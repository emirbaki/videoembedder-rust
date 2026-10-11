//! The vector store. One file holds every record.
//!
//! Layout: magic "VEI1" | u32 dim | u32 count | record[count]
//! Record: u8 kind | u32 label_len | label | f32 start | f32 end | f32 vec[dim]
//!
//! A write goes to a temp file first. Then the temp file replaces the store.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::embed::dot;

const MAGIC: &[u8; 4] = b"VEI1";

pub const KIND_WINDOW: u8 = 0;
pub const KIND_NOTE: u8 = 1;
pub const KIND_FRAME: u8 = 2;
pub const KIND_AUDIO: u8 = 3;

/// True for a record that belongs to a video file.
///
/// A window record is the old shape. A frame or audio record is the new shape.
pub fn is_video(kind: u8) -> bool {
    matches!(kind, KIND_WINDOW | KIND_FRAME | KIND_AUDIO)
}

/// Make one label form for one video path.
///
/// Windows gives a path with backslashes or with slashes, or with both. The
/// console and the web binary must agree, so every label becomes a slash form.
pub fn normalize_label(path: &str) -> String {
    path.replace('\\', "/")
}

/// The finished list beside the store. `store/index.vei` gives `store/index.done`.
fn done_path(store: &Path) -> PathBuf {
    store.with_extension("done")
}

/// Read the finished list. A missing file gives an empty set.
fn read_done(store: &Path) -> HashSet<String> {
    let text = std::fs::read_to_string(done_path(store)).unwrap_or_default();
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(normalize_label)
        .collect()
}

/// One record before it enters the store. The vector travels with it.
pub struct NewRecord {
    pub kind: u8,
    pub label: String,
    pub start: f32,
    pub end: f32,
    pub vec: Vec<f32>,
}

/// The id of one record, in Google's shape.
///
/// The label holds the source, and the two times hold the place in it. A
/// second pass over one video writes the same ids, so the write replaces the
/// rows of the first pass.
pub fn record_id(label: &str, start: f32, end: f32) -> String {
    format!("{}/{}-{}", label, millis(start), millis(end))
}

/// One time in milliseconds, rounded to the nearest one.
fn millis(seconds: f32) -> i64 {
    (seconds as f64 * 1000.0).round() as i64
}

#[derive(Clone)]
pub struct Record {
    pub kind: u8,
    pub label: String,
    pub start: f32,
    pub end: f32,
    pub vec: Vec<f32>,
}

pub struct Store {
    path: PathBuf,
    pub dim: usize,
    pub records: Vec<Record>,
    /// The video paths that finished an index run.
    done: HashSet<String>,
}

impl Store {
    /// Open the store. A missing file gives an empty store.
    pub fn open(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self {
                path: path.to_path_buf(),
                dim: 0,
                records: Vec::new(),
                done: read_done(path),
            });
        }
        let bytes = std::fs::read(path).context("read the store")?;
        let mut store = Self::decode(path.to_path_buf(), &bytes)?;
        store.done = read_done(&store.path);
        Ok(store)
    }

    fn decode(path: PathBuf, bytes: &[u8]) -> Result<Self> {
        let mut cur = Cursor { bytes, at: 0 };
        if cur.take(4)? != MAGIC {
            bail!("{} is not a store file", path.display());
        }
        let dim = cur.u32()? as usize;
        let count = cur.u32()? as usize;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let kind = cur.u8()?;
            let label_len = cur.u32()? as usize;
            let label = String::from_utf8(cur.take(label_len)?.to_vec())
                .context("store label is not utf8")?;
            let start = cur.f32()?;
            let end = cur.f32()?;
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(cur.f32()?);
            }
            records.push(Record {
                kind,
                label,
                start,
                end,
                vec,
            });
        }
        Ok(Self {
            path,
            dim,
            records,
            done: HashSet::new(),
        })
    }

    /// Append records and write the store.
    pub fn append(&mut self, new_records: Vec<Record>) -> Result<()> {
        if new_records.is_empty() {
            return Ok(());
        }
        if self.dim == 0 {
            self.dim = new_records[0].vec.len();
        }
        for record in &new_records {
            if record.vec.len() != self.dim {
                bail!(
                    "record vector size {} does not match the store size {}",
                    record.vec.len(),
                    self.dim
                );
            }
        }
        self.records.extend(new_records);
        self.write()
    }

    /// Remove every video record of one video path.
    ///
    /// The path separators do not matter, like `holds_video`.
    pub fn forget(&mut self, path: &str) -> Result<usize> {
        let want = normalize_label(path);
        let before = self.records.len();
        self.records
            .retain(|r| !(is_video(r.kind) && normalize_label(&r.label) == want));
        let removed = before - self.records.len();
        if removed > 0 {
            self.write()?;
            self.done.remove(&want);
            self.write_done()?;
        }
        Ok(removed)
    }

    /// True when this video finished an index run.
    ///
    /// The path separators do not matter, like `holds_video`.
    pub fn is_done(&self, label: &str) -> bool {
        self.done.contains(&normalize_label(label))
    }

    /// Record that this video finished an index run.
    pub fn mark_done(&mut self, label: &str) -> Result<()> {
        self.done.insert(normalize_label(label));
        self.write_done()
    }

    /// The finished set, one path on each line.
    fn write_done(&self) -> Result<()> {
        let mut out = String::new();
        for label in &self.done {
            out.push_str(label);
            out.push('\n');
        }
        let path = done_path(&self.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("done.tmp");
        std::fs::write(&tmp, out).context("write the finished list")?;
        std::fs::rename(&tmp, &path).context("replace the finished list")?;
        Ok(())
    }

    /// Build the store bytes. `write` sends them to the file.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.records.len() as u32).to_le_bytes());
        for record in &self.records {
            out.push(record.kind);
            out.extend_from_slice(&(record.label.len() as u32).to_le_bytes());
            out.extend_from_slice(record.label.as_bytes());
            out.extend_from_slice(&record.start.to_le_bytes());
            out.extend_from_slice(&record.end.to_le_bytes());
            for x in &record.vec {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// True when the store holds a video record with this exact path.
    ///
    /// The path separators do not matter, so the console and the web binary
    /// find the same record.
    pub fn holds_video(&self, path: &str) -> bool {
        let want = normalize_label(path);
        self.records
            .iter()
            .any(|record| is_video(record.kind) && normalize_label(&record.label) == want)
    }

    fn write(&self) -> Result<()> {
        let out = self.encode();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("vei.tmp");
        let mut file = std::fs::File::create(&tmp).context("create the temp store")?;
        file.write_all(&out)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, &self.path).context("replace the store")?;
        Ok(())
    }

    /// Rank every record against one query vector. Best first.
    pub fn rank(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| (i, dot(query, &r.vec)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.at + n > self.bytes.len() {
            bail!("the store file ended early");
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Build store bytes for a test. The real path goes through `Store::encode`.
#[cfg(test)]
fn encode_for_test(dim: usize, records: &[Record]) -> Vec<u8> {
    let store = Store {
        path: PathBuf::from("unused.vei"),
        dim,
        records: records.to_vec(),
        done: HashSet::new(),
    };
    store.encode()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(label: &str) -> Record {
        Record {
            kind: KIND_WINDOW,
            label: label.to_string(),
            start: 0.0,
            end: 4.0,
            vec: vec![1.0, 0.0],
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_store_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn store_of(records: Vec<Record>) -> Store {
        Store {
            path: PathBuf::from("unused.vei"),
            dim: 2,
            records,
            done: HashSet::new(),
        }
    }

    #[test]
    fn holds_video_matches_one_exact_path() {
        let store = store_of(vec![record("D:/video/a.mp4"), record("D:/video/b.mp4")]);
        assert!(store.holds_video("D:/video/a.mp4"));
        assert!(!store.holds_video("D:/video/zz.mp4"));
        assert!(!store.holds_video(""));
    }

    #[test]
    fn holds_video_keeps_a_path_with_a_space_and_a_unicode_letter() {
        let store = store_of(vec![record("D:/video/doğum günü.mp4")]);
        assert!(store.holds_video("D:/video/doğum günü.mp4"));
        assert!(!store.holds_video("D:/video/dogum gunu.mp4"));
    }

    #[test]
    fn holds_video_ignores_a_note_with_the_same_text() {
        let mut note = record("birthday");
        note.kind = KIND_NOTE;
        let store = store_of(vec![note]);
        assert!(!store.holds_video("birthday"));
    }

    #[test]
    fn is_video_accepts_the_three_video_kinds() {
        assert!(is_video(KIND_WINDOW));
        assert!(is_video(KIND_FRAME));
        assert!(is_video(KIND_AUDIO));
        assert!(!is_video(KIND_NOTE));
    }

    #[test]
    fn holds_video_matches_a_frame_record() {
        let mut frame = record("D:/video/a.mp4");
        frame.kind = KIND_FRAME;
        let store = store_of(vec![frame]);
        assert!(store.holds_video("D:/video/a.mp4"));
        assert!(!store.holds_video("D:/video/b.mp4"));
    }

    #[test]
    fn forget_removes_frame_and_audio_records_of_one_video() {
        let dir = std::env::temp_dir().join(format!("vei_forget_kind_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut frame = record("D:/video/a.mp4");
        frame.kind = KIND_FRAME;
        let mut audio = record("D:/video/a.mp4");
        audio.kind = KIND_AUDIO;
        let mut store = Store {
            path: dir.join("index.vei"),
            dim: 2,
            records: vec![frame, audio, record("D:/video/b.mp4")],
            done: HashSet::new(),
        };
        let removed = store.forget("D:/video/a.mp4").unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.records.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalize_label_turns_backslashes_into_forward_slashes() {
        assert_eq!(normalize_label("D:\\video\\a.mp4"), "D:/video/a.mp4");
        assert_eq!(normalize_label("D:/video/a.mp4"), "D:/video/a.mp4");
        assert_eq!(normalize_label("../spike/a.mp4"), "../spike/a.mp4");
    }

    #[test]
    fn holds_video_matches_across_path_separators() {
        let store = store_of(vec![record("D:\\video\\a.mp4")]);
        assert!(store.holds_video("D:/video/a.mp4"));
        assert!(store.holds_video("D:\\video\\a.mp4"));
    }

    #[test]
    fn forget_removes_a_label_across_path_separators() {
        let dir = std::env::temp_dir().join(format!("vei_forget_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store {
            path: dir.join("index.vei"),
            dim: 2,
            records: vec![record("D:\\video\\a.mp4"), record("D:\\video\\b.mp4")],
            done: HashSet::new(),
        };
        let removed = store.forget("D:/video/a.mp4").unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.records.len(), 1);
        assert_eq!(store.records[0].label, "D:\\video\\b.mp4");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_store_round_trips_through_the_file_format() {
        let records = vec![record("D:/video/a.mp4"), record("D:/video/b.mp4")];
        let bytes = encode_for_test(2, &records);
        let decoded = Store::decode(PathBuf::from("unused.vei"), &bytes).unwrap();
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.records.len(), 2);
        assert_eq!(decoded.records[1].label, "D:/video/b.mp4");
        assert_eq!(decoded.records[0].start, 0.0);
        assert_eq!(decoded.records[0].end, 4.0);
        assert_eq!(decoded.records[0].vec, vec![1.0, 0.0]);
    }

    #[test]
    fn a_damaged_file_is_rejected_with_a_message() {
        let err = match Store::decode(PathBuf::from("broken.vei"), b"nope") {
            Ok(_) => panic!("a damaged file was accepted"),
            Err(err) => err,
        };
        assert!(format!("{err:#}").contains("broken.vei"));
    }

    /// A finished video is known after the store reopens.
    #[test]
    fn a_finished_video_survives_a_reopen() {
        let dir = scratch("done");
        let path = dir.join("index.vei");
        let mut store = Store::open(&path).unwrap();
        store.mark_done("a.mp4").unwrap();

        assert!(store.is_done("a.mp4"));
        assert!(!store.is_done("b.mp4"));

        let again = Store::open(&path).unwrap();
        assert!(again.is_done("a.mp4"), "the mark must reach the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A video that is forgotten must index again.
    #[test]
    fn forget_clears_the_finished_mark() {
        let dir = scratch("done_forget");
        let path = dir.join("index.vei");
        let mut store = Store::open(&path).unwrap();
        store.append(vec![record("a.mp4")]).unwrap();
        store.mark_done("a.mp4").unwrap();

        store.forget("a.mp4").unwrap();

        assert!(!store.is_done("a.mp4"));
        let again = Store::open(&path).unwrap();
        assert!(!again.is_done("a.mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The mark ignores the slash direction, like every other path test.
    #[test]
    fn a_windows_path_form_still_matches_the_mark() {
        let dir = scratch("done_slash");
        let path = dir.join("index.vei");
        let mut store = Store::open(&path).unwrap();
        store.mark_done(r"sub\a.mp4").unwrap();
        assert!(store.is_done("sub/a.mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_id_holds_the_path_and_the_range() {
        assert_eq!(record_id("D:/video/a.mp4", 0.25, 0.5), "D:/video/a.mp4/250-500");
        assert_eq!(record_id("a note", 0.0, 0.0), "a note/0-0");
    }

    #[test]
    fn the_same_record_gives_the_same_id_twice() {
        let first = record_id("a.mp4", 1.0, 1.25);
        let second = record_id("a.mp4", 1.0, 1.25);
        assert_eq!(first, second, "a second pass must write the same id");
    }
}
