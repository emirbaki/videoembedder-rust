//! The vector store. One SQLite database holds every record.
//!
//! The vectors live in the `vec_records` table of `db`. The metadata lives in
//! the `records` and `videos` tables. The in-memory list holds the metadata
//! alone, so a large store costs memory for the small part.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, Transaction, params};

use crate::db;

const MAGIC: &[u8; 4] = b"VEI1";

pub const KIND_WINDOW: u8 = 0;
pub const KIND_NOTE: u8 = 1;
pub const KIND_FRAME: u8 = 2;
pub const KIND_AUDIO: u8 = 3;

/// The plan of the run that fills the store. Window mode changes these.
const PLAN_WINDOW: f64 = 0.25;
const PLAN_FRAMES: i64 = 1;
const PLAN_AUDIO: i64 = 1;

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

/// One record in the store. The vector stays on disk.
#[derive(Clone)]
pub struct Record {
    pub rid: i64,
    pub id: String,
    pub kind: u8,
    pub label: String,
    pub start: f32,
    pub end: f32,
}

pub struct Store {
    path: PathBuf,
    conn: Connection,
    pub dim: usize,
    pub records: Vec<Record>,
    /// The video paths that finished an index run.
    done: HashSet<String>,
}

impl Store {
    /// Open the store. A missing file gives an empty store.
    ///
    /// An old `VEI1` file beside the database imports one time. The old file
    /// stays, so a fault costs nothing.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = db::open(path)?;
        let mut store = Self {
            path: path.to_path_buf(),
            conn,
            dim: db::DIM,
            records: Vec::new(),
            done: HashSet::new(),
        };
        store.import_old_file()?;
        store.load_records()?;
        store.done = store.load_done()?;
        Ok(store)
    }

    /// Import the records of an old `VEI1` file beside the store.
    ///
    /// The import runs when the old file is here and the database holds no
    /// record. Every imported video takes the finished mark, so the next run
    /// does not index it again.
    fn import_old_file(&mut self) -> Result<()> {
        let old = self.path.with_extension("vei");
        if !old.exists() {
            return Ok(());
        }
        let held: i64 = self
            .conn
            .query_row("select count(*) from records", [], |row| row.get(0))?;
        if held > 0 {
            return Ok(());
        }
        let bytes = std::fs::read(&old)
            .with_context(|| format!("read the old store {}", old.display()))?;
        let (dim, records) =
            vei_records(&bytes).with_context(|| format!("read the old store {}", old.display()))?;
        if dim != self.dim {
            bail!(
                "the old store {} holds vectors of size {dim}, but the model gives size {}",
                old.display(),
                self.dim
            );
        }
        let labels: Vec<String> = records
            .iter()
            .filter(|r| is_video(r.kind))
            .map(|r| r.label.clone())
            .collect();
        // The base `index` command appended, so one old store can hold one
        // record twice. The id joins the two copies, and the first one wins.
        let mut seen = HashSet::new();
        let mut fresh = Vec::with_capacity(records.len());
        for record in records {
            if seen.insert(record_id(&record.label, record.start, record.end)) {
                fresh.push(record);
            }
        }
        self.append(fresh)
            .with_context(|| format!("import the old store {}", old.display()))?;
        for label in labels {
            self.mark_done(&label)
                .with_context(|| format!("import the old store {}", old.display()))?;
        }
        println!("Imported {} records from {}", self.records.len(), old.display());
        Ok(())
    }

    /// Read the metadata of every record, in row order.
    fn load_records(&mut self) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("select rid, id, kind, path, start, end from records order by rid")?;
        let rows = stmt.query_map([], |row| {
            Ok(Record {
                rid: row.get(0)?,
                id: row.get(1)?,
                kind: row.get::<_, i64>(2)? as u8,
                label: row.get(3)?,
                start: row.get::<_, f64>(4)? as f32,
                end: row.get::<_, f64>(5)? as f32,
            })
        })?;
        self.records.clear();
        for row in rows {
            self.records.push(row?);
        }
        Ok(())
    }

    /// Read the finished marks.
    fn load_done(&self) -> Result<HashSet<String>> {
        let mut stmt = self.conn.prepare("select path from videos where done = 1")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
    }

    /// Append records and write the store.
    ///
    /// A path that arrives again replaces its own rows. One transaction holds
    /// the records, their vectors, and the video rows.
    pub fn append(&mut self, new_records: Vec<NewRecord>) -> Result<()> {
        if new_records.is_empty() {
            return Ok(());
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

        let tx = self.conn.transaction().context("begin the write")?;
        // The path match ignores the separator direction, so a batch label and
        // a stored label name one video across `\` and `/`.
        let mut wants: Vec<String> = new_records.iter().map(|r| normalize_label(&r.label)).collect();
        wants.sort_unstable();
        wants.dedup();
        for want in &wants {
            let rids: Vec<i64> = self
                .records
                .iter()
                .filter(|record| normalize_label(&record.label) == *want)
                .map(|record| record.rid)
                .collect();
            delete_rows(&tx, &rids)?;
        }
        for record in &new_records {
            tx.execute(
                "insert into records(id, path, kind, start, end) values (?1, ?2, ?3, ?4, ?5)",
                params![
                    record_id(&record.label, record.start, record.end),
                    record.label,
                    record.kind as i64,
                    record.start as f64,
                    record.end as f64
                ],
            )?;
            let rid = tx.last_insert_rowid();
            tx.execute(
                "insert into vec_records(rid, path, embedding) values (?1, ?2, vec_f32(?3))",
                params![rid, record.label, db::vec_to_blob(&record.vec)],
            )?;
        }
        write_videos(&tx, &new_records)?;
        tx.commit().context("commit the write")?;

        self.load_records()
    }

    /// Remove every video record of one video path.
    ///
    /// The path separators do not matter, like `holds_video`. A note keeps its
    /// place, because a note is not a video record. The stored `path` column
    /// keeps the label as it shows, so the rows are found by id, not by path.
    pub fn forget(&mut self, path: &str) -> Result<usize> {
        let want = normalize_label(path);
        let rids: Vec<i64> = self
            .records
            .iter()
            .filter(|record| record.kind != KIND_NOTE && normalize_label(&record.label) == want)
            .map(|record| record.rid)
            .collect();
        let removed = rids.len();

        let tx = self.conn.transaction().context("begin the delete")?;
        delete_rows(&tx, &rids)?;
        tx.execute("delete from videos where path = ?1", [&want])?;
        tx.commit().context("commit the delete")?;

        self.done.remove(&want);
        self.load_records()?;
        Ok(removed)
    }

    /// True when this video finished an index run.
    pub fn is_done(&self, label: &str) -> bool {
        self.done.contains(&normalize_label(label))
    }

    /// Record that this video finished an index run.
    pub fn mark_done(&mut self, label: &str) -> Result<()> {
        let want = normalize_label(label);
        self.conn.execute(
            "insert into videos(path, plan_window, plan_stride, plan_frames, plan_audio, done)
             values (?1, ?2, ?2, ?3, ?4, 1)
             on conflict(path) do update set done = 1",
            params![want, PLAN_WINDOW, PLAN_FRAMES, PLAN_AUDIO],
        )?;
        self.done.insert(want);
        Ok(())
    }

    /// True when the store holds a video record with this exact path.
    pub fn holds_video(&self, path: &str) -> bool {
        let want = normalize_label(path);
        self.records
            .iter()
            .any(|record| is_video(record.kind) && normalize_label(&record.label) == want)
    }

    /// Read the vectors of these rows, in the order of the ids.
    ///
    /// The measure command uses this. The vectors stay on disk, so a caller
    /// asks for the ones it needs.
    pub fn vectors(&self, rids: &[i64]) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("select embedding from vec_records where rid = ?1")?;
        let mut out = Vec::with_capacity(rids.len());
        for rid in rids {
            let blob: Vec<u8> = stmt
                .query_row([rid], |row| row.get(0))
                .with_context(|| format!("read the vector of row {rid}"))?;
            out.push(db::blob_to_vec(&blob));
        }
        Ok(out)
    }

    /// The path of the store file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The position of one row id in the record list.
    ///
    /// A row id without a record means the metadata and the vector table have
    /// drifted apart. One transaction keeps them in step, so this is a signal
    /// of drift, not a normal case. The rank keeps going and names the row.
    fn position_of(&self, rid: i64) -> Option<usize> {
        match self.records.iter().position(|record| record.rid == rid) {
            Some(at) => Some(at),
            None => {
                eprintln!("the vector table holds row {rid} with no record");
                None
            }
        }
    }

    /// Rank every record against one query vector. Best first.
    ///
    /// The query asks the vector table for every row and its distance. The
    /// table holds cosine distance, so the score is one minus the distance.
    /// The rank needs every row, because the median of the store is the
    /// baseline of the result.
    ///
    /// A database fault surfaces as an error. Only an empty store gives an
    /// empty list, because a non-empty store always has a row to score.
    pub fn rank(&self, query: &[f32]) -> Result<Vec<(usize, f32)>> {
        if self.records.is_empty() {
            return Ok(Vec::new());
        }
        let sql = "select rid, distance from vec_records where embedding match vec_f32(?1) and k = ?2";
        let mut stmt = self
            .conn
            .prepare(sql)
            .context("prepare the rank query")?;
        let rows = stmt.query_map(
            params![db::vec_to_blob(query), self.records.len() as i64],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)? as f32)),
        )
        .context("run the rank query")?;
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for row in rows {
            let (rid, distance) = row.context("read a rank row")?;
            if let Some(at) = self.position_of(rid) {
                scored.push((at, 1.0 - distance));
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }
}

/// Delete the rows with these ids from both tables.
fn delete_rows(tx: &Transaction, rids: &[i64]) -> Result<()> {
    for rid in rids {
        tx.execute("delete from vec_records where rid = ?1", [rid])?;
        tx.execute("delete from records where rid = ?1", [rid])?;
    }
    Ok(())
}

/// Write one `videos` row for each video path in the batch.
///
/// The stored path is the normalized form, because `is_done` normalizes when
/// it reads the mark.
fn write_videos(tx: &Transaction, records: &[NewRecord]) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    for record in records {
        if !is_video(record.kind) || !seen.insert(record.label.as_str()) {
            continue;
        }
        let (size, mtime) = fingerprint(&record.label);
        tx.execute(
            "insert into videos(path, plan_window, plan_stride, plan_frames, plan_audio,
                                source_size, source_mtime, done)
             values (?1, ?2, ?2, ?3, ?4, ?5, ?6, 0)
             on conflict(path) do update set
               plan_window = excluded.plan_window,
               plan_stride = excluded.plan_stride,
               plan_frames = excluded.plan_frames,
               plan_audio = excluded.plan_audio,
               source_size = excluded.source_size,
               source_mtime = excluded.source_mtime",
            params![
                normalize_label(&record.label),
                PLAN_WINDOW,
                PLAN_FRAMES,
                PLAN_AUDIO,
                size,
                mtime
            ],
        )?;
    }
    Ok(())
}

/// The size and the time of the source file. A note gives two `None` values.
fn fingerprint(label: &str) -> (Option<i64>, Option<i64>) {
    let Ok(meta) = std::fs::metadata(label) else {
        return (None, None);
    };
    let size = meta.len() as i64;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);
    (Some(size), mtime)
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

/// Read the records of an old `VEI1` file. The import uses this.
fn vei_records(bytes: &[u8]) -> Result<(usize, Vec<NewRecord>)> {
    let mut cur = Cursor { bytes, at: 0 };
    if cur.take(4)? != MAGIC {
        bail!("the old store file is not a VEI1 file");
    }
    let dim = cur.u32()? as usize;
    let count = cur.u32()? as usize;
    let mut out = Vec::with_capacity(count);
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
        out.push(NewRecord {
            kind,
            label,
            start,
            end,
            vec,
        });
    }
    Ok((dim, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A frame record for one video. The vector size is the model size.
    fn new_record(label: &str, start: f32, end: f32) -> NewRecord {
        let mut vec = vec![0.0f32; db::DIM];
        vec[0] = 1.0;
        NewRecord {
            kind: KIND_FRAME,
            label: label.to_string(),
            start,
            end,
            vec,
        }
    }

    /// A store on a fresh database in the scratch folder.
    fn opened(name: &str) -> (PathBuf, Store) {
        let dir = scratch(name);
        let path = dir.join("index.db");
        let store = Store::open(&path).unwrap();
        (dir, store)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_store_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A stored record, for the tests that read one.
    fn record(label: &str) -> Record {
        Record {
            rid: 0,
            id: record_id(label, 0.0, 4.0),
            kind: KIND_WINDOW,
            label: label.to_string(),
            start: 0.0,
            end: 4.0,
        }
    }

    /// A store that holds these records, on a fresh database.
    ///
    /// The folder is unique per call, so tests under the parallel runner do
    /// not share one database.
    fn store_of(records: Vec<Record>) -> Store {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "vei_store_of_{}_{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        let new: Vec<NewRecord> = records
            .into_iter()
            .map(|r| NewRecord {
                kind: r.kind,
                label: r.label,
                start: r.start,
                end: r.end,
                vec: {
                    let mut vec = vec![0.0f32; db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            })
            .collect();
        store.append(new).unwrap();
        store
    }

    #[test]
    fn a_record_reaches_the_database_and_comes_back() {
        let (dir, mut store) = opened("round");
        store
            .append(vec![new_record("D:/video/a.mp4", 0.25, 0.5)])
            .unwrap();

        let again = Store::open(&dir.join("index.db")).unwrap();

        assert_eq!(again.records.len(), 1);
        assert_eq!(again.records[0].label, "D:/video/a.mp4");
        assert_eq!(again.records[0].id, "D:/video/a.mp4/250-500");
        assert_eq!(again.records[0].kind, KIND_FRAME);
        assert_eq!(again.records[0].start, 0.25);
        assert_eq!(again.records[0].end, 0.5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A second pass over one video replaces its rows. It does not double them.
    #[test]
    fn a_second_pass_leaves_one_set_of_rows() {
        let (dir, mut store) = opened("twice");
        store
            .append(vec![
                new_record("a.mp4", 0.0, 0.25),
                new_record("a.mp4", 0.25, 0.5),
            ])
            .unwrap();
        store
            .append(vec![
                new_record("a.mp4", 0.0, 0.25),
                new_record("a.mp4", 0.25, 0.5),
            ])
            .unwrap();

        assert_eq!(store.records.len(), 2);
        let again = Store::open(&dir.join("index.db")).unwrap();
        assert_eq!(again.records.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The delete of one video touches the records and the vectors.
    #[test]
    fn a_delete_removes_the_rows_and_the_vectors() {
        let (dir, mut store) = opened("delete");
        store
            .append(vec![new_record("a.mp4", 0.0, 0.25), new_record("b.mp4", 0.0, 0.25)])
            .unwrap();

        let removed = store.forget("a.mp4").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(store.records.len(), 1);
        let left: i64 = store
            .conn
            .query_row("select count(*) from vec_records", [], |row| row.get(0))
            .unwrap();
        assert_eq!(left, 1, "the vector of the removed record must go");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two paths where one starts like the other stay apart.
    #[test]
    fn a_path_that_starts_like_another_keeps_the_delete_to_one_video() {
        let (dir, mut store) = opened("prefix");
        store
            .append(vec![
                new_record("a/b.mp4", 0.0, 0.25),
                new_record("a/b.mp4x", 0.0, 0.25),
            ])
            .unwrap();

        let removed = store.forget("a/b.mp4").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(store.records.len(), 1);
        assert_eq!(store.records[0].label, "a/b.mp4x");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fault in the middle of a write leaves no half record.
    ///
    /// The two records hold the same path, the same range, and so the same
    /// id. The second insert breaks the unique rule. One transaction holds
    /// the two, so the first insert goes back with the second.
    #[test]
    fn a_fault_in_the_middle_of_a_write_leaves_no_rows() {
        let (dir, mut store) = opened("fault");
        let outcome = store.append(vec![
            new_record("a.mp4", 0.0, 0.25),
            new_record("a.mp4", 0.0, 0.25),
        ]);

        assert!(outcome.is_err(), "the same id must break the write");
        assert!(store.records.is_empty(), "the store keeps no half record");
        let rows: i64 = store
            .conn
            .query_row("select count(*) from vec_records", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 0, "no vector may stay after the fault");
        let again = Store::open(&dir.join("index.db")).unwrap();
        assert!(again.records.is_empty(), "the file keeps no half record");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A space and a non-ASCII letter survive the database.
    #[test]
    fn a_unicode_path_and_a_note_round_trip() {
        let (dir, mut store) = opened("unicode");
        let mut note = new_record("doğum günü", 0.0, 0.0);
        note.kind = KIND_NOTE;
        store
            .append(vec![new_record("D:/video/doğum günü.mp4", 0.0, 0.25), note])
            .unwrap();

        let again = Store::open(&dir.join("index.db")).unwrap();

        assert_eq!(again.records.len(), 2);
        assert!(again.holds_video("D:/video/doğum günü.mp4"));
        assert_eq!(again.records[1].kind, KIND_NOTE);
        assert_eq!(again.records[1].id, "doğum günü/0-0");
        let _ = std::fs::remove_dir_all(&dir);
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
        let (dir, mut store) = opened("forget_kind");
        let mut audio = new_record("D:/video/a.mp4", 0.0, 1.0);
        audio.kind = KIND_AUDIO;
        store
            .append(vec![
                new_record("D:/video/a.mp4", 0.0, 0.25),
                audio,
                new_record("D:/video/b.mp4", 0.0, 0.25),
            ])
            .unwrap();

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
        let (dir, mut store) = opened("forget_slash");
        store
            .append(vec![
                new_record("D:\\video\\a.mp4", 0.0, 0.25),
                new_record("D:\\video\\b.mp4", 0.0, 0.25),
            ])
            .unwrap();

        let removed = store.forget("D:/video/a.mp4").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(store.records[0].label, "D:\\video\\b.mp4");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A finished video is known after the store reopens.
    #[test]
    fn a_finished_video_survives_a_reopen() {
        let dir = scratch("done");
        let path = dir.join("index.db");
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
        let path = dir.join("index.db");
        let mut store = Store::open(&path).unwrap();
        store.append(vec![new_record("a.mp4", 0.0, 4.0)]).unwrap();
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
        let path = dir.join("index.db");
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

    /// A vector with one 1.0 at `at`, on the model size.
    fn basis(at: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; db::DIM];
        vec[at] = 1.0;
        vec
    }

    #[test]
    fn the_rank_returns_every_record_best_first() {
        let (dir, mut store) = opened("rank");
        let mut near = new_record("a.mp4", 0.0, 0.25);
        near.vec = basis(0);
        let mut far = new_record("b.mp4", 0.0, 0.25);
        far.vec = basis(1);
        store.append(vec![near, far]).unwrap();

        let ranked = store.rank(&basis(0)).unwrap();

        assert_eq!(ranked.len(), 2, "the median needs every score");
        assert_eq!(store.records[ranked[0].0].label, "a.mp4");
        assert!(
            ranked[0].1 > ranked[1].1,
            "the near record must score higher"
        );
        assert!(
            (ranked[0].1 - 1.0).abs() < 1e-4,
            "a unit vector against itself scores 1.0, got {}",
            ranked[0].1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_rank_of_an_empty_store_is_empty() {
        let (dir, store) = opened("rank_empty");
        assert!(store.rank(&basis(0)).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A note takes part in the rank, like a frame record.
    #[test]
    fn the_rank_holds_a_note() {
        let (dir, mut store) = opened("rank_note");
        let mut note = new_record("birthday", 0.0, 0.0);
        note.kind = KIND_NOTE;
        note.vec = basis(0);
        store.append(vec![note]).unwrap();

        let ranked = store.rank(&basis(0)).unwrap();

        assert_eq!(ranked.len(), 1);
        assert_eq!(store.records[ranked[0].0].kind, KIND_NOTE);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A database fault must surface. It must not read as an empty store.
    #[test]
    fn a_fault_in_the_rank_query_is_an_error() {
        let (dir, mut store) = opened("rank_fault");
        store.append(vec![new_record("a.mp4", 0.0, 0.25)]).unwrap();
        store
            .conn
            .execute_batch("drop table vec_records")
            .unwrap();

        let outcome = store.rank(&basis(0));

        assert!(outcome.is_err(), "a dropped table must not read as no matches");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build an old `VEI1` file from these records.
    fn vei_bytes(dim: usize, records: &[NewRecord]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for record in records {
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

    #[test]
    fn an_old_store_imports_into_the_database() {
        let dir = scratch("import");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let old = vec![new_record("D:/video/a.mp4", 0.0, 0.25)];
        std::fs::write(dir.join("index.vei"), vei_bytes(db::DIM, &old)).unwrap();

        let store = Store::open(&path).unwrap();

        assert_eq!(store.records.len(), 1);
        assert_eq!(store.records[0].label, "D:/video/a.mp4");
        assert!(store.is_done("D:/video/a.mp4"), "an old video is finished");
        assert!(
            dir.join("index.vei").exists(),
            "the old file must stay after the import"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_import_runs_one_time() {
        let dir = scratch("import_once");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let old = vec![new_record("a.mp4", 0.0, 0.25)];
        std::fs::write(dir.join("index.vei"), vei_bytes(db::DIM, &old)).unwrap();
        let _ = Store::open(&path).unwrap();

        let again = Store::open(&path).unwrap();

        assert_eq!(again.records.len(), 1, "the import must not double the rows");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An old store from another model must not import.
    #[test]
    fn an_old_store_of_another_size_is_refused() {
        let dir = scratch("import_size");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let mut record = new_record("a.mp4", 0.0, 0.25);
        record.vec = vec![1.0, 0.0];
        std::fs::write(dir.join("index.vei"), vei_bytes(2, &[record])).unwrap();

        let err = match Store::open(&path) {
            Ok(_) => panic!("an old store of size 2 was accepted"),
            Err(err) => err,
        };

        assert!(format!("{err:#}").contains("index.vei"), "got {err:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An old store can hold one record twice, because the base `index`
    /// command appended and a second run over one video wrote the same ids.
    /// The import must land one set of rows.
    #[test]
    fn a_repeated_record_in_an_old_store_imports_once() {
        let dir = scratch("import_repeat");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let first = new_record("D:/video/a.mp4", 0.0, 0.25);
        let second = new_record("D:/video/a.mp4", 0.0, 0.25);
        std::fs::write(dir.join("index.vei"), vei_bytes(db::DIM, &[first, second])).unwrap();

        let store = Store::open(&path).unwrap();

        assert_eq!(store.records.len(), 1, "the import must not hold a copy");
        let rids: Vec<i64> = store.records.iter().map(|r| r.rid).collect();
        assert_eq!(store.vectors(&rids).unwrap().len(), 1);
        assert!(store.is_done("D:/video/a.mp4"), "an old video is finished");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_vectors_come_back_in_the_order_of_the_ids() {
        let (dir, mut store) = opened("vectors");
        let mut first = new_record("a.mp4", 0.0, 0.25);
        first.vec = basis(0);
        let mut second = new_record("a.mp4", 0.25, 0.5);
        second.vec = basis(1);
        store.append(vec![first, second]).unwrap();
        let rids: Vec<i64> = store.records.iter().map(|r| r.rid).collect();

        let vectors = store.vectors(&[rids[1], rids[0]]).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], basis(1), "the order of the ids must hold");
        assert_eq!(vectors[1], basis(0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
