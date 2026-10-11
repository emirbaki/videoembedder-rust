//! The folder scan, the frame and audio plan, and the index run.
//!
//! One frame covers 0.25 s. One audio clip covers 1 s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::LlamaContext;
use serde::Serialize;

use crate::embed::Embedder;
use crate::media;
use crate::store::{self, NewRecord, Record};

/// An item below this score never enters a result group.
pub const FLOOR: f32 = 0.4;
/// The most items one video may add to one group.
pub const GROUP_CAP: usize = 50;
/// The number of frames one ffmpeg call cuts.
pub const FRAME_BATCH: usize = 120;
/// The number of audio clips one ffmpeg call cuts.
pub const AUDIO_BATCH: usize = 60;
/// Frames a second in the frame plan.
pub const FRAME_RATE: f64 = 4.0;
/// The time step between two frames in seconds.
pub const FRAME_STEP: f64 = 0.25;
/// The length of one audio clip in seconds.
pub const AUDIO_SECONDS: f64 = 1.0;

/// The largest gap between two hits that still joins one moment.
///
/// The gap is one frame step. Hits of one run of frames touch, so the gap is
/// zero. A run with one frame missing has a gap of one step.
pub const MOMENT_GAP: f32 = FRAME_STEP as f32;

/// The median score of one ranked list. An empty list gives 0.0.
///
/// The median is the baseline of the store for one query. A hit shows its
/// score minus this value, because an embedding score alone has no scale.
pub fn median(ranked: &[(usize, f32)]) -> f32 {
    if ranked.is_empty() {
        return 0.0;
    }
    let mut scores: Vec<f32> = ranked.iter().map(|(_, score)| *score).collect();
    scores.sort_by(f32::total_cmp);
    let n = scores.len();
    if n % 2 == 1 {
        scores[n / 2]
    } else {
        (scores[n / 2 - 1] + scores[n / 2]) / 2.0
    }
}

/// The gap between the best score of one result and the weak cut.
pub const WEAK_GAP: f32 = 0.20;

/// The cut that marks a weak result.
///
/// The best score of the result anchors the cut, as Google's photo feature
/// does. The floor of 0.4 keeps the cut above a hopeless row.
pub fn weak_cut(top: f32) -> f32 {
    (top - WEAK_GAP).max(FLOOR)
}

/// Is one score below the cut of its result?
pub fn is_weak(score: f32, cut: f32) -> bool {
    score < cut
}

const VIDEO_EXTENSIONS: [&str; 5] = ["mp4", "mkv", "mov", "avi", "webm"];

/// The progress of the running index job. The server reads a copy.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    pub running: bool,
    pub folder: String,
    pub file: String,
    pub file_index: usize,
    pub file_count: usize,
    pub window: usize,
    pub window_count: usize,
    pub indexed: usize,
    pub error: Option<String>,
    /// The device of the loaded model, "GPU" or "CPU".
    pub device: String,
    /// True when this machine can put layers on a card.
    pub gpu_available: bool,
    /// The milliseconds to load the model.
    pub load_ms: u64,
    /// The milliseconds of the running job.
    pub elapsed_ms: u64,
    /// The items of the running job for each second.
    pub per_second: f32,
}

/// One matching item inside a video.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub start: f32,
    pub end: f32,
    pub score: f32,
    pub margin: f32,
}

/// One slice of a video, with the score of the whole slice.
///
/// A group holds one cell for every video record, also the ones below the
/// floor. The web bar draws the cells, so the bar covers the whole video.
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub start: f32,
    pub end: f32,
    pub score: f32,
}

/// The hits of one video.
#[derive(Clone, Debug, Serialize)]
pub struct Group {
    pub video: String,
    pub duration: f32,
    pub score: f32,
    pub hits: Vec<Hit>,
    pub cells: Vec<Cell>,
}

/// List the video files under one folder, in name order.
pub fn list_videos(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    walk(folder, &mut found)?;
    found.sort();
    Ok(found)
}

fn walk(dir: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            walk(&path, found)?;
        } else if is_video(&path) {
            found.push(path);
        }
    }
    Ok(())
}

/// True when the file name ends with one of the video extensions.
pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|part| part.to_str())
        .map(|part| VIDEO_EXTENSIONS.contains(&part.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// The start time of every frame. One frame each 0.25 s.
///
/// A frame needs its full step inside the video. A video shorter than one step
/// has no frame, so a caller refuses it.
pub fn frames(duration: f64) -> Vec<f64> {
    let mut plan = Vec::new();
    let mut t = 0.0;
    while t + FRAME_STEP <= duration {
        plan.push(t);
        t += FRAME_STEP;
    }
    plan
}

/// The start time of every one-second audio clip.
pub fn audio_starts(duration: f64) -> Vec<f64> {
    let mut plan = Vec::new();
    let mut t = 0.0;
    while t + AUDIO_SECONDS <= duration {
        plan.push(t);
        t += AUDIO_SECONDS;
    }
    plan
}

/// A fresh working folder for one batch of ffmpeg output.
pub fn work_dir(at: usize) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("vei_{}_{n}_{at}", std::process::id()))
}

/// The phase of an index run.
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Frames,
    Audio,
    Done,
}

/// A stepwise index run for one video.
///
/// The run cuts media in batches, so the temp disk stays bounded. The caller
/// drives it: one `step` call embeds one frame or one audio clip.
pub struct IndexRun {
    video: PathBuf,
    label: String,
    duration: f64,
    frames: Vec<f64>,
    audios: Vec<f64>,
    frame_at: usize,
    audio_at: usize,
    phase: Phase,
    dir: PathBuf,
    files: Vec<PathBuf>,
    batch_from: usize,
    done: usize,
}

impl IndexRun {
    /// Probe the video and build the frame and the audio plan.
    ///
    /// The plan follows the length of each stream, not the length of the file.
    /// A clip past the end of its stream holds no media, and the encoder then
    /// refuses it. A short audio track is normal, so the audio plan is often
    /// shorter than the frame plan.
    pub fn start(video: &Path) -> Result<Self> {
        let duration = media::probe_duration(video)?;
        let video_len = media::probe_stream_duration(video, "v")?
            .unwrap_or(duration)
            .min(duration);
        let audio_len = if media::has_audio(video) {
            media::probe_stream_duration(video, "a")?
                .unwrap_or(duration)
                .min(duration)
        } else {
            0.0
        };
        let frames = frames(video_len);
        if frames.is_empty() {
            bail!("{} is shorter than one frame", video.display());
        }
        Ok(Self {
            video: video.to_path_buf(),
            label: store::normalize_label(&video.to_string_lossy()),
            duration,
            frames,
            audios: audio_starts(audio_len),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
        })
    }

    /// The number of records this run will make.
    pub fn total(&self) -> usize {
        self.frames.len() + self.audios.len()
    }

    /// The number of records made so far.
    pub fn done(&self) -> usize {
        self.done
    }

    /// Embed one item. Returns None when the run is complete.
    pub fn step(
        &mut self,
        embedder: &Embedder,
        ctx: &mut LlamaContext<'_>,
    ) -> Result<Option<NewRecord>> {
        if self.phase == Phase::Done {
            return Ok(None);
        }
        if self.remaining_in_batch() == 0 {
            self.cut_batch()?;
            if self.phase == Phase::Done {
                return Ok(None);
            }
        }
        let local = self.at() - self.batch_from;
        let path = self.files[local].clone();
        let start = self.at_time();
        let (kind, end, vec) = match self.phase {
            Phase::Frames => (
                store::KIND_FRAME,
                start + FRAME_STEP,
                embedder.embed_image_in(ctx, &path)?,
            ),
            Phase::Audio => (
                store::KIND_AUDIO,
                start + AUDIO_SECONDS,
                embedder.embed_audio_in(ctx, &path)?,
            ),
            Phase::Done => unreachable!(),
        };
        self.advance();
        self.done += 1;
        Ok(Some(NewRecord {
            kind,
            label: self.label.clone(),
            start: start as f32,
            end: end.min(self.duration) as f32,
            vec,
        }))
    }

    fn plan(&self) -> &[f64] {
        match self.phase {
            Phase::Frames => &self.frames,
            Phase::Audio => &self.audios,
            Phase::Done => &[],
        }
    }

    fn at(&self) -> usize {
        match self.phase {
            Phase::Frames => self.frame_at,
            Phase::Audio => self.audio_at,
            Phase::Done => 0,
        }
    }

    fn at_time(&self) -> f64 {
        self.plan().get(self.at()).copied().unwrap_or(0.0)
    }

    fn remaining_in_batch(&self) -> usize {
        self.files.len().saturating_sub(self.at() - self.batch_from)
    }

    fn advance(&mut self) {
        match self.phase {
            Phase::Frames => {
                self.frame_at += 1;
                if self.frame_at >= self.frames.len() {
                    // A video with no audio track has no second phase.
                    self.phase = if self.audios.is_empty() {
                        Phase::Done
                    } else {
                        Phase::Audio
                    };
                    self.audio_at = 0;
                    self.files.clear();
                    self.batch_from = 0;
                }
            }
            Phase::Audio => {
                self.audio_at += 1;
                if self.audio_at >= self.audios.len() {
                    self.phase = Phase::Done;
                }
            }
            Phase::Done => {}
        }
    }

    fn cut_batch(&mut self) -> Result<()> {
        self.clear_dir();
        if self.phase == Phase::Done || self.plan().is_empty() {
            self.phase = Phase::Done;
            return Ok(());
        }
        let at = self.at();
        let (count, step) = match self.phase {
            Phase::Frames => (FRAME_BATCH, FRAME_STEP),
            Phase::Audio => (AUDIO_BATCH, AUDIO_SECONDS),
            Phase::Done => unreachable!(),
        };
        let from = self.plan()[at];
        let last = (at + count).min(self.plan().len()) - 1;
        let to = (self.plan()[last] + step).min(self.duration);
        self.dir = work_dir(at);
        std::fs::create_dir_all(&self.dir)?;
        self.files = match self.phase {
            Phase::Frames => media::extract_frames(&self.video, from, to, FRAME_RATE, &self.dir)?,
            Phase::Audio => media::extract_audio(&self.video, from, to, &self.dir)?,
            Phase::Done => unreachable!(),
        };
        self.batch_from = at;
        if self.files.is_empty() {
            if self.phase == Phase::Audio {
                // A video without an audio track is normal.
                self.phase = Phase::Done;
                return Ok(());
            }
            bail!("no frames came from {}", self.video.display());
        }
        Ok(())
    }

    fn clear_dir(&mut self) {
        if !self.dir.as_os_str().is_empty() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
        self.files.clear();
    }
}

impl Drop for IndexRun {
    fn drop(&mut self) {
        self.clear_dir();
    }
}

/// Index one video. Embed every frame and every audio clip.
///
/// The callback runs after each record with the number made so far, counting
/// from 1, and the record count of the whole video.
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Vec<NewRecord>> {
    let mut run = IndexRun::start(video)?;
    let total = run.total();
    let mut ctx = embedder.new_context()?;
    let mut records = Vec::with_capacity(total);
    while let Some(record) = run.step(embedder, &mut ctx)? {
        records.push(record);
        on_step(run.done(), total);
    }
    Ok(records)
}

/// The items of one job for each second.
///
/// A time of zero or less gives 0.0, so a fast job cannot divide by zero.
pub fn per_second(items: usize, seconds: f64) -> f32 {
    if seconds <= 0.0 {
        return 0.0;
    }
    (items as f64 / seconds) as f32
}

/// Group the ranked records by video.
///
/// `ranked` comes from `Store::rank`, so it is sorted by score, best first.
pub fn group_hits(records: &[Record], ranked: &[(usize, f32)], baseline: f32) -> Vec<Group> {
    // The score of every record, at the index of that record. `ranked` holds
    // every record, so the bar can read a score below the floor too.
    let mut score_of = vec![0.0f32; records.len()];
    for (index, score) in ranked {
        if let Some(slot) = score_of.get_mut(*index) {
            *slot = *score;
        }
    }

    let mut durations: HashMap<String, f32> = HashMap::new();
    for record in records {
        if !crate::store::is_video(record.kind) {
            continue;
        }
        let held = durations
            .entry(crate::store::normalize_label(&record.label))
            .or_insert(0.0);
        if record.end > *held {
            *held = record.end;
        }
    }

    let mut groups: Vec<Group> = Vec::new();
    for (index, score) in ranked {
        if *score < FLOOR {
            break;
        }
        let record = &records[*index];
        if !crate::store::is_video(record.kind) {
            continue;
        }
        let name = crate::store::normalize_label(&record.label);
        let hit = Hit {
            start: record.start,
            end: record.end,
            score: *score,
            margin: *score - baseline,
        };
        match groups.iter_mut().find(|g| g.video == name) {
            Some(group) => {
                if group.hits.len() < GROUP_CAP {
                    group.hits.push(hit);
                }
            }
            None => {
                let duration = durations.get(&name).copied().unwrap_or(record.end);
                groups.push(Group {
                    video: name,
                    duration,
                    score: *score,
                    hits: vec![hit],
                    cells: Vec::new(),
                });
            }
        }
    }

    // One cell for every record of every grouped video, in time order. The
    // floor picks the hits. The bar shows all of them.
    //
    // A frame cell is 0.25 s and an audio cell is 1 s, so the two overlap. The
    // bar takes the frames when the video has them, because they are finer.
    for group in groups.iter_mut() {
        let has_frames = records.iter().any(|record| {
            record.kind == crate::store::KIND_FRAME
                && crate::store::normalize_label(&record.label) == group.video
        });
        group.cells = records
            .iter()
            .enumerate()
            .filter(|(_, record)| {
                let right_kind = if has_frames {
                    record.kind == crate::store::KIND_FRAME
                } else {
                    crate::store::is_video(record.kind)
                };
                right_kind && crate::store::normalize_label(&record.label) == group.video
            })
            .map(|(index, record)| Cell {
                start: record.start,
                end: record.end,
                score: score_of[index],
            })
            .collect();
        group.cells.sort_by(|a, b| a.start.total_cmp(&b.start));
    }

    // A run of touching hits reads as one moment. The cap counts moments, not
    // frames, so one video may show many moments.
    for group in groups.iter_mut() {
        let hits = std::mem::take(&mut group.hits);
        let mut moments = merge_hits(hits);
        moments.truncate(GROUP_CAP);
        group.hits = moments;
    }
    groups
}

/// Join hits that touch into one moment.
///
/// The moment spans the whole run. Its score and its margin come from the best
/// hit of the run, so a moment never claims more than one hit earned.
pub fn merge_hits(hits: Vec<Hit>) -> Vec<Hit> {
    if hits.len() < 2 {
        return hits;
    }
    let mut by_time = hits;
    by_time.sort_by(|a, b| a.start.total_cmp(&b.start));

    let mut moments: Vec<Hit> = Vec::new();
    for hit in by_time {
        match moments.last_mut() {
            Some(last) if hit.start - last.end <= MOMENT_GAP => {
                if hit.score > last.score {
                    last.score = hit.score;
                    last.margin = hit.margin;
                }
                if hit.end > last.end {
                    last.end = hit.end;
                }
            }
            _ => moments.push(hit),
        }
    }
    moments.sort_by(|a, b| b.score.total_cmp(&a.score));
    moments
}

/// The indexes of the records that stay after the thinning.
///
/// The thinning measures the cost of a coarser time grid. Every `factor`th
/// frame of one video stays, and the audio keeps its one second grid.
pub fn thin_indexes(records: &[Record], factor: usize) -> Vec<usize> {
    if factor < 2 {
        return (0..records.len()).collect();
    }
    // The frames of each video, in time order. A store may hold the frames of
    // one video twice, so the order is taken from the start time.
    let mut order: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        if record.kind == crate::store::KIND_FRAME {
            order
                .entry(crate::store::normalize_label(&record.label))
                .or_default()
                .push(index);
        }
    }
    let mut keep = vec![true; records.len()];
    for indexes in order.values_mut() {
        indexes.sort_by(|a, b| records[*a].start.total_cmp(&records[*b].start));
        for (at, index) in indexes.iter().enumerate() {
            if at % factor != 0 {
                keep[*index] = false;
            }
        }
    }
    (0..records.len()).filter(|at| keep[*at]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_test_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn record_of(kind: u8, label: &str, start: f32, end: f32) -> Record {
        Record {
            rid: 0,
            id: store::record_id(label, start, end),
            kind,
            label: label.to_string(),
            start,
            end,
        }
    }

    fn record(label: &str, start: f32, end: f32) -> Record {
        record_of(crate::store::KIND_FRAME, label, start, end)
    }

    #[test]
    fn list_videos_walks_subfolders_and_filters_extensions() {
        let dir = scratch("list");
        std::fs::write(dir.join("a.mp4"), b"x").unwrap();
        std::fs::write(dir.join("b.MKV"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("c.webm"), b"x").unwrap();

        let names: Vec<String> = list_videos(&dir)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert_eq!(names.len(), 3, "found {names:?}");
        assert!(names.contains(&"a.mp4".to_string()));
        assert!(names.contains(&"b.MKV".to_string()));
        assert!(names.contains(&"c.webm".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_path_with_a_space_and_a_unicode_letter_survives_the_scan() {
        let dir = scratch("unicode");
        std::fs::write(dir.join("doğum günü.mp4"), b"x").unwrap();
        let found = list_videos(&dir).unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].to_string_lossy().contains("doğum günü"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn frames_cover_a_thirty_second_video_at_four_a_second() {
        let plan = frames(30.0);
        assert_eq!(plan.len(), 120);
        assert_eq!(plan[0], 0.0);
        assert!((plan[1] - 0.25).abs() < 1e-9);
        assert!((plan[119] - 29.75).abs() < 1e-9);
    }

    #[test]
    fn a_video_shorter_than_one_frame_has_no_frame() {
        assert!(frames(0.2).is_empty());
        assert_eq!(frames(0.25).len(), 1);
    }

    #[test]
    fn audio_starts_cover_a_thirty_second_video() {
        let plan = audio_starts(30.0);
        assert_eq!(plan.len(), 30);
        assert_eq!(plan[0], 0.0);
        assert_eq!(plan[29], 29.0);
    }

    #[test]
    fn a_video_shorter_than_one_second_has_no_audio_clip() {
        assert!(audio_starts(0.5).is_empty());
        assert_eq!(audio_starts(1.0).len(), 1);
    }

    #[test]
    fn the_progress_object_matches_the_spec() {
        let json = serde_json::to_value(Progress::default()).unwrap();
        assert_eq!(json["running"], false);
        assert_eq!(json["error"], serde_json::Value::Null);
        assert_eq!(json["window_count"], 0);
        assert_eq!(json["indexed"], 0);
    }

    #[test]
    fn group_hits_drops_low_scores_and_sorts_groups() {
        let records = vec![
            record("a.mp4", 0.0, 4.0),
            record("a.mp4", 2.0, 6.0),
            record("b.mp4", 0.0, 4.0),
            record("b.mp4", 4.0, 8.0),
        ];
        let ranked = vec![(2, 0.9), (3, 0.7), (0, 0.5), (1, 0.2)];

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].video, "b.mp4");
        assert_eq!(groups[0].score, 0.9);
        assert_eq!(groups[0].duration, 8.0);
        // The two records of b.mp4 touch, so the moment covers the whole run.
        assert_eq!(groups[0].hits.len(), 1);
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 8.0).abs() < 1e-6);
        assert_eq!(groups[1].video, "a.mp4");
        assert_eq!(groups[1].hits.len(), 1);
    }

    /// One video gathers under one group, whatever separator its label carries.
    #[test]
    fn group_hits_gathers_a_video_across_path_separators() {
        let records = vec![
            record(r"D:\video\a.mp4", 0.0, 4.0),
            record("D:/video/a.mp4", 4.0, 8.0),
        ];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 1, "one video makes one group");
        assert_eq!(groups[0].video, "D:/video/a.mp4");
    }

    /// The bar needs every record, not only the ones above the floor.
    ///
    /// The floor picks the hit list. The bar shows the whole video, so a
    /// weak part of the video still takes its place on the bar.
    #[test]
    fn a_group_carries_one_cell_for_every_video_record() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("a.mp4", 0.25, 0.5),
            record("a.mp4", 0.5, 0.75),
        ];
        // The third record stays below the floor. It must still reach the bar.
        let ranked = vec![(0, 0.80), (2, 0.45), (1, 0.30)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups.len(), 1);
        // Two hits reach the floor. They touch, so they read as one moment
        // over the whole span. The middle record stays below the floor, and
        // the moment spans it.
        assert_eq!(groups[0].hits.len(), 1, "two hits joined into one moment");
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 0.75).abs() < 1e-6);
        assert!((groups[0].hits[0].score - 0.80).abs() < 1e-6);

        let cells = &groups[0].cells;
        assert_eq!(cells.len(), 3, "the bar needs every record");
        assert!((cells[0].start - 0.0).abs() < 1e-6);
        assert!((cells[0].score - 0.80).abs() < 1e-6);
        assert!((cells[1].start - 0.25).abs() < 1e-6);
        assert!((cells[1].score - 0.30).abs() < 1e-6);
        assert!((cells[2].start - 0.5).abs() < 1e-6);
        assert!((cells[2].score - 0.45).abs() < 1e-6);
    }

    /// The bar takes the finest records of the video.
    ///
    /// A frame cell is 0.25 s and an audio cell is 1 s, so the two overlap.
    /// One bar holds one of them, and the frames give the finer picture.
    #[test]
    fn the_bar_uses_the_frames_when_the_video_has_them() {
        let records = vec![
            record_of(crate::store::KIND_FRAME, "a.mp4", 0.0, 0.25),
            record_of(crate::store::KIND_FRAME, "a.mp4", 0.25, 0.5),
            record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0),
        ];
        let ranked = vec![(0, 0.80), (2, 0.70), (1, 0.60)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].cells.len(), 2, "the frames cover the bar");
        assert!((groups[0].cells[0].end - 0.25).abs() < 1e-6);
        assert!((groups[0].cells[1].end - 0.5).abs() < 1e-6);
    }

    /// A video with no frame record still gets a bar.
    #[test]
    fn the_bar_falls_back_to_the_audio_records_of_an_old_store() {
        let records = vec![
            record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0),
            record_of(crate::store::KIND_AUDIO, "a.mp4", 1.0, 2.0),
        ];
        let ranked = vec![(0, 0.80), (1, 0.70)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups[0].cells.len(), 2);
        assert!((groups[0].cells[0].score - 0.80).abs() < 1e-6);
        assert!((groups[0].cells[1].score - 0.70).abs() < 1e-6);
    }

    #[test]
    fn group_hits_caps_each_group_at_fifty() {
        // The records must not touch. Touching records merge into one moment,
        // and the cap would never bite.
        let records: Vec<Record> = (0..60)
            .map(|i| record("a.mp4", i as f32 * 10.0, i as f32 * 10.0 + 4.0))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..60).map(|i| (i, 0.9 - i as f32 * 0.001)).collect();

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hits.len(), 50);
        assert_eq!(groups[0].duration, 594.0);
    }

    #[test]
    fn group_hits_returns_nothing_when_every_score_is_below_the_floor() {
        let records = vec![record("a.mp4", 0.0, 4.0)];
        assert!(group_hits(&records, &[(0, 0.39)], 0.0).is_empty());
        assert!(group_hits(&[], &[], 0.0).is_empty());
    }

    #[test]
    fn the_median_of_an_odd_list_is_the_middle_value() {
        let ranked = vec![(0, 0.1), (1, 0.9), (2, 0.5)];
        assert!((median(&ranked) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_median_of_an_even_list_is_the_middle_pair() {
        let ranked = vec![(0, 0.1), (1, 0.3), (2, 0.5), (3, 0.9)];
        assert!((median(&ranked) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn the_median_of_an_empty_list_is_zero() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn the_cut_follows_the_best_score() {
        assert!((weak_cut(0.9) - 0.70).abs() < 1e-6);
        assert!(
            (weak_cut(0.5) - FLOOR).abs() < 1e-6,
            "a weak best score leaves the floor in place"
        );
    }

    #[test]
    fn a_score_below_the_cut_is_weak() {
        assert!(is_weak(0.60, 0.70));
        assert!(!is_weak(0.70, 0.70), "the cut itself stands");
        assert!(!is_weak(0.80, 0.70));
    }

    #[test]
    fn group_hits_reports_the_margin_over_the_baseline() {
        // The two records must not touch. Touching records merge, and the
        // second hit would not exist.
        let records = vec![record("a.mp4", 0.0, 4.0), record("a.mp4", 10.0, 14.0)];
        let ranked = vec![(1, 0.70), (0, 0.65)];
        let groups = group_hits(&records, &ranked, 0.60);
        assert_eq!(groups[0].hits[0].score, 0.70);
        assert!((groups[0].hits[0].margin - 0.10).abs() < 1e-6);
        assert!((groups[0].hits[1].margin - 0.05).abs() < 1e-6);
    }

    /// A contract pin for `index_video`.
    ///
    /// `index_video` needs the model, so it has no unit test of its own. This
    /// test pins the callback contract that it obeys: the number counts from 1,
    /// and it reaches the item count on the last item.
    #[test]
    fn the_callback_numbering_matches_the_item_plan() {
        let plan: Vec<f64> = frames(30.0)
            .into_iter()
            .chain(audio_starts(30.0))
            .collect();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let mut report = |n: usize, total: usize| seen.push((n, total));
        for (n, _) in plan.iter().enumerate() {
            report(n + 1, plan.len());
        }
        assert_eq!(seen.len(), 150);
        assert_eq!(seen[0], (1, 150));
        assert_eq!(seen[149], (150, 150));
    }

    /// A video with no audio track has an empty audio plan.
    ///
    /// The run must then end after the last frame. It must not ask the plan
    /// for an audio clip, because that plan holds no item.
    #[test]
    fn the_run_ends_after_the_frames_when_the_audio_plan_is_empty() {
        let mut run = IndexRun {
            video: PathBuf::from("silent.mp4"),
            label: "silent.mp4".to_string(),
            duration: 1.0,
            frames: vec![0.0, 0.25, 0.5, 0.75],
            audios: Vec::new(),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
        };

        for _ in 0..run.frames.len() {
            run.advance();
        }

        assert!(run.phase == Phase::Done);
        assert_eq!(run.remaining_in_batch(), 0);
        assert!(run.cut_batch().is_ok());
        assert!(run.phase == Phase::Done);
    }

    #[test]
    fn start_counts_the_frames_and_the_audio_clips_of_the_test_video() {
        let video =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spike/testvideo_nolabels.mp4");
        if !video.exists() || !Path::new(crate::media::FFPROBE).exists() {
            eprintln!("skipped: ffprobe or the test video is missing");
            return;
        }
        let format = crate::media::probe_duration(&video).unwrap();
        let video_len = crate::media::probe_stream_duration(&video, "v")
            .unwrap()
            .unwrap_or(format)
            .min(format);
        let audio_len = match crate::media::probe_stream_duration(&video, "a").unwrap() {
            Some(len) => len.min(format),
            None => 0.0,
        };
        let run = IndexRun::start(&video).unwrap();

        assert_eq!(
            run.total(),
            frames(video_len).len() + audio_starts(audio_len).len()
        );
        assert!(run.total() > 100, "got {}", run.total());
        // The test video holds about 23 s of audio inside a 30 s file. A clip
        // past the end of the audio track holds no sound, so the plan stops at
        // the track length.
        assert!(audio_len + 1.0 < format, "the track is not short here");
        assert!(
            run.total() < frames(video_len).len() + audio_starts(format).len(),
            "the plan asks for a clip past the end of the audio track"
        );
    }

    /// A run of touching frames is one moment, not many hits.
    #[test]
    fn touching_hits_join_one_moment() {
        let hits = vec![
            Hit { start: 0.0, end: 0.25, score: 0.70, margin: 0.10 },
            Hit { start: 0.25, end: 0.5, score: 0.80, margin: 0.20 },
            Hit { start: 0.5, end: 0.75, score: 0.60, margin: 0.05 },
        ];
        let moments = merge_hits(hits);

        assert_eq!(moments.len(), 1, "the run is one moment");
        assert!((moments[0].start - 0.0).abs() < 1e-6);
        assert!((moments[0].end - 0.75).abs() < 1e-6);
        assert!((moments[0].score - 0.80).abs() < 1e-6);
        assert!((moments[0].margin - 0.20).abs() < 1e-6);
    }

    /// Two runs with a hole between them stay apart.
    #[test]
    fn a_gap_of_two_frame_steps_keeps_two_moments() {
        let hits = vec![
            Hit { start: 0.0, end: 0.25, score: 0.80, margin: 0.20 },
            Hit { start: 1.0, end: 1.25, score: 0.70, margin: 0.10 },
        ];
        let moments = merge_hits(hits);

        assert_eq!(moments.len(), 2, "the hole splits the run");
    }

    /// The first hit of the list is the best moment.
    #[test]
    fn the_moments_come_back_best_first() {
        let hits = vec![
            Hit { start: 0.0, end: 0.25, score: 0.60, margin: 0.01 },
            Hit { start: 2.0, end: 2.25, score: 0.90, margin: 0.30 },
        ];
        let moments = merge_hits(hits);

        assert_eq!(moments.len(), 2);
        assert!((moments[0].score - 0.90).abs() < 1e-6);
        assert!((moments[1].score - 0.60).abs() < 1e-6);
    }

    #[test]
    fn one_hit_stays_one_hit() {
        let hits = vec![Hit { start: 1.0, end: 1.25, score: 0.5, margin: 0.0 }];
        assert_eq!(merge_hits(hits).len(), 1);
        assert!(merge_hits(Vec::new()).is_empty());
    }

    /// Audio records are 1 s long and they touch, so they join.
    ///
    /// A store from before the frame records holds audio only. The merge then
    /// gives one moment over the whole matching run. That is the honest answer
    /// for a 1 s grid, so the behaviour is pinned here.
    #[test]
    fn touching_audio_hits_join_one_moment() {
        let hits = vec![
            Hit { start: 0.0, end: 1.0, score: 0.70, margin: 0.10 },
            Hit { start: 1.0, end: 2.0, score: 0.75, margin: 0.15 },
            Hit { start: 2.0, end: 3.0, score: 0.72, margin: 0.12 },
        ];
        assert_eq!(merge_hits(hits).len(), 1);
    }

    /// The merge runs before the cap, so one video may show many moments.
    #[test]
    fn a_group_joins_its_run_of_frames_into_one_moment() {
        let records: Vec<Record> = (0..4)
            .map(|i| record("a.mp4", i as f32 * 0.25, i as f32 * 0.25 + 0.25))
            .collect();
        let ranked = vec![(0, 0.90), (1, 0.85), (2, 0.80), (3, 0.75)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hits.len(), 1, "four touching frames make one moment");
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_rate_is_items_a_second() {
        assert!((per_second(20, 10.0) - 2.0).abs() < 1e-6);
        assert!((per_second(207, 29.0) - 7.137_931).abs() < 1e-4);
    }

    /// A zero time must not divide by zero.
    #[test]
    fn a_zero_time_gives_a_zero_rate() {
        assert_eq!(per_second(10, 0.0), 0.0);
        assert_eq!(per_second(10, -1.0), 0.0);
        assert_eq!(per_second(0, 10.0), 0.0);
    }

    /// The page reads these fields, so the shape is a contract.
    #[test]
    fn the_progress_carries_the_speed_fields() {
        let json = serde_json::to_value(Progress::default()).unwrap();
        assert_eq!(json["device"], "");
        assert_eq!(json["gpu_available"], false);
        assert_eq!(json["load_ms"], 0);
        assert_eq!(json["elapsed_ms"], 0);
        assert_eq!(json["per_second"], 0.0);
    }

    /// A factor of four keeps one frame of every four, and keeps the audio.
    #[test]
    fn thinning_keeps_one_frame_of_each_group() {
        let mut records: Vec<Record> = (0..8)
            .map(|i| record("a.mp4", i as f32 * 0.25, i as f32 * 0.25 + 0.25))
            .collect();
        records.push(record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0));
        records.push(record_of(crate::store::KIND_AUDIO, "a.mp4", 1.0, 2.0));

        let kept = thin_indexes(&records, 4);

        assert_eq!(kept.len(), 4, "two frames and two audio records");
        let starts: Vec<f32> = kept.iter().map(|at| records[*at].start).collect();
        assert_eq!(starts, vec![0.0, 1.0, 0.0, 1.0]);
    }

    /// A factor of one keeps every record.
    #[test]
    fn a_factor_of_one_keeps_everything() {
        let records = vec![record("a.mp4", 0.0, 0.25), record("a.mp4", 0.25, 0.5)];
        assert_eq!(thin_indexes(&records, 1).len(), 2);
        assert_eq!(thin_indexes(&records, 0).len(), 2);
    }

    /// Each video thins on its own, so a short video keeps its first frame.
    #[test]
    fn each_video_thins_on_its_own() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("b.mp4", 0.0, 0.25),
            record("a.mp4", 0.25, 0.5),
            record("b.mp4", 0.25, 0.5),
        ];
        let kept = thin_indexes(&records, 2);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|at| records[*at].label == "a.mp4"));
        assert!(kept.iter().any(|at| records[*at].label == "b.mp4"));
    }

    /// One video thins as one set, whatever separator its label carries.
    #[test]
    fn thinning_gathers_a_video_across_path_separators() {
        let records = vec![
            record(r"D:\video\a.mp4", 0.0, 0.25),
            record("D:/video/a.mp4", 0.25, 0.5),
            record(r"D:\video\a.mp4", 0.5, 0.75),
        ];

        let kept = thin_indexes(&records, 2);

        assert_eq!(kept, vec![0, 2], "the two forms make one time order");
    }

    /// The thinning gives the indexes of the kept records, in store order.
    #[test]
    fn the_thinning_gives_the_indexes_of_the_kept_records() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("a.mp4", 0.25, 0.5),
            record("a.mp4", 0.5, 0.75),
            record("a.mp4", 0.75, 1.0),
        ];

        let kept = thin_indexes(&records, 2);

        assert_eq!(kept, vec![0, 2]);
    }
}
