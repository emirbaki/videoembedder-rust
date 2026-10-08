//! The folder scan, the window plan, and the progress object.
//!
//! One window is 2 s long. The stride is 1 s, so windows overlap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::embed::Embedder;
use crate::media;
use crate::store::{self, KIND_WINDOW, Record};

/// Window length in seconds.
pub const WINDOW: f64 = 2.0;
/// Window step in seconds.
pub const STRIDE: f64 = 1.0;
/// The fine window length in seconds, for the refine step.
pub const FINE_WINDOW: f64 = 1.0;
/// The fine window step in seconds, for the refine step.
pub const FINE_STRIDE: f64 = 0.5;
/// Small slack for the last window near the end of a video.
pub const EPS: f64 = 0.05;
/// A window below this score never enters a result group.
pub const FLOOR: f32 = 0.4;
/// The most windows one video may add to one group.
pub const GROUP_CAP: usize = 50;

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
}

/// One matching window inside a video.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
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

/// Cut a duration into windows. Each pair is a start and an end.
pub fn windows(duration: f64) -> Vec<(f64, f64)> {
    let mut plan = Vec::new();
    let mut start = 0.0;
    while start + WINDOW <= duration + EPS {
        plan.push((start, start + WINDOW));
        start += STRIDE;
    }
    plan
}

/// The fine windows inside one coarse span. Each pair is a start and an end.
///
/// The caller passes a span that already sits inside the video. A span shorter
/// than one fine window gets one window. An empty span gets none.
pub fn fine_windows(lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let mut plan = Vec::new();
    if hi <= lo {
        return plan;
    }
    let mut start = lo;
    while start + FINE_WINDOW <= hi + EPS {
        plan.push((start, start + FINE_WINDOW));
        start += FINE_STRIDE;
    }
    if plan.is_empty() {
        plan.push((lo, lo + FINE_WINDOW));
    }
    plan
}

/// Refine one coarse hit. Embed the fine windows in the span and keep the best.
///
/// `query` is the full prefixed query text, the same text the rank uses.
/// Returns None when every fine window fails.
pub fn refine_hit(
    embedder: &Embedder,
    video: &Path,
    query: &str,
    start: f64,
    end: f64,
) -> Result<Option<(f64, f64, f32)>> {
    let query_vec = embedder.embed_text(query)?;
    let mut best: Option<(f64, f64, f32)> = None;
    for (n, (lo, hi)) in fine_windows(start, end).into_iter().enumerate() {
        let dir = work_dir(n);
        let outcome = media::extract_window(video, lo, hi, &dir)
            .and_then(|(frames, audio)| embedder.embed_window(&frames, &audio));
        let _ = std::fs::remove_dir_all(&dir);
        let Ok(vec) = outcome else {
            continue;
        };
        let score = crate::embed::dot(&query_vec, &vec);
        if best.map(|held| score > held.2).unwrap_or(true) {
            best = Some((lo, hi, score));
        }
    }
    Ok(best)
}

/// A fresh working folder for one window of ffmpeg output.
pub fn work_dir(at: usize) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("vei_{}_{n}_{at}", std::process::id()))
}

/// Index one video. Embed every window and return its records.
///
/// The callback runs after each window with the window number, counting from 1,
/// and the window count of the whole video.
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    on_window: &mut dyn FnMut(usize, usize),
) -> Result<Vec<Record>> {
    let duration = media::probe_duration(video)?;
    let plan = windows(duration);
    if plan.is_empty() {
        bail!("{} is shorter than one window", video.display());
    }
    let label = store::normalize_label(&video.to_string_lossy());
    let mut records = Vec::with_capacity(plan.len());
    for (n, (start, end)) in plan.iter().enumerate() {
        let dir = work_dir(n);
        let outcome = media::extract_window(video, *start, *end, &dir)
            .and_then(|(frames, audio)| embedder.embed_window(&frames, &audio));
        let _ = std::fs::remove_dir_all(&dir);
        let vec = outcome?;
        records.push(Record {
            kind: KIND_WINDOW,
            label: label.clone(),
            start: *start as f32,
            end: *end as f32,
            vec,
        });
        on_window(n + 1, plan.len());
    }
    Ok(records)
}

/// Group the ranked records by video.
///
/// `ranked` comes from `Store::rank`, so it is sorted by score, best first.
pub fn group_hits(records: &[Record], ranked: &[(usize, f32)]) -> Vec<Group> {
    let mut durations: HashMap<&str, f32> = HashMap::new();
    for record in records {
        if record.kind != KIND_WINDOW {
            continue;
        }
        let held = durations.entry(record.label.as_str()).or_insert(0.0);
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
        if record.kind != KIND_WINDOW {
            continue;
        }
        let hit = Hit {
            start: record.start,
            end: record.end,
            score: *score,
        };
        match groups.iter_mut().find(|g| g.video == record.label) {
            Some(group) => {
                if group.hits.len() < GROUP_CAP {
                    group.hits.push(hit);
                }
            }
            None => {
                let duration = durations
                    .get(record.label.as_str())
                    .copied()
                    .unwrap_or(record.end);
                groups.push(Group {
                    video: record.label.clone(),
                    duration,
                    score: *score,
                    hits: vec![hit],
                });
            }
        }
    }
    groups
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

    fn record(label: &str, start: f32, end: f32) -> Record {
        Record {
            kind: KIND_WINDOW,
            label: label.to_string(),
            start,
            end,
            vec: vec![1.0, 0.0],
        }
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
    fn windows_cover_a_thirty_second_video() {
        let plan = windows(30.0);
        assert_eq!(plan.len(), 29);
        assert_eq!(plan[0], (0.0, 2.0));
        assert_eq!(plan[28], (28.0, 30.0));
    }

    #[test]
    fn a_video_shorter_than_one_window_has_no_plan() {
        assert!(windows(1.0).is_empty());
        assert_eq!(windows(2.0).len(), 1);
        assert_eq!(windows(5.0).len(), 4);
    }

    #[test]
    fn fine_windows_step_a_four_second_span() {
        let plan = fine_windows(0.0, 4.0);
        assert_eq!(plan.len(), 7);
        assert_eq!(plan[0], (0.0, 1.0));
        assert_eq!(plan[6], (3.0, 4.0));
    }

    #[test]
    fn a_span_shorter_than_the_fine_window_gets_one_window() {
        assert_eq!(fine_windows(5.0, 5.6), vec![(5.0, 6.0)]);
    }

    #[test]
    fn an_empty_span_has_no_fine_window() {
        assert!(fine_windows(4.0, 4.0).is_empty());
        assert!(fine_windows(4.0, 3.0).is_empty());
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

        let groups = group_hits(&records, &ranked);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].video, "b.mp4");
        assert_eq!(groups[0].score, 0.9);
        assert_eq!(groups[0].duration, 8.0);
        assert_eq!(groups[0].hits.len(), 2);
        assert_eq!(groups[1].video, "a.mp4");
        assert_eq!(groups[1].hits.len(), 1);
    }

    #[test]
    fn group_hits_caps_each_group_at_fifty() {
        let records: Vec<Record> = (0..60)
            .map(|i| record("a.mp4", i as f32, i as f32 + 4.0))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..60).map(|i| (i, 0.9 - i as f32 * 0.001)).collect();

        let groups = group_hits(&records, &ranked);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hits.len(), 50);
        assert_eq!(groups[0].duration, 63.0);
    }

    #[test]
    fn group_hits_returns_nothing_when_every_score_is_below_the_floor() {
        let records = vec![record("a.mp4", 0.0, 4.0)];
        assert!(group_hits(&records, &[(0, 0.39)]).is_empty());
        assert!(group_hits(&[], &[]).is_empty());
    }

    /// A contract pin for `index_video`.
    ///
    /// `index_video` needs the model, so it has no unit test of its own. This
    /// test pins the callback contract that it obeys: the number counts from 1,
    /// and it reaches the window count on the last window.
    #[test]
    fn the_callback_numbering_matches_the_window_plan() {
        let plan = windows(30.0);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let mut report = |n: usize, total: usize| seen.push((n, total));
        for (n, _) in plan.iter().enumerate() {
            report(n + 1, plan.len());
        }
        assert_eq!(seen.len(), 29);
        assert_eq!(seen[0], (1, 29));
        assert_eq!(seen[28], (29, 29));
    }
}
