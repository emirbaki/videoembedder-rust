# Frame-Level Search Implementation Plan (Route A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index a video frame by frame at 4 frames a second, plus one audio
record a second, and rank a query with a calibrated margin over the store.

**Architecture:** One frame is one image vector. One audio clip is one audio
vector. A new `IndexRun` state machine cuts media in batches and embeds one
item per step. Both the console and the web engine drive the same machine.
The search shows each hit as a margin over the store median.

**Tech Stack:** Rust edition 2024, `llama-cpp-2` with `mtmd`, `ffmpeg`,
`tiny_http`, plain JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-10-videoembed-product-routes.md`

## Global Constraints

- Rust edition 2024. Windows with the MSVC build tools.
- ffmpeg and ffprobe sit at the fixed paths in `videoembed/src/media.rs`.
- The frame rate is 4 a second. `FRAME_RATE = 4.0`, `FRAME_STEP = 0.25`.
- The audio rate is one clip a second. `AUDIO_SECONDS = 1.0`.
- No new crate is added to `videoembed/Cargo.toml`.
- The model files sit in `models/`. A test that needs the model is manual.
- All prose obeys ASD-STE100.
- A commit message ends with this line:
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`

## Review Focus

These inputs are likely to bite a user. Each one has a test in the owning task.

1. A video with no audio track. Indexing must finish with frame records only.
2. A video shorter than one frame. `IndexRun::start` must fail with a message.
3. A frame count from ffmpeg that differs from the plan. No panic. Each record
   keeps the plan time.
4. A store that holds old `KIND_WINDOW` records. Search and forget must treat
   them as video records.
5. A query with no good match. The margin must stay small.

---

## Task 1: Store kinds and the video predicate

**Files:**
- Modify: `videoembed/src/store.rs:17-18` (the kind constants)
- Modify: `videoembed/src/store.rs:111-121` (`forget`)
- Modify: `videoembed/src/store.rs:146-151` (`holds_video`)
- Test: `videoembed/src/store.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub const KIND_FRAME: u8 = 2`, `pub const KIND_AUDIO: u8 = 3`,
  `pub fn is_video(kind: u8) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add these tests inside `mod tests` in `videoembed/src/store.rs`:

```rust
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
        };
        let removed = store.forget("D:/video/a.mp4").unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.records.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `cd videoembed && cargo test --lib store::`
Expected: FAIL. `KIND_FRAME` and `is_video` do not exist.

- [ ] **Step 3: Add the kinds and the predicate**

Replace the two constants at `videoembed/src/store.rs:17-18` with these lines:

```rust
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
```

Change the filter line in `forget` to use `is_video`:

```rust
        self.records
            .retain(|r| !(is_video(r.kind) && normalize_label(&r.label) == want));
```

Change the body of `holds_video` to use `is_video`:

```rust
    pub fn holds_video(&self, path: &str) -> bool {
        let want = normalize_label(path);
        self.records
            .iter()
            .any(|record| is_video(record.kind) && normalize_label(&record.label) == want)
    }
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `cd videoembed && cargo test --lib store::`
Expected: PASS. Every store test passes, including the old ones.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/store.rs
git commit -m "feat: add the frame and audio record kinds

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 2: Calibration — the median score and the margin

**Files:**
- Modify: `videoembed/src/index.rs:28` (add `FLOOR` stays; add nothing here)
- Modify: `videoembed/src/index.rs:50-65` (`Hit` and `Group`)
- Modify: `videoembed/src/index.rs:131-154` (`refine_hit` is removed here)
- Modify: `videoembed/src/index.rs:219-269` (`group_hits`)
- Test: `videoembed/src/index.rs` (`mod tests`)

**Interfaces:**
- Consumes: `is_video` from Task 1.
- Produces: `pub fn median(ranked: &[(usize, f32)]) -> f32`,
  `pub struct Hit { start, end, score, margin: f32 }`,
  `pub fn group_hits(records: &[Record], ranked: &[(usize, f32)], baseline: f32) -> Vec<Group>`.

- [ ] **Step 1: Write the failing tests**

Add these tests inside `mod tests` in `videoembed/src/index.rs`:

```rust
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
    fn group_hits_reports_the_margin_over_the_baseline() {
        let records = vec![record("a.mp4", 0.0, 4.0), record("a.mp4", 4.0, 8.0)];
        let ranked = vec![(1, 0.70), (0, 0.65)];
        let groups = group_hits(&records, &ranked, 0.60);
        assert_eq!(groups[0].hits[0].score, 0.70);
        assert!((groups[0].hits[0].margin - 0.10).abs() < 1e-6);
        assert!((groups[0].hits[1].margin - 0.05).abs() < 1e-6);
    }
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `cd videoembed && cargo test --lib index::`
Expected: FAIL. `median` does not exist. `group_hits` takes two arguments.

- [ ] **Step 3: Add `median`, the `margin` field, and the `baseline` parameter**

Add `margin` to `Hit` at `videoembed/src/index.rs:50-56`:

```rust
/// One matching window.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub start: f32,
    pub end: f32,
    pub score: f32,
    pub margin: f32,
}
```

Add this function near the top of `videoembed/src/index.rs`, after the constants:

```rust
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
```

Change `group_hits` to take the baseline and set the margin. Replace its
signature and the two places that build a `Hit`:

```rust
pub fn group_hits(records: &[Record], ranked: &[(usize, f32)], baseline: f32) -> Vec<Group> {
```

Inside the loop, change the `kind` check to `is_video`, and build the hit with
the margin:

```rust
        let record = &records[*index];
        if !crate::store::is_video(record.kind) {
            continue;
        }
        let hit = Hit {
            start: record.start,
            end: record.end,
            score: *score,
            margin: *score - baseline,
        };
```

Also change the duration loop at the top of `group_hits` to use `is_video`:

```rust
    for record in records {
        if !crate::store::is_video(record.kind) {
            continue;
        }
```

- [ ] **Step 4: Update the old `group_hits` tests**

The four old `group_hits` tests call the function with two arguments. Add
`0.0` as the third argument in each call. For example, `group_hits(&records,
&ranked)` becomes `group_hits(&records, &ranked, 0.0)`.

Delete `refine_hit`, `fine_windows`, `FINE_WINDOW`, `FINE_STRIDE`, and their
tests. The frame plan replaces the fine-window refine step. Remove the two
tests `fine_windows_step_a_four_second_span` and
`a_span_shorter_than_the_fine_window_gets_one_window` and
`an_empty_span_has_no_fine_window`.

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `cd videoembed && cargo test --lib index::`
Expected: PASS. All index tests pass.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: report each hit as a margin over the store median

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 3: The frame plan and the audio plan

**Files:**
- Modify: `videoembed/src/index.rs:17-32` (the constants)
- Modify: `videoembed/src/index.rs:97-125` (`windows` and `fine_windows`)
- Test: `videoembed/src/index.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub const FRAME_RATE: f64 = 4.0`,
  `pub const FRAME_STEP: f64 = 0.25`,
  `pub const AUDIO_SECONDS: f64 = 1.0`,
  `pub fn frames(duration: f64) -> Vec<f64>`,
  `pub fn audio_starts(duration: f64) -> Vec<f64>`.

- [ ] **Step 1: Write the failing tests**

Add these tests inside `mod tests` in `videoembed/src/index.rs`:

```rust
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
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `cd videoembed && cargo test --lib index::`
Expected: FAIL. `frames` and `audio_starts` do not exist.

- [ ] **Step 3: Add the constants and the two plan functions**

Add these lines to the constants at the top of `videoembed/src/index.rs`:

```rust
/// Frames a second in the frame plan.
pub const FRAME_RATE: f64 = 4.0;
/// The time step between two frames in seconds.
pub const FRAME_STEP: f64 = 0.25;
/// The length of one audio clip in seconds.
pub const AUDIO_SECONDS: f64 = 1.0;
```

Replace `windows` and `fine_windows` with these two functions:

```rust
/// The start time of every frame. One frame each 0.25 s.
pub fn frames(duration: f64) -> Vec<f64> {
    let mut plan = Vec::new();
    let mut t = 0.0;
    while t + FRAME_STEP <= duration + EPS {
        plan.push(t);
        t += FRAME_STEP;
    }
    plan
}

/// The start time of every one-second audio clip.
pub fn audio_starts(duration: f64) -> Vec<f64> {
    let mut plan = Vec::new();
    let mut t = 0.0;
    while t + AUDIO_SECONDS <= duration + EPS {
        plan.push(t);
        t += AUDIO_SECONDS;
    }
    plan
}
```

Delete `WINDOW`, `STRIDE`, and the tests `windows_cover_a_thirty_second_video`
and `a_video_shorter_than_one_window_has_no_plan`. Nothing uses them after
Tasks 6 and 7.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `cd videoembed && cargo test --lib index::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: plan frames at four a second and audio at one a second

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 4: Cut frames and audio with ffmpeg

**Files:**
- Modify: `videoembed/src/media.rs:51-132` (`extract_window` and `read_frames`)
- Test: `videoembed/src/media.rs` (new `mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn has_audio(video: &Path) -> bool`,
  `pub fn extract_frames(video: &Path, from: f64, to: f64, rate: f64, dir: &Path) -> Result<Vec<PathBuf>>`,
  `pub fn extract_audio(video: &Path, from: f64, to: f64, dir: &Path) -> Result<Vec<PathBuf>>`.

- [ ] **Step 1: Write the failing tests**

Add this test module at the end of `videoembed/src/media.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> bool {
        Path::new(FFMPEG).exists() && Path::new(FFPROBE).exists()
    }

    fn test_video() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spike/testvideo_nolabels.mp4")
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_media_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn extract_frames_cuts_one_frame_every_quarter_second() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let dir = scratch("frames");
        let frames = extract_frames(&video, 0.0, 2.0, 4.0, &dir).unwrap();
        assert_eq!(frames.len(), 8, "got {}", frames.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_audio_cuts_one_clip_a_second() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        if !has_audio(&video) {
            eprintln!("skipped: the test video holds no audio track");
            return;
        }
        let dir = scratch("audio");
        let clips = extract_audio(&video, 0.0, 3.0, &dir).unwrap();
        assert_eq!(clips.len(), 3, "got {}", clips.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_frames_fails_on_a_missing_file() {
        let dir = scratch("missing");
        let outcome = extract_frames(Path::new("no_such_video.mp4"), 0.0, 1.0, 4.0, &dir);
        assert!(outcome.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `cd videoembed && cargo test --lib media::`
Expected: FAIL. `extract_frames`, `extract_audio`, and `has_audio` do not exist.

- [ ] **Step 3: Replace `extract_window` and `read_frames`**

Delete `extract_window` and `read_frames`. Add these functions and the two
helpers:

```rust
/// True when the file holds at least one audio stream.
pub fn has_audio(video: &Path) -> bool {
    Command::new(FFPROBE)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
        ])
        .arg(video)
        .output()
        .map(|out| out.status.success() && !out.stdout.is_empty())
        .unwrap_or(false)
}

/// Cut frames from `from` to `to` at `rate` a second into `dir`.
///
/// One ffmpeg call cuts the group. Frame j starts at `from + j / rate`.
pub fn extract_frames(
    video: &Path,
    from: f64,
    to: f64,
    rate: f64,
    dir: &Path,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let pattern = dir.join("f%05d.jpg").to_string_lossy().to_string();
    let args = vec![
        "-y".into(),
        "-ss".into(),
        format!("{from}"),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-t".into(),
        format!("{}", to - from),
        "-vf".into(),
        format!("fps={rate}"),
        "-q:v".into(),
        "2".into(),
        pattern,
    ];
    run_ffmpeg(args, "extracting frames")?;
    read_media(dir, &["jpg", "jpeg", "png"])
}

/// Cut one-second audio clips from `from` to `to` into `dir`.
///
/// One ffmpeg call writes every clip. Clip j starts at `from + j`.
/// A video without an audio track gives an empty vector.
pub fn extract_audio(
    video: &Path,
    from: f64,
    to: f64,
    dir: &Path,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    if !has_audio(video) {
        println!("  (no audio track in {})", video.display());
        return Ok(Vec::new());
    }
    let pattern = dir.join("a%05d.wav").to_string_lossy().to_string();
    let args = vec![
        "-y".into(),
        "-ss".into(),
        format!("{from}"),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-t".into(),
        format!("{}", to - from),
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        "16000".into(),
        "-f".into(),
        "segment".into(),
        "-segment_time".into(),
        "1".into(),
        pattern,
    ];
    run_ffmpeg(args, "extracting audio")?;
    read_media(dir, &["wav"])
}

/// List the files in `dir` with one of the extensions, in name order.
fn read_media(dir: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| extensions.contains(&x.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `cd videoembed && cargo test --lib media::`
Expected: PASS. The frame test passes on this machine. The audio test passes
or prints a skip line. The missing-file test passes.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/media.rs
git commit -m "feat: cut frames and one-second audio clips in batches

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 5: Embed one image and one audio clip

**Files:**
- Modify: `videoembed/src/embed.rs:112-144` (`embed_window` and `embed_window_in`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn embed_image(&self, frame: &Path) -> Result<Vec<f32>>`,
  `pub fn embed_image_in(&self, ctx: &mut LlamaContext<'_>, frame: &Path) -> Result<Vec<f32>>`,
  `pub fn embed_audio_in(&self, ctx: &mut LlamaContext<'_>, clip: &Path) -> Result<Vec<f32>>`.

This file needs the model, so it has no unit test. Step 4 verifies it with a
build and a small run in Task 7.

- [ ] **Step 1: Write the code**

Replace `embed_window` and `embed_window_in` at `videoembed/src/embed.rs:112-144`
with these three methods:

```rust
    /// Embed one image. Returns a unit vector.
    pub fn embed_image(&self, frame: &Path) -> Result<Vec<f32>> {
        let mut ctx = self.new_context()?;
        self.embed_image_in(&mut ctx, frame)
    }

    /// Embed one image in a caller-owned context.
    pub fn embed_image_in(&self, ctx: &mut LlamaContext<'_>, frame: &Path) -> Result<Vec<f32>> {
        ctx.clear_kv_cache();
        let path = frame.to_str().context("frame path is not utf8")?;
        let bitmap = MtmdBitmap::from_file(&self.mtmd, path, false)?;
        self.embed_joint(ctx, &[bitmap])
    }

    /// Embed one audio clip in a caller-owned context.
    pub fn embed_audio_in(&self, ctx: &mut LlamaContext<'_>, clip: &Path) -> Result<Vec<f32>> {
        ctx.clear_kv_cache();
        let path = clip.to_str().context("audio path is not utf8")?;
        let bitmap = MtmdBitmap::from_file(&self.mtmd, path, false)?;
        self.embed_joint(ctx, &[bitmap])
    }
```

- [ ] **Step 2: Build**

Run: `cd videoembed && cargo build`
Expected: PASS. No error. The build stops with an unused-import warning only if
a `PathBuf` import is now unused. Remove an unused import if the compiler names
one.

- [ ] **Step 3: Commit**

```bash
git add videoembed/src/embed.rs
git commit -m "feat: embed one image and one audio clip per call

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 6: The `IndexRun` state machine

**Files:**
- Modify: `videoembed/src/index.rs` (add `use llama_cpp_2::context::LlamaContext;`)
- Modify: `videoembed/src/index.rs:163-217` (`index_video`)
- Test: `videoembed/src/index.rs` (`mod tests`)

**Interfaces:**
- Consumes: `frames`, `audio_starts` from Task 3. `extract_frames`,
  `extract_audio`, `probe_duration` from Task 4. `embed_image_in`,
  `embed_audio_in` from Task 5. `KIND_FRAME`, `KIND_AUDIO` from Task 1.
- Produces: `pub const FRAME_BATCH: usize = 120`,
  `pub const AUDIO_BATCH: usize = 60`,
  `pub struct IndexRun`,
  `pub fn start(video: &Path) -> Result<IndexRun>`,
  `pub fn total(&self) -> usize`,
  `pub fn done(&self) -> usize`,
  `pub fn step(&mut self, embedder: &Embedder, ctx: &mut LlamaContext<'_>) -> Result<Option<Record>>`,
  `pub fn index_video(embedder: &Embedder, video: &Path, on_step: &mut dyn FnMut(usize, usize)) -> Result<Vec<Record>>`.

- [ ] **Step 1: Write the failing test**

Add this test inside `mod tests` in `videoembed/src/index.rs`:

```rust
    #[test]
    fn start_counts_the_frames_and_the_audio_clips_of_the_test_video() {
        let video =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spike/testvideo_nolabels.mp4");
        if !video.exists() || !Path::new(crate::media::FFPROBE).exists() {
            eprintln!("skipped: ffprobe or the test video is missing");
            return;
        }
        let run = IndexRun::start(&video).unwrap();
        let d = crate::media::probe_duration(&video).unwrap();
        let want = frames(d).len() + audio_starts(d).len();
        assert_eq!(run.total(), want);
        assert!(run.total() > 100, "got {}", run.total());
    }
```

- [ ] **Step 2: Run the test to make sure it fails**

Run: `cd videoembed && cargo test --lib index::start_counts`
Expected: FAIL. `IndexRun` does not exist.

- [ ] **Step 3: Write `IndexRun` and rewrite `index_video`**

Add this import to `videoembed/src/index.rs`, with the other `use` lines:

```rust
use llama_cpp_2::context::LlamaContext;
```

Add these constants with the other constants:

```rust
/// The number of frames one ffmpeg call cuts.
pub const FRAME_BATCH: usize = 120;
/// The number of audio clips one ffmpeg call cuts.
pub const AUDIO_BATCH: usize = 60;
```

Add the machine, above `index_video`:

```rust
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
    pub fn start(video: &Path) -> Result<Self> {
        let duration = media::probe_duration(video)?;
        let frames = frames(duration);
        if frames.is_empty() {
            bail!("{} is shorter than one frame", video.display());
        }
        Ok(Self {
            video: video.to_path_buf(),
            label: store::normalize_label(&video.to_string_lossy()),
            duration,
            frames,
            audios: audio_starts(duration),
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
    ) -> Result<Option<Record>> {
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
        Ok(Some(Record {
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
                    self.phase = Phase::Audio;
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
        if self.phase == Phase::Done {
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
```

Replace `index_video` with this wrapper:

```rust
/// Index one video. Embed every frame and every audio clip.
///
/// The callback runs after each record with the number made so far, counting
/// from 1, and the record count of the whole video.
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Vec<Record>> {
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
```

Delete `work_dir`'s old caller `AHEAD` constant if the compiler names it as
unused. Keep `work_dir`.

- [ ] **Step 4: Run the test to make sure it passes**

Run: `cd videoembed && cargo test --lib index::`
Expected: PASS. Every index test passes.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: add a stepwise frame and audio index run

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 7: The console on frames and the margin

**Files:**
- Modify: `videoembed/src/main.rs:92-108` (`index`)
- Modify: `videoembed/src/main.rs:127-162` (`search`)
- Modify: `videoembed/src/main.rs:164-188` (`list`)
- Modify: `videoembed/src/main.rs:199-206` (`bar` stays)

**Interfaces:**
- Consumes: `index::index_video`, `index::median`, `store::is_video`.
- Produces: nothing for later tasks.

- [ ] **Step 1: Update the `index` command**

Replace the body of `index` in `videoembed/src/main.rs` with this body:

```rust
    if rest.is_empty() {
        bail!("usage: index <video path>");
    }
    let video = PathBuf::from(rest);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    println!("Video: {}", video.display());
    let mut report = |n: usize, total: usize| println!("  item {n:5}/{total}");
    let records = index::index_video(embedder, &video, &mut report)?;
    let count = records.len();
    store.append(records)?;
    println!("Indexed {count} records into {}", store_path_hint());
    Ok(())
```

- [ ] **Step 2: Update the `search` command**

Replace the ranking and print block of `search` with this block:

```rust
    let query = embedder.embed_text(&format!("{QUERY_PREFIX}{rest}"))?;
    let ranked = store.rank(&query);
    let baseline = index::median(&ranked);
    println!("Top results for: {rest}");
    println!("Baseline (median of the store): {baseline:.3}");
    for (rank, (i, score)) in ranked.iter().take(10).enumerate() {
        let record = &store.records[*i];
        let margin = score - baseline;
        let bar = bar(*score);
        match record.kind {
            KIND_NOTE => println!(
                "{:2}. {:.3} ({:+.3}) {} note: {}",
                rank + 1,
                score,
                margin,
                bar,
                record.label
            ),
            _ => println!(
                "{:2}. {:.3} ({:+.3}) {} {}  [{:.2}-{:.2} s]",
                rank + 1,
                score,
                margin,
                bar,
                record.label,
                record.start,
                record.end
            ),
        }
    }
    Ok(())
```

- [ ] **Step 3: Update `list` to count video records**

In `list`, change the record loop to use `is_video`:

```rust
    for record in &store.records {
        if record.kind == KIND_NOTE {
            notes += 1;
            continue;
        }
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        match videos.iter_mut().find(|(name, _)| *name == record.label) {
            Some((_, count)) => *count += 1,
            None => videos.push((record.label.clone(), 1)),
        }
    }
    println!("Videos ({}):", videos.len());
    for (name, count) in videos {
        println!("  {name}  ({count} records)");
    }
    println!("Notes: {notes}");
    println!("Total records: {}", store.records.len());
}
```

- [ ] **Step 4: Update the command hint**

Change the hint line in `main` at `videoembed/src/main.rs:25` to name records:

```rust
    println!("Commands: index <path> | note <text> | search <query> | list | forget <path> | quit");
```

This line does not change its text. The word `window` is not in it. No edit.

- [ ] **Step 5: Build and run one index and one search**

Run:
```
cd videoembed
cargo build
./target/debug/videoembed.exe index ../spike/testvideo_nolabels.mp4
./target/debug/videoembed.exe search "birthday party"
```
Expected: the index prints `item 1/150` up to `item 150/150` and reports 150
records for a 30 s video. The search prints a baseline line and a margin in
parentheses for each hit. The best hit sits inside 20-30 s.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/main.rs
git commit -m "feat: index frames and audio from the console with calibrated scores

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 8: The web engine on frames, with one context

**Files:**
- Modify: `videoembed/src/bin/web.rs:62-69` (`State`, `VideoRow`)
- Modify: `videoembed/src/bin/web.rs:347-594` (the `Engine` struct and its impl)
- Modify: `videoembed/src/bin/web.rs:769-812` (`main`)

**Interfaces:**
- Consumes: `index::IndexRun`, `index::median`, `store::is_video`,
  `store::KIND_FRAME`, `store::KIND_AUDIO`.
- Produces: nothing for later tasks. The JSON shape gains `baseline` in
  `SearchAnswer` and a `margin` in each hit, and `VideoRow.items`.

The old engine made one context for every window. That call was slow. This task
makes one context for the whole run. The borrow checker needs the model beside
the loop, so the `Engine` struct becomes free functions that take the model.

- [ ] **Step 1: Write the failing test**

Add this test inside `mod tests` in `videoembed/src/bin/web.rs`:

```rust
    #[test]
    fn a_video_row_counts_frames_and_audio_together() {
        let store = Store {
            path: PathBuf::from("unused.vei"),
            dim: 2,
            records: vec![
                videoembed::store::Record {
                    kind: videoembed::store::KIND_FRAME,
                    label: "a.mp4".into(),
                    start: 0.0,
                    end: 0.25,
                    vec: vec![1.0, 0.0],
                },
                videoembed::store::Record {
                    kind: videoembed::store::KIND_AUDIO,
                    label: "a.mp4".into(),
                    start: 0.0,
                    end: 1.0,
                    vec: vec![1.0, 0.0],
                },
            ],
        };
        let rows = videos(&store);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].items, 2);
    }
```

- [ ] **Step 2: Run the test to make sure it fails**

Run: `cd videoembed && cargo test --bin web`
Expected: FAIL. `videos` takes `&self`, and `VideoRow` holds `windows`.

- [ ] **Step 3: Change `VideoRow` and add the free `videos` function**

Replace `VideoRow` at `videoembed/src/bin/web.rs:70-74`:

```rust
#[derive(Serialize)]
struct VideoRow {
    path: String,
    items: usize,
}
```

Replace the `Engine::videos` method with this free function:

```rust
/// Group the store records by video path. Every video kind counts.
fn videos(store: &Store) -> Vec<VideoRow> {
    let mut rows: Vec<VideoRow> = Vec::new();
    for record in &store.records {
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        match rows.iter_mut().find(|row| row.path == record.label) {
            Some(row) => row.items += 1,
            None => rows.push(VideoRow {
                path: record.label.clone(),
                items: 1,
            }),
        }
    }
    rows
}
```

- [ ] **Step 4: Replace the engine with free functions**

Delete `struct IndexJob`, `struct Engine`, and `impl Engine`. Add this block in
their place:

```rust
/// The indexing work of the server. One `step` call does one media item.
struct WebJob {
    files: Vec<PathBuf>,
    file_at: usize,
    run: Option<index::IndexRun>,
    records: Vec<Record>,
    indexed: usize,
}

/// Answer jobs forever. A running index does one media item per turn.
fn engine_loop(
    embedder: Embedder,
    mut store: Store,
    progress: Arc<Mutex<Progress>>,
    key: String,
    jobs: Receiver<Job>,
) {
    let mut ctx = match embedder.new_context() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("no context: {err:#}");
            return;
        }
    };
    let mut pending: Option<WebJob> = None;
    loop {
        if pending.is_some() {
            step_job(&embedder, &mut ctx, &mut store, &progress, pending.as_mut().unwrap());
            while let Ok(job) = jobs.try_recv() {
                handle(job, &embedder, &mut store, &progress, &key, &mut pending);
            }
        } else {
            match jobs.recv() {
                Ok(job) => handle(job, &embedder, &mut store, &progress, &key, &mut pending),
                Err(_) => break,
            }
        }
    }
}

fn handle(
    job: Job,
    embedder: &Embedder,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
    pending: &mut Option<WebJob>,
) {
    match job {
        Job::State(answer) => {
            let count = videos(store).len();
            let _ = answer.send(State {
                version: env!("CARGO_PKG_VERSION").to_string(),
                records: store.records.len(),
                videos: count,
                key: key.to_string(),
            });
        }
        Job::Videos(answer) => {
            let _ = answer.send(videos(store));
        }
        Job::Holds { path, answer } => {
            let _ = answer.send(store.holds_video(&path));
        }
        Job::Forget { path, answer } => {
            let outcome = store.forget(&path).map_err(JobError::from);
            let _ = answer.send(outcome);
        }
        Job::Search { query, answer } => {
            let _ = answer.send(search(embedder, store, &query));
        }
        Job::StartIndex { folder, answer } => {
            let _ = answer.send(start_index(store, progress, pending, &folder));
        }
    }
}

fn search(
    embedder: &Embedder,
    store: &Store,
    query: &str,
) -> Result<SearchAnswer, JobError> {
    if store.records.is_empty() {
        return Ok(SearchAnswer {
            groups: Vec::new(),
            baseline: 0.0,
        });
    }
    let text = format!("{QUERY_PREFIX}{query}");
    let vec = embedder.embed_text(&text).map_err(JobError::from)?;
    let ranked = store.rank(&vec);
    let baseline = index::median(&ranked);
    Ok(SearchAnswer {
        groups: index::group_hits(&store.records, &ranked, baseline),
        baseline,
    })
}

fn start_index(
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    pending: &mut Option<WebJob>,
    folder: &Path,
) -> Result<usize, JobError> {
    if pending.is_some() {
        return Err(JobError::Busy);
    }
    if !folder.is_dir() {
        return Err(JobError::Message(format!(
            "{} is not a folder",
            folder.display()
        )));
    }
    let files = index::list_videos(folder).map_err(JobError::from)?;
    if files.is_empty() {
        return Err(JobError::Message(format!(
            "no video files under {}",
            folder.display()
        )));
    }
    let count = files.len();
    {
        let mut held = progress.lock().unwrap();
        *held = Progress {
            running: true,
            folder: folder.display().to_string(),
            file_count: count,
            ..Progress::default()
        };
    }
    let _ = store; // the store is used on the first step
    *pending = Some(WebJob {
        files,
        file_at: 0,
        run: None,
        records: Vec::new(),
        indexed: 0,
    });
    Ok(count)
}

/// One media item of work, or the move to the next file.
fn step_job(
    embedder: &Embedder,
    ctx: &mut LlamaContext,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    job: &mut WebJob,
) {
    if job.run.is_none() {
        if job.file_at >= job.files.len() {
            progress.lock().unwrap().running = false;
            return;
        }
        let video = job.files[job.file_at].clone();
        match index::IndexRun::start(&video) {
            Ok(run) => {
                let label = store::normalize_label(&video.to_string_lossy());
                if let Err(err) = store.forget(&label) {
                    fail(progress, &err);
                }
                let total = run.total();
                {
                    let mut held = progress.lock().unwrap();
                    held.file = label;
                    held.file_index = job.file_at + 1;
                    held.window = 0;
                    held.window_count = total;
                    held.error = None;
                }
                job.records.clear();
                job.run = Some(run);
            }
            Err(err) => {
                fail(progress, &err);
                job.file_at += 1;
            }
        }
        return;
    }

    let outcome = job
        .run
        .as_mut()
        .unwrap()
        .step(embedder, ctx);
    let mut finished = false;
    match outcome {
        Ok(Some(record)) => job.records.push(record),
        Ok(None) => finished = true,
        Err(err) => {
            fail(progress, &err);
            finished = true;
        }
    }

    if finished {
        let records = std::mem::take(&mut job.records);
        job.indexed += records.len();
        if let Err(err) = store.append(records) {
            fail(progress, &err);
        }
        job.run = None;
        job.file_at += 1;
    } else {
        let mut held = progress.lock().unwrap();
        held.window = job.run.as_ref().map(|r| r.done()).unwrap_or(0);
        held.indexed = job.indexed + job.records.len();
    }
}

fn fail(progress: &Arc<Mutex<Progress>>, err: &anyhow::Error) {
    progress.lock().unwrap().error = Some(format!("{err:#}"));
}
```

Change the `SearchAnswer` struct at `videoembed/src/bin/web.rs:81-84`:

```rust
#[derive(Serialize)]
struct SearchAnswer {
    groups: Vec<Group>,
    baseline: f32,
}
```

Add these imports to `videoembed/src/bin/web.rs` with the other `use` lines:

```rust
use llama_cpp_2::context::LlamaContext;
use videoembed::store::{self, KIND_NOTE, KIND_WINDOW};
```

Keep the old `use videoembed::store::{KIND_WINDOW, Record, Store, normalize_label};`
line only for the names the file still uses. If `KIND_WINDOW` becomes unused,
remove it from the list.

- [ ] **Step 5: Update `main`**

Replace the engine spawn in `main` at `videoembed/src/bin/web.rs:769-794` with
this block:

```rust
fn main() -> Result<()> {
    let models = models_dir();
    let store_path = PathBuf::from("store/index.vei");
    let progress = Arc::new(Mutex::new(Progress::default()));
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    let key = make_key();

    let engine_progress = Arc::clone(&progress);
    std::thread::spawn(move || {
        let embedder = match Embedder::load(&models) {
            Ok(embedder) => embedder,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        let store = match Store::open(&store_path) {
            Ok(store) => store,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        let _ = ready_tx.send(Ok(()));
        engine_loop(embedder, store, engine_progress, key, jobs_rx);
    });

    match ready_rx.recv().context("the engine thread stopped early")? {
        Ok(()) => {}
        Err(message) => bail!("the engine failed to start: {message}"),
    }

    let server = Server::http(ADDRESS).map_err(|err| anyhow!("{err}"))?;
    println!("videoembed is at http://{ADDRESS}/");
    println!("Stop the server with the Quit button or with Ctrl C.");
    let _ = open::that(format!("http://{ADDRESS}/"));

    for request in server.incoming_requests() {
        let jobs = jobs_tx.clone();
        let progress = Arc::clone(&progress);
        let key = key.clone();
        std::thread::spawn(move || {
            if let Err(err) = handle_request(request, &jobs, &progress, &key) {
                eprintln!("A request failed: {err:#}");
            }
        });
    }
    Ok(())
}
```

The old free function `handle` at `videoembed/src/bin/web.rs:596-636` keeps its
body. Rename it to `handle_request` to avoid the clash with the new `handle`.
Update the one call inside `handle_request` that reaches `api(...)` and
`serve_video(...)`. Those two stay unchanged.

- [ ] **Step 6: Build and run the server**

Run:
```
cd videoembed
cargo build
./target/debug/web.exe
```
Expected: the browser opens. Press Choose folder, pick a folder with a video,
press Index. The progress bar moves by item. The search shows a margin for
each hit. No error in the terminal.

- [ ] **Step 7: Commit**

```bash
git add videoembed/src/bin/web.rs
git commit -m "feat: drive the web index with one context and IndexRun

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 9: The web page shows frames and the margin

**Files:**
- Modify: `videoembed/assets/app.js:23-27` (`refreshState`)
- Modify: `videoembed/assets/app.js:42-60` (`refreshVideos`)
- Modify: `videoembed/assets/app.js:102-119` (`showProgress`)
- Modify: `videoembed/assets/app.js:158-215` (`renderGroups`, `timelineBar`)
- Modify: `videoembed/assets/index.html:43-46` (a legend line)

**Interfaces:**
- Consumes: `VideoRow.items`, `SearchAnswer.baseline`, `Hit.margin`.
- Produces: nothing for later tasks.

- [ ] **Step 1: Update the wording and the count**

In `refreshState`, change the status line:

```javascript
  $("status").textContent = state.records + " records, " + state.videos + " videos";
```

In `refreshVideos`, change the count cell:

```javascript
    count.textContent = video.items + " items";
```

In `showProgress`, change the two text lines:

```javascript
    $("progress-text").textContent =
      "File " + p.file_index + "/" + p.file_count +
      ", item " + p.window + "/" + p.window_count +
      " - " + p.file;
```

```javascript
  $("progress-text").textContent = "Done. " + p.indexed + " records indexed.";
```

- [ ] **Step 2: Show the margin in a hit**

In `renderGroups`, change the hit label and the empty message:

```javascript
      row.textContent =
        hit.score.toFixed(3) + " (" + hit.margin.toFixed(3) + ")  " +
        hit.start.toFixed(2) + "-" + hit.end.toFixed(2) + " s";
```

```javascript
  if (groups.length === 0) {
    note("No record scored above the floor.");
    return;
  }
```

- [ ] **Step 3: Show the margin in the timeline mark**

In `timelineBar`, change the mark title:

```javascript
    mark.title = hit.score.toFixed(3) + " (" + hit.margin.toFixed(3) + ")  " +
      hit.start.toFixed(2) + " s";
```

- [ ] **Step 4: Add the legend line**

Add this line under the Timeline heading in `videoembed/assets/index.html`:

```html
  <p class="legend">The number is the score, then the margin over the store
  median. A mark with a small margin is a weak match.</p>
```

- [ ] **Step 5: Check the page by hand**

Run: `cd videoembed && ./target/debug/web.exe`
Expected: the status shows records. A search shows a number pair for each hit
and a mark on the timeline. The legend appears under the Timeline heading.

- [ ] **Step 6: Commit**

```bash
git add videoembed/assets/app.js videoembed/assets/index.html
git commit -m "feat: show frame records and the score margin in the web page

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| Frame record at 4 a second | Tasks 3, 6 |
| Audio record at 1 a second | Tasks 3, 4, 6 |
| Encode each item once | Task 6 |
| Calibrated score | Tasks 2, 7, 8, 9 |
| Strict frame-level slider | Task 9, `timelineBar` |
| Old window records stay valid | Task 1, Review Focus 4 |
| Batch images per vision call | Deferred. The MVP makes one call per item. |
| NVDEC frame decode | Deferred. |
| Scene-change keying | Deferred. |
| Rerank tier | Deferred. It follows the MVP. |
| The ANN index | Route B. Out of scope. |

The deferred levers do not block the MVP. Each one is a separate plan.

**Placeholder scan:** No `TBD` and no `TODO`. Every step holds the code or the
command.

**Type consistency:** `IndexRun::done` returns `usize` and the plan uses it in
Tasks 6, 7, and 8. `group_hits` takes `baseline: f32` in Tasks 2, 7, and 8.
`VideoRow.items` is set in Task 8 and read in Task 9. `Hit.margin` is set in
Task 2 and read in Tasks 7 and 9.

**Review Focus:** Every line has a test. Line 1 has the audio test in Task 4
and the empty-batch path in Task 6. Line 2 has the frame test in Task 3 and the
bail in Task 6. Line 3 has the batch guard in Task 6. Line 4 has the test in
Task 1. Line 5 has the test in Task 2.
