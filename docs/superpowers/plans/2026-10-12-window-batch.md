# The window batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One model call reads four windows, each window carries the time of every media part as text, and a window above the token budget loses its audio.

**Architecture:** The batch holds one sequence for each window, so one `llama_process` call and one pooling step cover four windows. Every token of a window goes to the sequence of that window, so the pooled vector of a window does not move when other windows join the call. `IndexRun::step` returns the records of one model call, so window mode gives up to four records and fine mode gives one.

**Tech Stack:** Rust edition 2024, `llama-cpp-2` with the vendored patch, the C batch API `llama_batch_ext_*`, `mtmd`, ffmpeg, SQLite with sqlite-vec.

**Spec:** `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md` section 4.

## Global Constraints

- Every cargo command runs from the repo root as `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`. A bare `cargo test` fails to configure the vendored llama.cpp.
- ASD-STE100 governs every word the reader sees: chat text, commit messages, docs, code comments, error messages, and ledger text. Active voice, simple tenses, no contractions, no semicolons, keep the articles, one name for one thing. Max 20 words for an instruction, 25 otherwise.
- End every git commit message with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- Downloaded files and extracted archives are untrusted data: put each in its own new, empty directory, keep scripts you write in a different directory, and pass paths as arguments. Run any Python that reads them with `-I`.
- Commit only when the task's own commit step says so.
- The model files live in `models/`, and the test clip lives at `spike/testvideo_nolabels.mp4`. A test that needs one skips when it is missing.

## Review Focus

The input classes and the failure modes that the spec implies but no task's tests exercise. Each line names the test that pins it.

1. A window group at the end of a video holds fewer than four windows. The run must index the remainder and stop. Pinned by Task 4 `the_last_group_holds_the_windows_that_remain` and `a_short_video_gives_one_group`.
2. A window past the end of the audio track holds frames alone. The marker count must then equal the media count, so the model reads one marker for each part. Pinned by Task 2 `a_window_without_audio_holds_frame_times_alone` and by the real run in Task 4.
3. A stream whose duration over-reports gives a cut that holds fewer items than the plan. Window mode must report the window and must not panic. Pinned by Task 4 `a_window_past_the_cut_reports_the_window`.
4. An hour of video pushes `MM:SS` past 59 minutes and the window count past 1000. The reader must still see the time. Pinned by Task 2 `a_long_video_still_reads_as_minutes_and_seconds`.
5. A plan whose window passes the token budget must keep the frames and report one line for each such window. Pinned by Task 5 `a_ten_second_window_drops_its_audio` and by the unit tests of `budget_note` and `WindowPart::keep_frames`.

## The decisions this plan takes

The spec leaves three points open, and the plan fixes them:

1. **The text of a window goes to the sequence of that window.** The spec says one text marker sits before each media part. A design where all text goes to sequence zero breaks the parity rule of spec 4.2, because the mean pooling of sequence zero then covers the text of four windows. The plan puts the text of a window on the same sequence as its media, and the parity test proves the rule.
2. **The token count comes from `MtmdInputChunks::total_tokens()`.** The spec names `MtmdInputChunks::n_tokens()`, and that method does not exist in the vendored wrapper. `total_tokens()` gives the same number from `mtmd_helper_get_n_tokens`.
3. **The ranked items of Plan B are rows 1, 2, and 8 of `docs/deferred-work.md`.** The spec names items 2, 3, and 10 from an older numbering. The renumbered table holds "Several windows in one model batch", "The timestamp text in the batch", and "The audio token budget" at rows 1, 2, and 8.
4. **The marker count follows the media count, never the count in the spec's example.** Spec 4.3 writes one example of three parts: `00:03 <__media__> 00:04 <__media__> 00:03 <__media__>`. That example names two frames and one clip. The real `clips_in` gives every clip that touches a 2 s window, and that is two clips, so a 2 s Object window holds four parts. The rule of 4.3, "the marker count always equals the bitmap count", is the binding one. Task 2 pins the spec's exact three-part string in a unit test of `window_text`, and Task 1 measures the real part count.

The plan names one more limit, as the spec asks: a window whose frames alone pass `BUDGET_TOKENS` still goes into the batch, because the batch holds 2048 tokens. The run does not refuse such a plan. Task 6 writes the limit into `docs/deferred-work.md`.

---

### Task 1: The token count of a window

**Files:**
- Modify: `videoembed/src/embed.rs` (add `tokens_of`, `tokenize`, `N_WINDOW_BATCH`, `BUDGET_TOKENS`)
- Create: `videoembed/tests/model_batch.rs`
- Test: `videoembed/tests/model_batch.rs`

**Interfaces:**
- Consumes: `Embedder::load(models: &Path, asked: Option<Device>) -> Result<Embedder>`, `Embedder::new_context()`, `media::cut_frames(video: &Path, rate: f64, dir: &Path) -> Result<Vec<PathBuf>>`, `media::cut_audio(video: &Path, dir: &Path) -> Result<Vec<PathBuf>>`, `index::frames_in(cut: &[PathBuf], from: f64, window: f64, rate: f64) -> Vec<PathBuf>`, `index::clips_in(cut: &[PathBuf], from: f64, window: f64) -> Vec<PathBuf>`, `plan::Plan::of(Preset)`, `plan::Plan::frame_rate()`.
- Produces:
  - `embed::N_WINDOW_BATCH: usize`
  - `embed::BUDGET_TOKENS: usize`
  - `Embedder::tokens_of(&self, text: &str, media: &[PathBuf]) -> Result<usize>`
  - the file `videoembed/tests/model_batch.rs` with the helpers `models()`, `video()`, `scratch(name)`, `one_second_cut(video, dir)`, and `the_model() -> Option<(&'static Embedder, MutexGuard<'static, ()>)>`.

- [ ] **Step 1: Write the test file with its helpers**

Create `videoembed/tests/model_batch.rs`:

```rust
//! The tests that need the real model.
//!
//! The lib tests never load the model, because `LlamaBackend::init` runs one
//! time in a process. This file is its own test binary and its own process, so
//! one backend serves every test here.
//!
//! Run from the repo root:
//!   CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --test model_batch

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use videoembed::embed::{self, Embedder};
use videoembed::{index, media, plan};

/// One media call at a time. The media encoder of llama.cpp is not thread safe,
/// so the tests of this file share one model and run one after the other.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The model folder. None when the two files are not here.
fn models() -> Option<PathBuf> {
    let models = PathBuf::from(std::env::var("VEI_MODELS").unwrap_or("../models".into()));
    let text = models.join("embeddinggemma-2-Q8_0.gguf");
    let mmproj = models.join("mmproj-embeddinggemma-2-Q8_0.gguf");
    (text.is_file() && mmproj.is_file()).then_some(models)
}

/// The one model of this test binary.
///
/// `LlamaBackend::init` runs one time in a process, so every test here shares
/// one embedder. A model that is not here gives None, and the test skips.
fn the_model() -> Option<(&'static Embedder, MutexGuard<'static, ()>)> {
    static SHARED: OnceLock<Option<Embedder>> = OnceLock::new();
    let guard = ONE_AT_A_TIME.lock().unwrap_or_else(|err| err.into_inner());
    let held = SHARED.get_or_init(|| models().and_then(|path| Embedder::load(&path, None).ok()));
    held.as_ref().map(|embedder| (embedder, guard))
}

/// The test clip. None when the file is not here.
fn video() -> Option<PathBuf> {
    let path = PathBuf::from("../spike/testvideo_nolabels.mp4");
    path.is_file().then_some(path)
}

/// A fresh folder for the cut of one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vei_batch_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the work folder is made");
    dir
}

/// One second of frames, and the one second clips, of the test video.
fn one_second_cut(video: &Path, dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let frames = media::cut_frames(video, 1.0, &dir.join("frames")).expect("the frames are cut");
    let clips = media::cut_audio(video, &dir.join("audio")).expect("the clips are cut");
    (frames, clips)
}
```

- [ ] **Step 2: Write the failing test**

Add to `videoembed/tests/model_batch.rs`:

```rust
/// The media of one Object window of the cut, frames first and clips last.
fn window_media(frames: &[PathBuf], clips: &[PathBuf], from: f64) -> Vec<PathBuf> {
    let plan = plan::Plan::of(plan::Preset::Object);
    let mut media = index::frames_in(frames, from, plan.window, plan.frame_rate());
    media.extend(index::clips_in(clips, from, plan.window));
    media
}

/// The token count of the Object window, and the room of a full batch.
///
/// The numbers pin `N_UBATCH` and `BUDGET_TOKENS`. The count comes from the
/// tokenizer of the real model, because a guess is worthless here.
#[test]
fn the_object_window_sits_under_the_batch_limit() {
    let Some((embedder, _guard)) = the_model() else {
        eprintln!("skipped: the model is not here");
        return;
    };
    let Some(video) = video() else {
        eprintln!("skipped: the test video is not here");
        return;
    };
    let dir = scratch("tokens");
    let (frames, clips) = one_second_cut(&video, &dir);
    let plan = plan::Plan::of(plan::Preset::Object);

    let media = window_media(&frames, &clips, 0.0);
    let times = index::window_times(0.0, plan.frame_rate(), 2, 2);
    let count = embedder
        .tokens_of(&embed::window_text(&times), &media)
        .expect("the count comes back");
    println!("one Object window: {count} tokens over {} parts", media.len());

    let ctx = embedder.new_context().expect("the context is made");
    assert_eq!(media.len(), times.len(), "one marker for each part");
    assert!(count > 0, "the tokenizer gave no token");
    assert!(
        count < embed::BUDGET_TOKENS,
        "one Object window holds {count} tokens, above the budget"
    );
    assert!(
        count * embed::N_WINDOW_BATCH < ctx.n_ubatch() as usize,
        "four windows of {count} tokens pass the batch limit of {}",
        ctx.n_ubatch()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
```

`index::window_times` arrives in Task 2. Until then the test does not compile, and that is the failure of Step 3.

- [ ] **Step 3: Run the test and watch it fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --test model_batch`

Expected: FAIL to compile, with `no method named \`tokens_of\` found for struct \`Embedder\`` and `cannot find function \`window_times\``.

- [ ] **Step 4: Add the constants, `tokenize`, and `tokens_of` to `embed.rs`**

Add above `pub struct Embedder` in `videoembed/src/embed.rs`:

```rust
/// The windows in one model call.
pub const N_WINDOW_BATCH: usize = 4;

/// The token count that one window may hold before it loses its audio.
///
/// The count of one window comes from the tokenizer, never from a guess. Task
/// 1 of the window-batch plan measures the Object window and pins the number.
pub const BUDGET_TOKENS: usize = 1024;
```

Add inside `impl Embedder`, above `embed_text`:

```rust
    /// Tokenize the text of one window with its media.
    ///
    /// The text carries one marker for each bitmap, in the same order.
    fn tokenize(&self, text: &str, media: &[&MtmdBitmap]) -> Result<MtmdInputChunks> {
        let input = MtmdInputText {
            text: text.to_string(),
            add_special: true,
            parse_special: true,
        };
        Ok(self.mtmd.tokenize(input, media)?)
    }

    /// The number of tokens that one window costs, as the tokenizer counts it.
    ///
    /// The count comes from the model, so a budget check uses the real number.
    pub fn tokens_of(&self, text: &str, media: &[PathBuf]) -> Result<usize> {
        if media.is_empty() {
            bail!("a window holds no media");
        }
        let mut bitmaps = Vec::with_capacity(media.len());
        for path in media {
            let path = path.to_str().context("media path is not utf8")?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self.tokenize(text, &refs)?;
        Ok(chunks.total_tokens())
    }
```

Add `PathBuf` to the `use std::path::Path;` line of `videoembed/src/embed.rs`.

- [ ] **Step 5: Run the test and watch it pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --test model_batch`

Expected: PASS, and the line `one Object window: <N> tokens over 4 parts`, where `<N>` sits between 100 and 1024.

- [ ] **Step 6: Pin the measured number**

Read the printed `<N>` from the last step, and replace the sentence `Task 1 of the window-batch plan measures the Object window and pins the number.` in the `BUDGET_TOKENS` doc comment with the measurement, for example `The Object window of the test video measures 304 tokens over four parts.` Write the measured number in the ledger too.

- [ ] **Step 7: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`

Expected: the lib tests, the console tests, and the web tests stay green, and the new target `model_batch` passes its one test.

- [ ] **Step 8: Commit**

```bash
git add videoembed/src/embed.rs videoembed/tests/model_batch.rs
git commit -m "feat: read the token count of one window

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: The time of every part, as text

**Files:**
- Modify: `videoembed/src/embed.rs` (add `mmss` and `window_text`)
- Modify: `videoembed/src/index.rs` (add `first_frame`, `last_frame`, `window_times`, and use them in `frames_in`)
- Test: the `tests` module of `videoembed/src/embed.rs` and of `videoembed/src/index.rs`

**Interfaces:**
- Consumes: `mtmd_default_marker() -> &str` from `llama-cpp-2`.
- Produces:
  - `embed::mmss(at: f64) -> String`
  - `embed::window_text(times: &[f64]) -> String`
  - `index::window_times(from: f64, rate: f64, frames: usize, clips: usize) -> Vec<f64>`

- [ ] **Step 1: Write the failing tests in `embed.rs`**

Add to the `tests` module of `videoembed/src/embed.rs`:

```rust
    /// The marker text of one window names each part with `MM:SS`.
    #[test]
    fn the_window_text_names_every_part() {
        // The example of the spec: a 2 s window at 3 s, two frames and a clip.
        assert_eq!(
            window_text(&[3.0, 4.0, 3.0]),
            "00:03 <__media__> 00:04 <__media__> 00:03 <__media__>"
        );
    }

    /// One marker for one media part, and no marker for no part.
    #[test]
    fn the_marker_count_matches_the_part_count() {
        assert_eq!(window_text(&[7.0]).matches(mtmd_default_marker()).count(), 1);
        assert_eq!(
            window_text(&[7.0, 7.0, 7.0])
                .matches(mtmd_default_marker())
                .count(),
            3
        );
        assert_eq!(window_text(&[]), "");
    }

    /// The seconds round down, so a part reads the second that holds it.
    #[test]
    fn a_part_reads_the_second_that_holds_it() {
        assert_eq!(mmss(0.0), "00:00");
        assert_eq!(mmss(3.0), "00:03");
        assert_eq!(mmss(3.75), "00:03");
        assert_eq!(mmss(59.999), "00:59");
        assert_eq!(mmss(-1.0), "00:00");
    }

    /// An hour of video keeps a readable time.
    #[test]
    fn a_long_video_still_reads_as_minutes_and_seconds() {
        assert_eq!(mmss(60.0), "01:00");
        assert_eq!(mmss(600.0), "10:00");
        assert_eq!(mmss(3600.0), "60:00");
        assert_eq!(mmss(3661.0), "61:01");
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib embed::tests`

Expected: FAIL to compile with `cannot find function \`window_text\` in this scope`.

- [ ] **Step 3: Write `mmss` and `window_text`**

Add beside `unit` and `dot` in `videoembed/src/embed.rs`:

```rust
/// The time of one frame or clip as `MM:SS`.
///
/// The seconds round down, so a part at 3.75 s reads `00:03`, the second that
/// holds that part. The minutes pass 59 for a long video, because a video here
/// can hold more than one hour of media.
pub fn mmss(at: f64) -> String {
    let seconds = at.max(0.0).floor() as u64;
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

/// The text of one window: one `MM:SS` marker before each media part.
///
/// The marker names the start time of its part, so the model reads the order
/// of the events and not only the pictures. The count of the markers always
/// matches the count of the media parts.
pub fn window_text(times: &[f64]) -> String {
    times
        .iter()
        .map(|at| format!("{} {}", mmss(*at), mtmd_default_marker()))
        .collect::<Vec<String>>()
        .join(" ")
}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib embed::tests`

Expected: PASS, 4 new tests, and the earlier embed tests stay green.

- [ ] **Step 5: Write the failing tests in `index.rs`**

Add to the `tests` module of `videoembed/src/index.rs`:

```rust
    /// The parts of one window carry the time of each part.
    ///
    /// The frames come first, then the clips, in the order the embedder reads
    /// them. A frame sits at its own place in the cut, and a clip sits at the
    /// second that it covers.
    #[test]
    fn the_part_times_of_a_window_name_each_part() {
        // The example of the spec: a 2 s window at 3 s, one frame a second.
        assert_eq!(window_times(3.0, 1.0, 2, 1), vec![3.0, 4.0, 3.0]);
        // Four frames a second sit at the quarter steps of the cut.
        assert_eq!(
            window_times(3.0, 4.0, 4, 2),
            vec![3.0, 3.25, 3.5, 3.75, 3.0, 4.0]
        );
    }

    /// A window past the end of the audio track holds frames alone.
    ///
    /// The marker count then equals the frame count, so the model reads one
    /// marker for each part it gets.
    #[test]
    fn a_window_without_audio_holds_frame_times_alone() {
        assert_eq!(window_times(23.0, 1.0, 2, 0), vec![23.0, 24.0]);
        assert_eq!(window_times(0.0, 1.0, 0, 0), Vec::<f64>::new());
    }

    /// The times of a window follow the files that the cut gives back.
    #[test]
    fn the_part_times_match_the_files_of_the_window() {
        let cut: Vec<PathBuf> = (0..8).map(|i| PathBuf::from(format!("f{i}.jpg"))).collect();
        let files = frames_in(&cut, 1.0, 1.0, 4.0);
        let times = window_times(1.0, 4.0, 4, 0);
        assert_eq!(files.len(), times.len());
        assert_eq!(times, vec![1.0, 1.25, 1.5, 1.75]);
    }
```

- [ ] **Step 6: Run the tests and watch them fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::the_part_times`

Expected: FAIL to compile with `cannot find function \`window_times\` in this scope`.

- [ ] **Step 7: Write `first_frame`, `last_frame`, and `window_times`**

Add above `frames_in` in `videoembed/src/index.rs`, and replace the body of `frames_in` with calls to the two new helpers:

```rust
/// The index of the first frame of one span, in the whole-video cut.
///
/// The cut gives frame `n` at `n / rate`, so this index names the time of the
/// first frame of the span.
fn first_frame(from: f64, rate: f64) -> usize {
    (from * rate).round().max(0.0) as usize
}

/// The index after the last frame of one span, in the whole-video cut.
fn last_frame(from: f64, window: f64, rate: f64) -> usize {
    ((from + window) * rate).round().max(0.0) as usize
}

/// The time of every media part of one window.
///
/// The frames come first, then the clips, in the order the embedder reads them.
/// A frame sits at its own place in the cut, so these times match the files
/// that `frames_in` gives. A clip sits at the second that it covers.
pub fn window_times(from: f64, rate: f64, frames: usize, clips: usize) -> Vec<f64> {
    let first = first_frame(from, rate);
    let mut times: Vec<f64> = (0..frames).map(|i| (first + i) as f64 / rate).collect();
    times.extend((0..clips).map(|k| from.floor() + k as f64));
    times
}

pub fn frames_in(cut: &[PathBuf], from: f64, window: f64, rate: f64) -> Vec<PathBuf> {
    let first = first_frame(from, rate);
    let end = last_frame(from, window, rate).min(cut.len());
    if first >= end {
        return Vec::new();
    }
    cut[first..end].to_vec()
}
```

- [ ] **Step 8: Run the tests and watch them pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests`

Expected: PASS, 3 new tests, and the earlier index tests stay green.

- [ ] **Step 9: Commit**

```bash
git add videoembed/src/embed.rs videoembed/src/index.rs
git commit -m "feat: name the time of each media part as text

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Four windows in one model call

**Files:**
- Modify: `videoembed/src/embed.rs` (the batch path)
- Test: `videoembed/tests/model_batch.rs`

**Interfaces:**
- Consumes: `embed::window_text`, `embed::N_WINDOW_BATCH`, `Embedder::tokenize`.
- Produces:
  - `embed::WindowIn { text: String, media: Vec<PathBuf> }`
  - `Embedder::embed_windows_in(&self, ctx: &mut LlamaContext<'_>, windows: &[WindowIn]) -> Result<Vec<Vec<f32>>>`
  - `Embedder::embed_text`, `embed_image_in`, `embed_audio_in`, and `embed_window_in` keep their names and their answers.

- [ ] **Step 1: Write the failing parity test**

Add to `videoembed/tests/model_batch.rs`:

```rust
/// Four windows of one plan, from the start of the video.
fn object_windows(frames: &[PathBuf], clips: &[PathBuf], plan: &plan::Plan) -> Vec<embed::WindowIn> {
    let rate = plan.frame_rate();
    (0..embed::N_WINDOW_BATCH)
        .map(|at| {
            let from = at as f64 * plan.stride;
            let parts = index::frames_in(frames, from, plan.window, rate);
            let window_clips = index::clips_in(clips, from, plan.window);
            let times = index::window_times(from, rate, parts.len(), window_clips.len());
            let mut media = parts;
            media.extend(window_clips);
            embed::WindowIn {
                text: embed::window_text(&times),
                media,
            }
        })
        .collect()
}

/// Four windows in one call give the same vectors as one at a time.
///
/// This is the gate of the window batch. A vector that moves with the batch
/// size is worthless, because the search then reads one vector for a window
/// and the record holds another.
#[test]
fn one_batch_of_four_matches_four_calls_of_one() {
    let Some((embedder, _guard)) = the_model() else {
        eprintln!("skipped: the model is not here");
        return;
    };
    let Some(video) = video() else {
        eprintln!("skipped: the test video is not here");
        return;
    };
    let dir = scratch("parity");
    let (frames, clips) = one_second_cut(&video, &dir);
    let plan = plan::Plan::of(plan::Preset::Object);
    let windows = object_windows(&frames, &clips, &plan);
    assert_eq!(windows.len(), embed::N_WINDOW_BATCH);

    let mut ctx = embedder.new_context().expect("the context is made");
    let together = embedder
        .embed_windows_in(&mut ctx, &windows)
        .expect("the batch comes back");
    assert_eq!(together.len(), embed::N_WINDOW_BATCH);

    for (at, window) in windows.iter().enumerate() {
        let alone = embedder
            .embed_windows_in(&mut ctx, std::slice::from_ref(window))
            .expect("one window comes back");
        let cos = embed::dot(&together[at], &alone[0]);
        println!("window {at}: cosine {cos}");
        assert!(cos > 0.999, "window {at} moved with the batch: cosine {cos}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --test model_batch one_batch_of_four`

Expected: FAIL to compile with `cannot find struct \`WindowIn\``.

- [ ] **Step 3: Add `WindowIn`, `chunk_owners`, and `embed_windows_in`**

Add above `pub struct Embedder` in `videoembed/src/embed.rs`:

```rust
/// One window of a model call: its text and its media files.
///
/// The text carries one marker for each media file, in the same order.
#[derive(Clone, Debug, Default)]
pub struct WindowIn {
    /// The marker text of the window.
    pub text: String,
    /// The media files in the order the model reads them: the frames first.
    pub media: Vec<PathBuf>,
}
```

Add inside `impl Embedder`, above `embed_image_in`:

```rust
    /// Embed several windows in one model call. One vector for each window.
    ///
    /// Every window takes its own sequence, so the pooled vector of a window
    /// holds the tokens of that window alone. The text and the media of a
    /// window share its sequence for the same reason. The answer follows the
    /// order of the call.
    pub fn embed_windows_in(
        &self,
        ctx: &mut LlamaContext<'_>,
        windows: &[WindowIn],
    ) -> Result<Vec<Vec<f32>>> {
        if windows.is_empty() {
            bail!("a batch holds no window");
        }
        if windows.len() > N_WINDOW_BATCH {
            bail!(
                "the batch holds {} windows, above the limit of {N_WINDOW_BATCH}",
                windows.len()
            );
        }
        ctx.clear_kv_cache();

        let mut bitmaps = Vec::new();
        let mut text = String::new();
        let mut parts = Vec::with_capacity(windows.len());
        for (at, window) in windows.iter().enumerate() {
            if window.media.is_empty() {
                bail!("window {at} holds no media");
            }
            for path in &window.media {
                let path = path.to_str().context("media path is not utf8")?;
                bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
            }
            if at > 0 {
                // The marker of the next window needs its own boundary.
                text.push(' ');
            }
            text.push_str(&window.text);
            parts.push(window.media.len());
        }

        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self.tokenize(&text, &refs)?;
        let owners = chunk_owners(&chunks, &parts)?;

        let n_embd_inp = unsafe { sys::llama_model_n_embd_inp(self.model.as_ptr()) };
        if n_embd_inp <= 0 {
            bail!("the model reports no input embedding size");
        }
        let n_embd_inp = n_embd_inp as usize;

        let batch = unsafe { sys::llama_batch_ext_init(ctx.as_ptr()) };
        if batch.is_null() {
            bail!("llama_batch_ext_init returned null");
        }
        let filled = self.fill_windows(ctx, &chunks, &owners, windows.len(), batch, n_embd_inp);
        let raw = match filled {
            Ok(()) => (0..windows.len())
                .map(|seq| unsafe { pooled_seq(ctx, seq as sys::llama_seq_id) })
                .collect::<Result<Vec<Vec<f32>>>>(),
            Err(err) => Err(err),
        };
        unsafe { sys::llama_batch_ext_free(batch) };

        let mut out = Vec::with_capacity(windows.len());
        for one in raw? {
            out.push(unit(&one)?);
        }
        Ok(out)
    }
```

Add the free function `chunk_owners` below `impl Embedder`:

```rust
/// The window of every chunk of one batch.
///
/// The media chunks come in window order. A text chunk belongs to the window
/// of the next media chunk, because that text names the part that follows it.
/// A text chunk after the last media chunk belongs to the window of the last
/// media chunk.
fn chunk_owners(chunks: &MtmdInputChunks, parts: &[usize]) -> Result<Vec<usize>> {
    let mut media_owner = Vec::new();
    for (window, count) in parts.iter().enumerate() {
        for _ in 0..*count {
            media_owner.push(window);
        }
    }
    let last = *media_owner
        .last()
        .ok_or_else(|| anyhow::anyhow!("a batch holds no media part"))?;

    let n_chunks = unsafe { sys::mtmd_input_chunks_size(chunks.as_ptr()) };
    let mut owners = Vec::with_capacity(n_chunks);
    let mut seen = 0usize;
    for i in 0..n_chunks {
        let chunk = chunks
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("chunk {i} is missing"))?;
        if chunk.chunk_type() == MtmdInputChunkType::Text {
            owners.push(media_owner.get(seen).copied().unwrap_or(last));
        } else {
            let owner = media_owner.get(seen).copied().ok_or_else(|| {
                anyhow::anyhow!("the tokenizer gave more media chunks than the batch holds parts")
            })?;
            owners.push(owner);
            seen += 1;
        }
    }
    Ok(owners)
}
```

Add `MtmdInputChunkType` to the `use llama_cpp_2::mtmd::{...}` list, and `anyhow::anyhow` to the `use anyhow::{...}` list.

- [ ] **Step 4: Replace the batch path with the window path**

Replace `embed_joint`, `fill_batch`, and `push` in `videoembed/src/embed.rs` with the code below, and make the single-window calls into thin wrappers:

```rust
    /// One batch holds every token and every media row of every window.
    ///
    /// The entry of a window names the sequence of that window, so one decode
    /// covers the whole batch and one pooling step gives one vector for each
    /// window.
    fn fill_windows(
        &self,
        ctx: &LlamaContext<'_>,
        chunks: &MtmdInputChunks,
        owners: &[usize],
        n_windows: usize,
        batch: *mut sys::llama_batch_ext,
        n_embd_inp: usize,
    ) -> Result<()> {
        let n_chunks = unsafe { sys::mtmd_input_chunks_size(chunks.as_ptr()) };
        let mut pos = vec![0 as sys::llama_pos; n_windows];

        for i in 0..n_chunks {
            let owner = *owners
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("chunk {i} has no window"))?;
            let chunk = unsafe { sys::mtmd_input_chunks_get(chunks.as_ptr(), i) };
            if chunk.is_null() {
                bail!("chunk {i} is null");
            }
            let kind = unsafe { sys::mtmd_input_chunk_get_type(chunk) };
            let seq = owner as sys::llama_seq_id;

            if kind == sys::MTMD_INPUT_CHUNK_TYPE_TEXT {
                let mut count = 0usize;
                let tokens = unsafe { sys::mtmd_input_chunk_get_tokens_text(chunk, &mut count) };
                if tokens.is_null() {
                    bail!("text chunk {i} has no tokens");
                }
                for t in 0..count {
                    let id = unsafe { *tokens.add(t) };
                    self.push(batch, seq, TokenOrRows::Token(id), n_embd_inp, &mut pos[owner])?;
                }
            } else {
                let rc = unsafe { sys::mtmd_encode_chunk(self.mtmd.as_ptr(), chunk) };
                if rc != 0 {
                    bail!("media encode failed for chunk {i} with code {rc}");
                }
                let n_tokens = unsafe { sys::mtmd_input_chunk_get_n_tokens(chunk) };
                let data = unsafe { sys::mtmd_get_output_embd(self.mtmd.as_ptr()) };
                if data.is_null() {
                    bail!("chunk {i} produced no embedding rows");
                }
                for t in 0..n_tokens {
                    let row = unsafe { data.add(t * n_embd_inp) };
                    self.push(batch, seq, TokenOrRows::Rows(row), n_embd_inp, &mut pos[owner])?;
                }
                // Media chunks can carry more positions than tokens. Text does not.
                let n_pos = unsafe { sys::mtmd_input_chunk_get_n_pos(chunk) };
                if n_pos > n_tokens as sys::llama_pos {
                    pos[owner] = pos[owner] + (n_pos - n_tokens as sys::llama_pos);
                }
            }
        }

        let rc = unsafe { sys::llama_process(ctx.as_ptr(), sys::LLAMA_PROCESS_TYPE_DECODE, batch) };
        if rc != 0 {
            bail!("llama_process failed with code {rc}");
        }
        Ok(())
    }

    fn push(
        &self,
        batch: *mut sys::llama_batch_ext,
        seq: sys::llama_seq_id,
        item: TokenOrRows,
        n_embd_inp: usize,
        pos: &mut sys::llama_pos,
    ) -> Result<()> {
        unsafe {
            let idx = match item {
                TokenOrRows::Token(id) => sys::llama_batch_ext_add_token(batch, seq, id),
                TokenOrRows::Rows(data) => {
                    let embd = sys::llama_embd {
                        data,
                        n_rows: 1,
                        n_embd: n_embd_inp,
                    };
                    sys::llama_batch_ext_add_embd(batch, seq, embd)
                }
            };
            if idx < 0 {
                bail!("the batch rejected an entry with code {idx}");
            }
            sys::llama_batch_ext_set_pos(batch, idx, pos);
            *pos += 1;
        }
        Ok(())
    }
```

The single-media calls become:

```rust
    /// Embed one media file in a caller-owned context.
    pub fn embed_image_in(&self, ctx: &mut LlamaContext<'_>, frame: &Path) -> Result<Vec<f32>> {
        let window = WindowIn {
            text: mtmd_default_marker().to_string(),
            media: vec![frame.to_path_buf()],
        };
        Ok(self.embed_windows_in(ctx, &[window])?.remove(0))
    }

    /// Embed one audio clip in a caller-owned context.
    pub fn embed_audio_in(&self, ctx: &mut LlamaContext<'_>, clip: &Path) -> Result<Vec<f32>> {
        self.embed_image_in(ctx, clip)
    }

    /// Embed several media files as one window, in the given order.
    ///
    /// The caller passes the frames first and the audio clip last. One batch
    /// holds every part, and one decode follows. The probe uses this call to
    /// test window mode.
    pub fn embed_window_in(
        &self,
        ctx: &mut LlamaContext<'_>,
        media: &[&Path],
    ) -> Result<Vec<f32>> {
        if media.is_empty() {
            bail!("a window holds no media");
        }
        let window = WindowIn {
            text: mtmd_default_marker().repeat(media.len()),
            media: media.iter().map(|path| path.to_path_buf()).collect(),
        };
        Ok(self.embed_windows_in(ctx, &[window])?.remove(0))
    }
```

`embed_text` keeps its body, and it reads the pooled vector of sequence 0 through the new helper.

Replace `pooled` with:

```rust
/// Copy the pooled vector of one sequence.
unsafe fn pooled_seq(ctx: &LlamaContext<'_>, seq: sys::llama_seq_id) -> Result<Vec<f32>> {
    let out = unsafe { sys::llama_get_embeddings_seq(ctx.as_ptr(), seq) };
    if out.is_null() {
        bail!("the context returned no pooled embedding for sequence {seq}");
    }
    let n_out = unsafe { sys::llama_model_n_embd_out(ctx.model.as_ptr()) };
    if n_out <= 0 {
        bail!("the model reports no embedding size");
    }
    Ok(unsafe { std::slice::from_raw_parts(out, n_out as usize) }.to_vec())
}
```

Set the sequence count in `context_params`:

```rust
fn context_params() -> LlamaContextParams {
    LlamaContextParams::default()
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean)
        .with_attention_type(LlamaAttentionType::NonCausal)
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_BATCH)
        .with_n_ubatch(N_UBATCH)
        // One sequence for each window of one model call.
        .with_n_seq_max(N_WINDOW_BATCH as u32)
}
```

- [ ] **Step 5: Run the parity test and watch it pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --test model_batch`

Expected: PASS, and every window reports a cosine above 0.999. A failure here stops the plan: the spec says the gain is worthless when the vectors move. Write the cosine numbers in the ledger.

- [ ] **Step 6: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`

Expected: PASS. The probe example must still build: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo build --manifest-path videoembed/Cargo.toml --examples`.

- [ ] **Step 7: Commit**

```bash
git add videoembed/src/embed.rs videoembed/tests/model_batch.rs
git commit -m "feat: read four windows in one model call

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: The run takes four windows at a time

**Files:**
- Modify: `videoembed/src/index.rs` (`IndexRun`, `WindowPart`, `index_video`, `reindex`)
- Modify: `videoembed/src/bin/web.rs` (the step loop)
- Modify: `videoembed/examples/window_probe.rs` (the step loop)
- Modify: `videoembed/src/main.rs` (two call sites)
- Test: the `tests` module of `videoembed/src/index.rs` and `videoembed/tests/model_batch.rs`

**Interfaces:**
- Consumes: `embed::N_WINDOW_BATCH`, `embed::window_text`, `embed::WindowIn`, `embed::embed_windows_in`, `index::window_times`.
- Produces:
  - `index::WindowPart { start: f64, files: Vec<PathBuf>, times: Vec<f64>, frames: usize }`
  - `IndexRun::step(&mut self, embedder: &Embedder, ctx: &mut LlamaContext<'_>) -> Result<Vec<NewRecord>>`
  - `IndexRun::notes(&self) -> &[String]`
  - `index::index_video(...) -> Result<(Vec<NewRecord>, Vec<String>)>`
  - `index::reindex(...) -> Result<(usize, Vec<String>)>`

- [ ] **Step 1: Write the failing group tests**

Add to the `tests` module of `videoembed/src/index.rs`:

```rust
    /// One model call reads four windows.
    #[test]
    fn the_run_takes_four_windows_at_a_time() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("group_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();

        run.cut_batch().unwrap();

        assert_eq!(run.group.len(), N_WINDOW_BATCH);
        assert_eq!(run.group[0].frames, 2, "the Object window holds two frames");
        assert_eq!(run.group[0].files.len(), 4, "two frames and two clips");
        assert_eq!(run.group[0].times.len(), run.group[0].files.len());
        assert_eq!(run.remaining_in_batch(), 1, "one group is one model call");
    }

    /// The last group holds the windows that remain.
    #[test]
    fn the_last_group_holds_the_windows_that_remain() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("tail_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        let total = run.total();
        assert!(total % N_WINDOW_BATCH != 0, "the fixture divides by four here");
        run.window_at = total - 1;

        run.cut_batch().unwrap();

        assert_eq!(run.group.len(), 1, "the tail holds one window");
        assert_eq!(run.group[0].start, run.windows[total - 1]);
    }

    /// A video that holds fewer windows than one group gives one group.
    #[test]
    fn a_short_video_gives_one_group() {
        let Some(video) = spike_video_short() else {
            return;
        };
        let media = scratch("short_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();

        assert!(run.total() < N_WINDOW_BATCH, "the fixture is short here");
        run.cut_batch().unwrap();
        assert_eq!(run.group.len(), run.total());
    }

    /// A window past the cut names itself, and the run does not panic.
    #[test]
    fn a_window_past_the_cut_reports_the_window() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("cut_short_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        run.cut_batch().unwrap();
        // The stream claims more media than the cut holds.
        run.frames_cut.truncate(4);
        run.clips_cut.clear();
        run.window_at = 10;

        let err = run.cut_batch().unwrap_err();

        assert!(
            err.to_string().contains("no media came from the window"),
            "got {err}"
        );
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::the_run_takes_four`

Expected: FAIL to compile with `no field \`group\` on struct \`IndexRun\``.

- [ ] **Step 3: Add `WindowPart` and the run fields**

Add above `pub struct IndexRun` in `videoembed/src/index.rs`:

```rust
/// One window of a group: its media, the time of each part, and its frames.
///
/// The files hold the frames first, then the clips. The times sit in the same
/// order, so the marker of a part names the time of that part.
#[derive(Clone, Debug, Default)]
pub struct WindowPart {
    /// The start of the window in the video.
    pub start: f64,
    /// The media files of the window, the frames first.
    pub files: Vec<PathBuf>,
    /// The time of each part, in the order of the files.
    pub times: Vec<f64>,
    /// The number of frames at the front of the files.
    pub frames: usize,
}

impl WindowPart {
    /// Drop the audio of this window. The frames stay.
    pub fn keep_frames(&mut self) {
        self.files.truncate(self.frames);
        self.times.truncate(self.frames);
    }
}
```

Add two fields to `IndexRun`:

```rust
    /// The windows of the current model call. Empty in fine mode.
    group: Vec<WindowPart>,
    /// The lines that the run reported, in order.
    notes: Vec<String>,
```

Add `group: Vec::new(),` and `notes: Vec::new(),` to the two `Ok(Self {` blocks and to the hand-built `IndexRun` literal in the test `the_run_ends_after_the_frames_when_the_audio_plan_is_empty`.

- [ ] **Step 4: Group the windows and return the records of one call**

Replace `window_files` with:

```rust
    /// The next group of windows, up to one model call of them.
    fn window_group(&self) -> Vec<WindowPart> {
        let rate = self.plan.frame_rate();
        let end = (self.window_at + N_WINDOW_BATCH).min(self.windows.len());
        (self.window_at..end)
            .map(|at| {
                let from = self.windows[at];
                let mut files = frames_in(&self.frames_cut, from, self.plan.window, rate);
                let frames = files.len();
                let clips = clips_in(&self.clips_cut, from, self.plan.window);
                let times = window_times(from, rate, frames, clips.len());
                files.extend(clips);
                WindowPart {
                    start: from,
                    files,
                    times,
                    frames,
                }
            })
            .collect()
    }
```

Replace the window arm of `cut_batch` with:

```rust
        if self.phase == Phase::Windows {
            let group = self.window_group();
            if group.is_empty() {
                // The plan asks for a window that the cut does not hold.
                let at = self.windows.get(self.window_at).copied().unwrap_or_default();
                bail!("no media came from the window at {at} s");
            }
            self.group = group;
            return Ok(());
        }
```

Replace `step` with:

```rust
    /// Embed the next model call. Returns the records of that call.
    ///
    /// Window mode reads up to `N_WINDOW_BATCH` windows in one call, so it
    /// gives up to that many records. Fine mode gives one.
    pub fn step(
        &mut self,
        embedder: &Embedder,
        ctx: &mut LlamaContext<'_>,
    ) -> Result<Vec<NewRecord>> {
        if self.phase == Phase::Done {
            return Ok(Vec::new());
        }
        if self.remaining_in_batch() == 0 {
            self.cut_batch()?;
            if self.phase == Phase::Done {
                return Ok(Vec::new());
            }
        }
        if self.phase == Phase::Windows {
            return self.step_group(embedder, ctx);
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
            Phase::Windows | Phase::Done => unreachable!(),
        };
        self.advance();
        self.done += 1;
        Ok(vec![NewRecord {
            kind,
            label: self.label.clone(),
            start: start as f32,
            end: end.min(self.duration) as f32,
            vec,
        }])
    }

    /// Embed the loaded group of windows in one model call.
    fn step_group(
        &mut self,
        embedder: &Embedder,
        ctx: &mut LlamaContext<'_>,
    ) -> Result<Vec<NewRecord>> {
        let parts = std::mem::take(&mut self.group);
        let windows: Vec<embed::WindowIn> = parts
            .iter()
            .map(|part| embed::WindowIn {
                text: embed::window_text(&part.times),
                media: part.files.clone(),
            })
            .collect();
        let vecs = embedder.embed_windows_in(ctx, &windows)?;
        if vecs.len() != parts.len() {
            bail!(
                "the model gave {} vectors for {} windows",
                vecs.len(),
                parts.len()
            );
        }

        let mut out = Vec::with_capacity(parts.len());
        for (part, vec) in parts.iter().zip(vecs) {
            out.push(NewRecord {
                kind: store::KIND_WINDOW,
                label: self.label.clone(),
                start: part.start as f32,
                end: (part.start + self.plan.window).min(self.duration) as f32,
                vec,
            });
        }

        let made = out.len();
        self.window_at += made;
        self.done += made;
        if self.window_at >= self.windows.len() {
            self.phase = Phase::Done;
        }
        Ok(out)
    }
```

Replace `remaining_in_batch` with:

```rust
    fn remaining_in_batch(&self) -> usize {
        if self.phase == Phase::Windows {
            // One group of windows fills one model call, so one item is ready.
            return usize::from(!self.group.is_empty());
        }
        self.files.len().saturating_sub(self.at() - self.batch_from)
    }
```

Keep the `advance` arm for windows, and put the reason in the comment:

```rust
            // Window mode advances a whole group, in `step_group`.
            Phase::Windows | Phase::Done => {}
```

Add `self.group.clear();` to `clear_media`.

Add the accessor below `done`:

```rust
    /// The lines that the run reported, in order.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
```

Import `embed::{N_WINDOW_BATCH, WindowIn}` in the `use` block of `videoembed/src/index.rs`.

- [ ] **Step 5: Update the three call sites of `step`**

`index_video` becomes:

```rust
/// Index one video under one plan. Embed every item of the plan.
///
/// The callback runs after each model call with the number of records made so
/// far and the record count of the whole video. The answer holds the records
/// and the lines that the run reported.
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    plan: &crate::plan::Plan,
    media: &Path,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<NewRecord>, Vec<String>)> {
    let mut run = IndexRun::start_with(video, plan, media)?;
    let total = run.total();
    let mut ctx = embedder.new_context()?;
    let mut records = Vec::with_capacity(total);
    loop {
        let made = run.step(embedder, &mut ctx)?;
        if made.is_empty() {
            break;
        }
        records.extend(made);
        on_step(run.done(), total);
    }
    Ok((records, run.notes().to_vec()))
}
```

`reindex` returns `Result<(usize, Vec<String>)>`, it takes the pair from `index_video`, and it ends with:

```rust
    let count = records.len();
    store.forget(&label)?;
    store.append(records)?;
    store.set_plan(&label, plan)?;
    Ok((count, notes))
```

`videoembed/src/main.rs` at the `index` command becomes:

```rust
    let (records, notes) = index::index_video(embedder, &video, &plan, &media, &mut report)?;
    for line in &notes {
        println!("{line}");
    }
    let count = records.len();
```

and at the `plan` command:

```rust
    let (count, notes) = index::reindex(embedder, store, &path, &wanted, &mut report)?;
    for line in &notes {
        println!("{line}");
    }
```

`videoembed/src/bin/web.rs` at the `step` call becomes:

```rust
    let outcome = job.run.as_mut().unwrap().step(embedder, ctx);
    let whole = run_left_a_whole_index(&outcome);
    let mut finished = false;
    match outcome {
        Ok(records) => {
            finished = records.is_empty();
            job.records.extend(records);
        }
        Err(err) => {
            fail(progress, &err);
            finished = true;
        }
    }
```

`run_left_a_whole_index` becomes:

```rust
fn run_left_a_whole_index(outcome: &Result<Vec<NewRecord>>) -> bool {
    matches!(outcome, Ok(records) if records.is_empty())
}
```

`videoembed/examples/window_probe.rs` at the `step` call becomes:

```rust
    while let Ok(records) = run.step(embedder, &mut ctx) {
        if records.is_empty() {
            break;
        }
        for record in records {
            if record.kind == KIND_FRAME {
                out.push((record.start as f64, record.end as f64, record.vec));
            }
        }
    }
```

- [ ] **Step 6: Write the failing end-to-end test**

Add to `videoembed/tests/model_batch.rs`:

```rust
/// The run indexes every window of the test video through one batch path.
///
/// This is the end-to-end pin. It costs about one minute on the CPU build,
/// because the video holds 29 windows.
#[test]
fn the_run_indexes_every_window_of_the_test_video() {
    let Some((embedder, _guard)) = the_model() else {
        eprintln!("skipped: the model is not here");
        return;
    };
    let Some(video) = video() else {
        eprintln!("skipped: the test video is not here");
        return;
    };
    let media = scratch("index_media");
    let plan = plan::Plan::of(plan::Preset::Object);
    let duration = media::probe_duration(&video).expect("the duration comes back");
    let want = plan.window_starts(duration).len();

    let mut report = |_made: usize, _total: usize| {};
    let (records, notes) = index::index_video(embedder, &video, &plan, &media, &mut report)
        .expect("the run ends");

    assert_eq!(records.len(), want, "one record for each window");
    assert!(notes.is_empty(), "the Object window holds the budget: {notes:?}");
    assert!(
        records.iter().all(|record| !record.vec.is_empty()),
        "every record holds a vector"
    );
    for pair in records.windows(2) {
        assert!(pair[1].start > pair[0].start, "the windows come in time order");
    }
    // The last windows reach past the audio track, so they hold frames alone.
    assert!(
        records.last().unwrap().start as f64 > 20.0,
        "the fixture covers the audio end"
    );
    let _ = std::fs::remove_dir_all(&media);
}
```

- [ ] **Step 7: Run the tests and watch them pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --test model_batch`

Expected: PASS, one record for each of the 29 windows, and every start time in order.

Then run the whole suite:

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`

Expected: PASS, with the three call sites of `step` updated.

- [ ] **Step 8: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/bin/web.rs videoembed/examples/window_probe.rs videoembed/src/main.rs videoembed/tests/model_batch.rs
git commit -m "feat: group four windows into one index step

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: The token budget

**Files:**
- Modify: `videoembed/src/index.rs` (`budget_note`, the budget rule in `step_group`, `Progress.note`)
- Modify: `videoembed/src/bin/web.rs` (the step loop, `begin_file`)
- Modify: `videoembed/assets/app.js`
- Test: the `tests` module of `videoembed/src/index.rs`, `videoembed/src/bin/web.rs`, and `videoembed/tests/model_batch.rs`

**Interfaces:**
- Consumes: `embed::BUDGET_TOKENS`, `embed::tokens_of`, `WindowPart::keep_frames`, `IndexRun::notes`.
- Produces:
  - `index::budget_note(start: f64, tokens: usize) -> String`
  - `index::Progress::note: String`
  - `IndexRun::notes()` holds one line for each window that lost its audio.

- [ ] **Step 1: Write the failing unit tests**

Add to the `tests` module of `videoembed/src/index.rs`:

```rust
    /// The rule of one window, in words, for the reader.
    #[test]
    fn the_budget_note_names_the_window_and_the_count() {
        assert_eq!(
            budget_note(3.0, 1500),
            "window at 3.0 s holds 1500 tokens, above the budget of 1024, \
             so the record keeps the frames alone"
        );
    }

    /// A window that passes the budget keeps its frames and drops its clips.
    #[test]
    fn a_window_that_passes_the_budget_keeps_its_frames() {
        let mut part = WindowPart {
            start: 3.0,
            files: vec![
                PathBuf::from("f0.jpg"),
                PathBuf::from("f1.jpg"),
                PathBuf::from("a0.wav"),
                PathBuf::from("a1.wav"),
            ],
            times: vec![3.0, 4.0, 3.0, 4.0],
            frames: 2,
        };

        part.keep_frames();

        assert_eq!(part.frames, 2);
        assert_eq!(part.times, vec![3.0, 4.0], "the markers follow the media");
        for file in &part.files {
            let name = file.to_string_lossy();
            assert!(name.ends_with(".jpg"), "{name} is not a frame");
        }
    }
```

Add to the `tests` module of `videoembed/src/bin/web.rs`, beside the test `a_new_file_keeps_the_error_of_an_earlier_file`:

```rust
    /// A new file clears the line of the file before it.
    ///
    /// The line names one window of one file. A stale line then reads as a
    /// fault of the file that runs now.
    #[test]
    fn a_new_file_clears_the_note_of_the_last_one() {
        let progress = Arc::new(Mutex::new(Progress::default()));
        progress.lock().unwrap().note = "window at 3.0 s holds 1500 tokens".to_string();

        begin_file(&progress, "clip-two.mp4".to_string(), 2, 12);

        assert!(progress.lock().unwrap().note.is_empty());
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::the_budget_note`

Expected: FAIL to compile with `cannot find function \`budget_note\` in this scope`.

- [ ] **Step 3: Write `budget_note` and the rule**

Add above `pub struct IndexRun` in `videoembed/src/index.rs`:

```rust
/// The line about one window that lost its audio to the token budget.
///
/// The run reports every such window, so the reader learns why the record
/// holds no sound.
pub fn budget_note(start: f64, tokens: usize) -> String {
    format!(
        "window at {start:.1} s holds {tokens} tokens, above the budget of {}, \
         so the record keeps the frames alone",
        embed::BUDGET_TOKENS
    )
}
```

Insert the rule at the top of `step_group`, in place of the line `let parts = std::mem::take(&mut self.group);`:

```rust
        let mut parts = std::mem::take(&mut self.group);
        for part in &mut parts {
            if part.files.len() == part.frames {
                // A window with no audio cannot lose anything.
                continue;
            }
            let text = embed::window_text(&part.times);
            let count = embedder.tokens_of(&text, &part.files)?;
            if count > embed::BUDGET_TOKENS {
                part.keep_frames();
                self.notes.push(budget_note(part.start, count));
            }
        }
```

The lines below it stay as Task 4 wrote them.

- [ ] **Step 4: Add the progress field and the page line**

Add to `Progress` in `videoembed/src/index.rs`:

```rust
    /// The newest line of the running job. Empty when the job reports nothing.
    pub note: String,
```

Add to `begin_file` in `videoembed/src/bin/web.rs`:

```rust
    // The line of the file before it names a window of that file.
    held.note.clear();
```

Add to `step_job`, in the branch that updates a running job:

```rust
        held.note = job
            .run
            .as_ref()
            .and_then(|run| run.notes().last())
            .cloned()
            .unwrap_or_default();
```

Change the running branch of `showProgress` in `videoembed/assets/app.js`:

```js
  if (p.running) {
    const rate = p.per_second > 0 ? " - " + p.per_second.toFixed(2) + " items a second" : "";
    const note = p.note ? " " + p.note : "";
    $("progress-text").textContent =
      "File " + p.file_index + "/" + p.file_count +
      ", item " + p.window + "/" + p.window_count +
      " - " + p.file + rate + note;
  } else {
```

- [ ] **Step 5: Run the unit tests and watch them pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::the_budget_note`

Expected: PASS.

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin web`

Expected: PASS, with the new note test.

- [ ] **Step 6: Write the failing real test**

Add to `videoembed/tests/model_batch.rs`:

```rust
/// A window above the token budget loses its audio, and the run says so.
///
/// The plan names a 10 s window of 10 frames with its audio. Ten frames and
/// their clips pass the budget of 1024 tokens, so the window drops its clips.
/// The frames alone stay under the budget, so the record keeps them.
#[test]
fn a_ten_second_window_drops_its_audio() {
    let Some((embedder, _guard)) = the_model() else {
        eprintln!("skipped: the model is not here");
        return;
    };
    let Some(video) = video() else {
        eprintln!("skipped: the test video is not here");
        return;
    };
    let plan = plan::Plan {
        mode: plan::Mode::Window,
        window: 10.0,
        stride: 10.0,
        frames: 10,
        audio: true,
    };
    let media = scratch("budget_media");
    let mut run = index::IndexRun::start_with(&video, &plan, &media).expect("the run starts");
    let mut ctx = embedder.new_context().expect("the context is made");

    let mut records = Vec::new();
    loop {
        let made = run.step(embedder, &mut ctx).expect("the step ends");
        if made.is_empty() {
            break;
        }
        records.extend(made);
    }

    let duration = media::probe_duration(&video).expect("the duration comes back");
    assert_eq!(records.len(), plan.window_starts(duration).len());
    assert!(!run.notes().is_empty(), "a window over the budget reports one line");
    assert!(
        run.notes().iter().all(|line| line.contains("above the budget")),
        "got {:?}",
        run.notes()
    );

    // The frames alone stay under the budget, so the record keeps them.
    let (frames, _clips) = one_second_cut(&video, &scratch("budget_frames"));
    let alone = index::frames_in(&frames, 0.0, 10.0, 1.0);
    let times = index::window_times(0.0, 1.0, alone.len(), 0);
    let count = embedder
        .tokens_of(&embed::window_text(&times), &alone)
        .expect("the count comes back");
    println!("the frames alone: {count} tokens over {} parts", alone.len());
    assert!(
        count < embed::BUDGET_TOKENS,
        "the frames alone hold {count} tokens, above the budget"
    );
    let _ = std::fs::remove_dir_all(&media);
}
```

- [ ] **Step 7: Run the test and watch it pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --test model_batch a_ten_second`

Expected: PASS, with the printed frame count. Write the real token counts in the ledger.

- [ ] **Step 8: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`

Expected: PASS, lib and console and web and model_batch.

- [ ] **Step 9: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/bin/web.rs videoembed/assets/app.js videoembed/tests/model_batch.rs
git commit -m "feat: drop the audio of a window over the token budget

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: The docs of the batch

**Files:**
- Modify: `README.md`
- Modify: `docs/deferred-work.md`
- Modify: `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md`

**Interfaces:**
- Consumes: the measured numbers of Task 1 and Task 5 from the ledger.
- Produces: no code.

- [ ] **Step 1: Update the README**

In `README.md`, replace step 3 and step 4 of "How it works" with:

```
3. The model reads four windows in one call. Each window holds its frames, its
   audio, and a `MM:SS` marker before each part.
4. One decode runs, and one pooling step gives one vector for each window.
```

Replace the paragraph after the numbered list with:

```
Step 3 is the important part. The model sees the whole window at once. A
separate decode for each part would pool only the last part. Four windows share
one call, and each window holds its own sequence, so the vectors do not move
with the size of the batch. A window above 1024 tokens loses its audio, and the
run reports that window on one line.
```

- [ ] **Step 2: Update the deferred work list**

In `docs/deferred-work.md`, remove the rows "Several windows in one model batch", "The timestamp text in the batch", and "The audio token budget" from the table of section 1, and remove their subsections. Renumber the table and the subsections so the list runs from 1 without a gap.

Add to the paragraph above section 1:

```
Plan B landed on 2026-10-12 on the `window-batch` branch. It answered the
window batch, the timestamp text, and the audio token budget.
```

Add one row to the table of section 3:

```
| The frames alone over the budget | A window whose frames alone pass 1024 tokens keeps them and goes into the batch, because the batch holds 2048 tokens. The run reports the window and does not refuse the plan. The Object window measures N tokens, so the case needs a named window of many frames. |
```

Replace the `N` with the number that Task 1 measured.

- [ ] **Step 3: Update the spec status**

In `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md`, replace the status lines with:

```
Status: approved. The user settled the four open decisions on 2026-10-12.
Plan A: `docs/superpowers/plans/2026-10-12-cut-pass.md`.
Plan B: `docs/superpowers/plans/2026-10-12-window-batch.md`.
Plan C waits for its own plan document.
```

- [ ] **Step 4: Run the whole suite one more time**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/deferred-work.md docs/superpowers/specs/2026-10-12-cut-batch-search-design.md
git commit -m "docs: record the window batch and the token budget

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```
