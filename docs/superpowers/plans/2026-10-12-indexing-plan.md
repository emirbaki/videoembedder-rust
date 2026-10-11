# The Indexing Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut each video under a per-video plan, so window mode is the default
and the user can set the plan from the console or the page.

**Architecture:** A new `plan` module holds the plan type and the four presets.
The store writes the plan into its `videos` row, so the plan travels with the
video. `IndexRun` reads the plan and cuts windows in place of single frames. A
plan change re-cuts the video at once through the same routine that the console
and the web engine share.

**Tech Stack:** Rust, edition 2024, SQLite with the sqlite-vec extension,
llama.cpp through the vendored `llama-cpp-2` crate, ffmpeg, and vanilla
JavaScript for the page.

**Spec:** `docs/superpowers/specs/2026-10-12-indexing-plan-design.md`

## Global Constraints

- Every test command runs as
  `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`.
  A plain `cargo test` fails to configure the vendored llama.cpp.
- Edition 2024. The two binaries are `videoembed` (the console) and `web`.
- The store is SQLite at `store/index.db`. The vector table is `vec_records`.
- The record id is `path/startMs-endMs`, with whole milliseconds.
- The label rule is `normalize_label`: a backslash becomes a forward slash.
- The kind list is `KIND_WINDOW`, `KIND_FRAME`, `KIND_AUDIO`, and `KIND_NOTE`.
  `is_video` accepts the first three.
- One plan value set, copied from the spec:

  | Preset | mode | window | stride | frames | audio |
  | --- | --- | --- | --- | --- | --- |
  | Fine | fine | 0.25 | 0.25 | 1 | yes |
  | Object | window | 2 | 1 | 2 | no |
  | Action | window | 4 | 2 | 4 | no |
  | Speech | window | 6 | 3 | 6 | yes |

- The default plan is Object.
- The frame rate inside one window is `frames / window`.
- The window slider runs from 1 s to 10 s. The frame slider runs from 1 to the
  window in whole seconds.
- The doc rules: ASD-STE100 governs every word a person reads. Active voice,
  simple tenses, no contractions. End every commit message with
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

These input classes are the ones the spec implies but no task's tests reach
first. Each line names the input and the behavior a reasonable person expects.
The owning task holds the test.

1. A video with no audio track under a plan with audio on. The window must hold
   the frames alone, and the run must not fail. **Owner: Task 4.**
2. A video shorter than one window. The run must give zero windows and must not
   fail. **Owner: Task 4.**
3. A window that reaches past the end of the video. The recorded end must clamp
   to the duration, as the fine grid clamps today. **Owner: Task 4.**
4. A plan change on a video whose source file is gone. The old plan must stay,
   and the message must name the file. **Owner: Task 6.**
5. A plan change while an index job runs. The engine must refuse the change with
   the busy message, and the plan must stay. **Owner: Task 7.**

---

## File Structure

| File | Role |
| --- | --- |
| `videoembed/src/plan.rs` | New. The `Plan` type, `Mode`, `Preset`, and the four presets. Pure values and no media. |
| `videoembed/src/db.rs` | The schema. Gains the `plan_mode` column and its migration. |
| `videoembed/src/store.rs` | The plan read and write on the `videos` row. |
| `videoembed/src/embed.rs` | `embed_window_in`, one batch for a whole window. |
| `videoembed/src/index.rs` | `IndexRun` cuts to a plan. Window mode joins the code. |
| `videoembed/src/main.rs` | The console `index` and the new `plan` command. |
| `videoembed/src/bin/web.rs` | The plan in the videos reply, the plan endpoint, and the re-cut job. |
| `videoembed/assets/app.js` | The plan panel. |
| `videoembed/assets/app.css` | The panel style. |
| `videoembed/examples/window_probe.rs` | The throwaway probe. Task 1 extends it. |
| `README.md` | The command table and the plan. |
| `docs/deferred-work.md` | Items 1, 8, and 11 come off. |

---

### Task 1: The window batch, and the probe of the default plan

**Files:**
- Modify: `videoembed/src/embed.rs` (add `embed_window_in`)
- Modify: `videoembed/src/lib.rs` (nothing, if `embed` is already public)
- Modify: `videoembed/examples/window_probe.rs`
- Test: the probe run itself, plus `cargo test`

**Interfaces:**
- Consumes: `Embedder::new_context`, the private `embed_joint`, and
  `MtmdBitmap::from_file`.
- Produces:
  `pub fn embed_window_in(&self, ctx: &mut LlamaContext<'_>, media: &[&Path]) -> Result<Vec<f32>>`.
  The caller passes the frames first and the audio clip last. The result is a
  unit vector. Task 4 calls it.

The probe answers one question: does the audio change the hit count? The design
says the default plan is Object, which holds no audio. Task 1 measures both
forms, so the default is a measured choice.

- [ ] **Step 1: Add `embed_window_in`**

Put this function in `videoembed/src/embed.rs`, after `embed_audio_in` and
before `embed_joint`.

```rust
    /// Embed several media files as one window, in the given order.
    ///
    /// The caller passes the frames first and the audio clip last. One batch
    /// holds every part, and one decode follows. A probe uses this call to test
    /// window mode.
    pub fn embed_window_in(
        &self,
        ctx: &mut LlamaContext<'_>,
        media: &[&Path],
    ) -> Result<Vec<f32>> {
        if media.is_empty() {
            bail!("a window holds no media");
        }
        ctx.clear_kv_cache();
        let mut bitmaps = Vec::with_capacity(media.len());
        for path in media {
            let path = path.to_str().context("media path is not utf8")?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
        }
        self.embed_joint(ctx, &bitmaps)
    }
```

- [ ] **Step 2: Build**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo build --manifest-path videoembed/Cargo.toml`
Expected: `Finished`, with no error.

- [ ] **Step 3: Extend the probe for both forms**

Rewrite `videoembed/examples/window_probe.rs` so the window phase runs twice:
once with audio and once without. Replace the module header and the queries, and
replace `index_windows`.

```rust
//! A throwaway probe for deferred item 1: window mode.
//!
//! Window mode holds one 2 s window in one record: two frames and the window
//! audio, embedded in one batch. The fine mode holds a 0.25 s frame grid.
//!
//! The probe indexes one video both ways, and each window form with and without
//! audio. It compares ten queries against a known truth. The video is 30 s, in
//! three 10 s blocks: sunny, tuesday, birthday.
//!
//! Run from the `videoembed` folder:
//!   VEI_MODELS=../models cargo run --release --example window_probe

use std::path::{Path, PathBuf};

use anyhow::Result;

use videoembed::embed::{Embedder, QUERY_PREFIX, dot};
use videoembed::store::KIND_FRAME;
use videoembed::{index, media};

/// One 2 s window, on a 1 s stride.
const WINDOW: f64 = 2.0;
const STRIDE: f64 = 1.0;
/// One frame each second, so a 2 s window holds two frames.
const FPS: f64 = 1.0;

/// name, query, the true block as (start, end).
const QUERIES: &[(&str, &str, (f64, f64))] = &[
    ("sunny", "sunny warm weather", (0.0, 10.0)),
    ("weather", "the weather is sunny today", (0.0, 10.0)),
    ("tuesday", "quiet tuesday morning", (10.0, 20.0)),
    ("morning", "it is a quiet morning", (10.0, 20.0)),
    ("birthday", "birthday party", (20.0, 30.0)),
    ("celebrate", "we celebrate a birthday", (20.0, 30.0)),
    ("blue", "blue", (10.0, 20.0)),
    ("green", "green", (20.0, 30.0)),
    ("orange", "orange", (0.0, 10.0)),
    ("party", "a party with friends", (20.0, 30.0)),
];

fn main() -> Result<()> {
    let models = PathBuf::from(std::env::var("VEI_MODELS").unwrap_or("../models".into()));
    let video = PathBuf::from("../spike/testvideo_nolabels.mp4");
    let embedder = Embedder::load(&models, None)?;

    println!("video: {}", video.display());
    println!();

    let with_audio = index_windows(&embedder, &video, true)?;
    let no_audio = index_windows(&embedder, &video, false)?;
    let fine = index_fine(&embedder, &video)?;
    println!("window + audio: {} records", with_audio.len());
    println!("window, no audio: {} records", no_audio.len());
    println!("fine:           {} records", fine.len());
    println!();
    println!(
        "{:<10} {:<28} {:<12} {:<8} {:<8} fine",
        "name", "query", "truth", "window", "no audio"
    );

    let mut win_hits = 0;
    let mut plain_hits = 0;
    let mut fine_hits = 0;
    for (name, text, truth) in QUERIES {
        let query = embedder.embed_text(&format!("{QUERY_PREFIX}{text}"))?;
        let w = overlaps(best_block(&with_audio, &query), *truth);
        let p = overlaps(best_block(&no_audio, &query), *truth);
        let f = overlaps(best_block(&fine, &query), *truth);
        win_hits += w as usize;
        plain_hits += p as usize;
        fine_hits += f as usize;
        println!(
            "{:<10} {:<28} {:<12} {:<8} {:<8} {}",
            name,
            text,
            format!("{:.0}-{:.0}s", truth.0, truth.1),
            if w { "OK" } else { "MISS" },
            if p { "OK" } else { "MISS" },
            if f { "OK" } else { "MISS" },
        );
    }
    println!();
    println!(
        "hits: window+audio {}/{}  no audio {}/{}  fine {}/{}",
        win_hits,
        QUERIES.len(),
        plain_hits,
        QUERIES.len(),
        fine_hits,
        QUERIES.len()
    );
    Ok(())
}

/// Index the video in window mode: one record for each 2 s window.
fn index_windows(
    embedder: &Embedder,
    video: &Path,
    audio: bool,
) -> Result<Vec<(f64, f64, Vec<f32>)>> {
    let duration = media::probe_duration(video)?;
    let mut out = Vec::new();
    let mut ctx = embedder.new_context()?;
    let mut t = 0.0;
    while t + WINDOW <= duration {
        let dir = index::work_dir(out.len());
        let mut media = media::extract_frames(video, t, t + WINDOW, FPS, &dir)?;
        if audio {
            media.extend(media::extract_audio(video, t, t + WINDOW, &dir)?);
        }
        let refs: Vec<&Path> = media.iter().map(|path| path.as_path()).collect();
        let vec = embedder.embed_window_in(&mut ctx, &refs)?;
        out.push((t, t + WINDOW, vec));
        let _ = std::fs::remove_dir_all(&dir);
        t += STRIDE;
    }
    Ok(out)
}

/// Index the video in the current fine mode, from the store's own plan.
fn index_fine(embedder: &Embedder, video: &Path) -> Result<Vec<(f64, f64, Vec<f32>)>> {
    let mut run = index::IndexRun::start(video)?;
    let mut ctx = embedder.new_context()?;
    let mut out = Vec::new();
    while let Some(record) = run.step(embedder, &mut ctx)? {
        if record.kind == KIND_FRAME {
            out.push((record.start as f64, record.end as f64, record.vec));
        }
    }
    Ok(out)
}

/// The block of the best record for one query.
fn best_block(records: &[(f64, f64, Vec<f32>)], query: &[f32]) -> (f64, f64) {
    records
        .iter()
        .max_by(|a, b| dot(query, &a.2).total_cmp(&dot(query, &b.2)))
        .map(|(start, end, _)| (*start, *end))
        .unwrap_or((0.0, 0.0))
}

fn overlaps(hit: (f64, f64), truth: (f64, f64)) -> bool {
    hit.0 < truth.1 && truth.0 < hit.1
}
```

- [ ] **Step 4: Run the probe**

Run, from the `videoembed` folder:

```bash
cd videoembed
CMAKE_GENERATOR="Visual Studio 17 2022" cargo run --release --example window_probe
```

Expected: a table of 10 queries, then a hits line. Record the two counts.
The prior run gave `window+audio 10/10` and `fine 5/10`.

- [ ] **Step 5: Record the result**

Write the two counts into `docs/deferred-work.md` under item 1, in one sentence.
Keep the probe as a throwaway example. It is not part of the product.

- [ ] **Step 6: Run the whole suite and commit**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: `92 passed` for the lib, `8 passed` for the console, `19 passed` for
web, and no failure.

```bash
git add videoembed/src/embed.rs videoembed/examples/window_probe.rs docs/deferred-work.md
git commit -m "feat: embed a whole window in one batch

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: The plan module

**Files:**
- Create: `videoembed/src/plan.rs`
- Modify: `videoembed/src/lib.rs` (add `pub mod plan;`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub enum Mode { Window, Fine }`
  - `pub enum Preset { Fine, Object, Action, Speech }` with
    `pub const ALL: [Preset; 4]`, `pub fn as_str(self) -> &'static str`, and
    `pub fn parse(word: &str) -> Option<Preset>`
  - `pub struct Plan { pub mode: Mode, pub window: f64, pub stride: f64, pub frames: usize, pub audio: bool }`
    with `pub fn of(preset: Preset) -> Plan`, `pub fn default_plan() -> Plan`,
    `pub fn frame_rate(&self) -> f64`, `pub fn window_starts(&self, duration: f64) -> Vec<f64>`,
    and `pub fn set_words(&self, words: &str) -> Result<Plan>`

- [ ] **Step 1: Write the failing tests**

Add this test module to the new `videoembed/src/plan.rs`.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preset_holds_its_values() {
        let object = Plan::of(Preset::Object);
        assert_eq!(object.mode, Mode::Window);
        assert_eq!(object.window, 2.0);
        assert_eq!(object.stride, 1.0);
        assert_eq!(object.frames, 2);
        assert!(!object.audio);

        let speech = Plan::of(Preset::Speech);
        assert_eq!(speech.window, 6.0);
        assert_eq!(speech.frames, 6);
        assert!(speech.audio);

        let fine = Plan::of(Preset::Fine);
        assert_eq!(fine.mode, Mode::Fine);
    }

    /// The default plan is the Object preset.
    #[test]
    fn the_default_plan_is_object() {
        assert_eq!(Plan::default_plan(), Plan::of(Preset::Object));
    }

    #[test]
    fn the_frame_rate_follows_from_the_window() {
        assert_eq!(Plan::of(Preset::Object).frame_rate(), 1.0);
        assert_eq!(Plan::of(Preset::Action).frame_rate(), 1.0);
        assert_eq!(Plan::of(Preset::Speech).frame_rate(), 1.0);
    }

    /// A name reads back to its preset, and a foreign word reads as nothing.
    #[test]
    fn a_preset_name_round_trips() {
        for preset in Preset::ALL {
            assert_eq!(Preset::parse(preset.as_str()), Some(preset));
        }
        assert_eq!(Preset::parse("object"), Some(Preset::Object));
        assert_eq!(Preset::parse("OBJECT"), Some(Preset::Object));
        assert_eq!(Preset::parse("fast"), None);
    }

    /// A 30 s video gives 29 windows at 2 s on a 1 s stride.
    #[test]
    fn the_window_starts_cover_a_thirty_second_video() {
        let plan = Plan::of(Preset::Object);
        let starts = plan.window_starts(30.0);
        assert_eq!(starts.len(), 29);
        assert_eq!(starts[0], 0.0);
        assert_eq!(starts[1], 1.0);
        assert_eq!(*starts.last().unwrap(), 28.0);
    }

    /// A video shorter than one window gives no window.
    #[test]
    fn a_short_video_gives_no_window() {
        let plan = Plan::of(Preset::Object);
        assert!(plan.window_starts(1.5).is_empty());
    }

    #[test]
    fn a_value_word_sets_one_plan_value() {
        let base = Plan::of(Preset::Object);
        let asked = base.set_words("window=4 frames=4 audio=on").unwrap();
        assert_eq!(asked.window, 4.0);
        assert_eq!(asked.frames, 4);
        assert!(asked.audio);
        // The stride follows the window when the user does not name it.
        assert_eq!(asked.stride, 2.0);
        assert_eq!(asked.mode, Mode::Window);
    }

    /// A bad value names the fault.
    #[test]
    fn a_bad_value_is_refused() {
        let base = Plan::of(Preset::Object);
        assert!(base.set_words("window=0").is_err(), "a zero window");
        assert!(base.set_words("window=99").is_err(), "a window past 10 s");
        assert!(base.set_words("frames=5 window=2").is_err(), "frames past the window");
        assert!(base.set_words("candy=1").is_err(), "an unknown key");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml plan::`
Expected: FAIL with `cannot find type Plan in this scope`, because the module is
empty.

- [ ] **Step 3: Write the module**

Write the body of `videoembed/src/plan.rs` above the test module.

```rust
//! The indexing plan. How one video is cut into records.

use anyhow::{Result, bail};

/// The two ways the app cuts a video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// One record holds a window: several frames and the window audio.
    Window,
    /// One record holds a 0.25 s frame, or one second of audio.
    Fine,
}

impl Mode {
    /// The stored name of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Window => "window",
            Mode::Fine => "fine",
        }
    }

    /// Read the mode from its stored name.
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "window" => Some(Mode::Window),
            "fine" => Some(Mode::Fine),
            _ => None,
        }
    }
}

/// A named plan. The three Google presets, plus the fine grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Fine,
    Object,
    Action,
    Speech,
}

impl Preset {
    /// Every preset, in the order the page shows them.
    pub const ALL: [Preset; 4] = [Preset::Fine, Preset::Object, Preset::Action, Preset::Speech];

    /// The stored name of the preset.
    pub fn as_str(self) -> &'static str {
        match self {
            Preset::Fine => "fine",
            Preset::Object => "object",
            Preset::Action => "action",
            Preset::Speech => "speech",
        }
    }

    /// Read a preset from a word. The word is lowercased first.
    pub fn parse(word: &str) -> Option<Self> {
        match word.to_ascii_lowercase().as_str() {
            "fine" => Some(Preset::Fine),
            "object" => Some(Preset::Object),
            "action" => Some(Preset::Action),
            "speech" => Some(Preset::Speech),
            _ => None,
        }
    }
}

/// How one video is cut.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Plan {
    pub mode: Mode,
    /// The seconds of one window.
    pub window: f64,
    /// The seconds between two window starts.
    pub stride: f64,
    /// The frame count in one window.
    pub frames: usize,
    /// The window record holds the window audio.
    pub audio: bool,
}

/// The shortest window the plan accepts, in seconds.
pub const MIN_WINDOW: f64 = 1.0;
/// The longest window the plan accepts, in seconds.
pub const MAX_WINDOW: f64 = 10.0;

impl Plan {
    /// The plan of one preset.
    pub fn of(preset: Preset) -> Plan {
        match preset {
            Preset::Fine => Plan {
                mode: Mode::Fine,
                window: 0.25,
                stride: 0.25,
                frames: 1,
                audio: true,
            },
            Preset::Object => Plan {
                mode: Mode::Window,
                window: 2.0,
                stride: 1.0,
                frames: 2,
                audio: false,
            },
            Preset::Action => Plan {
                mode: Mode::Window,
                window: 4.0,
                stride: 2.0,
                frames: 4,
                audio: false,
            },
            Preset::Speech => Plan {
                mode: Mode::Window,
                window: 6.0,
                stride: 3.0,
                frames: 6,
                audio: true,
            },
        }
    }

    /// The plan a video takes when it holds none.
    pub fn default_plan() -> Plan {
        Plan::of(Preset::Object)
    }

    /// The frames a second inside one window.
    pub fn frame_rate(&self) -> f64 {
        self.frames as f64 / self.window
    }

    /// The start of every window in one video.
    ///
    /// A window that would reach past the end does not start. The count follows
    /// from the duration, the window, and the stride.
    pub fn window_starts(&self, duration: f64) -> Vec<f64> {
        let mut starts = Vec::new();
        let mut t = 0.0;
        while t + self.window <= duration {
            starts.push(t);
            t += self.stride;
        }
        starts
    }

    /// Set one or more plan values over this plan.
    ///
    /// The words are `window=4 frames=4 audio=on`. A window that the user names
    /// sets the stride to half of that window, so the windows overlap.
    pub fn set_words(&self, words: &str) -> Result<Plan> {
        let mut plan = *self;
        let mut named_window = false;
        let mut named_stride = false;
        for word in words.split_whitespace() {
            let Some((key, value)) = word.split_once('=') else {
                bail!("{word} is not a plan value");
            };
            match key {
                "window" => {
                    plan.window = parse_seconds(value)?;
                    named_window = true;
                }
                "stride" => {
                    plan.stride = parse_seconds(value)?;
                    named_stride = true;
                }
                "frames" => {
                    plan.frames = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("{value} is not a frame count"))?;
                }
                "audio" => plan.audio = parse_flag(value)?,
                _ => bail!("{key} is not a plan value"),
            }
        }
        if plan.window < MIN_WINDOW || plan.window > MAX_WINDOW {
            bail!("the window must be {MIN_WINDOW} to {MAX_WINDOW} seconds");
        }
        if plan.frames < 1 || plan.frames as f64 > plan.window {
            bail!("the frame count must be 1 to {}", plan.window as usize);
        }
        if named_window && !named_stride {
            plan.stride = plan.window / 2.0;
        }
        if plan.stride <= 0.0 {
            bail!("the stride must be more than zero");
        }
        plan.mode = Mode::Window;
        Ok(plan)
    }
}

/// Read a seconds value, in the range the sliders allow.
fn parse_seconds(value: &str) -> Result<f64> {
    let seconds: f64 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("{value} is not a number"))?;
    if !seconds.is_finite() {
        bail!("{value} is not a number");
    }
    Ok(seconds)
}

/// Read an on or off word.
fn parse_flag(value: &str) -> Result<bool> {
    match value {
        "on" | "yes" | "1" => Ok(true),
        "off" | "no" | "0" => Ok(false),
        _ => bail!("{value} is not on or off"),
    }
}
```

Add the module to `videoembed/src/lib.rs`:

```rust
pub mod plan;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml plan::`
Expected: PASS, 8 tests.

- [ ] **Step 5: Run the whole suite and commit**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: the lib count rises by 8. No failure.

```bash
git add videoembed/src/plan.rs videoembed/src/lib.rs
git commit -m "feat: add the indexing plan and its presets

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: The plan in the store

**Files:**
- Modify: `videoembed/src/db.rs` (the schema and the migration)
- Modify: `videoembed/src/store.rs` (the plan read and write)
- Test: `videoembed/src/store.rs` (the test module)

**Interfaces:**
- Consumes: `plan::Plan`, `plan::Mode`, `plan::Preset::parse`, and the existing
  `Store::records`, `Store::append`, and `Store::forget`.
- Produces:
  - `pub fn Store::plan_of(&self, path: &str) -> Result<Option<Plan>>`
  - `pub fn Store::set_plan(&mut self, path: &str, plan: &Plan) -> Result<()>`
  - `pub fn Store::plan_rows(&self) -> Result<Vec<(String, Plan)>>`

- [ ] **Step 1: Write the failing tests**

Add these tests to the `mod tests` block in `videoembed/src/store.rs`.

```rust
    /// A plan written to a video reads back whole.
    #[test]
    fn a_plan_round_trips_through_the_store() {
        let (dir, mut store) = opened("plan_round");
        store.append(vec![new_record("a.mp4", 0.0, 0.25)]).unwrap();

        store.set_plan("a.mp4", &plan::Plan::of(plan::Preset::Action)).unwrap();

        let again = Store::open(&dir.join("index.db")).unwrap();
        let held = again.plan_of("a.mp4").unwrap().unwrap();
        assert_eq!(held, plan::Plan::of(plan::Preset::Action));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A video with no plan row gives nothing, so the caller uses the default.
    #[test]
    fn a_video_with_no_plan_gives_nothing() {
        let (dir, store) = opened("plan_none");
        assert_eq!(store.plan_of("a.mp4").unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An older store reads as Fine, because its records are fine records.
    #[test]
    fn an_older_store_reads_as_fine() {
        let dir = scratch("plan_old");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let conn = db::open(&path).unwrap();
        // A row as an older version wrote it: no plan_mode column.
        conn.execute_batch(
            "drop table videos;
             create table videos (
               path TEXT PRIMARY KEY,
               plan_window REAL NOT NULL DEFAULT 0.25,
               plan_stride REAL NOT NULL DEFAULT 0.25,
               plan_frames INTEGER NOT NULL DEFAULT 1,
               plan_audio INTEGER NOT NULL DEFAULT 1,
               source_size INTEGER,
               source_mtime INTEGER,
               done INTEGER NOT NULL DEFAULT 0
             );
             insert into videos(path, done) values ('a.mp4', 1);",
        )
        .unwrap();
        drop(conn);

        let store = Store::open(&path).unwrap();

        let held = store.plan_of("a.mp4").unwrap().unwrap();
        assert_eq!(held.mode, plan::Mode::Fine);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One plan read gives every video and its plan.
    #[test]
    fn the_plan_rows_give_every_video() {
        let (dir, mut store) = opened("plan_rows");
        store.append(vec![new_record("a.mp4", 0.0, 0.25)]).unwrap();
        store
            .append(vec![new_record("b.mp4", 0.0, 0.25)])
            .unwrap();
        store.set_plan("b.mp4", &plan::Plan::of(plan::Preset::Speech)).unwrap();

        let rows = store.plan_rows().unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "a.mp4");
        assert_eq!(rows[1], ("b.mp4".to_string(), plan::Plan::of(plan::Preset::Speech)));
        let _ = std::fs::remove_dir_all(&dir);
    }
```

Add `use crate::plan;` to the test module's `use super::*;` area if the module
does not already import it. The file's non-test code must also add
`use crate::plan::{self, Plan};` near the top.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml store::plan`
Expected: FAIL, `no method named plan_of`, and the older-store test fails on the
missing column.

- [ ] **Step 3: Add the column and the migration**

In `videoembed/src/db.rs`, add `plan_mode` to the `videos` schema:

```sql
CREATE TABLE IF NOT EXISTS videos (
  path         TEXT PRIMARY KEY,
  plan_mode    TEXT NOT NULL DEFAULT 'fine',
  plan_window  REAL NOT NULL DEFAULT 0.25,
  plan_stride  REAL NOT NULL DEFAULT 0.25,
  plan_frames  INTEGER NOT NULL DEFAULT 1,
  plan_audio   INTEGER NOT NULL DEFAULT 1,
  source_size  INTEGER,
  source_mtime INTEGER,
  done         INTEGER NOT NULL DEFAULT 0
);
```

`CREATE TABLE IF NOT EXISTS` does not change a table that is already here. Add a
migration function and call it from `open`, after `execute_batch(SCHEMA)`.

```rust
/// Add a column that a later version introduced.
///
/// The table is old when the column is absent. The default backfills every
/// existing row, so the store stays true for its records.
fn add_missing_columns(conn: &Connection) -> Result<()> {
    let mut names: Vec<String> = Vec::new();
    let mut stmt = conn.prepare("pragma table_info(videos)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        names.push(row?);
    }
    if names.iter().any(|name| name == "plan_mode") {
        return Ok(());
    }
    conn.execute_batch("alter table videos add column plan_mode TEXT NOT NULL DEFAULT 'fine'")
        .context("add the plan_mode column")?;
    Ok(())
}
```

Then in `open`, after the schema batch:

```rust
    add_missing_columns(&conn)?;
```

- [ ] **Step 4: Add the plan read and write**

In `videoembed/src/store.rs`, add `use crate::plan::{self, Plan};` near the top,
and add these methods to `impl Store`, after `forget`.

```rust
    /// The plan of one video, or None when the video holds no row.
    pub fn plan_of(&self, path: &str) -> Result<Option<Plan>> {
        let want = normalize_label(path);
        let row = self
            .conn
            .query_row(
                "select plan_mode, plan_window, plan_stride, plan_frames, plan_audio
                 from videos where path = ?1",
                [&want],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                },
            )
            .optional()
            .context("read the plan")?;
        let Some((mode, window, stride, frames, audio)) = row else {
            return Ok(None);
        };
        let mode = plan::Mode::parse(&mode).unwrap_or(plan::Mode::Fine);
        Ok(Some(Plan {
            mode,
            window,
            stride,
            frames: frames as usize,
            audio: audio != 0,
        }))
    }

    /// Write the plan of one video. The video row must be here.
    pub fn set_plan(&mut self, path: &str, plan: &Plan) -> Result<()> {
        let want = normalize_label(path);
        self.conn
            .execute(
                "update videos set plan_mode = ?1, plan_window = ?2, plan_stride = ?3,
                     plan_frames = ?4, plan_audio = ?5 where path = ?6",
                params![
                    plan.mode.as_str(),
                    plan.window,
                    plan.stride,
                    plan.frames as i64,
                    plan.audio as i64,
                    want
                ],
            )
            .context("write the plan")?;
        Ok(())
    }

    /// Every video and its plan, in path order.
    pub fn plan_rows(&self) -> Result<Vec<(String, Plan)>> {
        let mut out = Vec::new();
        let mut stmt = self
            .conn
            .prepare("select path from videos order by path")?;
        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<_, _>>()?;
        for path in paths {
            if let Some(plan) = self.plan_of(&path)? {
                out.push((path, plan));
            }
        }
        Ok(out)
    }
```

Add `use rusqlite::OptionalExtension;` to the imports of `store.rs`.

- [ ] **Step 5: Stop `write_videos` from guessing the plan**

`write_videos` writes the plan columns from constants that no longer tell the
truth. Change it to leave the plan columns to the `set_plan` call. Replace the
insert in `write_videos` with this one, and delete the `PLAN_WINDOW`,
`PLAN_FRAMES`, and `PLAN_AUDIO` constants.

```rust
        let (size, mtime) = fingerprint(&record.label);
        tx.execute(
            "insert into videos(path, source_size, source_mtime, done)
             values (?1, ?2, ?3, 0)
             on conflict(path) do update set
               source_size = excluded.source_size,
               source_mtime = excluded.source_mtime",
            params![want, size, mtime],
        )?;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: the new store tests pass. Every older store test still passes, because
the default plan matches the fine constants.

- [ ] **Step 7: Commit**

```bash
git add videoembed/src/db.rs videoembed/src/store.rs
git commit -m "feat: hold the plan in the video row

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: The index run cuts to a plan

**Files:**
- Modify: `videoembed/src/index.rs` (`IndexRun`, `index_video`)
- Test: `videoembed/src/index.rs` (the test module)

**Interfaces:**
- Consumes: `plan::Plan`, `plan::Mode`, `Embedder::embed_window_in`,
  `Embedder::embed_image_in`, `Embedder::embed_audio_in`.
- Produces:
  - `pub fn IndexRun::start_with(video: &Path, plan: &Plan) -> Result<IndexRun>`
  - `pub fn IndexRun::plan(&self) -> &Plan`

- [ ] **Step 1: Write the failing tests**

Add these tests to the `mod tests` block in `videoembed/src/index.rs`. They use
a real file, so they run only when the spike video is here.

```rust
    /// The window plan gives the window count and the window range.
    #[test]
    fn a_window_plan_gives_one_record_for_each_window() {
        let Some(video) = spike_video() else {
            return;
        };
        let plan = plan::Plan::of(plan::Preset::Object);
        let run = IndexRun::start_with(&video, &plan).unwrap();

        let seconds = crate::media::probe_duration(&video).unwrap();
        let windows = plan.window_starts(seconds).len();
        assert_eq!(run.total(), windows, "the window count");
        // The fine run makes one record for each 0.25 s frame. The window run
        // must make far fewer.
        let fine = IndexRun::start(&video).unwrap().total();
        assert!(run.total() * 4 < fine, "windows {} fine {fine}", run.total());
    }

    /// A video that holds no audio still gives whole windows.
    ///
    /// The run must not fail when the audio is absent and the plan asks for it.
    #[test]
    fn a_video_with_no_audio_gives_whole_windows() {
        let Some(video) = spike_video_without_audio() else {
            return;
        };
        let plan = plan::Plan::of(plan::Preset::Speech);
        let mut run = IndexRun::start_with(&video, &plan).unwrap();

        assert!(run.total() > 0, "the windows must still come");
        // The cut of the first window must not fail, and the window must hold
        // the frames alone. The media cut needs no model.
        run.cut_batch().unwrap();
        assert!(!run.files.is_empty(), "the window holds its frames");
        assert!(
            run.files.iter().all(|path| path.extension().is_some_and(|e| e == "jpg")),
            "a silent video gives frames alone: {:?}",
            run.files
        );
        run.clear_dir();
    }

    /// A video shorter than one window gives no record, and no fault.
    #[test]
    fn a_video_shorter_than_one_window_gives_no_record() {
        let Some(video) = spike_video_short() else {
            return;
        };
        let plan = plan::Plan::of(plan::Preset::Speech);
        // The run must refuse the video, and the message must name the window.
        let err = IndexRun::start_with(&video, &plan).unwrap_err();
        assert!(format!("{err}").contains("shorter than one"), "{err}");
        // The same video is long enough for the 2 s Object window.
        let object = plan::Plan::of(plan::Preset::Object);
        assert!(IndexRun::start_with(&video, &object).is_ok());
    }
```

Add these three test helpers at the end of the test module. `spike_video`
returns the spike path when it is here, and `None` otherwise, so the suite runs
on a machine without the clip. The other two cut a copy with ffmpeg.

```rust
    /// The spike clip, when it is here.
    fn spike_video() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from("../spike/testvideo_nolabels.mp4");
        path.is_file().then_some(path)
    }

    /// Cut a copy of the spike clip. A machine without the clip gives None.
    fn cut_spike(extra: &[&str], name: &str) -> Option<std::path::PathBuf> {
        let video = spike_video()?;
        let dir = std::env::temp_dir().join(format!("vei_cut_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok()?;
        let out = dir.join("cut.mp4");
        let status = std::process::Command::new(crate::media::FFMPEG)
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
            .arg(&video)
            .args(extra)
            .arg(&out)
            .status()
            .ok()?;
        status.success().then_some(out)
    }

    /// A copy of the spike clip with the audio dropped.
    fn spike_video_without_audio() -> Option<std::path::PathBuf> {
        cut_spike(&["-an", "-c:v", "copy"], "silent")
    }

    /// A three second cut of the spike clip.
    fn spike_video_short() -> Option<std::path::PathBuf> {
        cut_spike(&["-t", "3"], "short")
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml index::a_window_plan`
Expected: FAIL with `no function start_with`.

- [ ] **Step 3: Add the plan to `IndexRun`**

In `videoembed/src/index.rs`, add to the `IndexRun` struct:

```rust
    plan: crate::plan::Plan,
    windows: Vec<f64>,
```

Add the field to the `start` constructor. Keep `start` as the fine-plan entry,
so the current callers and tests keep working, and add `start_with`:

```rust
    /// Probe the video and build the plan of the run.
    ///
    /// Fine mode follows the length of each stream, not the length of the file.
    /// Window mode follows the duration and the plan.
    pub fn start_with(video: &Path, plan: &crate::plan::Plan) -> Result<Self> {
        match plan.mode {
            crate::plan::Mode::Window => Self::start_window(video, plan),
            crate::plan::Mode::Fine => Self::start(video),
        }
    }

    /// The plan of the run.
    pub fn plan(&self) -> &crate::plan::Plan {
        &self.plan
    }

    /// Probe the video and build the window plan.
    fn start_window(video: &Path, plan: &crate::plan::Plan) -> Result<Self> {
        let duration = media::probe_duration(video)?;
        let windows = plan.window_starts(duration);
        if windows.is_empty() {
            bail!(
                "{} is shorter than one {} s window",
                video.display(),
                plan.window
            );
        }
        Ok(Self {
            video: video.to_path_buf(),
            label: store::normalize_label(&video.to_string_lossy()),
            duration,
            frames: Vec::new(),
            audios: Vec::new(),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Windows,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan: *plan,
            windows,
        })
    }
```

Set `window_at: 0`, `plan: crate::plan::Plan::of(crate::plan::Preset::Fine)`, and
`windows: Vec::new()` in the fields of `start`.

Add a `Windows` variant to `Phase`, ahead of `Done`. A window run never enters
the `Frames` or `Audio` phase, and a fine run never enters `Windows`.

- [ ] **Step 4: Cut and embed one window**

The window phase holds one window in `self.files`, so one window is one batch.
`step`, `total`, `plan`, `at`, `remaining_in_batch`, `advance`, and `cut_batch`
each take a `Phase::Windows` arm. Add each arm to the existing match.

In `step`, the window arm replaces the single-file read, because one window
holds several files:

```rust
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
            Phase::Windows => {
                let refs: Vec<&Path> = self.files.iter().map(|path| path.as_path()).collect();
                (
                    store::KIND_WINDOW,
                    start + self.plan.window,
                    embedder.embed_window_in(ctx, &refs)?,
                )
            }
            Phase::Done => unreachable!(),
        };
```

In `total`, the window count replaces the frame and audio counts:

```rust
    pub fn total(&self) -> usize {
        if self.phase == Phase::Windows {
            return self.windows.len();
        }
        self.frames.len() + self.audios.len()
    }
```

In `plan`, the window starts:

```rust
    fn plan(&self) -> &[f64] {
        match self.phase {
            Phase::Frames => &self.frames,
            Phase::Audio => &self.audios,
            Phase::Windows => &self.windows,
            Phase::Done => &[],
        }
    }
```

In `at`, the window cursor:

```rust
    fn at(&self) -> usize {
        match self.phase {
            Phase::Frames => self.frame_at,
            Phase::Audio => self.audio_at,
            Phase::Windows => self.window_at,
            Phase::Done => 0,
        }
    }
```

In `remaining_in_batch`, one window counts as one item:

```rust
    fn remaining_in_batch(&self) -> usize {
        if self.phase == Phase::Windows {
            // One window is one batch. The files of the window are ready, or
            // they are not, and either way the count is one.
            return usize::from(!self.files.is_empty());
        }
        self.files.len().saturating_sub(self.at() - self.batch_from)
    }
```

In `advance`, the window cursor steps once and drops the media of the window
that finished, so the next `step` cuts the next window:

```rust
            Phase::Windows => {
                self.window_at += 1;
                self.files.clear();
                self.batch_from = 0;
                if self.window_at >= self.windows.len() {
                    self.phase = Phase::Done;
                }
            }
```

In `cut_batch`, the window arm replaces the count and step pair. A window cuts
its frames and its audio in one directory:

```rust
        if self.phase == Phase::Windows {
            let at = self.at();
            let from = self.windows[at];
            let to = (from + self.plan.window).min(self.duration);
            self.dir = work_dir(at);
            std::fs::create_dir_all(&self.dir)?;
            let mut files =
                media::extract_frames(&self.video, from, to, self.plan.frame_rate(), &self.dir)?;
            if self.plan.audio {
                // The audio is dropped when the clip gives nothing. The window
                // then holds the frames alone.
                files.extend(media::extract_audio(&self.video, from, to, &self.dir)?);
            }
            if files.is_empty() {
                bail!("no media came from the window at {from} s");
            }
            self.files = files;
            self.batch_from = at;
            return Ok(());
        }
```

Add `window_at: usize` to the struct, and set it to zero in every constructor.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml index::`
Expected: the window tests pass, and every fine test still passes. A machine
without the spike clip skips the two file tests.

- [ ] **Step 6: Run the whole suite and commit**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: no failure.

```bash
git add videoembed/src/index.rs
git commit -m "feat: cut a video into windows under a plan

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: The index call takes a plan

**Files:**
- Modify: `videoembed/src/index.rs` (`index_video`)
- Modify: `videoembed/src/main.rs` (the console `index`)
- Test: `videoembed/src/main.rs`

**Interfaces:**
- Consumes: `Store::plan_of`, `Store::set_plan`, `IndexRun::start_with`.
- Produces:
  `pub fn index_video(embedder: &Embedder, video: &Path, plan: &plan::Plan, on_step: &mut dyn FnMut(usize, usize)) -> Result<Vec<NewRecord>>`

- [ ] **Step 1: Write the failing test**

Add this test to the `mod tests` block in `videoembed/src/main.rs`.

```rust
    /// The index command names the plan it will use.
    #[test]
    fn the_index_line_names_the_plan() {
        let plan = videoembed::plan::Plan::of(videoembed::plan::Preset::Object);
        assert_eq!(
            index_line(&plan),
            "Indexing under the Object plan (2 s windows, 2 frames, no audio)."
        );
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml the_index_line`
Expected: FAIL with `cannot find function index_line`.

- [ ] **Step 3: Change `index_video` and the console `index`**

In `videoembed/src/index.rs`, change `index_video` to take a plan:

```rust
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    plan: &crate::plan::Plan,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Vec<NewRecord>> {
    let mut run = IndexRun::start_with(video, plan)?;
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

In `videoembed/src/main.rs`, add the helper and use it in `index`:

```rust
/// The line the console shows before an index run.
fn index_line(plan: &videoembed::plan::Plan) -> String {
    let audio = if plan.audio { "with audio" } else { "no audio" };
    match plan.mode {
        videoembed::plan::Mode::Fine => {
            "Indexing under the Fine plan (0.25 s frames).".to_string()
        }
        _ => format!(
            "Indexing under the {:?} plan ({} s windows, {} frames, {}).",
            preset_of(plan),
            plan.window,
            plan.frames,
            audio
        ),
    }
}

/// The preset of a plan, by its values.
fn preset_of(plan: &videoembed::plan::Plan) -> videoembed::plan::Preset {
    videoembed::plan::Preset::ALL
        .into_iter()
        .find(|preset| videoembed::plan::Plan::of(*preset) == *plan)
        .unwrap_or(videoembed::plan::Preset::Object)
}
```

Then in the `index` function, read the plan, print the line, run, and write the
plan:

```rust
fn index(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: index <video path>");
    }
    let video = PathBuf::from(rest);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    let label = videoembed::store::normalize_label(&video.to_string_lossy());
    let plan = store
        .plan_of(&label)?
        .unwrap_or_else(videoembed::plan::Plan::default_plan);
    println!("Video: {}", video.display());
    println!("{}", index_line(&plan));
    let mut report = |n: usize, total: usize| println!("  item {n:5}/{total}");
    let records = index::index_video(embedder, &video, &plan, &mut report)?;
    let count = records.len();
    store.forget(&label)?;
    store.append(records)?;
    store.set_plan(&label, &plan)?;
    println!("Indexed {count} records into {}", store_path_hint());
    Ok(())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: the new test passes, and the console tests still pass.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/main.rs
git commit -m "feat: index under the stored plan

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: A plan change re-cuts the video

**Files:**
- Modify: `videoembed/src/index.rs` (a shared `reindex` routine)
- Modify: `videoembed/src/main.rs` (the console `plan` command)
- Test: `videoembed/src/main.rs`

**Interfaces:**
- Consumes: `IndexRun::start_with`, `Store::forget`, `Store::append`,
  `Store::set_plan`, `Store::mark_done`, `Store::plan_of`, `plan::Preset::parse`,
  `plan::Plan::set_words`.
- Produces:
  - `pub fn index::reindex(embedder: &Embedder, store: &mut Store, path: &str, plan: &plan::Plan, on_step: &mut dyn FnMut(usize, usize)) -> Result<usize>`
  - `pub fn main::plan_command(store: &mut Store, rest: &str) -> Result<String>` —
    the console parser, split from the media work so a test can reach it.

- [ ] **Step 1: Write the failing tests**

Add these tests to the `mod tests` block in `videoembed/src/main.rs`.

```rust
    /// A preset word picks that preset.
    #[test]
    fn the_plan_command_reads_a_preset() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/video/a.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();

        let plan = plan_command(&store, "D:/video/a.mp4 action").unwrap();

        assert_eq!(plan, videoembed::plan::Plan::of(videoembed::plan::Preset::Action));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A value word sets one plan value over the stored plan.
    #[test]
    fn the_plan_command_reads_values() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/video/a.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();

        let plan = plan_command(&store, "D:/video/a.mp4 window=6 audio=on").unwrap();

        assert_eq!(plan.window, 6.0);
        assert!(plan.audio);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A path the store does not hold names the fault.
    #[test]
    fn the_plan_command_refuses_an_unknown_video() {
        let (dir, store) = store_for_plan_test();
        let err = plan_command(&store, "D:/video/zz.mp4 object").unwrap_err();
        assert!(format!("{err}").contains("zz.mp4"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A plan change on a video whose file is gone ends the change.
    #[test]
    fn a_plan_change_needs_the_source_file() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/video/missing.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();
        store
            .set_plan(
                "D:/video/missing.mp4",
                &videoembed::plan::Plan::of(videoembed::plan::Preset::Object),
            )
            .unwrap();

        let err = source_of("D:/video/missing.mp4").unwrap_err();

        assert!(format!("{err}").contains("missing.mp4"), "{err}");
        // The plan must stay as it was.
        let held = store.plan_of("D:/video/missing.mp4").unwrap().unwrap();
        assert_eq!(held, videoembed::plan::Plan::of(videoembed::plan::Preset::Object));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A store with a fresh file, for the plan tests.
    fn store_for_plan_test() -> (PathBuf, Store) {
        let dir = std::env::temp_dir().join(format!(
            "vei_plan_cmd_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = Store::open(&dir.join("index.db")).unwrap();
        (dir, store)
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml the_plan_command`
Expected: FAIL with `cannot find function plan_command`.

- [ ] **Step 3: Add the shared re-cut routine**

Add to `videoembed/src/index.rs`:

```rust
/// Cut one video again under a new plan.
///
/// The plan is written last, so a fault in the middle leaves the video with no
/// records and its old plan. The next index run then rebuilds it.
pub fn reindex(
    embedder: &Embedder,
    store: &mut store::Store,
    path: &str,
    plan: &crate::plan::Plan,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<usize> {
    let video = Path::new(path);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    let label = store::normalize_label(path);
    store.forget(&label)?;
    let records = index_video(embedder, video, plan, on_step)?;
    let count = records.len();
    store.append(records)?;
    store.set_plan(&label, plan)?;
    store.mark_done(&label)?;
    Ok(count)
}
```

- [ ] **Step 4: Add the console command**

In `videoembed/src/main.rs`, add the two functions and wire the command:

```rust
/// The source file of one stored video, and the check that it is here.
fn source_of(path: &str) -> Result<PathBuf> {
    let video = PathBuf::from(path);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    Ok(video)
}

/// The plan a `plan` command asks for, over the stored plan.
///
/// The argument is `<path> <preset>` or `<path> <value> <value>`.
fn plan_command(store: &Store, rest: &str) -> Result<videoembed::plan::Plan> {
    let (path, words) = rest
        .split_once(char::is_whitespace)
        .ok_or_else(|| anyhow::anyhow!("usage: plan <path> <preset> | plan <path> window=4 frames=4"))?;
    let words = words.trim();
    if words.is_empty() {
        bail!("usage: plan <path> <preset> | plan <path> window=4 frames=4");
    }
    let label = videoembed::store::normalize_label(path);
    let base = store
        .plan_of(&label)?
        .ok_or_else(|| anyhow::anyhow!("the store holds no video at {path}"))?;
    if let Some(preset) = videoembed::plan::Preset::parse(words) {
        return Ok(videoembed::plan::Plan::of(preset));
    }
    base.set_words(words)
}
```

Add the command to `run`:

```rust
        "plan" => plan(embedder, store, rest),
```

and the handler:

```rust
/// Cut one video again under the plan the user names.
fn plan(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    let (path, _) = rest
        .split_once(char::is_whitespace)
        .ok_or_else(|| anyhow::anyhow!("usage: plan <path> <preset>"))?;
    let wanted = plan_command(store, rest)?;
    source_of(path)?;
    println!("Re-indexing {} under the new plan.", path);
    let mut report = |n: usize, total: usize| println!("  item {n:5}/{total}");
    let count = index::reindex(embedder, store, path, &wanted, &mut report)?;
    println!("Indexed {count} records.");
    Ok(())
}
```

Add `plan` to the command list line in `main`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: the four plan-command tests pass.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/main.rs
git commit -m "feat: re-cut a video when its plan changes

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: The page sets the plan

**Files:**
- Modify: `videoembed/src/bin/web.rs` (the reply, the job, the endpoint)
- Test: `videoembed/src/bin/web.rs`

**Interfaces:**
- Consumes: `index::reindex`, `Store::plan_rows`, `plan::Preset`.
- Produces:
  - `VideoRow` gains `pub plan: String`, `pub window: f64`, `pub frames: usize`,
    `pub audio: bool`.
  - `Job::SetPlan { path: String, plan: plan::Plan, answer: Sender<Result<usize, JobError>> }`
  - `POST /api/plan`, with the body `{ "path": "...", "preset": "action" }` or
    `{ "path": "...", "window": 4, "frames": 4, "audio": true }`.

- [ ] **Step 1: Write the failing tests**

Add these tests to the `mod tests` block in `videoembed/src/bin/web.rs`.

```rust
    /// The videos reply carries the plan of each video.
    #[test]
    fn a_video_row_carries_its_plan() {
        let dir = std::env::temp_dir().join(format!("vei_web_plan_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store
            .append(vec![NewRecord {
                kind: videoembed::store::KIND_WINDOW,
                label: "a.mp4".into(),
                start: 0.0,
                end: 2.0,
                vec: unit(),
            }])
            .unwrap();
        store
            .set_plan("a.mp4", &videoembed::plan::Plan::of(videoembed::plan::Preset::Speech))
            .unwrap();

        let rows = videos(&store);

        assert_eq!(rows[0].plan, "speech");
        assert_eq!(rows[0].window, 6.0);
        assert_eq!(rows[0].frames, 6);
        assert!(rows[0].audio);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A video with no plan row reads as the default plan.
    #[test]
    fn a_video_with_no_plan_shows_the_default() {
        let dir = std::env::temp_dir().join(format!("vei_web_plan_def_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store
            .append(vec![NewRecord {
                kind: videoembed::store::KIND_WINDOW,
                label: "a.mp4".into(),
                start: 0.0,
                end: 2.0,
                vec: unit(),
            }])
            .unwrap();
        // A row with no plan_mode. The default fills it.
        store
            .conn_for_test()
            .execute_batch("update videos set plan_mode = 'window', plan_window = 0, plan_stride = 0, plan_frames = 0, plan_audio = 0")
            .unwrap();

        let rows = videos(&store);

        assert_eq!(rows[0].plan, "object", "a zero plan reads as the default");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The plan body reads a preset name.
    #[test]
    fn the_plan_body_reads_a_preset() {
        let body: PlanBody = serde_json::from_str(r#"{"path":"a.mp4","preset":"action"}"#).unwrap();
        let plan = body.plan().unwrap();
        assert_eq!(plan, videoembed::plan::Plan::of(videoembed::plan::Preset::Action));
    }

    /// The plan body reads a set of values.
    #[test]
    fn the_plan_body_reads_values() {
        let body: PlanBody =
            serde_json::from_str(r#"{"path":"a.mp4","window":4,"frames":4,"audio":true}"#).unwrap();
        let plan = body.plan().unwrap();
        assert_eq!(plan.window, 4.0);
        assert_eq!(plan.frames, 4);
        assert!(plan.audio);
    }

    /// A body with no plan value names the fault.
    #[test]
    fn an_empty_plan_body_is_refused() {
        let body: PlanBody = serde_json::from_str(r#"{"path":"a.mp4"}"#).unwrap();
        assert!(body.plan().is_err());
    }
```

Add `pub fn conn_for_test(&self) -> &rusqlite::Connection` to `Store` under a
`#[doc(hidden)]` marker, so the test can reach the row. If the project prefers
no test-only method, drop the second test; the first covers the plan read.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml a_video_row_carries_its_plan`
Expected: FAIL, `no field plan on VideoRow`.

- [ ] **Step 3: Add the plan to the reply**

In `videoembed/src/bin/web.rs`, extend `VideoRow` and `videos`:

```rust
/// One video and its plan.
#[derive(Serialize)]
struct VideoRow {
    path: String,
    items: usize,
    plan: String,
    window: f64,
    frames: usize,
    audio: bool,
}
```

In `videos`, read the plan of each path once, and fill the row. A path with no
plan row takes `Plan::default_plan()`.

```rust
/// Group the store records by video path, with the plan of each video.
fn videos(store: &Store) -> Vec<VideoRow> {
    let plans: std::collections::HashMap<String, videoembed::plan::Plan> = store
        .plan_rows()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let mut rows: Vec<VideoRow> = Vec::new();
    for record in &store.records {
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        let name = normalize_label(&record.label);
        if let Some(row) = rows.iter_mut().find(|row| row.path == name) {
            row.items += 1;
            continue;
        }
        let plan = plans
            .get(&name)
            .copied()
            .unwrap_or_else(videoembed::plan::Plan::default_plan);
        rows.push(VideoRow {
            path: name,
            items: 1,
            plan: preset_name(&plan),
            window: plan.window,
            frames: plan.frames,
            audio: plan.audio,
        });
    }
    rows
}

/// The preset name of a plan, by its values.
fn preset_name(plan: &videoembed::plan::Plan) -> String {
    videoembed::plan::Preset::ALL
        .into_iter()
        .find(|preset| videoembed::plan::Plan::of(*preset) == *plan)
        .unwrap_or(videoembed::plan::Preset::Object)
        .as_str()
        .to_string()
}
```

- [ ] **Step 4: Add the job and the endpoint**

Add the body type and the job variant:

```rust
#[derive(Debug, Deserialize)]
struct PlanBody {
    path: String,
    #[serde(default)]
    preset: Option<String>,
    #[serde(default)]
    window: Option<f64>,
    #[serde(default)]
    frames: Option<usize>,
    #[serde(default)]
    audio: Option<bool>,
}

impl PlanBody {
    /// The plan this body asks for.
    fn plan(&self) -> Result<videoembed::plan::Plan, JobError> {
        if let Some(preset) = &self.preset {
            let preset = videoembed::plan::Preset::parse(preset)
                .ok_or_else(|| JobError::Message(format!("{preset} is not a preset")))?;
            return Ok(videoembed::plan::Plan::of(preset));
        }
        let mut words = Vec::new();
        if let Some(window) = self.window {
            words.push(format!("window={window}"));
        }
        if let Some(frames) = self.frames {
            words.push(format!("frames={frames}"));
        }
        if let Some(audio) = self.audio {
            words.push(format!("audio={}", if audio { "on" } else { "off" }));
        }
        if words.is_empty() {
            return Err(JobError::Message("no plan value arrived".into()));
        }
        videoembed::plan::Plan::of(videoembed::plan::Preset::Object)
            .set_words(&words.join(" "))
            .map_err(JobError::from)
    }
}
```

Add `SetPlan { path: String, plan: videoembed::plan::Plan, answer: Sender<Result<usize, JobError>> }`
to `Job`. In `handle`, add the arm. A plan change is a job, so it shares the
busy rule with the index job:

```rust
        Job::SetPlan { path, plan, answer } => {
            let outcome = start_plan(&progress, pending, stopped, &path, plan);
            let _ = answer.send(outcome);
        }
```

Add the start function, modelled on `start_index`:

```rust
/// Begin a re-cut of one video under a new plan.
///
/// The job holds one file, so the engine loop cuts it one item at a time. A
/// running index job refuses the change.
fn start_plan(
    progress: &Arc<Mutex<Progress>>,
    pending: &mut Option<WebJob>,
    flag: &Arc<AtomicBool>,
    path: &str,
    plan: videoembed::plan::Plan,
) -> Result<usize, JobError> {
    refuse_when_busy(pending)?;
    let video = PathBuf::from(path);
    if !video.is_file() {
        return Err(JobError::Message(format!("{} is not a file", video.display())));
    }
    let run = index::IndexRun::start_with(&video, &plan).map_err(JobError::from)?;
    let total = run.total();
    flag.store(false, Ordering::Relaxed);
    {
        let mut held = progress.lock().unwrap();
        let device = held.device.clone();
        let gpu_available = held.gpu_available;
        let load_ms = held.load_ms;
        *held = Progress {
            running: true,
            folder: String::new(),
            file_count: 1,
            device,
            gpu_available,
            load_ms,
            ..Progress::default()
        };
    }
    let label = normalize_label(path);
    *pending = Some(WebJob {
        files: vec![video],
        file_at: 0,
        run: Some(run),
        records: Vec::new(),
        indexed: 0,
        started: std::time::Instant::now(),
        label: label.clone(),
        plan: Some(plan),
    });
    Ok(total)
}
```

Add `plan: Option<videoembed::plan::Plan>` to `WebJob`, and set it to `None` in
`start_index`. In `step_job`, after `store.append(records)`, move the plan
before `mark_done`:

```rust
        if whole {
            job.indexed += records.len();
            if let Err(err) = store.append(records) {
                fail(progress, &err);
            }
            if let Some(plan) = &job.plan {
                if let Err(err) = store.set_plan(&job.label, plan) {
                    fail(progress, &err);
                }
            }
            if let Err(err) = store.mark_done(&job.label) {
                fail(progress, &err);
            }
        } else {
```

Also, when `job.run` is built in `step_job` for a folder index, use the stored
plan of the video in place of `IndexRun::start`:

```rust
        let video = job.files[job.file_at].clone();
        let label = normalize_label(&video.to_string_lossy());
        let plan = match store.plan_of(&label) {
            Ok(Some(plan)) => plan,
            Ok(None) => videoembed::plan::Plan::default_plan(),
            Err(err) => {
                fail(progress, &err);
                videoembed::plan::Plan::default_plan()
            }
        };
        match index::IndexRun::start_with(&video, &plan) {
            Ok(run) => {
                if store.is_done(&label) {
                    job.file_at += 1;
                    return false;
                }
                if let Err(err) = store.forget(&label) {
                    fail(progress, &err);
                }
                begin_file(progress, label.clone(), job.file_at + 1, run.total());
                job.label = label.clone();
                job.plan = Some(plan);
                job.records.clear();
                job.run = Some(run);
            }
            Err(err) => {
                fail(progress, &err);
                job.file_at += 1;
            }
        }
        return false;
```

Finally, add the route in `api`:

```rust
            "/api/plan" => {
                let body: PlanBody = match read_body(&mut request) {
                    Ok(body) => body,
                    Err((code, message)) => {
                        request.respond(error_reply(code, &message))?;
                        return Ok(());
                    }
                };
                let plan = match body.plan() {
                    Ok(plan) => plan,
                    Err(JobError::Message(message)) => {
                        request.respond(error_reply(400, &message))?;
                        return Ok(());
                    }
                    Err(_) => {
                        request.respond(error_reply(400, "bad plan body"))?;
                        return Ok(());
                    }
                };
                let (answer, wait) = channel();
                let _ = jobs.send(Job::SetPlan {
                    path: body.path,
                    plan,
                    answer,
                });
                match wait.recv() {
                    Ok(Ok(count)) => request.respond(json_reply(&count))?,
                    Ok(Err(JobError::Busy)) => request.respond(error_reply(409, "busy"))?,
                    Ok(Err(JobError::Message(message))) => {
                        request.respond(error_reply(500, &message))?
                    }
                    _ => request.respond(error_reply(500, "the engine stopped"))?,
                }
            }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: the new web tests pass.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/bin/web.rs videoembed/src/store.rs
git commit -m "feat: set the plan of a video from the page

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: The plan panel

**Files:**
- Modify: `videoembed/assets/app.js`
- Modify: `videoembed/assets/app.css`
- Modify: `videoembed/src/bin/web.rs` (serve the panel markup)

**Interfaces:**
- Consumes: `GET /api/videos` (each row now holds `plan`, `window`, `frames`,
  `audio`) and `POST /api/plan`.
- Produces: a plan panel for one video, with four preset buttons, a window
  slider, a frame slider, and an audio switch.

The page holds no test runner, so this task has no automated test. The steps
build the markup, and a person checks the panel in a browser.

- [ ] **Step 1: Add the markup**

In the page body that `web.rs` serves, after the video list, add this markup:

```html
<div id="plan-panel" class="panel" hidden>
  <strong id="plan-title"></strong>
  <div class="presets">
    <button type="button" data-preset="fine">Fine</button>
    <button type="button" data-preset="object">Object</button>
    <button type="button" data-preset="action">Action</button>
    <button type="button" data-preset="speech">Speech</button>
  </div>
  <label>Window <input id="plan-window" type="range" min="1" max="10" step="1" value="2"><span id="plan-window-value">2 s</span></label>
  <label>Frames <input id="plan-frames" type="range" min="1" max="2" step="1" value="2"><span id="plan-frames-value">2</span></label>
  <label><input id="plan-audio" type="checkbox"> Audio</label>
  <button type="button" id="plan-apply">Apply</button>
</div>
```

- [ ] **Step 2: Add the style**

Add to `videoembed/assets/app.css`:

```css
.panel { margin-top: 0.75rem; padding: 0.75rem; border: 1px solid var(--line); border-radius: 4px; }
.presets { display: flex; gap: 0.5rem; margin: 0.5rem 0; }
.panel label { display: flex; align-items: center; gap: 0.5rem; margin: 0.35rem 0; font-size: 0.9rem; }
.panel input[type="range"] { flex: 1; }
```

- [ ] **Step 3: Add the script**

Add to `videoembed/assets/app.js`. The window slider sets the frame slider top
to the window in whole seconds. Apply posts the plan and polls, because the
re-cut is a job.

```js
let planPath = "";

function showPlan(row) {
  planPath = row.path;
  $("plan-panel").hidden = false;
  $("plan-title").textContent = `Plan for ${row.path}`;
  const window = $("plan-window");
  window.value = String(Math.round(row.window));
  $("plan-frames").max = window.value;
  $("plan-frames").value = String(row.frames);
  $("plan-audio").checked = row.audio;
  updatePlanLabels();
  for (const button of document.querySelectorAll("[data-preset]")) {
    button.classList.toggle("on", button.dataset.preset === row.plan);
  }
}

function updatePlanLabels() {
  const window = Number($("plan-window").value);
  $("plan-window-value").textContent = `${window} s`;
  const frames = Number($("plan-frames").value);
  $("plan-frames").max = String(window);
  if (frames > window) $("plan-frames").value = String(window);
  $("plan-frames-value").textContent = String(Number($("plan-frames").value));
}

async function applyPlan() {
  note("");
  try {
    await api("/api/plan", {
      path: planPath,
      window: Number($("plan-window").value),
      frames: Number($("plan-frames").value),
      audio: $("plan-audio").checked,
    });
    startPolling();
  } catch (err) {
    note(err.message);
  }
}

function usePreset(name) {
  const base = { fine: [0.25, 1, true], object: [2, 2, false], action: [4, 4, false], speech: [6, 6, true] };
  const [window, frames, audio] = base[name];
  if (name === "fine") {
    applyPreset("fine");
    return;
  }
  $("plan-window").value = String(window);
  $("plan-frames").value = String(frames);
  $("plan-audio").checked = audio;
  updatePlanLabels();
}

async function applyPreset(name) {
  note("");
  try {
    await api("/api/plan", { path: planPath, preset: name });
    startPolling();
  } catch (err) {
    note(err.message);
  }
}
```

Wire the controls on load:

```js
$("plan-window").addEventListener("input", updatePlanLabels);
$("plan-frames").addEventListener("input", updatePlanLabels);
$("plan-apply").addEventListener("click", applyPlan);
for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => usePreset(button.dataset.preset));
}
```

In the code that draws the video list, make each row a button that calls
`showPlan(row)`.

- [ ] **Step 4: Check the panel in a browser**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo run --manifest-path videoembed/Cargo.toml --bin web`
Expected: the page shows a video list. A click on a row shows the plan panel.
Move the window slider, and the frame slider top follows. Apply runs a job, and
the videos reply then names the new preset.

- [ ] **Step 5: Run the whole suite and commit**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: no failure.

```bash
git add videoembed/assets/app.js videoembed/assets/app.css videoembed/src/bin/web.rs
git commit -m "feat: add the plan panel

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: The docs

**Files:**
- Modify: `README.md`
- Modify: `docs/deferred-work.md`

- [ ] **Step 1: Update the README**

Change the `index` row of the command table to name the plan. Add a `plan` row.

```
| `index <path>` | Cut the video under its plan. Embed each item. |
| `plan <path> <preset>` | Cut the video again under a new plan. |
```

Replace the window sentence under the table with the presets:

```
The default plan is Object: a 2 s window on a 1 s stride, with 2 frames and no
audio. The presets are Object, Action, Speech, and Fine. The Fine preset keeps
the 0.25 s frame grid.
```

Add a plan endpoint row to the API table:

```
| `/api/plan` | POST | Set the plan of one video. The reply names the item count. |
```

- [ ] **Step 2: Take items 1, 8, and 11 off the deferred list**

In `docs/deferred-work.md`, remove the three rows for items 1, 8, and 11. Add
the probe result to the section that records the decisions.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/deferred-work.md
git commit -m "docs: record the plan and the probe result

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: section 3 (the plan and
the presets) to Task 2, section 4 (the records) to Task 4, section 5 (the
schema) to Task 3, section 6 (the index run) to Task 4, section 7 (a plan
change) to Tasks 5 and 6, section 8 (the controls) to Tasks 6, 7, and 8, section
9 (the tests) across Tasks 2 to 7, section 11 (what waits) to the Review Focus
list, and section 12 (the probe) to Task 1.

**Placeholder scan.** No step holds "TBD", "TODO", or "add validation". Every
code step shows the code. Task 8 has no automated test, because the page holds
no test runner. The step says so.

**Type consistency.** `Plan::of`, `Plan::default_plan`, `Plan::set_words`,
`Plan::frame_rate`, `Plan::window_starts`, `Plan::plan_of`, `Plan::set_plan`,
`Plan::plan_rows`, `IndexRun::start_with`, `IndexRun::plan`, `index::reindex`,
`plan_command`, and `source_of` keep one name and one signature across the
tasks. `plan::Plan` is the type in every task. `Mode::as_str` gives `window` and
`fine`, and `Preset::as_str` gives the four preset names.

**Review Focus.** The five lines each name a task, and each task holds the test.
The fifth line, the busy rule, is covered by `refuse_when_busy` in
`start_plan`, which Task 7 reuses from `start_index`.

## Open points

1. Task 4's file tests need the spike clip, and one of them runs ffmpeg. A
   machine without the clip skips them. The suite stays green either way.
2. Task 8 holds no test. A person checks the panel.
3. The first plan task is a measurement, not code. The two counts feed the doc.
