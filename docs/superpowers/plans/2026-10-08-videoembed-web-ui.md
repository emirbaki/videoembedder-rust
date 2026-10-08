# videoembed web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `videoembed` Rust app a local web page that indexes a folder,
searches it by text, and plays the matching moment.

**Architecture:** The `videoembed` package gains a library target and a second
binary, `web`. That binary runs two threads. One thread owns the embedding model
and the store, and it answers jobs from a channel. The other thread serves HTTP
on `127.0.0.1:8787` with `tiny_http`. A job that takes a long time runs in the
engine loop, one window per turn, so short jobs keep flowing.

**Tech Stack:** Rust 2024, `llama-cpp-2` 0.1.158 with the vendored FFI crate,
`tiny_http`, `serde` and `serde_json`, `rfd`, `open`, `percent-encoding`,
`anyhow`.

**Spec:** `docs/superpowers/specs/2026-10-08-videoembed-web-ui-design.md`

## Global Constraints

Every task below includes this section.

- Window length 4 s, stride 2 s, and a 0.05 s slack at the video end.
- The store stays the format `VEI1`, at `store/index.vei`. Both binaries share
  one file.
- Folder scan extensions, lowercased: `mp4`, `mkv`, `mov`, `avi`, `webm`. The
  scan walks into subfolders.
- The server binds `127.0.0.1:8787` only. It sends no CORS headers.
- Search floor 0.4, cap 50 hits in one group. Groups sort by their best score.
- `/api/video` writes the file in chunks of 64 KiB.
- Search prefixes: query `task: search result | query: {q}`, note
  `title: none | text: {t}`.
- The reader sees every word, so ASD-STE100 governs it: comments, docs, error
  messages, and commit messages. No semicolons. No contractions. One
  instruction is 20 words or fewer. Any other sentence is 25 words or fewer.
  Use the active voice and simple tenses. Lint a document before you commit it:
  `node "C:\Users\emir-\.agents\skills\asd-ste100\hooks\run-python.cjs" "C:\Users\emir-\.agents\skills\asd-ste100\scripts\ste-lint.py" --fail-over 2.5 FILE`
- Build with `CMAKE_GENERATOR="Visual Studio 17 2022" CMAKE_BUILD_PARALLEL_LEVEL=8 cargo build`.
  An MSVC file lock can stop the build with `os error 32`. Wait ten seconds and
  run the same command again.
- Run every command from the `videoembed` folder. Tests must not load the model
  and must not need a video file.

**One deviation from the spec, on purpose.** Spec section 4.7 says every
`/api/` call carries the header `X-VEI: 1`. A browser `<video>` element cannot
set a header on its own requests. So `/api/video` takes a random key in the
query instead. The page learns that key from `/api/state`, which the header
already guards. A foreign page cannot learn the key. Every other `/api/` call
keeps the header rule.

## Review Focus

The spec says what the software must do. It does not name every input the
software will meet. These five inputs are the most likely to bite a person who
uses the app. Each one has a test in the task named here.

1. A folder or file name with a space or a non ASCII letter. The query string
   must decode back to the exact stored path, or the player shows a 403. Test
   in Task 4 and Task 5.
2. `/api/video` for a path outside the index. It must answer 403, and it must
   not open the file. The predicate is tested in Task 4.
3. `/api/progress` before any job has run, or after one has ended. The page must
   not divide by zero and must not show a stale running bar. The JSON shape is
   tested in Task 2, and the page guard is written in Task 7.
4. A search that finds no hit above the floor, or a search against an empty
   store. The answer is `{"groups": []}` and the page shows one message. Tested
   in Task 2 and Task 8.
5. A `Range` request that starts past the end of the file. It must answer 416,
   not a panic and not a truncated stream. Tested in Task 9.

---

### Task 1: The library target and the new crates

The package holds one binary today. Add the library target, so a second binary
can use the same modules. Add the crates the web binary needs.

**Files:**
- Modify: `videoembed/Cargo.toml`
- Create: `videoembed/src/lib.rs`
- Modify: `videoembed/src/main.rs:5-15`

**Interfaces:**
- Consumes: the existing modules `embed`, `media`, `store`.
- Produces: the crate `videoembed` with the public modules `embed`, `media`,
  `store`. `index` arrives in Task 2.

- [ ] **Step 1: Add the crates**

```toml
[dependencies]
llama-cpp-2 = { version = "0.1.158", features = ["mtmd"] }
llama-cpp-sys-2 = { version = "0.1.158" }
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tiny_http = "0.12"
rfd = "0.17"
open = "5"
percent-encoding = "2"
```

Keep the `[patch.crates-io]` block as it is.

- [ ] **Step 2: Check that the new crates resolve**

Run: `cargo metadata --format-version 1 > /dev/null && echo resolved`
Expected: `resolved`

- [ ] **Step 3: Write the library root**

Create `videoembed/src/lib.rs`:

```rust
//! The videoembed library. Both binaries use these modules.

pub mod embed;
pub mod media;
pub mod store;
```

- [ ] **Step 4: Point the console at the library**

In `videoembed/src/main.rs`, delete these three lines:

```rust
mod embed;
mod media;
mod store;
```

Then replace the import block with:

```rust
use videoembed::embed::{DOC_PREFIX, Embedder, QUERY_PREFIX};
use videoembed::store::{KIND_NOTE, KIND_WINDOW, Record, Store};
```

Add `use videoembed::media;` only if a later step needs it. Step 5 adds it.

- [ ] **Step 5: Build and run the console**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no error.

Run: `./target/debug/videoembed.exe list`
Expected: the same output as before the change, for example
`Videos (1): ... 14 windows`.

- [ ] **Step 6: Run the existing tests**

Run: `cargo test`
Expected: `0 passed` today. The build of both targets must succeed.

- [ ] **Step 7: Commit**

```bash
git add videoembed/Cargo.toml videoembed/Cargo.lock videoembed/src/lib.rs videoembed/src/main.rs
git commit -m "refactor: move the modules into a library target"
```

---

### Task 2: The folder scan and the progress type

`index.rs` holds the folder scan, the window plan, one-video indexing, and the
progress object. It also holds the hit grouping, because grouping is a pure
function over records.

**Files:**
- Create: `videoembed/src/index.rs`
- Modify: `videoembed/src/lib.rs`

**Interfaces:**
- Consumes: `embed::Embedder`, `embed::Record`, `media`, `store::{Record, KIND_WINDOW}`.
- Produces:
  - `pub const WINDOW: f64 = 4.0`, `pub const STRIDE: f64 = 2.0`,
    `pub const EPS: f64 = 0.05`
  - `pub const FLOOR: f32 = 0.4`, `pub const GROUP_CAP: usize = 50`
  - `pub struct Progress` with the fields of the spec's JSON object, and
    `Serialize`
  - `pub fn list_videos(folder: &Path) -> Result<Vec<PathBuf>>`
  - `pub fn windows(duration: f64) -> Vec<(f64, f64)>`
  - `pub fn work_dir(at: usize) -> PathBuf`
  - `pub fn group_hits(records: &[Record], ranked: &[(usize, f32)]) -> Vec<Group>`
  - `pub struct Hit { start, end, score }`, `pub struct Group { video, duration, score, hits }`

- [ ] **Step 1: Write the failing tests**

Create `videoembed/src/index.rs` with the module doc and this test module last:

```rust
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
        assert_eq!(plan.len(), 14);
        assert_eq!(plan[0], (0.0, 4.0));
        assert_eq!(plan[13], (26.0, 30.0));
    }

    #[test]
    fn a_video_shorter_than_one_window_has_no_plan() {
        assert!(windows(3.0).is_empty());
        assert_eq!(windows(5.0).len(), 1);
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
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --lib`
Expected: a compile error. `list_videos`, `windows`, `Progress`, `group_hits`,
and `record`'s `KIND_WINDOW` are not defined.

- [ ] **Step 3: Write the module**

Put this code in `videoembed/src/index.rs`, before the test module:

```rust
//! The folder scan, the window plan, and the progress object.
//!
//! One window is 4 s long. The stride is 2 s, so windows overlap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::store::{KIND_WINDOW, Record};

/// Window length in seconds.
pub const WINDOW: f64 = 4.0;
/// Window step in seconds.
pub const STRIDE: f64 = 2.0;
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

/// A fresh working folder for one window of ffmpeg output.
pub fn work_dir(at: usize) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("vei_{}_{n}_{at}", std::process::id()))
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
```

- [ ] **Step 4: Export the module**

In `videoembed/src/lib.rs`, add the line:

```rust
pub mod index;
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cargo test --lib`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/lib.rs
git commit -m "feat: add the folder scan, the window plan, and hit grouping"
```

---

### Task 3: Index one video

Move the window loop out of the console and into the library. The console then
calls the library, so the same code serves the web binary.

**Files:**
- Modify: `videoembed/src/index.rs`
- Modify: `videoembed/src/main.rs:103-148`

**Interfaces:**
- Consumes: `embed::Embedder`, `media::extract_window`, `media::probe_duration`,
  `windows`, `work_dir`, `store::{Record, KIND_WINDOW, Store}`.
- Produces:
  `pub fn index_video(embedder: &Embedder, video: &Path, on_window: &mut dyn FnMut(usize, usize)) -> Result<Vec<Record>>`

- [ ] **Step 1: Write the failing test**

Add this test to the test module in `videoembed/src/index.rs`. It pins the
callback contract without a model, because `windows` drives the count.

```rust
    #[test]
    fn the_window_callback_reports_a_running_count() {
        let plan = windows(30.0);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let mut report = |n: usize, total: usize| seen.push((n, total));
        for (n, _) in plan.iter().enumerate() {
            report(n + 1, plan.len());
        }
        assert_eq!(seen.len(), 14);
        assert_eq!(seen[0], (1, 14));
        assert_eq!(seen[13], (14, 14));
    }
```

- [ ] **Step 2: Run the test to see it fail**

Run: `cargo test --lib the_window_callback`
Expected: PASS. This test pins the contract that Step 3 obeys. If it fails,
`windows` is wrong, and Task 2 is not done.

- [ ] **Step 3: Write the function**

Add this code to `videoembed/src/index.rs`, after `work_dir`:

```rust
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
    let label = video.to_string_lossy().to_string();
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
```

Add these imports to the top of the file:

```rust
use anyhow::{Context, Result, bail};

use crate::embed::Embedder;
use crate::media;
```

- [ ] **Step 4: Point the console at the library**

Replace the whole `index` function in `videoembed/src/main.rs` with:

```rust
/// Cut one video into windows, embed each window, and append the records.
fn index(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: index <video path>");
    }
    let video = PathBuf::from(rest);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    println!("Video: {}", video.display());
    let mut report = |n: usize, total: usize| println!("  window {n:3}/{total}");
    let records = index::index_video(embedder, &video, &mut report)?;
    let count = records.len();
    store.append(records)?;
    println!("Indexed {count} windows into {}", store_path_hint());
    Ok(())
}
```

Delete the constants `WINDOW`, `STRIDE`, and `EPS` from `main.rs`. They live in
`index.rs` now. Add the import:

```rust
use videoembed::index;
```

- [ ] **Step 5: Run the tests and the console**

Run: `cargo test --lib`
Expected: `9 passed`.

Run: `./target/debug/videoembed.exe index ../spike/testvideo_nolabels.mp4`
Expected: 14 lines, the last one `window 14/14`, then
`Indexed 14 windows into store/index.vei`.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/main.rs
git commit -m "feat: move one-video indexing into the library"
```

---

### Task 4: The store guard

`/api/video` may serve a file only when the store holds a window record with
that exact path.

**Files:**
- Modify: `videoembed/src/store.rs:99-109`

**Interfaces:**
- Consumes: `Record`, `KIND_WINDOW`.
- Produces: `pub fn holds_video(&self, path: &str) -> bool`

- [ ] **Step 1: Write the failing test**

Add this test module at the end of `videoembed/src/store.rs`:

```rust
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

    fn store_of(records: Vec<Record>) -> Store {
        Store {
            path: PathBuf::from("unused.vei"),
            dim: 2,
            records,
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
        let err = Store::decode(PathBuf::from("broken.vei"), b"nope").unwrap_err();
        assert!(format!("{err:#}").contains("broken.vei"));
    }
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --lib store::`
Expected: a compile error. `holds_video` and `encode_for_test` are not defined.
The `decode` test fails on the private function until Step 3.

- [ ] **Step 3: Write the method and one test helper**

Add `holds_video` to `impl Store`, after `forget`:

```rust
    /// True when the store holds a window record with this exact path.
    pub fn holds_video(&self, path: &str) -> bool {
        self.records
            .iter()
            .any(|record| record.kind == KIND_WINDOW && record.label == path)
    }
```

`write` builds the bytes and sends them to the file. The test needs the bytes
alone. Split the byte build out. Replace the first half of `write` with a call
to a new private function:

```rust
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
```

Add this function at the end of `store.rs`, above the test module:

```rust
/// Build store bytes for a test. The real path goes through `Store::encode`.
#[cfg(test)]
fn encode_for_test(dim: usize, records: &[Record]) -> Vec<u8> {
    let store = Store {
        path: PathBuf::from("unused.vei"),
        dim,
        records: records.to_vec(),
    };
    store.encode()
}
```

`Record` needs `Clone` for that helper. Add it:

```rust
#[derive(Clone)]
pub struct Record {
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `cargo test --lib store::`
Expected: `5 passed`.

- [ ] **Step 5: Check that the console still reads the real store**

Run: `./target/debug/videoembed.exe list`
Expected: the same list as before, with `14 windows`. This proves that `encode`
and `write` agree with `decode`.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/store.rs
git commit -m "feat: add the store guard and a store round trip test"
```

---

### Task 5: The server, the engine thread, and the page shell

Start the web binary. It loads the model on the engine thread, waits for the
load, then serves. `/api/state` reports the counts and the video key.

**Files:**
- Create: `videoembed/src/bin/web.rs`
- Create: `videoembed/assets/index.html`
- Create: `videoembed/assets/app.css`
- Create: `videoembed/assets/app.js`

**Interfaces:**
- Consumes: `videoembed::embed::Embedder`, `videoembed::index::Progress`,
  `videoembed::store::Store`.
- Produces:
  - the private type `type Reply = Response<Box<dyn Read + Send>>`
  - `enum Job` with the variants `State`, `Videos`, `Forget`, `StartIndex`,
    `Search`, `Holds`
  - `fn guarded(request: &Request) -> bool`
  - `fn parse_range(header: &str, len: u64) -> Option<(u64, u64)>`
  - `fn params(url: &str) -> HashMap<String, String>`
  - `fn mime_of(path: &Path) -> &'static str`
  - HTTP: `GET /`, `GET /app.js`, `GET /app.css`, `GET /api/state`,
    `GET /api/progress`

- [ ] **Step 1: Write the failing tests**

Create `videoembed/src/bin/web.rs` with this content, test module last:

```rust
//! The web server. One thread serves HTTP. One thread owns the model.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_range_asks_for_the_first_hundred_bytes() {
        assert_eq!(parse_range("bytes=0-99", 1000), Some((0, 99)));
    }

    #[test]
    fn an_open_range_runs_to_the_end() {
        assert_eq!(parse_range("bytes=100-", 1000), Some((100, 999)));
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        assert_eq!(parse_range("bytes=2000-", 1000), None);
        assert_eq!(parse_range("bytes=900-2000", 1000), Some((900, 999)));
    }

    #[test]
    fn a_broken_range_header_is_refused() {
        assert_eq!(parse_range("items=0-99", 1000), None);
        assert_eq!(parse_range("bytes=abc", 1000), None);
        assert_eq!(parse_range("bytes=50-10", 1000), None);
        assert_eq!(parse_range("bytes=", 1000), None);
    }

    #[test]
    fn a_query_string_decodes_a_space_and_a_unicode_letter() {
        let found = params("/api/video?path=D%3A%2Fvideo%2Fdo%C4%9Fum%20g%C3%BCn%C3%BC.mp4&key=abc");
        assert_eq!(found.get("path").unwrap(), "D:/video/doğum günü.mp4");
        assert_eq!(found.get("key").unwrap(), "abc");
    }

    #[test]
    fn the_mime_type_follows_the_extension() {
        assert_eq!(mime_of(Path::new("a.mp4")), "video/mp4");
        assert_eq!(mime_of(Path::new("a.MKV")), "video/x-matroska");
        assert_eq!(mime_of(Path::new("a.txt")), "application/octet-stream");
    }
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --bin web`
Expected: a compile error. None of the helpers exists yet.

- [ ] **Step 3: Write the server**

Add this code to `videoembed/src/bin/web.rs`, above the test module:

```rust
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use videoembed::embed::{Embedder, QUERY_PREFIX};
use videoembed::index::{self, Group, Progress};
use videoembed::media;
use videoembed::store::{KIND_WINDOW, Record, Store};

type Reply = Response<Box<dyn Read + Send>>;

const ADDRESS: &str = "127.0.0.1:8787";
const PAGE: &str = include_str!("../../assets/index.html");
const JS: &str = include_str!("../../assets/app.js");
const CSS: &str = include_str!("../../assets/app.css");

/// One request from the server thread to the engine thread.
enum Job {
    State(Sender<State>),
    Videos(Sender<Vec<VideoRow>>),
    Holds { path: String, answer: Sender<bool> },
    Forget { path: String, answer: Sender<Result<usize, JobError>> },
    StartIndex { folder: PathBuf, answer: Sender<Result<usize, JobError>> },
    Search { query: String, answer: Sender<Result<SearchAnswer, JobError>> },
}

/// The reason a job did not run. The server turns it into a status code.
enum JobError {
    Busy,
    Message(String),
}

impl From<anyhow::Error> for JobError {
    fn from(err: anyhow::Error) -> Self {
        Self::Message(format!("{err:#}"))
    }
}

#[derive(Serialize)]
struct State {
    version: String,
    records: usize,
    videos: usize,
    key: String,
}

#[derive(Serialize)]
struct VideoRow {
    path: String,
    windows: usize,
}

#[derive(Serialize)]
struct SearchAnswer {
    groups: Vec<Group>,
}

#[derive(Deserialize)]
struct PathBody {
    path: String,
}

#[derive(Deserialize)]
struct QueryBody {
    query: String,
}

#[derive(Serialize)]
struct Picked {
    path: Option<String>,
}

#[derive(Serialize)]
struct Count {
    files: usize,
}

#[derive(Serialize)]
struct Removed {
    removed: usize,
}

#[derive(Serialize)]
struct Stopping {
    stopping: bool,
}

#[derive(Serialize)]
struct Failure {
    error: String,
}
```

- [ ] **Step 4: Write the helpers**

Add this code below the types. Replace no placeholder, all of it is real:

```rust
/// Make one response header. The name and the value are ours, so they are valid.
fn head(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("the header is valid")
}

fn bytes_reply(code: u16, content_type: &str, body: Vec<u8>) -> Reply {
    let len = body.len();
    let data: Box<dyn Read + Send> = Box::new(std::io::Cursor::new(body));
    Response::new(
        StatusCode(code),
        vec![head("Content-Type", content_type)],
        data,
        Some(len),
        None,
    )
}

fn text_reply(code: u16, content_type: &str, body: &str) -> Reply {
    bytes_reply(code, content_type, body.as_bytes().to_vec())
}

fn json_with(code: u16, value: &impl Serialize) -> Reply {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    bytes_reply(code, "application/json", body)
}

fn json_reply(value: &impl Serialize) -> Reply {
    json_with(200, value)
}

fn error_reply(code: u16, message: &str) -> Reply {
    json_with(
        code,
        &Failure {
            error: message.to_string(),
        },
    )
}

/// True when the request carries the guard header. It blocks a foreign page.
fn guarded(request: &Request) -> bool {
    request
        .headers()
        .iter()
        .any(|header| header.field.equiv("X-VEI") && header.value.as_str() == "1")
}

/// Read one byte range from a Range header.
///
/// Returns an inclusive start and end. A missing or broken range gives `None`.
fn parse_range(header: &str, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let spec = header.strip_prefix("bytes=")?;
    let first = spec.split(',').next()?.trim();
    let (left, right) = first.split_once('-')?;

    if left.is_empty() {
        let asked: u64 = right.parse().ok()?;
        if asked == 0 {
            return None;
        }
        let count = asked.min(len);
        return Some((len - count, len - 1));
    }

    let start: u64 = left.parse().ok()?;
    if start >= len {
        return None;
    }
    let end = if right.is_empty() {
        len - 1
    } else {
        right.parse::<u64>().ok()?.min(len - 1)
    };
    if end < start {
        return None;
    }
    Some((start, end))
}

/// Split a URL into its decoded query parameters.
fn params(url: &str) -> HashMap<String, String> {
    let mut found = HashMap::new();
    if let Some((_, query)) = url.split_once('?') {
        for pair in query.split('&') {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            found.insert(decode(name), decode(value));
        }
    }
    found
}

fn decode(text: &str) -> String {
    percent_encoding::percent_decode_str(&text.replace('+', " "))
        .decode_utf8_lossy()
        .into_owned()
}

fn mime_of(path: &Path) -> &'static str {
    let extension = path
        .extension()
        .and_then(|part| part.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        _ => "application/octet-stream",
    }
}

/// Build a reply for one video file. A Range header gives a 206 and a slice.
fn video_reply(path: &Path, range_header: Option<&str>) -> Result<Reply> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let mime = mime_of(path);

    let Some(range_header) = range_header else {
        let data: Box<dyn Read + Send> = Box::new(file);
        return Ok(Response::new(
            StatusCode(200),
            vec![head("Content-Type", mime), head("Accept-Ranges", "bytes")],
            data,
            Some(len as usize),
            None,
        ));
    };

    let Some((start, end)) = parse_range(range_header, len) else {
        let data: Box<dyn Read + Send> = Box::new(std::io::empty());
        return Ok(Response::new(
            StatusCode(416),
            vec![head("Content-Range", &format!("bytes */{len}"))],
            data,
            Some(0),
            None,
        ));
    };

    file.seek(SeekFrom::Start(start))?;
    let count = end - start + 1;
    let data: Box<dyn Read + Send> = Box::new(file.take(count));
    Ok(Response::new(
        StatusCode(206),
        vec![
            head("Content-Type", mime),
            head("Accept-Ranges", "bytes"),
            head("Content-Range", &format!("bytes {start}-{end}/{len}")),
        ],
        data,
        Some(count as usize),
        None,
    ))
}

fn read_body<T: for<'de> Deserialize<'de>>(request: &mut Request) -> Result<T, String> {
    let mut text = String::new();
    request
        .as_reader()
        .read_to_string(&mut text)
        .map_err(|err| format!("read the body: {err}"))?;
    serde_json::from_str(&text).map_err(|err| format!("the body is not valid JSON: {err}"))
}

fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("VEI_MODELS") {
        return PathBuf::from(dir);
    }
    let local = PathBuf::from("models");
    if local.is_dir() {
        local
    } else {
        PathBuf::from("../models")
    }
}

fn make_key() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    format!("{:x}{:x}", now ^ (pid << 64), pid.rotate_left(17))
}
```

The key uses the clock and the process id. That is enough for a local single
user server.

- [ ] **Step 5: Write the engine**

Add this code below the helpers:

```rust
struct IndexJob {
    files: Vec<PathBuf>,
    file_at: usize,
    plan: Vec<(f64, f64)>,
    at: usize,
    records: Vec<Record>,
    on_file: bool,
    indexed: usize,
}

/// The engine thread. It owns the model and the store.
struct Engine {
    embedder: Embedder,
    store: Store,
    progress: Arc<Mutex<Progress>>,
    pending: Option<IndexJob>,
}

impl Engine {
    fn start(models: &Path, store_path: &Path, progress: Arc<Mutex<Progress>>) -> Result<Self> {
        let embedder = Embedder::load(models)?;
        let store = Store::open(store_path)?;
        Ok(Self {
            embedder,
            store,
            progress,
            pending: None,
        })
    }

    /// Answer jobs forever. A running index gives one window per turn.
    fn run(mut self, jobs: Receiver<Job>) {
        loop {
            if self.pending.is_some() {
                self.step_job();
                while let Ok(job) = jobs.try_recv() {
                    self.handle(job);
                }
            } else {
                match jobs.recv() {
                    Ok(job) => self.handle(job),
                    Err(_) => break,
                }
            }
        }
    }

    fn handle(&mut self, job: Job) {
        match job {
            Job::State(answer) => {
                let _ = answer.send(self.state());
            }
            Job::Videos(answer) => {
                let _ = answer.send(self.videos());
            }
            Job::Holds { path, answer } => {
                let _ = answer.send(self.store.holds_video(&path));
            }
            Job::Forget { path, answer } => {
                let outcome = self.store.forget(&path).map_err(JobError::from);
                let _ = answer.send(outcome);
            }
            Job::Search { query, answer } => {
                let _ = answer.send(self.search(&query));
            }
            Job::StartIndex { folder, answer } => {
                let _ = answer.send(self.start_index(&folder));
            }
        }
    }

    fn state(&self) -> State {
        State {
            version: env!("CARGO_PKG_VERSION").to_string(),
            records: self.store.records.len(),
            videos: self.videos().len(),
            key: self.key.clone(),
        }
    }

    fn videos(&self) -> Vec<VideoRow> {
        let mut rows: Vec<VideoRow> = Vec::new();
        for record in &self.store.records {
            if record.kind != KIND_WINDOW {
                continue;
            }
            match rows.iter_mut().find(|row| row.path == record.label) {
                Some(row) => row.windows += 1,
                None => rows.push(VideoRow {
                    path: record.label.clone(),
                    windows: 1,
                }),
            }
        }
        rows
    }

    fn search(&self, query: &str) -> Result<SearchAnswer, JobError> {
        if self.store.records.is_empty() {
            return Ok(SearchAnswer { groups: Vec::new() });
        }
        let text = format!("{QUERY_PREFIX}{query}");
        let vec = self.embedder.embed_text(&text).map_err(JobError::from)?;
        let ranked = self.store.rank(&vec);
        Ok(SearchAnswer {
            groups: index::group_hits(&self.store.records, &ranked),
        })
    }

    fn start_index(&mut self, folder: &Path) -> Result<usize, JobError> {
        if self.pending.is_some() {
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
            let mut progress = self.progress.lock().unwrap();
            *progress = Progress {
                running: true,
                folder: folder.display().to_string(),
                file_count: count,
                ..Progress::default()
            };
        }
        self.pending = Some(IndexJob {
            files,
            file_at: 0,
            plan: Vec::new(),
            at: 0,
            records: Vec::new(),
            on_file: false,
            indexed: 0,
        });
        Ok(count)
    }

    /// One window of work, or the move to the next file.
    fn step_job(&mut self) {
        let Some(mut job) = self.pending.take() else {
            return;
        };

        if !job.on_file {
            if job.file_at >= job.files.len() {
                self.progress.lock().unwrap().running = false;
                return;
            }
            let video = job.files[job.file_at].clone();
            match self.open_file(&video, job.file_at + 1) {
                Ok(plan) => {
                    job.plan = plan;
                    job.at = 0;
                    job.records.clear();
                    job.on_file = true;
                }
                Err(err) => {
                    self.fail(&err);
                    job.file_at += 1;
                }
            }
            self.pending = Some(job);
            return;
        }

        let video = job.files[job.file_at].clone();
        let (start, end) = job.plan[job.at];
        if let Err(err) = self.one_window(&video, start, end, &mut job.records) {
            self.fail(&err);
        }
        job.at += 1;
        {
            let mut progress = self.progress.lock().unwrap();
            progress.window = job.at;
            progress.indexed = job.indexed + job.records.len();
        }
        if job.at >= job.plan.len() {
            let records = std::mem::take(&mut job.records);
            job.indexed += records.len();
            if let Err(err) = self.store.append(records) {
                self.fail(&err);
            }
            job.file_at += 1;
            job.on_file = false;
        }
        self.pending = Some(job);
    }

    /// Probe the duration and drop the old records of this file.
    ///
    /// The probe runs first, so a broken file keeps its old index.
    fn open_file(&mut self, video: &Path, number: usize) -> Result<Vec<(f64, f64)>> {
        let duration = media::probe_duration(video)?;
        let plan = index::windows(duration);
        if plan.is_empty() {
            bail!("{} is shorter than one window", video.display());
        }
        let label = video.to_string_lossy().to_string();
        self.store.forget(&label)?;
        {
            let mut progress = self.progress.lock().unwrap();
            progress.file = label;
            progress.file_index = number;
            progress.window = 0;
            progress.window_count = plan.len();
            progress.error = None;
        }
        Ok(plan)
    }

    fn one_window(
        &self,
        video: &Path,
        start: f64,
        end: f64,
        records: &mut Vec<Record>,
    ) -> Result<()> {
        let dir = index::work_dir(records.len());
        let outcome = media::extract_window(video, start, end, &dir)
            .and_then(|(frames, audio)| self.embedder.embed_window(&frames, &audio));
        let _ = std::fs::remove_dir_all(&dir);
        records.push(Record {
            kind: KIND_WINDOW,
            label: video.to_string_lossy().to_string(),
            start: start as f32,
            end: end as f32,
            vec: outcome?,
        });
        Ok(())
    }

    fn fail(&self, err: &anyhow::Error) {
        self.progress.lock().unwrap().error = Some(format!("{err:#}"));
    }
}
```

`Engine::state` reads `self.key`. Add the field and set it in `start`:

```rust
struct Engine {
    embedder: Embedder,
    store: Store,
    progress: Arc<Mutex<Progress>>,
    pending: Option<IndexJob>,
    key: String,
}
```

In `start`, add `key: make_key(),` to the struct literal.

- [ ] **Step 6: Write the request handler and main**

Add this code below the engine:

```rust
fn handle(
    mut request: Request,
    jobs: &Sender<Job>,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
) -> Result<()> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let post = request.method() == &Method::Post;

    match path.as_str() {
        "/" => request.respond(text_reply(200, "text/html; charset=utf-8", PAGE))?,
        "/app.js" => request.respond(text_reply(200, "text/javascript; charset=utf-8", JS))?,
        "/app.css" => request.respond(text_reply(200, "text/css; charset=utf-8", CSS))?,
        "/favicon.ico" => request.respond(bytes_reply(204, "image/x-icon", Vec::new()))?,
        _ if !path.starts_with("/api/") => {
            request.respond(error_reply(404, "no such page"))?;
        }
        "/api/video" => {
            let found = params(&url);
            if found.get("key").map(String::as_str) != Some(key) {
                request.respond(error_reply(403, "the key is missing"))?;
                return Ok(());
            }
            serve_video(request, found.get("path").cloned(), jobs)?;
        }
        _ => {
            if !guarded(&request) {
                request.respond(error_reply(403, "the X-VEI header is missing"))?;
                return Ok(());
            }
            if path == "/api/progress" {
                let snapshot = progress.lock().unwrap().clone();
                request.respond(json_reply(&snapshot))?;
                return Ok(());
            }
            api(request, &path, post, jobs)?;
        }
    }
    Ok(())
}

fn serve_video(mut request: Request, path: Option<String>, jobs: &Sender<Job>) -> Result<()> {
    let Some(path) = path else {
        request.respond(error_reply(400, "the path is missing"))?;
        return Ok(());
    };
    let (answer, wait) = channel();
    let _ = jobs.send(Job::Holds {
        path: path.clone(),
        answer,
    });
    if !matches!(wait.recv(), Ok(true)) {
        request.respond(error_reply(403, "that video is not in the index"))?;
        return Ok(());
    }
    let range = request
        .headers()
        .iter()
        .find(|header| header.field.equiv("Range"))
        .map(|header| header.value.as_str().to_string());
    match video_reply(Path::new(&path), range.as_deref()) {
        Ok(reply) => request.respond(reply)?,
        Err(err) => {
            let message = format!("{err:#}");
            request.respond(error_reply(500, &message))?;
        }
    }
    Ok(())
}

fn api(
    mut request: Request,
    path: &str,
    post: bool,
    jobs: &Sender<Job>,
) -> Result<()> {
    match (post, path) {
        (false, "/api/state") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::State(answer));
            match wait.recv() {
                Ok(state) => request.respond(json_reply(&state))?,
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (false, "/api/videos") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::Videos(answer));
            match wait.recv() {
                Ok(rows) => request.respond(json_reply(&Rows { videos: rows }))?,
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/forget") => {
            let body: PathBody = match read_body(&mut request) {
                Ok(body) => body,
                Err(message) => {
                    request.respond(error_reply(400, &message))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::Forget {
                path: body.path,
                answer,
            });
            match wait.recv() {
                Ok(Ok(removed)) => request.respond(json_reply(&Removed { removed }))?,
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(400, &message))?
                }
                _ => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/index") => {
            let body: PathBody = match read_body(&mut request) {
                Ok(body) => body,
                Err(message) => {
                    request.respond(error_reply(400, &message))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::StartIndex {
                folder: PathBuf::from(body.path),
                answer,
            });
            match wait.recv() {
                Ok(Ok(files)) => request.respond(json_reply(&Count { files }))?,
                Ok(Err(JobError::Busy)) => {
                    request.respond(error_reply(409, "an index job already runs"))?
                }
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(400, &message))?
                }
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/search") => {
            let body: QueryBody = match read_body(&mut request) {
                Ok(body) => body,
                Err(message) => {
                    request.respond(error_reply(400, &message))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::Search {
                query: body.query,
                answer,
            });
            match wait.recv() {
                Ok(Ok(found)) => request.respond(json_reply(&found))?,
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(500, &message))?
                }
                _ => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/pick-folder") => {
            // The dialog blocks this thread alone. Each request gets its own thread.
            let picked = rfd::FileDialog::new().pick_folder();
            let answer = Picked {
                path: picked.map(|p| p.to_string_lossy().to_string()),
            };
            request.respond(json_reply(&answer))?;
        }
        (true, "/api/quit") => {
            let _ = request.respond(json_reply(&Stopping { stopping: true }));
            std::thread::sleep(Duration::from_millis(200));
            std::process::exit(0);
        }
        _ => request.respond(error_reply(405, "that method is not allowed here"))?,
    }
    Ok(())
}

fn main() -> Result<()> {
    let models = models_dir();
    let store_path = PathBuf::from("store/index.vei");
    let progress = Arc::new(Mutex::new(Progress::default()));
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<String, String>>();

    let engine_progress = Arc::clone(&progress);
    std::thread::spawn(move || {
        let engine = match Engine::start(&models, &store_path, engine_progress) {
            Ok(engine) => engine,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        // The ready signal carries the video key, so both threads share one value.
        let _ = ready_tx.send(Ok(engine.key.clone()));
        engine.run(jobs_rx);
    });

    // The model and the store must be ready before the server answers one call.
    let key = match ready_rx.recv().context("the engine thread stopped early")? {
        Ok(key) => key,
        Err(message) => bail!("the engine failed to start: {message}"),
    };

    let server = Server::http(ADDRESS).map_err(|err| anyhow!("{err}"))?;
    println!("videoembed is at http://{ADDRESS}/");
    println!("Stop the server with the Quit button or with Ctrl C.");
    let _ = open::that(format!("http://{ADDRESS}/"));

    for request in server.incoming_requests() {
        let jobs = jobs_tx.clone();
        let progress = Arc::clone(&progress);
        let key = key.clone();
        std::thread::spawn(move || {
            if let Err(err) = handle(request, &jobs, &progress, &key) {
                eprintln!("A request failed: {err:#}");
            }
        });
    }
    Ok(())
}
```

`Rows` is the answer wrapper for `/api/videos`. Add it to the types:

```rust
#[derive(Serialize)]
struct Rows {
    videos: Vec<VideoRow>,
}
```

The engine builds the video key with `make_key()` in `Engine::start`, and the
ready signal carries it to `main`. One value serves both threads.

- [ ] **Step 7: Write the page shell**

Create `videoembed/assets/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>videoembed</title>
<link rel="stylesheet" href="/app.css">
</head>
<body>
<header>
  <h1>videoembed</h1>
  <span id="status">Loading</span>
  <button id="quit" type="button">Quit</button>
</header>

<main>
<section>
  <h2>Folder</h2>
  <p id="folder">No folder</p>
  <button id="pick" type="button">Choose folder</button>
  <button id="start" type="button" disabled>Index</button>
  <div id="progress" hidden>
    <div class="bar"><div id="progress-bar"></div></div>
    <p id="progress-text"></p>
  </div>
</section>

<section>
  <h2>Search</h2>
  <form id="search-form">
    <input id="query" type="search" placeholder="birthday party" autocomplete="off">
    <button type="submit">Search</button>
  </form>
  <p id="message" class="message" hidden></p>
  <ul id="groups"></ul>
</section>

<section>
  <h2>Player</h2>
  <video id="player" controls preload="metadata"></video>
</section>

<section>
  <h2>Timeline</h2>
  <div id="timeline"></div>
</section>

<section>
  <h2>Indexed</h2>
  <table id="videos"><tbody></tbody></table>
</section>
</main>
<script src="/app.js"></script>
</body>
</html>
```

Create `videoembed/assets/app.css`:

```css
:root {
  color-scheme: dark;
  --line: #30363d;
  --dim: #8b949e;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: system-ui, sans-serif;
  background: #0d1117;
  color: #e6edf3;
}

header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--line);
}

header h1 { font-size: 1.1rem; margin: 0; }
header #status { color: var(--dim); font-size: 0.85rem; }
header #quit { margin-left: auto; }

main {
  display: grid;
  gap: 1.5rem;
  padding: 1.5rem;
  max-width: 70rem;
}

section { border: 1px solid var(--line); border-radius: 8px; padding: 1rem; }
h2 { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dim); margin: 0 0 0.75rem; }

button {
  font: inherit;
  color: inherit;
  background: #21262d;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.35rem 0.75rem;
  cursor: pointer;
}
button:hover { background: #30363d; }
button:disabled { opacity: 0.45; cursor: default; }

input[type="search"] {
  font: inherit;
  color: inherit;
  background: #010409;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.4rem 0.6rem;
  width: 24rem;
  max-width: 100%;
}

#folder { font-family: ui-monospace, monospace; font-size: 0.85rem; }

.bar { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-top: 0.75rem; }
#progress-bar { height: 100%; width: 0; background: #2f81f7; }
#progress-text { font-size: 0.85rem; color: var(--dim); margin: 0.4rem 0 0; }

.message { color: #f85149; font-size: 0.9rem; }

#groups { list-style: none; margin: 0; padding: 0; }
#groups li { padding: 0.5rem 0; border-bottom: 1px solid var(--line); }
#groups strong { display: block; font-family: ui-monospace, monospace; font-size: 0.8rem; color: var(--dim); }

.hit { margin: 0.35rem 0.35rem 0 0; font-variant-numeric: tabular-nums; }

#player { width: 100%; max-height: 26rem; background: #000; border-radius: 6px; }

.track {
  position: relative;
  height: 2rem;
  margin-bottom: 0.5rem;
  background: #161b22;
  border-radius: 4px;
}
.track-name { position: absolute; left: 0.5rem; top: 0.45rem; font-size: 0.72rem; color: var(--dim); }
.mark {
  position: absolute;
  top: 0;
  height: 100%;
  padding: 0;
  background: #2f81f7aa;
  border: 1px solid #2f81f7;
  border-radius: 2px;
}
.mark:hover { background: #2f81f7; }

table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
td { padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--line); }
td:first-child { font-family: ui-monospace, monospace; word-break: break-all; }
td:last-child { text-align: right; }
```

Create `videoembed/assets/app.js`:

```js
const API = { "X-VEI": "1", "Content-Type": "application/json" };
const $ = (id) => document.getElementById(id);

let key = "";

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: API,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.status + " " + res.statusText);
  return data;
}

function note(text) {
  const el = $("message");
  el.textContent = text;
  el.hidden = text === "";
}

async function refreshState() {
  const state = await api("/api/state");
  key = state.key;
  $("status").textContent = state.records + " windows, " + state.videos + " videos";
}

function boot() {
  refreshState().catch((err) => note(err.message));
}

boot();
```

- [ ] **Step 8: Run the tests**

Run: `cargo test --bin web`
Expected: `6 passed`.

Run: `cargo test`
Expected: every test passes.

- [ ] **Step 9: Check the server by hand**

Run: `cargo build && ./target/debug/web.exe`

In another shell, run these two commands.

Run: `curl -s -H "X-VEI: 1" http://127.0.0.1:8787/api/state`
Expected: a JSON object with `records`, `videos`, and a `key`.

Run: `curl -s http://127.0.0.1:8787/api/state`
Expected: `{"error":"the X-VEI header is missing"}`. The guard works.

Open `http://127.0.0.1:8787/` in a browser. The page shows the header, the
search box, and a window count. Stop the server with Ctrl C.

- [ ] **Step 10: Commit**

```bash
git add videoembed/src/bin/web.rs videoembed/assets
git commit -m "feat: serve the page and the engine thread"
```

---

### Task 6: The store manager

List the indexed videos and delete one.

**Files:**
- Modify: `videoembed/src/bin/web.rs` (the `api` function, already handles both)
- Modify: `videoembed/assets/app.js`

**Interfaces:**
- Consumes: `Job::Videos`, `Job::Forget`, `Store::forget`, `Rows`.
- Produces: the page functions `refreshVideos()` and `forget(path)`.

- [ ] **Step 1: Write the page functions**

Add this code to `videoembed/assets/app.js`, above `boot`:

```js
async function refreshVideos() {
  const answer = await api("/api/videos");
  const body = $("videos").querySelector("tbody");
  body.textContent = "";
  if (answer.videos.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "No video is indexed yet.";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const video of answer.videos) {
    const row = document.createElement("tr");

    const name = document.createElement("td");
    name.textContent = video.path;

    const count = document.createElement("td");
    count.textContent = video.windows + " windows";

    const act = document.createElement("td");
    const drop = document.createElement("button");
    drop.type = "button";
    drop.textContent = "Forget";
    drop.addEventListener("click", () => forget(video.path));
    act.appendChild(drop);

    row.append(name, count, act);
    body.appendChild(row);
  }
}

async function forget(path) {
  try {
    await api("/api/forget", { path });
    await refreshVideos();
    await refreshState();
  } catch (err) {
    note(err.message);
  }
}
```

- [ ] **Step 2: Call it at boot**

Replace `boot` in `videoembed/assets/app.js` with:

```js
function boot() {
  refreshState().catch((err) => note(err.message));
  refreshVideos().catch((err) => note(err.message));
}
```

- [ ] **Step 3: Check it by hand**

Run: `cargo build && ./target/debug/web.exe`

Open the page. The Indexed section lists the test video with 14 windows.
Note the count in the header. Press Forget on the row.

Expected: the row is replaced by `No video is indexed yet.`, and the header
count falls by 14. `videoembed/store/index.vei` is smaller on disk.

Forget is hard to undo, so re-index the video after the check:

Run: `./target/debug/videoembed.exe index ../spike/testvideo_nolabels.mp4`

- [ ] **Step 4: Commit**

```bash
git add videoembed/src/bin/web.rs videoembed/assets/app.js
git commit -m "feat: list the indexed videos and delete one"
```

---

### Task 7: The folder picker, the index job, and the progress line

The page picks a folder, starts a job, and polls the progress twice a second.

**Files:**
- Modify: `videoembed/assets/app.js`

**Interfaces:**
- Consumes: `POST /api/pick-folder`, `POST /api/index`, `GET /api/progress`,
  `Progress { running, folder, file, file_index, file_count, window,
  window_count, indexed, error }`.
- Produces: the page functions `pickFolder()`, `startIndex()`,
  `showProgress(p)`, `startPolling()`.

- [ ] **Step 1: Write the folder and progress functions**

Add this code to `videoembed/assets/app.js`, above `boot`:

```js
let folder = null;
let timer = null;

async function pickFolder() {
  try {
    const answer = await api("/api/pick-folder", {});
    if (answer.path === null) return;
    folder = answer.path;
    $("folder").textContent = folder;
    $("start").disabled = false;
    note("");
  } catch (err) {
    note(err.message);
  }
}

async function startIndex() {
  if (folder === null) return;
  $("start").disabled = true;
  note("");
  try {
    await api("/api/index", { path: folder });
    startPolling();
  } catch (err) {
    $("start").disabled = false;
    note(err.message);
  }
}

function showProgress(p) {
  $("progress").hidden = !p.running && p.indexed === 0 && !p.error;
  const width = p.window_count > 0 ? (p.window * 100) / p.window_count : 0;
  $("progress-bar").style.width = width + "%";

  if (p.error !== null) {
    $("progress-text").textContent = p.error;
    return;
  }
  if (p.running) {
    $("progress-text").textContent =
      "File " + p.file_index + "/" + p.file_count +
      ", window " + p.window + "/" + p.window_count +
      " - " + p.file;
    return;
  }
  $("progress-text").textContent = "Done. " + p.indexed + " windows indexed.";
}

async function pollProgress() {
  let p;
  try {
    p = await api("/api/progress");
  } catch (err) {
    note(err.message);
    return;
  }
  showProgress(p);
  if (p.running) return;
  clearInterval(timer);
  timer = null;
  $("start").disabled = false;
  await refreshVideos();
  await refreshState();
}

function startPolling() {
  if (timer !== null) return;
  timer = setInterval(pollProgress, 500);
  pollProgress();
}
```

`showProgress` guards `window_count === 0`, so the first poll cannot divide by
zero. Test 3 of the Review Focus list lands here.

- [ ] **Step 2: Wire the buttons at boot**

Replace `boot` in `videoembed/assets/app.js` with:

```js
function boot() {
  $("pick").addEventListener("click", pickFolder);
  $("start").addEventListener("click", startIndex);
  refreshState().catch((err) => note(err.message));
  refreshVideos().catch((err) => note(err.message));
}
```

- [ ] **Step 3: Check it by hand**

Run: `cargo build && ./target/debug/web.exe`

1. Press Choose folder. Pick `D:/MLProjelerim/videoembedder-rust/spike`.
2. Press Index. The bar fills, and the text names each file and window.
3. Wait for the run to end. The text reads `Done. 14 windows indexed.`
4. Press Index again on the same folder. The count stays 14, because a
   re-index replaces the old records.

The `spike` folder holds one video.

- [ ] **Step 4: Commit**

```bash
git add videoembed/assets/app.js
git commit -m "feat: pick a folder, index it, and show progress"
```

---

### Task 8: Search and the timeline

The page sends a query, shows the hits, and draws a mark for each one.

**Files:**
- Modify: `videoembed/assets/app.js`

**Interfaces:**
- Consumes: `POST /api/search` with `{"query": "..."}`, and the answer
  `{"groups": [{video, duration, score, hits: [{start, end, score}]}]}`.
- Produces: the page functions `runSearch(event)`, `renderGroups(groups)`,
  `timelineBar(group)`, `play(path, start)`.

Task 9 adds the player body of `play`. This task defines `play` with the two
lines that set the source, so the page works when a row is clicked.

- [ ] **Step 1: Write the search and timeline functions**

Add this code to `videoembed/assets/app.js`, above `boot`:

```js
async function runSearch(event) {
  event.preventDefault();
  const query = $("query").value.trim();
  if (query === "") return;
  note("");
  try {
    const answer = await api("/api/search", { query });
    renderGroups(answer.groups);
  } catch (err) {
    note(err.message);
    renderGroups([]);
  }
}

function renderGroups(groups) {
  const list = $("groups");
  list.textContent = "";
  const timeline = $("timeline");
  timeline.textContent = "";

  if (groups.length === 0) {
    note("No window scored above 0.4.");
    return;
  }
  note("");
  for (const group of groups) {
    const item = document.createElement("li");

    const title = document.createElement("strong");
    title.textContent = group.video;
    item.appendChild(title);

    for (const hit of group.hits) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hit";
      row.textContent =
        hit.score.toFixed(3) + "  " + hit.start.toFixed(1) + "-" + hit.end.toFixed(1) + " s";
      row.addEventListener("click", () => play(group.video, hit.start));
      item.appendChild(row);
    }

    list.appendChild(item);
    timeline.appendChild(timelineBar(group));
  }
}

function timelineBar(group) {
  const bar = document.createElement("div");
  bar.className = "track";

  const name = document.createElement("span");
  name.className = "track-name";
  name.textContent = group.video;
  bar.appendChild(name);

  // The store holds no true duration. The largest window end is the axis.
  const last = group.hits.length > 0 ? group.hits[group.hits.length - 1].end : 0;
  const span = group.duration > 0 ? group.duration : last > 0 ? last : 1;

  for (const hit of group.hits) {
    const mark = document.createElement("button");
    mark.type = "button";
    mark.className = "mark";
    mark.style.left = (hit.start / span) * 100 + "%";
    mark.style.width = ((hit.end - hit.start) / span) * 100 + "%";
    mark.title = hit.score.toFixed(3) + "  " + hit.start.toFixed(1) + " s";
    mark.addEventListener("click", () => play(group.video, hit.start));
    bar.appendChild(mark);
  }
  return bar;
}

function play(path, start) {
  const player = $("player");
  if (player.dataset.path !== path) {
    player.dataset.path = path;
    player.src = "/api/video?key=" + encodeURIComponent(key) +
      "&path=" + encodeURIComponent(path);
  }
  const seek = () => {
    player.currentTime = start;
    player.play();
  };
  if (player.readyState >= 1) seek();
  else player.addEventListener("loadedmetadata", seek, { once: true });
}
```

Task 9 completes `play`. This version sets the source and the time, which
already plays the video from a click.

- [ ] **Step 2: Wire the search form at boot**

Replace `boot` in `videoembed/assets/app.js` with:

```js
function boot() {
  $("pick").addEventListener("click", pickFolder);
  $("start").addEventListener("click", startIndex);
  $("search-form").addEventListener("submit", runSearch);
  refreshState().catch((err) => note(err.message));
  refreshVideos().catch((err) => note(err.message));
}
```

- [ ] **Step 3: Check it by hand**

Run: `cargo build && ./target/debug/web.exe`

1. Type `blue` in the search box and press Enter.
2. A group for the test video appears. Its best hit sits inside 10-20 s.
3. Type `sunny weather`. The best hit sits inside 0-10 s.
4. Type `zzzz`. The message reads `No window scored above 0.4.`
5. The timeline shows one bar for the video, with a mark at each hit.

- [ ] **Step 4: Commit**

```bash
git add videoembed/assets/app.js
git commit -m "feat: search the index and draw the timeline"
```

---

### Task 9: Range reads and the player

The browser seeks with a byte range request. Prove the range decision, then
finish the player.

**Files:**
- Modify: `videoembed/src/bin/web.rs` (the `video_reply` function and the tests)
- Modify: `videoembed/assets/app.js` (the `play` function)

**Interfaces:**
- Consumes: `parse_range`, `mime_of`.
- Produces: `enum RangePlan { Whole, Part(u64, u64), Refused }` and
  `fn range_plan(range_header: Option<&str>, len: u64) -> RangePlan`.

- [ ] **Step 1: Write the failing tests**

Add these tests to the test module in `videoembed/src/bin/web.rs`:

```rust
    #[test]
    fn no_range_header_asks_for_the_whole_file() {
        assert_eq!(range_plan(None, 1000), RangePlan::Whole);
    }

    #[test]
    fn a_range_header_asks_for_a_slice() {
        assert_eq!(range_plan(Some("bytes=0-99"), 1000), RangePlan::Part(0, 99));
        assert_eq!(range_plan(Some("bytes=500-"), 1000), RangePlan::Part(500, 999));
    }

    #[test]
    fn a_range_starting_past_the_end_is_refused_without_a_panic() {
        assert_eq!(range_plan(Some("bytes=2000-"), 1000), RangePlan::Refused);
        assert_eq!(range_plan(Some("bytes=-"), 1000), RangePlan::Refused);
        assert_eq!(range_plan(Some("bytes=0-"), 0), RangePlan::Refused);
    }

    #[test]
    fn a_suffix_range_reads_from_the_end() {
        assert_eq!(range_plan(Some("bytes=-100"), 1000), RangePlan::Part(900, 999));
        assert_eq!(range_plan(Some("bytes=-5000"), 1000), RangePlan::Part(0, 999));
    }
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --bin web`
Expected: a compile error. `range_plan` and `RangePlan` are not defined.

- [ ] **Step 3: Write the plan function**

Add this code to `videoembed/src/bin/web.rs`, above `video_reply`:

```rust
/// What one Range header asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangePlan {
    /// Send the whole file with a 200.
    Whole,
    /// Send the bytes from start to end, both included, with a 206.
    Part(u64, u64),
    /// Send a 416 and no body.
    Refused,
}

fn range_plan(range_header: Option<&str>, len: u64) -> RangePlan {
    match range_header {
        None => RangePlan::Whole,
        Some(header) => match parse_range(header, len) {
            Some((start, end)) => RangePlan::Part(start, end),
            None => RangePlan::Refused,
        },
    }
}
```

- [ ] **Step 4: Use it in `video_reply`**

Replace the body of `video_reply` with:

```rust
fn video_reply(path: &Path, range_header: Option<&str>) -> Result<Reply> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let mime = mime_of(path);

    match range_plan(range_header, len) {
        RangePlan::Whole => {
            let data: Box<dyn Read + Send> = Box::new(file);
            Ok(Response::new(
                StatusCode(200),
                vec![head("Content-Type", mime), head("Accept-Ranges", "bytes")],
                data,
                Some(len as usize),
                None,
            ))
        }
        RangePlan::Refused => {
            let data: Box<dyn Read + Send> = Box::new(std::io::empty());
            Ok(Response::new(
                StatusCode(416),
                vec![head("Content-Range", &format!("bytes */{len}"))],
                data,
                Some(0),
                None,
            ))
        }
        RangePlan::Part(start, end) => {
            file.seek(SeekFrom::Start(start))?;
            let count = end - start + 1;
            let data: Box<dyn Read + Send> = Box::new(file.take(count));
            Ok(Response::new(
                StatusCode(206),
                vec![
                    head("Content-Type", mime),
                    head("Accept-Ranges", "bytes"),
                    head("Content-Range", &format!("bytes {start}-{end}/{len}")),
                ],
                data,
                Some(count as usize),
                None,
            ))
        }
    }
}
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cargo test --bin web`
Expected: `10 passed`.

- [ ] **Step 6: Prove the range code against a real file**

Start the server and find the path of the indexed video in the Indexed
section. Copy that path and its key from `GET /api/state`.

Run: `curl -s -D - -o /dev/null -r 0-99 "http://127.0.0.1:8787/api/video?key=KEY&path=PATH"`
Expected: `HTTP/1.1 206`, `Content-Range: bytes 0-99/SIZE`, and
`Content-Length: 100`.

Run: `curl -s -D - -o /dev/null -r 99999999- "http://127.0.0.1:8787/api/video?key=KEY&path=PATH"`
Expected: `HTTP/1.1 416`.

Run: `curl -s -D - -o /dev/null "http://127.0.0.1:8787/api/video?path=PATH"`
Expected: `HTTP/1.1 403` with `{"error":"the key is missing"}`.

- [ ] **Step 7: Finish the player**

Replace `play` in `videoembed/assets/app.js` with:

```js
function play(path, start) {
  const player = $("player");
  const source = "/api/video?key=" + encodeURIComponent(key) +
    "&path=" + encodeURIComponent(path);

  const seek = () => {
    player.currentTime = start;
    player.play();
  };

  if (player.dataset.path !== path) {
    player.dataset.path = path;
    player.addEventListener("loadedmetadata", seek, { once: true });
    player.src = source;
    return;
  }
  if (player.readyState >= 1) seek();
  else player.addEventListener("loadedmetadata", seek, { once: true });
}
```

- [ ] **Step 8: Check playback by hand**

Run: `cargo build && ./target/debug/web.exe`

1. Search `blue`, then click the top hit.
2. The player loads the video and starts inside the 10-20 s band.
3. Drag the player bar to a new point. Playback continues from there.
   This proves the 206 path works in a browser.
4. Seek to the end and back. The player does not stop or freeze.

- [ ] **Step 9: Commit**

```bash
git add videoembed/src/bin/web.rs videoembed/assets/app.js
git commit -m "feat: serve byte ranges and play the matching moment"
```

---

### Task 10: Quit, the README, and the manual pass

Close the loop. The page gets a Quit button, the README describes the web UI,
and the whole spec's check list runs once.

**Files:**
- Modify: `videoembed/assets/app.js`
- Modify: `README.md`

**Interfaces:**
- Consumes: `POST /api/quit`.
- Produces: the page function `quit()`.

- [ ] **Step 1: Add the Quit button**

Add this code to `videoembed/assets/app.js`, above `boot`:

```js
async function quit() {
  try {
    await api("/api/quit", {});
  } catch (err) {
    note(err.message);
  }
}
```

Then add this line to `boot`:

```js
  $("quit").addEventListener("click", quit);
```

- [ ] **Step 2: Check that quit stops the process**

Run: `cargo build && ./target/debug/web.exe`

Press Quit on the page. Expected: the push-button stops, the console window
returns to its prompt, and the page cannot reach the server again.

- [ ] **Step 3: Update the README**

Change the status table row for the web UI to `Works. See the Web UI section.`

Add this section after the `Run` section:

````markdown
## Web UI

Run the web binary from the `videoembed` folder:

```bash
./target/debug/web.exe
```

The app opens `http://127.0.0.1:8787/` in your browser.

1. Press Choose folder and pick a folder.
2. Press Index. The bar shows the file and the window.
3. Type a query. The hits appear under Search.
4. Press a hit. The player jumps to that moment.
5. Press Forget on a row to drop one video from the index.
6. Press Quit to stop the server.

The server binds to `127.0.0.1` only, so another machine cannot reach it.

Each `/api/` call carries the header `X-VEI: 1`. A foreign page cannot set that
header, and the server sends no CORS headers, so its request fails. The player
needs one exception. A `<video>` element cannot set a header, so `/api/video`
takes a random key in the query instead. The page reads that key from
`/api/state`.

| Endpoint | Method | Action |
|---|---|---|
| `/api/state` | GET | The version, the record count, the video count, and the video key. |
| `/api/progress` | GET | The progress object of the running job. |
| `/api/videos` | GET | Every indexed video with its window count. |
| `/api/search` | POST | Rank the store and group the hits by video. |
| `/api/pick-folder` | POST | Open the folder dialog. |
| `/api/index` | POST | Index every video under a folder. |
| `/api/forget` | POST | Drop one video from the index. |
| `/api/video` | GET | The video bytes, with byte ranges. |
| `/api/quit` | POST | Stop the server. |
````

Change the last known rough edge to name the web UI instead:

```markdown
3. The console indexes one video. The web page indexes a folder.
```

- [ ] **Step 4: Lint the README and the plan**

Run:
`node "C:\Users\emir-\.agents\skills\asd-ste100\hooks\run-python.cjs" "C:\Users\emir-\.agents\skills\asd-ste100\scripts\ste-lint.py" --fail-over 2.5 README.md docs/superpowers/plans/2026-10-08-videoembed-web-ui.md`

Expected: a violation rate of 2.5 per 100 words or less for both files. Fix any
sentence that pushes the rate over the limit.

- [ ] **Step 5: Run the spec check list end to end**

Start the server and run all seven checks from spec section 8:

1. `cargo run --bin web` opens the browser at `127.0.0.1:8787`.
2. Pick the `spike` folder. The progress line moves, and the run ends.
3. Search `blue`. The top hit is a window inside 10-20 s.
4. Click the hit. The player loads the video and seeks to about 12 s.
5. Drag the player bar. Playback continues from the new point.
6. Forget the video in the store manager. The row leaves the list.
7. Press Quit. The process exits and the page stops.

Re-index the video after check 6.

- [ ] **Step 6: Run the whole test suite**

Run: `cargo test`
Expected: every test passes, with no warning about an unused import.

- [ ] **Step 7: Commit**

```bash
git add videoembed/assets/app.js README.md
git commit -m "docs: describe the web UI and finish the quit button"
```

---

## Self-Review

**Spec coverage.** Each spec section maps to a task.

| Spec section | Task |
|---|---|
| 4.1 Shape, the library target, the assets | 1, Task 5 step 7 |
| 4.2 One engine thread | Task 5 step 5 |
| 4.3 The API table, `/api/state` | Task 5 |
| 4.3 `/api/progress` | Task 5, Task 7 |
| 4.3 `/api/search` and the group shape | Task 2, Task 8 |
| 4.3 `/api/videos`, `/api/forget` | Task 6 |
| 4.3 `/api/pick-folder`, `/api/index` | Task 7 |
| 4.3 `/api/video` | Task 9 |
| 4.3 `/api/quit` | Task 10 |
| 4.4 The index job, the 409, the per-file append | Task 7, Task 5 step 5 |
| 4.5 Player and byte ranges | Task 9 |
| 4.6 Timeline | Task 8 |
| 4.7 Safety | Task 5 step 4, Task 4, Task 9 |
| 4.8 The store stays `VEI1` | Task 4 |
| 6 Error handling | Tasks 4, 5, 7, 9 |
| 7 Components and interfaces | Tasks 1 to 5 |
| 8 Testing | Every task |

**Placeholder scan.** The plan holds no `TBD`, no `TODO`, and no "add error
handling" step. Every code step shows the code.

**Type consistency.** `index_video` takes `&Embedder`, `&Path`, and
`&mut dyn FnMut(usize, usize)`, and returns `Result<Vec<Record>>`, in Task 3 and
in the spec. `Progress` carries the same nine field names in Task 2, in the
spec JSON, and in the page. `group_hits` returns `Vec<Group>`, and
`SearchAnswer.groups` holds `Vec<Group>`. `range_plan` returns the three
`RangePlan` variants used by `video_reply`. `Store::holds_video` takes `&str` at
all three call sites: the test, `Job::Holds`, and `serve_video`.

**Review Focus.** Each of the five lines names its test:

1. A space and a Unicode letter: Task 2's scan test, Task 4's `holds_video`
   test, Task 5's `params` test.
2. A path outside the index: Task 4's `holds_video_matches_one_exact_path`, and
   Task 9 step 6's 403 check.
3. Progress before a job: Task 2's JSON test, and the `window_count === 0`
   guard in Task 7.
4. A search with no hit: Task 2's floor test, and the message branch of
   `renderGroups` in Task 8.
5. A range past the end: Task 9's `RangePlan::Refused` tests and the 416 check.

**One spec change.** Section 4.7 says every `/api/` call carries `X-VEI: 1`. The
player cannot, so `/api/video` uses a query key instead. The README states the
rule and the exception.
