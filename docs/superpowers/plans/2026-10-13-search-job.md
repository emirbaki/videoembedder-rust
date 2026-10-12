# The Search and the Job Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A search shows one row for each moment, the web job pauses and resumes,
and a folder run clears the done mark of a video that holds no records.

**Architecture:** One fold rule joins the near rows of one video, and the command
line and the web both read it. The pause is a flag on the engine thread. The
engine reads the flag between items, so a paused job keeps its place and its cut
files. The reconcile pass reads the done marks at the start of a folder run.

**Tech Stack:** Rust, `llama.cpp` through the vendored `llama-cpp-2` crate,
SQLite with `sqlite-vec`, a small HTTP server (`tiny_http`), plain JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md`

## Global Constraints

- Every test run uses this command from the repo root:
  `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`.
  Another generator fails to configure the vendored `llama.cpp`.
- Every word the reader sees follows ASD-STE100, default mode. That rule covers
  chat text, commit messages, docs, and code comments.
- The tests follow section 6 of the spec. Media work uses the real `ffmpeg` and
  the real test clip. A pure function takes a plain unit test. A web route takes
  a unit test of its handler, plus one test of the state the page reads.
- The constants of Plans A and B stay as they are. `MERGE_GAP = 0.5` is new.
- One name for one thing. A folded row is a "moment". A held job is "paused".
- End every commit message with this line:
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- Watch each test fail before the code that passes it. Run the whole suite
  before each commit.

## Review Focus

These inputs and conditions are likely to bite a person. No task tests them by
accident, so the named task pins each one.

1. A search under the Fine plan. The rows sit 0.25 s apart, so the fold must
   chain many rows into one moment, not leave three rows. (Task 1)
2. A search answer that holds two videos. A fold must never join the rows of one
   video to the rows of another. (Task 1)
3. A note among the ten best rows. A note holds no time, so a note must stay its
   own row. (Task 1)
4. A pause that a person presses twice. The first press must hold the job, and
   the second must let it run. A paused job must keep its place. (Task 2)
5. A pause that is still set when the job ends, and a stop that arrives during a
   pause. The page must not show a job that runs forever. (Task 2)
6. A store that holds a done mark with no records, from a fault between the two
   writes. The folder run must clear the mark, and a done video that holds
   records must keep its own. (Task 3)

---

### Task 1: The merged hit

**Files:**
- Modify: `videoembed/src/index.rs` (the constant block near line 36, `merge_hits` at 915-939, the test module at 972)
- Modify: `videoembed/src/main.rs` (`search` at 375-410, `hit_line` at 412-439, the test at 529)
- Test: the `mod tests` blocks of both files

**Interfaces:**
- Consumes: `Record`, `Hit`, `store::is_video`, `store::normalize_label`, `store::KIND_NOTE`, `merge_hits`.
- Produces:
  - `pub const MERGE_GAP: f32 = 0.5;`
  - `pub fn merge_hits_with(hits: Vec<Hit>, gap: f32) -> Vec<Hit>`
  - `pub struct Row { pub note: bool, pub label: String, pub start: f32, pub end: f32, pub score: f32, pub margin: f32 }`
  - `pub fn search_rows(records: &[Record], ranked: &[(usize, f32)], baseline: f32, cap: usize, gap: f32) -> Vec<Row>`
  - In `main.rs`: `const ROW_CAP: usize = 10;` and `fn rows_of(records, ranked, baseline) -> Vec<index::Row>`

- [ ] **Step 1: Write the failing tests for the fold**

Add to `videoembed/src/index.rs`, inside `mod tests`. Use the helpers that the
other tests of the module use.

```rust
    /// One record of a search, as the test writes it.
    fn searched(label: &str, kind: u8, start: f32, end: f32) -> Record {
        Record {
            rid: 0,
            id: store::record_id(label, start, end),
            kind,
            label: label.to_string(),
            start,
            end,
        }
    }

    /// The ten best rows of the window grid fold into one moment.
    #[test]
    fn two_touching_windows_of_one_video_join_one_row() {
        let records = vec![
            searched("a.mp4", crate::store::KIND_FRAME, 0.0, 2.0),
            searched("a.mp4", crate::store::KIND_FRAME, 2.0, 4.0),
        ];
        let ranked = vec![(0, 0.9), (1, 0.5)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 1, "the two windows make one moment");
        assert_eq!(rows[0].start, 0.0);
        assert_eq!(rows[0].end, 4.0);
        assert_eq!(rows[0].score, 0.9, "the moment keeps the best score");
    }

    /// Two frames that sit past the gap stay two moments.
    #[test]
    fn two_frames_past_the_gap_keep_two_rows() {
        let records = vec![
            searched("a.mp4", crate::store::KIND_FRAME, 0.0, 0.25),
            searched("a.mp4", crate::store::KIND_FRAME, 1.0, 1.25),
        ];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 2, "a gap of 0.75 s holds two moments");
    }

    /// The rows of two videos never join, whatever their times.
    #[test]
    fn a_row_of_another_video_keeps_its_own_row() {
        let records = vec![
            searched("a.mp4", crate::store::KIND_FRAME, 0.0, 2.0),
            searched("b.mp4", crate::store::KIND_FRAME, 2.0, 4.0),
        ];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 2, "the two videos keep one row each");
    }

    /// A note holds no time, so a note never joins a video row.
    #[test]
    fn a_note_never_joins_a_row() {
        let records = vec![
            searched("a.mp4", crate::store::KIND_FRAME, 0.0, 2.0),
            searched("blue", crate::store::KIND_NOTE, 0.0, 0.0),
        ];
        let ranked = vec![(0, 0.9), (1, 0.85)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 2, "the note stands alone");
        assert!(rows.iter().any(|row| row.note), "the note is marked");
    }

    /// The cap counts the rows that enter the fold.
    #[test]
    fn the_cap_counts_the_rows_before_the_fold() {
        let records: Vec<Record> = (0..12)
            .map(|at| searched("a.mp4", crate::store::KIND_FRAME, at as f32, at as f32 + 0.25))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..12).map(|at| (at, 0.9 - at as f32 / 100.0)).collect();

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 10, "a gap of 0.75 s keeps every kept row");
    }

    /// Ten rows that sit near each other leave one moment.
    #[test]
    fn ten_near_rows_leave_one_row() {
        let records: Vec<Record> = (0..12)
            .map(|at| searched("a.mp4", crate::store::KIND_FRAME, at as f32 * 0.2, at as f32 * 0.2 + 0.25))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..12).map(|at| (at, 0.9 - at as f32 / 100.0)).collect();

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 1, "the ten rows chain into one moment");
        assert_eq!(rows[0].score, 0.9, "the moment keeps the best score");
    }

    /// The folded rows come back best first.
    #[test]
    fn the_folded_rows_come_back_best_first() {
        let records = vec![
            searched("a.mp4", crate::store::KIND_FRAME, 0.0, 0.25),
            searched("b.mp4", crate::store::KIND_FRAME, 9.0, 9.25),
        ];
        let ranked = vec![(1, 0.95), (0, 0.6)];

        let rows = search_rows(&records, &ranked, 0.5, 10, MERGE_GAP);

        assert_eq!(rows[0].label, "b.mp4");
        assert_eq!(rows[0].margin, 0.45, "the margin follows the baseline");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib`
Expected: FAIL with `cannot find function search_rows in this scope` and
`cannot find value MERGE_GAP in this scope`.

- [ ] **Step 3: Write the constant and the gap fold**

In `videoembed/src/index.rs`, after the `MOMENT_GAP` line (line 36), add:

```rust
/// The distance in seconds below which two rows of one video join one moment.
///
/// Google folds hits within 500 ms into one interval
/// (`VideoMomentFinderViewModel.kt:1020-1036`). A search shows a moment, so
/// three near hits read as one row.
pub const MERGE_GAP: f32 = 0.5;
```

Then replace the body of `merge_hits` (line 915) with a gap parameter, and keep
`merge_hits` as the wrapper that the moment fold of one video uses:

```rust
/// Fold the hits of one video that sit near each other into moments.
///
/// The hits arrive in any order. A hit that starts within `gap` of the end of
/// the moment before it joins that moment. The moment runs from the first start
/// to the last end, and it keeps the best score. The answer comes back best
/// first.
pub fn merge_hits_with(hits: Vec<Hit>, gap: f32) -> Vec<Hit> {
    if hits.len() < 2 {
        return hits;
    }
    let mut by_time = hits;
    by_time.sort_by(|a, b| a.start.total_cmp(&b.start));

    let mut moments: Vec<Hit> = Vec::new();
    for hit in by_time {
        match moments.last_mut() {
            Some(last) if hit.start - last.end <= gap => {
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

/// Fold the touching hits of one video into moments.
pub fn merge_hits(hits: Vec<Hit>) -> Vec<Hit> {
    merge_hits_with(hits, MOMENT_GAP)
}
```

- [ ] **Step 4: Write the row and the search fold**

Add to `videoembed/src/index.rs`, after the `Group` struct (line 128):

```rust
/// One row of a search answer.
///
/// The row holds a whole moment after the fold, so the range is the range the
/// user reads. A note is a row too. A note holds no time, so a note never folds.
#[derive(Clone, Debug)]
pub struct Row {
    pub note: bool,
    pub label: String,
    pub start: f32,
    pub end: f32,
    pub score: f32,
    pub margin: f32,
}
```

Add the fold, after `group_hits`:

```rust
/// The rows of a search answer, best first, with the near rows of one video
/// joined into one moment.
///
/// The list is cut to `cap` rows first, because one merged moment counts as one
/// row of the cap. Then the rows of one video fold when they sit within `gap` of
/// each other. A note holds no time, so a note stays its own row.
pub fn search_rows(
    records: &[Record],
    ranked: &[(usize, f32)],
    baseline: f32,
    cap: usize,
    gap: f32,
) -> Vec<Row> {
    let mut videos: Vec<(String, Hit)> = Vec::new();
    let mut notes: Vec<Row> = Vec::new();
    for (index, score) in ranked.iter().take(cap) {
        let Some(record) = records.get(*index) else {
            continue;
        };
        if !crate::store::is_video(record.kind) {
            notes.push(Row {
                note: true,
                label: record.label.clone(),
                start: record.start,
                end: record.end,
                score: *score,
                margin: *score - baseline,
            });
            continue;
        }
        videos.push((
            crate::store::normalize_label(&record.label),
            Hit {
                start: record.start,
                end: record.end,
                score: *score,
                margin: *score - baseline,
            },
        ));
    }

    // One video at a time, because a moment belongs to one video.
    let mut labels: Vec<String> = videos.iter().map(|(label, _)| label.clone()).collect();
    labels.sort();
    labels.dedup();
    let mut rows: Vec<Row> = Vec::new();
    for label in labels {
        let hits: Vec<Hit> = videos
            .iter()
            .filter(|(held, _)| *held == label)
            .map(|(_, hit)| hit.clone())
            .collect();
        for hit in merge_hits_with(hits, gap) {
            rows.push(Row {
                note: false,
                label: label.clone(),
                start: hit.start,
                end: hit.end,
                score: hit.score,
                margin: hit.margin,
            });
        }
    }
    rows.extend(notes);
    rows.sort_by(|a, b| b.score.total_cmp(&a.score));
    rows
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib`
Expected: PASS. The seven new tests pass, and the older moment tests
(`touching_hits_join_one_moment`, `a_gap_of_two_frame_steps_keeps_two_moments`)
still pass.

- [ ] **Step 6: Write the failing test for the command line rows**

In `videoembed/src/main.rs`, in `mod tests`, replace
`a_row_below_the_cut_is_marked_weak` with the two tests below. The row now
carries its own score and margin, so the line takes a row.

```rust
    /// One row of a search, as the test writes it.
    fn row(label: &str, start: f32, end: f32, score: f32, note: bool) -> index::Row {
        index::Row {
            note,
            label: label.to_string(),
            start,
            end,
            score,
            margin: score - 0.5,
        }
    }

    /// A row below the cut reads as weak.
    #[test]
    fn a_row_below_the_cut_is_marked_weak() {
        let cut = 0.7;

        let below = hit_line(0, &row("a.mp4", 0.0, 4.0, 0.55, false), cut);
        let above = hit_line(0, &row("a.mp4", 0.0, 4.0, 0.9, false), cut);

        assert!(below.ends_with(" weak"), "got {below}");
        assert!(!above.ends_with(" weak"), "got {above}");
    }

    /// One line shows the range of the whole moment.
    #[test]
    fn the_line_shows_the_range_of_the_whole_moment() {
        let line = hit_line(0, &row("a.mp4", 0.0, 4.0, 0.9, false), 0.7);

        assert!(line.contains("[0.00-4.00 s]"), "got {line}");
    }

    /// A search of the window grid shows one row for a near pair.
    #[test]
    fn a_search_shows_one_row_for_a_near_pair() {
        let records = vec![
            videoembed::store::Record {
                rid: 0,
                id: "a.mp4/0-2000".to_string(),
                kind: videoembed::store::KIND_FRAME,
                label: "a.mp4".to_string(),
                start: 0.0,
                end: 2.0,
            },
            videoembed::store::Record {
                rid: 1,
                id: "a.mp4/2000-4000".to_string(),
                kind: videoembed::store::KIND_FRAME,
                label: "a.mp4".to_string(),
                start: 2.0,
                end: 4.0,
            },
        ];
        let ranked = vec![(0, 0.9), (1, 0.85)];

        let rows = rows_of(&records, &ranked, 0.0);

        assert_eq!(rows.len(), 1, "the pair joins the cap of ten");
        assert_eq!(rows[0].end, 4.0, "the moment runs to the last end");
    }
```

- [ ] **Step 7: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin videoembed`
Expected: FAIL. `hit_line` takes five arguments, and `rows_of` does not exist.

- [ ] **Step 8: Change the search to print moments**

In `videoembed/src/main.rs`, add the cap beside the other constants:

```rust
/// The number of rows a search shows.
const ROW_CAP: usize = 10;
```

Replace the print loop of `search` (line 400) with:

```rust
    for (rank, row) in rows_of(&store.records, &ranked, baseline).iter().enumerate() {
        println!("{}", hit_line(rank, row, cut));
    }
```

Add the helper after `search`:

```rust
/// The rows a search shows: the ten best, folded into moments.
fn rows_of(
    records: &[videoembed::store::Record],
    ranked: &[(usize, f32)],
    baseline: f32,
) -> Vec<index::Row> {
    index::search_rows(records, ranked, baseline, ROW_CAP, index::MERGE_GAP)
}
```

Replace `hit_line` with:

```rust
/// One line of a search answer.
///
/// The row holds the whole moment, so the line shows the range of the moment.
fn hit_line(rank: usize, row: &index::Row, cut: f32) -> String {
    let mark = if index::is_weak(row.score, cut) {
        " weak"
    } else {
        ""
    };
    if row.note {
        return format!(
            "{:2}. {:.3} ({:+.3}) {} note: {}{}",
            rank + 1,
            row.score,
            row.margin,
            bar(row.score),
            row.label,
            mark
        );
    }
    format!(
        "{:2}. {:.3} ({:+.3}) {} {}  [{:.2}-{:.2} s]{}",
        rank + 1,
        row.score,
        row.margin,
        bar(row.score),
        row.label,
        row.start,
        row.end,
        mark
    )
}
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS for every test binary.

- [ ] **Step 10: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/main.rs
git commit -m "feat: fold the near rows of a search into one moment

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: The pause

**Files:**
- Modify: `videoembed/src/index.rs` (`Progress` at 74-98)
- Modify: `videoembed/src/bin/web.rs` (`Job` 42-76, `Stopping` 203-206, `engine_loop` 541-633, `handle` 658-713, `start_index` 751-801, `step_job` 870-983, `api` route table, `main` 1324-1336, `mod tests` 1393)
- Modify: `videoembed/assets/index.html` (the folder section at 10-34)
- Modify: `videoembed/assets/app.js` (`showProgress` 256-286, `boot` 507-524)

**Interfaces:**
- Consumes: `Progress`, `JobError`, `WebJob`, the `stopped: Arc<AtomicBool>` flag.
- Produces:
  - `Progress.paused: bool`
  - `Job::SetPause { answer: Sender<Result<bool, JobError>> }`
  - `fn toggle_pause(running: bool, held: &Arc<AtomicBool>, progress: &Arc<Mutex<Progress>>) -> Result<bool, JobError>`
  - `fn hold_when_paused(flag: &Arc<AtomicBool>, progress: &Arc<Mutex<Progress>>) -> bool`
  - `fn end_job(progress: &Arc<Mutex<Progress>>)`
  - `#[derive(Serialize)] struct Paused { paused: bool }`
  - The route `POST /api/pause`, the page element `#pause`, and the page
    function `togglePause`

- [ ] **Step 1: Write the failing test for the progress answer**

Add to `videoembed/src/bin/web.rs`, inside `mod tests`:

```rust
    /// The page reads the paused state from the progress answer.
    ///
    /// A reloaded page takes the state from this answer, so the button reads
    /// Resume after the reload.
    #[test]
    fn the_progress_answer_carries_the_paused_state() {
        let stopped = Progress {
            running: true,
            paused: false,
            ..Progress::default()
        };
        let paused = Progress {
            running: true,
            paused: true,
            ..Progress::default()
        };

        let json = serde_json::to_string(&stopped).unwrap();
        assert!(json.contains("\"paused\":false"), "got {json}");
        let json = serde_json::to_string(&paused).unwrap();
        assert!(json.contains("\"paused\":true"), "got {json}");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin videoembed`
Expected: FAIL with `no field paused on type Progress`.

- [ ] **Step 3: Add the paused field**

In `videoembed/src/index.rs`, add to `Progress`, after `running`:

```rust
    /// True when the running job waits for a resume.
    pub paused: bool,
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin videoembed`
Expected: PASS.

- [ ] **Step 5: Write the failing tests for the pause control**

Add to `videoembed/src/bin/web.rs`, inside `mod tests`. The helper `flag` is a
clear pause flag, and the helper `board` is a running job.

```rust
    /// A pause request needs a running job.
    #[test]
    fn a_job_that_does_not_run_refuses_a_pause() {
        let flag = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Progress::default()));

        let outcome = toggle_pause(false, &flag, &board);

        assert!(matches!(outcome, Err(JobError::Message(_))), "got {outcome:?}");
        assert!(!flag.load(Ordering::Relaxed), "the flag stays clear");
    }

    /// One call holds the job, and the next call lets it run again.
    #[test]
    fn a_pause_toggles_the_flag_and_the_board() {
        let flag = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Progress {
            running: true,
            ..Progress::default()
        }));

        let first = toggle_pause(true, &flag, &board).unwrap();
        assert!(first, "the first call holds the job");
        assert!(board.lock().unwrap().paused, "the page reads the state");

        let second = toggle_pause(true, &flag, &board).unwrap();
        assert!(!second, "the second call lets it run");
        assert!(!board.lock().unwrap().paused, "the page reads the state");
    }

    /// The gate holds a paused job, and it shows the state on the board.
    #[test]
    fn a_paused_job_waits_at_the_gate() {
        let flag = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Progress {
            running: true,
            ..Progress::default()
        }));

        assert!(!hold_when_paused(&flag, &board), "a clear flag holds nothing");
        assert!(!board.lock().unwrap().paused);

        flag.store(true, Ordering::Relaxed);
        assert!(hold_when_paused(&flag, &board), "a set flag holds the job");
        assert!(board.lock().unwrap().paused, "the page reads the state");
    }

    /// The end of a job clears the pause with it.
    ///
    /// A pause that outlives its job leaves an item that advances on no item,
    /// and the page shows a job that runs forever.
    #[test]
    fn the_end_of_a_job_clears_the_pause() {
        let board = Arc::new(Mutex::new(Progress {
            running: true,
            paused: true,
            ..Progress::default()
        }));

        end_job(&board);

        let held = board.lock().unwrap();
        assert!(!held.running, "the job ends");
        assert!(!held.paused, "the pause goes with it");
    }
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin videoembed`
Expected: FAIL with `cannot find function toggle_pause`, `hold_when_paused`, and
`end_job` in this scope.

- [ ] **Step 7: Write the pause control**

In `videoembed/src/bin/web.rs`, add these three functions near `fail` (line 999):

```rust
/// Hold or release the running job. A job that does not run refuses the call.
///
/// The page sends one call for each press. The flag holds the state, and the
/// board carries the state to the page.
fn toggle_pause(
    running: bool,
    held: &Arc<AtomicBool>,
    progress: &Arc<Mutex<Progress>>,
) -> Result<bool, JobError> {
    if !running {
        return Err(JobError::Message("no job runs".to_string()));
    }
    let next = !held.load(Ordering::Relaxed);
    held.store(next, Ordering::Relaxed);
    progress.lock().unwrap().paused = next;
    Ok(next)
}

/// True when the pause holds the job.
///
/// A paused job keeps its place and its cut files. The caller returns without
/// work, and the loop waits for the next request.
fn hold_when_paused(flag: &Arc<AtomicBool>, progress: &Arc<Mutex<Progress>>) -> bool {
    if !flag.load(Ordering::Relaxed) {
        return false;
    }
    progress.lock().unwrap().paused = true;
    true
}

/// End the running job, and clear its pause.
fn end_job(progress: &Arc<Mutex<Progress>>) {
    let mut held = progress.lock().unwrap();
    held.running = false;
    held.paused = false;
}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin videoembed`
Expected: PASS.

- [ ] **Step 9: Wire the pause into the engine**

In `videoembed/src/bin/web.rs`:

1. Add the job variant to `Job`, after `StopIndex`:

```rust
    /// Hold or release the running index job. The answer names the new state.
    SetPause { answer: Sender<Result<bool, JobError>> },
```

2. Add the answer struct beside `Stopping`:

```rust
#[derive(Serialize)]
struct Paused {
    paused: bool,
}
```

3. Give `step_job` the flag. Add the parameter after `flag`:

```rust
    paused: &Arc<AtomicBool>,
```

and put the gate at the top of the body, before the stop check:

```rust
    if hold_when_paused(paused, progress) {
        // The job keeps its place and its cut files. The model waits on no job,
        // so the pause costs nothing after the current window ends.
        return false;
    }
```

4. Replace the three end transitions in `step_job`
(`progress.lock().unwrap().running = false;` at lines 884, 889, and 964) with
`end_job(progress);`.

5. Give `engine_loop` the flag. Replace the line that makes `stopped` (line 558)
with:

```rust
    // The pause holds the job between items, and the stop ends it.
    let stopped = Arc::new(AtomicBool::new(false));
```

and add the parameter `paused: Arc<AtomicBool>` after `jobs: Receiver<Job>`.

6. Change the loop head, so a paused job waits on the channel instead of
spinning:

```rust
    loop {
        let request = if pending.is_some() && !paused.load(Ordering::Relaxed) {
```

7. Add the arms to `handle`. Give `handle` the parameter `paused: &Arc<AtomicBool>`
after `stopped`, pass `&paused` at both call sites in `engine_loop`, and add:

```rust
        Job::SetPause { answer } => {
            let _ = answer.send(toggle_pause(pending.is_some(), paused, progress));
        }
```

8. Replace the `StopIndex` arm, so a stop lands during a pause:

```rust
        Job::StopIndex { answer } => {
            let running = pending.is_some();
            // A stop ends a pause too. A paused job never reads the stop flag.
            paused.store(false, Ordering::Relaxed);
            progress.lock().unwrap().paused = false;
            stopped.store(true, Ordering::Relaxed);
            let _ = answer.send(running);
        }
```

9. Pass the flag through `start_index` and `start_plan`, which clear the stop
flag. Add the parameter `paused: &Arc<AtomicBool>` after `flag` to each, and
clear the pause beside the stop clear:

```rust
    flag.store(false, Ordering::Relaxed);
    paused.store(false, Ordering::Relaxed);
```

10. Change the two `handle` call sites and the two starter calls to match the new
parameters, and add the flag to `step_job`'s call site.

11. In `main`, make the flag and give it to the engine thread:

```rust
    let paused = Arc::new(AtomicBool::new(false));
```

before the thread, then clone it beside `engine_progress`:

```rust
    let engine_paused = Arc::clone(&paused);
```

and pass it to `engine_loop` after `jobs_rx`:

```rust
        engine_loop(
            embedder,
            models,
            store,
            engine_progress,
            engine_paused,
            engine_key,
            jobs_rx,
        );
```

- [ ] **Step 10: Add the route**

In `videoembed/src/bin/web.rs`, in the `api` match, add the arm beside
`(true, "/api/stop")`:

```rust
        (true, "/api/pause") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::SetPause { answer });
            match wait.recv() {
                Ok(Ok(paused)) => request.respond(json_reply(&Paused { paused }))?,
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(400, &message))?
                }
                Ok(Err(JobError::Busy)) => {
                    request.respond(error_reply(409, "an index job already runs"))?
                }
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
```

- [ ] **Step 11: Add the button and its handler**

In `videoembed/assets/index.html`, after the stop button (line 22):

```html
    <button id="pause" type="button" disabled>Pause</button>
```

In `videoembed/assets/app.js`, add the handler after `stopIndex`:

```javascript
async function togglePause() {
  try {
    await api("/api/pause", {});
  } catch (err) {
    note(err.message);
  }
}
```

Add the listener in `boot`:

```javascript
  $("pause").addEventListener("click", togglePause);
```

In `showProgress`, add the button state after the device line, and the paused
text at the head of the running branch:

```javascript
  // A pause works on a running job only, and it names the next action.
  $("pause").disabled = !p.running;
  $("pause").textContent = p.paused ? "Resume" : "Pause";
```

```javascript
    if (p.running) {
      if (p.paused) {
        $("progress-text").textContent =
          "Paused. File " + p.file_index + "/" + p.file_count +
          ", item " + p.window + "/" + p.window_count + " - " + p.file;
        return;
      }
```

- [ ] **Step 12: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS for every test binary, and no warning.

- [ ] **Step 13: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/bin/web.rs videoembed/assets/app.js videoembed/assets/index.html
git commit -m "feat: pause and resume the running index job

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: The reconcile pass

**Files:**
- Modify: `videoembed/src/store.rs` (`is_done` and `mark_done` at 392-407, `mod tests`)
- Modify: `videoembed/src/bin/web.rs` (`start_index` at 751-801, the `StartIndex` arm at 690-692, `mod tests`)
- Test: the `mod tests` blocks of both files

**Interfaces:**
- Consumes: `Store.done: HashSet<String>`, `Store.holds_video(&self, path: &str) -> bool`, `Store.conn`, `index::list_videos`.
- Produces: `pub fn Store::reconcile(&mut self) -> Result<usize>`, and
  `start_index(store: &mut Store, progress, pending, flag, paused, folder)`

- [ ] **Step 1: Write the failing tests for the pass**

Add to `videoembed/src/store.rs`, inside `mod tests`, beside the other done tests:

```rust
    /// A mark with no records goes, and the pass counts it.
    #[test]
    fn a_done_video_with_no_records_loses_its_mark() {
        let dir = scratch("reconcile");
        let path = dir.join("index.db");
        let mut store = Store::open(&path).unwrap();
        store.mark_done("a.mp4").unwrap();

        let count = store.reconcile().unwrap();

        assert_eq!(count, 1, "one mark holds no record");
        assert!(!store.is_done("a.mp4"), "the mark goes");
        let again = Store::open(&path).unwrap();
        assert!(!again.is_done("a.mp4"), "the mark must reach the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A done video that holds records keeps its mark.
    #[test]
    fn a_done_video_with_records_keeps_its_mark() {
        let dir = scratch("reconcile_held");
        let path = dir.join("index.db");
        let mut store = Store::open(&path).unwrap();
        store.append(vec![new_record("a.mp4", 0.0, 4.0)]).unwrap();
        store.mark_done("a.mp4").unwrap();

        let count = store.reconcile().unwrap();

        assert_eq!(count, 0, "every mark holds a record");
        assert!(store.is_done("a.mp4"), "the mark stays");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The pass counts every stale mark, and it keeps the good ones.
    #[test]
    fn the_pass_counts_every_stale_mark() {
        let dir = scratch("reconcile_pair");
        let path = dir.join("index.db");
        let mut store = Store::open(&path).unwrap();
        store.append(vec![new_record("kept.mp4", 0.0, 4.0)]).unwrap();
        store.mark_done("kept.mp4").unwrap();
        store.mark_done("gone.mp4").unwrap();
        store.mark_done("also_gone.mp4").unwrap();

        let count = store.reconcile().unwrap();

        assert_eq!(count, 2, "two marks hold no record");
        assert!(store.is_done("kept.mp4"), "a held video keeps its mark");
        assert!(!store.is_done("gone.mp4"));
        assert!(!store.is_done("also_gone.mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib`
Expected: FAIL with `no method named reconcile found for struct Store`.

- [ ] **Step 3: Write the pass**

In `videoembed/src/store.rs`, after `mark_done` (line 407):

```rust
    /// Clear the done mark of every video that holds no records.
    ///
    /// A fault can land between the record write and the mark write, so a video
    /// can hold a mark with no records. Every later run then skips it. A run
    /// calls this pass at the start, and the caller reports the count.
    pub fn reconcile(&mut self) -> Result<usize> {
        let stale: Vec<String> = self
            .done
            .iter()
            .filter(|label| !self.holds_video(label))
            .cloned()
            .collect();
        if stale.is_empty() {
            return Ok(0);
        }

        let tx = self.conn.transaction().context("begin the reconcile")?;
        for label in &stale {
            tx.execute("update videos set done = 0 where path = ?1", [label])?;
        }
        tx.commit().context("commit the reconcile")?;

        for label in &stale {
            self.done.remove(label);
        }
        Ok(stale.len())
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib`
Expected: PASS, with the three new tests green.

- [ ] **Step 5: Write the failing test for the folder run**

Add to `videoembed/src/bin/web.rs`, inside `mod tests`:

```rust
    /// A folder run clears the mark of a video that holds no records.
    ///
    /// The mark would keep the video out of every later run, so the run must
    /// clear it before it walks the folder.
    #[test]
    fn a_folder_run_clears_the_mark_of_a_video_with_no_records() {
        let dir = std::env::temp_dir().join(format!("vei_web_reconcile_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("a.mp4");
        std::fs::write(&clip, b"not a video").unwrap();
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store.mark_done(&clip.to_string_lossy()).unwrap();

        let progress = Arc::new(Mutex::new(Progress::default()));
        let flag = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let mut pending = None;

        let count = start_index(&mut store, &progress, &mut pending, &flag, &paused, &dir).unwrap();

        assert_eq!(count, 1, "the folder holds one video");
        assert!(
            !store.is_done(&clip.to_string_lossy()),
            "the stale mark goes"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin videoembed`
Expected: FAIL with `this function takes 5 arguments but 6 arguments were supplied`.
This shows the run holds the mark, because the run never calls the pass.

- [ ] **Step 7: Call the pass from the folder run**

In `videoembed/src/bin/web.rs`, give `start_index` the store as its first
parameter, after `folder`'s checks and before the stop clear:

```rust
fn start_index(
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    pending: &mut Option<WebJob>,
    flag: &Arc<AtomicBool>,
    paused: &Arc<AtomicBool>,
    folder: &Path,
) -> Result<usize, JobError> {
```

Insert the pass after the `files.is_empty()` check:

```rust
    // A video can hold a mark with no records, because a fault lands between
    // the two writes. The pass clears those marks, so the run indexes them.
    let cleared = store.reconcile().map_err(JobError::from)?;
    if cleared > 0 {
        println!("Cleared {cleared} finished marks with no records.");
    }
```

Change the `StartIndex` arm of `handle` to pass the store:

```rust
        Job::StartIndex { folder, answer } => {
            let _ = answer.send(start_index(
                store, progress, pending, stopped, paused, &folder,
            ));
        }
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS for every test binary.

- [ ] **Step 9: Commit**

```bash
git add videoembed/src/store.rs videoembed/src/bin/web.rs
git commit -m "feat: clear a done mark that holds no records

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: The documents

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md` (the header)
- Modify: `docs/deferred-work.md`

**Interfaces:**
- Consumes: the delivered behavior of Tasks 1, 2, and 3.
- Produces: no code.

- [ ] **Step 1: Read the three documents**

Run: `head -40 README.md` and read the spec header and
`docs/deferred-work.md`. Find the part of the `README` that names the web page
and the part that names the search.

- [ ] **Step 2: Write the README lines**

Add these facts to the `README`, in the part that names the web page and the
search. Keep the voice of the lines around them.

```markdown
- A search shows one row for each moment. Two rows of one video that sit within
  0.5 s join one row, and the row shows the whole range.
- The Pause button holds the running job. The job keeps its place and its cut
  files. The button reads Resume while the job waits.
- A folder run clears the finished mark of a video that holds no records. The
  server prints the count when the count is not zero.
```

- [ ] **Step 3: Write the spec header**

In `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md`, replace the
line that names Plan C with:

```markdown
Plan C: `docs/superpowers/plans/2026-10-13-search-job.md`.
```

- [ ] **Step 4: Write the deferred-work rows**

In `docs/deferred-work.md`, remove any row that Task 1, 2, or 3 now delivers.
Add one row for the work that waits, in the table style of the file:

```markdown
| A pause that lands inside one window | The engine reads the pause flag between items, so a press waits for the window in the model. A finer check costs a flag read inside `IndexRun::step`. |
```

Renumber the table and its subsections, as the file does.

- [ ] **Step 5: Lint the prose**

Run:
`node "C:\Users\emir-\.agents\skills\asd-ste100\hooks\run-python.cjs" "C:\Users\emir-\.agents\skills\asd-ste100\scripts\ste-lint.py" --fail-over 2.5 README.md docs/deferred-work.md`
Expected: no file over 2.5 violations for each 100 words.

- [ ] **Step 6: Commit**

```bash
git add README.md docs
git commit -m "docs: record the moment, the pause, and the pass

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Notes for the executor

- Task 2 changes the signature of `handle`, `start_index`, `start_plan`, and
  `step_job`. The compiler names every call site. Work through them in one pass.
- Task 2 needs the button and the handler in the page. The page lives in the
  binary through `include_str!`, so the compiler does not check it. Read the
  changed part of `app.js` by eye, and check the element id against
  `index.html`.
- Task 3 depends on Task 2 for the `paused` parameter of `start_index`. Run Task
  2 before Task 3.
- A pause costs the run one window, because the model call inside `IndexRun::step`
  does not stop. The note of Task 4 records this.

