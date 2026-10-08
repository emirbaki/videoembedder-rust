# Fine-Only Window Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index every video with a 2 s window and a 1 s stride, so one search returns a 2 s answer range with no inference at query time.

**Architecture:** One change to a shared constant pair. `index::WINDOW` and `index::STRIDE` drive the window plan for the console, the web UI, and the indexer. Both binaries call `index::windows` and `index::index_video`, so the two constants are the only production change. The tests that pin the plan, the module doc, and the README change with them.

**Tech Stack:** Rust, `anyhow`, `serde`, `llama-cpp-2` (unchanged), `cargo test`.

**Spec:** None. This is a bounded change. The user approved the shape directly: fine only, 2 s window, 1 s stride. The evidence sits in `spike/08_window_sweep.py` (2 s windows rank as well as 4 s) and `spike/09_coarse_fine.py` (a fine pass lands inside the true block).

## Global Constraints

- Window length 2 s and stride 1 s. Copy the values verbatim: `WINDOW = 2.0`, `STRIDE = 1.0`.
- Keep the store format `VEI1` and the magic bytes. Do not add a version field.
- Keep `FINE_WINDOW`, `FINE_STRIDE`, `fine_windows`, and `refine_hit`. They stay for an on-demand refine to 1 s.
- Keep `EPS = 0.05`, `FLOOR = 0.4`, and `GROUP_CAP = 50`.
- Build command: `CMAKE_GENERATOR="Visual Studio 17 2022" CMAKE_BUILD_PARALLEL_LEVEL=8 cargo build`.
- Run every cargo command from the `videoembed` folder.

## Review Focus

- **A stale store.** A store written at the old 4 s shape stays valid to `Store::open`. The web UI calls `holds_video` and reports the video as indexed, so the client skips a re-index. Old 4 s records then mix with new 2 s records. A reasonable person expects one shape. The fix is an instruction to clear the store, plus a manual check in Task 2.
- **A short video.** A video below 2 s now yields no window, and `index` fails with `is shorter than one window`. A reasonable person expects a clear message, not a crash. Pinned by `a_video_shorter_than_one_window_has_no_plan`.
- **A long video and the group cap.** A 10 minute video at a 1 s stride makes about 600 windows. `group_hits` keeps at most 50 hits per video. A reasonable person expects the cap to hold. Pinned by `group_hits_caps_each_group_at_fifty`.
- **The end-of-video slack.** `EPS` is 0.05 s. With a 2 s window the slack is larger in relative terms. A video of exactly 2 s must still give one window. Pinned in the short-video test.
- **Store size.** The record count doubles. Search stays a flat dot product, and a 10 minute video adds about 300 records at 2 s, so the size stays small. No code change. A note only.

---

## Task 1: Move the window plan to 2 s and 1 s

**Files:**

- Modify: `videoembed/src/index.rs` (the two constants, the module doc, three tests)
- Modify: `README.md:84,91-92`

**Interfaces:**

- Consumes: nothing from an earlier task.
- Produces: `index::WINDOW = 2.0`, `index::STRIDE = 1.0`. `index::windows(duration) -> Vec<(f64, f64)>` keeps its signature. Every caller keeps working. Both `main.rs` and `bin/web.rs` read the constants through `index::windows`.

- [ ] **Step 1: Rewrite the three plan tests to the new shape**

In `videoembed/src/index.rs`, change the three tests that pin the window plan. Keep every other test as is.

```rust
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
```

- [ ] **Step 2: Run the tests and watch them fail**

Run from `videoembed`: `cargo test --lib`
Expected: FAIL. `windows_cover_a_thirty_second_video` reports left 14, right 29. `a_video_shorter_than_one_window_has_no_plan` fails on `windows(2.0)`, because a 4 s window gives no plan. `the_callback_numbering_matches_the_window_plan` reports left 14, right 29.

- [ ] **Step 3: Change the two constants and the module doc**

In `videoembed/src/index.rs`, set the constants. Keep the type and the comment shape.

```rust
/// Window length in seconds.
pub const WINDOW: f64 = 2.0;
/// Window step in seconds.
pub const STRIDE: f64 = 1.0;
```

Change the module doc at the top of the file to match.

```rust
//! The folder scan, the window plan, and the progress object.
//!
//! One window is 2 s long. The stride is 1 s, so windows overlap.
```

- [ ] **Step 4: Run the tests and watch them pass**

Run from `videoembed`: `cargo test --lib`
Expected: PASS. All lib tests green, 0 failed.

- [ ] **Step 5: Drop the stale "coarse" wording in the refine doc**

In `videoembed/src/index.rs`, the `refine_hit` doc says the query is "the same text the coarse rank uses". Change it to "the same text the rank uses", because the coarse pass is gone.

- [ ] **Step 6: Update the README**

In `README.md`, change line 84 to:

```text
| `index <path>` | Cut the video into 2 s windows on a 1 s stride. Embed each window. |
```

Change lines 91 and 92 to:

```text
The stride is 1 s, so the windows overlap. A hit at 20.0-22.0 s can start at
19 s or 21 s in the true scene.
```

- [ ] **Step 7: Run the whole suite**

Run from `videoembed`: `cargo test`
Expected: PASS. The lib tests and the web binary tests all green, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add videoembed/src/index.rs README.md
git commit -m "feat: index with a 2 s window and a 1 s stride"
```

---

## Task 2: Re-index and verify end to end

**Files:**

- Modify: none. This task runs the built binary and checks the output.
- Remove: `videoembed/store/index.vei` (the stale 4 s store).

**Interfaces:**

- Consumes: the constants and the plan from Task 1.
- Produces: a store at the 2 s shape, and the measured window counts that prove it.

- [ ] **Step 1: Clear the stale store**

Run from `videoembed`: `rm -f store/index.vei`
Expected: no output. The next index starts from an empty store. This step is required, because `holds_video` makes the web UI skip an already-indexed video.

- [ ] **Step 2: Build the binaries**

Run from `videoembed`: `CMAKE_GENERATOR="Visual Studio 17 2022" CMAKE_BUILD_PARALLEL_LEVEL=8 cargo build`
Expected: a clean build, and a fresh `target/debug/videoembed.exe` and `target/debug/web.exe`.

- [ ] **Step 3: Index the football clip**

Run from `videoembed`:

```bash
./target/debug/videoembed.exe index "C:/Users/emir-/Downloads/2026-10-07_biebrntionx/2026-10-07_biebrntionx_2107656533844828272_6lCwR1KHqzGZ6vJS.mp4"
```

Expected: `Indexed 8 windows`. The clip is 9.26 s, so the plan runs 0-2 through 7-9.

- [ ] **Step 4: Search the woman and the crying players**

Run from `videoembed`:

```bash
./target/debug/videoembed.exe search "football players crying"
./target/debug/videoembed.exe search "a woman"
```

Expected: the first query puts a window that starts at or after 4 s on top. The second puts a window that starts before 4 s on top. Every window spans 2 s.

- [ ] **Step 5: Index the 30 s test video and check the count**

Run from `videoembed`:

```bash
./target/debug/videoembed.exe index "../spike/testvideo_nolabels.mp4"
```

Expected: `Indexed 29 windows`, up from 14 at the old shape.

- [ ] **Step 6: Search the birthday block**

Run from `videoembed`:

```bash
./target/debug/videoembed.exe search "birthday party"
```

Expected: a window inside 20-30 s sits on top. With 1 s granularity the top start sits near 20 s.

- [ ] **Step 7: Record the outcome**

Write the four measured lines (the two index counts and the two top hits) into the task ledger. No commit. The build artifacts and the store are ignored by git.
