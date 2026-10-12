# Both grids for one video - implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One video holds a window grid and a fine grid at the same time, and
one search ranks both.

**Architecture:** The window plan gains a `fine` flag. The flag adds the fixed
0.25 s grid beside the window records. One run writes both sets. The search
already ranks every record kind in one list, so it gains only the kind on a hit.

**Tech Stack:** Rust with `anyhow`, `rusqlite` with `sqlite-vec`, a small web
server on `tiny_http`, and plain JavaScript with no build step.

**Spec:** `docs/superpowers/specs/2026-10-13-hybrid-grid-design.md`.

## Global Constraints

- Every cargo command runs from the repo root as
  `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`.
  A plain command fails to configure the vendored llama.cpp.
- `videoembed/assets/app.js` and `videoembed/assets/index.html` hold CRLF line
  endings. A test that scans them finds `"\n}"`, never `"\n}\n"`.
- The web test suite mirrors no route and no browser. A web test asserts on a
  pure helper or on the text of the `JS` and `PAGE` constants.
- Prose follows ASD-STE100. Code, identifiers, and command syntax do not.
- One name for one thing. The flag is called `fine` in Rust, in SQL, in JSON,
  and in the page.
- Every commit message ends with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

- A plan whose rate is not a whole divisor of 4 fps. The dense cut must serve
  both grids at whole steps, and a wrong step gives wrong frame times.
- A video with no audio track and the flag set. The frame plan and the window
  plan must both finish.
- A window that runs past the end of the fine grid.
- A first run on a store from before this change. The new column must arrive.
- The flag on the Fine preset. It must change nothing.

---

### Task 1: The plan flag

**Files:**
- Modify: `videoembed/src/plan.rs:70-123` and `:153-193`
- Test: `videoembed/src/plan.rs` (the `tests` module)

**Interfaces:**
- Produces: `Plan.fine: bool`, and the plan word `fine=on|off`.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `videoembed/src/plan.rs`:

```rust
    /// The fine key adds the grid and leaves the window values alone.
    #[test]
    fn the_fine_key_adds_the_grid_beside_the_window() {
        let base = Plan::of(Preset::Object);
        let both = base.set_words("fine=on").unwrap();
        assert!(both.fine);
        assert_eq!(both.window, base.window);
        assert_eq!(both.frames, base.frames);
        assert!(both.audio);
        let off = base.set_words("window=4 fine=off").unwrap();
        assert!(!off.fine);
        assert_eq!(off.window, 4.0);
        assert_eq!(off.stride, 2.0);
    }
```

In `a_preset_holds_its_values`, after `assert!(object.audio);`, add:

```rust
        assert!(!object.fine, "a preset holds no fine grid");
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib plan::`
Expected: FAIL to compile, `no field fine on type Plan`.

- [ ] **Step 3: Write the implementation**

In the `Plan` struct, after `pub audio: bool,`, add:

```rust
    /// The plan also cuts the 0.25 s fine grid.
    pub fine: bool,
```

In `Plan::of`, add `fine: false,` to each of the four preset literals.

In `set_words`, in the `match key` block after the `"audio"` arm, add:

```rust
                "fine" => plan.fine = parse_flag(value)?,
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib plan::`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib`
Expected: PASS, 157 tests.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/plan.rs
git commit -m "feat: add the fine flag to a plan

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: The store column

**Files:**
- Modify: `videoembed/src/db.rs:19-28` and `:85-99`
- Modify: `videoembed/src/store.rs:310-365`
- Test: `videoembed/src/store.rs` (the `tests` module)

**Interfaces:**
- Consumes: `Plan.fine` from Task 1.
- Produces: the store carries `fine` through `plan_of` and `set_plan`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module of `videoembed/src/store.rs`:

```rust
    /// The fine flag survives the store.
    #[test]
    fn the_fine_flag_round_trips_through_the_store() {
        let (dir, mut store) = opened("plan_fine");
        store
            .append(vec![new_record("a.mp4", 0.0, 0.25)])
            .unwrap();
        let mut plan = plan::Plan::of(plan::Preset::Object);
        plan.fine = true;
        store.set_plan("a.mp4", &plan).unwrap();

        let again = Store::open(&dir.join("index.db")).unwrap();
        let held = again.plan_of("a.mp4").unwrap().unwrap();
        assert!(held.fine, "the flag reads back");
        assert_eq!(held.window, plan.window);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A store from before this change gains the fine column.
    #[test]
    fn an_old_store_gains_the_fine_column() {
        let dir = scratch("old_store");
        let path = dir.join("index.db");
        {
            let store = Store::open(&path).unwrap();
            drop(store);
            // A store of the version before the flag holds no fine column.
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch("alter table videos drop column plan_fine")
                .unwrap();
        }
        let store = Store::open(&path).unwrap();
        let plan = plan::Plan::of(plan::Preset::Object);
        store.set_plan("a.mp4", &plan).unwrap();
        let held = store.plan_of("a.mp4").unwrap().unwrap();
        assert!(!held.fine, "the new column reads back");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib store::tests::the_fine_flag`
Expected: FAIL, `the flag reads back`.

- [ ] **Step 3: Write the implementation**

In `db.rs`, in the `videos` table of `SCHEMA`, after the `plan_audio` line, add:

```sql
  plan_fine    INTEGER NOT NULL DEFAULT 0,
```

Replace the body of `add_missing_columns` (db.rs:85). The current body reads
the column names and returns when it finds `plan_mode`. Replace it with:

```rust
fn add_missing_columns(conn: &Connection) -> Result<()> {
    let mut names: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare("pragma table_info(videos)")?;
        let mut rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows.by_ref() {
            names.push(row?);
        }
    }
    if !names.iter().any(|name| name == "plan_mode") {
        conn.execute_batch("alter table videos add column plan_mode TEXT NOT NULL DEFAULT 'fine'")
            .context("add the plan_mode column")?;
    }
    if !names.iter().any(|name| name == "plan_fine") {
        conn.execute_batch("alter table videos add column plan_fine INTEGER NOT NULL DEFAULT 0")
            .context("add the plan_fine column")?;
    }
    Ok(())
}
```

Keep the call site. The old early `return Ok(())` goes away, so each column
gets its own check.

In `store.rs`, in `plan_of`, change the select and the row read:

```rust
                "select plan_mode, plan_window, plan_stride, plan_frames, plan_audio,
                        plan_fine
                 from videos where path = ?1",
                [&want],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
```

and the binding:

```rust
        let Some((mode, window, stride, frames, audio, fine)) = row else {
            return Ok(None);
        };
        let mode = plan::Mode::parse(&mode).unwrap_or(plan::Mode::Fine);
        Ok(Some(Plan {
            mode,
            window,
            stride,
            frames: frames as usize,
            audio: audio != 0,
            fine: fine != 0,
        }))
```

In `set_plan`, add `plan_fine` to the column list, the values list, the
`do update set` list, and the parameters:

```rust
                "insert into videos(path, plan_mode, plan_window, plan_stride, plan_frames,
                                    plan_audio, plan_fine)
                 values (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 on conflict(path) do update set
                   plan_mode = excluded.plan_mode,
                   plan_window = excluded.plan_window,
                   plan_stride = excluded.plan_stride,
                   plan_frames = excluded.plan_frames,
                   plan_audio = excluded.plan_audio,
                   plan_fine = excluded.plan_fine",
                params![
                    want,
                    plan.mode.as_str(),
                    plan.window,
                    plan.stride,
                    plan.frames as i64,
                    plan.audio as i64,
                    plan.fine as i64
                ],
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib store::`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib`
Expected: PASS, 158 tests.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/db.rs videoembed/src/store.rs
git commit -m "feat: store the fine flag with the plan

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: One cut, two grids

**Files:**
- Modify: `videoembed/src/index.rs:219-238`, `:636-655`, `:659-681`
- Test: `videoembed/src/index.rs` (the `tests` module)

**Interfaces:**
- Consumes: `Plan.fine` from Task 1.
- Produces:
  - `pub fn dense_rate(plan_rate: f64) -> f64`
  - `pub fn frames_in_dense(cut: &[PathBuf], from: f64, window: f64, rate: f64, cut_rate: f64) -> Vec<PathBuf>`
  - `pub fn window_times_dense(from: f64, rate: f64, frames: usize, clips: usize, cut_rate: f64) -> Vec<f64>`
  - `IndexRun::cut_rate(&self) -> f64`

`frames_in` and `window_times` keep their names and their meaning. They call the
dense forms with `cut_rate` equal to `rate`, so every caller outside this file
stands.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module of `videoembed/src/index.rs`:

```rust
    /// The dense cut serves both rates at whole steps.
    #[test]
    fn the_dense_cut_serves_both_rates() {
        assert_eq!(dense_rate(1.0), 4.0);
        assert_eq!(dense_rate(3.0), 12.0);
        for rate in [0.5, 1.0, 1.5, 2.5, 4.0, 10.0] {
            let cut = dense_rate(rate);
            let plan_step = cut / rate;
            let fine_step = cut / FRAME_RATE;
            assert!(
                (plan_step - plan_step.round()).abs() < 1e-9,
                "the window step is whole at {rate} fps"
            );
            assert!(
                (fine_step - fine_step.round()).abs() < 1e-9,
                "the fine step is whole at {rate} fps"
            );
        }
    }

    /// A dense cut gives the window the frames of its own rate.
    #[test]
    fn a_dense_cut_gives_the_window_its_own_frames() {
        // Four frames a second, so index `n` sits at `n / 4` s.
        let cut: Vec<PathBuf> = (0..16).map(|n| PathBuf::from(format!("f{n}.jpg"))).collect();
        // A 2 s window at 1 fps wants 2 frames, at 0 s and 1 s.
        let files = frames_in_dense(&cut, 0.0, 2.0, 1.0, 4.0);
        assert_eq!(files, vec![PathBuf::from("f0.jpg"), PathBuf::from("f4.jpg")]);
        let times = window_times_dense(0.0, 1.0, files.len(), 0, 4.0);
        assert_eq!(times, vec![0.0, 1.0]);
        // The second window starts at 1 s, on its own frame.
        let next = frames_in_dense(&cut, 1.0, 2.0, 1.0, 4.0);
        assert_eq!(next[0], PathBuf::from("f4.jpg"));
        // A window past the end of the cut takes what is left, and no more.
        let left = frames_in_dense(&cut, 3.0, 2.0, 1.0, 4.0);
        assert_eq!(left, vec![PathBuf::from("f12.jpg")]);
        assert!(frames_in_dense(&cut, 40.0, 2.0, 1.0, 4.0).is_empty());
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::the_dense_cut`
Expected: FAIL to compile, `cannot find function dense_rate`.

- [ ] **Step 3: Write the implementation**

Replace `frames_in` (index.rs:219) with the pair:

```rust
/// The frames of one span, from the list of a whole-video cut.
///
/// Frame `i` of the list sits at `i / rate`, and the span `[from, from + window)`
/// takes the frames inside it. A span that runs past the list takes what is left.
pub fn frames_in(cut: &[PathBuf], from: f64, window: f64, rate: f64) -> Vec<PathBuf> {
    frames_in_dense(cut, from, window, rate, rate)
}

/// The frames of one span, from a cut that holds the fine grid too.
///
/// A plan with the fine flag cuts at a rate above its own, so the window path
/// takes every `cut_rate / rate` frame of the span. The frames sit at the times
/// of its own rate, and the window keeps their count.
pub fn frames_in_dense(
    cut: &[PathBuf],
    from: f64,
    window: f64,
    rate: f64,
    cut_rate: f64,
) -> Vec<PathBuf> {
    let first = first_frame(from, cut_rate);
    let end = last_frame(from, window, cut_rate).min(cut.len());
    if first >= end {
        return Vec::new();
    }
    let step = (cut_rate / rate).round().max(1.0) as usize;
    cut[first..end].iter().step_by(step).cloned().collect()
}
```

Replace `window_times` (index.rs:233) with the pair:

```rust
/// The time of every media part of one window.
///
/// The frames come first, then the clips, in the order the embedder reads them.
/// A frame sits at its own place in the cut, so these times match the files
/// that `frames_in` gives. A clip sits at the second that it covers.
pub fn window_times(from: f64, rate: f64, frames: usize, clips: usize) -> Vec<f64> {
    window_times_dense(from, rate, frames, clips, rate)
}

/// The time of every part of one window, from a cut that holds the fine grid.
///
/// The picked frames sit at the times of the plan rate, and the cut rate names
/// the place of each one in the dense cut.
pub fn window_times_dense(
    from: f64,
    rate: f64,
    frames: usize,
    clips: usize,
    cut_rate: f64,
) -> Vec<f64> {
    let first = first_frame(from, cut_rate);
    let step = (cut_rate / rate).round().max(1.0) as usize;
    let mut times: Vec<f64> = (0..frames)
        .map(|i| (first + i * step) as f64 / cut_rate)
        .collect();
    times.extend((0..clips).map(|k| from.floor() + k as f64));
    times
}
```

Add `dense_rate` before `first_frame`:

```rust
/// The rate of a cut that holds the window grid and the fine grid.
///
/// The cut is a whole multiple of the plan rate and of 4 fps, so both paths
/// take whole steps over it. A plan at 1 fps gives 4 fps. A plan at 3 fps
/// gives 12 fps.
pub fn dense_rate(plan_rate: f64) -> f64 {
    let base = plan_rate.max(FRAME_RATE);
    for n in 1..=64u32 {
        let rate = base * f64::from(n);
        let plan_step = rate / plan_rate;
        let fine_step = rate / FRAME_RATE;
        if (plan_step - plan_step.round()).abs() < 1e-9
            && (fine_step - fine_step.round()).abs() < 1e-9
        {
            return rate;
        }
    }
    base
}
```

Add `cut_rate` to `impl IndexRun`, near `window_group` (index.rs:636):

```rust
    /// The rate of the whole-video cut.
    ///
    /// The cut holds the fine grid when the plan asks for it, so it runs at a
    /// rate that serves both paths at whole steps.
    fn cut_rate(&self) -> f64 {
        match self.plan.mode {
            crate::plan::Mode::Window if self.plan.fine => dense_rate(self.plan.frame_rate()),
            crate::plan::Mode::Window => self.plan.frame_rate(),
            crate::plan::Mode::Fine => FRAME_RATE,
        }
    }
```

In `window_group` (index.rs:636), change the first two lines and the two
helper calls:

```rust
    fn window_group(&self) -> Vec<WindowPart> {
        let rate = self.plan.frame_rate();
        let cut_rate = self.cut_rate();
        let end = (self.window_at + N_WINDOW_BATCH).min(self.windows.len());
        (self.window_at..end)
            .map(|at| {
                let from = self.windows[at];
                let mut files =
                    frames_in_dense(&self.frames_cut, from, self.plan.window, rate, cut_rate);
                let frames = files.len();
                let clips = clips_in(&self.clips_cut, from, self.plan.window);
                let times = window_times_dense(from, rate, frames, clips.len(), cut_rate);
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

In `cut_media` (index.rs:659), replace the rate match with the one call:

```rust
        let rate = self.cut_rate();
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS. The window path of a plan without the flag cuts and picks
exactly as before, so every existing test stands.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: cut one dense grid for both paths

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: One index pass

**Files:**
- Modify: `videoembed/src/index.rs:424-470`, `:483-500`, `:600-625`, `:683-745`
- Test: `videoembed/src/index.rs` (the `tests` module)

**Interfaces:**
- Consumes: `IndexRun::cut_rate` from Task 3, and `Plan.fine` from Task 1.
- Produces: a run with the flag writes the window records and the fine records.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module of `videoembed/src/index.rs`:

```rust
    /// A plan with the flag runs the fine grid and then the windows.
    #[test]
    fn the_flag_runs_the_fine_grid_and_the_windows() {
        let mut plan = plan::Plan::of(plan::Preset::Object);
        plan.fine = true;
        let mut run = IndexRun {
            source: PathBuf::from("a.mp4"),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: "a.mp4".to_string(),
            duration: 2.0,
            frames: vec![0.0, 0.25],
            audios: vec![0.0],
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan,
            windows: vec![0.0],
            group: Vec::new(),
            notes: Vec::new(),
        };

        assert_eq!(run.total(), 4, "two frames, one clip, one window");
        run.advance();
        run.advance();
        assert!(run.phase == Phase::Audio, "the grid comes before the windows");
        run.advance();
        assert!(run.phase == Phase::Windows, "the windows come last");
        run.advance();
        assert!(run.phase == Phase::Windows, "a window waits for its group");
    }

    /// A grid with no audio track hands its video straight to the windows.
    #[test]
    fn a_grid_with_no_audio_goes_to_its_windows() {
        let mut plan = plan::Plan::of(plan::Preset::Object);
        plan.fine = true;
        let mut run = IndexRun {
            source: PathBuf::from("a.mp4"),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: "a.mp4".to_string(),
            duration: 2.0,
            frames: vec![0.0],
            audios: Vec::new(),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan,
            windows: vec![0.0],
            group: Vec::new(),
            notes: Vec::new(),
        };

        assert_eq!(run.total(), 2, "one frame and one window");
        run.advance();
        assert!(
            run.phase == Phase::Windows,
            "no audio track, so the grid hands over"
        );
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::the_flag_runs`
Expected: FAIL, `two frames, one clip, one window` reads 3.

- [ ] **Step 3: Write the implementation**

In `total()` (index.rs:468), replace the body with:

```rust
    /// The number of records this run will make.
    pub fn total(&self) -> usize {
        self.frames.len() + self.audios.len() + self.windows.len()
    }
```

In `start_window` (index.rs:434), after the `windows.is_empty()` check that
ends at line 443, add:

```rust
        // The flag adds the fine grid beside the windows. The grid follows
        // each stream length, as the fine mode does.
        let (frames, audios) = if plan.fine {
            let video_len = media::probe_stream_duration(source, "v")?
                .unwrap_or(duration)
                .min(duration);
            let audio_len = if media::has_audio(source) {
                media::probe_stream_duration(source, "a")?
                    .unwrap_or(duration)
                    .min(duration)
            } else {
                0.0
            };
            (frames(video_len), audio_starts(audio_len))
        } else {
            (Vec::new(), Vec::new())
        };
```

In the struct literal of `start_window` (index.rs:450-454), replace

```rust
            frames: Vec::new(),
            audios: Vec::new(),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Windows,
```

with

```rust
            frames,
            audios,
            frame_at: 0,
            audio_at: 0,
            phase: if plan.fine { Phase::Frames } else { Phase::Windows },
```

Add the phase helper to `impl IndexRun`, before `advance` (index.rs:609):

```rust
    /// The phase after the fine grid of a video.
    ///
    /// A window plan with the flag runs its windows after the grid. A fine
    /// plan has no window, so its run ends.
    fn after_fine(&self) -> Phase {
        if self.windows.is_empty() {
            Phase::Done
        } else {
            Phase::Windows
        }
    }
```

In `advance` (index.rs:609), replace the `Phase::Frames` and the `Phase::Audio`
arms:

```rust
            Phase::Frames => {
                self.frame_at += 1;
                if self.frame_at >= self.frames.len() {
                    // A video with no audio track goes to its windows.
                    self.phase = if self.audios.is_empty() {
                        self.after_fine()
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
                    self.phase = self.after_fine();
                }
            }
```

In `cut_batch` (index.rs:686), replace the whole function with:

```rust
    fn cut_batch(&mut self) -> Result<()> {
        self.files.clear();
        if self.phase == Phase::Done {
            return Ok(());
        }
        if self.starts().is_empty() {
            // The fine stream of this video is over. The windows may still run.
            self.phase = self.after_fine();
            return Ok(());
        }
        self.cut_media()?;
        if self.phase == Phase::Windows {
            let group = self.window_group();
            // The plan asks for a window that the cut does not hold. A window
            // with no media has no vector, so the run reports the first one.
            let empty = group
                .iter()
                .find(|part| part.files.is_empty())
                .map(|part| part.start)
                .or_else(|| {
                    group
                        .is_empty()
                        .then(|| self.windows.get(self.window_at).copied().unwrap_or_default())
                });
            if let Some(start) = empty {
                bail!("no media came from the window at {start} s");
            }
            self.group = group;
            return Ok(());
        }
        // Fine mode slices the whole-video cut by item, because one frame sits
        // at one fine step and one clip at one audio second.
        let at = self.at();
        if self.phase == Phase::Frames {
            // A dense cut holds more frames than the fine grid wants, so the
            // grid takes every `cut_rate / 4` frame.
            let step = (self.cut_rate() / FRAME_RATE).round().max(1.0) as usize;
            let end = (at + FRAME_BATCH).min(self.frames.len());
            if at >= end {
                self.phase = if self.audios.is_empty() {
                    self.after_fine()
                } else {
                    Phase::Audio
                };
                self.audio_at = 0;
                self.files.clear();
                self.batch_from = 0;
                return Ok(());
            }
            self.files = self
                .frames_cut
                .iter()
                .step_by(step)
                .skip(at)
                .take(end - at)
                .cloned()
                .collect();
            self.batch_from = at;
            return Ok(());
        }
        let end = (at + AUDIO_BATCH).min(self.clips_cut.len());
        if at >= end {
            self.phase = self.after_fine();
            self.audio_at = 0;
            self.files.clear();
            self.batch_from = 0;
            return Ok(());
        }
        self.files = self.clips_cut[at..end].to_vec();
        self.batch_from = at;
        Ok(())
    }
```

`FRAME_BATCH` and `AUDIO_BATCH` keep their values. Where the old code chose
`count` and `whole` by phase, the two branches name their own list now.

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: write both grids in one index pass

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: The kind on a hit

**Files:**
- Modify: `videoembed/src/index.rs:106-146`, `:863-868`, `:935-994`, `:1011-1018`
- Modify: `videoembed/src/main.rs:567-576`
- Test: `videoembed/src/index.rs` (the `tests` module)

**Interfaces:**
- Consumes: the window record and the frame record of one video.
- Produces: `Hit.kind: u8` and `Row.kind: u8`, holding `store::KIND_WINDOW`,
  `store::KIND_FRAME`, `store::KIND_AUDIO`, or `store::KIND_NOTE`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module of `videoembed/src/index.rs`:

```rust
    /// A hit names the kind of its record.
    #[test]
    fn a_hit_names_its_kind() {
        let records = vec![
            record_of(crate::store::KIND_WINDOW, "a.mp4", 10.0, 12.0),
            record_of(crate::store::KIND_FRAME, "a.mp4", 10.25, 10.5),
        ];
        let ranked = vec![(0usize, 0.9f32), (1usize, 0.5f32)];
        let rows = search_rows(&records, &ranked, 0.2, 10, MERGE_GAP);
        assert_eq!(rows.len(), 2, "the two sit apart, so neither folds");
        let kinds: Vec<u8> = rows.iter().map(|row| row.kind).collect();
        assert!(kinds.contains(&crate::store::KIND_WINDOW));
        assert!(kinds.contains(&crate::store::KIND_FRAME));
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::tests::a_hit_names_its_kind`
Expected: FAIL to compile, `no field kind`.

- [ ] **Step 3: Write the implementation**

In `Hit` (index.rs:106), after `pub margin: f32,`, add:

```rust
    /// The record kind of this hit, from `store::KIND_*`.
    pub kind: u8,
```

In `Row` (index.rs:139), after `pub note: bool,`, add:

```rust
    /// The record kind of this row, from `store::KIND_*`.
    pub kind: u8,
```

In `group_hits` (index.rs:863), add the kind to the `Hit` literal:

```rust
        let hit = Hit {
            start: record.start,
            end: record.end,
            score: *score,
            margin: *score - baseline,
            kind: record.kind,
        };
```

In `search_rows` (index.rs:935), add the kind at each of the three places a
row is built. The note row (index.rs:949):

```rust
            notes.push(Row {
                note: true,
                kind: record.kind,
                label: record.label.clone(),
                start: record.start,
                end: record.end,
                score: *score,
                margin: *score - baseline,
            });
```

The video hit (index.rs:961):

```rust
            Hit {
                start: record.start,
                end: record.end,
                score: *score,
                margin: *score - baseline,
                kind: record.kind,
            },
```

The answer row (index.rs:982):

```rust
            rows.push(Row {
                note: false,
                kind: hit.kind,
                label: label.clone(),
                start: hit.start,
                end: hit.end,
                score: hit.score,
                margin: hit.margin,
            });
```

In `merge_hits_with` (index.rs:1011), the folded moment keeps the kind of its
best hit, as it keeps the score:

```rust
                if hit.score > last.score {
                    last.score = hit.score;
                    last.margin = hit.margin;
                    last.kind = hit.kind;
                }
```

In `main.rs`, the test helper `row` (main.rs:567) builds an `index::Row`, so it
gains the field:

```rust
    /// One row of a search, as the test writes it.
    fn row(label: &str, start: f32, end: f32, score: f32, note: bool) -> index::Row {
        index::Row {
            note,
            kind: 0,
            label: label.to_string(),
            start,
            end,
            score,
            margin: score - 0.5,
        }
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --lib index::`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: name the record kind on every hit

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: The panel and the row

**Files:**
- Modify: `videoembed/src/bin/web.rs:129-136`, `:171-209`, `:545-560`
- Modify: `videoembed/assets/index.html:80`
- Modify: `videoembed/assets/app.js:74-79`, `:97-152`, `:347-392`, `:528-546`
- Test: `videoembed/src/bin/web.rs` (the `tests` module)

**Interfaces:**
- Consumes: `Plan.fine` from Task 1, `Row.kind` from Task 5.
- Produces: the page sets the flag, and a hit row names its kind.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module of `videoembed/src/bin/web.rs`:

```rust
    /// The plan body reads the fine flag.
    #[test]
    fn the_plan_body_reads_the_fine_flag() {
        let body: PlanBody = serde_json::from_str(
            r#"{"path":"a.mp4","window":2.0,"frames":2,"audio":true,"fine":true}"#,
        )
        .unwrap();
        assert!(body.plan().unwrap().fine);
        let off: PlanBody =
            serde_json::from_str(r#"{"path":"a.mp4","window":2.0,"frames":2,"audio":true}"#)
                .unwrap();
        assert!(!off.plan().unwrap().fine);
    }

    /// The page holds a checkbox for the fine grid.
    #[test]
    fn the_page_holds_a_box_for_the_fine_grid() {
        assert!(PAGE.contains(r#"id="plan-fine""#));
        assert!(JS.contains(r#"$("plan-fine").checked"#));
        assert!(JS.contains("fine: $(\"plan-fine\").checked"));
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin web`
Expected: FAIL, the body has no `fine` field.

- [ ] **Step 3: Write the implementation**

In `web.rs`, in `PlanBody` (web.rs:171), after `audio: Option<bool>,`, add:

```rust
    #[serde(default)]
    fine: Option<bool>,
```

and in `PlanBody::plan`, before the `if words.is_empty()` check:

```rust
        if let Some(fine) = self.fine {
            words.push(format!("fine={}", if fine { "on" } else { "off" }));
        }
```

In `VideoRow` (web.rs:129), after `audio: bool,`, add:

```rust
    fine: bool,
```

and in the row build (web.rs:549), after `audio: plan.audio,`, add
`fine: plan.fine,`.

In `index.html`, after the audio label on line 80, add:

```html
    <label><input id="plan-fine" type="checkbox"> Both grids</label>
```

In `app.js`:

`planWords` (app.js:74), replace with:

```js
// One video's plan, as one short line for the table. An empty name means the
// plan matches no preset. The values then stand alone.
function planWords(video) {
  if (video.plan === "fine") return "Fine, " + video.window + " s frames";
  const head = video.plan === "" ? "" : video.plan + ", ";
  return head + video.window + " s window, " + video.frames +
    " frames" + (video.audio ? ", audio" : "") + (video.fine ? ", both grids" : "");
}
```

`showPlan` (app.js:97), add after `$("plan-audio").checked = video.audio;`:

```js
  $("plan-fine").checked = video.fine;
```

`usePreset` (app.js:179), add after `$("plan-audio").checked = audio;`:

```js
  $("plan-fine").checked = false;
```

`applyPlan` (app.js:142), add to the body after `audio: ...`:

```js
      fine: $("plan-fine").checked,
```

`boot` (app.js:538), add beside the audio listener:

```js
  $("plan-fine").addEventListener("change", planChanged);
```

`renderGroups` (app.js:379), add the kind to the hit text:

```js
      const kind = hit.kind === 0 ? "window" : hit.kind === 2 ? "frame" : "audio";
      const text = document.createElement("span");
      text.textContent =
        kind + "  " + hit.score.toFixed(3) + " (" + hit.margin.toFixed(3) + ")  " +
        hit.start.toFixed(2) + "-" + hit.end.toFixed(2) + " s";
```

The fine box clears the preset through `planChanged`, so the body always takes
the word path when the box changes. A body that still names a preset drops the
flag, because `PlanBody::plan` returns at the preset.

- [ ] **Step 4: Run the test to verify it passes**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml --bin web`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/bin/web.rs videoembed/assets/index.html videoembed/assets/app.js
git commit -m "feat: show the fine flag and the kind of a hit on the page

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## After the tasks

Run the whole suite one more time, then the branch review, per the execution
skill. The two small jobs of the spec, the reload guard and the speed report,
follow on this branch as their own commits.
