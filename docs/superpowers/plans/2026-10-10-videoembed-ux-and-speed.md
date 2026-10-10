# videoembed UX and Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four features from the Google comparison to the videoembed web app, and measure the cost of a coarser time grid before any format change.

**Architecture:** Every feature is a small, testable unit. Three of the four units are pure functions in `index.rs` or `media.rs`, so they need no model and no server. The server work reuses the existing `Job` channel, the `Holds` check, and the `Progress` object. The store keeps its VEI1 layout. Resume state lives in a sibling file, not in the record format.

**Tech Stack:** Rust 2021, hyper 1, serde, ffmpeg through `std::process::Command`, llama.cpp through the vendored `llama-cpp-2`, plain HTML and CSS and JavaScript.

**Spec:** `docs/google-gallery-comparison.md`

## Global Constraints

- Build the CPU form by default: `cargo build --manifest-path videoembed/Cargo.toml`.
- Run every test with: `cargo test --manifest-path videoembed/Cargo.toml`.
- The CUDA form stays optional: `--features cuda`. Do not make CUDA a default feature. Tests run on the CPU form.
- Assets are embedded with `include_str!` in `src/bin/web.rs:24-26`. A change to `assets/` needs a rebuild of `web.exe`.
- ffmpeg and ffprobe sit at fixed paths in `src/media.rs:8-11`. A test that needs them returns early when the files are missing, with a `skipped:` line on stderr.
- The store keeps the VEI1 layout from `src/store.rs:1-6`. No task changes the record layout.
- One name for one thing: a **record** is one stored vector; a **hit** is a record above the floor; a **moment** is one joined run of hits; a **cell** is one record drawn on the bar.
- Every line of prose follows ASD-STE100.

## Review Focus

These are the input classes no single task fully covers. Each gets its own test in the owning task.

1. Two separate matching runs inside one video. The merge must keep them apart, and a run split by a hole must not become one moment.
2. A store that holds only audio records. The audio records are 1 s long and they touch, so the merge joins them. The behaviour must be pinned by a test, not left to chance.
3. A frame request for a time past the end of the video, and a frame request for a path that is not in the index. Each must give a clear refusal, never a hang and never a body of zero bytes.
4. Stop pressed twice, and Stop pressed when no job runs. The page must stay usable and the job must stop.
5. A resume after a run that died part way. The unfinished video must index again. A finished video must not index again.

---

### Task 1: Join touching hits into one moment

**Files:**
- Modify: `videoembed/src/index.rs` (add `MOMENT_GAP`, `merge_hits`, and its tests; call the function from `group_hits`)
- Test: `videoembed/src/index.rs` (the `#[cfg(test)]` module)

**Interfaces:**
- Consumes: `Hit` (`src/index.rs:67-73`), `FRAME_STEP` (`src/index.rs:28`), `GROUP_CAP` (`src/index.rs:20`), `group_hits` (`src/index.rs:396`).
- Produces: `pub const MOMENT_GAP: f32` and `pub fn merge_hits(hits: Vec<Hit>) -> Vec<Hit>`. The answer comes sorted by score, best first. One moment spans the first start of its run to the last end. Its score and its margin come from the best hit of the run.

The rule: two hits join when the second starts no later than one frame step after the first ends. Hits of one run of frames touch, so they join. A run with one frame missing has a gap of one step, so it still joins. A run with two frames missing has a gap of two steps, so it splits.

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/index.rs`:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path videoembed/Cargo.toml merge_hits`
Expected: FAIL with `cannot find function merge_hits in this scope`.

Run: `cargo test --manifest-path videoembed/Cargo.toml moment`
Expected: the same failure, and `a_group_joins_its_run_of_frames_into_one_moment` fails on `assert_eq!(groups[0].hits.len(), 1)` with `4`.

- [ ] **Step 3: Write the implementation**

Add after `FRAME_STEP` in `src/index.rs`:

```rust
/// The largest gap between two hits that still joins one moment.
///
/// The gap is one frame step. Hits of one run of frames touch, so the gap is
/// zero. A run with one frame missing has a gap of one step.
pub const MOMENT_GAP: f32 = FRAME_STEP;
```

Add after `group_hits` in `src/index.rs`:

```rust
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
```

At the end of `group_hits`, before `groups`, add the merge and the cap. Replace the closing `groups` line of the function with:

```rust
    // A run of touching hits reads as one moment. The cap counts moments, not
    // frames, so one video may show many moments.
    for group in groups.iter_mut() {
        let hits = std::mem::take(&mut group.hits);
        let mut moments = merge_hits(hits);
        moments.truncate(GROUP_CAP);
        group.hits = moments;
    }
    groups
```

The existing `if group.hits.len() < GROUP_CAP` guard in the group-building loop stays. It bounds the work before the merge.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS, with the six new tests green and every earlier test still green. Note the count.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/index.rs
git commit -m "feat: join touching hits into one moment"
```

---

### Task 2: Report the indexing speed

**Files:**
- Modify: `videoembed/src/index.rs` (`Progress` fields, `per_second`, and its tests)
- Modify: `videoembed/src/bin/web.rs` (measure the load and the job, set the fields)
- Modify: `videoembed/assets/app.js`, `videoembed/assets/index.html` (show the speed)
- Test: `videoembed/src/index.rs`, `videoembed/src/bin/web.rs`

**Interfaces:**
- Consumes: `Progress` (`src/index.rs:54-64`), `WebJob` (`src/bin/web.rs:349-355`), `step_job` (`src/bin/web.rs:508`), `main` (`src/bin/web.rs:754`), `Embedder::load` (`src/embed.rs:72`).
- Produces: `pub fn per_second(items: usize, seconds: f64) -> f32`, and four new `Progress` fields: `backend: String`, `load_ms: u64`, `elapsed_ms: u64`, `per_second: f32`.

The rule follows the Google benchmark: time with a monotonic clock, report the model load apart from the job, and report a rate rather than a blended figure.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `src/index.rs`:

```rust
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
        assert_eq!(json["backend"], "");
        assert_eq!(json["load_ms"], 0);
        assert_eq!(json["elapsed_ms"], 0);
        assert_eq!(json["per_second"], 0.0);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path videoembed/Cargo.toml per_second`
Expected: FAIL with `cannot find function per_second in this scope`.

Run: `cargo test --manifest-path videoembed/Cargo.toml speed_fields`
Expected: FAIL, because `json["backend"]` is null, so it does not equal `""`.

- [ ] **Step 3: Write the implementation**

In `src/index.rs`, add these fields to `Progress`, after `error`:

```rust
    /// The compute backend of this build, "CPU" or "CUDA".
    pub backend: String,
    /// The milliseconds to load the model.
    pub load_ms: u64,
    /// The milliseconds of the running job.
    pub elapsed_ms: u64,
    /// The items of the running job for each second.
    pub per_second: f32,
```

In `src/index.rs`, add before `group_hits`:

```rust
/// The items of one job for each second.
///
/// A time of zero or less gives 0.0, so a fast job cannot divide by zero.
pub fn per_second(items: usize, seconds: f64) -> f32 {
    if seconds <= 0.0 {
        return 0.0;
    }
    (items as f64 / seconds) as f32
}
```

In `src/bin/web.rs`, add near `ADDRESS`:

```rust
/// The compute backend of this build. The page shows the name.
#[cfg(feature = "cuda")]
const BACKEND: &str = "CUDA";
#[cfg(not(feature = "cuda"))]
const BACKEND: &str = "CPU";
```

In `src/bin/web.rs`, add a field to `WebJob`:

```rust
    started: std::time::Instant,
```

In `src/bin/web.rs`, in `start_index`, name the backend in the `Progress`
literal:

```rust
        let mut held = progress.lock().unwrap();
        *held = Progress {
            running: true,
            folder: folder.display().to_string(),
            file_count: count,
            backend: BACKEND.to_string(),
            ..Progress::default()
        };
```

In `start_index`, add the clock to the `WebJob` literal:

```rust
    *pending = Some(WebJob {
        files,
        file_at: 0,
        run: None,
        records: Vec::new(),
        indexed: 0,
        started: std::time::Instant::now(),
    });
```

In `src/bin/web.rs`, in `step_job`, replace the tail that writes progress:

```rust
    } else {
        let done = job.indexed + job.records.len();
        let seconds = job.started.elapsed().as_secs_f64();
        let mut held = progress.lock().unwrap();
        held.window = job.run.as_ref().map(|r| r.done()).unwrap_or(0);
        held.indexed = done;
        held.elapsed_ms = (seconds * 1000.0) as u64;
        held.per_second = index::per_second(done, seconds);
    }
```

In `src/bin/web.rs`, in `main`, time the model load. Replace the start of the engine thread body:

```rust
        let started = std::time::Instant::now();
        let embedder = match Embedder::load(&models) {
            Ok(embedder) => embedder,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        let load_ms = started.elapsed().as_millis() as u64;
        {
            let mut held = engine_progress.lock().unwrap();
            held.backend = BACKEND.to_string();
            held.load_ms = load_ms;
        }
```

In `assets/app.js`, in `showProgress`, replace the running branch:

```js
  if (p.running) {
    const rate = p.per_second > 0 ? " - " + p.per_second.toFixed(2) + " items a second" : "";
    $("progress-text").textContent =
      "File " + p.file_index + "/" + p.file_count +
      ", item " + p.window + "/" + p.window_count +
      " - " + p.file + rate;
    return;
  }
```

and replace the done branch:

```js
  const rate = p.per_second > 0 ? " at " + p.per_second.toFixed(2) + " items a second" : "";
  $("progress-text").textContent =
    "Done. " + p.indexed + " records indexed" + rate +
    ". " + p.backend + ", model load " + p.load_ms + " ms.";
```

In `assets/index.html`, replace the legend line of the progress paragraph block by adding a speed line under it:

```html
    <p id="progress-text"></p>
    <p id="speed" class="legend"></p>
```

In `assets/app.js`, in `showProgress`, after the text is set, add:

```js
  $("speed").textContent = p.elapsed_ms > 0
    ? ((p.elapsed_ms / 1000).toFixed(1) + " s of work, " + p.backend)
    : "";
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS, with the three new tests green.

Run: `cargo build --manifest-path videoembed/Cargo.toml`
Expected: the build succeeds with no warning about an unused field.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/bin/web.rs videoembed/assets/app.js videoembed/assets/index.html
git commit -m "feat: report the indexing speed and the backend"
```

---

### Task 3: Stop a running index job

**Files:**
- Modify: `videoembed/src/bin/web.rs` (`Job`, `WebJob`, `step_job`, `engine_loop`, the `api` dispatcher, `main`)
- Modify: `videoembed/assets/app.js`, `videoembed/assets/index.html` (a Stop button)
- Test: `videoembed/src/bin/web.rs`

**Interfaces:**
- Consumes: `Job` (`src/bin/web.rs:29-48`), `step_job` (`src/bin/web.rs:508`), `engine_loop` (`src/bin/web.rs:376`), `Progress.running` (`src/index.rs:55`), `Stopping` (`src/bin/web.rs:113`).
- Produces: the route `POST /api/stop` with the reply `{"stopping": bool}`, and the new `step_job` return type `bool`. The value is true when the job has finished or stopped, so `engine_loop` clears `pending`.

Stop is checked between items, never inside an item. An item in flight always finishes.

- [ ] **Step 1: Write the failing check**

No unit test covers this task. The stop flag is read inside `step_job`,
and `step_job` needs a loaded model, so the seam is the HTTP route. Write a
small script in the workspace scratch directory. It starts a job, stops it,
and reads the progress. Save it as `<workspace>/check_stop.sh`:

```bash
#!/bin/sh
# The stop route must answer, and the job must leave the running state.
set -e
H='-H X-VEI:1 -H Content-Type:application/json'
code=$(curl -s -o /tmp/veifix/stop.json -w '%{http_code}' $H -d '{}' \
  http://127.0.0.1:8787/api/stop)
echo "stop reply: $code $(cat /tmp/veifix/stop.json)"
test "$code" = "200" || { echo "FAIL: no stop route"; exit 1; }
grep -q '"stopping":true' /tmp/veifix/stop.json || { echo "FAIL: no job stopped"; exit 1; }
sleep 1
running=$(curl -s $H http://127.0.0.1:8787/api/progress | grep -o '"running":[a-z]*')
test "$running" = '"running":false' || { echo "FAIL: still running"; exit 1; }
code2=$(curl -s -o /dev/null -w '%{http_code}' $H -d '{}' http://127.0.0.1:8787/api/stop)
test "$code2" = "200" || { echo "FAIL: the second stop broke"; exit 1; }
echo "PASS"
```

This check covers the whole Review Focus line 4: a stop, a second stop, and a
stop when no job runs.

- [ ] **Step 2: Run the check to verify it fails**

Start the server on the build from Task 2, start a job, then run the check:

```bash
cd /tmp/veifix/run && "D:/MLProjelerim/videoembedder-rust/videoembed/target/release/web.exe" &
curl -s -H "X-VEI: 1" -H "Content-Type: application/json" \
  -d '{"path":"D:/MLProjelerim/videoembedder-rust/spike"}' http://127.0.0.1:8787/api/index
sh /tmp/veifix/check_stop.sh
```

Expected: `stop reply: 404 ...` then `FAIL: no stop route`. The route does
not exist yet.

- [ ] **Step 3: Write the implementation**

In `src/bin/web.rs`, add the job:

```rust
    /// Stop the running index job. The answer says whether one ran.
    StopIndex { answer: Sender<bool> },
```

In `src/bin/web.rs`, add a field to `WebJob`:

```rust
    stopped: bool,
```

In `src/bin/web.rs`, in `start_index`, add to the `WebJob` literal:

```rust
        stopped: false,
```

In `src/bin/web.rs`, change the signature and the body of `step_job`:

```rust
/// One media item of work. True means the whole job has finished or stopped.
fn step_job(
    embedder: &Embedder,
    ctx: &mut LlamaContext<'_>,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    flag: &Arc<AtomicBool>,
    job: &mut WebJob,
) -> bool {
    if flag.load(Ordering::Relaxed) {
        // Drop the partial work of this video. A half-indexed video reads as a
        // whole one on the next search.
        job.run = None;
        job.records.clear();
        job.stopped = true;
        progress.lock().unwrap().running = false;
        return true;
    }
    if job.run.is_none() {
        if job.file_at >= job.files.len() {
            progress.lock().unwrap().running = false;
            return true;
        }
        let video = job.files[job.file_at].clone();
        match index::IndexRun::start(&video) {
            Ok(run) => {
                let label = normalize_label(&video.to_string_lossy());
                if let Err(err) = store.forget(&label) {
                    fail(progress, &err);
                }
                begin_file(progress, label, job.file_at + 1, run.total());
                job.records.clear();
                job.run = Some(run);
            }
            Err(err) => {
                fail(progress, &err);
                job.file_at += 1;
            }
        }
        return false;
    }

    let outcome = job.run.as_mut().unwrap().step(embedder, ctx);
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
        if job.file_at >= job.files.len() {
            progress.lock().unwrap().running = false;
            return true;
        }
    } else {
        let done = job.indexed + job.records.len();
        let seconds = job.started.elapsed().as_secs_f64();
        let mut held = progress.lock().unwrap();
        held.window = job.run.as_ref().map(|r| r.done()).unwrap_or(0);
        held.indexed = done;
        held.elapsed_ms = (seconds * 1000.0) as u64;
        held.per_second = index::per_second(done, seconds);
    }
    false
}
```

Add the imports at the top of `src/bin/web.rs`:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
```

In `src/bin/web.rs`, in `engine_loop`, replace the pending branch:

```rust
    let stopped = Arc::new(AtomicBool::new(false));
    ...
        if pending.is_some() {
            let done = step_job(
                &embedder,
                &mut ctx,
                &mut store,
                &progress,
                &stopped,
                pending.as_mut().unwrap(),
            );
            if done {
                pending = None;
            }
            while let Ok(job) = jobs.try_recv() {
                handle(job, &embedder, &mut store, &progress, &key, &stopped, &mut pending);
            }
        } else {
            match jobs.recv() {
                Ok(job) => handle(job, &embedder, &mut store, &progress, &key, &stopped, &mut pending),
                Err(_) => break,
            }
        }
```

Declare `let stopped = Arc::new(AtomicBool::new(false));` at the top of `engine_loop`, before the loop.

In `src/bin/web.rs`, change `handle` to take the flag, and add the arm:

```rust
fn handle(
    job: Job,
    embedder: &Embedder,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
    stopped: &Arc<AtomicBool>,
    pending: &mut Option<WebJob>,
) {
```

and inside the match:

```rust
        Job::StopIndex { answer } => {
            let running = pending.is_some();
            stopped.store(true, Ordering::Relaxed);
            let _ = answer.send(running);
        }
```

When a new job starts, clear the flag. In `start_index`, add before `*pending = Some(...)`:

```rust
    flag.store(false, Ordering::Relaxed);
```

`start_index` gains a parameter: `flag: &Arc<AtomicBool>`, passed from the `StartIndex` arm.

In `src/bin/web.rs`, in `main`, create the flag and send it to both places:

```rust
    let stopped = Arc::new(AtomicBool::new(false));
    let server_stopped = Arc::clone(&stopped);
    let engine_stopped = Arc::clone(&stopped);
```

Pass `server_stopped` to `handle_request`, and `engine_stopped` to `engine_loop`.

In `src/bin/web.rs`, in `api`, add the route before the fallback arm:

```rust
        (true, "/api/stop") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::StopIndex { answer });
            match wait.recv() {
                Ok(running) => request.respond(json_reply(&Stopping { stopping: running }))?,
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
```

In `assets/index.html`, add the button after the Index button:

```html
  <button id="stop" type="button" disabled>Stop</button>
```

In `assets/app.js`, in `boot`, wire the button:

```js
  $("stop").addEventListener("click", stopIndex);
```

and add the function:

```js
async function stopIndex() {
  try {
    await api("/api/stop", {});
  } catch (err) {
    note(err.message);
  }
}
```

In `assets/app.js`, in `startIndex`, enable the Stop button, and in `pollProgress`'s finish branch, disable it:

```js
  $("stop").disabled = false;
```

and in `pollProgress`:

```js
  $("stop").disabled = true;
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS on the whole suite.

Build, start the server, start a job, then run the check:

```bash
CMAKE_GENERATOR="Visual Studio 17 2022" CMAKE_BUILD_PARALLEL_LEVEL=8 cargo build --release --features cuda
cd /tmp/veifix/run && "D:/MLProjelerim/videoembedder-rust/videoembed/target/release/web.exe" &
curl -s -H "X-VEI: 1" -H "Content-Type: application/json" \
  -d '{"path":"D:/MLProjelerim/videoembedder-rust/spike"}' http://127.0.0.1:8787/api/index
sh /tmp/veifix/check_stop.sh
```

Expected: `stop reply: 200 {"stopping":true}` then `PASS`. The progress
reply shows `"running":false`. The second stop answers `200` with
`{"stopping":false}`.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/bin/web.rs videoembed/assets/app.js videoembed/assets/index.html
git commit -m "feat: stop a running index job"
```

---

### Task 4: Skip a video that is already indexed

**Files:**
- Modify: `videoembed/src/store.rs` (a `done` set, `is_done`, `mark_done`, and tests)
- Modify: `videoembed/src/bin/web.rs` (`WebJob.label`, the skip check, the mark)
- Test: `videoembed/src/store.rs`

**Interfaces:**
- Consumes: `Store` (`src/store.rs:46-50`), `Store::open` (`src/store.rs:54`), `Store::forget` (`src/store.rs:120`), `normalize_label` (`src/store.rs:33`), `WebJob` (`src/bin/web.rs:349`).
- Produces: `pub fn is_done(&self, label: &str) -> bool` and `pub fn mark_done(&mut self, label: &str) -> Result<()>`. The state lives in a sibling file of the store, named `index.done` beside `index.vei`. One finished video path occupies one line.

The state must not enter the VEI1 record layout. A sibling file keeps the format as it is. A missing sibling file means no video is finished.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `src/store.rs`:

The test module of `src/store.rs` has a one-argument `record(label)` helper
(`src/store.rs:234`) and no `scratch` helper. Add the `scratch` helper beside
`record`, and use the one-argument form:

```rust
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_store_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A finished video is known after the store reopens.
    #[test]
    fn a_finished_video_survives_a_reopen() {
        let dir = scratch("done");
        let path = dir.join("index.vei");
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
        let path = dir.join("index.vei");
        let mut store = Store::open(&path).unwrap();
        store.append(vec![record("a.mp4")]).unwrap();
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
        let path = dir.join("index.vei");
        let mut store = Store::open(&path).unwrap();
        store.mark_done(r"sub\a.mp4").unwrap();
        assert!(store.is_done("sub/a.mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path videoembed/Cargo.toml is_done`
Expected: FAIL with `no method named is_done found for struct Store`.

- [ ] **Step 3: Write the implementation**

In `src/store.rs`, add the import:

```rust
use std::collections::HashSet;
```

In `src/store.rs`, add a field to `Store`:

```rust
    /// The video paths that finished an index run.
    done: HashSet<String>,
```

In `src/store.rs`, in `open`, fill the set in both branches. Replace the missing-file branch:

```rust
        if !path.exists() {
            return Ok(Self {
                path: path.to_path_buf(),
                dim: 0,
                records: Vec::new(),
                done: read_done(&path),
            });
        }
```

and the success tail:

```rust
        let mut store = Self::decode(path.to_path_buf(), &bytes)?;
        store.done = read_done(&store.path);
        Ok(store)
```

with the signature changed to `let mut store = Self::decode(...)?;`. `Self::decode` sets `done: HashSet::new()` in its `Ok(Self { ... })`.

In `src/store.rs`, add the methods after `forget`:

```rust
    /// True when this video finished an index run.
    ///
    /// The path separators do not matter, like `holds_video`.
    pub fn is_done(&self, label: &str) -> bool {
        self.done.contains(&normalize_label(label))
    }

    /// Record that this video finished an index run.
    pub fn mark_done(&mut self, label: &str) -> Result<()> {
        self.done.insert(normalize_label(label));
        self.write_done()
    }

    /// The finished set, one path on each line.
    fn write_done(&self) -> Result<()> {
        let mut out = String::new();
        for label in &self.done {
            out.push_str(label);
            out.push('\n');
        }
        let path = done_path(&self.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("done.tmp");
        std::fs::write(&tmp, out).context("write the finished list")?;
        std::fs::rename(&tmp, &path).context("replace the finished list")?;
        Ok(())
    }
```

In `src/store.rs`, in `forget`, clear the mark:

```rust
        let removed = before - self.records.len();
        if removed > 0 {
            self.write()?;
            self.done.remove(&want);
            self.write_done()?;
        }
        Ok(removed)
```

In `src/store.rs`, add the free functions near `normalize_label`:

```rust
/// The finished list beside the store. `store/index.vei` gives `store/index.done`.
fn done_path(store: &Path) -> PathBuf {
    store.with_extension("done")
}

/// Read the finished list. A missing file gives an empty set.
fn read_done(store: &Path) -> HashSet<String> {
    let text = std::fs::read_to_string(done_path(store)).unwrap_or_default();
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(normalize_label)
        .collect()
}
```

In `src/store.rs`, in `decode`, add `done: HashSet::new(),` to the returned `Self`.

The new field breaks every `Store` literal in the file. Add
`done: HashSet::new(),` to each one:

- `src/store.rs:93`, the tail of `open`.
- `src/store.rs:222`, inside `encode_of`, a test helper.
- `src/store.rs:244`, inside `store_of`, a test helper.
- `src/store.rs:300` and `src/store.rs:329`, two tests that build a store by
  hand.

The compiler names any literal that is missed, so run the test command and
read the list.

In `src/bin/web.rs`, add a field to `WebJob`:

```rust
    label: String,
```

In `src/bin/web.rs`, in `start_index`, add to the `WebJob` literal:

```rust
        label: String::new(),
```

In `src/bin/web.rs`, in `step_job`, replace the `IndexRun::start` success arm:

```rust
            Ok(run) => {
                let label = normalize_label(&video.to_string_lossy());
                if store.is_done(&label) {
                    job.file_at += 1;
                    return false;
                }
                if let Err(err) = store.forget(&label) {
                    fail(progress, &err);
                }
                begin_file(progress, label.clone(), job.file_at + 1, run.total());
                job.label = label;
                job.records.clear();
                job.run = Some(run);
            }
```

In `src/bin/web.rs`, in `step_job`, in the finished branch, mark the video:

```rust
    if finished {
        let records = std::mem::take(&mut job.records);
        job.indexed += records.len();
        if let Err(err) = store.append(records) {
            fail(progress, &err);
        }
        if let Err(err) = store.mark_done(&job.label) {
            fail(progress, &err);
        }
        job.run = None;
        job.file_at += 1;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS, with the three new tests green.

Check the resume by hand. Index a folder, stop the job part way, then index the same folder again:

```bash
curl -s -H "X-VEI: 1" -H "Content-Type: application/json" -d '{"path":"<a folder with several videos>"}' http://127.0.0.1:8787/api/index
curl -s -H "X-VEI: 1" -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8787/api/stop
curl -s -H "X-VEI: 1" -H "Content-Type: application/json" -d '{"path":"<the same folder>"}' http://127.0.0.1:8787/api/index
```

Expected: the second run passes over every video that reported "Done" in the first run, and it indexes the video that stopped. Every video that finished has one line in `store/index.done`.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/store.rs videoembed/src/bin/web.rs
git commit -m "feat: skip a video that is already indexed"
```

---

### Task 5: Show a thumbnail for each hit

**Files:**
- Modify: `videoembed/src/media.rs` (`thumb_jpeg`, `run_ffmpeg_out`, and tests)
- Modify: `videoembed/src/bin/web.rs` (the `/api/frame` route and the thumbnail cache, and a test)
- Modify: `videoembed/assets/app.js`, `videoembed/assets/app.css` (the picture in each hit)
- Test: `videoembed/src/media.rs`, `videoembed/src/bin/web.rs`

**Interfaces:**
- Consumes: `FFMPEG` (`src/media.rs:8`), `run_ffmpeg` (`src/media.rs:65`), the `Holds` job (`src/bin/web.rs:32-35`), `bytes_reply` (`src/bin/web.rs:127`), `params` (`src/bin/web.rs:205`), `handle_request` (`src/bin/web.rs:581`).
- Produces: `pub fn thumb_jpeg(video: &Path, at: f64, width: u32) -> Result<Vec<u8>>`, the route `GET /api/frame?key=<key>&path=<path>&t=<seconds>&w=<pixels>`, and `pub fn thumb_key(path: &str, at: f32, width: u32) -> String`.

The route never runs in the engine thread. It checks the index with one `Holds` round trip, then cuts the frame on the request thread, so a search and an index both keep running.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `src/media.rs`:

```rust
    /// A thumbnail is a JPEG, and the bytes start with the JPEG marker.
    #[test]
    fn a_thumb_is_a_jpeg() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let bytes = thumb_jpeg(&video, 1.0, 320).unwrap();
        assert!(bytes.len() > 200, "got {} bytes", bytes.len());
        assert_eq!(bytes[0], 0xff);
        assert_eq!(bytes[1], 0xd8);
    }

    /// A cut past the end gives no frame, so the call must refuse.
    #[test]
    fn a_thumb_past_the_end_is_an_error() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        assert!(thumb_jpeg(&video, 1000.0, 320).is_err());
    }

    #[test]
    fn a_thumb_of_a_missing_file_is_an_error() {
        assert!(thumb_jpeg(Path::new("no_such_video.mp4"), 0.0, 320).is_err());
    }
```

Add to the test module in `src/bin/web.rs`:

```rust
    /// Two hits at the same time share one cache entry, and a width splits it.
    #[test]
    fn the_thumb_key_covers_the_path_the_time_and_the_width() {
        assert_eq!(thumb_key("a.mp4", 1.0, 320), thumb_key("a.mp4", 1.0, 320));
        assert_ne!(thumb_key("a.mp4", 1.0, 320), thumb_key("a.mp4", 1.5, 320));
        assert_ne!(thumb_key("a.mp4", 1.0, 320), thumb_key("a.mp4", 1.0, 640));
        assert_ne!(thumb_key("a.mp4", 1.0, 320), thumb_key("b.mp4", 1.0, 320));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path videoembed/Cargo.toml thumb`
Expected: FAIL with `cannot find function thumb_jpeg in this scope` and `cannot find function thumb_key in this scope`.

- [ ] **Step 3: Write the implementation**

In `src/media.rs`, add after `run_ffmpeg`:

```rust
/// Run ffmpeg and give back its standard output.
///
/// The frame pipe of `thumb_jpeg` writes the picture bytes there.
fn run_ffmpeg_out(args: Vec<String>, what: &str) -> Result<Vec<u8>> {
    let mut full: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-nostdin".into(),
    ];
    full.extend(args);
    let out = Command::new(FFMPEG).args(&full).output()?;
    if !out.status.success() {
        bail!("ffmpeg failed while {what}");
    }
    if out.stdout.is_empty() {
        bail!("ffmpeg gave no picture while {what}");
    }
    Ok(out.stdout)
}
```

In `src/media.rs`, add after `run_ffmpeg_out`:

```rust
/// Cut one JPEG frame at a time, scaled to a width. The bytes come back.
///
/// The seek flag comes before the input, so ffmpeg seeks fast. A time past the
/// end of the video gives no frame, and the call refuses.
pub fn thumb_jpeg(video: &Path, at: f64, width: u32) -> Result<Vec<u8>> {
    let args = vec![
        "-y".into(),
        "-ss".into(),
        format!("{at}"),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-frames:v".into(),
        "1".into(),
        "-vf".into(),
        format!("scale={width}:-2"),
        "-f".into(),
        "image2pipe".into(),
        "-vcodec".into(),
        "mjpeg".into(),
        "-".into(),
    ];
    run_ffmpeg_out(args, "cutting a thumbnail")
}
```

In `src/bin/web.rs`, add the cache and the key. Near the other `const` items:

```rust
/// The number of thumbnails kept in memory.
const THUMB_CAP: usize = 256;
/// The width of one thumbnail in pixels.
const THUMB_WIDTH: u32 = 320;
```

In `src/bin/web.rs`, add the key function near `mime_of`:

```rust
/// One key for one thumbnail. The time is rounded to the tenth of a second.
fn thumb_key(path: &str, at: f32, width: u32) -> String {
    format!("{path}|{:.1}|{width}", at)
}

/// Put one thumbnail in the cache. The oldest entry leaves when it is full.
fn cache_thumb(cache: &mut HashMap<String, Arc<Vec<u8>>>, key: String, bytes: Vec<u8>) {
    if cache.len() >= THUMB_CAP && !cache.contains_key(&key) {
        if let Some(first) = cache.keys().next().cloned() {
            cache.remove(&first);
        }
    }
    cache.insert(key, Arc::new(bytes));
}
```

Add to the imports at the top of `src/bin/web.rs`:

```rust
use std::collections::HashMap;
use std::sync::Arc;
```

In `src/bin/web.rs`, in `handle_request`, add the route before the `match`
on the path (`src/bin/web.rs:591`), so the picture never waits on the engine
loop:

```rust
    let query = url.split('?').nth(1).map(|text| text.to_string());
    if path == "/api/frame" && request.method() == &Method::Get {
        return serve_frame(request, query, jobs, thumbs);
    }
```

`handle_request` gains the cache parameter, after `key`:
`thumbs: &Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>>`.

In `src/bin/web.rs`, add the handler:

```rust
/// One thumbnail. The index check runs on the engine. The cut runs here.
fn serve_frame(
    request: Request,
    query: Option<String>,
    jobs: &Sender<Job>,
    thumbs: &Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>>,
) -> Result<()> {
    let Some(query) = query else {
        request.respond(error_reply(400, "the query is missing"))?;
        return Ok(());
    };
    let args = params(&query);
    let path = args.get("path").cloned().unwrap_or_default();
    if path.is_empty() {
        request.respond(error_reply(400, "the path is missing"))?;
        return Ok(());
    }
    let at = args
        .get("t")
        .and_then(|text| text.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0);
    let Some(at) = at else {
        request.respond(error_reply(400, "the time is not a number"))?;
        return Ok(());
    };
    let width = args
        .get("w")
        .and_then(|text| text.parse::<u32>().ok())
        .filter(|value| *value > 0 && *value <= 1024)
        .unwrap_or(THUMB_WIDTH);

    let key = thumb_key(&path, at, width);
    if let Some(bytes) = thumbs.lock().unwrap().get(&key).cloned() {
        request.respond(bytes_reply(200, "image/jpeg", bytes.as_ref().clone()))?;
        return Ok(());
    }

    let (answer, wait) = channel();
    let _ = jobs.send(Job::Holds {
        path: path.clone(),
        answer,
    });
    if !matches!(wait.recv(), Ok(true)) {
        request.respond(error_reply(403, "that video is not in the index"))?;
        return Ok(());
    }

    match media::thumb_jpeg(Path::new(&path), at as f64, width) {
        Ok(bytes) => {
            cache_thumb(&mut thumbs.lock().unwrap(), key, bytes.clone());
            request.respond(bytes_reply(200, "image/jpeg", bytes))?;
        }
        Err(err) => {
            request.respond(error_reply(400, &format!("{err:#}")))?;
        }
    }
    Ok(())
}
```

In `src/bin/web.rs`, in `main`, create the cache and pass it to every request thread:

```rust
    let thumbs = Arc::new(Mutex::new(HashMap::new()));
```

```rust
        let thumbs = Arc::clone(&thumbs);
        std::thread::spawn(move || {
            if let Err(err) = handle_request(request, &jobs, &progress, &key, &thumbs) {
                eprintln!("A request failed: {err:#}");
            }
        });
```

In `assets/app.js`, in `renderGroups`, replace the hit row builder:

```js
    for (const hit of group.hits) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hit";

      const shot = document.createElement("img");
      shot.className = "shot";
      shot.loading = "lazy";
      shot.alt = "frame at " + hit.start.toFixed(2) + " s";
      shot.src = "/api/frame?key=" + encodeURIComponent(key) +
        "&path=" + encodeURIComponent(group.video) +
        "&t=" + hit.start.toFixed(2) + "&w=320";

      const text = document.createElement("span");
      text.textContent =
        hit.score.toFixed(3) + " (" + hit.margin.toFixed(3) + ")  " +
        hit.start.toFixed(2) + "-" + hit.end.toFixed(2) + " s";

      row.append(shot, text);
      row.addEventListener("click", () => play(group.video, hit.start));
      item.appendChild(row);
    }
```

In `assets/app.css`, add after the `.hit` rule:

```css
.hit { display: flex; align-items: center; gap: 0.5rem; }

.shot {
  width: 4rem;
  height: 2.25rem;
  object-fit: cover;
  border-radius: 3px;
  background: #161b22;
  flex: none;
}
```

- [ ] **Step 4: Run the tests, then check the route by hand**

Run: `cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS, with the four new tests green.

Run: `cargo build --manifest-path videoembed/Cargo.toml`
Expected: the build succeeds.

Then rebuild the web binary and check the route with the server running:

```bash
curl -s -H "X-VEI: 1" -o /tmp/veifix/shot.jpg -w "%{content_type} %{size_download}\n" \
  "http://127.0.0.1:8787/api/frame?key=<the key from /api/state>&path=<a path from /api/videos>&t=1.0&w=320"
```

Expected: `image/jpeg 12xx`. A path that is not in the index gives `403`. A time past the end gives `400`. A `t` of `abc` gives `400`.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/media.rs videoembed/src/bin/web.rs videoembed/assets/app.js videoembed/assets/app.css
git commit -m "feat: show a thumbnail for each hit"
```

---

### Task 6: Measure the cost of a coarser time grid

**Files:**
- Modify: `videoembed/src/index.rs` (`thin_frames` and its tests)
- Modify: `videoembed/src/main.rs` (the `measure` command)
- Test: `videoembed/src/index.rs`

**Interfaces:**
- Consumes: `Record` (`src/store.rs:37-44`), `KIND_FRAME` (`src/store.rs:19`), `Store::rank` (`src/store.rs:177`), `median` (`src/index.rs:36`), `group_hits` (`src/index.rs:396`), `Embedder::embed_text` (`src/embed.rs:94`), `QUERY_PREFIX` (`src/embed.rs:21`).
- Produces: `pub fn thin_frames(records: &[Record], factor: usize) -> Vec<Record>`, and the console command `measure <query>`.

The measurement answers one question: how far does the best answer move when the time grid coarsens? It uses the vectors already in the store, so it costs one text embedding for each query and nothing else. It does not measure the audio packing or the timestamp text. Say so in the output.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `src/index.rs`:

```rust
    /// A factor of four keeps one frame of every four, and keeps the audio.
    #[test]
    fn thinning_keeps_one_frame_of_each_group() {
        let mut records: Vec<Record> = (0..8)
            .map(|i| record("a.mp4", i as f32 * 0.25, i as f32 * 0.25 + 0.25))
            .collect();
        records.push(record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0));
        records.push(record_of(crate::store::KIND_AUDIO, "a.mp4", 1.0, 2.0));

        let thin = thin_frames(&records, 4);

        assert_eq!(thin.len(), 4, "two frames and two audio records");
        let frames: Vec<f32> = thin
            .iter()
            .filter(|r| r.kind == crate::store::KIND_FRAME)
            .map(|r| r.start)
            .collect();
        assert_eq!(frames, vec![0.0, 1.0]);
    }

    /// A factor of one keeps every record.
    #[test]
    fn a_factor_of_one_keeps_everything() {
        let records = vec![record("a.mp4", 0.0, 0.25), record("a.mp4", 0.25, 0.5)];
        assert_eq!(thin_frames(&records, 1).len(), 2);
        assert_eq!(thin_frames(&records, 0).len(), 2);
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
        let thin = thin_frames(&records, 2);
        assert_eq!(thin.len(), 2);
        assert!(thin.iter().any(|r| r.label == "a.mp4"));
        assert!(thin.iter().any(|r| r.label == "b.mp4"));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path videoembed/Cargo.toml thin`
Expected: FAIL with `cannot find function thin_frames in this scope`.

- [ ] **Step 3: Write the implementation**

In `src/index.rs`, add after `merge_hits`:

```rust
/// Keep one frame of every group of `factor` frames of each video.
///
/// The thinning measures the cost of a coarser time grid. Every other record
/// stays, so the audio keeps its one second grid.
pub fn thin_frames(records: &[Record], factor: usize) -> Vec<Record> {
    if factor < 2 {
        return records.to_vec();
    }
    // The frames of each video, in time order. A store may hold the frames of
    // one video twice, so the order is taken from the start time.
    let mut order: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        if record.kind == crate::store::KIND_FRAME {
            order.entry(record.label.as_str()).or_default().push(index);
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
    records
        .iter()
        .enumerate()
        .filter(|(index, _)| keep[*index])
        .map(|(_, record)| record.clone())
        .collect()
}
```

In `src/main.rs`, add `KIND_FRAME` and `dot` to the imports. The file is a
binary, so its crate root is the binary, not the library. Write the full
`videoembed::` path:

```rust
use videoembed::embed::{DOC_PREFIX, Embedder, QUERY_PREFIX, dot};
use videoembed::store::{KIND_FRAME, KIND_NOTE, Record, Store};
```

In `src/main.rs`, add the command to the dispatch:

```rust
        "measure" => measure(embedder, store, rest),
```

In `src/main.rs`, add the function:

```rust
/// Show where the best answer lands on a coarser time grid.
///
/// The frames are already embedded. A factor of 4 turns the 4 fps grid into a
/// 1 fps grid, so the run costs one text embedding for each query. The run
/// does not measure the audio packing or the timestamp text of the Google
/// design. It measures the time grid alone.
fn measure(embedder: &Embedder, store: &Store, rest: &str) -> Result<()> {
    let queries: Vec<&str> = rest
        .split('|')
        .map(|query| query.trim())
        .filter(|query| !query.is_empty())
        .collect();
    if queries.is_empty() {
        bail!("usage: measure <query> | <query> ...");
    }
    let vectors: Vec<Vec<f32>> = queries
        .iter()
        .map(|query| embedder.embed_text(&format!("{QUERY_PREFIX}{query}")))
        .collect::<Result<Vec<_>>>()?;

    println!(
        "{:>6}  {:>7}  {:>6}  {:>16}  query",
        "factor", "grid", "frames", "top moment"
    );
    for factor in [1usize, 2, 4, 8] {
        let thin = index::thin_frames(&store.records, factor);
        let frames = thin
            .iter()
            .filter(|r| r.kind == KIND_FRAME)
            .count();
        if frames == 0 {
            println!("{factor:>6}  no frame records");
            continue;
        }
        let grid = index::FRAME_STEP * factor as f64;
        let mut line = Vec::new();
        for (at, query) in queries.iter().enumerate() {
            let ranked: Vec<(usize, f32)> = thin
                .iter()
                .enumerate()
                .map(|(i, r)| (i, dot(&vectors[at], &r.vec)))
                .collect();
            let mut ranked = ranked;
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let baseline = index::median(&ranked);
            let groups = index::group_hits(&thin, &ranked, baseline);
            let top = match groups.first().and_then(|g| g.hits.first()) {
                Some(hit) => format!("{:.2}-{:.2} s", hit.start, hit.end),
                None => "none".to_string(),
            };
            line.push(format!("{query}: {top}"));
        }
        println!(
            "{factor:>6}  {grid:>6.2}s  {frames:>6}  {}",
            line.join("  |  ")
        );
    }
    println!();
    println!("A factor of 4 makes a 1.00 s grid and a factor of 8 makes a 2.00 s grid.");
    println!("The run does not measure the audio packing or the timestamp text.");
    Ok(())
}
```

The `dot` function lives in `src/embed.rs:280`, not in `src/store.rs`. Write
`crate::embed::dot`. Do not add a wrapper for it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --manifest-path videoembed/Cargo.toml`
Expected: PASS, with the three new tests green.

Run the measurement on a real store. Use a copy, so the store is not touched:

```bash
cd /tmp/veifix/run && "D:/MLProjelerim/videoembedder-rust/videoembed/target/release/videoembed.exe" \
  "measure a person talking | music | blue" 2>&1 | tail -20
```

Expected: four lines, one for each factor. The `top moment` of the factor 1 line is the reference. Read the other lines against it and write down how far the start time moves.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/index.rs videoembed/src/main.rs
git commit -m "feat: measure the cost of a coarser time grid"
```

---

## After the plan

Write the measurement result into `docs/google-gallery-comparison.md`, in section 5, note 1. That note owns the window-mode decision, and the decision waits on the numbers.

## Self-review notes

- Spec coverage: feature 1 (thumbnail) is Task 5, feature 3 (speed report) is Task 2, feature 4 (merged hits) is Task 1, and the cancel half of feature 2 is Task 3, the resume half is Task 4. The measurement the plan promised is Task 6. Every review-focus line has an owner: line 1 and 2 in Task 1, line 3 in Task 5, line 4 in Task 3, line 5 in Task 4.
- Type consistency: `merge_hits`, `per_second`, `thin_frames`, `thumb_jpeg`, `thumb_key`, `is_done`, `mark_done` keep one spelling through every task. `step_job` returns `bool` from Task 3 on, and Task 4 and Task 5 use that form.
- Known gap: the window mode itself is not built. Task 6 measures the grid cost only, so it cannot settle the audio packing or the timestamp text. Section 5 of the comparison says so.
