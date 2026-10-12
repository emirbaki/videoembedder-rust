# The cut, the batch, and the search - design

Date: 2026-10-12
Status: approved. The user settled the four open decisions on 2026-10-12.
Plan A: `docs/superpowers/plans/2026-10-12-cut-pass.md`.
Plan B: `docs/superpowers/plans/2026-10-12-window-batch.md`.
Plan C: `docs/superpowers/plans/2026-10-13-search-job.md`.

Sources:

- `docs/deferred-work.md` section 1 holds the ranked list.
- `docs/google-gallery-comparison.md` holds the Google facts, with a file and a
  line for each one.
- `videoembed/src/media.rs`, `videoembed/src/embed.rs`,
  `videoembed/src/index.rs`, `videoembed/src/store.rs`, and
  `videoembed/src/bin/web.rs` hold the current code.

## 1. The goal

Eight items of the ranked list become three plans, in order. Item 4 is dropped.

| Plan | Items | What it gives |
| --- | --- | --- |
| A. The cut | 1, 7 | One ffmpeg pass for the whole video, a progressive MP4, and a still that is not black |
| B. The batch | 2, 3, 10 | Four windows in one model call, the frame times as text, and a token budget |
| C. The search and the job | 5, 6 | Near hits merged into one moment, a pause, and a reconcile pass |

Each plan ships on its own branch, and each merges into `main` after its tests
pass. Each plan gets its own design section below.

## 2. The decisions

The decisions on 2026-10-12:

1. Three plans, run in order: A, then B, then C.
2. Item 4, two grids for one video, is dropped. A plan holds one mode. Two
   grids double the records of a video, and one result list would mix two score
   scales. The Fine preset already answers "the exact second".
3. One model call holds four windows.
4. A fragmented MP4 is rewritten once, and the copy stays beside the store.

### 2.1 The rulings I made

The user named no value for these four, so I chose them:

1. The timestamp text uses `MM:SS`, and one marker names the time of each media
   part.
2. A window that would pass the token budget loses its audio, and the run
   continues. This is Google's rule.
3. A black still steps forward to the next probe offset. The offsets are
   Google's six, from 0 to 4000 ms.
4. The pause is a web control. The reconcile pass runs at the start of a run.

## 3. Plan A: the cut

### 3.1 One pass for the frames and the audio

Today one window costs two ffmpeg starts, one for the frames and one for the
audio, and each start seeks. A 30 s clip at 2 s on a 1 s stride gives 29
windows and 58 starts.

Two functions replace the per-span cuts:

- `media::cut_frames(video, rate, dir) -> Result<Vec<PathBuf>>` runs one
  `-vf fps={rate}` pass over the whole video, and it gives back the frames in
  time order.
- `media::cut_audio(video, dir) -> Result<Vec<PathBuf>>` runs one pass with the
  segment muxer at one second a clip, and it gives back the clips in time
  order. The current per-span function already uses this muxer.

Frame number `n` counts from one and sits at `(n - 1) / rate`. Clip number `k`
counts from zero and covers `[k, k + 1)`.

A window from `t` to `t + window` takes:

- the frames with a time in `[t, t + window)`
- the clips that touch the span, from `floor(t)` to `ceil(t + window)`

Fine mode keeps the same two functions. It takes every frame at
`FRAME_RATE` (4), and one clip for each `KIND_AUDIO` record. The batching
constants `FRAME_BATCH`, `FRAME_STEP`, `AUDIO_BATCH`, and `AUDIO_SECONDS`
hold the slicing, and they no longer drive a cut.

One window of the Object plan takes two frames and two clips. Neighbouring
windows share a frame and a clip, because the stride is shorter than the window.

### 3.2 The work directory

The run cuts every frame and every clip once, so the whole media of the video
sits on disk at one time. The run keeps the files until the video ends, then it
removes the directory. A 30 minute video at 4 fps gives about 7200 frames, so
the disk use is real. The design accepts it, because a second pass over the
video costs more than the disk.

### 3.3 The fragmented MP4

Two functions join `media.rs`:

- `media::is_fragmented_mp4(path) -> Result<bool>` walks the top level boxes of
  the file, and it reports true for a `moof` box or for a `mvex` box inside the
  movie box.
- `media::progressive_copy(video, store_dir) -> Result<PathBuf>` runs
  `ffmpeg -i <source> -c copy -movflags +faststart <copy>`, and it gives back
  the copy path. The copy lives in `<store>/media/`, and its name is the source
  name plus a hash of the full path, so two videos with one name do not collide.

The run reads the copy when the source is fragmented. It reuses an existing
copy when the copy is newer than the source, so a re-cut skips the rewrite. A
stale copy is made again.

The run reports the rewrite on its own line, because a long video takes time to
copy.

### 3.4 The black opening frame

`media::luma(video, at) -> Result<f64>` reads the average luma of one frame
from the `signalstats` filter, and it gives back `YAVG`.

`media::thumb_jpeg` gains a rule. The probe offsets are 0, 200, 500, 1000,
2000, and 4000 ms. A request inside the first four seconds takes the first
offset at or after the request with a luma above `BLACK_LUMA` (16). A request
after the first four seconds keeps its own time.

The web server holds one chosen offset for each video path, so the probe runs
once for each video in the life of the process.

## 4. Plan B: the batch

### 4.1 Four windows in one call

The batch entry calls already take a sequence id, and the context already
answers one pooled vector for each sequence. Today every entry names sequence
zero, and the context holds one sequence.

The changes:

- `N_WINDOW_BATCH = 4`.
- The context takes `.with_n_seq_max(N_WINDOW_BATCH)`.
- `IndexRun` cuts four windows, fills one batch with four sequences, runs one
  `llama_process`, and reads four vectors.
- The last batch of a video holds one to four windows, because the window count
  rarely divides by four.

The batch and the context grow, so `N_BATCH` and `N_UBATCH` need a check
against the real token count of four windows. The first task of the plan
measures that count and pins the constants.

### 4.2 The parity rule

Four windows in one call must give the same vectors as one window in one call.
The plan proves this with a test: embed four windows one at a time, then in one
batch, and compare each pair. The cosine of each pair must pass 0.999.

A failure here stops the plan. The gain is worthless when the vectors move.

### 4.3 The timestamp text

Google puts the frame time into the sequence as text, for example `00:03`
(`EmbedPart.kt:35`), so the model sees the order of events.

`embed.rs` gains a pure function that builds the text of one window. Each media
part takes one `MM:SS` marker before it, and the marker names the start time of
that part. A 2 s window at 3 s with two frames and its audio reads:

```text
00:03 <__media__> 00:04 <__media__> 00:03 <__media__>
```

The order of the media stays as it is today: the frames first, then the audio.
The marker count always equals the bitmap count.

The offset names the video time, not the window time.

### 4.4 The token budget

`BUDGET_TOKENS = 1024`, as Google uses. The count of one window comes from the
tokenizer, not from a guess: `MtmdInputChunks::n_tokens()` answers it.

The rule: the run builds the window, counts the tokens, and builds the window
again with no audio when the count passes the budget. The record then holds the
frames alone, and the run reports each such window on one line.

The plan measures the real count of the Object window and records the number in
a comment. A plan whose window passes the budget without audio is a design
fault, and the plan names it.

## 5. Plan C: the search and the job

### 5.1 The merged hit

Google merges hits within 500 ms of each other into one interval
(`VideoMomentFinderViewModel.kt:1020-1036`). This app shows every hit, so three
near hits read as three moments.

`MERGE_GAP = 0.5`. The rule: sort the hits of one video by start time, then
fold a hit into the one before it when its start sits within `MERGE_GAP` of
that hit's end. The merged range runs from the first start to the last end, and
the score is the best score of the group.

The merge runs after the top-10 cap, so a merged moment counts as one row. The
rule holds for the window grid and for the fine grid.

### 5.2 The pause

The web job gains a paused state:

- `Progress` gains `paused: bool`.
- `Job` gains a pause flag that the job thread checks between items.
- `/api/pause` (POST) toggles the flag. A job that is not running refuses the
  call with a message.
- The page gains a Pause button. While the job is paused the button reads
  Resume.
- The progress answer carries the paused state, so a reloaded page shows it.

A paused job keeps its cut files and its place in the video. The model waits on
no job, so a pause costs the run nothing after the current window ends.

### 5.3 The reconcile pass

A video can hold a done mark with no records, because a fault lands between the
two writes. Google marks such a project unprocessed at the start of a run
(`VideoMomentFinderViewModel.kt:1067-1095`).

`Store::reconcile() -> Result<usize>` clears the done mark of every video that
holds no video records, and it gives back the count. The web folder run calls
it once at the start, and it writes the count to the server line when the count
is not zero.

## 6. The tests

Each plan carries its own tests, in the plan's task order. The rules:

- Media work uses the real ffmpeg and the real test clip.
- A pure function takes a plain unit test.
- The batch parity test uses the real model.
- A web route takes a unit test of its handler, plus one test of the state the
  page reads.

## 7. What waits

- Items 8, 9, 11, and 12 of the ranked list: the card fallback, the model
  download, the cold and warm speed report, and finer steps in the plan panel.
- Section 2 of `docs/deferred-work.md`: the six small fixes from the
  device-mode review.

Item 4 is dropped, and it does not wait.

## 8. The next step

The writing-plans skill makes Plan A from this document. Plans B and C follow
it, one at a time, because each plan changes the tree that the next one reads.
