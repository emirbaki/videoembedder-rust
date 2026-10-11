# The indexing plan - design

Date: 2026-10-12
Status: approved in chat. The parts are ready to plan.

Sources:

- `docs/google-gallery-comparison.md` section 2.1 holds Google's plan, its
  presets, and its slider ranges.
- `docs/store-format-change.md` holds the store analysis.
- `docs/deferred-work.md` holds items 1, 8, and 11.
- `videoembed/examples/window_probe.rs` holds the throwaway probe.

## 1. The goal

The app cuts each video under one plan. Three deferred items meet here:

| Item | What it gives |
| --- | --- |
| 1 | Window mode: one 2 s window holds 2 frames and its audio |
| 8 | A plan for each video, with presets |
| 11 | Sliders for the window and the frame count |

Window mode cuts the work about tenfold. The probe measured it at 10 hits of
10 on one clip, against 5 hits of 10 for the current fine grid. The plan makes
that gain the default, and it gives the user control of the trade.

The design keeps the fine grid as a preset, because it answers "find the exact
second" and the window grid does not.

## 2. The decisions

The decisions on 2026-10-12:

1. Window mode lands as the default plan. Fine mode stays as a preset.
2. The presets are the three Google presets plus Fine: Object, Action, Speech,
   and Fine.
3. A plan change re-cuts the video at once. The user does not run a second
   command.
4. The stride is half of the window. The probe used this setting and Google
   uses zero overlap. The design follows the probe.
5. The store format does not change. One column joins the `videos` table.
6. An older store reads as the Fine plan, because its records are fine records.

## 3. The plan and the presets

A plan says how one video is cut. Five values:

| Value | Meaning |
| --- | --- |
| `mode` | `window` or `fine` |
| `window` | the seconds of one window |
| `stride` | the seconds between two window starts |
| `frames` | the frame count in one window |
| `audio` | the window record holds the window audio |

The four presets:

| Preset | mode | window | stride | frames | audio |
| --- | --- | --- | --- | --- | --- |
| Fine | fine | 0.25 | 0.25 | 1 | yes |
| Object | window | 2 | 1 | 2 | no |
| Action | window | 4 | 2 | 4 | no |
| Speech | window | 6 | 3 | 6 | yes |

The default plan is Object.

In fine mode the `window`, `stride`, and `frames` values have no effect. Fine
mode keeps the 0.25 s frame grid and the 1 s audio grid, as the code does today.

The frame rate inside one window is `frames / window`. The Object preset gives
1 frame for each second, so a 2 s window holds 2 frames.

## 4. The records

Fine mode keeps the two record kinds. One `KIND_FRAME` record covers 0.25 s,
and one `KIND_AUDIO` record covers one second.

Window mode writes one `KIND_WINDOW` record for each window. The label is the
video path, the range is the window, and the vector holds the frames and the
audio together. One batch holds every part, and one decode follows. The store
already names the kind `KIND_WINDOW`.

A video in window mode writes no frame record and no audio record. One pass
writes one kind or the other.

## 5. The schema

The `videos` table gains one column:

```sql
plan_mode TEXT NOT NULL DEFAULT 'fine'
```

The four plan columns already exist: `plan_window`, `plan_stride`,
`plan_frames`, and `plan_audio`. The `videos` row already holds the finished
mark, so the plan and the mark live in one row.

`CREATE TABLE IF NOT EXISTS` does not change an existing table. The open call
therefore reads `pragma table_info(videos)`, and it runs the `ALTER TABLE` when
`plan_mode` is absent. The default `'fine'` backfills every existing row, and
the existing plan values are the Fine values, so the row is true for its
records.

## 6. The index run

`IndexRun` takes a plan, in place of the fixed frame and audio steps.

- **Window mode.** The run cuts each window to `frames` JPEG files at the rate
  `frames / window`, plus the window WAV when `audio` is on. It calls
  `Embedder::embed_window_in` with the frames first and the audio last. The
  record range is `[start, start + window)`. The next start adds `stride`.
- **Fine mode.** The run keeps the current two phases, the frames phase and the
  audio phase.

The run cuts media in batches, so the temp disk stays bounded. The caller
drives it with one `step` call for each record. The progress counts windows
done of the window total in window mode, and items done of the item total in
fine mode.

## 7. A plan change

A plan change re-cuts the video at once. The steps:

1. Write the new plan into the `videos` row.
2. Find the source file at the stored path. A missing file stops the change,
   and the old plan stays.
3. Remove the old records of the video.
4. Cut and embed the video again, under the new plan.
5. Write the new records and the finished mark.

The engine loop runs the steps, so the page and the console share one path. A
long video reports progress, like the index job. One re-index runs at a time,
because the engine holds one job.

A fault between step 3 and step 5 leaves the video with no records. The plan
row stays, so the next index run rebuilds the video.

## 8. The controls

The console takes a preset name and a set of values:

```
plan <path> object
plan <path> window=4 frames=4 audio=on
```

A value word sets one plan value on top of the current plan. The two forms do
not mix in one call.

The page adds a panel for one video:

- four preset buttons,
- a window slider from 1 s to 10 s,
- a frame slider from 1 to the window, and
- an audio switch.

Apply runs the change in section 7. The panel reads the current plan of the
video from the videos reply.

## 9. The tests

Each change starts with a failing test.

- The plan type reads and writes each preset, and the frame rate follows from
  `frames / window`.
- A window cut gives the expected record count and the expected range.
- A window record holds the frames and the audio in one batch.
- A plan change re-cuts the video, and the old records leave.
- A missing source file keeps the old plan.
- An older store reads as Fine.
- The console text for the plan command, both forms.
- The page panel sends the plan of the video.

## 10. What stays the same

- The label rule, `normalize_label`, and the slash form of a path.
- The kind list, and `is_video`.
- The record id, `path/startMs-endMs`.
- The atomic write through one transaction.
- The one vector size for the whole store.
- The store path, `store/index.db`.
- The rank, the search scope, and the weak cut.

## 11. What waits

- **Audio alone.** The Speech preset holds audio. A video with no audio track
  gives a window of frames alone, and the run must not fail.
- **The audio budget.** Google drops the audio of a window that would pass the
  token budget. The design does not measure the budget yet.
- **The timestamp text.** Google puts the frame times into the batch as text.
  The design does not add that text.
- **The fine grid beside the window grid.** A plan holds one mode. A store can
  hold both grids for one video only when the user changes the plan back and
  forth. The design does not hold both at one time.
- **MP4 normalisation.** Item 10 stays.

## 12. The next step

One more probe run measures the Object plan, because the probe measured the
audio form only. The run answers one question: does the audio change the hit
count? The first plan task runs it.

Then the writing-plans skill makes the implementation plan from this document.
