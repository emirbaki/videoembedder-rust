# Deferred work

The state on 2026-10-12. The list holds the work that we chose not to do yet.
The rank comes from `docs/google-gallery-comparison.md` section 5, and from the
review of the device-mode branch.

Items 1, 8, and 11 of section 5 are done. Window mode, the per-video plan with
its presets, and the plan panel landed on 2026-10-12. Items 2, 4, 6, 7, and 9
of that list landed before them, on other branches: the thumbnail in each
result, the inline speed report, the scoped search, the relative score cut, and
the upsert by record id.

Item 3 landed on 2026-10-13. The web server stops a run, it holds the run and
lets it go again, and a folder run clears the mark of a video with no records.
Item 5 still waits.

Items 1 and 7 of section 1 of this list are done. The `cut-pass` branch landed
the one-pass cut, the progressive copy of a fragmented MP4, and the black-frame
probe. The list below holds the rank of the work that remains.

The `indexing-plan` branch merged into `main` on 2026-10-12. The full suite
passes on the merged tree: 108 lib tests, 15 console tests, and 25 web tests.

The probe of 2026-10-12 measured window mode on `spike/testvideo_nolabels.mp4`
against ten queries. Window mode with audio hit 10 of 10. Window mode with no
audio hit 6 of 10. The fine grid hit 5 of 10. The probe window is the Object
plan, so the audio form answers for the Object shape.

The default plan changed on 2026-10-12. It is Object with the window audio,
because of the probe result. Google's table gives Object no audio, and the
design recorded that table. The default now departs from it.

A video that is already indexed keeps its stored plan. A new default does not
re-cut it. Change the plan of that video to take the new default.

Plan B landed on 2026-10-12 on the `window-batch` branch. It answered the
window batch, the timestamp text, and the audio token budget.

Plan C waits on the `search-job` branch for its merge. It answered the merged
moment, the pause and the resume, and the reconcile pass.

## 1. Features that wait

The rank weighs the gain against the cost.

| # | Item | Gain | Cost |
| --- | --- | --- | --- |
| 1 | Both grids for one video | Medium | Medium |
| 2 | A fallback when the card fails | Medium | Low |
| 3 | The model download | Medium | Medium |
| 4 | The cold and warm speed report | Small | Low |
| 5 | Finer steps in the plan panel | Small | Low |
| 6 | A pause inside one window | Small | Low |

### 1.1 Both grids for one video

A plan holds one mode, and a plan change drops the old records. A video that
holds the window grid and the fine grid at the same time answers "what happens
here" and "which second" together. The comparison doc calls this a hybrid.

The cost is more records for one video, plus a search that mixes two grids.

### 1.2 A fallback when the card fails

Google tries the preferred backend and falls back to the CPU when init fails
(`OnDeviceEmbedder.kt:374-409`). The web server answers an error instead. This
app keeps the audio on the same context as the frames, and Google keeps the
audio on the CPU in every branch (`OnDeviceEmbedder.kt:318-333`).

### 1.3 The model download

The user fetches the model by hand. Google downloads with a range resume and a
temporary file.

### 1.4 The cold and warm speed report

The page shows items a second and the model load in milliseconds. Google
reports the cold init apart from the warm init, and it keeps the raw samples for
each run. It never blends the numbers into one figure.

### 1.5 Finer steps in the plan panel

The two sliders run in whole seconds. The Fine preset holds a 0.25 s window, so
the panel cannot show it. The Fine button applies at once for that reason. A
step of 0.25 s in the window slider would show every preset in the controls.

### 1.6 A pause inside one window

The engine reads the pause flag between items, so a press waits for the window
in the model. A finer check costs a flag read inside `IndexRun::step`.

## 2. Small fixes from the device-mode review

| # | Item |
| --- | --- |
| 1 | A switch and a Start in one engine turn are order-dependent. |
| 2 | A refused second switch says an index job runs. No such job exists. |
| 3 | `/api/device` reloads the model when the mode is already active. The console guards this. The page does not. |
| 4 | The page shows the picked mode before the load ends, and holds stale text after. |
| 5 | The `--gpu` and `--cpu` flags are stripped from any place in the arguments. |
| 6 | A load failure on the server answers HTTP 400. |

## 3. Optimisations

| Item | Limit today |
| --- | --- |
| The store search | The rank asks the vector table for k = every record of the scope. sqlite-vec scans that partition for such an ask. A small k, with a second pass over the top rows, would cut the scan. |
| The record list in memory | The store reads every record row into memory at open. One million records is heavy, and the search uses the rows only for the time range. |
| The repeated cut | A plan change cuts the whole video again, and the video leaves the search for the length of the cut. Write the new records under a staging id, then swap them for the old ones in one write. |
| The switch cost | A device switch opens one extra context. The cost is small, and the switch happens on request only. |
| The whole media on disk | The run cuts every frame of the whole video before the first window reaches the model, so the frames of the whole video sit in a temporary folder. A 30 s video at 4 frames a second gives about 120 stills. |
| The frames alone over the budget | A window whose frames alone pass 1024 tokens keeps them and goes into the batch, because the batch holds 4096 tokens. The run reports the window and does not refuse the plan. The Object window measures 257 tokens, so the case needs a named window of many frames. |

## 4. Kept as they are, by choice

- The task prefixes.
- The median baseline and the margin.
- The 0.25 s frame grid.
- One binary, with CUDA as an option rather than a requirement.
