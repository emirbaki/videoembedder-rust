# Deferred work

The state on 2026-10-13. The list holds the work that we chose not to do yet.
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
probe.

Item 2 of section 1 landed on 2026-10-13. A failed load on the card loads on
the processor instead. The list below holds the rank of the work that remains.

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

Plan C landed on 2026-10-13. It answered the merged moment, the pause and the
resume, and the reconcile pass.

The `small-fixes` branch merged into `main` on 2026-10-13. It answered five
small fixes from the device-mode review, the fallback when the card fails, and
one more fix from the branch review. The full suite passes on the merged tree:
156 library tests, 21 console tests, 38 web tests, and 4 model tests.

The `hybrid-grid` branch merged into `main` on 2026-10-13. It answered item 1
and item 3 of section 1, and item 3 of section 2. A plan now takes a `fine`
flag, one dense cut serves both grids, and one pass writes both kinds of
record. The full suite passes on the merged tree: 166 library tests, 21 console
tests, 45 web tests, and 4 model tests.

The `fix/ubatch-group-budget` branch landed on 2026-10-13. The cost of one
frame follows the size of the frame: a 480x270 frame costs 88 tokens, a
720x720 frame costs 229, and a 1920x1920 frame costs 1093. A group of large
windows therefore passed the 4096 token limit of one model call, and the run
aborted on the assert at `llama-context.cpp:1512`. The run now measures each
window and gives the model the longest leading part of the group that fits.
A window that alone passes the limit keeps its first frames, and the run
reports that window on one line. The full suite passes on the branch: 167
library tests, 21 console tests, 45 web tests, and 6 model tests.

## 1. Features that wait

The rank weighs the gain against the cost.

| # | Item | Gain | Cost |
| --- | --- | --- | --- |
| 2 | The model download | Medium | Medium |
| 4 | Finer steps in the plan panel | Small | Low |
| 5 | A pause inside one window | Small | Low |

Items 1 and 3 landed on `hybrid-grid` on 2026-10-13.

### 1.2 The model download

The user fetches the model by hand. Google downloads with a range resume and a
temporary file.

### 1.4 Finer steps in the plan panel

The two sliders run in whole seconds. The Fine preset holds a 0.25 s window, so
the panel cannot show it. The Fine button applies at once for that reason. A
step of 0.25 s in the window slider would show every preset in the controls.

### 1.5 A pause inside one window

The engine reads the pause flag between items, so a press waits for the window
in the model. A finer check costs a flag read inside `IndexRun::step`.

## 2. Small fixes from the device-mode review

Five of the six landed on `small-fixes` on 2026-10-13: the turn order, the
refusal text, the page mode, the flag stripping, and the load-failure code.
The fallback when the card fails also landed there. Item 3 landed on
`hybrid-grid` on 2026-10-13. The page now guards the pick of the device in use,
so the click answers at once. No item waits.

## 3. Optimisations

| Item | Limit today |
| --- | --- |
| The store search | The rank asks the vector table for k = every record of the scope. sqlite-vec scans that partition for such an ask. A small k, with a second pass over the top rows, would cut the scan. |
| The record list in memory | The store reads every record row into memory at open. One million records is heavy, and the search uses the rows only for the time range. |
| The repeated cut | A plan change cuts the whole video again, and the video leaves the search for the length of the cut. Write the new records under a staging id, then swap them for the old ones in one write. |
| The switch cost | A device switch opens one extra context. The cost is small, and the switch happens on request only. |
| The whole media on disk | The run cuts every frame of the whole video before the first window reaches the model, so the frames of the whole video sit in a temporary folder. A 30 s video at 4 frames a second gives about 120 stills. |
| The text path over the batch limit | `embed_text` passes `n_batch = 8192` to the text path, above `n_ubatch = 4096`. A query or a note above 4096 tokens hits the same assert as the window path. A query of that length is unlikely, so the item waits. |

## 4. Kept as they are, by choice

- The task prefixes.
- The median baseline and the margin.
- The 0.25 s frame grid.
- One binary, with CUDA as an option rather than a requirement.
