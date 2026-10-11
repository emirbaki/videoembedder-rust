# Deferred work

The state on 2026-10-12. The list holds the work that we chose not to do yet.
The rank comes from `docs/google-gallery-comparison.md` section 5, and from the
review of the device-mode branch.

## 1. Features that wait

| # | Item | Why it waits |
| --- | --- | --- |
| 10 | MP4 normalisation, and a black-frame probe | Needs more ffmpeg work. |

Items 1, 8, and 11 are done. Window mode, the per-video plan with its presets,
and the plan panel all landed on 2026-10-12.

The probe of 2026-10-12 measured window mode on `spike/testvideo_nolabels.mp4`
against ten queries. Window mode with audio hit 10 of 10. Window mode with no
audio hit 6 of 10. The fine grid hit 5 of 10. The probe window is the Object
plan, so the audio form answers for the Object shape.

The default plan changed on 2026-10-12. It is Object with the window audio,
because of the probe result. Google's table gives Object no audio, and the
design recorded that table. The default now departs from it.

A video that is already indexed keeps its stored plan. A new default does not
re-cut it. Change the plan of that video to take the new default.

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
| The store search | One query over every vector for each query. The vector table answers it. A vector index comes near one million records. |
| The switch cost | The probe opens one extra context per switch. The cost is small, and the switch happens on request only. |

## 4. Kept as they are, by choice

- The task prefixes.
- The median baseline and the margin.
- The 0.25 s frame grid.
- One binary, with CUDA as an option rather than a requirement.
