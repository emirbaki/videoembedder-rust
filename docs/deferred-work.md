# Deferred work

The state on 2026-10-11. The list holds the work that we chose not to do yet.
The rank comes from `docs/google-gallery-comparison.md` section 5, and from the
review of the device-mode branch.

## 1. Features that wait

| # | Item | Why it waits |
| --- | --- | --- |
| 1 | Window mode: one 2 s window holds 2 frames and its audio | Changes the store format. It cuts the work about tenfold. One clip cannot settle the accuracy question, so the decision stays open. |
| 6 | A search inside one video | Needs a filter on the rank. Small cost. |
| 7 | A relative score cut, like the Google greyout | Changes the number of results. Small cost. |
| 8 | A per-video plan, with presets | Needs plan fields in the store. |
| 9 | Upsert and delete by record id | Needs a real store format. Items 8 and 11 wait on it. |
| 10 | MP4 normalisation, and a black-frame probe | Needs more ffmpeg work. |
| 11 | Sliders for the window and the frame count | Follows item 8. |

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
| The store search | One linear scan over every record, for each query. Fine for a few thousand records. A vector index comes later. |
| The store write | The whole store stays in RAM, and each append rewrites the file. A crash-safe append log or a real format removes both. |
| The switch cost | The probe opens one extra context per switch. The cost is small, and the switch happens on request only. |

## 4. Kept as they are, by choice

- The task prefixes.
- The median baseline and the margin.
- The 0.25 s frame grid.
- The flat store, until item 9 earns its cost.
- One binary, with CUDA as an option rather than a requirement.
