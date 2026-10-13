# videoembed

Search inside a video by text. The app embeds a video in short windows, then
returns the window that matches a query.

Example. A 30 s clip shows a party at 20 s. The query `birthday party` returns
the window 21.0-23.0 s with a score of 0.78.

## Status

| Part | State |
|---|---|
| Console app | Works. |
| Web UI | Works. See the Web UI section. |
| GPU | Optional. The `cuda` feature runs the model on an NVIDIA card. |

## How it works

1. ffmpeg cuts the whole video one time: its frames, and every one second audio
   clip.
2. A window takes its own frames and clips from that one cut, so a window costs
   no further ffmpeg call.
3. The model reads four windows in one call. Each window holds its frames, its
   audio, and a `MM:SS` marker before each part.
4. One decode runs, and one pooling step gives one vector for each window.
5. The vector goes into a store. A query embeds the same way and ranks the store
   by cosine similarity.

Step 3 is the important part. The model sees the whole window at once. A
separate decode for each part would pool only the last part. Four windows share
one call, and each window holds its own sequence, so the vectors do not move
with the size of the batch. A window above 1024 tokens loses its audio, and the
run reports that window on one line.

The encoder cannot split a batch, so one call holds no more than 4096 tokens.
A group of large windows therefore gives more than one call. A large frame
costs many tokens, and the cost follows the size of the frame. A window that
alone passes the limit keeps only its first frames, and the run reports that
window on one line.

Two steps keep the picture right. A fragmented MP4 is rewritten one time into a
progressive copy, and the run reads that copy. A still near the opening steps
forward past a black frame to the first frame with a picture.

### Both grids

A plan can hold the window grid and the 0.25 s fine grid at the same time. Set
`fine=on` on the `plan` command, or tick Both grids on the page. One dense cut
then serves both paths, because the cut rate divides by both steps: a plan of
1 frame a second gives a 4 fps cut, and a plan of 3 fps gives 12 fps. One pass
writes both kinds of record, and one search ranks them together. The flag adds
the grid beside the windows. It never moves the windows.

A video without the flag keeps its old shape.

The model is `google/embeddinggemma-2`, in the `Q8_0` format, with its
multimodal projection. The engine is `llama.cpp`, linked into the binary
through the `llama-cpp-2` crate.

## Requirements

1. Windows with the MSVC build tools and the Visual Studio 2022 generator.
2. CMake on the path.
3. Rust, edition 2024.
4. ffmpeg and ffprobe.
5. Two model files.

## Setup

Put the two model files in `models/`:

```
models/embeddinggemma-2-Q8_0.gguf
models/mmproj-embeddinggemma-2-Q8_0.gguf
```

Put the ffmpeg binaries under `tools/dl/ff/`. The path sits in
`videoembed/src/media.rs` as the constants `FFMPEG` and `FFPROBE`.

Fetch the submodule after the first clone:

```
git submodule update --init
```

## Build

```
cd videoembed
CMAKE_GENERATOR="Visual Studio 17 2022" CMAKE_BUILD_PARALLEL_LEVEL=8 cargo build
```

The first build takes about ten minutes. It compiles `llama.cpp`.

An MSVC file lock can stop the build with `os error 32`. Wait ten seconds and
run the same command again.

### Build with a CUDA card

Add the `cuda` feature to run the model on an NVIDIA card:

```bash
cd videoembed
CMAKE_GENERATOR="Visual Studio 17 2022" CMAKE_BUILD_PARALLEL_LEVEL=20 \
  CMAKE_CUDA_ARCHITECTURES=86 cargo build --release --features cuda
```

Set `CMAKE_CUDA_ARCHITECTURES` to the code of your card. The RTX 3050 Ti is
`86`. Leave the value out and the build makes every architecture, which takes
about seven times longer.

On a 30 s clip the CUDA build indexes 29 windows in 7 s. The CPU build takes
119 s. The two sets of vectors agree at a cosine of 0.9998.

## Run

Run from the `videoembed` folder. The model path and the store path resolve
from there.

```
./target/debug/videoembed.exe
```

The app prints a prompt. It also takes one command as an argument, which suits
a script.

| Command | Action |
|---|---|
| `index <path>` | Cut the video under its plan. Embed each item. |
| `plan <path> <preset>` | Cut the video again under a new plan. |
| `device gpu\|cpu` | Load the model on that device. |
| `note <text>` | Add one free text record. |
| `search <query> [\| <path>]` | Rank every record, or one video. Show the top 10, and mark a weak row. |
| `list` | Group the records by video. |
| `forget <path>` | Drop one video from the index. |
| `quit` | Stop. |

The default plan is Object: a 2 s window on a 1 s stride, with 2 frames and the
window audio. The probe measured that shape at 10 hits of 10 with the audio,
against 6 of 10 without it. The presets are Object, Action, Speech, and Fine.
The Fine preset keeps the 0.25 s frame grid. A video that the store does not
know takes the default.
An older video keeps the Fine plan that it was cut under.

The `plan` command also reads values, as `plan <path> window=4 frames=4
audio=on fine=on`. The window takes 1 to 10 seconds, and the frame count stops
at the window in seconds. A preset holds no fine grid, so set `fine` by hand.

Start the app with `--gpu` or `--cpu` to pick the device. The default is the
card when the card is here.

The store sits at `videoembed/store/index.db`. Both the console and the web
binary use that one file. The format is SQLite with the sqlite-vec extension.
One write is one transaction, so a crash cannot corrupt it. The app imports an
old `VEI1` store one time at open, when the database holds no record. A database
from an older version gains its new columns at open, so no manual step is
needed.

Set `VEI_MODELS` to point at another model folder.

## Web UI

Run the web binary from the `videoembed` folder:

```bash
./target/debug/web.exe
```

The app opens `http://127.0.0.1:8787/` in your browser.

1. Press Choose folder and pick a folder.
2. Press Index. The bar shows the file and the window.
3. Type a query. The hits appear under Search. Each hit names its kind: window,
   frame, or audio.
4. Press a hit. The player jumps to that moment.
5. Press the name of a video to show its plan. Pick a preset, or move the two
   sliders, and tick Both grids. Press Apply to cut the video again.
6. Press Forget on a row to drop one video from the index.
7. Press Quit to stop the server.

The Search list shows one row for each moment.

- Two rows of one video that sit within 0.5 s join one row, and the row shows
  the whole range.
- The Pause button holds the running job. The job keeps its place and its cut
  files. The button reads Resume while the job waits.
- A folder run clears the finished mark of a video that holds no records. The
  server prints the count when the count is not zero.
- The device pick loads the model on the card or the processor. A pick of the
  device in use costs no load, so the click answers at once.
- The speed line shows the load of the model in milliseconds beside the work of
  the job. It also lists the last five runs, so a slow run stays visible.

The server binds to `127.0.0.1` only, so another machine cannot reach it.

Each `/api/` call carries the header `X-VEI: 1`. A foreign page cannot set that
header, and the server sends no CORS headers, so its request fails. The player
needs one exception. A `<video>` element cannot set a header, so `/api/video`
takes a random key in the query instead. The page reads that key from
`/api/state`.

| Endpoint | Method | Action |
|---|---|---|
| `/api/state` | GET | The version, the record count, the video count, and the video key. |
| `/api/progress` | GET | The progress object of the running job. |
| `/api/videos` | GET | Every indexed video, with its item count and its plan. |
| `/api/search` | POST | Rank the store, or one video, and group the hits. The reply names the weak cut. |
| `/api/pick-folder` | POST | Open the folder dialog. |
| `/api/index` | POST | Index every video under a folder. |
| `/api/plan` | POST | Set the plan of one video. The reply names the item count. |
| `/api/device` | POST | Load the model on the card or the processor. A pick of the device in use reloads nothing. |
| `/api/stop` | POST | Stop the running job. The reply says whether a job ran. |
| `/api/pause` | POST | Hold or release the running job. The reply names the new state. |
| `/api/forget` | POST | Drop one video from the index. |
| `/api/video` | GET | The video bytes, with byte ranges. |
| `/api/quit` | POST | Stop the server. |

## Repository layout

| Path | Role |
|---|---|
| `videoembed/` | The Rust crate. The app and its modules. |
| `vendor/llama-cpp-2/` | The wrapper crate, with five pointer accessors added. |
| `vendor/llama-cpp-sys-2/` | The FFI crate, with the upstream project as a submodule. |
| `spike/` | The Python checks that measure the Rust port. |
| `docs/superpowers/specs/` | The design documents. |
| `models/` | The model files. Not in git. |
| `tools/dl/` | The ffmpeg download. Not in git. |

## Verification

`spike/06_video_search.py` makes the same index in Python, and
`spike/07_parity_rust_store.py` compares the two. The vectors agree at a cosine
of 0.96 to 0.99 for every window.

Run the Python check from the `spike` folder with the machine learning
environment:

```
VEI_WIN=2 VEI_STRIDE=1 VEI_SUFFIX=_w2 python 06_video_search.py index testvideo_nolabels.mp4
python 07_parity_rust_store.py index_testvideo_nolabels_w2.npz
```

Four queries work against `testvideo_nolabels.mp4`:

| Query | Expected window | True scene |
|---|---|---|
| `birthday party` | 21.0-23.0 s | 20-30 s |
| `blue` | 12.0-14.0 s | 10-20 s |
| `sunny weather` | 0.0-2.0 s | 0-10 s |
| `tuesday morning` | 11.0-13.0 s | 10-20 s |

## Known rough edges

1. The ffmpeg path is a fixed absolute path in `videoembed/src/media.rs`.
2. The console indexes one video. The web page indexes a folder.
3. A plan change cuts the whole video again, and the video leaves the search for
   the length of the cut.
