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

1. ffmpeg cuts a window into four JPEG frames and one WAV clip.
2. The model reads the frames, the audio, and a text marker in one batch.
3. One decode runs, and one pooling step produces one vector for the window.
4. The vector goes into a store. A query embeds the same way and ranks the store
   by cosine similarity.

Step 2 is the important part. The model sees the whole window at once. A
separate decode for each part would pool only the last part.

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
| `index <path>` | Cut the video into 2 s windows on a 1 s stride. Embed each window. |
| `note <text>` | Add one free text record. |
| `search <query> [\| <path>]` | Rank every record, or one video. Show the top 10, and mark a weak row. |
| `list` | Group the records by video. |
| `forget <path>` | Drop one video from the index. |
| `quit` | Stop. |

The stride is 1 s, so the windows overlap. A hit at 20.0-22.0 s can start at
19 s or 21 s in the true scene.

The store sits at `videoembed/store/index.db`. Both the console and the web
binary use that one file. The format is SQLite with the sqlite-vec extension.
One write is one transaction, so a crash cannot corrupt it. The app imports an
old `VEI1` store one time at open, when the database holds no record.

Set `VEI_MODELS` to point at another model folder.

## Web UI

Run the web binary from the `videoembed` folder:

```bash
./target/debug/web.exe
```

The app opens `http://127.0.0.1:8787/` in your browser.

1. Press Choose folder and pick a folder.
2. Press Index. The bar shows the file and the window.
3. Type a query. The hits appear under Search.
4. Press a hit. The player jumps to that moment.
5. Press Forget on a row to drop one video from the index.
6. Press Quit to stop the server.

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
| `/api/videos` | GET | Every indexed video with its window count. |
| `/api/search` | POST | Rank the store, or one video, and group the hits. The reply names the weak cut. |
| `/api/pick-folder` | POST | Open the folder dialog. |
| `/api/index` | POST | Index every video under a folder. |
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
2. The console has no progress line during a long index.
3. The console indexes one video. The web page indexes a folder.
