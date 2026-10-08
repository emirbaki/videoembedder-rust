# videoembed

Search inside a video by text. The app embeds a video in short windows, then
returns the window that matches a query.

Example. A 30 s clip shows a party at 20 s. The query `birthday party` returns
the window 20.0-24.0 s with a score of 0.76.

## Status

| Part | State |
|---|---|
| Console app | Works. |
| Web UI | Designed, not built. See `docs/superpowers/specs/`. |
| GPU | Not used. The model runs on the CPU. |

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
| `index <path>` | Cut the video into 4 s windows on a 2 s stride. Embed each window. |
| `note <text>` | Add one free text record. |
| `search <query>` | Rank every record. Show the top 10. |
| `list` | Group the records by video. |
| `forget <path>` | Drop one video from the index. |
| `quit` | Stop. |

The stride is 2 s, so the windows overlap. A hit at 20.0-24.0 s can start at
18 s or 22 s in the true scene.

The store sits at `videoembed/store/index.vei`. Both the console and the future
web binary use that one file. The format is `VEI1`, and a crash cannot corrupt
it, because a write goes through a temp file and a rename.

Set `VEI_MODELS` to point at another model folder.

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
VEI_WIN=4 VEI_STRIDE=2 VEI_SUFFIX=_w4 python 06_video_search.py index testvideo_nolabels.mp4
python 07_parity_rust_store.py index_testvideo_nolabels_w4.npz
```

Four queries work against `testvideo_nolabels.mp4`:

| Query | Expected window | True scene |
|---|---|---|
| `birthday party` | 20.0-24.0 s | 20-30 s |
| `blue` | 12.0-16.0 s | 10-20 s |
| `sunny weather` | 0.0-4.0 s | 0-10 s |
| `tuesday morning` | 10.0-14.0 s | 10-20 s |

## Known rough edges

1. The ffmpeg path is a fixed absolute path in `videoembed/src/media.rs`.
2. The console has no progress line during a long index.
3. The app loads one video at a time. There is no folder scan yet.
