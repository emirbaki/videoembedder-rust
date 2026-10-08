# videoembed web UI - design

Date: 2026-10-08
Status: approved, ready for a plan

## 1. Goal

Give the video search app a user interface. The user picks a folder, the app
indexes the videos inside it, the user types a query, and the app shows the
matching moments. A player jumps to the chosen moment.

The console app already does the search. The UI adds a folder picker, a search
box, a result list, a timeline, and a player.

## 2. Scope

In scope for this version:

1. Folder index. Pick a folder and index every video inside it, with progress.
2. Search plus a timeline that marks each matching window.
3. A player that seeks to the chosen window.
4. A store manager that lists indexed videos and deletes one.

Out of scope:

- Notes in the UI. The console `note` command stays.
- GPU use, a release build, and a transcript view.
- User accounts and a network-facing server.

## 3. Current state

The package `videoembed` holds one binary, a console.

| File | Role |
|---|---|
| `src/main.rs` | The console loop and the commands. |
| `src/embed.rs` | The model, the joint batch, text and window embedding. |
| `src/media.rs` | ffprobe duration, frame and audio extraction with ffmpeg. |
| `src/store.rs` | The `VEI1` store file, append, forget, rank. |

A window is 4 s long and the stride is 2 s. The store sits at `store/index.vei`
and holds one vector for each window and note.

## 4. Design

### 4.1 Shape

The package gains a second binary, `web`. It serves a page on
`127.0.0.1:8787` and opens the default browser.

To share code, the three modules move into a library target. Both binaries use
the library. The console binary keeps every command it has.

```text
videoembed/
  Cargo.toml
  src/
    lib.rs        exports embed, media, store, index
    index.rs      scan a folder, index one video, report progress
    main.rs       the console, unchanged behavior
    bin/web.rs    the server, the endpoints, the engine thread
  assets/
    index.html
    app.js
    app.css
```

### 4.2 One engine thread

The model lives on one thread. That thread also owns the store. A request
arrives over a channel, and the answer leaves over a second channel.

Reasons:

1. llama.cpp runs best on one thread.
2. The wrapper types are not proven `Send`, so the model stays on its own
   thread and never crosses a thread boundary.
3. Search and index run one at a time. That is correct for a CPU-bound job, and
   it removes every lock around the store.

The server thread never touches the model. It sends a job and waits for the
answer.

### 4.3 HTTP API

All calls return JSON. A failed call returns an object with an `error` field
and a 4xx or 5xx code.

| Method | Path | Body | Answer |
|---|---|---|---|
| GET | `/` | - | The page. |
| GET | `/app.js`, `/app.css` | - | The assets. |
| GET | `/api/state` | - | Version, record count, video count. |
| POST | `/api/pick-folder` | - | `{"path": "..."}` or `{"path": null}`. |
| POST | `/api/index` | `{"path": "..."}` | `{"files": 4}` or an error. |
| GET | `/api/progress` | - | The progress object below. |
| POST | `/api/search` | `{"query": "..."}` | `{"groups": [...]}`. |
| GET | `/api/videos` | - | `{"videos": [{"path": "...", "windows": 14}]}`. |
| POST | `/api/forget` | `{"path": "..."}` | `{"removed": 14}`. |
| GET | `/api/video?path=...` | - | The video bytes, with range support. |
| POST | `/api/quit` | - | `{"stopping": true}` and the process exits. |

The progress object:

```json
{
  "running": true,
  "folder": "D:/video",
  "file": "D:/video/a.mp4",
  "file_index": 2,
  "file_count": 4,
  "window": 7,
  "window_count": 14,
  "indexed": 21,
  "error": null
}
```

The page polls `/api/progress` twice a second while a job runs.

The search answer groups the hits by video:

```json
{
  "groups": [
    {
      "video": "D:/video/a.mp4",
      "duration": 30.0,
      "score": 0.78,
      "hits": [{"start": 20.0, "end": 24.0, "score": 0.78}]
    }
  ]
}
```

A group holds every window of that video whose score sits above a floor of
0.4, capped at 50 windows. The groups sort by their best score.

### 4.4 The index job

`POST /api/index` does these steps:

1. Check that the path is a directory.
2. Refuse the call with a 409 code if a job already runs.
3. Scan the folder for the extensions `mp4`, `mkv`, `mov`, `avi`, and `webm`.
   The scan walks into subfolders.
4. Start a thread that runs the job.
5. Return the file count at once, so the page can show progress.

The job, for each file:

1. Probe the duration with ffprobe.
2. Drop the old records of that file, so a re-index replaces them.
3. Cut the windows on a 2 s stride and embed each one.
4. Append the new records after each file, not at the end, so a stop keeps the
   work that is done.
5. Write the progress object after each window.

An error on one file goes into the progress `error` field. The job moves to the
next file.

### 4.5 Player and byte ranges

The page uses the browser `<video>` element. Play, pause, and seek come from the
browser.

Seeking needs byte ranges. `GET /api/video` must read the `Range` header and
answer with a 206 code, a `Content-Range` header, and the requested bytes. A
call without a `Range` header returns the whole file with a 200 code.

The handler reads the file in chunks of 64 KiB and writes them to the socket.
It never loads a whole video into memory.

### 4.6 Timeline

Each group carries the video length and the hit windows. The page draws one bar
for the group. A mark sits at `start / duration * 100` percent of the width.

A click on a mark, or on a row, sets the player source and seeks to the window
start.

### 4.7 Safety

1. The server binds to `127.0.0.1` only.
2. `/api/video` serves a file only if the store holds a record with that exact
   path. A path outside the index gives a 403 code.
3. `/api/index` accepts a directory path only.
4. Every `/api/` call must carry the header `X-VEI: 1`. Another web page cannot
   set a custom header without a preflight, and the server sends no CORS
   headers, so that preflight fails. This blocks a request from a foreign page.

### 4.8 Store

The store format stays `VEI1`. The current `store/index.vei` keeps working, and
both binaries share it.

The timeline axis uses the largest window end as the video length. That value
can sit up to 4 s below the true length. The bar tolerates that, and the player
reads the true length from the browser.

## 5. Data flow

Index:

```text
page -> POST /api/index {path}
     -> server thread -> channel -> engine thread
     -> scan, cut, embed, append to store
     -> progress object (shared, read by the server thread)
page -> GET /api/progress (twice a second)
```

Search:

```text
page -> POST /api/search {query}
     -> server thread -> channel -> engine thread
     -> embed the query, rank the store, group by video
     -> JSON -> page -> rows and timeline marks
```

Play:

```text
page -> click a mark
     -> video.src = /api/video?path=...
     -> browser sends a range request
     -> server reads the range from disk and replies 206
     -> browser plays and seeks
```

## 6. Error handling

| Case | Behavior |
|---|---|
| The folder holds no video | `POST /api/index` returns 400 with a message. |
| A job already runs | `POST /api/index` returns 409. |
| The folder dialog closes with no pick | `{"path": null}`. The page keeps the old path. |
| ffmpeg fails on one window | The job records the error and skips the window. |
| The store file is missing | The store starts empty, as today. |
| The store file is damaged | The server exits and prints the parse error. |
| A video path is not in the index | `/api/video` returns 403. |
| The model fails to load | The process exits before the server starts. |

## 7. Components

| File | Change | Purpose |
|---|---|---|
| `videoembed/Cargo.toml` | edit | Add the new crates. |
| `videoembed/src/lib.rs` | new | Export `embed`, `media`, `store`, `index`. |
| `videoembed/src/index.rs` | new | Folder scan, one-video index, progress type. |
| `videoembed/src/main.rs` | edit | Use the library. Keep every command. |
| `videoembed/src/bin/web.rs` | new | Server, endpoints, engine thread, range reads. |
| `videoembed/assets/index.html` | new | The page. |
| `videoembed/assets/app.js` | new | Fetch calls, result list, timeline, player. |
| `videoembed/assets/app.css` | new | Layout. |

New crates:

| Crate | Use |
|---|---|
| `tiny_http` | The server. Small, blocking, no async runtime. |
| `serde`, `serde_json` | JSON in and out. |
| `rfd` | The native Windows folder dialog. |
| `open` | Start the default browser. |

### Interfaces

`src/index.rs` exports these items. Both binaries use them.

```rust
pub struct Progress { .. }              // Copy of the JSON object above.

/// List the video files under one folder, in name order.
pub fn list_videos(folder: &Path) -> Result<Vec<PathBuf>>;

/// Index one video and return its records.
/// The callback runs after each window.
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    on_window: &mut dyn FnMut(usize, usize),
) -> Result<Vec<Record>>;
```

`src/store.rs` gains one method for the safety check:

```rust
/// True when the store holds a window record with this exact path.
pub fn holds_video(&self, path: &str) -> bool;
```

## 8. Testing

Unit tests, run by `cargo test`. These need no model and no video:

1. Store round trip. Encode records, decode them, compare.
2. `list_videos` returns only the video extensions, and it walks subfolders.
3. `holds_video` is true for an indexed path and false for a foreign path.
4. The `Range` header parser: `bytes=0-99`, `bytes=100-`, and a bad header.

Manual checks, run by hand:

1. `cargo run --bin web` opens the browser at `127.0.0.1:8787`.
2. Pick the `spike` folder. The progress line moves, and the run ends.
3. Search `blue`. The top hit is a window inside 10-20 s.
4. Click the hit. The player loads the video and seeks to 12 s.
5. Drag the player bar. Playback continues from the new point. This proves the
   range code works.
6. Delete the video in the store manager. The row leaves the list.
7. Press quit. The process exits and the page stops.

## 9. Risks

| Risk | Answer |
|---|---|
| `tiny_http` does no range handling | The handler is manual and unit-tested. |
| The folder dialog blocks a worker thread | The dialog opens on its own thread, and the HTTP call waits for it. |
| A long index blocks search | Accepted. The engine thread serializes the work. The progress line shows why. |
| `embed.rs` uses raw pointers | Unchanged. The model never crosses a thread. |
| The refactor breaks the console | The console keeps its commands and its tests run first. |

## 10. Build estimate

About 2 hours. The page, the timeline, and the range code take most of it.

## 11. Open items

None. The four scope items, the folder dialog, and the browser shape are fixed.
