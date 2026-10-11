# Comparison: google-ai-edge/gallery against this app

Date: 2026-10-10
Source of the comparison: the public repository `google-ai-edge/gallery`, branch `main`.
Method: a code read only. I did not build or run the Android app. Every Google fact below cites a file and a line.

## 1. Why this comparison matters

Google ships a feature that does almost the same job as this app. It is called the Video Moment Finder. It takes a video, indexes it with the same Gemma embedding model, and answers a text query with time ranges. It runs on a phone, so it fights a hard memory and speed budget. Its answers to that budget are worth reading.

I read six areas of the repository:

| Area | Package |
|---|---|
| The embedding service | `services/semanticretrieval/` |
| The video search feature | `customtasks/videomomentfinder/` |
| The photo search feature | `customtasks/smartalbum/` |
| Media handling | `services/photolibrary/`, `common/ImageUtils.kt` |
| Models and hardware | `data/Model*.kt`, `common/HardwareUtils.kt` |
| Speed measurement | `ui/benchmark/`, `common/metrics/` |

## 2. The closest analogue: Video Moment Finder

This feature is this app, built by Google, on a phone.

### 2.1 The indexing plan

| Item | Google Video Moment Finder | This app |
|---|---|---|
| Unit of one record | One window | One frame, or one audio second |
| Window length | 2 s default (`DEFAULT_WINDOW_DURATION_SEC`) | 4 fps, so 0.25 s |
| Frames in a window | 2 (`DEFAULT_FRAMES_PER_WINDOW`) | 1 |
| Window overlap | 0 s (`DEFAULT_OVERLAP_DURATION_SEC`) | Not applicable |
| Audio | In the same window record | A separate record |
| Records for a 41.5 s video | About 21 | 207 (166 frames plus 41 audio) |
| Embedding calls for the same video | About 21 | 207 |

The last row is the important one. **Google makes about ten times fewer embedding calls for the same video.** One call embeds a whole window: two frames and their audio, in one vector. This app embeds every frame on its own and every audio second on its own.

That difference explains the whole speed gap. The GPU build fixed the wall-clock time for this app. The window model would cut the work itself.

The two designs trade precision for work:

- This app resolves a match to 0.25 s. Good for "find the exact second".
- Google resolves a match to 2 s, but its vector holds sound and picture together. Good for "find the moment".

Google lets the user move that line. It offers three presets and three sliders (`ProcessingConfigDialog.kt:72-89,157-230`):

| Preset | Window | Frames | Audio |
|---|---|---|---|
| Object | 2 s | 2 | No |
| Action | 4 s | 4 | No |
| Speech | 6 s | 6 | Yes |

The sliders run from 1 s to 10 s for the window, and from 1 to `windowDuration` for the frame count.

### 2.2 How a window becomes a model input

Google interleaves text, audio, and image parts in one call (`AudioInterleaving.kt:63-92`):

```
[timestamp][audio][timestamp][image]
```

The frame times sit inside the sequence as text, for example `"00:03"` (`EmbedPart.kt:35`). The model then sees the order of events, not only the pictures.

Google measures its audio budget from the mel frontend. The frontend makes 100 mel frames a second. A shrink factor of 16 gives 6.25 audio tokens a second (`AudioInterleaving.kt:29-33`). The estimator is a deliberate upper bound. When a window would overflow the 1024 token budget, Google drops the audio for that window instead of failing (`VideoMomentFinderViewModel.kt:1299-1313`).

If the engine rejects audio once, the whole pass restarts with frames only (`VideoMomentFinderViewModel.kt:1389-1410`).

### 2.3 Storage and the time map

| Item | Google | This app |
|---|---|---|
| Store | SQLite through MediaPipe `SqliteVectorStore` | One SQLite database, `store/index.db` |
| Table | `rag_vector_store` | `videos`, `records`, `vec_records` |
| Record id | `"<projectId>/<startTimeMs>-<endTimeMs>"` | `"path/startMs-endMs"` |
| Time map | `start_time_ms` and `end_time_ms` metadata | `start` and `end` fields |
| Write path | Idempotent upsert by id | An upsert by id, in one transaction |
| Delete | By id, or by id prefix | By id, one video at a time |

Google keeps the time range in metadata and reads it back after a search (`VideoMomentFinderViewModel.kt:777-790`). The record id already holds the range, so a rewrite is idempotent. A second pass over the same video overwrites the same rows.

This app stores the range in the record itself. That is simpler. But a re-index of one video cannot update in place. The whole file rewrites.

### 2.4 The search and the score

| Item | Google | This app |
|---|---|---|
| Result count | `DEFAULT_TOP_K = 5` | Floor 0.4, then a cap of 50 for each video |
| Score threshold | None | An absolute floor of 0.4 |
| Score shown | Raw cosine, `%.2f` | Score and margin over the median |
| Score scale | Absolute | Relative to the store median |
| Normalisation | L2, then cosine | Cosine, with an image-only frame vector |
| Scope filter | `parent_id` metadata filter | None, the whole store searches |
| Merge of near hits | Yes, within 500 ms | No |

Google uses a scoped search. It filters the store to the selected project before it ranks (`VideoMomentFinderViewModel.kt:758-762`). This app ranks every record and groups the result by video afterwards.

Google merges results that sit within 500 ms of each other into one interval (`VideoMomentFinderViewModel.kt:1020-1036`). This app shows every hit.

Google shows a raw cosine value and applies no cutoff in this feature. A weak result still appears. The photo feature is stricter: it greys out every card below `max(0.40, topScore - 0.20)` (`SmartAlbumSearchScreen.kt:829-831`). That is a relative cut, anchored to the best score. This app anchors its cut to the median instead, and shows the gap as a margin.

### 2.5 The results on screen

Google's result card holds three things. This app's row holds text only.

| Element | Google | This app |
|---|---|---|
| Preview | A video thumbnail | None |
| Score | Cosine, `%.2f` | Score and margin |
| Time range | `MM:SS - MM:SS` | `start-end s` |
| Timeline | A `Canvas` bar, 48 dp high, 8 dp track | A gradient bar, 1.1 rem high |
| Bar marks | An interval marker for each hit | A white peak line for the best hit |
| Selection | A 4 dp border | None |
| Layout | A horizontal row, 12 dp padding | A list |

The Google bar marks **every** merged interval (`ProjectDetailsScreen.kt:1190-1250`). This app's bar paints every cell with a colour, and marks only the best hit with a line. The two designs answer different questions. The Google bar says where the matches are. The colour bar here says how strong each part is.

A thumbnail is the largest user-experience gap. A time range in text makes the user click to learn what the moment holds.

### 2.6 Long jobs

| Item | Google | This app |
|---|---|---|
| Cancel | Yes | No |
| Resume | Yes, indexed ids skip | No, a restart begins at zero |
| Pause | Yes | No |
| Progress | Windows done of total | Items done of total |
| Failure | Partial records are dropped | The run stops |
| Stale records | Removed by id prefix before a rewrite | Left in place |

Google deletes the stale records of a project first, then writes the new ones (`VideoMomentFinderViewModel.kt:1225-1227`). It drops partial records when a pass fails (`VideoMomentFinderViewModel.kt:1465-1466`). On start, it marks any project that lost its vectors as unprocessed (`VideoMomentFinderViewModel.kt:1067-1095`). That is a reconciliation step. It keeps the store and the file list consistent.

The photo feature adds a worker with a notification, a pause button, and a per-item delay of 100 ms while the gallery scrolls (`SmartAlbumIndexingWorker.kt:344-345,540-561`).

### 2.7 Media handling

Google does four things this app does not.

1. **It rewrites a fragmented MP4 into a progressive MP4** before it indexes (`VideoMomentFinderViewModel.kt:331-340`). Seeking works after the rewrite. This app meets the same class of fault as a corrupt audio track, and reports it as an empty audio clip.
2. **It probes several offsets for a thumbnail**, at 0, 200, 500, 1000, 2000, and 4000 ms, and skips a black opening frame (`VideoThumbnails.kt:35-62`).
3. **It checks decoder availability** before a long job starts (`VideoDecoderSupport.kt:52-92`).
4. **It caps the input at 5 minutes** for the video feature (`VideosTab.kt:62`).

For pictures, Google decodes at a target dimension of 1024 for indexing (`SmartAlbumIndexingWorker.kt:288`), and it keeps an LRU bitmap cache of one eighth of the heap, keyed by asset id and size (`PhotoLibraryService.kt:204-247`).

### 2.8 Models and hardware

| Item | Google | This app |
|---|---|---|
| Backend choice | CPU, GPU, NPU, TPU in a listed order | A build flag, `--features cuda` |
| Fallback | A non-CPU failure falls back to CPU | None |
| Audio backend | Always CPU | The same context as the frames |
| Device gate | `minDeviceMemoryInGb`, a warning | None |
| Download | Range resume, `.gallerytmp` file | The user fetches the model |

Google tries the preferred backend and falls back to the CPU when init fails (`OnDeviceEmbedder.kt:374-409`). It forces the audio encoder onto the CPU in every branch (`OnDeviceEmbedder.kt:318-333`). This app uses one llama context for all three modalities and one GPU setting.

### 2.9 Speed measurement

Google measures speed in detail, and this app measures nothing.

| Metric | Google | This app |
|---|---|---|
| Cold init | Reported apart from warm init | Not measured |
| Steady init | Reported | Not measured |
| Time to first token | Reported | Not measured |
| Prefill speed | Tokens a second | Not measured |
| Decode speed | Tokens a second | Not measured |
| Runs | 1 to 10, default 3 | Not applicable |
| Aggregate | Min, max, mean, median | Not applicable |
| Raw samples | Kept for each run | Not applicable |

Google never blends the numbers into one figure (`benchmark.proto:50-60`). It times with a monotonic clock (`LitertlmInferenceMetricsTracker.kt:261`). It counts the decode speed over `outputTokens - 1`, so the first token does not count twice (`LitertlmInferenceMetricsTracker.kt:382-401`).

Two honest notes from this area:

- The benchmark computes p25 and p75 but never shows them (`BenchmarkViewModel.kt:423-424`).
- The photo indexer reports a count and a percent only. It does not report items a second. It also throttles itself, so its wall-clock time understates its real speed (`SmartAlbumIndexingWorker.kt:344-345`).

## 3. Where this app is ahead

1. **Task prefixes.** This app gives the query the prefix `task: search result | query: ` and the media the prefix `title: none | text: ` (`embed.rs:21-22`). Google passes the text raw (`SemanticRetrievalService.kt:619`). The prefix is a real accuracy gain, because the model was trained to separate the two sides of a search.
2. **An honest score scale.** Google shows a raw cosine. A cosine value alone has no scale, so 0.65 and 0.70 look alike. This app shows the gap over the store median, and the bar turns that gap into colour.
3. **Finer time resolution.** 0.25 s against 2 s. For a query about a single instant, this app is more precise.
4. **Audio encoded once.** This app encodes each audio second one time. Google re-reads the audio of every window. With an overlap of zero the waste is small, but the multiplication grows with overlap.
5. **A small database dependency.** One crate, `rusqlite`, with the bundled
   SQLite. Google needs SQLite, MediaPipe, WorkManager, and Proto DataStore.
6. **A GPU path with all layers offloaded.** `N_GPU_LAYERS = 999` gives the measured 40 times speedup.

## 4. Where Google is ahead

1. **Ten times fewer embedding calls**, from the window model.
2. **A thumbnail in each result**, so the user sees the moment.
3. **A cancellable, resumable, reconciling index job.**
4. **A scoped search** by project, instead of a rank over the whole store.
5. **Merged intervals**, so near hits become one moment.
6. **User control of the plan**, through presets and sliders.
7. **A measured speed report**, instead of a guess.
8. **Media repair**: the MP4 rewrite and the black-frame probe.
9. **A fallback path** when the chosen accelerator fails.

## 5. Candidate features, ranked

The rank weighs the user gain against the cost.

| # | Feature | Gain | Cost | Risk |
|---|---|---|---|---|
| 1 | Window mode: pack frames and audio into one record | About 10 times less work | High | Changes the store format |
| 2 | A thumbnail in each result | Large | Medium | Needs a new server route |
| 3 | Resumable and cancellable indexing | Large | Medium | Needs per-video state |
| 4 | An inline speed report | Medium | Low | None |
| 5 | Merge near hits into one moment | Medium | Low | Changes the hit list |
| 6 | A scoped search by video | Medium | Low | Needs a filter on rank |
| 7 | A relative score cut, like Google's greyout | Medium | Low | Changes the result count |
| 8 | A per-video plan, with presets | Medium | Medium | Needs plan fields |
| 9 | Upsert and delete by record id | Medium | High | Needs a real store format |
| 10 | MP4 normalisation and a black-frame probe | Medium | Medium | Needs ffmpeg work |
| 11 | Sliders for window and frame count | Small | Low | Follows item 8 |

### Notes on the top four

**1. Window mode.** This is the only change that attacks the work itself rather than the wall clock. A 2 s window with 2 frames and audio cuts 207 calls to about 21 for the same 41.5 s clip. An A/B test would settle the accuracy question: index one video both ways, run the same ten queries, and compare the returned seconds against the truth. The cost is a new record kind and a re-index of every video. A hybrid is possible: keep the 4 fps frames for precision, and add windows for speed on long videos.

**Measurement (2026-10-10).** The `measure` command ranks a store on a 0.25 s, 0.5 s, 1 s, and 2 s grid. The run uses the vectors already in the store, so it costs one text embedding for each query. It does not measure the audio packing or the timestamp text.

On a 572 record store of four copies of `spike/testvideo_nolabels.mp4`:

| factor | grid | frame records | blue | a person talking | music | tuesday |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.25 s | 480 | 10.00 s | 5.00 s | 3.00 s | 11.00 s |
| 2 | 0.50 s | 240 | 10.00 s | 5.00 s | 3.00 s | 11.00 s |
| 4 | 1.00 s | 120 | 10.00 s | 5.00 s | 3.00 s | 11.00 s |
| 8 | 2.00 s | 60 | 10.00 s | 5.00 s | 3.00 s | 11.00 s |

The best frame holds its second at every factor, and the record count falls eightfold. A coarser grid keeps frame 0, 4, 8, and so on. A peak that sits on a kept sample survives every factor. A peak between two samples can move by half of the new step. One clip cannot show that case, so the window mode decision stays open.

The merged moment does not measure the grid. It spans every touching hit above the floor, so it widens from 10.00-20.00 s to the whole video as the grid coarsens.

**2. A thumbnail per result.** This is the largest gain for the least architectural change. The server already serves video by range, and ffmpeg already cuts frames. One route, `/api/frame?path=&t=`, returns a JPEG at a time. The card then shows the picture, the score, and the range, as Google does.

**3. Resume and cancel.** A 20 minute job that dies at minute 19 wastes the whole run. Google records which ids are indexed and skips them on the next pass. This app needs a per-video record of completion, in the store or beside it, plus a cancel flag that the run checks between items. The progress object already exists, so the plumbing is small.

**4. A speed report.** The work is small and the payoff is trust. Show items a second, the wall-clock time, and the backend. Time with a monotonic clock. Report cold and warm runs apart, as Google does. Keep the raw samples so variance stays visible.

**Measurement (2026-10-11).** The run-time mode picks GPU or CPU for the loaded model. One release binary with the `cuda` feature serves both modes. The `web` server indexed `spike/testvideo_nolabels.mp4` on each device in turn. The run holds 143 items.

| device | load | work | items a second |
| --- | --- | --- | --- |
| GPU | 2.13 s | 7.57 s | 18.88 |
| CPU | 1.26 s | 232.30 s | 0.62 |

The card is about 31 times faster on this clip. The card costs about a second more at load. Both modes returned 143 records for the same 143 items.

## 6. What to keep as it is

- The task prefixes. Google does not have them. Do not trade them away for parity.
- The median baseline and the margin. It is the reason a grey bar can say "no match".
- The 0.25 s frame grid, at least for short clips.
- The single binary, with CUDA as an option rather than a requirement.
