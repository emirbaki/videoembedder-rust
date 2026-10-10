# videoembed product routes - design

Date: 2026-10-10
Status: draft for review

## 1. Goal

One engine searches video and audio at the frame level. A user types "a man
with a ball" and gets the exact frame, plus a timeline of near hits. Three
products share the engine: a local tool, a web service, and a mobile app.

## 2. Fix the score before the granularity

The scoring doubt is real, and it comes first. Today "a man with a ball" gives
0.65 to a wrong frame and 0.70 to the right one. That margin of 0.05 is noise.

Root cause: an embedding model ranks, it does not calibrate. Cosine scores
compress into a narrow band. The band makes the absolute number meaningless.

Frame level indexing does not fix this. It gives more scores in the same narrow
band. The fixes are these, in order:

1. Calibrate the score. Show the query score as a margin over the median frame
   of that store. A z-score works too. This is cheap, and it fixes the display.
2. Rerank the top hits. Run a small second model on the top 20 frames. That
   model scores the query and the frame jointly, and it gives a wide 0.0 to
   1.0 margin.
3. Raise the floor. Drop any frame whose score sits below a per-query cutoff.

These three apply to every route below.

## 3. One frame representation

- A frame vector holds 768 floats at fp16, or 1536 bytes.
- A frame record holds the image only. It runs at 4 frames a second.
- An audio record holds one second of audio. It runs at 1 a second. The two
  kinds stay apart, so the encoder reads each audio second one time.
- The frame rate is adaptive. The default is 4 frames a second. 30 frames a
  second is a fine-cut option only.

Storage for one hour of video:

| Rate | Frames | Video vectors | Audio vectors | Total |
|---|---|---|---|---|
| 4 fps | 14,400 | 22 MB | 5.5 MB | 28 MB |
| 30 fps | 108,000 | 166 MB | 5.5 MB | 172 MB |

Encode time on one RTX 3050 Ti, measured at about 30 ms per image:

| Rate | Frames per hour | Encode time |
|---|---|---|
| 4 fps | 14,400 | 7 minutes |
| 30 fps | 108,000 | 54 minutes |

## 4. The three routes

### 4.1 Route A - the local tool

For a studio or a creator. The engine runs on their own card. The corpus is
their own drives. No server, no upload.

Optimizations:

1. Keep the CUDA build. It is done, and it gives 17 times.
2. Encode each frame once. Today overlapping windows encode a shared frame
   twice. Frame level removes that.
3. Batch many images in one vision call.
4. Decode frames on the GPU with NVDEC, not with CPU JPEG.
5. Store fp16.

### 4.2 Route B - the web service

For many users. The index lives in the cloud. A user uploads a video, waits,
then searches. The corpus is one shared index, so one query can cross many
videos.

Optimizations:

1. Run the encode on a server GPU, in batches and in a queue.
2. Shard the index by video. Rank each shard on its own worker.
3. Add an ANN index, HNSW or FAISS, for top-k over millions of vectors.
4. Store fp16. Keep hot videos in memory and cold ones on object storage.
5. Rerank the top hits with a cross-encoder, since this is where the margin fix
   matters most.

Costs: GPU hours for the encode, plus storage. The ANN index and the queue are
the new parts.

### 4.3 Route C - the mobile app

For a casual user on a phone. A meme search over saved videos.

On-device indexing is heavy. The model is about 1 GB. Two choices exist:

1. Hybrid. The phone uploads the video. The cloud indexes it and returns hits.
   The phone shows the slider. This is the practical MVP.
2. On-device. A small quantized model, Q4, runs on Core ML or a GPU delegate.
   This is slower, and it is a later goal.

Optimizations for the hybrid path:

1. Reuse Route B. The phone is a thin client.
2. Stream hits as the index builds, so the user sees progress.
3. Cache the slider locally after the first search.

## 5. Shared engine levers

These help every route:

| Lever | Effect |
|---|---|
| Encode each frame once | Removes the overlap duplication. |
| Batch images per vision call | Cuts the per-call overhead, up to 10 times. |
| NVDEC frame decode | Moves the frame cut from CPU to GPU. |
| fp16 storage | Halves the disk and the memory. |
| Scene-change keying | Cuts frames with no change, 5 to 20 times. |
| Rerank tier | Fixes the score margin. |

## 6. The ANN index question

Brute force cosine is fine to about 1 million vectors. Beyond that, one query
scans too much. Add HNSW or FAISS. This matters only for Route B.

## 7. Decisions

1. Route A is first. The engine is here, and it needs no server work.
2. The default frame rate is 4 a second.
3. The MVP ships the cheap calibration. The rerank tier comes after.

## 8. Why Route A first

Route A validates frame search on the current engine. The cheap calibration
lands in the same pass. The rerank tier follows, because it is the real fix for
the 0.65 against 0.70 problem. The ANN index waits for Route B.
