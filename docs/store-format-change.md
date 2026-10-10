# The store format change - analysis

Date: 2026-10-11
Status: analysis, no decision yet

## 1. The items that need the change

The deferred list holds seven features. Four of them touch the store:

| # | Item | Needs from the store |
| --- | --- | --- |
| 1 | Window mode: one 2 s window holds 2 frames and its audio | A new record kind, and the plan of the run |
| 8 | A per-video plan, with presets | The plan fields, stored beside the video |
| 9 | Upsert and delete by record id | An id for every record |
| 11 | Sliders for the window and the frame count | Item 8, and the plan of the run that filled the store |

Three items stay out. Item 6 (a scoped search) filters the rank. Item 7 (a
relative score cut) changes the display. Item 10 (MP4 normalisation) changes
the media step. None of the three needs a new format.

## 2. The store today

The facts below come from `videoembed/src/store.rs`.

| Fact | Where |
| --- | --- |
| One file holds every record. The magic is `VEI1`. | [store.rs:3](videoembed/src/store.rs#L3) |
| The header is `magic | u32 dim | u32 count`. | [store.rs:191](videoembed/src/store.rs#L191) |
| A record is `u8 kind | u32 label_len | label | f32 start | f32 end | f32 vec[dim]`. | [store.rs:196](videoembed/src/store.rs#L196) |
| The kinds are window, note, frame, and audio. | [store.rs:18](videoembed/src/store.rs#L18) |
| The label is the join key. A video path is the identity, in slash form. | [store.rs:34](videoembed/src/store.rs#L34) |
| The vector is `f32`, so one record carries 3,072 bytes at `dim = 768`. | [store.rs:202](videoembed/src/store.rs#L202) |
| One write rewrites the whole file, through a temp file and a rename. | [store.rs:220](videoembed/src/store.rs#L220) |
| `append` and `forget` both rewrite the whole file. | [store.rs:123](videoembed/src/store.rs#L123) |
| The open call reads the whole file into memory. | [store.rs:81](videoembed/src/store.rs#L81) |
| A rank scores every record, then sorts the list. | [store.rs:235](videoembed/src/store.rs#L235) |
| The finished videos live in a second file, `store/index.done`. | [store.rs:38](videoembed/src/store.rs#L38) |

The record holds no id, no timestamp, no source fingerprint, and no plan. The
plan of a run exists only in the code that ran it, so a later session cannot
tell four frames a second from two frames a second by reading the store.

## 3. The limits, in numbers

The spec `docs/superpowers/specs/2026-10-10-videoembed-product-routes.md` gives
the scale. One hour of video at 4 frames a second holds 14,400 frame records.

| Limit | At 14,400 records | At 100,000 records |
| --- | --- | --- |
| The vector bytes, `f32` | 44 MB | 307 MB |
| The vector bytes, `fp16` | 22 MB | 154 MB |
| The whole file in RAM | 44 MB | 307 MB |
| One rewrite for one new video | 44 MB to write | 307 MB to write |
| One query | 14,400 dots | 100,000 dots |

A `f32` store costs twice the disk of the fp16 plan in the spec, and the spec
already names fp16 as a shared lever. The rewrite cost is the sharper limit: a
second video of one hour makes the append rewrite the first hour again.

## 4. What each item needs

**Item 1, window mode.** A window record holds more than one frame, so the
record needs the plan that built it: the window seconds, the stride, the frame
count in the window, and the audio flag. The frame kind stays for the fine
grid, and the window kind sits beside it.

**Item 9, upsert and delete by id.** An id per record gives a target for a
delete and a replace. Today `forget` matches every record whose label equals
one path, and every append adds records. A re-index without a `forget` first
doubles the records of one video.

**Items 8 and 11, the plan and the sliders.** The plan belongs to the video,
not to one record. Two shapes exist: repeat the plan inside every record, or
keep one plan row for each video. The repeat is simpler and it needs no second
section. At 16 bytes for each record it costs about 230 KB in a one-hour store.

## 5. The three shapes

**Shape A, the flat file at version 2.** Keep one file and one rewrite. Add a
version word, an id per record, the plan fields, and an optional `fp16` vector.
The change is small, and the tests for the current format carry over.

**Shape B, an append log with a sidecar.** The record file becomes a log: an
append writes only the new records, a delete writes a tombstone, and a
compaction merges the log on request. A sidecar holds the ids, the plans, and
the finished list. This gives a true upsert and a cheap append, and it removes
the second file, `index.done`, by folding it in. The cost is one more file to
keep consistent.

**Shape C, an embedded database.** A store such as SQLite with a vector
extension, or `redb` with a vector index crate. It brings transactions, ids,
and a mature format. It also brings a new dependency, a build step on Windows,
and a migration for every existing store. The project rule is one binary with
CUDA as an option, so a heavy dependency works against that rule.

| | A: flat file v2 | B: log and sidecar | C: embedded database |
| --- | --- | --- | --- |
| New dependency | none | none | one, with a C build |
| Format change | small | medium | large |
| Append cost | full rewrite | one record | one record |
| Delete by id | yes, with a rewrite | yes, cheap | yes |
| Memory limit | whole file | whole file | the database page cache |
| Unlocks items 1, 8, 9, 11 | yes | yes | yes |
| Migration of an old store | a read of `VEI1`, then a write of `VEI2` | same | a full import |
| Risk | low | medium | high |

## 6. The recommendation

Shape A now, and shape B when a store passes a size where one rewrite hurts.

The reasons: the items in scope need fields, not a new engine. Shape A adds the
fields and keeps every current test. The spec puts the ANN index and the shard
work in Route B, so the store stays a single file for the local tool. The fp16
vector lands inside shape A, and it halves the disk and the memory at once.

Shape B waits for one of two signals: a store that holds several long videos,
or a delete that must not rewrite the file. Shape C waits for a server, where
transactions and many writers matter.

## 7. Open questions

1. Which shape: A, B, or C?
2. Does window mode land in the same change, or after the format is done? Two
   steps are safer, because the format change alone is testable on its own.
3. Does the store need a source fingerprint (size and mtime)? Today a changed
   file with the same path stays marked as finished, so the run skips it.
4. Do the plans repeat inside every record, or do they get one row for each
   video?
5. Does the `.done` list fold into the store in shape A too, or does it stay a
   sidecar until shape B?

## 8. What stays the same

- The label rule, `normalize_label`, and the slash form of a path.
- The kind list, and `is_video`.
- The temp-file-then-rename write, so a crash cannot leave a half store.
- The one vector size for the whole file.
- The store path, `store/index.vei`.
