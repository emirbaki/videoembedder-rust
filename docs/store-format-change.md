# The store format change - analysis

Date: 2026-10-11
Status: the decision is SQLite, with the sqlite-vec extension. The schema
design is open.

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
| The header is `magic \| u32 dim \| u32 count`. | [store.rs:191](videoembed/src/store.rs#L191) |
| A record is `u8 kind \| u32 label_len \| label \| f32 start \| f32 end \| f32 vec[dim]`. | [store.rs:196](videoembed/src/store.rs#L196) |
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
and a mature format. It also brings a new dependency, a wider file set, and a migration for every
existing store. A database holds no vector type, so the search stays ours.

| | A: flat file v2 | B: log and sidecar | C: embedded database |
| --- | --- | --- | --- |
| New dependency | none | none | one crate, and a wider file set |
| Format change | small | medium | large |
| Append cost | full rewrite | one record | one record |
| Delete by id | yes, with a rewrite | yes, cheap | yes |
| Memory limit | whole file | whole file | the database page cache |
| Unlocks items 1, 8, 9, 11 | yes | yes | yes |
| Migration of an old store | a read of `VEI1`, then a write of `VEI2` | same | a full import |
| Risk | low | medium | high |

## 6. Why SQLite is the risk it is

Google chose SQLite. The comparison doc records the shape: the app stores
through a MediaPipe `SqliteVectorStore`, in the table `rag_vector_store`, and
the record id is `"<projectId>/<startTimeMs>-<endTimeMs>"`
(`docs/google-gallery-comparison.md` section 2.3). The reason fits Google's
case, and it does not transfer to this one.

Why Google took it:

1. SQLite is part of Android, so the app adds no library for the store.
2. The id holds the project and the time range, so a second pass over one video
   overwrites the same rows. That is an upsert with no bookkeeping.
3. A delete on the id prefix removes the records of one project in one call.
4. A filter on `parent_id` scopes a search to one project.

Why the same choice costs this project:

| Risk | Detail | Weight |
| --- | --- | --- |
| The search does not get faster | SQLite holds no vector type. The engine still scores every row. The brute force and the ANN work stay with us in every shape. | High |
| Every store call site changes | `open`, `append`, `forget`, `rank`, and the finished list move to SQL. The current tests need a rewrite, not an edit. | Medium |
| The file set widens | A database brings its own file, plus a journal or a write-ahead log. The store holds two files today (`index.vei` and `index.done`), and it would hold four or five. | Medium |
| The migration | Every existing store must import. A fault in the middle leaves two stores, and the run must know which one serves. | Medium |
| A new crate to audit | `rusqlite` reaches SQLite through C. The toolchain is already here, because llama.cpp builds with CMake and a C++ compiler. | Low |

Notice two facts. The store is two files today, so a wider file set is not a new
idea for this project. And the build step is the weakest of the five reasons,
because the C toolchain is already in the build. The sharpest reason is the
first: SQLite buys the writes, and it buys nothing for the search.

## 7. The alternatives, if the one-file rule goes

The real fork is not the file count. The fork is which layer owns the vectors.

| Alternative | What it gives | Cost |
| --- | --- | --- |
| SQLite, with a vector extension | An id, an upsert, a prefix delete, and a vector search | One crate, an extension build, a wider file set |
| A pure-Rust key-value store (`redb` or `sled`), plus an ANN crate | No C, transactions, a real index | Two crates, and our own layout for the vectors |
| A memory-mapped vector file, plus an ANN crate (`hnsw_rs` or `usearch`) | The file format stays ours, and the search scales | We build the metadata side: id, plan, finished list |
| A columnar vector format (Lance) | A format made for vectors, with an ANN path | A large dependency, and an unfamiliar file |
| DuckDB, with a vector extension | SQL, plus an index | A large dependency, and a server-grade engine |

If the count stays under about one million records, brute force holds, and the
shape is only about the writes and the metadata. The spec names that line: brute
force is fine to about one million vectors
(`docs/superpowers/specs/2026-10-10-videoembed-product-routes.md:121`). If the
count passes it, the ANN crate decides the design, and the database behind it is
the smaller choice.

### 7.1 The embedded vector databases

A vector database is more than a store. It holds an index, a metadata filter,
and a write path that knows about vectors. Four of them fit a local tool, and
three do not.

| Option | What it is | Index | Filter | Shape | Cost |
| --- | --- | --- | --- | --- | --- |
| LanceDB (the Lance format) | An embedded vector database, written in Rust | IVF-PQ, plus brute force under a threshold | A columnar dataset with filters | In-process | A large dependency, Arrow and Lance, a versioned file set |
| DuckDB with the VSS extension | An embedded SQL engine | HNSW | SQL | One file | A large C++ binary, and an extension |
| sqlite-vec | A SQLite extension | Brute force, with quantization | SQL | One file plus the extension | An extension build |
| usearch | An ANN library | HNSW, and other graphs | None | A library | We own the storage and the metadata |
| hnsw_rs | A Rust ANN index | HNSW | None | A library | The same, in Rust |
| Faiss | A C++ ANN library | Several | None | A library | A C++ build, and bindings |
| Qdrant | A vector search server, in Rust | HNSW | Yes | A server process | A process to run and to ship |

The library rows are indexes, not databases. An index answers "the nearest
vectors to this one". It holds no path, no plan, and no finished list, so the
metadata side stays ours in every library row.

For this project the shortlist is three: LanceDB, an embedded vector database
with a real index, DuckDB with the VSS extension, and a small ANN library over
our own file. LanceDB and DuckDB each bring a large dependency. That weight buys
an index, and the index only matters near the top of the range in section 3.

### 7.2 The blob column or the vec0 table

Two ways put a vector in SQLite.

The blob way: an ordinary table holds the vector in a `BLOB` column, and the
engine scores the rows in Rust. One table holds the metadata and the vector, so
one query answers a scoped search, and our own `dot` runs on unit vectors.

The `vec0` way: the sqlite-vec extension holds the vectors in a virtual table,
and the search runs inside SQLite, in C. The vector leaves the records table, so
the path and the kind come back through a join on the id.

| | The blob way | The vec0 way |
| --- | --- | --- |
| The scan runs | In Rust | In SQLite, in C |
| The metric | Our `dot`, on unit vectors | The extension distance, with cosine |
| The vector sits | In the record row | In a second table, keyed by the id |
| A scoped search | One query, then a scan of those rows | A partition key, or a filter on the join |
| The extension | None | sqlite-vec, built and loaded on each connection |
| The index | None | None. `vec0` is brute force, with optional quantization |

The last row is the one that matters here. `vec0` is still an exhaustive scan.
It moves the scan from Rust into C, and binary quantization can shrink it. It
adds no graph and no tree.

### 7.3 An ANN library beside the database

Yes, an ANN library can index the same vectors. The pattern is a database as the
source of truth, with an index beside it:

1. The database holds the rows, the ids, the plans, and the finished list.
2. The index holds the vectors alone, under the same ids.
3. A query asks the index for the nearest ids, then reads those rows from the
   database.

`hnsw_rs` or `usearch` fills that role in Rust, in the process, with no server.
The cost is one more thing to keep in step. An append must reach both, a delete
must reach both, and a fault between the two writes leaves an index that holds a
record the database dropped. A rebuild from the database repairs it.

This pairing stands apart from the `vec0` question. The database writes the
rows, and the index is a cache over them. The open choice is when the index
earns its cost: near one million records, by the estimate in section 3.

## 8. The scoped search, with and without SQL

Item 6 lets a user search one video only. The store must answer one question:
score the records of this one path.

With SQL the answer is one line:

```sql
SELECT id, vec FROM records WHERE path = ?;
```

The engine reads an index on `path`, so it touches only those rows. Our code
holds no filter logic of its own.

A key-value store holds no query language. It holds a map from a key to a
value. Our code then has two choices, and each one costs work:

1. Read every record, then skip the records whose path differs. The read
   touches the whole store, so the search slows down as the store grows.
2. Keep a second map from the path to the record keys. That map is a secondary
   index. We write it, we update it on every append, and we repair it on every
   delete.

So "there is no SQL for a scoped search" means this: the filter is code we
write, and each filter column becomes an index we maintain by hand. SQLite,
DuckDB, and LanceDB each carry that work already.

## 9. The decision and what follows

The decision on 2026-10-11: the store moves to a database. The one-file rule
goes. The shape is not A.

The engine decision on 2026-10-11: SQLite, with the sqlite-vec extension. The
blob way and the `vec0` way are not exclusive. SQLite holds the rows, the ids,
the plans, and the finished list, and one `vec0` table holds the vectors.

What follows from the two decisions:

1. SQLite owns the rows, the ids, the plans, and the finished list. One `videos`
   row holds the plan of each video and its finished mark, so `index.done` folds
   in.
2. A scoped search becomes a query, not our own filter code.
3. The vector sits in the `vec0` table, not in the record row. One copy, one id.
4. An ANN index waits. `vec0` scores every row, and the index earns its cost
   only near one million records. The index is a library beside the database
   when it comes (section 7.3), not a second engine.
5. The `fp16` plan needs a new shape. `vec0` holds a `float[768]` column, so it
   takes 4 bytes for each value, or a binary form on the quantized path. The
   extension carries no `fp16` type, so the disk saving comes from a rebuild in
   `f32` first, and from quantization later.

## 10. Open questions

1. Does window mode land in the same change, or after the format is done? Two
   steps are safer, because the format change alone is testable on its own.
2. Does the store need a source fingerprint (size and mtime)? Today a changed
   file with the same path stays marked as finished, so the run skips it.
3. Does the import of an old `VEI1` store run once at open, or as its own
   command?
4. Does the store keep the name `store/index.vei`, or take a new one?

## 11. What stays the same

- The label rule, `normalize_label`, and the slash form of a path.
- The kind list, and `is_video`.
- The temp-file-then-rename write, so a crash cannot leave a half store.
- The one vector size for the whole file.
- The store path, `store/index.vei`.
