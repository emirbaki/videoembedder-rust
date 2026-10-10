# The SQLite store - design

Date: 2026-10-11
Status: approved in chat. The parts are ready to plan.

Sources:

- `docs/store-format-change.md` holds the analysis and the decisions.
- `docs/deferred-work.md` holds the items that wait.
- `docs/google-gallery-comparison.md` section 2.3 holds Google's store.

## 1. The goal

The store moves from one flat file, `store/index.vei`, to SQLite with the
sqlite-vec extension. The change unlocks four deferred items: the window
record, the per-video plan, the upsert and delete by id, and the sliders.

The change does not touch the rank, the prefixes, or the callers.

## 2. The decisions

The decisions on 2026-10-11:

1. The engine is SQLite, with the sqlite-vec extension.
2. One `vec0` table holds the vector. The store keeps one copy of it.
3. The record id takes Google's shape, `"<path>/<startMs>-<endMs>"`.
4. One `videos` row holds the plan of each video and its finished mark.
5. Window mode lands after this change, not in it.
6. An old `VEI1` store imports at open.
7. The store file is `store/index.db`.

## 3. The schema

Three tables in one file.

```sql
CREATE TABLE videos (
  path         TEXT PRIMARY KEY,
  plan_window  REAL NOT NULL,
  plan_stride  REAL NOT NULL,
  plan_frames  INTEGER NOT NULL,
  plan_audio   INTEGER NOT NULL,
  source_size  INTEGER,
  source_mtime INTEGER,
  done         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE records (
  id    TEXT PRIMARY KEY,
  path  TEXT NOT NULL,
  kind  INTEGER NOT NULL,
  start REAL NOT NULL,
  end   REAL NOT NULL
);

CREATE INDEX records_path ON records (path);

CREATE VIRTUAL TABLE vec_records USING vec0(
  id        TEXT PRIMARY KEY,
  path      TEXT PARTITION KEY,
  embedding float[768]
);
```

The `videos` table folds in the old `index.done` file. A row for each video
holds the plan and the finished mark.

The `records` table holds no vector. The `vec_records` table holds the vector
and nothing else. The id joins the two.

The `path` column of `vec_records` is a partition key. A scoped search names
one path, and `vec0` scans that partition alone.

Two details wait for the build. Section 11 lists them.

## 4. The write path

One write is one transaction:

1. Begin.
2. Delete the `records` rows of the video, by the id prefix.
3. Delete the `vec_records` rows of the video, by the id prefix.
4. Insert the new `records` rows and their `vec_records` rows.
5. Insert or update the `videos` row, with the plan and the fingerprint.
6. Commit.

The prefix is `"<path>/"`. The time part follows. So a delete of one video
touches one video, and never a second one whose path starts the same way.

A crash before the commit leaves the store as it was. A second pass over one
video writes the same ids, so the pass replaces the old rows.

## 5. The read paths

**The global search.** The rank scores every row, because it needs the median
of the store. The `vec0` table gives the distance for each row, and the
`records` table gives the path, the kind, and the range:

```sql
SELECT vec_records.distance, records.id, records.path,
       records.kind, records.start, records.end
FROM vec_records
JOIN records ON records.id = vec_records.id
WHERE vec_records.embedding MATCH ?;
```

The rank keeps its shape: the median baseline, the floor of 0.4, the group by
video, and the cap of 50. The vector never leaves the database.

**The scoped search.** The query adds `AND vec_records.path = ?`. The
partition key keeps the scan inside one video.

## 6. The import of an old store

The open call reads `index.vei` when the old file is here and `index.db` is
not. The steps:

1. Make the schema.
2. Read the `VEI1` file.
3. Write every record in one transaction. The id comes from the label, the
   start, and the end.
4. Give each video row a default plan and a `done` mark of 1.
5. Keep `index.vei` in place, so a bad import loses nothing.

The `VEI1` reader stays for one release. A later change removes it.

## 7. The errors

1. A missing database is an empty store, as a missing file is today.
2. A locked database waits on a busy timeout.
3. An extension that fails to load stops the open, with a clear message.
   Nothing works without it.
4. The import is one transaction. A fault in the middle leaves the old file
   and no database.

## 8. The tests

The current store tests carry over to the new backend. Five tests are new:

1. An id round trip, from a write to a read.
2. A second pass over one video leaves one set of rows, not two.
3. A delete of one video removes its rows and its vectors.
4. A scoped search returns records of one video only.
5. The import of a small `VEI1` store lands the same records.

Two videos share a path prefix in test 3, so the prefix delete proves its
boundary.

## 9. What does not change

- The rank rules: the median baseline, the floor, the group by video, and the
  cap of 50.
- The label rule, `normalize_label`, and the slash form of a path.
- The kind list, and `is_video`.
- The store path, `store/index.vei`. The open call reads it.
- The callers in `main.rs` and `web.rs`.

## 10. Out of scope

- Window mode. It lands after this change.
- The sliders and the presets.
- An ANN index. A library joins later, near one million records.
- A check of the fingerprint, so a changed file with the same path reindexes.
  The column is here. The check comes later.

## 11. Open items for the build

1. The `vec0` primary key type. A text key needs a recent sqlite-vec. If the
   build rejects it, the key becomes an integer, and the `records` table
   carries the text id.
2. The `vec0` partition key syntax. If the build rejects it, the scoped search
   filters the join instead.
3. The build of the bundled SQLite and the extension on Windows, with the
   current toolchain.
