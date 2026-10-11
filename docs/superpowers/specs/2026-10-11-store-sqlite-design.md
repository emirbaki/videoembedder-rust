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
3. The record id takes Google's shape, `"path/startMs-endMs"`, with the two
   times in whole milliseconds.
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
  rid   INTEGER PRIMARY KEY,
  id    TEXT NOT NULL UNIQUE,
  path  TEXT NOT NULL,
  kind  INTEGER NOT NULL,
  start REAL NOT NULL,
  end   REAL NOT NULL
);

CREATE INDEX records_path ON records (path);

CREATE VIRTUAL TABLE vec_records USING vec0(
  rid       INTEGER PRIMARY KEY,
  path      TEXT PARTITION KEY,
  embedding float[768] distance_metric=cosine
);
```

The `videos` table folds in the old `index.done` file. A row for each video
holds the plan and the finished mark.

The `records` table holds no vector. The `vec_records` table holds the vector
and nothing else. The `rid` joins the two.

The `vec0` key is an integer. The extension documents `integer primary key`
only, so the text id stays in the `records` table. The key rule does not
change: the id holds the path, the start, and the end.

The `path` column of `vec_records` is a partition key. A scoped search names
one path, and `vec0` scans that partition alone.

The column takes the cosine metric. The rank uses the dot product today, and
cosine distance gives the same order for a unit vector.

The `path` column serves the delete. A delete names a path and touches that
path alone. Section 4 holds the rule.

## 4. The write path

One write is one transaction:

1. Begin.
2. Delete the rows of the video from both tables, by the `path` column.
3. Insert the new `records` rows and their `vec_records` rows.
4. Insert or update the `videos` row, with the plan and the fingerprint.
5. Commit.

The delete names the path: `where path = ?`. The column is exact, so a delete
of one video touches one video, and never a second one whose path starts the
same way.

The write names each path of the batch. A video path that arrives again
replaces its own rows. A note with the same text replaces itself.

A crash before the commit leaves the store as it was. A second pass over one
video writes the same ids, so the pass replaces the old rows.

## 5. The read paths

**The global search.** The rank scores every row, because it needs the median
of the store. The `vec0` table gives the distance for each row. The score is
one minus the distance.

```sql
SELECT rid, distance
FROM vec_records
WHERE embedding MATCH vec_f32(?) AND k = ?;
```

The rank keeps its shape: the median baseline, the floor of 0.4, the group by
video, and the cap of 50. The vector never leaves the database.

The in-memory record list holds the metadata alone. The list gives the
position of a `rid`, and the rank gives the distance. The two sides meet in
one map.

**The scoped search.** The query adds `AND path = ?`. The partition key keeps
the scan inside one video.

**The measure command.** The command reads the vectors of the kept records on
request. `Store::vectors(&[i64])` returns them in the order of the ids.

## 6. The import of an old store

The open call reads `index.vei` when the old file is here and the database
holds no record. The steps:

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

The current store tests carry over to the new backend. Eight tests are new:

1. An id round trip, from a write to a read.
2. A second pass over one video leaves one set of rows, not two.
3. A delete of one video removes its rows and its vectors.
4. A delete of one path leaves a second path that starts the same way.
5. A fault in the middle of a write leaves no row and no vector.
6. The rank of an empty store is empty.
7. A path with a space and a non-ASCII letter, and a free note, round trip.
8. The import of a small `VEI1` store lands the same records. An import of a
   store with another vector size fails, and the message names the file.

Two videos share a path prefix in test 4, so the delete proves its boundary.

## 9. What does not change

- The rank rules: the median baseline, the floor, the group by video, and the
  cap of 50.
- The label rule, `normalize_label`, and the slash form of a path.
- The kind list, and `is_video`.
- The commands, and the console text. The store path changes, and the three
  callers change with it.

The three callers change in one way. The `Record` type holds no vector, so
`main.rs` and `web.rs` build a `NewRecord` on the write path. The measure
command reads the vectors it needs from the store.

## 10. Out of scope

- Window mode. It lands after this change.
- The sliders and the presets.
- An ANN index. A library joins later, near one million records.
- A check of the fingerprint, so a changed file with the same path reindexes.
  The column is here. The check comes later.

## 11. Open items for the build

1. The `vec0` partition key syntax. If the build rejects it, the scoped search
   filters in Rust instead.
2. The vector form of a KNN query. The extension takes a blob or a JSON text.
   The first database test settles which one this build wants.
3. The build of the bundled SQLite and the extension on Windows, with the
   current toolchain.
