# The SQLite Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the store from the one flat file `store/index.vei` to SQLite with the sqlite-vec extension, so the store gains ids, an upsert, a delete by id, and a per-video plan.

**Architecture:** One SQLite file, `store/index.db`, holds three tables. The `videos` table holds one row for each video: the plan and the finished mark. The `records` table holds one row for each record: the id, the path, the kind, and the range. The `vec_records` table is a `vec0` virtual table that holds the vector alone, keyed by the row id of the record. The in-memory record list holds metadata only, so the vectors stay on disk.

**Tech Stack:** Rust 2024, `rusqlite` 0.40 with the `bundled` feature, `sqlite-vec` 0.1.9, `anyhow`.

**Spec:** `docs/superpowers/specs/2026-10-11-store-sqlite-design.md`

## Global Constraints

- The store file is `store/index.db`.
- The record id takes Google's shape: `"path/startMs-endMs"`, with the two times in whole milliseconds.
- The vector size is 768.
- One `vec0` table holds the vector. The store holds one copy.
- The rank rules do not change: the median baseline, the floor of 0.4, the group by video, and the cap of 50.
- The label rule does not change: `normalize_label` turns a backslash into a slash.
- The kind list does not change: window 0, note 1, frame 2, audio 3.
- Window mode does not land in this change.
- An old `VEI1` store imports at open, one time.
- The in-memory `Store::records` list holds no vector.
- The test command is `cargo test`. Use `--lib` or `--bin` to run one group.
- Every commit message ends with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

The five input classes that the tests must pin, most likely first. The test that pins each one follows the line.

1. Two paths where one holds the start of the other, such as `a/b.mp4` and `a/b.mp4x`. A delete and a scoped search must touch one video only. Test: `a_path_that_starts_like_another_keeps_the_delete_to_one_video`, Task 3.
2. A fault in the middle of a write. The store must keep the rows it had, and it must hold no half record. Test: `a_fault_in_the_middle_of_a_write_leaves_no_rows`, Task 3.
3. An old `VEI1` store whose vector size is not 768. The import must refuse it with a message that names the file. Test: `an_old_store_of_another_size_is_refused`, Task 5.
4. An empty store. A search must say the store is empty, and the rank must give an empty list. Test: `the_rank_of_an_empty_store_is_empty`, Task 4.
5. A path with a space and a non-ASCII letter, and a free note. The label and the id must round trip. Test: `a_unicode_path_and_a_note_round_trip`, Task 3.

## File Structure

| File | Change | What it holds |
| --- | --- | --- |
| `videoembed/Cargo.toml` | Modify | The two new dependencies. |
| `videoembed/src/lib.rs` | Modify | The new `db` module. |
| `videoembed/src/db.rs` | Create | The connection, the extension, the schema, and the vector blobs. |
| `videoembed/src/store.rs` | Modify | The store API over the database, and the old-file reader. |
| `videoembed/src/index.rs` | Modify | The run gives `NewRecord`. The thinning gives indexes. |
| `videoembed/src/main.rs` | Modify | The store path, the note, and the measure command. |
| `videoembed/src/bin/web.rs` | Modify | The store path and the job records. |
| `docs/deferred-work.md` | Modify | Item 9 comes off the list. |
| `docs/store-format-change.md` | Modify | The status line. |
| `docs/google-gallery-comparison.md` | Modify | The store row, and the "no database" gain. |

---

### Task 1: The database module and the schema

**Files:**
- Create: `videoembed/src/db.rs`
- Modify: `videoembed/Cargo.toml`
- Modify: `videoembed/src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `db::DIM` (768), `db::open(&Path) -> Result<Connection>`, `db::vec_to_blob(&[f32]) -> Vec<u8>`, `db::blob_to_vec(&[u8]) -> Vec<f32>`, and the three tables `videos`, `records`, `vec_records`.

- [ ] **Step 1: Add the two dependencies**

Run:

```bash
cd videoembed && cargo add rusqlite --features bundled && cargo add sqlite-vec@0.1.9
```

Expected: `Cargo.toml` holds `rusqlite = { version = "0.40.2", features = ["bundled"] }` and `sqlite-vec = "0.1.9"`.

- [ ] **Step 2: Write the failing tests**

Put this module at the end of a new file `videoembed/src/db.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_db_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A unit vector with one 1.0 at `at`. The size is the model size.
    fn vec_at(at: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; DIM];
        out[at] = 1.0;
        out
    }

    #[test]
    fn the_extension_answers_its_version() {
        let dir = scratch("version");
        let conn = open(&dir.join("index.db")).unwrap();
        let version: String = conn
            .query_row("select vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(version.starts_with('v'), "got {version}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_vector_round_trips_through_the_vec_table() {
        let dir = scratch("round");
        let conn = open(&dir.join("index.db")).unwrap();
        conn.execute(
            "insert into vec_records(rid, path, embedding) values (1, 'a.mp4', vec_f32(?1))",
            [vec_to_blob(&vec_at(0))],
        )
        .unwrap();

        let blob: Vec<u8> = conn
            .query_row("select embedding from vec_records where rid = 1", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert_eq!(blob_to_vec(&blob), vec_at(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_knn_query_returns_the_nearest_row() {
        let dir = scratch("knn");
        let conn = open(&dir.join("index.db")).unwrap();
        for at in 0..3 {
            conn.execute(
                "insert into vec_records(rid, path, embedding) values (?1, 'a.mp4', vec_f32(?2))",
                rusqlite::params![at + 1, vec_to_blob(&vec_at(at))],
            )
            .unwrap();
        }

        let best: i64 = conn
            .query_row(
                "select rid from vec_records where embedding match vec_f32(?1) and k = 1",
                [vec_to_blob(&vec_at(2))],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(best, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_partition_key_narrows_a_query() {
        let dir = scratch("part");
        let conn = open(&dir.join("index.db")).unwrap();
        conn.execute(
            "insert into vec_records(rid, path, embedding) values (1, 'a.mp4', vec_f32(?1))",
            [vec_to_blob(&vec_at(0))],
        )
        .unwrap();
        conn.execute(
            "insert into vec_records(rid, path, embedding) values (2, 'b.mp4', vec_f32(?1))",
            [vec_to_blob(&vec_at(1))],
        )
        .unwrap();

        let best: i64 = conn
            .query_row(
                "select rid from vec_records \
                 where embedding match vec_f32(?1) and k = 5 and path = 'b.mp4'",
                [vec_to_blob(&vec_at(1))],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(best, 2, "the query must stay inside one video");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `cd videoembed && cargo test --lib db::`

Expected: the build fails with `unresolved import` or `cannot find function open`. The module does not exist yet.

- [ ] **Step 4: Write the module**

Put this at the top of `videoembed/src/db.rs`, above the test module:

```rust
//! The database. One connection, the sqlite-vec extension, and the schema.
//!
//! The vectors live in the `vec_records` table. The metadata lives in the
//! `records` and `videos` tables. One `rid` joins a record to its vector.

use std::path::Path;
use std::sync::Once;
use std::time::Duration;

use anyhow::{Context, Result};
use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;

/// The vector size of the model. The schema bakes this value in.
pub const DIM: usize = 768;

/// The metadata tables. A missing database makes them.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS videos (
  path         TEXT PRIMARY KEY,
  plan_window  REAL NOT NULL DEFAULT 0.25,
  plan_stride  REAL NOT NULL DEFAULT 0.25,
  plan_frames  INTEGER NOT NULL DEFAULT 1,
  plan_audio   INTEGER NOT NULL DEFAULT 1,
  source_size  INTEGER,
  source_mtime INTEGER,
  done         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS records (
  rid   INTEGER PRIMARY KEY,
  id    TEXT NOT NULL UNIQUE,
  path  TEXT NOT NULL,
  kind  INTEGER NOT NULL,
  start REAL NOT NULL,
  end   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS records_path ON records (path);
";

/// Register the extension one time in this process.
///
/// The call is process wide, so every later connection carries the extension.
fn register() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    });
}

/// The vector table. The size comes from the model.
fn vec_schema() -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_records USING vec0(
           rid INTEGER PRIMARY KEY,
           path TEXT PARTITION KEY,
           embedding float[{DIM}] distance_metric=cosine
         );"
    )
}

/// Open the database at one path and make the schema.
///
/// A missing file makes an empty store.
pub fn open(path: &Path) -> Result<Connection> {
    register();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("make the store folder")?;
    }
    let conn = Connection::open(path).context("open the store")?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.execute_batch(SCHEMA).context("make the tables")?;
    conn.execute_batch(&vec_schema()).context("make the vector table")?;
    Ok(conn)
}

/// Write one vector as a blob. One value takes four bytes.
pub fn vec_to_blob(vec: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for x in vec {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Read a stored vector blob. One value takes four bytes.
pub fn blob_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes([part[0], part[1], part[2], part[3]]))
        .collect()
}
```

Then add the module to `videoembed/src/lib.rs`:

```rust
pub mod db;
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cd videoembed && cargo test --lib db::`

Expected: 4 tests pass. If `vec_version` or the `vec0` build fails, read the error and stop: the extension did not build, and no later task can work.

If the KNN test fails on the vector form alone, try the two next forms before you change anything else: `match ?1` without `vec_f32`, then `vec_f32(?1)` on a JSON string. The extension takes a blob or a JSON text for a vector, and the test settles which one this build wants. Record the form that works in a comment.

- [ ] **Step 6: Run the whole suite**

Run: `cd videoembed && cargo test`

Expected: the old counts hold: 60 lib tests, 2 console tests, 18 web tests, plus the 4 new ones.

- [ ] **Step 7: Commit**

```bash
git add videoembed/Cargo.toml videoembed/Cargo.lock videoembed/src/db.rs videoembed/src/lib.rs
git commit -m "feat: add the SQLite store module with sqlite-vec

The module opens the database, makes the three tables, and converts a
vector to a blob and back. The vec0 table takes cosine distance and a
text partition key on the path.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: The record id and the new record type

**Files:**
- Modify: `videoembed/src/store.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `store::NewRecord { kind: u8, label: String, start: f32, end: f32, vec: Vec<f32> }` and `store::record_id(label: &str, start: f32, end: f32) -> String`.

- [ ] **Step 1: Write the failing test**

Add these tests to the `tests` module of `videoembed/src/store.rs`:

```rust
    #[test]
    fn a_record_id_holds_the_path_and_the_range() {
        assert_eq!(record_id("D:/video/a.mp4", 0.25, 0.5), "D:/video/a.mp4/250-500");
        assert_eq!(record_id("a note", 0.0, 0.0), "a note/0-0");
    }

    #[test]
    fn the_same_record_gives_the_same_id_twice() {
        let first = record_id("a.mp4", 1.0, 1.25);
        let second = record_id("a.mp4", 1.0, 1.25);
        assert_eq!(first, second, "a second pass must write the same id");
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cd videoembed && cargo test --lib store::tests::a_record_id`

Expected: FAIL with `cannot find function record_id`.

- [ ] **Step 3: Write the type and the id**

Add this code to `videoembed/src/store.rs`, above `pub struct Record`:

```rust
/// One record before it enters the store. The vector travels with it.
pub struct NewRecord {
    pub kind: u8,
    pub label: String,
    pub start: f32,
    pub end: f32,
    pub vec: Vec<f32>,
}

/// The id of one record, in Google's shape.
///
/// The label holds the source, and the two times hold the place in it. A
/// second pass over one video writes the same ids, so the write replaces the
/// rows of the first pass.
pub fn record_id(label: &str, start: f32, end: f32) -> String {
    format!("{}/{}-{}", label, millis(start), millis(end))
}

/// One time in milliseconds, rounded to the nearest one.
fn millis(seconds: f32) -> i64 {
    (seconds as f64 * 1000.0).round() as i64
}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cd videoembed && cargo test --lib store::tests`

Expected: PASS. The two new tests pass, and every old test still passes.

- [ ] **Step 5: Commit**

```bash
git add videoembed/src/store.rs
git commit -m "feat: add the record id and the new record type

A record id holds the path and the range in milliseconds. A second pass
over one video writes the same ids.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: The store reads and writes the database

This is the largest task. It moves every store call to SQL, drops the vector from the in-memory record, and updates the three callers. The build is green at the end.

**Files:**
- Modify: `videoembed/src/store.rs`
- Modify: `videoembed/src/index.rs`
- Modify: `videoembed/src/main.rs`
- Modify: `videoembed/src/bin/web.rs`

**Interfaces:**
- Consumes: `db::DIM`, `db::open`, `db::vec_to_blob`, `record_id`, `NewRecord`.
- Produces: `Record { rid: i64, id: String, kind: u8, label: String, start: f32, end: f32 }` and `Store { pub dim: usize, pub records: Vec<Record>, .. }`, with `Store::open`, `Store::append(Vec<NewRecord>)`, `Store::forget`, `Store::is_done`, `Store::mark_done`, `Store::holds_video`.

- [ ] **Step 1: Write the failing tests**

Replace the old file-format tests in `videoembed/src/store.rs` (the tests `the_store_round_trips_through_the_file_format` and `a_damaged_file_is_rejected_with_a_message`) with these tests, and update the two helpers `record` and `store_of`:

```rust
    /// A frame record for one video. The vector size is the model size.
    fn new_record(label: &str, start: f32, end: f32) -> NewRecord {
        let mut vec = vec![0.0f32; db::DIM];
        vec[0] = 1.0;
        NewRecord {
            kind: KIND_FRAME,
            label: label.to_string(),
            start,
            end,
            vec,
        }
    }

    /// A store on a fresh database in the scratch folder.
    fn opened(name: &str) -> (PathBuf, Store) {
        let dir = scratch(name);
        let path = dir.join("index.db");
        let store = Store::open(&path).unwrap();
        (dir, store)
    }

    #[test]
    fn a_record_reaches_the_database_and_comes_back() {
        let (dir, mut store) = opened("round");
        store
            .append(vec![new_record("D:/video/a.mp4", 0.25, 0.5)])
            .unwrap();

        let again = Store::open(&dir.join("index.db")).unwrap();

        assert_eq!(again.records.len(), 1);
        assert_eq!(again.records[0].label, "D:/video/a.mp4");
        assert_eq!(again.records[0].id, "D:/video/a.mp4/250-500");
        assert_eq!(again.records[0].kind, KIND_FRAME);
        assert_eq!(again.records[0].start, 0.25);
        assert_eq!(again.records[0].end, 0.5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A second pass over one video replaces its rows. It does not double them.
    #[test]
    fn a_second_pass_leaves_one_set_of_rows() {
        let (dir, mut store) = opened("twice");
        store
            .append(vec![
                new_record("a.mp4", 0.0, 0.25),
                new_record("a.mp4", 0.25, 0.5),
            ])
            .unwrap();
        store
            .append(vec![
                new_record("a.mp4", 0.0, 0.25),
                new_record("a.mp4", 0.25, 0.5),
            ])
            .unwrap();

        assert_eq!(store.records.len(), 2);
        let again = Store::open(&dir.join("index.db")).unwrap();
        assert_eq!(again.records.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The delete of one video touches the records and the vectors.
    #[test]
    fn a_delete_removes_the_rows_and_the_vectors() {
        let (dir, mut store) = opened("delete");
        store
            .append(vec![new_record("a.mp4", 0.0, 0.25), new_record("b.mp4", 0.0, 0.25)])
            .unwrap();

        let removed = store.forget("a.mp4").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(store.records.len(), 1);
        let left: i64 = store
            .conn
            .query_row("select count(*) from vec_records", [], |row| row.get(0))
            .unwrap();
        assert_eq!(left, 1, "the vector of the removed record must go");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two paths where one starts like the other stay apart.
    #[test]
    fn a_path_that_starts_like_another_keeps_the_delete_to_one_video() {
        let (dir, mut store) = opened("prefix");
        store
            .append(vec![
                new_record("a/b.mp4", 0.0, 0.25),
                new_record("a/b.mp4x", 0.0, 0.25),
            ])
            .unwrap();

        let removed = store.forget("a/b.mp4").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(store.records.len(), 1);
        assert_eq!(store.records[0].label, "a/b.mp4x");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fault in the middle of a write leaves no half record.
    ///
    /// The two records hold the same path, the same range, and so the same
    /// id. The second insert breaks the unique rule. One transaction holds
    /// the two, so the first insert goes back with the second.
    #[test]
    fn a_fault_in_the_middle_of_a_write_leaves_no_rows() {
        let (dir, mut store) = opened("fault");
        let outcome = store.append(vec![
            new_record("a.mp4", 0.0, 0.25),
            new_record("a.mp4", 0.0, 0.25),
        ]);

        assert!(outcome.is_err(), "the same id must break the write");
        assert!(store.records.is_empty(), "the store keeps no half record");
        let rows: i64 = store
            .conn
            .query_row("select count(*) from vec_records", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 0, "no vector may stay after the fault");
        let again = Store::open(&dir.join("index.db")).unwrap();
        assert!(again.records.is_empty(), "the file keeps no half record");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A space and a non-ASCII letter survive the database.
    #[test]
    fn a_unicode_path_and_a_note_round_trip() {
        let (dir, mut store) = opened("unicode");
        let mut note = new_record("doğum günü", 0.0, 0.0);
        note.kind = KIND_NOTE;
        store
            .append(vec![new_record("D:/video/doğum günü.mp4", 0.0, 0.25), note])
            .unwrap();

        let again = Store::open(&dir.join("index.db")).unwrap();

        assert_eq!(again.records.len(), 2);
        assert!(again.holds_video("D:/video/doğum günü.mp4"));
        assert_eq!(again.records[1].kind, KIND_NOTE);
        assert_eq!(again.records[1].id, "doğum günü/0-0");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

Keep the other tests, and change each `Record { .. vec: vec![1.0, 0.0] }` construction to a `Record` with `rid`, `id`, and no `vec`. Use this helper:

```rust
    /// A stored record, for the tests that read one.
    fn record(label: &str) -> Record {
        Record {
            rid: 0,
            id: record_id(label, 0.0, 4.0),
            kind: KIND_WINDOW,
            label: label.to_string(),
            start: 0.0,
            end: 4.0,
        }
    }
```

Change `store_of` to build `Store::open` on a scratch database and append the records:

```rust
    /// A store that holds these records, on a fresh database.
    fn store_of(records: Vec<Record>) -> Store {
        let dir = std::env::temp_dir().join(format!(
            "vei_store_of_{}_{}",
            std::process::id(),
            records.len()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        let new: Vec<NewRecord> = records
            .into_iter()
            .map(|r| NewRecord {
                kind: r.kind,
                label: r.label,
                start: r.start,
                end: r.end,
                vec: {
                    let mut vec = vec![0.0f32; db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            })
            .collect();
        store.append(new).unwrap();
        store
    }
```

The test `forget_removes_frame_and_audio_records_of_one_video` and `forget_removes_a_label_across_path_separators` build a `Store` by hand. Change them to use `store_of` and `new_record`:

```rust
    #[test]
    fn forget_removes_frame_and_audio_records_of_one_video() {
        let (dir, mut store) = opened("forget_kind");
        let mut audio = new_record("D:/video/a.mp4", 0.0, 1.0);
        audio.kind = KIND_AUDIO;
        store
            .append(vec![new_record("D:/video/a.mp4", 0.0, 0.25), audio, new_record("D:/video/b.mp4", 0.0, 0.25)])
            .unwrap();

        let removed = store.forget("D:/video/a.mp4").unwrap();

        assert_eq!(removed, 2);
        assert_eq!(store.records.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forget_removes_a_label_across_path_separators() {
        let (dir, mut store) = opened("forget_slash");
        store
            .append(vec![
                new_record("D:\\video\\a.mp4", 0.0, 0.25),
                new_record("D:\\video\\b.mp4", 0.0, 0.25),
            ])
            .unwrap();

        let removed = store.forget("D:/video/a.mp4").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(store.records[0].label, "D:\\video\\b.mp4");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

Add `use crate::db;` to the top of the `tests` module, and change the store test paths from `index.vei` to `index.db` in `a_finished_video_survives_a_reopen`, `forget_clears_the_finished_mark`, and `a_windows_path_form_still_matches_the_mark`.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cd videoembed && cargo test --lib store::`

Expected: the build fails. `Record` holds no `vec` field yet, and `Store::append` takes `Vec<Record>`.

- [ ] **Step 3: Write the store**

Replace the top of `videoembed/src/store.rs` (the module doc, the imports, `MAGIC`, `done_path`, `read_done`, `Record`, `Store`, and every method that writes or reads the flat file) with this code:

```rust
//! The vector store. One SQLite database holds every record.
//!
//! The vectors live in the `vec_records` table of `db`. The metadata lives in
//! the `records` and `videos` tables. The in-memory list holds the metadata
//! alone, so a large store costs memory for the small part.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, Transaction, params};

use crate::db;

const MAGIC: &[u8; 4] = b"VEI1";

pub const KIND_WINDOW: u8 = 0;
pub const KIND_NOTE: u8 = 1;
pub const KIND_FRAME: u8 = 2;
pub const KIND_AUDIO: u8 = 3;

/// The plan of the run that fills the store. Window mode changes these.
const PLAN_WINDOW: f64 = 0.25;
const PLAN_FRAMES: i64 = 1;
const PLAN_AUDIO: i64 = 1;
```

Keep `is_video` and `normalize_label` as they are.

Then the record and the store:

```rust
/// One record in the store. The vector stays on disk.
#[derive(Clone)]
pub struct Record {
    pub rid: i64,
    pub id: String,
    pub kind: u8,
    pub label: String,
    pub start: f32,
    pub end: f32,
}

pub struct Store {
    path: PathBuf,
    conn: Connection,
    pub dim: usize,
    pub records: Vec<Record>,
    /// The video paths that finished an index run.
    done: HashSet<String>,
}
```

Then the methods:

```rust
impl Store {
    /// Open the store. A missing file gives an empty store.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = db::open(path)?;
        let mut store = Self {
            path: path.to_path_buf(),
            conn,
            dim: db::DIM,
            records: Vec::new(),
            done: HashSet::new(),
        };
        store.load_records()?;
        store.done = store.load_done()?;
        Ok(store)
    }

    /// Read the metadata of every record, in row order.
    fn load_records(&mut self) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("select rid, id, kind, label, start, end from records order by rid")?;
        let rows = stmt.query_map([], |row| {
            Ok(Record {
                rid: row.get(0)?,
                id: row.get(1)?,
                kind: row.get::<_, i64>(2)? as u8,
                label: row.get(3)?,
                start: row.get::<_, f64>(4)? as f32,
                end: row.get::<_, f64>(5)? as f32,
            })
        })?;
        self.records.clear();
        for row in rows {
            self.records.push(row?);
        }
        Ok(())
    }

    /// Read the finished marks.
    fn load_done(&self) -> Result<HashSet<String>> {
        let mut stmt = self.conn.prepare("select path from videos where done = 1")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
    }

    /// Append records and write the store.
    ///
    /// A path that arrives again replaces its own rows. One transaction holds
    /// the records, their vectors, and the video rows.
    pub fn append(&mut self, new_records: Vec<NewRecord>) -> Result<()> {
        if new_records.is_empty() {
            return Ok(());
        }
        for record in &new_records {
            if record.vec.len() != self.dim {
                bail!(
                    "record vector size {} does not match the store size {}",
                    record.vec.len(),
                    self.dim
                );
            }
        }

        let tx = self.conn.transaction().context("begin the write")?;
        let mut paths: Vec<&str> = new_records.iter().map(|r| r.label.as_str()).collect();
        paths.sort_unstable();
        paths.dedup();
        for path in paths {
            delete_rows(&tx, path)?;
        }
        for record in &new_records {
            tx.execute(
                "insert into records(id, path, kind, start, end) values (?1, ?2, ?3, ?4, ?5)",
                params![
                    record_id(&record.label, record.start, record.end),
                    record.label,
                    record.kind as i64,
                    record.start as f64,
                    record.end as f64
                ],
            )?;
            let rid = tx.last_insert_rowid();
            tx.execute(
                "insert into vec_records(rid, path, embedding) values (?1, ?2, vec_f32(?3))",
                params![rid, record.label, db::vec_to_blob(&record.vec)],
            )?;
        }
        write_videos(&tx, &new_records)?;
        tx.commit().context("commit the write")?;

        self.load_records()
    }

    /// Remove every video record of one video path.
    ///
    /// The path separators do not matter, like `holds_video`. A note keeps its
    /// place, because a note is not a video record.
    pub fn forget(&mut self, path: &str) -> Result<usize> {
        let want = normalize_label(path);
        let tx = self.conn.transaction().context("begin the delete")?;
        let removed: i64 = tx.query_row(
            "select count(*) from records where path = ?1 and kind <> 1",
            [&want],
            |row| row.get(0),
        )?;
        if removed > 0 {
            tx.execute("delete from vec_records where path = ?1", [&want])?;
            tx.execute("delete from records where path = ?1 and kind <> 1", [&want])?;
        }
        tx.execute("delete from videos where path = ?1", [&want])?;
        tx.commit().context("commit the delete")?;

        self.done.remove(&want);
        self.load_records()?;
        Ok(removed as usize)
    }

    /// True when this video finished an index run.
    pub fn is_done(&self, label: &str) -> bool {
        self.done.contains(&normalize_label(label))
    }

    /// Record that this video finished an index run.
    pub fn mark_done(&mut self, label: &str) -> Result<()> {
        let want = normalize_label(label);
        self.conn.execute(
            "insert into videos(path, plan_window, plan_stride, plan_frames, plan_audio, done)
             values (?1, ?2, ?2, ?3, ?4, 1)
             on conflict(path) do update set done = 1",
            params![want, PLAN_WINDOW, PLAN_FRAMES, PLAN_AUDIO],
        )?;
        self.done.insert(want);
        Ok(())
    }

    /// True when the store holds a video record with this exact path.
    pub fn holds_video(&self, path: &str) -> bool {
        let want = normalize_label(path);
        self.records
            .iter()
            .any(|record| is_video(record.kind) && normalize_label(&record.label) == want)
    }

    /// The path of the store file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Delete the rows of one path from both tables.
fn delete_rows(tx: &Transaction, path: &str) -> Result<()> {
    tx.execute("delete from vec_records where path = ?1", [path])?;
    tx.execute("delete from records where path = ?1", [path])?;
    Ok(())
}

/// Write one `videos` row for each video path in the batch.
fn write_videos(tx: &Transaction, records: &[NewRecord]) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    for record in records {
        if !is_video(record.kind) || !seen.insert(record.label.as_str()) {
            continue;
        }
        let (size, mtime) = fingerprint(&record.label);
        tx.execute(
            "insert into videos(path, plan_window, plan_stride, plan_frames, plan_audio,
                                source_size, source_mtime, done)
             values (?1, ?2, ?2, ?3, ?4, ?5, ?6, 0)
             on conflict(path) do update set
               plan_window = excluded.plan_window,
               plan_stride = excluded.plan_stride,
               plan_frames = excluded.plan_frames,
               plan_audio = excluded.plan_audio,
               source_size = excluded.source_size,
               source_mtime = excluded.source_mtime",
            params![record.label, PLAN_WINDOW, PLAN_FRAMES, PLAN_AUDIO, size, mtime],
        )?;
    }
    Ok(())
}

/// The size and the time of the source file. A note gives two `None` values.
fn fingerprint(label: &str) -> (Option<i64>, Option<i64>) {
    let Ok(meta) = std::fs::metadata(label) else {
        return (None, None);
    };
    let size = meta.len() as i64;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);
    (Some(size), mtime)
}
```

Keep `rank` as it is for now. It reads `r.vec`, which no longer exists. Change its body to this temporary form, and Task 4 replaces it:

```rust
    /// Rank every record against one query vector. Best first.
    ///
    /// The vectors sit in the database, so the search moves to a query in
    /// Task 4. This body keeps the callers working until then.
    pub fn rank(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let _ = query;
        self.records
            .iter()
            .enumerate()
            .map(|(at, _)| (at, 0.0))
            .collect()
    }
```

Change `decode` and `Cursor` into the old-file reader that Task 5 uses:

```rust
/// Read the records of an old `VEI1` file. The import uses this.
#[allow(dead_code)]
fn vei_records(bytes: &[u8]) -> Result<(usize, Vec<NewRecord>)> {
    let mut cur = Cursor { bytes, at: 0 };
    if cur.take(4)? != MAGIC {
        bail!("the old store file is not a VEI1 file");
    }
    let dim = cur.u32()? as usize;
    let count = cur.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = cur.u8()?;
        let label_len = cur.u32()? as usize;
        let label = String::from_utf8(cur.take(label_len)?.to_vec())
            .context("store label is not utf8")?;
        let start = cur.f32()?;
        let end = cur.f32()?;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            vec.push(cur.f32()?);
        }
        out.push(NewRecord {
            kind,
            label,
            start,
            end,
            vec,
        });
    }
    Ok((dim, out))
}
```

Delete `decode`, `encode`, `write`, `write_done`, `done_path`, `read_done`, and `encode_for_test`. Keep `Cursor` as it is.

The delete in `append` takes each label of the batch. A video path that arrives again replaces its own rows. A note with the same text replaces itself, so the same note never lands twice.

- [ ] **Step 4: Update the index run**

In `videoembed/src/index.rs`, change the import line to `use crate::store::{self, NewRecord, Record};` — `Record` stays, because the grouping, the thinning, and the tests read it. Then change `IndexRun::step` to return `Result<Option<NewRecord>>` and build `NewRecord` in its end:

```rust
        Ok(Some(NewRecord {
            kind,
            label: self.label.clone(),
            start: start as f32,
            end: end.min(self.duration) as f32,
            vec,
        }))
```

Change `index_video` to return `Result<Vec<NewRecord>>`. In the test module, change `record_of` to build a `NewRecord`, and `record` to keep building one.

- [ ] **Step 5: Update the console**

In `videoembed/src/main.rs`:

1. Change the import to `use videoembed::store::{KIND_FRAME, KIND_NOTE, NewRecord, Record, Store};`.
2. In `note`, build a `NewRecord`:

```rust
    store.append(vec![NewRecord {
        kind: KIND_NOTE,
        label: rest.to_string(),
        start: 0.0,
        end: 0.0,
        vec,
    }])?;
```

3. Leave every other call as it is. `store.records[*i].label`, `.kind`, `.start`, and `.end` still work.

- [ ] **Step 6: Update the web binary**

In `videoembed/src/bin/web.rs`:

1. Change the import to `use videoembed::store::{NewRecord, Record, Store, normalize_label};`.
2. Change `WebJob.records` to `Vec<NewRecord>`.
3. Change `run_left_a_whole_index(outcome: &Result<Option<NewRecord>>) -> bool`.
4. In the test `a_video_row_counts_frames_and_audio_together`, build three `NewRecord` values with a 768-value vector. Add this helper to the test module:

```rust
    /// A vector of the model size, with one 1.0 at the front.
    fn unit() -> Vec<f32> {
        let mut vec = vec![0.0f32; videoembed::db::DIM];
        vec[0] = 1.0;
        vec
    }
```

5. In the test `only_a_run_that_reached_its_end_is_whole`, build a `NewRecord` with `vec: unit()`.

- [ ] **Step 7: Run the tests and watch them pass**

Run: `cd videoembed && cargo test`

Expected: every test passes. The counts hold: 60 lib tests, 2 console tests, 18 web tests, plus the new store tests and the four database tests. The run prints no warning.

- [ ] **Step 8: Commit**

```bash
git add videoembed/src/store.rs videoembed/src/index.rs videoembed/src/main.rs videoembed/src/bin/web.rs
git commit -m "feat: move the store to the SQLite database

The store reads and writes the records and the videos tables, and one
transaction holds each write. A second pass over a video replaces its
own rows. The in-memory record list holds no vector.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: The rank runs in the vector table

**Files:**
- Modify: `videoembed/src/store.rs`

**Interfaces:**
- Consumes: `Store::records`, the `vec_records` table.
- Produces: `Store::index_of_rid(&self, rid: i64) -> Option<usize>` and the real `Store::rank`.

- [ ] **Step 1: Write the failing tests**

Add these tests to `videoembed/src/store.rs`:

```rust
    /// A vector with one 1.0 at `at`, on the model size.
    fn basis(at: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; db::DIM];
        vec[at] = 1.0;
        vec
    }

    #[test]
    fn the_rank_returns_every_record_best_first() {
        let (dir, mut store) = opened("rank");
        let mut near = new_record("a.mp4", 0.0, 0.25);
        near.vec = basis(0);
        let mut far = new_record("b.mp4", 0.0, 0.25);
        far.vec = basis(1);
        store.append(vec![near, far]).unwrap();

        let ranked = store.rank(&basis(0));

        assert_eq!(ranked.len(), 2, "the median needs every score");
        assert_eq!(store.records[ranked[0].0].label, "a.mp4");
        assert!(
            ranked[0].1 > ranked[1].1,
            "the near record must score higher"
        );
        assert!(
            (ranked[0].1 - 1.0).abs() < 1e-4,
            "a unit vector against itself scores 1.0, got {}",
            ranked[0].1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_rank_of_an_empty_store_is_empty() {
        let (dir, store) = opened("rank_empty");
        assert!(store.rank(&basis(0)).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A note takes part in the rank, like a frame record.
    #[test]
    fn the_rank_holds_a_note() {
        let (dir, mut store) = opened("rank_note");
        let mut note = new_record("birthday", 0.0, 0.0);
        note.kind = KIND_NOTE;
        note.vec = basis(0);
        store.append(vec![note]).unwrap();

        let ranked = store.rank(&basis(0));

        assert_eq!(ranked.len(), 1);
        assert_eq!(store.records[ranked[0].0].kind, KIND_NOTE);
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cd videoembed && cargo test --lib store::tests::the_rank`

Expected: FAIL. `the_rank_returns_every_record_best_first` fails on the score, because the temporary body gives 0.0 for every record.

- [ ] **Step 3: Write the rank**

Replace the temporary `rank` and add the position lookup:

```rust
    /// The position of one row id in the record list.
    fn position_of(&self, rid: i64) -> Option<usize> {
        self.records.iter().position(|record| record.rid == rid)
    }

    /// Rank every record against one query vector. Best first.
    ///
    /// The query asks the vector table for every row and its distance. The
    /// table holds cosine distance, so the score is one minus the distance.
    /// The rank needs every row, because the median of the store is the
    /// baseline of the result.
    pub fn rank(&self, query: &[f32]) -> Vec<(usize, f32)> {
        if self.records.is_empty() {
            return Vec::new();
        }
        let sql = "select rid, distance from vec_records where embedding match vec_f32(?1) and k = ?2";
        let mut stmt = match self.conn.prepare(sql) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map(
            params![db::vec_to_blob(query), self.records.len() as i64],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)? as f32)),
        );
        let Ok(rows) = rows else {
            return Vec::new();
        };
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for row in rows.flatten() {
            if let Some(at) = self.position_of(row.0) {
                scored.push((at, 1.0 - row.1));
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cd videoembed && cargo test --lib store::`

Expected: PASS, every store test.

- [ ] **Step 5: Run the whole suite**

Run: `cd videoembed && cargo test`

Expected: every test passes. The console test `the_console_search_ranks_a_query` style tests that read a real store still pass, because the rank keeps its order.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/store.rs
git commit -m "feat: rank the records inside the vector table

The query asks vec0 for every row and its cosine distance. The score is
one minus the distance. The rank keeps its order and its shape.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: The old store imports at open, and the store path changes

**Files:**
- Modify: `videoembed/src/store.rs`
- Modify: `videoembed/src/main.rs`
- Modify: `videoembed/src/bin/web.rs`

**Interfaces:**
- Consumes: `vei_records(&[u8])`, `Store::append`.
- Produces: `Store::open` imports the old file one time.

- [ ] **Step 1: Write the failing tests**

Add these tests to `videoembed/src/store.rs`:

```rust
    /// Build an old `VEI1` file from these records.
    fn vei_bytes(dim: usize, records: &[NewRecord]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for record in records {
            out.push(record.kind);
            out.extend_from_slice(&(record.label.len() as u32).to_le_bytes());
            out.extend_from_slice(record.label.as_bytes());
            out.extend_from_slice(&record.start.to_le_bytes());
            out.extend_from_slice(&record.end.to_le_bytes());
            for x in &record.vec {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn an_old_store_imports_into_the_database() {
        let dir = scratch("import");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let old = vec![new_record("D:/video/a.mp4", 0.0, 0.25)];
        std::fs::write(dir.join("index.vei"), vei_bytes(db::DIM, &old)).unwrap();

        let store = Store::open(&path).unwrap();

        assert_eq!(store.records.len(), 1);
        assert_eq!(store.records[0].label, "D:/video/a.mp4");
        assert!(store.is_done("D:/video/a.mp4"), "an old video is finished");
        assert!(
            dir.join("index.vei").exists(),
            "the old file must stay after the import"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_import_runs_one_time() {
        let dir = scratch("import_once");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let old = vec![new_record("a.mp4", 0.0, 0.25)];
        std::fs::write(dir.join("index.vei"), vei_bytes(db::DIM, &old)).unwrap();
        let _ = Store::open(&path).unwrap();

        let again = Store::open(&path).unwrap();

        assert_eq!(again.records.len(), 1, "the import must not double the rows");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An old store from another model must not import.
    #[test]
    fn an_old_store_of_another_size_is_refused() {
        let dir = scratch("import_size");
        let path = dir.join("index.db");
        std::fs::create_dir_all(&dir).unwrap();
        let mut record = new_record("a.mp4", 0.0, 0.25);
        record.vec = vec![1.0, 0.0];
        std::fs::write(dir.join("index.vei"), vei_bytes(2, &[record])).unwrap();

        let err = match Store::open(&path) {
            Ok(_) => panic!("an old store of size 2 was accepted"),
            Err(err) => err,
        };

        assert!(format!("{err:#}").contains("index.vei"), "got {err:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cd videoembed && cargo test --lib store::tests::an_old_store`

Expected: FAIL with `assertion failed: store.records.len() == 1`, because the open call reads no old file.

- [ ] **Step 3: Write the import**

In `videoembed/src/store.rs`, change `open` to call the import, and add the function:

```rust
    /// Open the store. A missing file gives an empty store.
    ///
    /// An old `VEI1` file beside the database imports one time. The old file
    /// stays, so a fault costs nothing.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = db::open(path)?;
        let mut store = Self {
            path: path.to_path_buf(),
            conn,
            dim: db::DIM,
            records: Vec::new(),
            done: HashSet::new(),
        };
        store.import_old_file()?;
        store.load_records()?;
        store.done = store.load_done()?;
        Ok(store)
    }

    /// Import the records of an old `VEI1` file beside the store.
    ///
    /// The import runs when the old file is here and the database holds no
    /// record. Every imported video takes the finished mark, so the next run
    /// does not index it again.
    fn import_old_file(&mut self) -> Result<()> {
        let old = self.path.with_extension("vei");
        if !old.exists() {
            return Ok(());
        }
        let held: i64 = self
            .conn
            .query_row("select count(*) from records", [], |row| row.get(0))?;
        if held > 0 {
            return Ok(());
        }
        let bytes = std::fs::read(&old)
            .with_context(|| format!("read the old store {}", old.display()))?;
        let (dim, records) =
            vei_records(&bytes).with_context(|| format!("read the old store {}", old.display()))?;
        if dim != self.dim {
            bail!(
                "the old store {} holds vectors of size {dim}, but the model gives size {}",
                old.display(),
                self.dim
            );
        }
        let labels: Vec<String> = records
            .iter()
            .filter(|r| is_video(r.kind))
            .map(|r| r.label.clone())
            .collect();
        self.append(records)?;
        for label in labels {
            self.mark_done(&label)?;
        }
        println!("Imported {} records from {}", self.records.len(), old.display());
        Ok(())
    }
```

Remove the `#[allow(dead_code)]` from `vei_records`.

- [ ] **Step 4: Change the store path**

In `videoembed/src/main.rs`:

```rust
    let store_path = PathBuf::from("store/index.db");
```

and:

```rust
fn store_path_hint() -> &'static str {
    "store/index.db"
}
```

In `videoembed/src/bin/web.rs`, change `let store_path = PathBuf::from("store/index.vei");` to `"store/index.db"`.

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cd videoembed && cargo test`

Expected: every test passes, including the three import tests.

- [ ] **Step 6: Commit**

```bash
git add videoembed/src/store.rs videoembed/src/main.rs videoembed/src/bin/web.rs
git commit -m "feat: import an old store at open, and move to store/index.db

The open call reads index.vei one time when the database holds no
record. The imported videos take the finished mark. The old file stays.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: The measure command reads the vectors it needs

**Files:**
- Modify: `videoembed/src/store.rs`
- Modify: `videoembed/src/index.rs`
- Modify: `videoembed/src/main.rs`

**Interfaces:**
- Consumes: `Store::records` with `rid`.
- Produces: `Store::vectors(&self, rids: &[i64]) -> Result<Vec<Vec<f32>>>` and `index::thin_indexes(records: &[Record], factor: usize) -> Vec<usize>`.

- [ ] **Step 1: Write the failing tests**

Add this test to `videoembed/src/store.rs`:

```rust
    #[test]
    fn the_vectors_come_back_in_the_order_of_the_ids() {
        let (dir, mut store) = opened("vectors");
        let mut first = new_record("a.mp4", 0.0, 0.25);
        first.vec = basis(0);
        let mut second = new_record("a.mp4", 0.25, 0.5);
        second.vec = basis(1);
        store.append(vec![first, second]).unwrap();
        let rids: Vec<i64> = store.records.iter().map(|r| r.rid).collect();

        let vectors = store.vectors(&[rids[1], rids[0]]).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], basis(1), "the order of the ids must hold");
        assert_eq!(vectors[1], basis(0));
        let _ = std::fs::remove_dir_all(&dir);
    }
```

Add this test to `videoembed/src/index.rs`:

```rust
    /// The thinning gives the indexes of the kept records, in store order.
    #[test]
    fn the_thinning_gives_the_indexes_of_the_kept_records() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("a.mp4", 0.25, 0.5),
            record("a.mp4", 0.5, 0.75),
            record("a.mp4", 0.75, 1.0),
        ];

        let kept = thin_indexes(&records, 2);

        assert_eq!(kept, vec![0, 2]);
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cd videoembed && cargo test --lib store::tests::the_vectors`

Expected: FAIL with `no method named vectors` and `cannot find function thin_indexes`.

- [ ] **Step 3: Write the two calls**

In `videoembed/src/store.rs`, add the method to `impl Store`:

```rust
    /// Read the vectors of these rows, in the order of the ids.
    ///
    /// The measure command uses this. The vectors stay on disk, so a caller
    /// asks for the ones it needs.
    pub fn vectors(&self, rids: &[i64]) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("select embedding from vec_records where rid = ?1")?;
        let mut out = Vec::with_capacity(rids.len());
        for rid in rids {
            let blob: Vec<u8> = stmt
                .query_row([rid], |row| row.get(0))
                .with_context(|| format!("read the vector of row {rid}"))?;
            out.push(db::blob_to_vec(&blob));
        }
        Ok(out)
    }
```

In `videoembed/src/index.rs`, add:

```rust
/// The indexes of the records that stay after the thinning.
///
/// The thinning measures the cost of a coarser time grid. Every `factor`th
/// frame of one video stays, and the audio keeps its one second grid.
pub fn thin_indexes(records: &[Record], factor: usize) -> Vec<usize> {
    if factor < 2 {
        return (0..records.len()).collect();
    }
    let mut order: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        if record.kind == crate::store::KIND_FRAME {
            order.entry(record.label.as_str()).or_default().push(index);
        }
    }
    let mut keep = vec![true; records.len()];
    for indexes in order.values_mut() {
        indexes.sort_by(|a, b| records[*a].start.total_cmp(&records[*b].start));
        for (at, index) in indexes.iter().enumerate() {
            if at % factor != 0 {
                keep[*index] = false;
            }
        }
    }
    (0..records.len()).filter(|at| keep[*at]).collect()
}
```

Delete `thin_frames`, and change the tests that used it to use `thin_indexes`. The test `thinning_keeps_one_frame_of_each_group` becomes:

```rust
    #[test]
    fn thinning_keeps_one_frame_of_each_group() {
        let mut records: Vec<Record> = (0..8)
            .map(|i| record("a.mp4", i as f32 * 0.25, i as f32 * 0.25 + 0.25))
            .collect();
        records.push(record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0));
        records.push(record_of(crate::store::KIND_AUDIO, "a.mp4", 1.0, 2.0));

        let kept = thin_indexes(&records, 4);

        assert_eq!(kept.len(), 4, "two frames and two audio records");
        let starts: Vec<f32> = kept.iter().map(|at| records[*at].start).collect();
        assert_eq!(starts, vec![0.0, 1.0, 0.0, 1.0]);
    }
```

Change `a_factor_of_one_keeps_everything` and `each_video_thins_on_its_own` to read the kept indexes in the same way.

- [ ] **Step 4: Update the measure command**

In `videoembed/src/main.rs`, replace the body of the factor loop in `measure`:

```rust
    for factor in [1usize, 2, 4, 8] {
        let kept = index::thin_indexes(&store.records, factor);
        let frames = kept
            .iter()
            .filter(|at| store.records[**at].kind == KIND_FRAME)
            .count();
        if frames == 0 {
            println!("{factor:>6}  no frame records");
            continue;
        }
        let rids: Vec<i64> = kept.iter().map(|at| store.records[*at].rid).collect();
        let vectors_of: Vec<Vec<f32>> = store.vectors(&rids)?;
        let grid = index::FRAME_STEP * factor as f64;
        let mut line = Vec::new();
        for (at, query) in queries.iter().enumerate() {
            let mut ranked: Vec<(usize, f32)> = (0..kept.len())
                .map(|i| (i, dot(&vectors[at], &vectors_of[i])))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let baseline = index::median(&ranked);
            let thin: Vec<Record> = kept.iter().map(|at| store.records[*at].clone()).collect();
            let groups = index::group_hits(&thin, &ranked, baseline);
            let best = ranked
                .first()
                .map(|(index, _)| format!("{:.2} s", thin[*index].start))
                .unwrap_or_else(|| "none".to_string());
            let moment = match groups.first().and_then(|g| g.hits.first()) {
                Some(hit) => format!("{:.2}-{:.2} s", hit.start, hit.end),
                None => "none".to_string(),
            };
            line.push(format!("{query}: {best} ({moment})"));
        }
        println!("{factor:>6}  {grid:>6.2}s  {frames:>6}  {}", line.join("  |  "));
    }
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cd videoembed && cargo test`

Expected: every test passes.

- [ ] **Step 6: Run the console against a real store by hand**

Run: `cd videoembed && cargo run --release -- measure blue` and then `index ../spike/testvideo_nolabels.mp4` and `search blue`.

Expected: the measure table prints four rows with a factor, a grid, a frame count, and a time for each query. `search blue` finds a moment near 10-20 s, as it does today.

- [ ] **Step 7: Commit**

```bash
git add videoembed/src/store.rs videoembed/src/index.rs videoembed/src/main.rs
git commit -m "feat: read the vectors of the thinned records for the measure

The thinning gives the indexes of the kept records, and the store reads
those vectors on request.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: The documents

**Files:**
- Modify: `docs/deferred-work.md`
- Modify: `docs/store-format-change.md`
- Modify: `docs/google-gallery-comparison.md`

**Interfaces:**
- Consumes: the finished store.
- Produces: three documents that match the code.

- [ ] **Step 1: Update the deferred list**

In `docs/deferred-work.md`, remove the row `| 9 | Upsert and delete by record id | Needs a real store format. Items 8 and 11 wait on it. |`, and change the row for item 8 to:

```
| 8 | A per-video plan, with presets | The store holds the plan. The indexer does not use it yet. |
```

Change the optimisation row `| The store write | ... |` to:

```
| The store search | One query over every vector for each query. The vector table answers it. A vector index comes near one million records. |
```

- [ ] **Step 2: Update the store analysis**

In `docs/store-format-change.md`, change the status line to:

```
Status: the store is SQLite with the sqlite-vec extension. The format change is done.
```

and add a line at the end of section 9:

```
The change landed on 2026-10-11. The code is in `videoembed/src/db.rs` and
`videoembed/src/store.rs`. Window mode is the next item.
```

- [ ] **Step 3: Update the comparison**

In `docs/google-gallery-comparison.md`:

1. In the section 2.3 table, change the "This app" row for Store to `One SQLite database, store/index.db`, the Table row to `videos`, `records`, `vec_records`, the Record id row to `"path/startMs-endMs"`, the Write path row to `An upsert by id, in one transaction`, and the Delete row to `By id, one video at a time`.
2. In section 3, remove the gain `**No database dependency.** One flat file, one binary. Google needs SQLite, MediaPipe, WorkManager, and Proto DataStore.` and replace it with:

```
5. **A small database dependency.** One crate, `rusqlite`, with the bundled
   SQLite. Google needs SQLite, MediaPipe, WorkManager, and Proto DataStore.
```

3. In section 4, remove `5. **Idempotent upsert and prefix delete**, so one video updates on its own.` and renumber the list.
4. In section 6, remove `- The flat store, until feature 9 earns its cost.`

- [ ] **Step 4: Lint the documents**

Run:

```bash
node "C:/Users/emir-/.agents/skills/asd-ste100/hooks/run-python.cjs" "C:/Users/emir-/.agents/skills/asd-ste100/scripts/ste-lint.py" --fail-over 2.5 docs/deferred-work.md docs/store-format-change.md docs/google-gallery-comparison.md
```

Expected: each file reports a `per100w` value below 2.5.

- [ ] **Step 5: Commit**

```bash
git add docs/deferred-work.md docs/store-format-change.md docs/google-gallery-comparison.md
git commit -m "docs: record the SQLite store

Item 9 comes off the deferred list. The comparison records the database,
the id, and the upsert.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| 2. The decisions | Tasks 1, 3, 4, 5 |
| 3. The schema | Task 1 |
| 4. The write path | Task 3 |
| 5. The read paths | Tasks 4, 6 |
| 6. The import | Task 5 |
| 7. The errors | Tasks 1 (busy timeout, extension), 3 (transaction), 4 (empty store), 5 (import fault) |
| 8. The tests | Tasks 3, 4, 5, 6 |
| 9. What does not change | Tasks 3, 4 |
| 10. Out of scope | No task, by design |
| 11. Open items | Task 1 pins the `vec0` key and the partition syntax on the machine |

The spec holds one change: it says the delete uses the id prefix. The plan uses the `path` column instead, because the revised schema carries that column. The prefix rule stays in the id, so a record of one video never reaches another.

**Placeholder scan:** none. Every step holds the code or the command.

**Type consistency:** `NewRecord` and `record_id` come from Task 2 and stay. `Record` gains `rid` and `id` and loses `vec` in Task 3, and every later task uses that shape. `thin_indexes` replaces `thin_frames` in Task 6 and the older name appears nowhere after it.
