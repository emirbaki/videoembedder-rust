//! The database. One connection, the sqlite-vec extension, and the schema.
//!
//! The vectors live in the `vec_records` table. The metadata lives in the
//! `records` and `videos` tables. One `rid` joins a record to its vector.

use std::path::Path;
use std::sync::Once;
use std::time::Duration;

use anyhow::{Context, Result, bail};
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

/// Write one vector as a blob. One value takes four bytes, little-endian.
pub fn vec_to_blob(vec: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for x in vec {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Read a stored vector blob. One value takes four bytes, little-endian.
///
/// A write gives the blob a whole number of values. A length that is not a
/// multiple of four names a damaged row, so the read refuses the blob.
pub fn blob_to_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!(
            "a vector blob of {} bytes does not hold 32-bit values",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes([part[0], part[1], part[2], part[3]]))
        .collect())
}

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

        assert_eq!(blob_to_vec(&blob).unwrap(), vec_at(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A blob that holds part of a value names a damaged row.
    #[test]
    fn a_blob_of_a_partial_value_is_refused() {
        let err = blob_to_vec(&[1u8, 2, 3, 4, 5, 6]).unwrap_err();

        assert!(format!("{err}").contains('6'), "{err}");
    }

    /// An empty blob is no vector, and the read says so without a fault.
    #[test]
    fn an_empty_blob_is_no_vector() {
        assert_eq!(blob_to_vec(&[]).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn a_knn_query_returns_the_nearest_row() {
        let dir = scratch("knn");
        let conn = open(&dir.join("index.db")).unwrap();
        for at in 0..3 {
            conn.execute(
                "insert into vec_records(rid, path, embedding) values (?1, 'a.mp4', vec_f32(?2))",
                rusqlite::params![(at + 1) as i64, vec_to_blob(&vec_at(at))],
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
