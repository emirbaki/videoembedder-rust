//! A throwaway probe for deferred item 1: window mode.
//!
//! Window mode holds one 2 s window in one record: two frames and the window
//! audio, embedded in one batch. The fine mode holds a 0.25 s frame grid.
//!
//! The probe indexes one video both ways, and each window form with and without
//! audio. It compares ten queries against a known truth. The video is 30 s, in
//! three 10 s blocks: sunny, tuesday, birthday.
//!
//! Run from the `videoembed` folder:
//!   VEI_MODELS=../models cargo run --release --example window_probe

use std::path::{Path, PathBuf};

use anyhow::Result;

use videoembed::embed::{Embedder, QUERY_PREFIX, dot};
use videoembed::store::KIND_FRAME;
use videoembed::{index, media};

/// One 2 s window, on a 1 s stride.
const WINDOW: f64 = 2.0;
const STRIDE: f64 = 1.0;
/// One frame each second, so a 2 s window holds two frames.
const FPS: f64 = 1.0;

/// name, query, the true block as (start, end).
const QUERIES: &[(&str, &str, (f64, f64))] = &[
    ("sunny", "sunny warm weather", (0.0, 10.0)),
    ("weather", "the weather is sunny today", (0.0, 10.0)),
    ("tuesday", "quiet tuesday morning", (10.0, 20.0)),
    ("morning", "it is a quiet morning", (10.0, 20.0)),
    ("birthday", "birthday party", (20.0, 30.0)),
    ("celebrate", "we celebrate a birthday", (20.0, 30.0)),
    ("blue", "blue", (10.0, 20.0)),
    ("green", "green", (20.0, 30.0)),
    ("orange", "orange", (0.0, 10.0)),
    ("party", "a party with friends", (20.0, 30.0)),
];

fn main() -> Result<()> {
    let models = PathBuf::from(std::env::var("VEI_MODELS").unwrap_or("../models".into()));
    let video = PathBuf::from("../spike/testvideo_nolabels.mp4");
    let embedder = Embedder::load(&models, None)?;

    println!("video: {}", video.display());
    println!();

    let with_audio = index_windows(&embedder, &video, true)?;
    let no_audio = index_windows(&embedder, &video, false)?;
    let fine = index_fine(&embedder, &video)?;
    println!("window + audio: {} records", with_audio.len());
    println!("window, no audio: {} records", no_audio.len());
    println!("fine:           {} records", fine.len());
    println!();
    println!(
        "{:<10} {:<28} {:<12} {:<8} {:<8} fine",
        "name", "query", "truth", "window", "no audio"
    );

    let mut win_hits = 0;
    let mut plain_hits = 0;
    let mut fine_hits = 0;
    for (name, text, truth) in QUERIES {
        let query = embedder.embed_text(&format!("{QUERY_PREFIX}{text}"))?;
        let w = overlaps(best_block(&with_audio, &query), *truth);
        let p = overlaps(best_block(&no_audio, &query), *truth);
        let f = overlaps(best_block(&fine, &query), *truth);
        win_hits += w as usize;
        plain_hits += p as usize;
        fine_hits += f as usize;
        println!(
            "{:<10} {:<28} {:<12} {:<8} {:<8} {}",
            name,
            text,
            format!("{:.0}-{:.0}s", truth.0, truth.1),
            if w { "OK" } else { "MISS" },
            if p { "OK" } else { "MISS" },
            if f { "OK" } else { "MISS" },
        );
    }
    println!();
    println!(
        "hits: window+audio {}/{}  no audio {}/{}  fine {}/{}",
        win_hits,
        QUERIES.len(),
        plain_hits,
        QUERIES.len(),
        fine_hits,
        QUERIES.len()
    );
    Ok(())
}

/// Index the video in window mode: one record for each 2 s window.
fn index_windows(
    embedder: &Embedder,
    video: &Path,
    audio: bool,
) -> Result<Vec<(f64, f64, Vec<f32>)>> {
    let duration = media::probe_duration(video)?;
    let mut out = Vec::new();
    let mut ctx = embedder.new_context()?;
    let mut t = 0.0;
    while t + WINDOW <= duration {
        let dir = index::work_dir(out.len());
        let mut parts = media::extract_frames(video, t, t + WINDOW, FPS, &dir)?;
        if audio {
            parts.extend(media::extract_audio(video, t, t + WINDOW, &dir)?);
        }
        let refs: Vec<&Path> = parts.iter().map(|path| path.as_path()).collect();
        let vec = embedder.embed_window_in(&mut ctx, &refs)?;
        out.push((t, t + WINDOW, vec));
        let _ = std::fs::remove_dir_all(&dir);
        t += STRIDE;
    }
    Ok(out)
}

/// Index the video in the current fine mode, from the store's own plan.
fn index_fine(embedder: &Embedder, video: &Path) -> Result<Vec<(f64, f64, Vec<f32>)>> {
    let mut run = index::IndexRun::start(video)?;
    let mut ctx = embedder.new_context()?;
    let mut out = Vec::new();
    while let Some(record) = run.step(embedder, &mut ctx)? {
        if record.kind == KIND_FRAME {
            out.push((record.start as f64, record.end as f64, record.vec));
        }
    }
    Ok(out)
}

/// The block of the best record for one query.
fn best_block(records: &[(f64, f64, Vec<f32>)], query: &[f32]) -> (f64, f64) {
    records
        .iter()
        .max_by(|a, b| dot(query, &a.2).total_cmp(&dot(query, &b.2)))
        .map(|(start, end, _)| (*start, *end))
        .unwrap_or((0.0, 0.0))
}

fn overlaps(hit: (f64, f64), truth: (f64, f64)) -> bool {
    hit.0 < truth.1 && truth.0 < hit.1
}
