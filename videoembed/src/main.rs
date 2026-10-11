//! A console for video and text embedding search.
//!
//! Type a command at the prompt. Each command works one step at a time.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use videoembed::embed::{DOC_PREFIX, Device, Embedder, QUERY_PREFIX, dot};
use videoembed::index;
use videoembed::store::{KIND_FRAME, KIND_NOTE, NewRecord, Record, Store};

fn main() -> Result<()> {
    let models = models_dir();
    let store_path = PathBuf::from("store/index.db");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (asked, args) = split_device(args);
    println!("Loading models from {}", models.display());
    let mut embedder = Embedder::load(&models, asked)?;
    let mut store = Store::open(&store_path)?;
    println!(
        "Store: {} ({} records)",
        store_path.display(),
        store.records.len()
    );
    println!("Commands: index <path> | note <text> | search <query> | list | forget <path> | device gpu|cpu | quit");
    println!("Measure the time grid: measure <query> | <another query>");
    println!("Start with --gpu or --cpu to pick the mode. The default is the card when it is here.");

    if !args.is_empty() {
        run(&mut embedder, &models, &mut store, &args.join(" "))?;
        return Ok(());
    }

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Err(err) = run(&mut embedder, &models, &mut store, line) {
            println!("Error: {err:#}");
        }
    }
    Ok(())
}

/// Take the mode flag out of the arguments.
///
/// `--gpu` and `--cpu` name the device for the model load. Every other word
/// stays in the command line, so a foreign flag reaches the command parser.
fn split_device(args: Vec<String>) -> (Option<Device>, Vec<String>) {
    let mut asked = None;
    let mut rest = Vec::new();
    for word in args {
        match word.strip_prefix("--").and_then(Device::parse) {
            Some(device) => asked = Some(device),
            None => rest.push(word),
        }
    }
    (asked, rest)
}

fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("VEI_MODELS") {
        return PathBuf::from(dir);
    }
    let local = PathBuf::from("models");
    if local.is_dir() {
        local
    } else {
        PathBuf::from("../models")
    }
}

fn run(embedder: &mut Embedder, models: &Path, store: &mut Store, line: &str) -> Result<()> {
    let (command, rest) = split_command(line);
    match command {
        "index" => index(embedder, store, rest),
        "note" => note(embedder, store, rest),
        "search" => search(embedder, store, rest),
        "measure" => measure(embedder, store, rest),
        "device" => device(embedder, models, rest),
        "list" => {
            list(store);
            Ok(())
        }
        "forget" => forget(store, rest),
        "quit" | "exit" => {
            std::process::exit(0);
        }
        other => {
            println!("Unknown command: {other}");
            Ok(())
        }
    }
}

fn split_command(line: &str) -> (&str, &str) {
    match line.find(char::is_whitespace) {
        Some(at) => (&line[..at], line[at..].trim()),
        None => (line, ""),
    }
}

/// Load the model again on another device.
///
/// The load takes some seconds, and the old model serves until the new one is
/// ready. A failure leaves the console on the device it had.
fn device(embedder: &mut Embedder, models: &Path, rest: &str) -> Result<()> {
    let Some(device) = Device::parse(rest) else {
        bail!("usage: device gpu | device cpu");
    };
    if device == embedder.device() {
        println!("Already on {}.", device.as_str());
        return Ok(());
    }
    println!("Loading the model on {}.", device.as_str());
    embedder.reload(models, device)?;
    println!("Now on {}.", device.as_str());
    Ok(())
}

/// Show where the best answer lands on a coarser time grid.
///
/// The frames are already embedded. A factor of 4 turns the 4 fps grid into a
/// 1 fps grid, so the run costs one text embedding for each query. The run
/// does not measure the audio packing or the timestamp text of the Google
/// design. It measures the time grid alone.
fn measure(embedder: &Embedder, store: &Store, rest: &str) -> Result<()> {
    let queries: Vec<&str> = rest
        .split('|')
        .map(|query| query.trim())
        .filter(|query| !query.is_empty())
        .collect();
    if queries.is_empty() {
        bail!("usage: measure <query> | <query> ...");
    }
    let vectors: Vec<Vec<f32>> = queries
        .iter()
        .map(|query| embedder.embed_text(&format!("{QUERY_PREFIX}{query}")))
        .collect::<Result<Vec<_>>>()?;

    println!(
        "{:>6}  {:>7}  {:>6}  query: best frame, then the moment",
        "factor", "grid", "frames"
    );
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
        // The vectors stay on disk, so the measure asks for the ones the
        // thinning kept, in the same order.
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
            // The best frame tells the grid cost. The moment tells what a
            // person sees, and it spans every touching hit above the floor.
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
    println!();
    println!("A factor of 4 makes a 1.00 s grid and a factor of 8 makes a 2.00 s grid.");
    println!("The run does not measure the audio packing or the timestamp text.");
    Ok(())
}

/// Cut one video into frames and audio clips, embed each one, and append.
fn index(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: index <video path>");
    }
    let video = PathBuf::from(rest);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    println!("Video: {}", video.display());
    let mut report = |n: usize, total: usize| println!("  item {n:5}/{total}");
    let records = index::index_video(embedder, &video, &mut report)?;
    let count = records.len();
    store.append(records)?;
    println!("Indexed {count} records into {}", store_path_hint());
    Ok(())
}

/// Embed one free text record.
fn note(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: note <text>");
    }
    let vec = embedder.embed_text(&format!("{DOC_PREFIX}{rest}"))?;
    store.append(vec![NewRecord {
        kind: KIND_NOTE,
        label: rest.to_string(),
        start: 0.0,
        end: 0.0,
        vec,
    }])?;
    println!("Added note: {rest}");
    Ok(())
}

/// Rank every record against the query.
fn search(embedder: &Embedder, store: &Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: search <query>");
    }
    if store.records.is_empty() {
        println!("The store is empty. Run index or note first.");
        return Ok(());
    }
    let query = embedder.embed_text(&format!("{QUERY_PREFIX}{rest}"))?;
    let ranked = store.rank(&query, None)?;
    let baseline = index::median(&ranked);
    println!("Top results for: {rest}");
    println!("Baseline (median of the store): {baseline:.3}");
    for (rank, (i, score)) in ranked.iter().take(10).enumerate() {
        let record = &store.records[*i];
        let margin = score - baseline;
        let bar = bar(*score);
        match record.kind {
            KIND_NOTE => println!(
                "{:2}. {:.3} ({:+.3}) {} note: {}",
                rank + 1,
                score,
                margin,
                bar,
                record.label
            ),
            _ => println!(
                "{:2}. {:.3} ({:+.3}) {} {}  [{:.2}-{:.2} s]",
                rank + 1,
                score,
                margin,
                bar,
                record.label,
                record.start,
                record.end
            ),
        }
    }
    Ok(())
}

/// Group the records by source.
fn list(store: &Store) {
    if store.records.is_empty() {
        println!("The store is empty.");
        return;
    }
    let mut videos: Vec<(String, usize)> = Vec::new();
    let mut notes = 0;
    for record in &store.records {
        if record.kind == KIND_NOTE {
            notes += 1;
            continue;
        }
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        match videos.iter_mut().find(|(name, _)| *name == record.label) {
            Some((_, count)) => *count += 1,
            None => videos.push((record.label.clone(), 1)),
        }
    }
    println!("Videos ({}):", videos.len());
    for (name, count) in videos {
        println!("  {name}  ({count} records)");
    }
    println!("Notes: {notes}");
    println!("Total records: {}", store.records.len());
}

fn forget(store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: forget <video path>");
    }
    let removed = store.forget(rest)?;
    println!("Removed {removed} records for {rest}");
    Ok(())
}

fn bar(score: f32) -> String {
    let filled = ((score.max(0.0) * 20.0).round() as usize).min(20);
    "#".repeat(filled)
}

fn store_path_hint() -> &'static str {
    "store/index.db"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The start reads the mode from a flag, and the flag leaves the command.
    #[test]
    fn the_mode_flag_comes_out_of_the_arguments() {
        let (asked, rest) = split_device(vec!["index".into(), "a.mp4".into()]);
        assert_eq!(asked, None);
        assert_eq!(rest, vec!["index", "a.mp4"]);

        let (asked, rest) = split_device(vec!["--cpu".into(), "index".into()]);
        assert_eq!(asked, Some(Device::Cpu));
        assert_eq!(rest, vec!["index"]);

        let (asked, rest) = split_device(vec!["search".into(), "--gpu".into(), "blue".into()]);
        assert_eq!(asked, Some(Device::Gpu));
        assert_eq!(rest, vec!["search", "blue"]);
    }

    /// A word that starts with `--` is not a mode, and it stays in the command.
    #[test]
    fn a_foreign_flag_stays_in_the_command() {
        let (asked, rest) = split_device(vec!["--verbose".into(), "list".into()]);
        assert_eq!(asked, None);
        assert_eq!(rest, vec!["--verbose", "list"]);
    }
}
