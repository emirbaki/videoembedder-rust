//! A console for video and text embedding search.
//!
//! Type a command at the prompt. Each command works one step at a time.

mod embed;
mod media;
mod store;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Result, bail};

use embed::{DOC_PREFIX, Embedder, QUERY_PREFIX};
use store::{KIND_NOTE, KIND_WINDOW, Record, Store};

/// Window length in seconds.
const WINDOW: f64 = 4.0;
/// Window step in seconds.
const STRIDE: f64 = 2.0;
/// Small slack for the last window near the end of a video.
const EPS: f64 = 0.05;

fn main() -> Result<()> {
    let models = models_dir();
    let store_path = PathBuf::from("store/index.vei");
    println!("Loading models from {}", models.display());
    let embedder = Embedder::load(&models)?;
    let mut store = Store::open(&store_path)?;
    println!(
        "Store: {} ({} records)",
        store_path.display(),
        store.records.len()
    );
    println!("Commands: index <path> | note <text> | search <query> | list | forget <path> | quit");

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        run(&embedder, &mut store, &args.join(" "))?;
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
        if let Err(err) = run(&embedder, &mut store, line) {
            println!("Error: {err:#}");
        }
    }
    Ok(())
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

fn run(embedder: &Embedder, store: &mut Store, line: &str) -> Result<()> {
    let (command, rest) = split_command(line);
    match command {
        "index" => index(embedder, store, rest),
        "note" => note(embedder, store, rest),
        "search" => search(embedder, store, rest),
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

/// Cut one video into windows, embed each window, and append the records.
fn index(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: index <video path>");
    }
    let video = PathBuf::from(rest);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    let duration = media::probe_duration(&video)?;
    let label = video.to_string_lossy().to_string();
    println!("Video: {label}");
    println!("Duration: {duration:.2} s");

    let work = std::env::temp_dir().join(format!("vei_{}", std::process::id()));
    let mut records = Vec::new();
    let mut start = 0.0;
    let mut n = 0;
    while start + WINDOW <= duration + EPS {
        let end = start + WINDOW;
        n += 1;
        let dir = work.join(format!("w{n:03}"));
        let (frames, audio) = media::extract_window(&video, start, end, &dir)?;
        let vec = embedder.embed_window(&frames, &audio)?;
        println!(
            "  window {n:3}  {start:6.1}-{end:5.1} s  {} frames  dim {}",
            frames.len(),
            vec.len()
        );
        records.push(Record {
            kind: KIND_WINDOW,
            label: label.clone(),
            start: start as f32,
            end: end as f32,
            vec,
        });
        start += STRIDE;
    }
    let _ = std::fs::remove_dir_all(&work);
    if records.is_empty() {
        bail!("the video is shorter than one window");
    }
    let count = records.len();
    store.append(records)?;
    println!("Indexed {count} windows into {}", store_path_hint());
    Ok(())
}

/// Embed one free text record.
fn note(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: note <text>");
    }
    let vec = embedder.embed_text(&format!("{DOC_PREFIX}{rest}"))?;
    store.append(vec![Record {
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
    let ranked = store.rank(&query);
    println!("Top results for: {rest}");
    for (rank, (i, score)) in ranked.iter().take(10).enumerate() {
        let record = &store.records[*i];
        let bar = bar(*score);
        match record.kind {
            KIND_NOTE => println!(
                "{:2}. {:.3} {} note: {}",
                rank + 1,
                score,
                bar,
                record.label
            ),
            _ => println!(
                "{:2}. {:.3} {} {}  [{:.1}-{:.1} s]",
                rank + 1,
                score,
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
        match videos.iter_mut().find(|(name, _)| *name == record.label) {
            Some((_, count)) => *count += 1,
            None => videos.push((record.label.clone(), 1)),
        }
    }
    println!("Videos ({}):", videos.len());
    for (name, count) in videos {
        println!("  {name}  ({count} windows)");
    }
    println!("Notes: {notes}");
    println!("Total records: {}", store.records.len());
}

fn forget(store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: forget <video path>");
    }
    let removed = store.forget(rest)?;
    println!("Removed {removed} windows for {rest}");
    Ok(())
}

fn bar(score: f32) -> String {
    let filled = ((score.max(0.0) * 20.0).round() as usize).min(20);
    "#".repeat(filled)
}

fn store_path_hint() -> &'static str {
    "store/index.vei"
}
