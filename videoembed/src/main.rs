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
    if store.imported > 0 {
        println!("Imported {} records from the old store.", store.imported);
    }
    println!(
        "Store: {} ({} records)",
        store_path.display(),
        store.records.len()
    );
    println!("Commands: index <path> | plan <path> <preset> | note <text> | search <query> [| <path>] | list | forget <path> | device gpu|cpu | quit");
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

/// Take the mode flag out of the head of the arguments.
///
/// `--gpu` and `--cpu` name the device for the model load, and they come
/// before the command. A flag past the command is a word of the command, so a
/// query with the text `--gpu` keeps it.
fn split_device(args: Vec<String>) -> (Option<Device>, Vec<String>) {
    let mut asked = None;
    let mut rest = Vec::new();
    let mut head = true;
    for word in args {
        if head {
            match word.strip_prefix("--").and_then(Device::parse) {
                Some(device) => {
                    asked = Some(device);
                    continue;
                }
                None => head = false,
            }
        }
        rest.push(word);
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
        "plan" => plan(embedder, store, rest),
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
        let thin: Vec<Record> = kept.iter().map(|at| store.records[*at].clone()).collect();
        let grid = index::FRAME_STEP * factor as f64;
        let mut line = Vec::new();
        for (at, query) in queries.iter().enumerate() {
            let mut ranked: Vec<(usize, f32)> = (0..kept.len())
                .map(|i| (i, dot(&vectors[at], &vectors_of[i])))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let baseline = index::median(&ranked);
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

/// The line the console shows before an index run.
fn index_line(plan: &videoembed::plan::Plan) -> String {
    if plan.mode == videoembed::plan::Mode::Fine {
        return "Indexing under the Fine plan (0.25 s frames).".to_string();
    }
    let audio = if plan.audio { "with audio" } else { "no audio" };
    format!(
        "Indexing under the {:?} plan ({} s windows, {} frames, {}).",
        preset_of(plan),
        plan.window,
        plan.frames,
        audio
    )
}

/// The line about the source of a cut. An empty line means no copy.
fn note_line(copy_from: Option<&str>) -> String {
    match copy_from {
        Some(path) => format!("Reading the progressive copy of {path}."),
        None => String::new(),
    }
}

/// The preset of a plan, by its values.
fn preset_of(plan: &videoembed::plan::Plan) -> videoembed::plan::Preset {
    videoembed::plan::Preset::ALL
        .into_iter()
        .find(|preset| videoembed::plan::Plan::of(*preset) == *plan)
        .unwrap_or(videoembed::plan::Preset::Object)
}

/// Cut one video under its plan, embed each item, and append.
fn index(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    if rest.is_empty() {
        bail!("usage: index <video path>");
    }
    let video = PathBuf::from(rest);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    let label = videoembed::store::normalize_label(&video.to_string_lossy());
    let plan = store
        .plan_of(&label)?
        .unwrap_or_else(videoembed::plan::Plan::default_plan);
    println!("Video: {}", video.display());
    println!("{}", index_line(&plan));
    let media = store.media_dir();
    let note = note_line(index::copy_note(&video, &media)?.as_deref());
    if !note.is_empty() {
        println!("{note}");
    }
    let mut report = |n: usize, total: usize| println!("  item {n:5}/{total}");
    let (records, notes) = index::index_video(embedder, &video, &plan, &media, &mut report)?;
    for line in &notes {
        println!("{line}");
    }
    let count = records.len();
    store.forget(&label)?;
    store.append(records)?;
    store.set_plan(&label, &plan)?;
    println!("Indexed {count} records into {}", store_path_hint());
    Ok(())
}

/// The source file of one stored video, and the check that it is here.
fn source_of(path: &str) -> Result<PathBuf> {
    let video = PathBuf::from(path);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    Ok(video)
}

/// Split a `plan` argument into the video path and the plan words.
///
/// The path may hold spaces. The plan words start at the first word that names
/// a preset or holds a value, so every word before that one belongs to the
/// path.
fn split_plan_args(rest: &str) -> Result<(String, String)> {
    let words: Vec<&str> = rest.split_whitespace().collect();
    for at in 1..words.len() {
        let word = words[at];
        if word.contains('=') || videoembed::plan::Preset::parse(word).is_some() {
            return Ok((words[..at].join(" "), words[at..].join(" ")));
        }
    }
    bail!("usage: plan <path> <preset> | plan <path> window=4 frames=4")
}

/// The plan a `plan` command asks for, over the stored plan.
///
/// The argument is `<path> <preset>` or `<path> <value> <value>`.
fn plan_command(store: &Store, rest: &str) -> Result<videoembed::plan::Plan> {
    let (path, words) = split_plan_args(rest)?;
    let label = videoembed::store::normalize_label(&path);
    let base = store
        .plan_of(&label)?
        .ok_or_else(|| anyhow::anyhow!("the store holds no video at {path}"))?;
    if let Some(preset) = videoembed::plan::Preset::parse(&words) {
        return Ok(videoembed::plan::Plan::of(preset));
    }
    base.set_words(&words)
}

/// Cut one video again under the plan the user names.
fn plan(embedder: &Embedder, store: &mut Store, rest: &str) -> Result<()> {
    let (path, _) = split_plan_args(rest)?;
    let wanted = plan_command(store, rest)?;
    source_of(&path)?;
    println!("Re-indexing {} under the new plan.", path);
    let media = store.media_dir();
    let note = note_line(index::copy_note(Path::new(&path), &media)?.as_deref());
    if !note.is_empty() {
        println!("{note}");
    }
    let mut report = |n: usize, total: usize| println!("  item {n:5}/{total}");
    let (count, notes) = index::reindex(embedder, store, &path, &wanted, &mut report)?;
    for line in &notes {
        println!("{line}");
    }
    println!("Indexed {count} records.");
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
/// Split the search argument into the query and an optional scope.
///
/// The last `|` holds the scope, so a query can keep one of its own. The
/// separator is the same one `measure` uses.
fn parse_search(rest: &str) -> (String, Option<String>) {
    match rest.rsplit_once('|') {
        Some((query, scope)) => {
            let scope = scope.trim();
            let query = query.trim().to_string();
            if scope.is_empty() {
                (query, None)
            } else {
                (query, Some(scope.to_string()))
            }
        }
        None => (rest.trim().to_string(), None),
    }
}

/// The line the console shows when the rank gives no row.
///
/// A scope that names no video takes its own line. No scope keeps the line
/// that names the start step.
fn no_result_line(scope: Option<&str>) -> String {
    match scope {
        Some(scope) => format!("No records for {scope}."),
        None => "The store is empty. Run index or note first.".to_string(),
    }
}

/// The number of rows a search shows.
const ROW_CAP: usize = 10;

fn search(embedder: &Embedder, store: &Store, rest: &str) -> Result<()> {
    let (text, scope) = parse_search(rest);
    if text.is_empty() {
        bail!("usage: search <query> [| <path>]");
    }
    if store.records.is_empty() {
        println!("{}", no_result_line(None));
        return Ok(());
    }
    let query = embedder.embed_text(&format!("{QUERY_PREFIX}{text}"))?;
    let ranked = store.rank(&query, scope.as_deref())?;
    if ranked.is_empty() {
        println!("{}", no_result_line(scope.as_deref()));
        return Ok(());
    }
    let baseline = index::median(&ranked);
    match &scope {
        Some(scope) => println!("Top results for: {text} in {scope}"),
        None => println!("Top results for: {text}"),
    }
    match &scope {
        Some(_) => println!("Baseline (median of the video): {baseline:.3}"),
        None => println!("Baseline (median of the store): {baseline:.3}"),
    }
    let cut = index::weak_cut(ranked[0].1);
    println!("Weak below: {cut:.3}");
    for (rank, row) in rows_of(&store.records, &ranked, baseline).iter().enumerate() {
        println!("{}", hit_line(rank, row, cut));
    }
    Ok(())
}

/// The rows a search shows: the ten best, folded into moments.
fn rows_of(records: &[Record], ranked: &[(usize, f32)], baseline: f32) -> Vec<index::Row> {
    index::search_rows(records, ranked, baseline, ROW_CAP, index::MERGE_GAP)
}

/// One line of the result list.
///
/// The row holds the whole moment, so the line shows the range of the moment. A
/// row below the cut takes the `weak` mark. The cut comes from the best score of
/// the result, so a strong match marks its weak rows.
fn hit_line(rank: usize, row: &index::Row, cut: f32) -> String {
    let mark = if index::is_weak(row.score, cut) {
        " weak"
    } else {
        ""
    };
    if row.note {
        return format!(
            "{:2}. {:.3} ({:+.3}) {} note: {}{}",
            rank + 1,
            row.score,
            row.margin,
            bar(row.score),
            row.label,
            mark
        );
    }
    format!(
        "{:2}. {:.3} ({:+.3}) {} {}  [{:.2}-{:.2} s]{}",
        rank + 1,
        row.score,
        row.margin,
        bar(row.score),
        row.label,
        row.start,
        row.end,
        mark
    )
}

/// Group the video records by their normalized path, in store order.
///
/// The path separators do not matter, so one video makes one row whatever
/// separator its label carries.
fn video_rows(records: &[Record]) -> Vec<(String, usize)> {
    let mut videos: Vec<(String, usize)> = Vec::new();
    for record in records {
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        let name = videoembed::store::normalize_label(&record.label);
        match videos.iter_mut().find(|(held, _)| *held == name) {
            Some((_, count)) => *count += 1,
            None => videos.push((name, 1)),
        }
    }
    videos
}

/// Group the records by source.
fn list(store: &Store) {
    if store.records.is_empty() {
        println!("The store is empty.");
        return;
    }
    let notes = store
        .records
        .iter()
        .filter(|record| record.kind == KIND_NOTE)
        .count();
    let videos = video_rows(&store.records);
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

        let (asked, rest) = split_device(vec!["--gpu".into(), "--cpu".into(), "index".into()]);
        assert_eq!(asked, Some(Device::Cpu), "the last leading flag wins");
        assert_eq!(rest, vec!["index"]);
    }

    /// A flag past the command is a word of the command, not a mode.
    ///
    /// The mode comes before the command. A word `--gpu` after the command
    /// belongs to the command, and the run must not load another model for it.
    #[test]
    fn a_flag_after_the_command_stays_in_the_command() {
        let (asked, rest) = split_device(vec!["search".into(), "--gpu".into(), "blue".into()]);
        assert_eq!(asked, None);
        assert_eq!(rest, vec!["search", "--gpu", "blue"]);
    }

    /// A word that starts with `--` is not a mode, and it stays in the command.
    #[test]
    fn a_foreign_flag_stays_in_the_command() {
        let (asked, rest) = split_device(vec!["--verbose".into(), "list".into()]);
        assert_eq!(asked, None);
        assert_eq!(rest, vec!["--verbose", "list"]);
    }

    /// One row of a search, as the test writes it.
    fn row(label: &str, start: f32, end: f32, score: f32, note: bool) -> index::Row {
        index::Row {
            note,
            label: label.to_string(),
            start,
            end,
            score,
            margin: score - 0.5,
        }
    }

    /// A row below the cut takes the weak mark.
    #[test]
    fn a_row_below_the_cut_is_marked_weak() {
        let weak = hit_line(0, &row("a.mp4", 0.0, 0.25, 0.55, false), 0.70);
        assert!(weak.ends_with(" weak"), "{weak}");
        let strong = hit_line(0, &row("a.mp4", 0.0, 0.25, 0.80, false), 0.70);
        assert!(!strong.contains("weak"), "{strong}");
        assert!(strong.contains("[0.00-0.25 s]"), "{strong}");

        let note = hit_line(1, &row("blue", 0.0, 0.0, 0.50, true), 0.70);
        assert!(note.contains("note: blue"), "{note}");
        assert!(note.ends_with(" weak"), "{note}");
    }

    /// One line shows the range of the whole moment.
    #[test]
    fn the_line_shows_the_range_of_the_whole_moment() {
        let line = hit_line(0, &row("a.mp4", 0.0, 4.0, 0.9, false), 0.7);

        assert!(line.contains("[0.00-4.00 s]"), "got {line}");
    }

    /// A search of the window grid shows one row for a near pair.
    ///
    /// The cap holds ten rows before the fold, so the near pair joins one row
    /// of the ten.
    #[test]
    fn a_search_shows_one_row_for_a_near_pair() {
        let records = vec![
            videoembed::store::Record {
                rid: 0,
                id: "a.mp4/0-2000".to_string(),
                kind: videoembed::store::KIND_FRAME,
                label: "a.mp4".to_string(),
                start: 0.0,
                end: 2.0,
            },
            videoembed::store::Record {
                rid: 1,
                id: "a.mp4/2000-4000".to_string(),
                kind: videoembed::store::KIND_FRAME,
                label: "a.mp4".to_string(),
                start: 2.0,
                end: 4.0,
            },
        ];
        let ranked = vec![(0, 0.9), (1, 0.85)];

        let rows = rows_of(&records, &ranked, 0.0);

        assert_eq!(rows.len(), 1, "the pair joins the cap of ten");
        assert_eq!(rows[0].end, 4.0, "the moment runs to the last end");
    }

    /// The search takes a scope after the separator.
    #[test]
    fn the_search_takes_a_scope_after_the_separator() {
        let (query, scope) = parse_search("blue | spike/a.mp4");
        assert_eq!(query, "blue");
        assert_eq!(scope.as_deref(), Some("spike/a.mp4"));

        let (query, scope) = parse_search("  blue  ");
        assert_eq!(query, "blue");
        assert_eq!(scope, None);
    }

    /// The last separator wins, so a query can hold the `|` itself.
    #[test]
    fn the_last_separator_holds_the_scope() {
        let (query, scope) = parse_search("a | b.mp4 | spike/c.mp4");
        assert_eq!(query, "a | b.mp4");
        assert_eq!(scope.as_deref(), Some("spike/c.mp4"));
    }

    /// An empty scope is no scope.
    #[test]
    fn an_empty_scope_is_no_scope() {
        let (query, scope) = parse_search("blue |");
        assert_eq!(query, "blue");
        assert_eq!(scope, None);
    }

    /// The line names the scope, or the start step when there is no scope.
    #[test]
    fn an_empty_result_names_the_scope_or_the_start() {
        assert_eq!(
            no_result_line(Some("spike/a.mp4")),
            "No records for spike/a.mp4."
        );
        assert_eq!(
            no_result_line(None),
            "The store is empty. Run index or note first."
        );
    }

    /// One video makes one row, whatever separator its label carries.
    #[test]
    fn the_list_gathers_a_video_across_path_separators() {
        let rows = video_rows(&[
            Record {
                rid: 1,
                id: r"D:\video\a.mp4/0-250".into(),
                kind: KIND_FRAME,
                label: r"D:\video\a.mp4".into(),
                start: 0.0,
                end: 0.25,
            },
            Record {
                rid: 2,
                id: "D:/video/a.mp4/250-500".into(),
                kind: KIND_FRAME,
                label: "D:/video/a.mp4".into(),
                start: 0.25,
                end: 0.5,
            },
            Record {
                rid: 3,
                id: "blue/0-0".into(),
                kind: KIND_NOTE,
                label: "blue".into(),
                start: 0.0,
                end: 0.0,
            },
        ]);

        assert_eq!(rows, vec![("D:/video/a.mp4".to_string(), 2)]);
    }

    /// The index command names the plan it will use.
    #[test]
    fn the_index_line_names_the_plan() {
        let plan = videoembed::plan::Plan::of(videoembed::plan::Preset::Object);
        assert_eq!(
            index_line(&plan),
            "Indexing under the Object plan (2 s windows, 2 frames, with audio)."
        );

        let fine = videoembed::plan::Plan::of(videoembed::plan::Preset::Fine);
        assert_eq!(index_line(&fine), "Indexing under the Fine plan (0.25 s frames).");
    }

    /// The index line names the rewrite of a fragmented video.
    #[test]
    fn the_note_line_names_a_rewrite() {
        assert_eq!(
            note_line(Some("D:/video/a.mp4")),
            "Reading the progressive copy of D:/video/a.mp4."
        );
        assert_eq!(note_line(None), "");
    }

    /// A preset word picks that preset.
    #[test]
    fn the_plan_command_reads_a_preset() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/video/a.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();

        let plan = plan_command(&store, "D:/video/a.mp4 action").unwrap();

        assert_eq!(plan, videoembed::plan::Plan::of(videoembed::plan::Preset::Action));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A value word sets one plan value over the stored plan.
    #[test]
    fn the_plan_command_reads_values() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/video/a.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();

        let plan = plan_command(&store, "D:/video/a.mp4 window=6 audio=on").unwrap();

        assert_eq!(plan.window, 6.0);
        assert!(plan.audio);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A path that holds a space still separates from the plan words.
    #[test]
    fn the_plan_command_reads_a_path_with_a_space() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/my videos/a.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();

        let plan = plan_command(&store, "D:/my videos/a.mp4 action").unwrap();

        assert_eq!(
            plan,
            videoembed::plan::Plan::of(videoembed::plan::Preset::Action)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The split hands the whole path to the caller, spaces and all.
    #[test]
    fn the_plan_split_names_the_path() {
        let (path, words) = split_plan_args("D:/my videos/a.mp4 window=4").unwrap();
        assert_eq!(path, "D:/my videos/a.mp4");
        assert_eq!(words, "window=4");
    }

    /// A path the store does not hold names the fault.
    #[test]
    fn the_plan_command_refuses_an_unknown_video() {
        let (dir, store) = store_for_plan_test();
        let err = plan_command(&store, "D:/video/zz.mp4 object").unwrap_err();
        assert!(format!("{err}").contains("zz.mp4"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A plan change on a video whose file is gone ends the change.
    #[test]
    fn a_plan_change_needs_the_source_file() {
        let (dir, mut store) = store_for_plan_test();
        store
            .append(vec![videoembed::store::NewRecord {
                kind: videoembed::store::KIND_FRAME,
                label: "D:/video/missing.mp4".into(),
                start: 0.0,
                end: 0.25,
                vec: {
                    let mut vec = vec![0.0f32; videoembed::db::DIM];
                    vec[0] = 1.0;
                    vec
                },
            }])
            .unwrap();
        store
            .set_plan(
                "D:/video/missing.mp4",
                &videoembed::plan::Plan::of(videoembed::plan::Preset::Object),
            )
            .unwrap();

        let err = source_of("D:/video/missing.mp4").unwrap_err();

        assert!(format!("{err}").contains("missing.mp4"), "{err}");
        // The plan must stay as it was.
        let held = store.plan_of("D:/video/missing.mp4").unwrap().unwrap();
        assert_eq!(
            held,
            videoembed::plan::Plan::of(videoembed::plan::Preset::Object)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A store on a fresh file, for the plan tests.
    ///
    /// The counter gives each caller its own file, because the tests run at the
    /// same time in one process.
    fn store_for_plan_test() -> (PathBuf, Store) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("vei_plan_cmd_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = Store::open(&dir.join("index.db")).unwrap();
        (dir, store)
    }
}
