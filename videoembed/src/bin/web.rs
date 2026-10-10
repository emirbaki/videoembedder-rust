//! The web server. One thread serves HTTP. One thread owns the model.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::LlamaContext;
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use std::sync::atomic::{AtomicBool, Ordering};

use videoembed::embed::{Embedder, QUERY_PREFIX};
use videoembed::index::{self, Group, Progress};
use videoembed::media;
use videoembed::store::{Record, Store, normalize_label};

type Reply = Response<Box<dyn Read + Send>>;

const ADDRESS: &str = "127.0.0.1:8787";
/// The compute backend of this build. The page shows the name.
#[cfg(feature = "cuda")]
const BACKEND: &str = "CUDA";
#[cfg(not(feature = "cuda"))]
const BACKEND: &str = "CPU";
/// The most bytes one request body may hold.
const MAX_BODY: usize = 1024 * 1024;
/// The number of thumbnails kept in memory.
const THUMB_CAP: usize = 256;
/// The width of one thumbnail in pixels.
const THUMB_WIDTH: u32 = 320;
/// The largest thumbnail width a request may ask for.
const THUMB_MAX_WIDTH: u32 = 1024;
const PAGE: &str = include_str!("../../assets/index.html");
const JS: &str = include_str!("../../assets/app.js");
const CSS: &str = include_str!("../../assets/app.css");

/// One request from the server thread to the engine thread.
enum Job {
    State(Sender<State>),
    Videos(Sender<Vec<VideoRow>>),
    /// Stop the running index job. The answer says whether one ran.
    StopIndex { answer: Sender<bool> },
    Holds {
        path: String,
        answer: Sender<bool>,
    },
    Forget {
        path: String,
        answer: Sender<Result<usize, JobError>>,
    },
    StartIndex {
        folder: PathBuf,
        answer: Sender<Result<usize, JobError>>,
    },
    Search {
        query: String,
        answer: Sender<Result<SearchAnswer, JobError>>,
    },
}

/// The reason a job did not run. The server turns it into a status code.
enum JobError {
    Busy,
    Message(String),
}

impl From<anyhow::Error> for JobError {
    fn from(err: anyhow::Error) -> Self {
        Self::Message(format!("{err:#}"))
    }
}

#[derive(Serialize)]
struct State {
    version: String,
    records: usize,
    videos: usize,
    key: String,
}

#[derive(Serialize)]
struct VideoRow {
    path: String,
    items: usize,
}

#[derive(Serialize)]
struct Rows {
    videos: Vec<VideoRow>,
}

#[derive(Serialize)]
struct SearchAnswer {
    groups: Vec<Group>,
    baseline: f32,
}

#[derive(Debug, Deserialize)]
struct PathBody {
    path: String,
}

#[derive(Debug, Deserialize)]
struct QueryBody {
    query: String,
}

#[derive(Serialize)]
struct Picked {
    path: Option<String>,
}

#[derive(Serialize)]
struct Count {
    files: usize,
}

#[derive(Serialize)]
struct Removed {
    removed: usize,
}

#[derive(Serialize)]
struct Stopping {
    stopping: bool,
}

#[derive(Serialize)]
struct Failure {
    error: String,
}

/// Make one response header. The name and the value are ours, so they are valid.
fn head(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("the header is valid")
}

fn bytes_reply(code: u16, content_type: &str, body: Vec<u8>) -> Reply {
    let len = body.len();
    let data: Box<dyn Read + Send> = Box::new(std::io::Cursor::new(body));
    Response::new(
        StatusCode(code),
        vec![head("Content-Type", content_type)],
        data,
        Some(len),
        None,
    )
}

fn text_reply(code: u16, content_type: &str, body: &str) -> Reply {
    bytes_reply(code, content_type, body.as_bytes().to_vec())
}

fn json_with(code: u16, value: &impl Serialize) -> Reply {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    bytes_reply(code, "application/json", body)
}

fn json_reply(value: &impl Serialize) -> Reply {
    json_with(200, value)
}

fn error_reply(code: u16, message: &str) -> Reply {
    json_with(
        code,
        &Failure {
            error: message.to_string(),
        },
    )
}

/// True when the request carries the guard header. It blocks a foreign page.
fn guarded(request: &Request) -> bool {
    request
        .headers()
        .iter()
        .any(|header| header.field.equiv("X-VEI") && header.value.as_str() == "1")
}

/// Read one byte range from a Range header.
///
/// Returns an inclusive start and end. A missing or broken range gives `None`.
fn parse_range(header: &str, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let spec = header.strip_prefix("bytes=")?;
    let first = spec.split(',').next()?.trim();
    let (left, right) = first.split_once('-')?;

    if left.is_empty() {
        let asked: u64 = right.parse().ok()?;
        if asked == 0 {
            return None;
        }
        let count = asked.min(len);
        return Some((len - count, len - 1));
    }

    let start: u64 = left.parse().ok()?;
    if start >= len {
        return None;
    }
    let end = if right.is_empty() {
        len - 1
    } else {
        right.parse::<u64>().ok()?.min(len - 1)
    };
    if end < start {
        return None;
    }
    Some((start, end))
}

/// Split a URL into its decoded query parameters.
fn params(url: &str) -> HashMap<String, String> {
    let mut found = HashMap::new();
    if let Some((_, query)) = url.split_once('?') {
        for pair in query.split('&') {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            found.insert(decode(name), decode(value));
        }
    }
    found
}

fn decode(text: &str) -> String {
    percent_encoding::percent_decode_str(&text.replace('+', " "))
        .decode_utf8_lossy()
        .into_owned()
}

/// One key for one thumbnail. The time is rounded to the tenth of a second.
fn thumb_key(path: &str, at: f32, width: u32) -> String {
    format!("{path}|{:.1}|{width}", at)
}

/// Put one thumbnail in the cache. The oldest entry leaves when it is full.
fn cache_thumb(cache: &mut HashMap<String, Arc<Vec<u8>>>, key: String, bytes: Vec<u8>) {
    if cache.len() >= THUMB_CAP && !cache.contains_key(&key) {
        if let Some(first) = cache.keys().next().cloned() {
            cache.remove(&first);
        }
    }
    cache.insert(key, Arc::new(bytes));
}

fn mime_of(path: &Path) -> &'static str {
    let extension = path
        .extension()
        .and_then(|part| part.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        _ => "application/octet-stream",
    }
}

/// What one Range header asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangePlan {
    /// Send the whole file with a 200.
    Whole,
    /// Send the bytes from start to end, both included, with a 206.
    Part(u64, u64),
    /// Send a 416 and no body.
    Refused,
}

fn range_plan(range_header: Option<&str>, len: u64) -> RangePlan {
    match range_header {
        None => RangePlan::Whole,
        Some(header) => match parse_range(header, len) {
            Some((start, end)) => RangePlan::Part(start, end),
            None => RangePlan::Refused,
        },
    }
}

/// Build a reply for one video file. A Range header gives a 206 and a slice.
fn video_reply(path: &Path, range_header: Option<&str>) -> Result<Reply> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let mime = mime_of(path);

    match range_plan(range_header, len) {
        RangePlan::Whole => {
            let data: Box<dyn Read + Send> = Box::new(file);
            Ok(Response::new(
                StatusCode(200),
                vec![head("Content-Type", mime), head("Accept-Ranges", "bytes")],
                data,
                Some(len as usize),
                None,
            ))
        }
        RangePlan::Refused => {
            let data: Box<dyn Read + Send> = Box::new(std::io::empty());
            Ok(Response::new(
                StatusCode(416),
                vec![head("Content-Range", &format!("bytes */{len}"))],
                data,
                Some(0),
                None,
            ))
        }
        RangePlan::Part(start, end) => {
            file.seek(SeekFrom::Start(start))?;
            let count = end - start + 1;
            let data: Box<dyn Read + Send> = Box::new(file.take(count));
            Ok(Response::new(
                StatusCode(206),
                vec![
                    head("Content-Type", mime),
                    head("Accept-Ranges", "bytes"),
                    head("Content-Range", &format!("bytes {start}-{end}/{len}")),
                ],
                data,
                Some(count as usize),
                None,
            ))
        }
    }
}

/// Read one request body. The read stops one byte past the limit.
fn read_body<T: for<'de> Deserialize<'de>>(request: &mut Request) -> Result<T, (u16, String)> {
    let mut raw = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|err| (400, format!("read the body: {err}")))?;
    parse_body(&raw)
}

/// Parse one request body. A body over the limit gives a 413.
///
/// The limit stops one local process from filling the memory of the server.
fn parse_body<T: for<'de> Deserialize<'de>>(raw: &[u8]) -> Result<T, (u16, String)> {
    if raw.len() > MAX_BODY {
        return Err((413, format!("the body is larger than {MAX_BODY} bytes")));
    }
    serde_json::from_slice(raw).map_err(|err| (400, format!("the body is not valid JSON: {err}")))
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

fn make_key() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    format!("{:x}{:x}", now ^ (pid << 64), pid.rotate_left(17))
}

/// The indexing work of the server. One `step` call does one media item.
struct WebJob {
    files: Vec<PathBuf>,
    file_at: usize,
    run: Option<index::IndexRun>,
    records: Vec<Record>,
    indexed: usize,
    started: std::time::Instant,
    label: String,
}

/// Group the store records by video path. Every video kind counts.
fn videos(store: &Store) -> Vec<VideoRow> {
    let mut rows: Vec<VideoRow> = Vec::new();
    for record in &store.records {
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        match rows.iter_mut().find(|row| row.path == record.label) {
            Some(row) => row.items += 1,
            None => rows.push(VideoRow {
                path: record.label.clone(),
                items: 1,
            }),
        }
    }
    rows
}

/// Answer jobs forever. A running index does one media item per turn.
fn engine_loop(
    embedder: Embedder,
    mut store: Store,
    progress: Arc<Mutex<Progress>>,
    key: String,
    jobs: Receiver<Job>,
) {
    let mut ctx = match embedder.new_context() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("no context: {err:#}");
            return;
        }
    };
    let mut pending: Option<WebJob> = None;
    let stopped = Arc::new(AtomicBool::new(false));
    loop {
        if pending.is_some() {
            let done = step_job(
                &embedder,
                &mut ctx,
                &mut store,
                &progress,
                &stopped,
                pending.as_mut().unwrap(),
            );
            if done {
                pending = None;
            }
            while let Ok(job) = jobs.try_recv() {
                handle(job, &embedder, &mut store, &progress, &key, &stopped, &mut pending);
            }
        } else {
            match jobs.recv() {
                Ok(job) => handle(job, &embedder, &mut store, &progress, &key, &stopped, &mut pending),
                Err(_) => break,
            }
        }
    }
}

fn handle(
    job: Job,
    embedder: &Embedder,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
    stopped: &Arc<AtomicBool>,
    pending: &mut Option<WebJob>,
) {
    match job {
        Job::State(answer) => {
            let count = videos(store).len();
            let _ = answer.send(State {
                version: env!("CARGO_PKG_VERSION").to_string(),
                records: store.records.len(),
                videos: count,
                key: key.to_string(),
            });
        }
        Job::Videos(answer) => {
            let _ = answer.send(videos(store));
        }
        Job::Holds { path, answer } => {
            let _ = answer.send(store.holds_video(&path));
        }
        Job::Forget { path, answer } => {
            let outcome = store.forget(&path).map_err(JobError::from);
            let _ = answer.send(outcome);
        }
        Job::Search { query, answer } => {
            let _ = answer.send(search(embedder, store, &query));
        }
        Job::StartIndex { folder, answer } => {
            let _ = answer.send(start_index(progress, pending, stopped, &folder));
        }
        Job::StopIndex { answer } => {
            let running = pending.is_some();
            stopped.store(true, Ordering::Relaxed);
            let _ = answer.send(running);
        }
    }
}

fn search(embedder: &Embedder, store: &Store, query: &str) -> Result<SearchAnswer, JobError> {
    if store.records.is_empty() {
        return Ok(SearchAnswer {
            groups: Vec::new(),
            baseline: 0.0,
        });
    }
    let text = format!("{QUERY_PREFIX}{query}");
    let vec = embedder.embed_text(&text).map_err(JobError::from)?;
    let ranked = store.rank(&vec);
    let baseline = index::median(&ranked);
    Ok(SearchAnswer {
        groups: index::group_hits(&store.records, &ranked, baseline),
        baseline,
    })
}

fn start_index(
    progress: &Arc<Mutex<Progress>>,
    pending: &mut Option<WebJob>,
    flag: &Arc<AtomicBool>,
    folder: &Path,
) -> Result<usize, JobError> {
    if pending.is_some() {
        return Err(JobError::Busy);
    }
    if !folder.is_dir() {
        return Err(JobError::Message(format!(
            "{} is not a folder",
            folder.display()
        )));
    }
    let files = index::list_videos(folder).map_err(JobError::from)?;
    if files.is_empty() {
        return Err(JobError::Message(format!(
            "no video files under {}",
            folder.display()
        )));
    }
    let count = files.len();
    flag.store(false, Ordering::Relaxed);
    {
        let mut held = progress.lock().unwrap();
        // The model state belongs to the server, not to this job. Hold it,
        // because the page shows it after the job ends.
        let device = held.device.clone();
        let gpu_available = held.gpu_available;
        let load_ms = held.load_ms;
        *held = Progress {
            running: true,
            folder: folder.display().to_string(),
            file_count: count,
            device,
            gpu_available,
            load_ms,
            ..Progress::default()
        };
    }
    *pending = Some(WebJob {
        files,
        file_at: 0,
        run: None,
        records: Vec::new(),
        indexed: 0,
        started: std::time::Instant::now(),
        label: String::new(),
    });
    Ok(count)
}

/// True when one finished run left a whole index for its video.
///
/// A finished video takes the finished mark, and the mark keeps it out of
/// every later run. A run that failed part way holds a partial answer, so it
/// takes no mark and the next run indexes the video again.
fn run_left_a_whole_index(outcome: &Result<Option<Record>>) -> bool {
    matches!(outcome, Ok(None))
}

/// One media item of work. True means the whole job has finished or stopped.
fn step_job(
    embedder: &Embedder,
    ctx: &mut LlamaContext<'_>,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    flag: &Arc<AtomicBool>,
    job: &mut WebJob,
) -> bool {
    if flag.load(Ordering::Relaxed) {
        // Drop the partial work of this video. A half-indexed video reads as a
        // whole one on the next search.
        job.run = None;
        job.records.clear();
        progress.lock().unwrap().running = false;
        return true;
    }
    if job.run.is_none() {
        if job.file_at >= job.files.len() {
            progress.lock().unwrap().running = false;
            return true;
        }
        let video = job.files[job.file_at].clone();
        match index::IndexRun::start(&video) {
            Ok(run) => {
                let label = normalize_label(&video.to_string_lossy());
                if store.is_done(&label) {
                    job.file_at += 1;
                    return false;
                }
                if let Err(err) = store.forget(&label) {
                    fail(progress, &err);
                }
                begin_file(progress, label.clone(), job.file_at + 1, run.total());
                job.label = label;
                job.records.clear();
                job.run = Some(run);
            }
            Err(err) => {
                fail(progress, &err);
                job.file_at += 1;
            }
        }
        return false;
    }

    let outcome = job.run.as_mut().unwrap().step(embedder, ctx);
    let whole = run_left_a_whole_index(&outcome);
    let mut finished = false;
    match outcome {
        Ok(Some(record)) => job.records.push(record),
        Ok(None) => finished = true,
        Err(err) => {
            fail(progress, &err);
            finished = true;
        }
    }

    if finished {
        let records = std::mem::take(&mut job.records);
        if whole {
            job.indexed += records.len();
            if let Err(err) = store.append(records) {
                fail(progress, &err);
            }
            if let Err(err) = store.mark_done(&job.label) {
                fail(progress, &err);
            }
        } else {
            // The run failed part way. The records of the partial answer go,
            // as they do when the user stops a job, and the video takes no
            // finished mark. The next run then indexes the whole video again.
        }
        job.run = None;
        job.file_at += 1;
        if job.file_at >= job.files.len() {
            progress.lock().unwrap().running = false;
            return true;
        }
    } else {
        let done = job.indexed + job.records.len();
        let seconds = job.started.elapsed().as_secs_f64();
        let mut held = progress.lock().unwrap();
        held.window = job.run.as_ref().map(|r| r.done()).unwrap_or(0);
        held.indexed = done;
        held.elapsed_ms = (seconds * 1000.0) as u64;
        held.per_second = index::per_second(done, seconds);
    }
    false
}

/// Point the progress at a new file.
///
/// The call does not touch the error field. A failure of an earlier file
/// must reach the page, and the page reads the error field.
fn begin_file(progress: &Arc<Mutex<Progress>>, label: String, number: usize, total: usize) {
    let mut held = progress.lock().unwrap();
    held.file = label;
    held.file_index = number;
    held.window = 0;
    held.window_count = total;
}

fn fail(progress: &Arc<Mutex<Progress>>, err: &anyhow::Error) {
    progress.lock().unwrap().error = Some(format!("{err:#}"));
}

fn handle_request(
    request: Request,
    jobs: &Sender<Job>,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
    thumbs: &Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>>,
) -> Result<()> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let post = request.method() == &Method::Post;

    // A thumbnail cuts a frame on this thread, so a search and an index keep
    // running. The route never waits on the engine loop.
    if path == "/api/frame" && request.method() == &Method::Get {
        return serve_frame(request, &url, key, jobs, thumbs);
    }

    match path.as_str() {
        "/" => request.respond(text_reply(200, "text/html; charset=utf-8", PAGE))?,
        "/app.js" => request.respond(text_reply(200, "text/javascript; charset=utf-8", JS))?,
        "/app.css" => request.respond(text_reply(200, "text/css; charset=utf-8", CSS))?,
        "/favicon.ico" => request.respond(bytes_reply(204, "image/x-icon", Vec::new()))?,
        _ if !path.starts_with("/api/") => {
            request.respond(error_reply(404, "no such page"))?;
        }
        "/api/video" => {
            let found = params(&url);
            if found.get("key").map(String::as_str) != Some(key) {
                request.respond(error_reply(403, "the key is missing"))?;
                return Ok(());
            }
            serve_video(request, found.get("path").cloned(), jobs)?;
        }
        _ => {
            if !guarded(&request) {
                request.respond(error_reply(403, "the X-VEI header is missing"))?;
                return Ok(());
            }
            if path == "/api/progress" {
                let snapshot = progress.lock().unwrap().clone();
                request.respond(json_reply(&snapshot))?;
                return Ok(());
            }
            api(request, &path, post, jobs)?;
        }
    }
    Ok(())
}

fn serve_video(request: Request, path: Option<String>, jobs: &Sender<Job>) -> Result<()> {
    let Some(path) = path else {
        request.respond(error_reply(400, "the path is missing"))?;
        return Ok(());
    };
    let (answer, wait) = channel();
    let _ = jobs.send(Job::Holds {
        path: path.clone(),
        answer,
    });
    if !matches!(wait.recv(), Ok(true)) {
        request.respond(error_reply(403, "that video is not in the index"))?;
        return Ok(());
    }
    let range = request
        .headers()
        .iter()
        .find(|header| header.field.equiv("Range"))
        .map(|header| header.value.as_str().to_string());
    match video_reply(Path::new(&path), range.as_deref()) {
        Ok(reply) => request.respond(reply)?,
        Err(err) => {
            let message = format!("{err:#}");
            request.respond(error_reply(500, &message))?;
        }
    }
    Ok(())
}

/// One thumbnail. The index check runs on the engine. The cut runs here.
fn serve_frame(
    request: Request,
    url: &str,
    http_key: &str,
    jobs: &Sender<Job>,
    thumbs: &Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>>,
) -> Result<()> {
    // `params` reads the whole URL, so a query alone gives no pair.
    let args = params(url);
    if args.get("key").map(String::as_str) != Some(http_key) {
        request.respond(error_reply(403, "the key is missing"))?;
        return Ok(());
    }
    let path = args.get("path").cloned().unwrap_or_default();
    if path.is_empty() {
        request.respond(error_reply(400, "the path is missing"))?;
        return Ok(());
    }
    let at = args
        .get("t")
        .and_then(|text| text.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0);
    let Some(at) = at else {
        request.respond(error_reply(400, "the time is not a number"))?;
        return Ok(());
    };
    let width = args
        .get("w")
        .and_then(|text| text.parse::<u32>().ok())
        .filter(|value| *value > 0 && *value <= THUMB_MAX_WIDTH)
        .unwrap_or(THUMB_WIDTH);

    let key = thumb_key(&path, at, width);
    if let Some(bytes) = thumbs.lock().unwrap().get(&key).cloned() {
        request.respond(bytes_reply(200, "image/jpeg", bytes.as_ref().clone()))?;
        return Ok(());
    }

    let (answer, wait) = channel();
    let _ = jobs.send(Job::Holds {
        path: path.clone(),
        answer,
    });
    if !matches!(wait.recv(), Ok(true)) {
        request.respond(error_reply(403, "that video is not in the index"))?;
        return Ok(());
    }

    match media::thumb_jpeg(Path::new(&path), at as f64, width) {
        Ok(bytes) => {
            cache_thumb(&mut thumbs.lock().unwrap(), key, bytes.clone());
            request.respond(bytes_reply(200, "image/jpeg", bytes))?;
        }
        Err(err) => {
            request.respond(error_reply(400, &format!("{err:#}")))?;
        }
    }
    Ok(())
}

fn api(mut request: Request, path: &str, post: bool, jobs: &Sender<Job>) -> Result<()> {
    match (post, path) {
        (false, "/api/state") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::State(answer));
            match wait.recv() {
                Ok(state) => request.respond(json_reply(&state))?,
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (false, "/api/videos") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::Videos(answer));
            match wait.recv() {
                Ok(rows) => request.respond(json_reply(&Rows { videos: rows }))?,
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/forget") => {
            let body: PathBody = match read_body(&mut request) {
                Ok(body) => body,
                Err((code, message)) => {
                    request.respond(error_reply(code, &message))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::Forget {
                path: body.path,
                answer,
            });
            match wait.recv() {
                Ok(Ok(removed)) => request.respond(json_reply(&Removed { removed }))?,
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(400, &message))?
                }
                _ => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/index") => {
            let body: PathBody = match read_body(&mut request) {
                Ok(body) => body,
                Err((code, message)) => {
                    request.respond(error_reply(code, &message))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::StartIndex {
                folder: PathBuf::from(body.path),
                answer,
            });
            match wait.recv() {
                Ok(Ok(files)) => request.respond(json_reply(&Count { files }))?,
                Ok(Err(JobError::Busy)) => {
                    request.respond(error_reply(409, "an index job already runs"))?
                }
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(400, &message))?
                }
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/search") => {
            let body: QueryBody = match read_body(&mut request) {
                Ok(body) => body,
                Err((code, message)) => {
                    request.respond(error_reply(code, &message))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::Search {
                query: body.query,
                answer,
            });
            match wait.recv() {
                Ok(Ok(found)) => request.respond(json_reply(&found))?,
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(500, &message))?
                }
                _ => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/pick-folder") => {
            // The dialog blocks this thread alone. Each request gets its own thread.
            let picked = rfd::FileDialog::new().pick_folder();
            let answer = Picked {
                path: picked.map(|p| p.to_string_lossy().to_string()),
            };
            request.respond(json_reply(&answer))?;
        }
        (true, "/api/stop") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::StopIndex { answer });
            match wait.recv() {
                Ok(running) => request.respond(json_reply(&Stopping { stopping: running }))?,
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
            }
        }
        (true, "/api/quit") => {
            let _ = request.respond(json_reply(&Stopping { stopping: true }));
            std::thread::sleep(Duration::from_millis(200));
            std::process::exit(0);
        }
        _ => request.respond(error_reply(405, "that method is not allowed here"))?,
    }
    Ok(())
}

fn main() -> Result<()> {
    let models = models_dir();
    let store_path = PathBuf::from("store/index.vei");
    let progress = Arc::new(Mutex::new(Progress::default()));
    let thumbs = Arc::new(Mutex::new(HashMap::new()));
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    let key = make_key();

    let engine_progress = Arc::clone(&progress);
    let engine_key = key.clone();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let embedder = match Embedder::load(&models, None) {
            Ok(embedder) => embedder,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        let load_ms = started.elapsed().as_millis() as u64;
        {
            let mut held = engine_progress.lock().unwrap();
            held.device = embedder.device().as_str().to_string();
            held.gpu_available = embedder.supports_gpu();
            held.load_ms = load_ms;
        }
        println!("Backend: {} on {}", BACKEND, embedder.device().as_str());
        let store = match Store::open(&store_path) {
            Ok(store) => store,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        let _ = ready_tx.send(Ok(()));
        engine_loop(embedder, store, engine_progress, engine_key, jobs_rx);
    });

    match ready_rx.recv().context("the engine thread stopped early")? {
        Ok(()) => {}
        Err(message) => bail!("the engine failed to start: {message}"),
    }

    let server = Server::http(ADDRESS).map_err(|err| anyhow!("{err}"))?;
    println!("videoembed is at http://{ADDRESS}/");
    println!("Stop the server with the Quit button or with Ctrl C.");
    let _ = open::that(format!("http://{ADDRESS}/"));

    for request in server.incoming_requests() {
        let jobs = jobs_tx.clone();
        let progress = Arc::clone(&progress);
        let key = key.clone();
        let thumbs = Arc::clone(&thumbs);
        std::thread::spawn(move || {
            if let Err(err) = handle_request(request, &jobs, &progress, &key, &thumbs) {
                eprintln!("A request failed: {err:#}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failure of one file must survive the start of the next file.
    ///
    /// The page reads the error field for the whole run. A new file that
    /// clears the field hides the failure of the file before it.
    #[test]
    fn a_new_file_keeps_the_error_of_an_earlier_file() {
        let progress = Arc::new(Mutex::new(Progress::default()));
        progress.lock().unwrap().error = Some("clip two failed".to_string());

        begin_file(&progress, "clip-three.mp4".to_string(), 3, 12);

        let held = progress.lock().unwrap();
        assert_eq!(held.error.as_deref(), Some("clip two failed"));
        assert_eq!(held.file, "clip-three.mp4");
        assert_eq!(held.file_index, 3);
        assert_eq!(held.window, 0);
        assert_eq!(held.window_count, 12);
    }

    #[test]
    fn a_plain_range_asks_for_the_first_hundred_bytes() {
        assert_eq!(parse_range("bytes=0-99", 1000), Some((0, 99)));
    }

    #[test]
    fn an_open_range_runs_to_the_end() {
        assert_eq!(parse_range("bytes=100-", 1000), Some((100, 999)));
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        assert_eq!(parse_range("bytes=2000-", 1000), None);
        assert_eq!(parse_range("bytes=900-2000", 1000), Some((900, 999)));
    }

    #[test]
    fn a_broken_range_header_is_refused() {
        assert_eq!(parse_range("items=0-99", 1000), None);
        assert_eq!(parse_range("bytes=abc", 1000), None);
        assert_eq!(parse_range("bytes=50-10", 1000), None);
        assert_eq!(parse_range("bytes=", 1000), None);
    }

    #[test]
    fn no_range_header_asks_for_the_whole_file() {
        assert_eq!(range_plan(None, 1000), RangePlan::Whole);
    }

    #[test]
    fn a_range_header_asks_for_a_slice() {
        assert_eq!(range_plan(Some("bytes=0-99"), 1000), RangePlan::Part(0, 99));
        assert_eq!(range_plan(Some("bytes=500-"), 1000), RangePlan::Part(500, 999));
    }

    #[test]
    fn a_range_starting_past_the_end_is_refused_without_a_panic() {
        assert_eq!(range_plan(Some("bytes=2000-"), 1000), RangePlan::Refused);
        assert_eq!(range_plan(Some("bytes=-"), 1000), RangePlan::Refused);
        assert_eq!(range_plan(Some("bytes=0-"), 0), RangePlan::Refused);
    }

    #[test]
    fn a_suffix_range_reads_from_the_end() {
        assert_eq!(range_plan(Some("bytes=-100"), 1000), RangePlan::Part(900, 999));
        assert_eq!(range_plan(Some("bytes=-5000"), 1000), RangePlan::Part(0, 999));
    }

    #[test]
    fn a_query_string_decodes_a_space_and_a_unicode_letter() {
        let found = params(
            "/api/video?path=D%3A%2Fvideo%2Fdo%C4%9Fum%20g%C3%BCn%C3%BC.mp4&key=abc",
        );
        assert_eq!(found.get("path").unwrap(), "D:/video/doğum günü.mp4");
        assert_eq!(found.get("key").unwrap(), "abc");
    }

    #[test]
    fn parse_body_reads_a_valid_body() {
        let body: PathBody = parse_body(br#"{"path":"D:/a.mp4"}"#).unwrap();
        assert_eq!(body.path, "D:/a.mp4");
    }

    #[test]
    fn parse_body_rejects_json_that_is_not_valid() {
        let err = parse_body::<PathBody>(b"not json").unwrap_err();
        assert_eq!(err.0, 400);
        assert!(err.1.contains("not valid JSON"), "{}", err.1);
    }

    #[test]
    fn parse_body_refuses_a_body_over_one_mib() {
        let raw = vec![b' '; MAX_BODY + 1];
        let err = parse_body::<PathBody>(&raw).unwrap_err();
        assert_eq!(err.0, 413);
    }

    #[test]
    fn the_mime_type_follows_the_extension() {
        assert_eq!(mime_of(Path::new("a.mp4")), "video/mp4");
        assert_eq!(mime_of(Path::new("a.MKV")), "video/x-matroska");
        assert_eq!(mime_of(Path::new("a.txt")), "application/octet-stream");
    }

    #[test]
    fn a_video_row_counts_frames_and_audio_together() {
        let dir = std::env::temp_dir().join(format!("vei_web_rows_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.vei")).unwrap();
        store
            .append(vec![
                Record {
                    kind: videoembed::store::KIND_FRAME,
                    label: "a.mp4".into(),
                    start: 0.0,
                    end: 0.25,
                    vec: vec![1.0, 0.0],
                },
                Record {
                    kind: videoembed::store::KIND_AUDIO,
                    label: "a.mp4".into(),
                    start: 0.0,
                    end: 1.0,
                    vec: vec![1.0, 0.0],
                },
                Record {
                    kind: videoembed::store::KIND_NOTE,
                    label: "a note".into(),
                    start: 0.0,
                    end: 0.0,
                    vec: vec![1.0, 0.0],
                },
            ])
            .unwrap();

        let rows = videos(&store);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].path, "a.mp4");
        assert_eq!(rows[0].items, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two hits at the same time share one cache entry, and a width splits it.
    #[test]
    fn the_thumb_key_covers_the_path_the_time_and_the_width() {
        assert_eq!(thumb_key("a.mp4", 1.0, 320), thumb_key("a.mp4", 1.0, 320));
        assert_ne!(thumb_key("a.mp4", 1.0, 320), thumb_key("a.mp4", 1.5, 320));
        assert_ne!(thumb_key("a.mp4", 1.0, 320), thumb_key("a.mp4", 1.0, 640));
        assert_ne!(thumb_key("a.mp4", 1.0, 320), thumb_key("b.mp4", 1.0, 320));
    }

    /// Only a run that reached its end may mark its video as finished.
    ///
    /// The mark keeps a video out of every later run, so a run that failed
    /// part way must leave no mark. The video then indexes again.
    #[test]
    fn only_a_run_that_reached_its_end_is_whole() {
        let more_work = Ok(Some(Record {
            kind: videoembed::store::KIND_FRAME,
            label: "a.mp4".to_string(),
            start: 0.0,
            end: 0.25,
            vec: vec![1.0, 0.0],
        }));
        assert!(!run_left_a_whole_index(&more_work), "the run is not finished");
        assert!(run_left_a_whole_index(&Ok(None)), "the run reached its end");
        assert!(
            !run_left_a_whole_index(&Err(anyhow!("a broken frame"))),
            "a failed run holds a partial answer"
        );
    }
}
