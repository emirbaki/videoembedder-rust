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

use videoembed::embed::{Device, Embedder, QUERY_PREFIX};
use videoembed::index::{self, Group, Progress};
use videoembed::media;
use videoembed::store::{NewRecord, Store, normalize_label};

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
    /// Hold or release the running index job. The answer names the new state.
    SetPause { answer: Sender<Result<bool, JobError>> },
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
        path: Option<String>,
        answer: Sender<Result<SearchAnswer, JobError>>,
    },
    /// Load the model again on another device. The answer names the new mode.
    SetDevice {
        device: Device,
        answer: Sender<Result<String, JobError>>,
    },
    /// Cut one video again under a new plan. The answer holds the record count.
    SetPlan {
        path: String,
        plan: videoembed::plan::Plan,
        answer: Sender<Result<usize, JobError>>,
    },
}

/// The reason a job did not run. The server turns it into a status code.
#[derive(Debug)]
enum JobError {
    Busy,
    Message(String),
}

impl From<anyhow::Error> for JobError {
    fn from(err: anyhow::Error) -> Self {
        Self::Message(format!("{err:#}"))
    }
}

/// The reply for a failed device switch, as a code and a line.
///
/// A switch fails for one reason: the model will not load on that device. That
/// fault belongs to the server, so the code is 500. A busy engine is the one
/// exception, and it answers 409.
fn switch_reply(err: JobError) -> (u16, String) {
    match err {
        JobError::Busy => (409, "the engine runs another job".to_string()),
        JobError::Message(message) => (500, message),
    }
}

#[derive(Serialize)]
struct State {
    version: String,
    records: usize,
    videos: usize,
    key: String,
}

/// One video, its record count, and its plan.
#[derive(Serialize)]
struct VideoRow {
    path: String,
    items: usize,
    plan: String,
    window: f64,
    frames: usize,
    audio: bool,
}

#[derive(Serialize)]
struct Rows {
    videos: Vec<VideoRow>,
}

#[derive(Serialize)]
struct SearchAnswer {
    groups: Vec<Group>,
    baseline: f32,
    /// The score below which a group reads as weak.
    cut: f32,
}

#[derive(Debug, Deserialize)]
struct PathBody {
    path: String,
}

#[derive(Debug, Deserialize)]
struct QueryBody {
    query: String,
    #[serde(default)]
    path: Option<String>,
}

/// The mode a switch asks for, as "gpu" or "cpu".
#[derive(Debug, Deserialize)]
struct DeviceBody {
    device: String,
}

/// The plan the page asks for. Either a preset name, or a set of values.
#[derive(Debug, Deserialize)]
struct PlanBody {
    path: String,
    #[serde(default)]
    preset: Option<String>,
    #[serde(default)]
    window: Option<f64>,
    #[serde(default)]
    frames: Option<usize>,
    #[serde(default)]
    audio: Option<bool>,
}

impl PlanBody {
    /// The plan this body asks for.
    fn plan(&self) -> Result<videoembed::plan::Plan, JobError> {
        if let Some(preset) = &self.preset {
            let preset = videoembed::plan::Preset::parse(preset)
                .ok_or_else(|| JobError::Message(format!("{preset} is not a preset")))?;
            return Ok(videoembed::plan::Plan::of(preset));
        }
        let mut words = Vec::new();
        if let Some(window) = self.window {
            words.push(format!("window={window}"));
        }
        if let Some(frames) = self.frames {
            words.push(format!("frames={frames}"));
        }
        if let Some(audio) = self.audio {
            words.push(format!("audio={}", if audio { "on" } else { "off" }));
        }
        if words.is_empty() {
            return Err(JobError::Message("no plan value arrived".into()));
        }
        videoembed::plan::Plan::of(videoembed::plan::Preset::Object)
            .set_words(&words.join(" "))
            .map_err(JobError::from)
    }
}

/// The mode the model now runs on.
#[derive(Serialize)]
struct DeviceName {
    device: String,
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
struct Paused {
    paused: bool,
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

/// The time of one still, with the black-frame probe.
///
/// The probe runs one time for each video in the life of the process, because
/// it costs one ffmpeg call for each offset it tries. A request past the probe
/// window keeps its own time and fills no map.
fn pick_thumb_time(chosen: &mut HashMap<String, f64>, video: &Path, at: f64) -> f64 {
    let last = media::PROBE_OFFSETS[media::PROBE_OFFSETS.len() - 1];
    if at > last {
        return at;
    }
    let key = video.to_string_lossy().to_string();
    if let Some(picked) = chosen.get(&key) {
        // A later request inside the window still reads the picked still, so
        // the page shows one picture of the opening.
        return picked.max(at);
    }
    match media::probe_offset(video, at) {
        Ok(picked) => {
            chosen.insert(key, picked);
            picked
        }
        Err(_) => at,
    }
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
    records: Vec<NewRecord>,
    indexed: usize,
    started: std::time::Instant,
    label: String,
    /// The plan of the current file. The store takes it when the file ends.
    plan: Option<videoembed::plan::Plan>,
}

/// Group the store records by video path. Every video kind counts.
///
/// The reply holds the plan of each video. A video with no plan row takes the
/// default plan.
fn videos(store: &Store) -> Vec<VideoRow> {
    let plans: HashMap<String, videoembed::plan::Plan> = store
        .plan_rows()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let mut rows: Vec<VideoRow> = Vec::new();
    for record in &store.records {
        if !videoembed::store::is_video(record.kind) {
            continue;
        }
        let name = normalize_label(&record.label);
        if let Some(row) = rows.iter_mut().find(|row| row.path == name) {
            row.items += 1;
            continue;
        }
        let plan = plans
            .get(&name)
            .copied()
            .unwrap_or_else(videoembed::plan::Plan::default_plan);
        rows.push(VideoRow {
            path: name,
            items: 1,
            plan: preset_name(&plan),
            window: plan.window,
            frames: plan.frames,
            audio: plan.audio,
        });
    }
    rows
}

/// The preset name of a plan, by its values. An empty name means no preset.
///
/// The page sends the name back to the server. A plan that matches no preset
/// must name no preset, or the page would replace a custom window and frame
/// count with that preset.
fn preset_name(plan: &videoembed::plan::Plan) -> String {
    videoembed::plan::Preset::ALL
        .into_iter()
        .find(|preset| videoembed::plan::Plan::of(*preset) == *plan)
        .map(|preset| preset.as_str().to_string())
        .unwrap_or_default()
}

/// Answer jobs forever. A running index does one media item per turn.
fn engine_loop(
    mut embedder: Embedder,
    models: PathBuf,
    mut store: Store,
    progress: Arc<Mutex<Progress>>,
    key: String,
    jobs: Receiver<Job>,
    paused: Arc<AtomicBool>,
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
        // A paused job waits on the channel, so the loop spins on no item.
        let request = if pending.is_some() && !paused.load(Ordering::Relaxed) {
            let done = step_job(
                &embedder,
                &mut ctx,
                &mut store,
                &progress,
                &stopped,
                &paused,
                pending.as_mut().unwrap(),
            );
            if done {
                pending = None;
            }
            let mut held = None;
            while let Ok(job) = jobs.try_recv() {
                if let Some(next) = handle(job, &embedder, &mut store, &progress, &key, &stopped, &paused, &mut pending) {
                    match held {
                        // One load at a time. The loop holds one request.
                        Some(_) => next.refuse(),
                        None => held = Some(next),
                    }
                }
            }
            held
        } else {
            match jobs.recv() {
                Ok(job) => handle(job, &embedder, &mut store, &progress, &key, &stopped, &paused, &mut pending),
                Err(_) => break,
            }
        };

        if let Some(Switch::SetDevice { device, answer }) = request {
            let started = std::time::Instant::now();
            // The new model loads before the old one leaves, so a failure
            // keeps the server on the device it had.
            match embedder.prepare(&models, device) {
                Ok((device, model, mtmd)) => {
                    // The new model must hold a context before the old model
                    // leaves. A device with no room for the cache then never
                    // takes over, and the old device keeps serving. The probe
                    // runs first, because a swap with no context after it
                    // leaves the server with no answer for any request.
                    if let Err(err) = embedder.probe(&model) {
                        let _ = answer.send(Err(JobError::from(err)));
                        continue;
                    }
                    // The context borrows the old model, so it goes first.
                    drop(ctx);
                    embedder.swap(device, model, mtmd);
                    ctx = match embedder.new_context() {
                        Ok(next) => next,
                        Err(err) => {
                            // The probe passed, so this is a fault of the
                            // runtime, and no device serves now.
                            fail(&progress, &err);
                            let _ = answer.send(Err(JobError::from(err)));
                            break;
                        }
                    };
                    let load_ms = started.elapsed().as_millis() as u64;
                    {
                        let mut held = progress.lock().unwrap();
                        held.device = device.as_str().to_string();
                        held.gpu_available = embedder.supports_gpu();
                        held.load_ms = load_ms;
                    }
                    let _ = answer.send(Ok(device.as_str().to_string()));
                }
                Err(err) => {
                    let _ = answer.send(Err(JobError::from(err)));
                }
            }
        }
    }
}

/// A change that the engine loop must make itself.
///
/// A job handler sees the store and the model by shared reference. The loop
/// owns the model and the working context, so a device switch lands there.
enum Switch {
    /// Load the model again on another device. The answer names the new mode.
    SetDevice {
        device: Device,
        answer: Sender<Result<String, JobError>>,
    },
}

impl Switch {
    /// Refuse this request, because the loop already holds one.
    fn refuse(self) {
        match self {
            Self::SetDevice { answer, .. } => {
                let _ = answer.send(Err(JobError::Busy));
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
    paused: &Arc<AtomicBool>,
    pending: &mut Option<WebJob>,
) -> Option<Switch> {
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
        Job::Search { query, path, answer } => {
            let _ = answer.send(search(embedder, store, &query, path.as_deref()));
        }
        Job::StartIndex { folder, answer } => {
            let _ = answer.send(start_index(
                store, progress, pending, stopped, paused, &folder,
            ));
        }
        Job::StopIndex { answer } => {
            let running = pending.is_some();
            // A stop ends a pause too, because a paused job never reads the
            // stop flag.
            paused.store(false, Ordering::Relaxed);
            progress.lock().unwrap().paused = false;
            stopped.store(true, Ordering::Relaxed);
            let _ = answer.send(running);
        }
        Job::SetPause { answer } => {
            let _ = answer.send(toggle_pause(pending.is_some(), paused, progress));
        }
        Job::SetPlan { path, plan, answer } => {
            let media = store.media_dir();
            let _ = answer.send(start_plan(
                progress, pending, stopped, paused, &path, plan, &media,
            ));
        }
        Job::SetDevice { device, answer } => {
            // The loop owns the model, so the load happens there.
            match refuse_when_busy(pending) {
                Ok(()) => return Some(Switch::SetDevice { device, answer }),
                Err(err) => {
                    let _ = answer.send(Err(err));
                }
            }
        }
    }
    None
}

fn search(
    embedder: &Embedder,
    store: &Store,
    query: &str,
    scope: Option<&str>,
) -> Result<SearchAnswer, JobError> {
    if store.records.is_empty() {
        return Ok(SearchAnswer {
            groups: Vec::new(),
            baseline: 0.0,
            cut: 0.0,
        });
    }
    let text = format!("{QUERY_PREFIX}{query}");
    let vec = embedder.embed_text(&text).map_err(JobError::from)?;
    let ranked = store.rank(&vec, scope).map_err(JobError::from)?;
    let baseline = index::median(&ranked);
    let cut = index::weak_cut(ranked.first().map(|(_, score)| *score).unwrap_or(0.0));
    Ok(SearchAnswer {
        groups: index::group_hits(&store.records, &ranked, baseline),
        baseline,
        cut,
    })
}

/// Refuse a new action while the engine runs a job.
///
/// The engine holds one job. A second index would fight the first, and a
/// switch of device would drop the model under it.
fn refuse_when_busy(pending: &Option<WebJob>) -> Result<(), JobError> {
    if pending.is_some() {
        return Err(JobError::Busy);
    }
    Ok(())
}

fn start_index(
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    pending: &mut Option<WebJob>,
    flag: &Arc<AtomicBool>,
    paused: &Arc<AtomicBool>,
    folder: &Path,
) -> Result<usize, JobError> {
    refuse_when_busy(pending)?;
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
    // A video can hold a mark with no records, because a fault lands between
    // the two writes. The pass clears those marks, so the run indexes them.
    let cleared = store.reconcile().map_err(JobError::from)?;
    if cleared > 0 {
        println!("Cleared {cleared} finished marks with no records.");
    }
    flag.store(false, Ordering::Relaxed);
    paused.store(false, Ordering::Relaxed);
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
        plan: None,
    });
    Ok(count)
}

/// Begin a re-cut of one video under a new plan.
///
/// The job holds one file, so the engine loop cuts it one item at a time. A
/// running index job refuses the change.
fn start_plan(
    progress: &Arc<Mutex<Progress>>,
    pending: &mut Option<WebJob>,
    flag: &Arc<AtomicBool>,
    paused: &Arc<AtomicBool>,
    path: &str,
    plan: videoembed::plan::Plan,
    media: &Path,
) -> Result<usize, JobError> {
    refuse_when_busy(pending)?;
    let video = PathBuf::from(path);
    if !video.is_file() {
        return Err(JobError::Message(format!(
            "{} is not a file",
            video.display()
        )));
    }
    let run = index::IndexRun::start_with(&video, &plan, media).map_err(JobError::from)?;
    let total = run.total();
    flag.store(false, Ordering::Relaxed);
    paused.store(false, Ordering::Relaxed);
    let label = normalize_label(path);
    {
        let mut held = progress.lock().unwrap();
        // The model state belongs to the server, not to this job. Hold it,
        // because the page shows it after the job ends.
        let device = held.device.clone();
        let gpu_available = held.gpu_available;
        let load_ms = held.load_ms;
        *held = Progress {
            running: true,
            folder: String::new(),
            file_count: 1,
            // The page reads the file name and the item count of the job.
            file: label.clone(),
            file_index: 1,
            window_count: total,
            device,
            gpu_available,
            load_ms,
            ..Progress::default()
        };
    }
    *pending = Some(WebJob {
        files: vec![video],
        file_at: 0,
        run: Some(run),
        records: Vec::new(),
        indexed: 0,
        started: std::time::Instant::now(),
        label,
        plan: Some(plan),
    });
    Ok(total)
}

/// True when one finished run left a whole index for its video.
///
/// A finished video takes the finished mark, and the mark keeps it out of
/// every later run. A run that failed part way holds a partial answer, so it
/// takes no mark and the next run indexes the video again.
fn run_left_a_whole_index(outcome: &Result<Vec<NewRecord>>) -> bool {
    matches!(outcome, Ok(records) if records.is_empty())
}

/// One media item of work. True means the whole job has finished or stopped.
fn step_job(
    embedder: &Embedder,
    ctx: &mut LlamaContext<'_>,
    store: &mut Store,
    progress: &Arc<Mutex<Progress>>,
    flag: &Arc<AtomicBool>,
    paused: &Arc<AtomicBool>,
    job: &mut WebJob,
) -> bool {
    if hold_when_paused(paused, progress) {
        // The job keeps its place and its cut files. The model waits on no job,
        // so the pause costs nothing after the current window ends.
        return false;
    }
    if flag.load(Ordering::Relaxed) {
        // Drop the partial work of this video. A half-indexed video reads as a
        // whole one on the next search.
        job.run = None;
        job.records.clear();
        end_job(progress);
        return true;
    }
    if job.run.is_none() {
        if job.file_at >= job.files.len() {
            end_job(progress);
            return true;
        }
        let video = job.files[job.file_at].clone();
        let label = normalize_label(&video.to_string_lossy());
        // The video takes the plan the store holds for it, or the default.
        let plan = match store.plan_of(&label) {
            Ok(Some(plan)) => plan,
            Ok(None) => videoembed::plan::Plan::default_plan(),
            Err(err) => {
                fail(progress, &err);
                videoembed::plan::Plan::default_plan()
            }
        };
        let media = store.media_dir();
        match index::IndexRun::start_with(&video, &plan, &media) {
            Ok(run) => {
                if store.is_done(&label) {
                    job.file_at += 1;
                    return false;
                }
                if let Err(err) = store.forget(&label) {
                    fail(progress, &err);
                }
                begin_file(progress, label.clone(), job.file_at + 1, run.total());
                job.label = label;
                job.plan = Some(plan);
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
    let finished = match outcome {
        Ok(records) => {
            let empty = records.is_empty();
            job.records.extend(records);
            empty
        }
        Err(err) => {
            fail(progress, &err);
            true
        }
    };

    if finished {
        let records = std::mem::take(&mut job.records);
        if whole {
            job.indexed += records.len();
            if let Err(err) = store.append(records) {
                fail(progress, &err);
            }
            if let Some(plan) = &job.plan {
                if let Err(err) = store.set_plan(&job.label, plan) {
                    fail(progress, &err);
                }
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
            end_job(progress);
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
        held.note = job
            .run
            .as_ref()
            .and_then(|run| run.notes().last())
            .cloned()
            .unwrap_or_default();
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
    // The line of the file before it names a window of that file.
    held.note.clear();
}

fn fail(progress: &Arc<Mutex<Progress>>, err: &anyhow::Error) {
    progress.lock().unwrap().error = Some(format!("{err:#}"));
}

/// Hold or release the running job. A job that does not run refuses the call.
///
/// The page sends one call for each press. The flag holds the state, and the
/// board carries the state to the page.
fn toggle_pause(
    running: bool,
    held: &Arc<AtomicBool>,
    progress: &Arc<Mutex<Progress>>,
) -> Result<bool, JobError> {
    if !running {
        return Err(JobError::Message("no job runs".to_string()));
    }
    let next = !held.load(Ordering::Relaxed);
    held.store(next, Ordering::Relaxed);
    progress.lock().unwrap().paused = next;
    Ok(next)
}

/// True when the pause holds the job.
///
/// A paused job keeps its place and its cut files. The caller returns without
/// work, and the loop waits for the next request.
fn hold_when_paused(flag: &Arc<AtomicBool>, progress: &Arc<Mutex<Progress>>) -> bool {
    if !flag.load(Ordering::Relaxed) {
        return false;
    }
    progress.lock().unwrap().paused = true;
    true
}

/// End the running job, and clear its pause.
fn end_job(progress: &Arc<Mutex<Progress>>) {
    let mut held = progress.lock().unwrap();
    held.running = false;
    held.paused = false;
}

fn handle_request(
    request: Request,
    jobs: &Sender<Job>,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
    thumbs: &Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>>,
    chosen: &Arc<Mutex<HashMap<String, f64>>>,
) -> Result<()> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let post = request.method() == &Method::Post;

    // A thumbnail cuts a frame on this thread, so a search and an index keep
    // running. The route never waits on the engine loop.
    if path == "/api/frame" && request.method() == &Method::Get {
        return serve_frame(request, &url, key, jobs, thumbs, chosen);
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
    chosen: &Arc<Mutex<HashMap<String, f64>>>,
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

    // The opening of a video is often black. The probe steps the time forward
    // to the first frame with a picture. The cache key keeps the time the page
    // asked for, so the map holds one entry for each requested time.
    let picked = {
        let mut map = chosen.lock().unwrap();
        pick_thumb_time(&mut map, Path::new(&path), at as f64)
    };

    match media::thumb_jpeg(Path::new(&path), picked, width) {
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
        (true, "/api/device") => {
            let body: DeviceBody = match read_body(&mut request) {
                Ok(body) => body,
                Err((code, message)) => {
                    request.respond(error_reply(code, &message))?;
                    return Ok(());
                }
            };
            let Some(device) = Device::parse(&body.device) else {
                request.respond(error_reply(400, "the device must be gpu or cpu"))?;
                return Ok(());
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::SetDevice { device, answer });
            match wait.recv() {
                Ok(Ok(device)) => request.respond(json_reply(&DeviceName { device }))?,
                Ok(Err(err)) => {
                    let (code, message) = switch_reply(err);
                    request.respond(error_reply(code, &message))?
                }
                Err(_) => request.respond(error_reply(500, "the engine stopped"))?,
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
        (true, "/api/plan") => {
            let body: PlanBody = match read_body(&mut request) {
                Ok(body) => body,
                Err((code, message)) => {
                    request.respond(error_reply(code, &message))?;
                    return Ok(());
                }
            };
            let plan = match body.plan() {
                Ok(plan) => plan,
                Err(JobError::Message(message)) => {
                    request.respond(error_reply(400, &message))?;
                    return Ok(());
                }
                Err(_) => {
                    request.respond(error_reply(400, "bad plan body"))?;
                    return Ok(());
                }
            };
            let (answer, wait) = channel();
            let _ = jobs.send(Job::SetPlan {
                path: body.path,
                plan,
                answer,
            });
            match wait.recv() {
                Ok(Ok(count)) => request.respond(json_reply(&count))?,
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
                path: body.path,
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
        (true, "/api/pause") => {
            let (answer, wait) = channel();
            let _ = jobs.send(Job::SetPause { answer });
            match wait.recv() {
                Ok(Ok(paused)) => request.respond(json_reply(&Paused { paused }))?,
                Ok(Err(JobError::Message(message))) => {
                    request.respond(error_reply(400, &message))?
                }
                Ok(Err(JobError::Busy)) => {
                    request.respond(error_reply(409, "an index job already runs"))?
                }
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
    let store_path = PathBuf::from("store/index.db");
    let progress = Arc::new(Mutex::new(Progress::default()));
    let thumbs = Arc::new(Mutex::new(HashMap::new()));
    // One picked offset for each video, so the probe runs one time only.
    let chosen = Arc::new(Mutex::new(HashMap::new()));
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    let key = make_key();

    let engine_progress = Arc::clone(&progress);
    let engine_key = key.clone();
    // The pause flag lives with the engine. The page reads the state from
    // the progress answer.
    let paused = Arc::new(AtomicBool::new(false));
    let engine_paused = Arc::clone(&paused);
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
        if store.imported > 0 {
            println!("Imported {} records from the old store.", store.imported);
        }
        let _ = ready_tx.send(Ok(()));
        engine_loop(
            embedder,
            models,
            store,
            engine_progress,
            engine_key,
            jobs_rx,
            engine_paused,
        );
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
        let chosen = Arc::clone(&chosen);
        std::thread::spawn(move || {
            if let Err(err) = handle_request(request, &jobs, &progress, &key, &thumbs, &chosen) {
                eprintln!("A request failed: {err:#}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vector of the model size, with one 1.0 at the front.
    fn unit() -> Vec<f32> {
        let mut vec = vec![0.0f32; videoembed::db::DIM];
        vec[0] = 1.0;
        vec
    }

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

    /// A new file clears the line of the file before it.
    ///
    /// The line names one window of one file. A stale line then reads as a
    /// fault of the file that runs now.
    #[test]
    fn a_new_file_clears_the_note_of_the_last_one() {
        let progress = Arc::new(Mutex::new(Progress::default()));
        progress.lock().unwrap().note = "window at 3.0 s holds 1500 tokens".to_string();

        begin_file(&progress, "clip-two.mp4".to_string(), 2, 12);

        assert!(progress.lock().unwrap().note.is_empty());
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

    /// The chosen offset stands for the life of the process.
    #[test]
    fn the_chosen_offset_stands_in_the_map() {
        let mut chosen = HashMap::new();
        let Some(video) = spike_video() else {
            return;
        };
        let first = pick_thumb_time(&mut chosen, &video, 0.0);
        assert_eq!(first, 0.0);
        assert_eq!(chosen.len(), 1);

        // A second call reads the map, and it cuts no frame of its own.
        let again = pick_thumb_time(&mut chosen, &video, 0.0);
        assert_eq!(again, 0.0);
        assert_eq!(chosen.len(), 1);
    }

    /// A request past the probe window keeps its own time, and it fills no map.
    #[test]
    fn a_late_request_fills_no_map() {
        let mut chosen = HashMap::new();
        let Some(video) = spike_video() else {
            return;
        };
        assert_eq!(pick_thumb_time(&mut chosen, &video, 8.0), 8.0);
        assert!(chosen.is_empty());
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
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store
            .append(vec![
                NewRecord {
                    kind: videoembed::store::KIND_FRAME,
                    label: "a.mp4".into(),
                    start: 0.0,
                    end: 0.25,
                    vec: unit(),
                },
                NewRecord {
                    kind: videoembed::store::KIND_AUDIO,
                    label: "a.mp4".into(),
                    start: 0.0,
                    end: 1.0,
                    vec: unit(),
                },
                NewRecord {
                    kind: videoembed::store::KIND_NOTE,
                    label: "a note".into(),
                    start: 0.0,
                    end: 0.0,
                    vec: unit(),
                },
            ])
            .unwrap();

        let rows = videos(&store);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].path, "a.mp4");
        assert_eq!(rows[0].items, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One video makes one row, whatever separator its label carries.
    #[test]
    fn a_video_row_gathers_a_video_across_path_separators() {
        let dir = std::env::temp_dir().join(format!("vei_web_sep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store
            .append(vec![
                NewRecord {
                    kind: videoembed::store::KIND_FRAME,
                    label: r"D:\video\a.mp4".into(),
                    start: 0.0,
                    end: 0.25,
                    vec: unit(),
                },
                NewRecord {
                    kind: videoembed::store::KIND_FRAME,
                    label: "D:/video/a.mp4".into(),
                    start: 0.25,
                    end: 0.5,
                    vec: unit(),
                },
            ])
            .unwrap();

        let rows = videos(&store);

        assert_eq!(rows.len(), 1, "one video makes one row");
        assert_eq!(rows[0].path, "D:/video/a.mp4");
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

    /// One job with no work, for the tests that need a busy engine.
    fn test_job() -> WebJob {
        WebJob {
            files: Vec::new(),
            file_at: 0,
            run: None,
            records: Vec::new(),
            indexed: 0,
            started: std::time::Instant::now(),
            label: String::new(),
            plan: None,
        }
    }

    /// The engine holds one job, so a busy engine refuses a new action.
    ///
    /// A switch of device drops the model, and a running index holds it.
    #[test]
    fn a_busy_engine_refuses_a_new_action() {
        let none: Option<WebJob> = None;
        assert!(refuse_when_busy(&none).is_ok());
        assert!(refuse_when_busy(&Some(test_job())).is_err());
    }

    /// Only a run that reached its end may mark its video as finished.
    ///
    /// The mark keeps a video out of every later run, so a run that failed
    /// part way must leave no mark. The video then indexes again.
    #[test]
    fn only_a_run_that_reached_its_end_is_whole() {
        let more_work = Ok(vec![NewRecord {
            kind: videoembed::store::KIND_FRAME,
            label: "a.mp4".to_string(),
            start: 0.0,
            end: 0.25,
            vec: unit(),
        }]);
        assert!(!run_left_a_whole_index(&more_work), "the run is not finished");
        assert!(
            run_left_a_whole_index(&Ok(Vec::new())),
            "the run reached its end"
        );
        assert!(
            !run_left_a_whole_index(&Err(anyhow!("a broken frame"))),
            "a failed run holds a partial answer"
        );
    }

    /// The videos reply carries the plan of each video.
    #[test]
    fn a_video_row_carries_its_plan() {
        let dir = std::env::temp_dir().join(format!("vei_web_plan_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store
            .append(vec![NewRecord {
                kind: videoembed::store::KIND_WINDOW,
                label: "a.mp4".into(),
                start: 0.0,
                end: 2.0,
                vec: unit(),
            }])
            .unwrap();
        store
            .set_plan(
                "a.mp4",
                &videoembed::plan::Plan::of(videoembed::plan::Preset::Speech),
            )
            .unwrap();

        let rows = videos(&store);

        assert_eq!(rows[0].plan, "speech");
        assert_eq!(rows[0].window, 6.0);
        assert_eq!(rows[0].frames, 6);
        assert!(rows[0].audio);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A plan that matches no preset names no preset.
    ///
    /// The page sends the name back to the server. A false name would replace
    /// the custom window and frames of the video with that preset.
    #[test]
    fn a_custom_plan_names_no_preset() {
        let custom = videoembed::plan::Plan::of(videoembed::plan::Preset::Object)
            .set_words("window=3 frames=3")
            .unwrap();

        assert_eq!(preset_name(&custom), "");
    }

    /// A plan job points the progress at the video that it cuts.
    ///
    /// A folder run clears the mark of a video that holds no records.
    ///
    /// The mark would keep the video out of every later run, so the run must
    /// clear it before it walks the folder.
    #[test]
    fn a_folder_run_clears_the_mark_of_a_video_with_no_records() {
        let dir = std::env::temp_dir().join(format!("vei_web_reconcile_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("a.mp4");
        std::fs::write(&clip, b"not a video").unwrap();
        let mut store = Store::open(&dir.join("index.db")).unwrap();
        store.mark_done(&clip.to_string_lossy()).unwrap();

        let progress = Arc::new(Mutex::new(Progress::default()));
        let flag = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let mut pending = None;

        let count = start_index(&mut store, &progress, &mut pending, &flag, &paused, &dir).unwrap();

        assert_eq!(count, 1, "the folder holds one video");
        assert!(
            !store.is_done(&clip.to_string_lossy()),
            "the stale mark goes"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The page shows the file name and the item count of the running job. A
    /// zero count gives an empty bar and an item count of "N/0".
    #[test]
    fn a_plan_job_fills_the_progress_fields() {
        let Some(video) = spike_video() else {
            return;
        };
        let progress = Arc::new(Mutex::new(Progress::default()));
        let flag = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let mut pending = None;

        let media = std::env::temp_dir().join(format!("vei_web_plan_media_{}", std::process::id()));
        let total = start_plan(
            &progress,
            &mut pending,
            &flag,
            &paused,
            &video.to_string_lossy(),
            videoembed::plan::Plan::of(videoembed::plan::Preset::Speech),
            &media,
        )
        .unwrap();

        let held = progress.lock().unwrap();
        assert!(held.running, "the plan job runs");
        assert_eq!(held.window_count, total, "the bar needs the item count");
        assert_eq!(held.file_count, 1);
        assert!(
            held.file.ends_with("testvideo_nolabels.mp4"),
            "the page names the file, got {:?}",
            held.file
        );
    }

    /// The spike clip, when it is here.
    fn spike_video() -> Option<PathBuf> {
        let path = PathBuf::from("../spike/testvideo_nolabels.mp4");
        path.is_file().then_some(path)
    }

    /// The plan body reads a preset name.
    #[test]
    fn the_plan_body_reads_a_preset() {
        let body: PlanBody = serde_json::from_str(r#"{"path":"a.mp4","preset":"action"}"#).unwrap();
        let plan = body.plan().unwrap();
        assert_eq!(
            plan,
            videoembed::plan::Plan::of(videoembed::plan::Preset::Action)
        );
    }

    /// The plan body reads a set of values.
    #[test]
    fn the_plan_body_reads_values() {
        let body: PlanBody =
            serde_json::from_str(r#"{"path":"a.mp4","window":4,"frames":4,"audio":true}"#).unwrap();
        let plan = body.plan().unwrap();
        assert_eq!(plan.window, 4.0);
        assert_eq!(plan.frames, 4);
        assert!(plan.audio);
    }

    /// A body with no plan value names the fault.
    #[test]
    fn an_empty_plan_body_is_refused() {
        let body: PlanBody = serde_json::from_str(r#"{"path":"a.mp4"}"#).unwrap();
        assert!(body.plan().is_err());
    }

    /// The page reads the paused state from the progress answer.
    ///
    /// A reloaded page takes the state from this answer, so the button reads
    /// Resume after the reload.
    #[test]
    fn the_progress_answer_carries_the_paused_state() {
        let stopped = Progress {
            running: true,
            paused: false,
            ..Progress::default()
        };
        let paused = Progress {
            running: true,
            paused: true,
            ..Progress::default()
        };

        let json = serde_json::to_string(&stopped).unwrap();
        assert!(json.contains("\"paused\":false"), "got {json}");
        let json = serde_json::to_string(&paused).unwrap();
        assert!(json.contains("\"paused\":true"), "got {json}");
    }

    /// A pause request needs a running job.
    #[test]
    fn a_job_that_does_not_run_refuses_a_pause() {
        let flag = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Progress::default()));

        let outcome = toggle_pause(false, &flag, &board);

        assert!(
            matches!(outcome, Err(JobError::Message(_))),
            "the call must refuse a job that does not run"
        );
        assert!(!flag.load(Ordering::Relaxed), "the flag stays clear");
    }

    /// One call holds the job, and the next call lets it run again.
    #[test]
    fn a_pause_toggles_the_flag_and_the_board() {
        let flag = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Progress {
            running: true,
            ..Progress::default()
        }));

        let first = toggle_pause(true, &flag, &board).unwrap();
        assert!(first, "the first call holds the job");
        assert!(board.lock().unwrap().paused, "the page reads the state");

        let second = toggle_pause(true, &flag, &board).unwrap();
        assert!(!second, "the second call lets it run");
        assert!(!board.lock().unwrap().paused, "the page reads the state");
    }

    /// The gate holds a paused job, and it shows the state on the board.
    #[test]
    fn a_paused_job_waits_at_the_gate() {
        let flag = Arc::new(AtomicBool::new(false));
        let board = Arc::new(Mutex::new(Progress {
            running: true,
            ..Progress::default()
        }));

        assert!(!hold_when_paused(&flag, &board), "a clear flag holds nothing");
        assert!(!board.lock().unwrap().paused);

        flag.store(true, Ordering::Relaxed);
        assert!(hold_when_paused(&flag, &board), "a set flag holds the job");
        assert!(board.lock().unwrap().paused, "the page reads the state");
    }

    /// The end of a job clears the pause with it.
    ///
    /// A pause that outlives its job leaves a job that advances on no item, and
    /// the page shows a job that runs forever.
    #[test]
    fn the_end_of_a_job_clears_the_pause() {
        let board = Arc::new(Mutex::new(Progress {
            running: true,
            paused: true,
            ..Progress::default()
        }));

        end_job(&board);

        let held = board.lock().unwrap();
        assert!(!held.running, "the job ends");
        assert!(!held.paused, "the pause goes with it");
    }

    /// A failed load is a fault of the server, and a busy engine is not.
    ///
    /// The page shows the code, and a person reads 400 as a bad request. A
    /// model that will not load on the card is not the fault of the caller.
    #[test]
    fn a_failed_load_answers_500_and_a_busy_switch_answers_409() {
        let (code, message) = switch_reply(JobError::Busy);
        assert_eq!(code, 409);
        // The engine refuses a switch for two reasons: a job runs, or another
        // switch loads. The line must fit both, and it must not name a job
        // that does not exist.
        assert_eq!(message, "the engine runs another job");

        let (code, message) = switch_reply(JobError::Message("the card is out of memory".to_string()));
        assert_eq!(code, 500, "a load fault belongs to the server");
        assert_eq!(message, "the card is out of memory");
    }

    /// A device switch reads the state again, so the speed line is not stale.
    ///
    /// The page holds no test runner, so this pins the call in the source. The
    /// select and the speed line both come from the progress answer.
    #[test]
    fn the_page_reads_the_state_again_after_a_switch() {
        let at = JS
            .find("async function switchDevice")
            .expect("the page holds the switch");
        // The file holds CRLF endings, so the scan looks for the brace alone.
        let body = &JS[at..];
        let open = body.find('{').expect("the switch opens");
        let rest = &body[open..];
        let end = rest.find("\n}").expect("the switch ends");
        assert!(
            rest[..end].contains("/api/progress"),
            "a switch must read the state again, or the page keeps the old device"
        );
    }
}
