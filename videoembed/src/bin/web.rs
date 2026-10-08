//! The web server. One thread serves HTTP. One thread owns the model.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use videoembed::embed::{Embedder, QUERY_PREFIX};
use videoembed::index::{self, Group, Progress};
use videoembed::media;
use videoembed::store::{KIND_WINDOW, Record, Store};

type Reply = Response<Box<dyn Read + Send>>;

const ADDRESS: &str = "127.0.0.1:8787";
const PAGE: &str = include_str!("../../assets/index.html");
const JS: &str = include_str!("../../assets/app.js");
const CSS: &str = include_str!("../../assets/app.css");

/// One request from the server thread to the engine thread.
enum Job {
    State(Sender<State>),
    Videos(Sender<Vec<VideoRow>>),
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
    windows: usize,
}

#[derive(Serialize)]
struct Rows {
    videos: Vec<VideoRow>,
}

#[derive(Serialize)]
struct SearchAnswer {
    groups: Vec<Group>,
}

#[derive(Deserialize)]
struct PathBody {
    path: String,
}

#[derive(Deserialize)]
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

fn read_body<T: for<'de> Deserialize<'de>>(request: &mut Request) -> Result<T, String> {
    let mut text = String::new();
    request
        .as_reader()
        .read_to_string(&mut text)
        .map_err(|err| format!("read the body: {err}"))?;
    serde_json::from_str(&text).map_err(|err| format!("the body is not valid JSON: {err}"))
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

struct IndexJob {
    files: Vec<PathBuf>,
    file_at: usize,
    plan: Vec<(f64, f64)>,
    at: usize,
    records: Vec<Record>,
    on_file: bool,
    indexed: usize,
}

/// The engine thread. It owns the model and the store.
struct Engine {
    embedder: Embedder,
    store: Store,
    progress: Arc<Mutex<Progress>>,
    pending: Option<IndexJob>,
    key: String,
}

impl Engine {
    fn start(models: &Path, store_path: &Path, progress: Arc<Mutex<Progress>>) -> Result<Self> {
        let embedder = Embedder::load(models)?;
        let store = Store::open(store_path)?;
        Ok(Self {
            embedder,
            store,
            progress,
            pending: None,
            key: make_key(),
        })
    }

    /// Answer jobs forever. A running index gives one window per turn.
    fn run(mut self, jobs: Receiver<Job>) {
        loop {
            if self.pending.is_some() {
                self.step_job();
                while let Ok(job) = jobs.try_recv() {
                    self.handle(job);
                }
            } else {
                match jobs.recv() {
                    Ok(job) => self.handle(job),
                    Err(_) => break,
                }
            }
        }
    }

    fn handle(&mut self, job: Job) {
        match job {
            Job::State(answer) => {
                let _ = answer.send(self.state());
            }
            Job::Videos(answer) => {
                let _ = answer.send(self.videos());
            }
            Job::Holds { path, answer } => {
                let _ = answer.send(self.store.holds_video(&path));
            }
            Job::Forget { path, answer } => {
                let outcome = self.store.forget(&path).map_err(JobError::from);
                let _ = answer.send(outcome);
            }
            Job::Search { query, answer } => {
                let _ = answer.send(self.search(&query));
            }
            Job::StartIndex { folder, answer } => {
                let _ = answer.send(self.start_index(&folder));
            }
        }
    }

    fn state(&self) -> State {
        State {
            version: env!("CARGO_PKG_VERSION").to_string(),
            records: self.store.records.len(),
            videos: self.videos().len(),
            key: self.key.clone(),
        }
    }

    fn videos(&self) -> Vec<VideoRow> {
        let mut rows: Vec<VideoRow> = Vec::new();
        for record in &self.store.records {
            if record.kind != KIND_WINDOW {
                continue;
            }
            match rows.iter_mut().find(|row| row.path == record.label) {
                Some(row) => row.windows += 1,
                None => rows.push(VideoRow {
                    path: record.label.clone(),
                    windows: 1,
                }),
            }
        }
        rows
    }

    fn search(&self, query: &str) -> Result<SearchAnswer, JobError> {
        if self.store.records.is_empty() {
            return Ok(SearchAnswer { groups: Vec::new() });
        }
        let text = format!("{QUERY_PREFIX}{query}");
        let vec = self.embedder.embed_text(&text).map_err(JobError::from)?;
        let ranked = self.store.rank(&vec);
        Ok(SearchAnswer {
            groups: index::group_hits(&self.store.records, &ranked),
        })
    }

    fn start_index(&mut self, folder: &Path) -> Result<usize, JobError> {
        if self.pending.is_some() {
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
        {
            let mut progress = self.progress.lock().unwrap();
            *progress = Progress {
                running: true,
                folder: folder.display().to_string(),
                file_count: count,
                ..Progress::default()
            };
        }
        self.pending = Some(IndexJob {
            files,
            file_at: 0,
            plan: Vec::new(),
            at: 0,
            records: Vec::new(),
            on_file: false,
            indexed: 0,
        });
        Ok(count)
    }

    /// One window of work, or the move to the next file.
    fn step_job(&mut self) {
        let Some(mut job) = self.pending.take() else {
            return;
        };

        if !job.on_file {
            if job.file_at >= job.files.len() {
                self.progress.lock().unwrap().running = false;
                return;
            }
            let video = job.files[job.file_at].clone();
            match self.open_file(&video, job.file_at + 1) {
                Ok(plan) => {
                    job.plan = plan;
                    job.at = 0;
                    job.records.clear();
                    job.on_file = true;
                }
                Err(err) => {
                    self.fail(&err);
                    job.file_at += 1;
                }
            }
            self.pending = Some(job);
            return;
        }

        let video = job.files[job.file_at].clone();
        let (start, end) = job.plan[job.at];
        if let Err(err) = self.one_window(&video, start, end, &mut job.records) {
            self.fail(&err);
        }
        job.at += 1;
        {
            let mut progress = self.progress.lock().unwrap();
            progress.window = job.at;
            progress.indexed = job.indexed + job.records.len();
        }
        if job.at >= job.plan.len() {
            let records = std::mem::take(&mut job.records);
            job.indexed += records.len();
            if let Err(err) = self.store.append(records) {
                self.fail(&err);
            }
            job.file_at += 1;
            job.on_file = false;
        }
        self.pending = Some(job);
    }

    /// Probe the duration and drop the old records of this file.
    ///
    /// The probe runs first, so a broken file keeps its old index.
    fn open_file(&mut self, video: &Path, number: usize) -> Result<Vec<(f64, f64)>> {
        let duration = media::probe_duration(video)?;
        let plan = index::windows(duration);
        if plan.is_empty() {
            bail!("{} is shorter than one window", video.display());
        }
        let label = video.to_string_lossy().to_string();
        self.store.forget(&label)?;
        {
            let mut progress = self.progress.lock().unwrap();
            progress.file = label;
            progress.file_index = number;
            progress.window = 0;
            progress.window_count = plan.len();
            progress.error = None;
        }
        Ok(plan)
    }

    fn one_window(
        &self,
        video: &Path,
        start: f64,
        end: f64,
        records: &mut Vec<Record>,
    ) -> Result<()> {
        let dir = index::work_dir(records.len());
        let outcome = media::extract_window(video, start, end, &dir)
            .and_then(|(frames, audio)| self.embedder.embed_window(&frames, &audio));
        let _ = std::fs::remove_dir_all(&dir);
        records.push(Record {
            kind: KIND_WINDOW,
            label: video.to_string_lossy().to_string(),
            start: start as f32,
            end: end as f32,
            vec: outcome?,
        });
        Ok(())
    }

    fn fail(&self, err: &anyhow::Error) {
        self.progress.lock().unwrap().error = Some(format!("{err:#}"));
    }
}

fn handle(
    request: Request,
    jobs: &Sender<Job>,
    progress: &Arc<Mutex<Progress>>,
    key: &str,
) -> Result<()> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let post = request.method() == &Method::Post;

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
                Err(message) => {
                    request.respond(error_reply(400, &message))?;
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
                Err(message) => {
                    request.respond(error_reply(400, &message))?;
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
                Err(message) => {
                    request.respond(error_reply(400, &message))?;
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
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<String, String>>();

    let engine_progress = Arc::clone(&progress);
    std::thread::spawn(move || {
        let engine = match Engine::start(&models, &store_path, engine_progress) {
            Ok(engine) => engine,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("{err:#}")));
                return;
            }
        };
        // The ready signal carries the video key, so both threads share one value.
        let _ = ready_tx.send(Ok(engine.key.clone()));
        engine.run(jobs_rx);
    });

    // The model and the store must be ready before the server answers one call.
    let key = match ready_rx.recv().context("the engine thread stopped early")? {
        Ok(key) => key,
        Err(message) => bail!("the engine failed to start: {message}"),
    };

    let server = Server::http(ADDRESS).map_err(|err| anyhow!("{err}"))?;
    println!("videoembed is at http://{ADDRESS}/");
    println!("Stop the server with the Quit button or with Ctrl C.");
    let _ = open::that(format!("http://{ADDRESS}/"));

    for request in server.incoming_requests() {
        let jobs = jobs_tx.clone();
        let progress = Arc::clone(&progress);
        let key = key.clone();
        std::thread::spawn(move || {
            if let Err(err) = handle(request, &jobs, &progress, &key) {
                eprintln!("A request failed: {err:#}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_mime_type_follows_the_extension() {
        assert_eq!(mime_of(Path::new("a.mp4")), "video/mp4");
        assert_eq!(mime_of(Path::new("a.MKV")), "video/x-matroska");
        assert_eq!(mime_of(Path::new("a.txt")), "application/octet-stream");
    }
}
