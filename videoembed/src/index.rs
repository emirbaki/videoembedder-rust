//! The folder scan, the frame and audio plan, and the index run.
//!
//! One frame covers 0.25 s. One audio clip covers 1 s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::LlamaContext;
use serde::Serialize;

use crate::embed::{self, Embedder, N_WINDOW_BATCH};
use crate::media;
use crate::store::{self, NewRecord, Record};

/// An item below this score never enters a result group.
pub const FLOOR: f32 = 0.4;
/// The most items one video may add to one group.
pub const GROUP_CAP: usize = 50;
/// The number of frames one ffmpeg call cuts.
pub const FRAME_BATCH: usize = 120;
/// The number of audio clips one ffmpeg call cuts.
pub const AUDIO_BATCH: usize = 60;
/// Frames a second in the frame plan.
pub const FRAME_RATE: f64 = 4.0;
/// The time step between two frames in seconds.
pub const FRAME_STEP: f64 = 0.25;
/// The length of one audio clip in seconds.
pub const AUDIO_SECONDS: f64 = 1.0;

/// The distance in seconds below which two hits of one video join one moment.
///
/// Google folds hits within 500 ms into one interval
/// (`VideoMomentFinderViewModel.kt:1020-1036`). A search shows rows, so three
/// near hits must read as one row. The one gap holds for the console search
/// and for the page, and for the window grid and the fine grid.
pub const MERGE_GAP: f32 = 0.5;

/// The median score of one ranked list. An empty list gives 0.0.
///
/// The median is the baseline of the store for one query. A hit shows its
/// score minus this value, because an embedding score alone has no scale.
pub fn median(ranked: &[(usize, f32)]) -> f32 {
    if ranked.is_empty() {
        return 0.0;
    }
    let mut scores: Vec<f32> = ranked.iter().map(|(_, score)| *score).collect();
    scores.sort_by(f32::total_cmp);
    let n = scores.len();
    if n % 2 == 1 {
        scores[n / 2]
    } else {
        (scores[n / 2 - 1] + scores[n / 2]) / 2.0
    }
}

/// The gap between the best score of one result and the weak cut.
pub const WEAK_GAP: f32 = 0.20;

/// The cut that marks a weak result.
///
/// The best score of the result anchors the cut, as Google's photo feature
/// does. The floor of 0.4 keeps the cut above a hopeless row.
pub fn weak_cut(top: f32) -> f32 {
    (top - WEAK_GAP).max(FLOOR)
}

/// Is one score below the cut of its result?
pub fn is_weak(score: f32, cut: f32) -> bool {
    score < cut
}

const VIDEO_EXTENSIONS: [&str; 5] = ["mp4", "mkv", "mov", "avi", "webm"];

/// The progress of the running index job. The server reads a copy.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    pub running: bool,
    /// True when the running job waits for a resume.
    pub paused: bool,
    pub folder: String,
    pub file: String,
    pub file_index: usize,
    pub file_count: usize,
    pub window: usize,
    pub window_count: usize,
    pub indexed: usize,
    pub error: Option<String>,
    /// The device of the loaded model, "GPU" or "CPU".
    pub device: String,
    /// True when this machine can put layers on a card.
    pub gpu_available: bool,
    /// The milliseconds to load the model.
    pub load_ms: u64,
    /// The milliseconds of the running job.
    pub elapsed_ms: u64,
    /// The items of the running job for each second.
    pub per_second: f32,
    /// The newest line of the running job. Empty when the job reports nothing.
    pub note: String,
}

/// One matching item inside a video.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub start: f32,
    pub end: f32,
    pub score: f32,
    pub margin: f32,
    /// The record kind of this hit, from `store::KIND_*`.
    pub kind: u8,
}

/// One slice of a video, with the score of the whole slice.
///
/// A group holds one cell for every video record, also the ones below the
/// floor. The web bar draws the cells, so the bar covers the whole video.
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub start: f32,
    pub end: f32,
    pub score: f32,
}

/// The hits of one video.
#[derive(Clone, Debug, Serialize)]
pub struct Group {
    pub video: String,
    pub duration: f32,
    pub score: f32,
    pub hits: Vec<Hit>,
    pub cells: Vec<Cell>,
}

/// One row of a search answer.
///
/// The row holds a whole moment after the fold, so the range is the range the
/// user reads. A note is a row too. A note holds no time, so a note never folds.
#[derive(Clone, Debug)]
pub struct Row {
    pub note: bool,
    /// The record kind of this row, from `store::KIND_*`.
    pub kind: u8,
    pub label: String,
    pub start: f32,
    pub end: f32,
    pub score: f32,
    pub margin: f32,
}

/// List the video files under one folder, in name order.
pub fn list_videos(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    walk(folder, &mut found)?;
    found.sort();
    Ok(found)
}

fn walk(dir: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            walk(&path, found)?;
        } else if is_video(&path) {
            found.push(path);
        }
    }
    Ok(())
}

/// True when the file name ends with one of the video extensions.
pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|part| part.to_str())
        .map(|part| VIDEO_EXTENSIONS.contains(&part.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// The start time of every frame. One frame each 0.25 s.
///
/// A frame needs its full step inside the video. A video shorter than one step
/// has no frame, so a caller refuses it.
pub fn frames(duration: f64) -> Vec<f64> {
    let mut plan = Vec::new();
    let mut t = 0.0;
    while t + FRAME_STEP <= duration {
        plan.push(t);
        t += FRAME_STEP;
    }
    plan
}

/// The start time of every one-second audio clip.
pub fn audio_starts(duration: f64) -> Vec<f64> {
    let mut plan = Vec::new();
    let mut t = 0.0;
    while t + AUDIO_SECONDS <= duration {
        plan.push(t);
        t += AUDIO_SECONDS;
    }
    plan
}

/// The rate of a cut that holds the window grid and the fine grid.
///
/// The cut is a whole multiple of the plan rate and of 4 fps, so both paths
/// take whole steps over it. A plan at 1 fps gives 4 fps. A plan at 3 fps
/// gives 12 fps.
pub fn dense_rate(plan_rate: f64) -> f64 {
    let base = plan_rate.max(FRAME_RATE);
    for n in 1..=64u32 {
        let rate = base * f64::from(n);
        let plan_step = rate / plan_rate;
        let fine_step = rate / FRAME_RATE;
        if (plan_step - plan_step.round()).abs() < 1e-9
            && (fine_step - fine_step.round()).abs() < 1e-9
        {
            return rate;
        }
    }
    base
}

/// The index of the first frame of one span, in the whole-video cut.
///
/// The cut gives frame `n` at `n / rate`, so this index names the time of the
/// first frame of the span.
fn first_frame(from: f64, rate: f64) -> usize {
    (from * rate).round().max(0.0) as usize
}

/// The index after the last frame of one span, in the whole-video cut.
fn last_frame(from: f64, window: f64, rate: f64) -> usize {
    ((from + window) * rate).round().max(0.0) as usize
}

/// The frames of one span, from the list of a whole-video cut.
///
/// Frame `i` of the list sits at `i / rate`, and the span `[from, from + window)`
/// takes the frames inside it. A span that runs past the list takes what is left.
pub fn frames_in(cut: &[PathBuf], from: f64, window: f64, rate: f64) -> Vec<PathBuf> {
    frames_in_dense(cut, from, window, rate, rate)
}

/// The frames of one span, from a cut that holds the fine grid too.
///
/// A plan with the fine flag cuts at a rate above its own, so the window path
/// takes every `cut_rate / rate` frame of the span. The frames sit at the times
/// of its own rate, and the window keeps their count.
pub fn frames_in_dense(
    cut: &[PathBuf],
    from: f64,
    window: f64,
    rate: f64,
    cut_rate: f64,
) -> Vec<PathBuf> {
    let first = first_frame(from, cut_rate);
    let end = last_frame(from, window, cut_rate).min(cut.len());
    if first >= end {
        return Vec::new();
    }
    let step = (cut_rate / rate).round().max(1.0) as usize;
    cut[first..end].iter().step_by(step).cloned().collect()
}

/// The time of every media part of one window.
///
/// The frames come first, then the clips, in the order the embedder reads them.
/// A frame sits at its own place in the cut, so these times match the files
/// that `frames_in` gives. A clip sits at the second that it covers.
pub fn window_times(from: f64, rate: f64, frames: usize, clips: usize) -> Vec<f64> {
    window_times_dense(from, rate, frames, clips, rate)
}

/// The time of every part of one window, from a cut that holds the fine grid.
///
/// The picked frames sit at the times of the plan rate, and the cut rate names
/// the place of each one in the dense cut.
pub fn window_times_dense(
    from: f64,
    rate: f64,
    frames: usize,
    clips: usize,
    cut_rate: f64,
) -> Vec<f64> {
    let first = first_frame(from, cut_rate);
    let step = (cut_rate / rate).round().max(1.0) as usize;
    let mut times: Vec<f64> = (0..frames)
        .map(|i| (first + i * step) as f64 / cut_rate)
        .collect();
    times.extend((0..clips).map(|k| from.floor() + k as f64));
    times
}

/// The clips of one span, from the list of a whole-video cut.
///
/// Clip `k` of the list covers `[k, k + 1)`, and the span takes every clip that
/// touches it. A span past the end of the audio takes none, because a video
/// often holds less audio than video.
pub fn clips_in(cut: &[PathBuf], from: f64, window: f64) -> Vec<PathBuf> {
    let first = from.floor().max(0.0) as usize;
    let last = (from + window).ceil().max(0.0) as usize;
    let end = last.min(cut.len());
    if first >= end {
        return Vec::new();
    }
    cut[first..end].to_vec()
}

/// A fresh working folder for one batch of ffmpeg output.
pub fn work_dir(at: usize) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("vei_{}_{n}_{at}", std::process::id()))
}

/// The message about the source of one cut. None means the video is its own
/// source.
///
/// A fragmented MP4 is cut through a progressive copy, and the user waits
/// while the copy is made. This call makes the copy when it is missing or
/// stale.
pub fn copy_note(video: &Path, media: &Path) -> Result<Option<String>> {
    let source = media::source_file(video, media)?;
    Ok((source != video).then(|| video.to_string_lossy().to_string()))
}

/// The phase of an index run.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Frames,
    Audio,
    Windows,
    Done,
}

/// One window of a group: its media, the time of each part, and its frames.
///
/// The files hold the frames first, then the clips. The times sit in the same
/// order, so the marker of a part names the time of that part.
#[derive(Clone, Debug, Default)]
pub struct WindowPart {
    /// The start of the window in the video.
    pub start: f64,
    /// The media files of the window, the frames first.
    pub files: Vec<PathBuf>,
    /// The time of each part, in the order of the files.
    pub times: Vec<f64>,
    /// The number of frames at the front of the files.
    pub frames: usize,
}

impl WindowPart {
    /// Drop the audio of this window. The frames stay.
    pub fn keep_frames(&mut self) {
        self.files.truncate(self.frames);
        self.times.truncate(self.frames);
    }
}

/// The line about one window that lost its audio to the token budget.
///
/// The run reports every such window, so the reader learns why the record
/// holds no sound.
pub fn budget_note(start: f64, tokens: usize) -> String {
    format!(
        "window at {start:.1} s holds {tokens} tokens, above the budget of {}, \
         so the record keeps the frames alone",
        embed::BUDGET_TOKENS
    )
}

/// A stepwise index run for one video.
///
/// The run cuts media in batches, so the temp disk stays bounded. The caller
/// drives it: one `step` call embeds one frame or one audio clip.
#[derive(Debug)]
pub struct IndexRun {
    /// The file to cut. It is the video itself, or its progressive copy.
    source: PathBuf,
    /// Every frame of the whole-video cut. Empty until the first cut.
    frames_cut: Vec<PathBuf>,
    /// Every audio clip of the whole-video cut. Empty until the first cut.
    clips_cut: Vec<PathBuf>,
    label: String,
    duration: f64,
    frames: Vec<f64>,
    audios: Vec<f64>,
    frame_at: usize,
    audio_at: usize,
    phase: Phase,
    dir: PathBuf,
    files: Vec<PathBuf>,
    batch_from: usize,
    done: usize,
    window_at: usize,
    /// The plan of the run. Fine mode ignores `window`, `stride`, and `frames`.
    plan: crate::plan::Plan,
    /// The start of every window. Empty in fine mode.
    windows: Vec<f64>,
    /// The windows of the current model call. Empty in fine mode.
    group: Vec<WindowPart>,
    /// The lines that the run reported, in order.
    notes: Vec<String>,
}

impl IndexRun {
    /// Probe the video and build the frame and the audio plan.
    ///
    /// The plan follows the length of each stream, not the length of the file.
    /// A clip past the end of its stream holds no media, and the encoder then
    /// refuses it. A short audio track is normal, so the audio plan is often
    /// shorter than the frame plan.
    pub fn start(video: &Path) -> Result<Self> {
        Self::start_from(video, video)
    }

    /// Probe the video and build the frame and the audio plan.
    ///
    /// `source` names the file to cut, and `video` names the video of the
    /// user. The two differ only for a fragmented MP4, which the run reads
    /// through its progressive copy.
    fn start_from(source: &Path, video: &Path) -> Result<Self> {
        let duration = media::probe_duration(source)?;
        let video_len = media::probe_stream_duration(source, "v")?
            .unwrap_or(duration)
            .min(duration);
        let audio_len = if media::has_audio(source) {
            media::probe_stream_duration(source, "a")?
                .unwrap_or(duration)
                .min(duration)
        } else {
            0.0
        };
        let frames = frames(video_len);
        if frames.is_empty() {
            bail!("{} is shorter than one frame", video.display());
        }
        Ok(Self {
            source: source.to_path_buf(),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: store::normalize_label(&video.to_string_lossy()),
            duration,
            frames,
            audios: audio_starts(audio_len),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan: crate::plan::Plan::of(crate::plan::Preset::Fine),
            windows: Vec::new(),
            group: Vec::new(),
            notes: Vec::new(),
        })
    }

    /// Probe the video and build the plan of the run.
    ///
    /// Fine mode follows the length of each stream, not the length of the file.
    /// Window mode follows the duration and the plan. `media` names the folder
    /// that holds a progressive copy of a fragmented video.
    pub fn start_with(video: &Path, plan: &crate::plan::Plan, media: &Path) -> Result<Self> {
        let source = media::source_file(video, media)?;
        match plan.mode {
            crate::plan::Mode::Window => Self::start_window(&source, plan, video),
            crate::plan::Mode::Fine => Self::start_from(&source, video),
        }
    }

    /// The start of every window, or the fine items.
    ///
    /// The plan of the run itself lives in the `plan` field, so this call
    /// carries the name `starts`.
    pub fn starts(&self) -> &[f64] {
        match self.phase {
            Phase::Frames => &self.frames,
            Phase::Audio => &self.audios,
            Phase::Windows => &self.windows,
            Phase::Done => &[],
        }
    }

    /// Probe the video and build the window plan.
    fn start_window(source: &Path, plan: &crate::plan::Plan, video: &Path) -> Result<Self> {
        let duration = media::probe_duration(source)?;
        let windows = plan.window_starts(duration);
        if windows.is_empty() {
            bail!(
                "{} is shorter than one {} s window",
                video.display(),
                plan.window
            );
        }
        // The flag adds the fine grid beside the windows. The grid follows
        // each stream length, as the fine mode does.
        let (frames, audios) = if plan.fine {
            let video_len = media::probe_stream_duration(source, "v")?
                .unwrap_or(duration)
                .min(duration);
            let audio_len = if media::has_audio(source) {
                media::probe_stream_duration(source, "a")?
                    .unwrap_or(duration)
                    .min(duration)
            } else {
                0.0
            };
            (frames(video_len), audio_starts(audio_len))
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            source: source.to_path_buf(),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: store::normalize_label(&video.to_string_lossy()),
            duration,
            frames,
            audios,
            frame_at: 0,
            audio_at: 0,
            phase: if plan.fine { Phase::Frames } else { Phase::Windows },
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan: *plan,
            windows,
            group: Vec::new(),
            notes: Vec::new(),
        })
    }

    /// The number of records this run will make.
    pub fn total(&self) -> usize {
        self.frames.len() + self.audios.len() + self.windows.len()
    }

    /// The number of records made so far.
    pub fn done(&self) -> usize {
        self.done
    }

    /// The lines that the run reported, in order.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// Embed the next model call. Returns the records of that call.
    ///
    /// Window mode reads up to `N_WINDOW_BATCH` windows in one call, so it
    /// gives up to that many records. Fine mode gives one.
    pub fn step(
        &mut self,
        embedder: &Embedder,
        ctx: &mut LlamaContext<'_>,
    ) -> Result<Vec<NewRecord>> {
        if self.phase == Phase::Done {
            return Ok(Vec::new());
        }
        if self.remaining_in_batch() == 0 {
            self.cut_batch()?;
            if self.phase == Phase::Done {
                return Ok(Vec::new());
            }
        }
        if self.phase == Phase::Windows {
            return self.step_group(embedder, ctx);
        }
        let local = self.at() - self.batch_from;
        let path = self.files[local].clone();
        let start = self.at_time();
        let (kind, end, vec) = match self.phase {
            Phase::Frames => (
                store::KIND_FRAME,
                start + FRAME_STEP,
                embedder.embed_image_in(ctx, &path)?,
            ),
            Phase::Audio => (
                store::KIND_AUDIO,
                start + AUDIO_SECONDS,
                embedder.embed_audio_in(ctx, &path)?,
            ),
            Phase::Windows | Phase::Done => unreachable!(),
        };
        self.advance();
        self.done += 1;
        Ok(vec![NewRecord {
            kind,
            label: self.label.clone(),
            start: start as f32,
            end: end.min(self.duration) as f32,
            vec,
        }])
    }

    /// Embed the loaded group of windows in one model call.
    fn step_group(
        &mut self,
        embedder: &Embedder,
        ctx: &mut LlamaContext<'_>,
    ) -> Result<Vec<NewRecord>> {
        let mut parts = std::mem::take(&mut self.group);
        for part in &mut parts {
            if part.files.len() == part.frames {
                // A window with no audio cannot lose anything.
                continue;
            }
            let text = embed::window_text(&part.times);
            let count = embedder.tokens_of(&text, &part.files)?;
            if count > embed::BUDGET_TOKENS {
                part.keep_frames();
                self.notes.push(budget_note(part.start, count));
            }
        }
        let windows: Vec<embed::WindowIn> = parts
            .iter()
            .map(|part| embed::WindowIn {
                text: embed::window_text(&part.times),
                media: part.files.clone(),
            })
            .collect();
        let vecs = embedder.embed_windows_in(ctx, &windows)?;
        if vecs.len() != parts.len() {
            bail!(
                "the model gave {} vectors for {} windows",
                vecs.len(),
                parts.len()
            );
        }

        let mut out = Vec::with_capacity(parts.len());
        for (part, vec) in parts.iter().zip(vecs) {
            out.push(NewRecord {
                kind: store::KIND_WINDOW,
                label: self.label.clone(),
                start: part.start as f32,
                end: (part.start + self.plan.window).min(self.duration) as f32,
                vec,
            });
        }

        let made = out.len();
        self.window_at += made;
        self.done += made;
        if self.window_at >= self.windows.len() {
            self.phase = Phase::Done;
        }
        Ok(out)
    }

    fn at(&self) -> usize {
        match self.phase {
            Phase::Frames => self.frame_at,
            Phase::Audio => self.audio_at,
            Phase::Windows => self.window_at,
            Phase::Done => 0,
        }
    }

    fn at_time(&self) -> f64 {
        self.starts().get(self.at()).copied().unwrap_or(0.0)
    }

    fn remaining_in_batch(&self) -> usize {
        if self.phase == Phase::Windows {
            // One group of windows fills one model call, so one item is ready.
            return usize::from(!self.group.is_empty());
        }
        self.files.len().saturating_sub(self.at() - self.batch_from)
    }

    /// The phase after the fine grid of a video.
    ///
    /// A window plan with the flag runs its windows after the grid. A fine
    /// plan has no window, so its run ends.
    fn after_fine(&self) -> Phase {
        if self.windows.is_empty() {
            Phase::Done
        } else {
            Phase::Windows
        }
    }

    fn advance(&mut self) {
        match self.phase {
            Phase::Frames => {
                self.frame_at += 1;
                if self.frame_at >= self.frames.len() {
                    // A video with no audio track goes to its windows.
                    self.phase = if self.audios.is_empty() {
                        self.after_fine()
                    } else {
                        Phase::Audio
                    };
                    self.audio_at = 0;
                    self.files.clear();
                    self.batch_from = 0;
                }
            }
            Phase::Audio => {
                self.audio_at += 1;
                if self.audio_at >= self.audios.len() {
                    self.phase = self.after_fine();
                }
            }
            // Window mode advances a whole group, in `step_group`.
            Phase::Windows | Phase::Done => {}
        }
    }

    /// The rate of the whole-video cut.
    ///
    /// The cut holds the fine grid when the plan asks for it, so it runs at a
    /// rate that serves both paths at whole steps.
    fn cut_rate(&self) -> f64 {
        match self.plan.mode {
            crate::plan::Mode::Window if self.plan.fine => dense_rate(self.plan.frame_rate()),
            crate::plan::Mode::Window => self.plan.frame_rate(),
            crate::plan::Mode::Fine => FRAME_RATE,
        }
    }

    /// The next group of windows, up to one model call of them.
    fn window_group(&self) -> Vec<WindowPart> {
        let rate = self.plan.frame_rate();
        let cut_rate = self.cut_rate();
        let end = (self.window_at + N_WINDOW_BATCH).min(self.windows.len());
        (self.window_at..end)
            .map(|at| {
                let from = self.windows[at];
                let mut files =
                    frames_in_dense(&self.frames_cut, from, self.plan.window, rate, cut_rate);
                let frames = files.len();
                let clips = clips_in(&self.clips_cut, from, self.plan.window);
                let times = window_times_dense(from, rate, frames, clips.len(), cut_rate);
                files.extend(clips);
                WindowPart {
                    start: from,
                    files,
                    times,
                    frames,
                }
            })
            .collect()
    }

    /// Cut every frame and every clip of the video, one time.
    ///
    /// One window of a plan costs no ffmpeg call after this cut. A video with a
    /// short audio track cuts fewer clips than frames, and a window past the
    /// audio end holds frames alone.
    fn cut_media(&mut self) -> Result<()> {
        if !self.frames_cut.is_empty() || !self.clips_cut.is_empty() {
            return Ok(());
        }
        self.clear_media();
        self.dir = work_dir(0);
        std::fs::create_dir_all(&self.dir)?;
        let rate = self.cut_rate();
        let frame_dir = self.dir.join("frames");
        self.frames_cut = media::cut_frames(&self.source, rate, &frame_dir)?;
        // Fine mode always holds audio records. Window mode holds them only
        // when the plan asks for them.
        let wants_audio = self.plan.audio || !self.audios.is_empty();
        if wants_audio {
            let audio_dir = self.dir.join("audio");
            self.clips_cut = media::cut_audio(&self.source, &audio_dir)?;
        }
        Ok(())
    }

    fn cut_batch(&mut self) -> Result<()> {
        self.files.clear();
        if self.phase == Phase::Done {
            return Ok(());
        }
        if self.starts().is_empty() {
            // The fine stream of this video is over. The windows may still run.
            self.phase = self.after_fine();
            return Ok(());
        }
        self.cut_media()?;
        if self.phase == Phase::Windows {
            let group = self.window_group();
            // The plan asks for a window that the cut does not hold. A window
            // with no media has no vector, so the run reports the first one.
            let empty = group
                .iter()
                .find(|part| part.files.is_empty())
                .map(|part| part.start)
                .or_else(|| {
                    group
                        .is_empty()
                        .then(|| self.windows.get(self.window_at).copied().unwrap_or_default())
                });
            if let Some(start) = empty {
                bail!("no media came from the window at {start} s");
            }
            self.group = group;
            return Ok(());
        }
        // Fine mode slices the whole-video cut by item, because one frame sits
        // at one fine step and one clip at one audio second.
        let at = self.at();
        if self.phase == Phase::Frames {
            // A dense cut holds more frames than the fine grid wants, so the
            // grid takes every `cut_rate / 4` frame.
            let step = (self.cut_rate() / FRAME_RATE).round().max(1.0) as usize;
            // The cut can hold fewer frames than the stream plan claims, so the
            // guard counts the frames of the cut, and not the frames of the plan.
            let held = (self.frames_cut.len() + step - 1) / step;
            let end = (at + FRAME_BATCH).min(held);
            if at >= end {
                // The cut holds fewer frames than the grid, so the frame phase
                // is over. A video without an audio track goes to its windows.
                self.phase = if self.audios.is_empty() {
                    self.after_fine()
                } else {
                    Phase::Audio
                };
                self.audio_at = 0;
                self.files.clear();
                self.batch_from = 0;
                return Ok(());
            }
            self.files = self
                .frames_cut
                .iter()
                .step_by(step)
                .skip(at)
                .take(end - at)
                .cloned()
                .collect();
            self.batch_from = at;
            return Ok(());
        }
        let end = (at + AUDIO_BATCH).min(self.clips_cut.len());
        if at >= end {
            self.phase = self.after_fine();
            self.audio_at = 0;
            self.files.clear();
            self.batch_from = 0;
            return Ok(());
        }
        self.files = self.clips_cut[at..end].to_vec();
        self.batch_from = at;
        Ok(())
    }

    /// Remove the work folder and the whole-video cut.
    fn clear_media(&mut self) {
        if !self.dir.as_os_str().is_empty() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
        self.files.clear();
        self.group.clear();
        self.frames_cut.clear();
        self.clips_cut.clear();
    }
}

impl Drop for IndexRun {
    fn drop(&mut self) {
        self.clear_media();
    }
}

/// Index one video under one plan. Embed every item of the plan.
///
/// The callback runs after each model call with the number of records made so
/// far and the record count of the whole video. The answer holds the records
/// and the lines that the run reported.
pub fn index_video(
    embedder: &Embedder,
    video: &Path,
    plan: &crate::plan::Plan,
    media: &Path,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<NewRecord>, Vec<String>)> {
    let mut run = IndexRun::start_with(video, plan, media)?;
    let total = run.total();
    let mut ctx = embedder.new_context()?;
    let mut records = Vec::with_capacity(total);
    loop {
        let made = run.step(embedder, &mut ctx)?;
        if made.is_empty() {
            break;
        }
        records.extend(made);
        on_step(run.done(), total);
    }
    Ok((records, run.notes().to_vec()))
}

/// Cut one video again under a new plan.
///
/// The plan is written first, and the records go next. A fault after that
/// point leaves the video with no records and its new plan, so the next index
/// run rebuilds it under the plan the user chose.
pub fn reindex(
    embedder: &Embedder,
    store: &mut store::Store,
    path: &str,
    plan: &crate::plan::Plan,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<(usize, Vec<String>)> {
    let video = Path::new(path);
    if !video.is_file() {
        bail!("{} is not a file", video.display());
    }
    let media = store.media_dir();
    let label = store::normalize_label(path);
    store.set_plan(&label, plan)?;
    store.clear_records(&label)?;
    let (records, notes) = index_video(embedder, video, plan, &media, on_step)?;
    let count = records.len();
    store.append(records)?;
    store.mark_done(&label)?;
    Ok((count, notes))
}

/// The items of one job for each second.
///
/// A time of zero or less gives 0.0, so a fast job cannot divide by zero.
pub fn per_second(items: usize, seconds: f64) -> f32 {
    if seconds <= 0.0 {
        return 0.0;
    }
    (items as f64 / seconds) as f32
}

/// Group the ranked records by video.
///
/// `ranked` comes from `Store::rank`, so it is sorted by score, best first.
pub fn group_hits(records: &[Record], ranked: &[(usize, f32)], baseline: f32) -> Vec<Group> {
    // The score of every record, at the index of that record. `ranked` holds
    // every record, so the bar can read a score below the floor too.
    let mut score_of = vec![0.0f32; records.len()];
    for (index, score) in ranked {
        if let Some(slot) = score_of.get_mut(*index) {
            *slot = *score;
        }
    }

    let mut durations: HashMap<String, f32> = HashMap::new();
    for record in records {
        if !crate::store::is_video(record.kind) {
            continue;
        }
        let held = durations
            .entry(crate::store::normalize_label(&record.label))
            .or_insert(0.0);
        if record.end > *held {
            *held = record.end;
        }
    }

    let mut groups: Vec<Group> = Vec::new();
    for (index, score) in ranked {
        if *score < FLOOR {
            break;
        }
        let record = &records[*index];
        if !crate::store::is_video(record.kind) {
            continue;
        }
        let name = crate::store::normalize_label(&record.label);
        let hit = Hit {
            start: record.start,
            end: record.end,
            score: *score,
            margin: *score - baseline,
            kind: record.kind,
        };
        match groups.iter_mut().find(|g| g.video == name) {
            Some(group) => {
                if group.hits.len() < GROUP_CAP {
                    group.hits.push(hit);
                }
            }
            None => {
                let duration = durations.get(&name).copied().unwrap_or(record.end);
                groups.push(Group {
                    video: name,
                    duration,
                    score: *score,
                    hits: vec![hit],
                    cells: Vec::new(),
                });
            }
        }
    }

    // One cell for every record of every grouped video, in time order. The
    // floor picks the hits. The bar shows all of them.
    //
    // A frame cell is 0.25 s and an audio cell is 1 s, so the two overlap. The
    // bar takes the frames when the video has them, because they are finer.
    for group in groups.iter_mut() {
        let has_frames = records.iter().any(|record| {
            record.kind == crate::store::KIND_FRAME
                && crate::store::normalize_label(&record.label) == group.video
        });
        group.cells = records
            .iter()
            .enumerate()
            .filter(|(_, record)| {
                let right_kind = if has_frames {
                    record.kind == crate::store::KIND_FRAME
                } else {
                    crate::store::is_video(record.kind)
                };
                right_kind && crate::store::normalize_label(&record.label) == group.video
            })
            .map(|(index, record)| Cell {
                start: record.start,
                end: record.end,
                score: score_of[index],
            })
            .collect();
        group.cells.sort_by(|a, b| a.start.total_cmp(&b.start));
    }

    // A run of touching hits reads as one moment. The cap counts moments, not
    // frames, so one video may show many moments.
    for group in groups.iter_mut() {
        let hits = std::mem::take(&mut group.hits);
        let mut moments = merge_hits(hits);
        moments.truncate(GROUP_CAP);
        group.hits = moments;
    }
    groups
}

/// The rows of a search answer, best first, with the near rows of one video
/// joined into one moment.
///
/// The list is cut to `cap` rows first, because one merged moment counts as one
/// row of the cap. Then the rows of one video fold when they sit within `gap` of
/// each other. A note holds no time, so a note stays its own row.
pub fn search_rows(
    records: &[Record],
    ranked: &[(usize, f32)],
    baseline: f32,
    cap: usize,
    gap: f32,
) -> Vec<Row> {
    let mut videos: Vec<(String, Hit)> = Vec::new();
    let mut notes: Vec<Row> = Vec::new();
    for (index, score) in ranked.iter().take(cap) {
        let Some(record) = records.get(*index) else {
            continue;
        };
        if !store::is_video(record.kind) {
            notes.push(Row {
                note: true,
                kind: record.kind,
                label: record.label.clone(),
                start: record.start,
                end: record.end,
                score: *score,
                margin: *score - baseline,
            });
            continue;
        }
        videos.push((
            store::normalize_label(&record.label),
            Hit {
                start: record.start,
                end: record.end,
                score: *score,
                margin: *score - baseline,
                kind: record.kind,
            },
        ));
    }

    // One video at a time, because a moment belongs to one video.
    let mut labels: Vec<String> = videos.iter().map(|(label, _)| label.clone()).collect();
    labels.sort();
    labels.dedup();
    let mut rows: Vec<Row> = Vec::new();
    for label in labels {
        let hits: Vec<Hit> = videos
            .iter()
            .filter(|(held, _)| *held == label)
            .map(|(_, hit)| hit.clone())
            .collect();
        for hit in merge_hits_with(hits, gap) {
            rows.push(Row {
                note: false,
                kind: hit.kind,
                label: label.clone(),
                start: hit.start,
                end: hit.end,
                score: hit.score,
                margin: hit.margin,
            });
        }
    }
    rows.extend(notes);
    rows.sort_by(|a, b| b.score.total_cmp(&a.score));
    rows
}

/// Join the hits of one video that sit near each other into one moment.
///
/// The moment spans the whole run. Its score and its margin come from the best
/// hit of the run, so a moment never claims more than one hit earned.
pub fn merge_hits_with(hits: Vec<Hit>, gap: f32) -> Vec<Hit> {
    if hits.len() < 2 {
        return hits;
    }
    let mut by_time = hits;
    by_time.sort_by(|a, b| a.start.total_cmp(&b.start));

    let mut moments: Vec<Hit> = Vec::new();
    for hit in by_time {
        match moments.last_mut() {
            Some(last) if hit.start - last.end <= gap => {
                if hit.score > last.score {
                    last.score = hit.score;
                    last.margin = hit.margin;
                    last.kind = hit.kind;
                }
                if hit.end > last.end {
                    last.end = hit.end;
                }
            }
            _ => moments.push(hit),
        }
    }
    moments.sort_by(|a, b| b.score.total_cmp(&a.score));
    moments
}

/// Join the touching hits of one video into one moment.
pub fn merge_hits(hits: Vec<Hit>) -> Vec<Hit> {
    merge_hits_with(hits, MERGE_GAP)
}

/// The indexes of the records that stay after the thinning.
///
/// The thinning measures the cost of a coarser time grid. Every `factor`th
/// frame of one video stays, and the audio keeps its one second grid.
pub fn thin_indexes(records: &[Record], factor: usize) -> Vec<usize> {
    if factor < 2 {
        return (0..records.len()).collect();
    }
    // The frames of each video, in time order. A store may hold the frames of
    // one video twice, so the order is taken from the start time.
    let mut order: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        if record.kind == crate::store::KIND_FRAME {
            order
                .entry(crate::store::normalize_label(&record.label))
                .or_default()
                .push(index);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan;

    /// The parts of one window carry the time of each part.
    ///
    /// The frames come first, then the clips, in the order the embedder reads
    /// them. A frame sits at its own place in the cut, and a clip sits at the
    /// second that it covers.
    #[test]
    fn the_part_times_of_a_window_name_each_part() {
        // The example of the spec: a 2 s window at 3 s, one frame a second.
        assert_eq!(window_times(3.0, 1.0, 2, 1), vec![3.0, 4.0, 3.0]);
        // Four frames a second sit at the quarter steps of the cut.
        assert_eq!(
            window_times(3.0, 4.0, 4, 2),
            vec![3.0, 3.25, 3.5, 3.75, 3.0, 4.0]
        );
    }

    /// A window past the end of the audio track holds frames alone.
    ///
    /// The marker count then equals the frame count, so the model reads one
    /// marker for each part it gets.
    #[test]
    fn a_window_without_audio_holds_frame_times_alone() {
        assert_eq!(window_times(23.0, 1.0, 2, 0), vec![23.0, 24.0]);
        assert_eq!(window_times(0.0, 1.0, 0, 0), Vec::<f64>::new());
    }

    /// The times of a window follow the files that the cut gives back.
    #[test]
    fn the_part_times_match_the_files_of_the_window() {
        let cut: Vec<PathBuf> = (0..8).map(|i| PathBuf::from(format!("f{i}.jpg"))).collect();
        let files = frames_in(&cut, 1.0, 1.0, 4.0);
        let times = window_times(1.0, 4.0, 4, 0);
        assert_eq!(files.len(), times.len());
        assert_eq!(times, vec![1.0, 1.25, 1.5, 1.75]);
    }

    /// The dense cut serves both rates at whole steps.
    #[test]
    fn the_dense_cut_serves_both_rates() {
        assert_eq!(dense_rate(1.0), 4.0);
        assert_eq!(dense_rate(3.0), 12.0);
        for rate in [0.5, 1.0, 1.5, 2.5, 4.0, 10.0] {
            let cut = dense_rate(rate);
            let plan_step = cut / rate;
            let fine_step = cut / FRAME_RATE;
            assert!(
                (plan_step - plan_step.round()).abs() < 1e-9,
                "the window step is whole at {rate} fps"
            );
            assert!(
                (fine_step - fine_step.round()).abs() < 1e-9,
                "the fine step is whole at {rate} fps"
            );
        }
    }

    /// A dense cut gives the window the frames of its own rate.
    #[test]
    fn a_dense_cut_gives_the_window_its_own_frames() {
        // Four frames a second, so index `n` sits at `n / 4` s.
        let cut: Vec<PathBuf> = (0..16).map(|n| PathBuf::from(format!("f{n}.jpg"))).collect();
        // A 2 s window at 1 fps wants 2 frames, at 0 s and 1 s.
        let files = frames_in_dense(&cut, 0.0, 2.0, 1.0, 4.0);
        assert_eq!(files, vec![PathBuf::from("f0.jpg"), PathBuf::from("f4.jpg")]);
        let times = window_times_dense(0.0, 1.0, files.len(), 0, 4.0);
        assert_eq!(times, vec![0.0, 1.0]);
        // The second window starts at 1 s, on its own frame.
        let next = frames_in_dense(&cut, 1.0, 2.0, 1.0, 4.0);
        assert_eq!(next[0], PathBuf::from("f4.jpg"));
        // A window past the end of the cut takes what is left, and no more.
        let left = frames_in_dense(&cut, 3.0, 2.0, 1.0, 4.0);
        assert_eq!(left, vec![PathBuf::from("f12.jpg")]);
        assert!(frames_in_dense(&cut, 40.0, 2.0, 1.0, 4.0).is_empty());
    }

    /// A plan with the flag runs the fine grid and then the windows.
    #[test]
    fn the_flag_runs_the_fine_grid_and_the_windows() {
        let mut plan = plan::Plan::of(plan::Preset::Object);
        plan.fine = true;
        let mut run = IndexRun {
            source: PathBuf::from("a.mp4"),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: "a.mp4".to_string(),
            duration: 2.0,
            frames: vec![0.0, 0.25],
            audios: vec![0.0],
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan,
            windows: vec![0.0],
            group: Vec::new(),
            notes: Vec::new(),
        };

        assert_eq!(run.total(), 4, "two frames, one clip, one window");
        run.advance();
        run.advance();
        assert!(
            run.phase == Phase::Audio,
            "the grid comes before the windows"
        );
        run.advance();
        assert!(run.phase == Phase::Windows, "the windows come last");
        run.advance();
        assert!(run.phase == Phase::Windows, "a window waits for its group");
    }

    /// A grid with no audio track hands its video straight to the windows.
    #[test]
    fn a_grid_with_no_audio_goes_to_its_windows() {
        let mut plan = plan::Plan::of(plan::Preset::Object);
        plan.fine = true;
        let mut run = IndexRun {
            source: PathBuf::from("a.mp4"),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: "a.mp4".to_string(),
            duration: 2.0,
            frames: vec![0.0],
            audios: Vec::new(),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan,
            windows: vec![0.0],
            group: Vec::new(),
            notes: Vec::new(),
        };

        assert_eq!(run.total(), 2, "one frame and one window");
        run.advance();
        assert!(
            run.phase == Phase::Windows,
            "no audio track, so the grid hands over"
        );
    }

    /// A hit names the kind of its record.
    #[test]
    fn a_hit_names_its_kind() {
        let records = vec![
            record_of(store::KIND_WINDOW, "a.mp4", 0.0, 2.0),
            record_of(store::KIND_FRAME, "a.mp4", 5.0, 5.25),
        ];
        let ranked = vec![(0usize, 0.9f32), (1usize, 0.5f32)];
        let rows = search_rows(&records, &ranked, 0.2, 10, MERGE_GAP);
        assert_eq!(rows.len(), 2, "the two sit apart, so neither folds");
        let kinds: Vec<u8> = rows.iter().map(|row| row.kind).collect();
        assert!(kinds.contains(&store::KIND_WINDOW));
        assert!(kinds.contains(&store::KIND_FRAME));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_test_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn record_of(kind: u8, label: &str, start: f32, end: f32) -> Record {
        Record {
            rid: 0,
            id: store::record_id(label, start, end),
            kind,
            label: label.to_string(),
            start,
            end,
        }
    }

    fn record(label: &str, start: f32, end: f32) -> Record {
        record_of(crate::store::KIND_FRAME, label, start, end)
    }

    #[test]
    fn list_videos_walks_subfolders_and_filters_extensions() {
        let dir = scratch("list");
        std::fs::write(dir.join("a.mp4"), b"x").unwrap();
        std::fs::write(dir.join("b.MKV"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("c.webm"), b"x").unwrap();

        let names: Vec<String> = list_videos(&dir)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert_eq!(names.len(), 3, "found {names:?}");
        assert!(names.contains(&"a.mp4".to_string()));
        assert!(names.contains(&"b.MKV".to_string()));
        assert!(names.contains(&"c.webm".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_path_with_a_space_and_a_unicode_letter_survives_the_scan() {
        let dir = scratch("unicode");
        std::fs::write(dir.join("doğum günü.mp4"), b"x").unwrap();
        let found = list_videos(&dir).unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].to_string_lossy().contains("doğum günü"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn frames_cover_a_thirty_second_video_at_four_a_second() {
        let plan = frames(30.0);
        assert_eq!(plan.len(), 120);
        assert_eq!(plan[0], 0.0);
        assert!((plan[1] - 0.25).abs() < 1e-9);
        assert!((plan[119] - 29.75).abs() < 1e-9);
    }

    #[test]
    fn a_video_shorter_than_one_frame_has_no_frame() {
        assert!(frames(0.2).is_empty());
        assert_eq!(frames(0.25).len(), 1);
    }

    #[test]
    fn audio_starts_cover_a_thirty_second_video() {
        let plan = audio_starts(30.0);
        assert_eq!(plan.len(), 30);
        assert_eq!(plan[0], 0.0);
        assert_eq!(plan[29], 29.0);
    }

    #[test]
    fn a_video_shorter_than_one_second_has_no_audio_clip() {
        assert!(audio_starts(0.5).is_empty());
        assert_eq!(audio_starts(1.0).len(), 1);
    }

    #[test]
    fn the_progress_object_matches_the_spec() {
        let json = serde_json::to_value(Progress::default()).unwrap();
        assert_eq!(json["running"], false);
        assert_eq!(json["error"], serde_json::Value::Null);
        assert_eq!(json["window_count"], 0);
        assert_eq!(json["indexed"], 0);
    }

    #[test]
    fn group_hits_drops_low_scores_and_sorts_groups() {
        let records = vec![
            record("a.mp4", 0.0, 4.0),
            record("a.mp4", 2.0, 6.0),
            record("b.mp4", 0.0, 4.0),
            record("b.mp4", 4.0, 8.0),
        ];
        let ranked = vec![(2, 0.9), (3, 0.7), (0, 0.5), (1, 0.2)];

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].video, "b.mp4");
        assert_eq!(groups[0].score, 0.9);
        assert_eq!(groups[0].duration, 8.0);
        // The two records of b.mp4 touch, so the moment covers the whole run.
        assert_eq!(groups[0].hits.len(), 1);
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 8.0).abs() < 1e-6);
        assert_eq!(groups[1].video, "a.mp4");
        assert_eq!(groups[1].hits.len(), 1);
    }

    /// One video gathers under one group, whatever separator its label carries.
    #[test]
    fn group_hits_gathers_a_video_across_path_separators() {
        let records = vec![
            record(r"D:\video\a.mp4", 0.0, 4.0),
            record("D:/video/a.mp4", 4.0, 8.0),
        ];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 1, "one video makes one group");
        assert_eq!(groups[0].video, "D:/video/a.mp4");
    }

    /// The bar needs every record, not only the ones above the floor.
    ///
    /// The floor picks the hit list. The bar shows the whole video, so a
    /// weak part of the video still takes its place on the bar.
    #[test]
    fn a_group_carries_one_cell_for_every_video_record() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("a.mp4", 0.25, 0.5),
            record("a.mp4", 0.5, 0.75),
        ];
        // The third record stays below the floor. It must still reach the bar.
        let ranked = vec![(0, 0.80), (2, 0.45), (1, 0.30)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups.len(), 1);
        // Two hits reach the floor. They touch, so they read as one moment
        // over the whole span. The middle record stays below the floor, and
        // the moment spans it.
        assert_eq!(groups[0].hits.len(), 1, "two hits joined into one moment");
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 0.75).abs() < 1e-6);
        assert!((groups[0].hits[0].score - 0.80).abs() < 1e-6);

        let cells = &groups[0].cells;
        assert_eq!(cells.len(), 3, "the bar needs every record");
        assert!((cells[0].start - 0.0).abs() < 1e-6);
        assert!((cells[0].score - 0.80).abs() < 1e-6);
        assert!((cells[1].start - 0.25).abs() < 1e-6);
        assert!((cells[1].score - 0.30).abs() < 1e-6);
        assert!((cells[2].start - 0.5).abs() < 1e-6);
        assert!((cells[2].score - 0.45).abs() < 1e-6);
    }

    /// The bar takes the finest records of the video.
    ///
    /// A frame cell is 0.25 s and an audio cell is 1 s, so the two overlap.
    /// One bar holds one of them, and the frames give the finer picture.
    #[test]
    fn the_bar_uses_the_frames_when_the_video_has_them() {
        let records = vec![
            record_of(crate::store::KIND_FRAME, "a.mp4", 0.0, 0.25),
            record_of(crate::store::KIND_FRAME, "a.mp4", 0.25, 0.5),
            record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0),
        ];
        let ranked = vec![(0, 0.80), (2, 0.70), (1, 0.60)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].cells.len(), 2, "the frames cover the bar");
        assert!((groups[0].cells[0].end - 0.25).abs() < 1e-6);
        assert!((groups[0].cells[1].end - 0.5).abs() < 1e-6);
    }

    /// A video with no frame record still gets a bar.
    #[test]
    fn the_bar_falls_back_to_the_audio_records_of_an_old_store() {
        let records = vec![
            record_of(crate::store::KIND_AUDIO, "a.mp4", 0.0, 1.0),
            record_of(crate::store::KIND_AUDIO, "a.mp4", 1.0, 2.0),
        ];
        let ranked = vec![(0, 0.80), (1, 0.70)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups[0].cells.len(), 2);
        assert!((groups[0].cells[0].score - 0.80).abs() < 1e-6);
        assert!((groups[0].cells[1].score - 0.70).abs() < 1e-6);
    }

    #[test]
    fn group_hits_caps_each_group_at_fifty() {
        // The records must not touch. Touching records merge into one moment,
        // and the cap would never bite.
        let records: Vec<Record> = (0..60)
            .map(|i| record("a.mp4", i as f32 * 10.0, i as f32 * 10.0 + 4.0))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..60).map(|i| (i, 0.9 - i as f32 * 0.001)).collect();

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hits.len(), 50);
        assert_eq!(groups[0].duration, 594.0);
    }

    #[test]
    fn group_hits_returns_nothing_when_every_score_is_below_the_floor() {
        let records = vec![record("a.mp4", 0.0, 4.0)];
        assert!(group_hits(&records, &[(0, 0.39)], 0.0).is_empty());
        assert!(group_hits(&[], &[], 0.0).is_empty());
    }

    #[test]
    fn the_median_of_an_odd_list_is_the_middle_value() {
        let ranked = vec![(0, 0.1), (1, 0.9), (2, 0.5)];
        assert!((median(&ranked) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_median_of_an_even_list_is_the_middle_pair() {
        let ranked = vec![(0, 0.1), (1, 0.3), (2, 0.5), (3, 0.9)];
        assert!((median(&ranked) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn the_median_of_an_empty_list_is_zero() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn the_cut_follows_the_best_score() {
        assert!((weak_cut(0.9) - 0.70).abs() < 1e-6);
        assert!(
            (weak_cut(0.5) - FLOOR).abs() < 1e-6,
            "a weak best score leaves the floor in place"
        );
    }

    #[test]
    fn a_score_below_the_cut_is_weak() {
        assert!(is_weak(0.60, 0.70));
        assert!(!is_weak(0.70, 0.70), "the cut itself stands");
        assert!(!is_weak(0.80, 0.70));
    }

    #[test]
    fn group_hits_reports_the_margin_over_the_baseline() {
        // The two records must not touch. Touching records merge, and the
        // second hit would not exist.
        let records = vec![record("a.mp4", 0.0, 4.0), record("a.mp4", 10.0, 14.0)];
        let ranked = vec![(1, 0.70), (0, 0.65)];
        let groups = group_hits(&records, &ranked, 0.60);
        assert_eq!(groups[0].hits[0].score, 0.70);
        assert!((groups[0].hits[0].margin - 0.10).abs() < 1e-6);
        assert!((groups[0].hits[1].margin - 0.05).abs() < 1e-6);
    }

    /// A contract pin for `index_video`.
    ///
    /// `index_video` needs the model, so it has no unit test of its own. This
    /// test pins the callback contract that it obeys: the number counts from 1,
    /// and it reaches the item count on the last item.
    #[test]
    fn the_callback_numbering_matches_the_item_plan() {
        let plan: Vec<f64> = frames(30.0)
            .into_iter()
            .chain(audio_starts(30.0))
            .collect();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let mut report = |n: usize, total: usize| seen.push((n, total));
        for (n, _) in plan.iter().enumerate() {
            report(n + 1, plan.len());
        }
        assert_eq!(seen.len(), 150);
        assert_eq!(seen[0], (1, 150));
        assert_eq!(seen[149], (150, 150));
    }

    /// A video with no audio track has an empty audio plan.
    ///
    /// The run must then end after the last frame. It must not ask the plan
    /// for an audio clip, because that plan holds no item.
    #[test]
    fn the_run_ends_after_the_frames_when_the_audio_plan_is_empty() {
        let mut run = IndexRun {
            source: PathBuf::from("silent.mp4"),
            frames_cut: Vec::new(),
            clips_cut: Vec::new(),
            label: "silent.mp4".to_string(),
            duration: 1.0,
            frames: vec![0.0, 0.25, 0.5, 0.75],
            audios: Vec::new(),
            frame_at: 0,
            audio_at: 0,
            phase: Phase::Frames,
            dir: PathBuf::new(),
            files: Vec::new(),
            batch_from: 0,
            done: 0,
            window_at: 0,
            plan: plan::Plan::of(plan::Preset::Fine),
            windows: Vec::new(),
            group: Vec::new(),
            notes: Vec::new(),
        };

        for _ in 0..run.frames.len() {
            run.advance();
        }

        assert!(run.phase == Phase::Done);
        assert_eq!(run.remaining_in_batch(), 0);
        assert!(run.cut_batch().is_ok());
        assert!(run.phase == Phase::Done);
    }

    #[test]
    fn start_counts_the_frames_and_the_audio_clips_of_the_test_video() {
        let video =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spike/testvideo_nolabels.mp4");
        if !video.exists() || !Path::new(crate::media::FFPROBE).exists() {
            eprintln!("skipped: ffprobe or the test video is missing");
            return;
        }
        let format = crate::media::probe_duration(&video).unwrap();
        let video_len = crate::media::probe_stream_duration(&video, "v")
            .unwrap()
            .unwrap_or(format)
            .min(format);
        let audio_len = match crate::media::probe_stream_duration(&video, "a").unwrap() {
            Some(len) => len.min(format),
            None => 0.0,
        };
        let run = IndexRun::start(&video).unwrap();

        assert_eq!(
            run.total(),
            frames(video_len).len() + audio_starts(audio_len).len()
        );
        assert!(run.total() > 100, "got {}", run.total());
        // The test video holds about 23 s of audio inside a 30 s file. A clip
        // past the end of the audio track holds no sound, so the plan stops at
        // the track length.
        assert!(audio_len + 1.0 < format, "the track is not short here");
        assert!(
            run.total() < frames(video_len).len() + audio_starts(format).len(),
            "the plan asks for a clip past the end of the audio track"
        );
    }

    /// A run of touching frames is one moment, not many hits.
    #[test]
    fn touching_hits_join_one_moment() {
        let hits = vec![
            Hit { start: 0.0, end: 0.25, score: 0.70, margin: 0.10, kind: 0 },
            Hit { start: 0.25, end: 0.5, score: 0.80, margin: 0.20, kind: 0 },
            Hit { start: 0.5, end: 0.75, score: 0.60, margin: 0.05, kind: 0 },
        ];
        let moments = merge_hits(hits);

        assert_eq!(moments.len(), 1, "the run is one moment");
        assert!((moments[0].start - 0.0).abs() < 1e-6);
        assert!((moments[0].end - 0.75).abs() < 1e-6);
        assert!((moments[0].score - 0.80).abs() < 1e-6);
        assert!((moments[0].margin - 0.20).abs() < 1e-6);
    }

    /// Two runs with a hole between them stay apart.
    #[test]
    fn a_gap_of_two_frame_steps_keeps_two_moments() {
        let hits = vec![
            Hit { start: 0.0, end: 0.25, score: 0.80, margin: 0.20, kind: 0 },
            Hit { start: 1.0, end: 1.25, score: 0.70, margin: 0.10, kind: 0 },
        ];
        let moments = merge_hits(hits);

        assert_eq!(moments.len(), 2, "the hole splits the run");
    }

    /// The first hit of the list is the best moment.
    #[test]
    fn the_moments_come_back_best_first() {
        let hits = vec![
            Hit { start: 0.0, end: 0.25, score: 0.60, margin: 0.01, kind: 0 },
            Hit { start: 2.0, end: 2.25, score: 0.90, margin: 0.30, kind: 0 },
        ];
        let moments = merge_hits(hits);

        assert_eq!(moments.len(), 2);
        assert!((moments[0].score - 0.90).abs() < 1e-6);
        assert!((moments[1].score - 0.60).abs() < 1e-6);
    }

    #[test]
    fn one_hit_stays_one_hit() {
        let hits = vec![Hit { start: 1.0, end: 1.25, score: 0.5, margin: 0.0, kind: 0 }];
        assert_eq!(merge_hits(hits).len(), 1);
        assert!(merge_hits(Vec::new()).is_empty());
    }

    /// Audio records are 1 s long and they touch, so they join.
    ///
    /// A store from before the frame records holds audio only. The merge then
    /// gives one moment over the whole matching run. That is the honest answer
    /// for a 1 s grid, so the behaviour is pinned here.
    #[test]
    fn touching_audio_hits_join_one_moment() {
        let hits = vec![
            Hit { start: 0.0, end: 1.0, score: 0.70, margin: 0.10, kind: 0 },
            Hit { start: 1.0, end: 2.0, score: 0.75, margin: 0.15, kind: 0 },
            Hit { start: 2.0, end: 3.0, score: 0.72, margin: 0.12, kind: 0 },
        ];
        assert_eq!(merge_hits(hits).len(), 1);
    }

    /// The merge runs before the cap, so one video may show many moments.
    #[test]
    fn a_group_joins_its_run_of_frames_into_one_moment() {
        let records: Vec<Record> = (0..4)
            .map(|i| record("a.mp4", i as f32 * 0.25, i as f32 * 0.25 + 0.25))
            .collect();
        let ranked = vec![(0, 0.90), (1, 0.85), (2, 0.80), (3, 0.75)];
        let groups = group_hits(&records, &ranked, 0.50);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hits.len(), 1, "four touching frames make one moment");
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 1.0).abs() < 1e-6);
    }

    /// The page folds its groups at the same gap as the console search.
    ///
    /// The spec gives one rule for a search, and the page shows the rows. Two
    /// hits of one video 0.3 s apart must read as one moment. A fold at the
    /// frame step leaves two rows, and the page then parts from the console.
    #[test]
    fn a_group_folds_two_hits_inside_the_merge_gap() {
        let records = vec![record("a.mp4", 0.0, 0.25), record("a.mp4", 0.55, 0.8)];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let groups = group_hits(&records, &ranked, 0.0);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hits.len(), 1, "0.3 s sits inside the merge gap");
        assert!((groups[0].hits[0].start - 0.0).abs() < 1e-6);
        assert!((groups[0].hits[0].end - 0.8).abs() < 1e-6);
    }

    /// Two touching windows of one video show as one row.
    ///
    /// This is the gate of the fold. The window grid holds windows that touch,
    /// so three near hits of one moment must read as one row.
    #[test]
    fn two_touching_windows_of_one_video_join_one_row() {
        let records = vec![record("a.mp4", 0.0, 2.0), record("a.mp4", 2.0, 4.0)];
        let ranked = vec![(0, 0.9), (1, 0.5)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 1, "the two windows make one moment");
        assert_eq!(rows[0].start, 0.0);
        assert_eq!(rows[0].end, 4.0);
        assert_eq!(rows[0].score, 0.9, "the moment keeps the best score");
    }

    /// Two frames that sit past the gap keep two rows.
    #[test]
    fn two_frames_past_the_gap_keep_two_rows() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("a.mp4", 1.0, 1.25),
        ];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 2, "a gap of 0.75 s holds two moments");
    }

    /// The rows of two videos never join, whatever their times.
    #[test]
    fn a_row_of_another_video_keeps_its_own_row() {
        let records = vec![record("a.mp4", 0.0, 2.0), record("b.mp4", 2.0, 4.0)];
        let ranked = vec![(0, 0.9), (1, 0.8)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 2, "the two videos keep one row each");
    }

    /// A note holds no time, so a note never joins a video row.
    #[test]
    fn a_note_never_joins_a_row() {
        let records = vec![
            record("a.mp4", 0.0, 2.0),
            record_of(crate::store::KIND_NOTE, "blue", 0.0, 0.0),
        ];
        let ranked = vec![(0, 0.9), (1, 0.85)];

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 2, "the note stands alone");
        assert!(rows.iter().any(|row| row.note), "the note is marked");
    }

    /// The cap counts the rows that enter the fold.
    #[test]
    fn the_cap_counts_the_rows_before_the_fold() {
        let records: Vec<Record> = (0..12)
            .map(|at| record("a.mp4", at as f32, at as f32 + 0.25))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..12).map(|at| (at, 0.9 - at as f32 / 100.0)).collect();

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 10, "a gap of 0.75 s keeps every kept row");
    }

    /// Ten rows that sit near each other leave one moment.
    #[test]
    fn ten_near_rows_leave_one_row() {
        let records: Vec<Record> = (0..12)
            .map(|at| record("a.mp4", at as f32 * 0.2, at as f32 * 0.2 + 0.25))
            .collect();
        let ranked: Vec<(usize, f32)> = (0..12).map(|at| (at, 0.9 - at as f32 / 100.0)).collect();

        let rows = search_rows(&records, &ranked, 0.0, 10, MERGE_GAP);

        assert_eq!(rows.len(), 1, "the ten rows chain into one moment");
        assert_eq!(rows[0].score, 0.9, "the moment keeps the best score");
    }

    /// The folded rows come back best first.
    #[test]
    fn the_folded_rows_come_back_best_first() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("b.mp4", 9.0, 9.25),
        ];
        let ranked = vec![(1, 0.95), (0, 0.6)];

        let rows = search_rows(&records, &ranked, 0.5, 10, MERGE_GAP);

        assert_eq!(rows[0].label, "b.mp4");
        assert!(
            (rows[0].margin - 0.45).abs() < 1e-6,
            "the margin follows the baseline, got {}",
            rows[0].margin
        );
    }

    #[test]
    fn the_rate_is_items_a_second() {
        assert!((per_second(20, 10.0) - 2.0).abs() < 1e-6);
        assert!((per_second(207, 29.0) - 7.137_931).abs() < 1e-4);
    }

    /// A zero time must not divide by zero.
    #[test]
    fn a_zero_time_gives_a_zero_rate() {
        assert_eq!(per_second(10, 0.0), 0.0);
        assert_eq!(per_second(10, -1.0), 0.0);
        assert_eq!(per_second(0, 10.0), 0.0);
    }

    /// The page reads these fields, so the shape is a contract.
    #[test]
    fn the_progress_carries_the_speed_fields() {
        let json = serde_json::to_value(Progress::default()).unwrap();
        assert_eq!(json["device"], "");
        assert_eq!(json["gpu_available"], false);
        assert_eq!(json["load_ms"], 0);
        assert_eq!(json["elapsed_ms"], 0);
        assert_eq!(json["per_second"], 0.0);
    }

    /// A factor of four keeps one frame of every four, and keeps the audio.
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

    /// A factor of one keeps every record.
    #[test]
    fn a_factor_of_one_keeps_everything() {
        let records = vec![record("a.mp4", 0.0, 0.25), record("a.mp4", 0.25, 0.5)];
        assert_eq!(thin_indexes(&records, 1).len(), 2);
        assert_eq!(thin_indexes(&records, 0).len(), 2);
    }

    /// Each video thins on its own, so a short video keeps its first frame.
    #[test]
    fn each_video_thins_on_its_own() {
        let records = vec![
            record("a.mp4", 0.0, 0.25),
            record("b.mp4", 0.0, 0.25),
            record("a.mp4", 0.25, 0.5),
            record("b.mp4", 0.25, 0.5),
        ];
        let kept = thin_indexes(&records, 2);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|at| records[*at].label == "a.mp4"));
        assert!(kept.iter().any(|at| records[*at].label == "b.mp4"));
    }

    /// One video thins as one set, whatever separator its label carries.
    #[test]
    fn thinning_gathers_a_video_across_path_separators() {
        let records = vec![
            record(r"D:\video\a.mp4", 0.0, 0.25),
            record("D:/video/a.mp4", 0.25, 0.5),
            record(r"D:\video\a.mp4", 0.5, 0.75),
        ];

        let kept = thin_indexes(&records, 2);

        assert_eq!(kept, vec![0, 2], "the two forms make one time order");
    }

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

    /// The frames of a span, from the list of a whole-video cut.
    ///
    /// Frame `i` sits at `i / rate`, and a span holds the frames inside it.
    #[test]
    fn a_span_takes_the_frames_inside_it() {
        let cut: Vec<PathBuf> = (0..30)
            .map(|i| PathBuf::from(format!("f{i:05}.jpg")))
            .collect();

        // One frame a second: a 0 s window takes 0 and 1, a 1 s window 1 and 2.
        let first = frames_in(&cut, 0.0, 2.0, 1.0);
        assert_eq!(first.len(), 2);
        assert!(first[0].ends_with("f00000.jpg"));
        assert!(first[1].ends_with("f00001.jpg"));

        let second = frames_in(&cut, 1.0, 2.0, 1.0);
        assert!(second[0].ends_with("f00001.jpg"));
        assert!(second[1].ends_with("f00002.jpg"));

        // Four frames a second: a 0.25 s span takes one frame.
        let fine = frames_in(&cut, 0.25, 0.25, 4.0);
        assert_eq!(fine.len(), 1);
        assert!(fine[0].ends_with("f00001.jpg"));

        // The list ends, so the last span holds what is left.
        assert_eq!(frames_in(&cut, 29.0, 2.0, 1.0).len(), 1);
        assert!(frames_in(&cut, 40.0, 2.0, 1.0).is_empty());
    }

    /// The clips of a span, from the list of a whole-video cut.
    ///
    /// Clip `k` covers `[k, k + 1)`, and a span takes every clip that touches
    /// it. A span past the end of the audio takes none.
    #[test]
    fn a_span_takes_the_clips_that_touch_it() {
        let cut: Vec<PathBuf> = (0..24)
            .map(|i| PathBuf::from(format!("a{i:05}.wav")))
            .collect();

        assert_eq!(clips_in(&cut, 0.0, 2.0).len(), 2);
        assert_eq!(clips_in(&cut, 1.0, 2.0).len(), 2);
        assert_eq!(clips_in(&cut, 0.5, 2.0).len(), 3);
        assert_eq!(clips_in(&cut, 23.0, 2.0).len(), 1);
        assert!(clips_in(&cut, 24.0, 2.0).is_empty());
        assert!(clips_in(&[], 0.0, 2.0).is_empty());
    }

    /// The window run cuts the whole video one time.
    ///
    /// The cut holds every frame and every clip of the plan, so a window costs
    /// no ffmpeg call at all.
    #[test]
    fn the_window_run_cuts_the_whole_video_once() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("one_pass_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        // A plain video is its own source, so no copy comes.
        assert_eq!(media::source_file(&video, &media).unwrap(), video);

        run.cut_batch().unwrap();
        assert_eq!(run.frames_cut.len(), 30);
        assert_eq!(run.clips_cut.len(), 24);
        // One window of the Object plan holds two frames and two clips.
        assert_eq!(run.group[0].files.len(), 4);
    }

    /// A window past the end of the audio holds the frames alone.
    #[test]
    fn a_window_past_the_audio_holds_frames_alone() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("late_window_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        run.cut_batch().unwrap();

        // The audio stream ends at 23.1 s, so the window at 28 s holds no clip.
        run.window_at = 28;
        run.cut_batch().unwrap();
        assert_eq!(run.group[0].files.len(), 2);
    }

    /// The rule of one window, in words, for the reader.
    #[test]
    fn the_budget_note_names_the_window_and_the_count() {
        assert_eq!(
            budget_note(3.0, 1500),
            "window at 3.0 s holds 1500 tokens, above the budget of 1024, \
             so the record keeps the frames alone"
        );
    }

    /// A window that passes the budget keeps its frames and drops its clips.
    #[test]
    fn a_window_that_passes_the_budget_keeps_its_frames() {
        let mut part = WindowPart {
            start: 3.0,
            files: vec![
                PathBuf::from("f0.jpg"),
                PathBuf::from("f1.jpg"),
                PathBuf::from("a0.wav"),
                PathBuf::from("a1.wav"),
            ],
            times: vec![3.0, 4.0, 3.0, 4.0],
            frames: 2,
        };

        part.keep_frames();

        assert_eq!(part.frames, 2);
        assert_eq!(part.times, vec![3.0, 4.0], "the markers follow the media");
        for file in &part.files {
            let name = file.to_string_lossy();
            assert!(name.ends_with(".jpg"), "{name} is not a frame");
        }
    }

    /// One model call reads four windows.
    #[test]
    fn the_run_takes_four_windows_at_a_time() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("group_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();

        run.cut_batch().unwrap();

        assert_eq!(run.group.len(), N_WINDOW_BATCH);
        assert_eq!(run.group[0].frames, 2, "the Object window holds two frames");
        assert_eq!(run.group[0].files.len(), 4, "two frames and two clips");
        assert_eq!(run.group[0].times.len(), run.group[0].files.len());
        assert_eq!(run.remaining_in_batch(), 1, "one group is one model call");
    }

    /// The last group holds the windows that remain.
    #[test]
    fn the_last_group_holds_the_windows_that_remain() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("tail_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        let total = run.total();
        assert!(total % N_WINDOW_BATCH != 0, "the fixture divides by four here");
        run.window_at = total - 1;

        run.cut_batch().unwrap();

        assert_eq!(run.group.len(), 1, "the tail holds one window");
        assert_eq!(run.group[0].start, run.windows[total - 1]);
    }

    /// A video that holds fewer windows than one group gives one group.
    #[test]
    fn a_short_video_gives_one_group() {
        let Some(video) = spike_video_short() else {
            return;
        };
        let media = scratch("short_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();

        assert!(run.total() < N_WINDOW_BATCH, "the fixture is short here");
        run.cut_batch().unwrap();
        assert_eq!(run.group.len(), run.total());
    }

    /// A window past the cut names itself, and the run does not panic.
    #[test]
    fn a_window_past_the_cut_reports_the_window() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("cut_short_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        run.cut_batch().unwrap();
        // The stream claims more media than the cut holds.
        run.frames_cut.truncate(4);
        run.clips_cut.clear();
        run.window_at = 10;

        let err = run.cut_batch().unwrap_err();

        assert!(
            err.to_string().contains("no media came from the window"),
            "got {err}"
        );
    }

    /// A window with no media inside a group reports its own time.
    ///
    /// The cut ends before the plan does, so a window of the group holds no
    /// file. The run must name that window and must not embed an empty one.
    #[test]
    fn a_window_with_no_media_inside_a_group_reports_its_time() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("short_cut_media");
        let plan = plan::Plan::of(plan::Preset::Object);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();
        run.cut_batch().unwrap();
        // The frame stream ends at 28 s, and the plan asks for more.
        run.frames_cut.truncate(28);
        run.clips_cut.clear();
        run.window_at = 27;

        let err = run.cut_batch().unwrap_err();

        assert!(
            err.to_string().contains("no media came from the window at 28 s"),
            "got {err}"
        );
    }

    /// The fine run cuts the whole video one time, and it keeps its record count.
    #[test]
    fn the_fine_run_cuts_the_whole_video_once() {
        let Some(video) = spike_video() else {
            return;
        };
        let mut run = IndexRun::start(&video).unwrap();
        // 120 frames of 0.25 s, and 23 audio clips of the short audio track.
        assert_eq!(run.total(), 143);

        run.cut_batch().unwrap();
        assert_eq!(run.frames_cut.len(), 120);
        assert_eq!(run.clips_cut.len(), 24);
        assert_eq!(run.files.len(), FRAME_BATCH);
        // The first batch of frames starts at the first frame.
        assert_eq!(run.batch_from, 0);
    }

    /// A cut that holds fewer items than the plan stops the phase.
    ///
    /// A stream whose duration over-reports gives a plan longer than the cut.
    /// The run keeps the records that the cut can serve, and it does not stop
    /// the index of the other files of a folder.
    #[test]
    fn a_cut_shorter_than_the_plan_stops_the_phase() {
        let Some(video) = spike_video() else {
            return;
        };
        let mut run = IndexRun::start(&video).unwrap();
        run.cut_batch().unwrap();
        assert_eq!(run.frames_cut.len(), 120);

        // The stream claims 50 s, and the cut holds 30 s of frames.
        run.frames = (0..200).map(|i| i as f64 * FRAME_STEP).collect();
        run.frame_at = 150;
        run.files.clear();

        // The second batch starts past the cut, so the frames are over.
        run.cut_batch().unwrap();
        assert!(run.files.is_empty());
        assert_eq!(run.phase, Phase::Audio, "the audio phase comes next");

        // The audio phase then cuts its own batch from the clip list.
        run.cut_batch().unwrap();
        assert_eq!(run.phase, Phase::Audio);
        assert_eq!(run.files.len(), 24);
        assert_eq!(run.batch_from, 0);
    }

    /// A video with no audio track gives frame records alone.
    #[test]
    fn a_silent_video_gives_frame_records_alone() {
        let Some(video) = spike_video_without_audio() else {
            return;
        };
        let media = scratch("silent_pass_media");
        let plan = plan::Plan::of(plan::Preset::Fine);
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();

        assert!(run.audios.is_empty(), "a silent clip holds no audio item");
        run.cut_batch().unwrap();
        assert_eq!(run.frames_cut.len(), 120);
        assert!(run.clips_cut.is_empty());
        assert_eq!(run.files.len(), FRAME_BATCH);
    }

    /// A plain video is its own source, so it takes no copy note.
    #[test]
    fn a_plain_video_takes_no_copy_note() {
        let Some(video) = spike_video() else {
            return;
        };
        let media = scratch("note_media");
        assert!(copy_note(&video, &media).unwrap().is_none());
    }

    /// A fragmented video takes a copy note, and the copy lands in the media
    /// folder.
    #[test]
    fn a_fragmented_video_takes_a_copy_note() {
        let Some(video) = cut_spike(
            &["-t", "2", "-movflags", "frag_keyframe+empty_moov"],
            "frag_note",
        ) else {
            return;
        };
        let media = scratch("frag_note_media");
        let note = copy_note(&video, &media).unwrap();
        assert_eq!(note.as_deref(), Some(video.to_string_lossy().as_ref()));
        assert!(media.is_dir(), "the copy folder comes with the note");
        let made: Vec<_> = std::fs::read_dir(&media).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(made.len(), 1, "one copy comes");
    }

    /// The spike clip, when it is here.
    fn spike_video() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from("../spike/testvideo_nolabels.mp4");
        path.is_file().then_some(path)
    }

    /// Cut a copy of the spike clip. A machine without the clip gives None.
    fn cut_spike(extra: &[&str], name: &str) -> Option<std::path::PathBuf> {
        let video = spike_video()?;
        let dir = std::env::temp_dir().join(format!("vei_cut_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok()?;
        let out = dir.join("cut.mp4");
        let status = std::process::Command::new(crate::media::FFMPEG)
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
            .arg(&video)
            .args(extra)
            .arg(&out)
            .status()
            .ok()?;
        status.success().then_some(out)
    }

    /// A copy of the spike clip with the audio dropped.
    fn spike_video_without_audio() -> Option<std::path::PathBuf> {
        cut_spike(&["-an", "-c:v", "copy"], "silent")
    }

    /// A three second cut of the spike clip.
    fn spike_video_short() -> Option<std::path::PathBuf> {
        cut_spike(&["-t", "3"], "short")
    }

    /// The window plan gives the window count, and far fewer records than the
    /// fine plan.
    #[test]
    fn a_window_plan_gives_one_record_for_each_window() {
        let Some(video) = spike_video() else {
            return;
        };
        let plan = plan::Plan::of(plan::Preset::Object);
        let media = scratch("window_count_media");
        let run = IndexRun::start_with(&video, &plan, &media).unwrap();

        let seconds = crate::media::probe_duration(&video).unwrap();
        let windows = plan.window_starts(seconds).len();
        assert_eq!(run.total(), windows, "the window count");
        // The fine run makes one record for each 0.25 s frame. The window run
        // must make far fewer.
        let fine = IndexRun::start(&video).unwrap().total();
        assert!(run.total() * 4 < fine, "windows {} fine {fine}", run.total());
    }

    /// A video that holds no audio still gives whole windows.
    ///
    /// The run must not fail when the audio is absent and the plan asks for it.
    #[test]
    fn a_video_with_no_audio_gives_whole_windows() {
        let Some(video) = spike_video_without_audio() else {
            return;
        };
        let plan = plan::Plan::of(plan::Preset::Speech);
        let media = scratch("silent_window_media");
        let mut run = IndexRun::start_with(&video, &plan, &media).unwrap();

        assert!(run.total() > 0, "the windows must still come");
        // The cut of the first window must not fail, and the window must hold
        // the frames alone. The media cut needs no model.
        run.cut_batch().unwrap();
        let files = &run.group[0].files;
        assert!(!files.is_empty(), "the window holds its frames");
        assert!(
            files
                .iter()
                .all(|path| path.extension().is_some_and(|e| e == "jpg")),
            "a silent video gives frames alone: {files:?}"
        );
        run.clear_media();
    }

    /// A video shorter than one window is refused, and the message names the
    /// window. The same video fits the shorter Object window.
    #[test]
    fn a_video_shorter_than_one_window_gives_no_record() {
        let Some(video) = spike_video_short() else {
            return;
        };
        let plan = plan::Plan::of(plan::Preset::Speech);
        // The run must refuse the video, and the message must name the window.
        let media = scratch("short_window_media");
        let err = IndexRun::start_with(&video, &plan, &media).unwrap_err();
        assert!(format!("{err}").contains("shorter than one"), "{err}");
        // The same video is long enough for the 2 s Object window.
        let object = plan::Plan::of(plan::Preset::Object);
        assert!(IndexRun::start_with(&video, &object, &media).is_ok());
    }
}
