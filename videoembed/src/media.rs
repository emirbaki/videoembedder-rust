//! ffmpeg helpers. Cut one window into frames and one wav file.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

pub const FFMPEG: &str =
    "D:/MLProjelerim/videoembedder-rust/tools/dl/ff/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe";
pub const FFPROBE: &str =
    "D:/MLProjelerim/videoembedder-rust/tools/dl/ff/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe";

/// Read the duration of a video in seconds.
pub fn probe_duration(video: &Path) -> Result<f64> {
    let out = Command::new(FFPROBE)
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(video)
        .output()
        .context("run ffprobe")?;
    if !out.status.success() {
        bail!("ffprobe failed for {}", video.display());
    }
    let text = String::from_utf8(out.stdout).context("ffprobe output is not utf8")?;
    text.trim()
        .parse::<f64>()
        .context("ffprobe duration is not a number")
}

fn run_ffmpeg(args: Vec<String>, what: &str) -> Result<()> {
    let mut full: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-nostdin".into(),
    ];
    full.extend(args);
    let status = Command::new(FFMPEG).args(&full).status()?;
    if !status.success() {
        bail!("ffmpeg failed while {what}");
    }
    Ok(())
}

/// Cut one window into JPEG frames at 1 fps and a 16 kHz mono WAV.
///
/// The seek flag comes before the input. The length flag comes after it.
pub fn extract_window(
    video: &Path,
    start: f64,
    end: f64,
    dir: &Path,
) -> Result<(Vec<PathBuf>, PathBuf)> {
    std::fs::create_dir_all(dir)?;
    let ss = format!("{start}");
    let span = format!("{}", end - start);
    let source = video.to_string_lossy().to_string();

    let frames_pattern = dir.join("f%02d.jpg").to_string_lossy().to_string();
    run_ffmpeg(
        vec![
            "-y".into(),
            "-ss".into(),
            ss.clone(),
            "-i".into(),
            source.clone(),
            "-t".into(),
            span.clone(),
            "-vf".into(),
            "fps=1".into(),
            "-q:v".into(),
            "2".into(),
            frames_pattern,
        ],
        "extracting frames",
    )?;

    let wav = dir.join("clip.wav");
    run_ffmpeg(
        vec![
            "-y".into(),
            "-ss".into(),
            ss,
            "-i".into(),
            source,
            "-t".into(),
            span,
            "-vn".into(),
            "-ac".into(),
            "1".into(),
            "-ar".into(),
            "16000".into(),
            "-f".into(),
            "wav".into(),
            wav.to_string_lossy().to_string(),
        ],
        "extracting audio",
    )?;

    Ok((read_frames(dir)?, wav))
}

/// List the frame files of a window dir in name order.
pub fn read_frames(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("jpg" | "jpeg" | "png")
            )
        })
        .collect();
    frames.sort();
    Ok(frames)
}
