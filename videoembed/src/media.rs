//! ffmpeg helpers. Cut one window into frames and one wav file.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};

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

/// Read the length of one stream in seconds. The kind is "v" or "a".
///
/// `Ok(None)` means the file holds no stream of that kind, or the stream
/// reports no length. A missing stream length is normal in some files.
pub fn probe_stream_duration(video: &Path, kind: &str) -> Result<Option<f64>> {
    let out = Command::new(FFPROBE)
        .args([
            "-v",
            "error",
            "-select_streams",
            kind,
            "-show_entries",
            "stream=duration",
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
    Ok(text
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .next())
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

/// True when the video holds at least one audio stream.
///
/// A video with no audio track gives false. An ffprobe that fails gives false.
pub fn has_audio(video: &Path) -> bool {
    let out = Command::new(FFPROBE)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
        ])
        .arg(video)
        .output();
    match out {
        Ok(out) => out.status.success() && !out.stdout.is_empty(),
        Err(_) => false,
    }
}

/// Cut one span into JPEG frames at `rate` frames a second.
///
/// One ffmpeg call cuts the whole span. The seek flag comes before the input,
/// so ffmpeg seeks fast and reads only the span.
pub fn extract_frames(
    video: &Path,
    from: f64,
    to: f64,
    rate: f64,
    dir: &Path,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let pattern = dir.join("f%05d.jpg").to_string_lossy().to_string();
    let args = vec![
        "-y".into(),
        "-ss".into(),
        format!("{from}"),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-t".into(),
        format!("{}", to - from),
        "-vf".into(),
        format!("fps={rate}"),
        "-q:v".into(),
        "2".into(),
        pattern,
    ];
    run_ffmpeg(args, "extracting frames")?;
    read_media(dir, &["jpg", "jpeg", "png"])
}

/// Cut one span into one-second 16 kHz mono WAV clips.
///
/// The segment muxer names the clips in time order. A video with no audio
/// track gives an empty list, because a cut would fail.
pub fn extract_audio(video: &Path, from: f64, to: f64, dir: &Path) -> Result<Vec<PathBuf>> {
    if !has_audio(video) {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(dir)?;
    let pattern = dir.join("a%05d.wav").to_string_lossy().to_string();
    let args = vec![
        "-y".into(),
        "-ss".into(),
        format!("{from}"),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-t".into(),
        format!("{}", to - from),
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        "16000".into(),
        "-f".into(),
        "segment".into(),
        "-segment_time".into(),
        "1".into(),
        "-reset_timestamps".into(),
        "1".into(),
        pattern,
    ];
    run_ffmpeg(args, "extracting audio")?;
    read_media(dir, &["wav"])
}

/// List the files of one dir with one of the given extensions, in name order.
fn read_media(dir: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| extensions.contains(&x.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Cut one window into JPEG frames at 1 fps and a 16 kHz mono WAV.
///
/// The seek flag comes before the input. The length flag comes after it.
/// The two ffmpeg calls run together, because each one waits on a process.
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
    let frames_args = vec![
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
    ];

    let wav = dir.join("clip.wav");
    let audio_args = vec![
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
    ];

    // Both closures own their argument list, so the two calls do not share state.
    let (frames_out, audio_out) = std::thread::scope(|scope| {
        let frames = scope.spawn(move || run_ffmpeg(frames_args, "extracting frames"));
        let audio = scope.spawn(move || run_ffmpeg(audio_args, "extracting audio"));
        let f = frames
            .join()
            .unwrap_or_else(|_| Err(anyhow!("the frames ffmpeg thread stopped")));
        let a = audio
            .join()
            .unwrap_or_else(|_| Err(anyhow!("the audio ffmpeg thread stopped")));
        (f, a)
    });
    frames_out?;
    audio_out?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> bool {
        Path::new(FFMPEG).exists() && Path::new(FFPROBE).exists()
    }

    fn test_video() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spike/testvideo_nolabels.mp4")
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vei_media_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_stream_probe_separates_the_video_and_the_audio_length() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let format = probe_duration(&video).unwrap();
        let video_len = probe_stream_duration(&video, "v").unwrap().unwrap();
        let audio_len = probe_stream_duration(&video, "a").unwrap().unwrap();

        assert!(video_len > 0.0);
        assert!(audio_len > 0.0);
        // The test video holds about 23 s of audio inside a 30 s file.
        assert!(audio_len < format, "audio {audio_len} format {format}");
        // A missing stream kind gives None, not an error.
        assert!(probe_stream_duration(&video, "s").unwrap().is_none());
    }

    #[test]
    fn extract_frames_cuts_one_frame_every_quarter_second() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let dir = scratch("frames");
        let frames = extract_frames(&video, 0.0, 2.0, 4.0, &dir).unwrap();
        assert_eq!(frames.len(), 8, "got {}", frames.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_audio_cuts_one_clip_a_second() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        if !has_audio(&video) {
            eprintln!("skipped: the test video holds no audio track");
            return;
        }
        let dir = scratch("audio");
        let clips = extract_audio(&video, 0.0, 3.0, &dir).unwrap();
        assert_eq!(clips.len(), 3, "got {}", clips.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_frames_fails_on_a_missing_file() {
        let dir = scratch("missing");
        let outcome = extract_frames(Path::new("no_such_video.mp4"), 0.0, 1.0, 4.0, &dir);
        assert!(outcome.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
