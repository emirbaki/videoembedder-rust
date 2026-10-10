//! ffmpeg helpers. Probe a video and cut frames and audio clips.

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

/// Run ffmpeg and give back its standard output.
///
/// The frame pipe of `thumb_jpeg` writes the picture bytes there.
fn run_ffmpeg_out(args: Vec<String>, what: &str) -> Result<Vec<u8>> {
    let mut full: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-nostdin".into(),
    ];
    full.extend(args);
    let out = Command::new(FFMPEG).args(&full).output()?;
    if !out.status.success() {
        bail!("ffmpeg failed while {what}");
    }
    if out.stdout.is_empty() {
        bail!("ffmpeg gave no picture while {what}");
    }
    Ok(out.stdout)
}

/// Cut one JPEG frame at a time, scaled to a width. The bytes come back.
///
/// The seek flag comes before the input, so ffmpeg seeks fast. A time past the
/// end of the video gives no frame, and the call refuses.
pub fn thumb_jpeg(video: &Path, at: f64, width: u32) -> Result<Vec<u8>> {
    let args = vec![
        "-y".into(),
        "-ss".into(),
        format!("{at}"),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-frames:v".into(),
        "1".into(),
        "-vf".into(),
        format!("scale={width}:-2"),
        "-f".into(),
        "image2pipe".into(),
        "-vcodec".into(),
        "mjpeg".into(),
        "-".into(),
    ];
    run_ffmpeg_out(args, "cutting a thumbnail")
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
    let found = read_media(dir, &["wav"])?;
    // Drop a segment with no samples, so the index skips it instead of
    // stopping on it.
    let mut kept = Vec::new();
    for path in found {
        match std::fs::read(&path) {
            Ok(bytes) if wav_has_samples(&bytes) => kept.push(path),
            Ok(_) => {}
            Err(err) => return Err(err).with_context(|| format!("read {}", path.display())),
        }
    }
    Ok(kept)
}

/// True when a wav file holds audio samples.
///
/// The data chunk size tells the truth. The stream duration of a video does
/// not always do so, and a cut past the real end of the audio gives a file
/// with a header and no samples.
fn wav_has_samples(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(512)];
    let Some(at) = head.windows(4).position(|window| window == b"data") else {
        return false;
    };
    let size_at = at + 4;
    if size_at + 4 > head.len() {
        return false;
    }
    let size = u32::from_le_bytes([
        head[size_at],
        head[size_at + 1],
        head[size_at + 2],
        head[size_at + 3],
    ]);
    size > 0
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A wav with an empty data chunk holds no audio.
    ///
    /// ffmpeg writes such a file when a cut lands past the real end of the
    /// audio. The encoder refuses the file, and the index run stops.
    #[test]
    fn a_wav_with_no_samples_is_refused() {
        let mut empty = Vec::new();
        empty.extend_from_slice(b"RIFF");
        empty.extend_from_slice(&34u32.to_le_bytes());
        empty.extend_from_slice(b"WAVEfmt ");
        empty.extend_from_slice(&[0u8; 12]);
        empty.extend_from_slice(b"data");
        empty.extend_from_slice(&0u32.to_le_bytes());
        assert!(!wav_has_samples(&empty));

        let mut full = empty.clone();
        let at = full.len() - 4;
        full[at..].copy_from_slice(&32000u32.to_le_bytes());
        assert!(wav_has_samples(&full));
    }

    #[test]
    fn a_file_that_is_not_a_wav_holds_no_samples() {
        assert!(!wav_has_samples(b""));
        assert!(!wav_has_samples(b"not a wav at all"));
    }

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

    /// A thumbnail is a JPEG, and the bytes start with the JPEG marker.
    #[test]
    fn a_thumb_is_a_jpeg() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let bytes = thumb_jpeg(&video, 1.0, 320).unwrap();
        assert!(bytes.len() > 200, "got {} bytes", bytes.len());
        assert_eq!(bytes[0], 0xff);
        assert_eq!(bytes[1], 0xd8);
    }

    /// A cut past the end gives no frame, so the call must refuse.
    #[test]
    fn a_thumb_past_the_end_is_an_error() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        assert!(thumb_jpeg(&video, 1000.0, 320).is_err());
    }

    #[test]
    fn a_thumb_of_a_missing_file_is_an_error() {
        assert!(thumb_jpeg(Path::new("no_such_video.mp4"), 0.0, 320).is_err());
    }
}
