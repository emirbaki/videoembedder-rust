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

/// Cut the audio of a whole video into one-second 16 kHz mono WAV clips.
///
/// One ffmpeg call covers the whole track. Clip number `k` counts from zero
/// and covers `[k, k + 1)`, so the caller slices the list by time. A video with
/// no audio track gives an empty list.
pub fn cut_audio(video: &Path, dir: &Path) -> Result<Vec<PathBuf>> {
    if !has_audio(video) {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(dir)?;
    let pattern = dir.join("a%05d.wav").to_string_lossy().to_string();
    let args = vec![
        "-y".into(),
        "-i".into(),
        video.to_string_lossy().to_string(),
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

/// The file to cut: the video itself, or its progressive copy.
///
/// A fragmented MP4 seeks badly, so the index run reads a copy. The copy goes
/// in `<store>/media/`, and this call makes it when it is missing or stale.
pub fn source_file(video: &Path, media: &Path) -> Result<PathBuf> {
    if !is_fragmented_mp4(video)? {
        return Ok(video.to_path_buf());
    }
    progressive_copy(video, media)
}

/// True when an MP4 holds fragmented media.
///
/// The walk reads the top level boxes. A fragmented file holds one `moof` box
/// for each fragment, and a plain file holds none. A file that is not an MP4
/// falls out of the walk and reads as plain.
pub fn is_fragmented_mp4(path: &Path) -> Result<bool> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let mut at = 0u64;
    while at + 8 <= len {
        file.seek(SeekFrom::Start(at))?;
        let mut head = [0u8; 8];
        if file.read_exact(&mut head).is_err() {
            return Ok(false);
        }
        let size = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as u64;
        if &head[4..8] == b"moof" {
            return Ok(true);
        }
        let step = match size {
            0 => len - at,
            1 => 16,
            _ => size,
        };
        if step < 8 {
            return Ok(false);
        }
        at += step;
    }
    Ok(false)
}

/// The name of the progressive copy of one video.
///
/// The name holds the source name and a hash of the full path, so two videos
/// with one name in two folders do not share a copy.
fn copy_name(video: &Path) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in video.to_string_lossy().to_lowercase().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let stem = video
        .file_stem()
        .and_then(|part| part.to_str())
        .unwrap_or("video");
    format!("{stem}_{hash:016x}.mp4")
}

/// True when the copy exists and it is not older than its source.
fn copy_is_fresh(copy: &Path, video: &Path) -> bool {
    let (Ok(copied), Ok(source)) = (copy.metadata(), video.metadata()) else {
        return false;
    };
    let (Ok(copied), Ok(source)) = (copied.modified(), source.modified()) else {
        return false;
    };
    copied >= source
}

/// Rewrite a fragmented MP4 as a progressive MP4, and give back the copy.
///
/// A copy that is not older than its source stands, so a re-cut skips the
/// rewrite. The copy holds the same streams, and the index run reads it.
pub fn progressive_copy(video: &Path, media: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(media)?;
    let copy = media.join(copy_name(video));
    if copy_is_fresh(&copy, video) {
        return Ok(copy);
    }
    let args = vec![
        "-y".into(),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-c".into(),
        "copy".into(),
        "-movflags".into(),
        "+faststart".into(),
        copy.to_string_lossy().to_string(),
    ];
    run_ffmpeg(args, "rewriting a fragmented mp4")?;
    Ok(copy)
}

/// Cut every frame of a video at `rate` frames a second.
///
/// One ffmpeg call covers the whole video, so a plan with many windows pays
/// for the seek once. The caller slices the list by time, because frame number
/// `n` counts from one and sits at `(n - 1) / rate`.
pub fn cut_frames(video: &Path, rate: f64, dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let pattern = dir.join("f%05d.jpg").to_string_lossy().to_string();
    let args = vec![
        "-y".into(),
        "-i".into(),
        video.to_string_lossy().to_string(),
        "-vf".into(),
        format!("fps={rate}"),
        "-q:v".into(),
        "2".into(),
        pattern,
    ];
    run_ffmpeg(args, "extracting frames")?;
    let files = read_media(dir, &["jpg", "jpeg", "png"])?;
    if files.is_empty() {
        bail!("no frame came from {}", video.display());
    }
    Ok(files)
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

    /// One pass over the whole video gives one frame for each step of the rate.
    #[test]
    fn a_whole_video_cut_gives_a_frame_a_step() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let dir = scratch("whole_frames");
        let files = cut_frames(&video, 1.0, &dir).unwrap();
        // The clip holds 30.064 s of format, so one frame a second gives 30.
        assert_eq!(files.len(), 30, "got {}", files.len());
        assert!(files.iter().all(|file| file.exists()));
        // The names climb, because ffmpeg numbers them in time order.
        let mut sorted = files.clone();
        sorted.sort();
        assert_eq!(files, sorted);

        let dir = scratch("whole_frames_fast");
        let fast = cut_frames(&video, 4.0, &dir).unwrap();
        assert_eq!(fast.len(), 120, "got {}", fast.len());
    }

    /// One pass gives one clip for each second of the audio stream.
    ///
    /// The audio stream of the clip is 23.1 s long, so the count follows the
    /// stream and not the 30 s of the file.
    #[test]
    fn a_whole_video_audio_cut_follows_the_audio_stream() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        let dir = scratch("whole_audio");
        let clips = cut_audio(&video, &dir).unwrap();
        assert_eq!(clips.len(), 24, "got {}", clips.len());
        assert!(clips.iter().all(|file| file.exists()));
        for clip in &clips {
            assert!(wav_has_samples(&std::fs::read(clip).unwrap()));
        }
    }

    /// A video with no audio track gives no clip, and no fault.
    #[test]
    fn a_silent_video_gives_no_clip() {
        if !ready() {
            eprintln!("skipped: ffmpeg is missing");
            return;
        }
        let dir = scratch("silent_source");
        std::fs::create_dir_all(&dir).unwrap();
        let silent = dir.join("silent.mp4");
        let made = Command::new(FFMPEG)
            .args(["-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=2", "-pix_fmt", "yuv420p"])
            .arg(&silent)
            .output();
        match made {
            Ok(out) if out.status.success() => {}
            _ => {
                eprintln!("skipped: ffmpeg cannot make the silent clip");
                return;
            }
        }
        let out = scratch("silent_out");
        assert!(cut_audio(&silent, &out).unwrap().is_empty());
    }

    /// A missing file gives a fault, and not an empty list.
    #[test]
    fn a_whole_video_cut_fails_on_a_missing_file() {
        let dir = scratch("missing");
        assert!(cut_frames(Path::new("no_such_video.mp4"), 4.0, &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fragmented MP4 holds a `moof` box, and a plain MP4 does not.
    #[test]
    fn the_fragment_check_reads_the_boxes() {
        let video = test_video();
        if !ready() || !video.exists() {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        }
        assert!(!is_fragmented_mp4(&video).unwrap());

        let dir = scratch("fragmented");
        std::fs::create_dir_all(&dir).unwrap();
        let fragmented = dir.join("frag.mp4");
        let made = Command::new(FFMPEG)
            .args(["-y", "-i"])
            .arg(&video)
            .args(["-t", "2", "-c", "copy", "-movflags", "frag_keyframe+empty_moov"])
            .arg(&fragmented)
            .output()
            .unwrap();
        assert!(made.status.success());
        assert!(is_fragmented_mp4(&fragmented).unwrap());
    }

    /// A file that is not an MP4 holds no box, and it reads as plain.
    #[test]
    fn a_short_file_reads_as_plain() {
        let dir = scratch("short_file");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tiny.mp4");
        std::fs::write(&file, b"not a real mp4").unwrap();
        assert!(!is_fragmented_mp4(&file).unwrap());
    }

    /// Make a fragmented copy of the test clip, for the copy tests.
    fn fragmented_clip(name: &str) -> Option<PathBuf> {
        let video = test_video();
        if !ready() || !video.exists() {
            return None;
        }
        let dir = scratch(name);
        std::fs::create_dir_all(&dir).ok()?;
        let fragmented = dir.join("frag.mp4");
        let made = Command::new(FFMPEG)
            .args(["-y", "-i"])
            .arg(&video)
            .args(["-t", "2", "-c", "copy", "-movflags", "frag_keyframe+empty_moov"])
            .arg(&fragmented)
            .output()
            .ok()?;
        made.status.success().then_some(fragmented)
    }

    /// The copy is progressive, it holds the same media, and a second call
    /// reuses it.
    #[test]
    fn a_fragmented_video_takes_one_progressive_copy() {
        let Some(fragmented) = fragmented_clip("copy_source") else {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        };
        let media = scratch("copy_media");
        let copy = progressive_copy(&fragmented, &media).unwrap();
        assert!(copy.exists());
        assert!(copy.starts_with(&media));
        assert!(!is_fragmented_mp4(&copy).unwrap());
        assert!(probe_duration(&copy).unwrap() > 0.0);

        // A second call gives the same file, and it does not copy again.
        let before = std::fs::metadata(&copy).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let again = progressive_copy(&fragmented, &media).unwrap();
        assert_eq!(again, copy);
        let after = std::fs::metadata(&again).unwrap().modified().unwrap();
        assert_eq!(before, after);
    }

    /// A stale copy is made again.
    #[test]
    fn a_stale_copy_is_made_again() {
        let Some(fragmented) = fragmented_clip("stale_source") else {
            eprintln!("skipped: ffmpeg or the test video is missing");
            return;
        };
        let media = scratch("stale_media");
        let copy = progressive_copy(&fragmented, &media).unwrap();
        let first = std::fs::metadata(&copy).unwrap().modified().unwrap();

        // A newer source makes the copy stale.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let bytes = std::fs::read(&fragmented).unwrap();
        std::fs::write(&fragmented, bytes).unwrap();

        let again = progressive_copy(&fragmented, &media).unwrap();
        let second = std::fs::metadata(&again).unwrap().modified().unwrap();
        assert!(second > first);
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
