//! The tests that need the real model.
//!
//! The lib tests never load the model, because `LlamaBackend::init` runs one
//! time in a process. This file is its own test binary and its own process, so
//! one backend serves every test here.
//!
//! Run from the repo root:
//!   CMAKE_GENERATOR="Visual Studio 17 2022" cargo test --test model_batch

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use llama_cpp_2::mtmd::mtmd_default_marker;
use videoembed::embed::{self, Embedder};
use videoembed::{index, media, plan};

/// One media call at a time. The media encoder of llama.cpp is not thread safe,
/// so the tests of this file share one model and run one after the other.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The model folder. None when the two files are not here.
fn models() -> Option<PathBuf> {
    let models = PathBuf::from(std::env::var("VEI_MODELS").unwrap_or("../models".into()));
    let text = models.join("embeddinggemma-2-Q8_0.gguf");
    let mmproj = models.join("mmproj-embeddinggemma-2-Q8_0.gguf");
    (text.is_file() && mmproj.is_file()).then_some(models)
}

/// The one model of this test binary.
///
/// `LlamaBackend::init` runs one time in a process, so every test here shares
/// one embedder. A model that is not here gives None, and the test skips.
fn the_model() -> Option<(&'static Embedder, MutexGuard<'static, ()>)> {
    static SHARED: OnceLock<Option<Embedder>> = OnceLock::new();
    let guard = ONE_AT_A_TIME.lock().unwrap_or_else(|err| err.into_inner());
    let held = SHARED.get_or_init(|| models().and_then(|path| Embedder::load(&path, None).ok()));
    held.as_ref().map(|embedder| (embedder, guard))
}

/// The test clip. None when the file is not here.
fn video() -> Option<PathBuf> {
    let path = PathBuf::from("../spike/testvideo_nolabels.mp4");
    path.is_file().then_some(path)
}

/// A fresh folder for the cut of one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vei_batch_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the work folder is made");
    dir
}

/// One second of frames, and the one second clips, of the test video.
fn one_second_cut(video: &Path, dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let frames = media::cut_frames(video, 1.0, &dir.join("frames")).expect("the frames are cut");
    let clips = media::cut_audio(video, &dir.join("audio")).expect("the clips are cut");
    (frames, clips)
}

/// The media of one Object window of the cut, frames first and clips last.
fn window_media(frames: &[PathBuf], clips: &[PathBuf], from: f64) -> Vec<PathBuf> {
    let plan = plan::Plan::of(plan::Preset::Object);
    let mut media = index::frames_in(frames, from, plan.window, plan.frame_rate());
    media.extend(index::clips_in(clips, from, plan.window));
    media
}

/// The text of one window of `parts` media parts.
///
/// The shape is the shape of the real window text. This task builds the text
/// here, because the function that builds it arrives with the next task.
fn parts_text(parts: usize) -> String {
    (0..parts)
        .map(|_| format!("00:00 {}", mtmd_default_marker()))
        .collect::<Vec<String>>()
        .join(" ")
}

/// The token count of the Object window, and the room of a full batch.
///
/// The numbers pin `N_UBATCH` and `BUDGET_TOKENS`. The count comes from the
/// tokenizer of the real model, because a guess is worthless here.
#[test]
fn the_object_window_sits_under_the_batch_limit() {
    let Some((embedder, _guard)) = the_model() else {
        eprintln!("skipped: the model is not here");
        return;
    };
    let Some(video) = video() else {
        eprintln!("skipped: the test video is not here");
        return;
    };
    let dir = scratch("tokens");
    let (frames, clips) = one_second_cut(&video, &dir);

    let media = window_media(&frames, &clips, 0.0);
    let count = embedder
        .tokens_of(&parts_text(media.len()), &media)
        .expect("the count comes back");
    println!("one Object window: {count} tokens over {} parts", media.len());

    let ctx = embedder.new_context().expect("the context is made");
    assert_eq!(media.len(), 4, "two frames and two clips");
    assert!(count > 0, "the tokenizer gave no token");
    assert!(
        count < embed::BUDGET_TOKENS,
        "one Object window holds {count} tokens, above the budget"
    );
    assert!(
        count * embed::N_WINDOW_BATCH < ctx.n_ubatch() as usize,
        "four windows of {count} tokens pass the batch limit of {}",
        ctx.n_ubatch()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
