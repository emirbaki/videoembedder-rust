//! The embedding model. Text, image, and audio embedding live here.
//!
//! One call embeds one image or one audio clip. The media and the marker text
//! go into one batch, so that one pooling step covers the whole item.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaAttentionType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use llama_cpp_sys_2 as sys;

pub const QUERY_PREFIX: &str = "task: search result | query: ";
pub const DOC_PREFIX: &str = "title: none | text: ";

const N_CTX: u32 = 8192;
const N_BATCH: u32 = 8192;
/// The largest micro batch. The encoder refuses a batch above this limit.
///
/// One model call holds four windows, and the budget allows 1024 tokens for
/// one window, so the worst call holds 4096 tokens. The measured Object window
/// holds 257 tokens, and four of them hold 1028.
const N_UBATCH: u32 = 4096;

/// Offload every layer of the text model to the GPU.
///
/// The value sits above the layer count, so llama.cpp keeps every layer on the
/// card. A build without the `cuda` feature ignores it.
const N_GPU_LAYERS: u32 = 999;

/// Where the model runs. The user picks this, and a switch reloads the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Every layer on the card, and the media encoder too.
    Gpu,
    /// Every layer on the processor, and the media encoder too.
    Cpu,
}

impl Device {
    /// The one name for this mode. The console, the page, and the JSON use it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }

    /// Read a mode from its name. The case and the space do not matter.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "gpu" => Some(Self::Gpu),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// True when this machine can put layers on a card.
///
/// The question goes to llama.cpp. A build without the `cuda` feature answers
/// false, and a CUDA build on a machine with no card answers false too.
pub fn gpu_available(backend: &LlamaBackend) -> bool {
    backend.supports_gpu_offload()
}

/// The device to load on. `None` asks for the card when it is there.
///
/// A named device that this machine cannot serve is an error, never a quiet
/// fall back, because the user asked for it.
fn choose_device(backend: &LlamaBackend, asked: Option<Device>) -> Result<Device> {
    let available = gpu_available(backend);
    match asked {
        Some(Device::Gpu) if !available => bail!("this build cannot put layers on a card"),
        Some(device) => Ok(device),
        None => Ok(if available { Device::Gpu } else { Device::Cpu }),
    }
}

/// The device to try after a load fails on `device`, if one is left.
///
/// The processor is the last device. A failure there has no fall back.
fn fallback_device(device: Device) -> Option<Device> {
    match device {
        Device::Gpu => Some(Device::Cpu),
        Device::Cpu => None,
    }
}

/// Load on one device, then on the fall back when the first load fails.
///
/// The card is the fast path, and a machine may hold one that cannot take the
/// model. Google falls back to the processor when init fails. The function
/// returns the device that answered, so the caller reports the truth.
fn try_devices<T, F>(device: Device, mut open: F) -> Result<(Device, T)>
where
    F: FnMut(Device) -> Result<T>,
{
    match open(device) {
        Ok(value) => Ok((device, value)),
        Err(err) => match fallback_device(device) {
            Some(next) => open(next).map(|value| (next, value)),
            None => Err(err),
        },
    }
}

/// The model parameters for one mode. The GPU keeps every layer on the card.
///
/// The CPU mode asks for zero layers in words. The default is -1, and
/// llama.cpp reads a negative count as every layer, so the default would
/// offload the whole model in a CUDA build.
fn model_params(device: Device) -> LlamaModelParams {
    let base = LlamaModelParams::default();
    match device {
        Device::Gpu => base.with_n_gpu_layers(N_GPU_LAYERS),
        Device::Cpu => base.with_n_gpu_layers(0),
    }
}

/// The multimodal parameters for one mode.
fn mtmd_params(device: Device) -> MtmdContextParams {
    MtmdContextParams {
        use_gpu: device == Device::Gpu,
        n_threads: 8,
        print_timings: false,
        ..Default::default()
    }
}

fn context_params() -> LlamaContextParams {
    LlamaContextParams::default()
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean)
        .with_attention_type(LlamaAttentionType::NonCausal)
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_BATCH)
        .with_n_ubatch(N_UBATCH)
        .with_n_seq_max(N_WINDOW_BATCH as u32)
}

/// The windows in one model call.
pub const N_WINDOW_BATCH: usize = 4;

/// The token count that one window may hold before it loses its audio.
///
/// The count of one window comes from the tokenizer, never from a guess. The
/// Object window of the test video measures 257 tokens over four parts, and
/// four of those windows fit the batch of 2048. The budget of 1024 comes from
/// Google.
pub const BUDGET_TOKENS: usize = 1024;

/// One window of a model call: its text and its media files.
///
/// The text carries one marker for each media file, in the same order.
#[derive(Clone, Debug, Default)]
pub struct WindowIn {
    /// The marker text of the window.
    pub text: String,
    /// The media files in the order the model reads them: the frames first.
    pub media: Vec<PathBuf>,
}

pub struct Embedder {
    backend: LlamaBackend,
    device: Device,
    model: LlamaModel,
    mtmd: MtmdContext,
}

impl Embedder {
    /// Load the text model and the multimodal projection on one device.
    ///
    /// A `None` device asks for the card when it is there, and the processor
    /// otherwise.
    pub fn load(models: &Path, asked: Option<Device>) -> Result<Self> {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        let backend = LlamaBackend::init()?;
        let device = choose_device(&backend, asked)?;
        let (device, (model, mtmd)) =
            try_devices(device, |device| Self::open(models, &backend, device))?;
        Ok(Self {
            backend,
            device,
            model,
            mtmd,
        })
    }

    /// Load the model and the projection for one mode.
    ///
    /// The caller keeps the backend. `LlamaBackend::init` runs one time in a
    /// process, so a switch of mode must not ask for a second one.
    fn open(
        models: &Path,
        backend: &LlamaBackend,
        device: Device,
    ) -> Result<(LlamaModel, MtmdContext)> {
        let model = LlamaModel::load_from_file(
            backend,
            &models.join("embeddinggemma-2-Q8_0.gguf"),
            &model_params(device),
        )?;
        let mmproj = models.join("mmproj-embeddinggemma-2-Q8_0.gguf");
        let mtmd = MtmdContext::init_from_file(
            mmproj.to_str().context("model path is not utf8")?,
            &model,
            &mtmd_params(device),
        )?;
        Ok((model, mtmd))
    }

    /// Load the model and the projection for one device, ready to swap in.
    ///
    /// The call takes the embedder by shared reference, so a caller may keep a
    /// context open while the new model loads. The old model stays in place
    /// until [`Embedder::swap`], so a failure here changes nothing. A failed
    /// load on the card falls back to the processor, and the answer names the
    /// device that took the model.
    pub fn prepare(
        &self,
        models: &Path,
        device: Device,
    ) -> Result<(Device, LlamaModel, MtmdContext)> {
        let device = choose_device(&self.backend, Some(device))?;
        let (device, (model, mtmd)) =
            try_devices(device, |device| Self::open(models, &self.backend, device))?;
        Ok((device, model, mtmd))
    }

    /// Prove that one model holds a context. The probe context goes at once.
    ///
    /// A switch calls this before the old model leaves. A device with no room
    /// for the key and value cache then never takes over, and the old device
    /// keeps serving. The caller must call this before [`Embedder::swap`],
    /// because a context borrows the model.
    pub fn probe(&self, model: &LlamaModel) -> Result<()> {
        model.new_context(&self.backend, context_params())?;
        Ok(())
    }

    /// Take a model pair from [`Embedder::prepare`].
    ///
    /// Every context of the old model is dead from here: a context borrows the
    /// model, so the caller must drop it before this call.
    pub fn swap(&mut self, device: Device, model: LlamaModel, mtmd: MtmdContext) {
        self.model = model;
        self.mtmd = mtmd;
        self.device = device;
    }

    /// Load the model again on another device.
    ///
    /// The old model stays in place until the new one is ready, so a failure
    /// leaves a working embedder. The caller must drop every context first,
    /// because a context borrows the model.
    pub fn reload(&mut self, models: &Path, device: Device) -> Result<()> {
        let (device, model, mtmd) = self.prepare(models, device)?;
        self.swap(device, model, mtmd);
        Ok(())
    }

    /// The device this embedder runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// True when this machine can put layers on a card.
    pub fn supports_gpu(&self) -> bool {
        gpu_available(&self.backend)
    }

    /// Tokenize the text of one window with its media.
    ///
    /// The text carries one marker for each bitmap, in the same order.
    fn tokenize(&self, text: &str, media: &[&MtmdBitmap]) -> Result<MtmdInputChunks> {
        let input = MtmdInputText {
            text: text.to_string(),
            add_special: true,
            parse_special: true,
        };
        Ok(self.mtmd.tokenize(input, media)?)
    }

    /// The number of tokens that one window costs, as the tokenizer counts it.
    ///
    /// The count comes from the model, so a budget check uses the real number.
    pub fn tokens_of(&self, text: &str, media: &[PathBuf]) -> Result<usize> {
        if media.is_empty() {
            bail!("a window holds no media");
        }
        let mut bitmaps = Vec::with_capacity(media.len());
        for path in media {
            let path = path.to_str().context("media path is not utf8")?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self.tokenize(text, &refs)?;
        Ok(chunks.total_tokens())
    }

    /// Embed one text. Returns a unit vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ctx = self.model.new_context(&self.backend, context_params())?;
        let input = MtmdInputText {
            text: text.to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = self.mtmd.tokenize(input, &[])?;
        chunks.eval_chunks(&self.mtmd, &ctx, 0, 0, N_BATCH as i32, true)?;
        unit(ctx.embeddings_seq_ith(0)?)
    }

    /// Make one context. A caller that embeds many items keeps this context
    /// and passes it to `embed_image_in`, so the GPU key/value cache stays put.
    pub fn new_context(&self) -> Result<LlamaContext<'_>> {
        Ok(self.model.new_context(&self.backend, context_params())?)
    }

    /// Embed one image. Returns a unit vector.
    ///
    /// This call makes its own context. Use it for one image. A loop uses
    /// `new_context` and `embed_image_in` instead.
    pub fn embed_image(&self, frame: &Path) -> Result<Vec<f32>> {
        let mut ctx = self.new_context()?;
        self.embed_image_in(&mut ctx, frame)
    }

    /// Embed several windows in one model call. One vector for each window.
    ///
    /// Every window takes its own sequence, so the pooled vector of a window
    /// holds the tokens of that window alone. The text and the media of a
    /// window share its sequence for the same reason. The answer follows the
    /// order of the call.
    pub fn embed_windows_in(
        &self,
        ctx: &mut LlamaContext<'_>,
        windows: &[WindowIn],
    ) -> Result<Vec<Vec<f32>>> {
        if windows.is_empty() {
            bail!("a batch holds no window");
        }
        if windows.len() > N_WINDOW_BATCH {
            bail!(
                "the batch holds {} windows, above the limit of {N_WINDOW_BATCH}",
                windows.len()
            );
        }
        ctx.clear_kv_cache();

        let n_embd_inp = unsafe { sys::llama_model_n_embd_inp(self.model.as_ptr()) };
        if n_embd_inp <= 0 {
            bail!("the model reports no input embedding size");
        }
        let n_embd_inp = n_embd_inp as usize;

        let batch = unsafe { sys::llama_batch_ext_init(ctx.as_ptr()) };
        if batch.is_null() {
            bail!("llama_batch_ext_init returned null");
        }
        let mrope = multi_axis_rope(self.model.as_ptr());
        let mut pos = vec![0 as sys::llama_pos; windows.len()];
        let mut filled = Ok(());
        for (at, window) in windows.iter().enumerate() {
            let held = self.fill_window(window, at as sys::llama_seq_id, batch, n_embd_inp, mrope, &mut pos[at]);
            if let Err(err) = held {
                filled = Err(err);
                break;
            }
        }
        let raw = match filled {
            Ok(()) => unsafe {
                let rc = sys::llama_process(ctx.as_ptr(), sys::LLAMA_PROCESS_TYPE_DECODE, batch);
                if rc != 0 {
                    Err(anyhow::anyhow!("llama_process failed with code {rc}"))
                } else {
                    (0..windows.len())
                        .map(|seq| pooled_seq(ctx, seq as sys::llama_seq_id))
                        .collect::<Result<Vec<Vec<f32>>>>()
                }
            },
            Err(err) => Err(err),
        };
        unsafe { sys::llama_batch_ext_free(batch) };

        let mut out = Vec::with_capacity(windows.len());
        for one in raw? {
            out.push(unit(&one)?);
        }
        Ok(out)
    }

    /// Embed one image in a caller-owned context.
    pub fn embed_image_in(&self, ctx: &mut LlamaContext<'_>, frame: &Path) -> Result<Vec<f32>> {
        let window = WindowIn {
            text: mtmd_default_marker().to_string(),
            media: vec![frame.to_path_buf()],
        };
        Ok(self.embed_windows_in(ctx, &[window])?.remove(0))
    }

    /// Embed one audio clip in a caller-owned context.
    pub fn embed_audio_in(&self, ctx: &mut LlamaContext<'_>, clip: &Path) -> Result<Vec<f32>> {
        self.embed_image_in(ctx, clip)
    }

    /// Embed several media files as one window, in the given order.
    ///
    /// The caller passes the frames first and the audio clip last. One batch
    /// holds every part, and one decode follows. A probe uses this call to test
    /// window mode.
    pub fn embed_window_in(
        &self,
        ctx: &mut LlamaContext<'_>,
        media: &[&Path],
    ) -> Result<Vec<f32>> {
        if media.is_empty() {
            bail!("a window holds no media");
        }
        let window = WindowIn {
            text: mtmd_default_marker().repeat(media.len()),
            media: media.iter().map(|path| path.to_path_buf()).collect(),
        };
        Ok(self.embed_windows_in(ctx, &[window])?.remove(0))
    }

    /// Add every token and every media row of one window to the batch.
    ///
    /// The window takes its own sequence, so its pooled vector holds its own
    /// entries alone. The text of the window goes to the tokenizer on its own,
    /// because one shared text gives the first window a start token and the
    /// last window a trailing text piece, and those two windows then differ
    /// from the same window read on its own.
    fn fill_window(
        &self,
        window: &WindowIn,
        seq: sys::llama_seq_id,
        batch: *mut sys::llama_batch_ext,
        n_embd_inp: usize,
        mrope: bool,
        pos: &mut sys::llama_pos,
    ) -> Result<()> {
        let mut bitmaps = Vec::with_capacity(window.media.len());
        for path in &window.media {
            let path = path.to_str().context("media path is not utf8")?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = self.tokenize(&window.text, &refs)?;
        let n_chunks = unsafe { sys::mtmd_input_chunks_size(chunks.as_ptr()) };

        for i in 0..n_chunks {
            let chunk = unsafe { sys::mtmd_input_chunks_get(chunks.as_ptr(), i) };
            if chunk.is_null() {
                bail!("chunk {i} is null");
            }
            let kind = unsafe { sys::mtmd_input_chunk_get_type(chunk) };

            if kind == sys::MTMD_INPUT_CHUNK_TYPE_TEXT {
                let mut count = 0usize;
                let tokens = unsafe { sys::mtmd_input_chunk_get_tokens_text(chunk, &mut count) };
                if tokens.is_null() {
                    bail!("text chunk {i} has no tokens");
                }
                for t in 0..count {
                    let id = unsafe { *tokens.add(t) };
                    // A text token carries one position, at any rope type.
                    self.push(batch, seq, TokenOrRows::Token(id), n_embd_inp, &[*pos])?;
                    *pos += 1;
                }
            } else {
                let rc = unsafe { sys::mtmd_encode_chunk(self.mtmd.as_ptr(), chunk) };
                if rc != 0 {
                    bail!("media encode failed for chunk {i} with code {rc}");
                }
                let n_tokens = unsafe { sys::mtmd_input_chunk_get_n_tokens(chunk) };
                let data = unsafe { sys::mtmd_get_output_embd(self.mtmd.as_ptr()) };
                if data.is_null() {
                    bail!("chunk {i} produced no embedding rows");
                }
                let base = *pos;
                for t in 0..n_tokens {
                    let row = unsafe { data.add(t * n_embd_inp) };
                    let at = media_pos(chunk, kind, base, t, mrope);
                    let width = if mrope { 4 } else { 1 };
                    self.push(batch, seq, TokenOrRows::Rows(row), n_embd_inp, &at[..width])?;
                }
                // A media chunk covers positions of its own, and the count can
                // pass its token count. The next chunk starts after that count.
                let n_pos = unsafe { sys::mtmd_input_chunk_get_n_pos(chunk) };
                *pos = base + n_pos.max(n_tokens as sys::llama_pos);
            }
        }
        Ok(())
    }

    /// Add one item to the batch, on the sequence of the window that owns it.
    ///
    /// The positions hold one value for each rotary section of the model. An
    /// embedding entry reads every value of the slice.
    fn push(
        &self,
        batch: *mut sys::llama_batch_ext,
        seq: sys::llama_seq_id,
        item: TokenOrRows,
        n_embd_inp: usize,
        pos: &[sys::llama_pos],
    ) -> Result<()> {
        unsafe {
            let idx = match item {
                TokenOrRows::Token(id) => sys::llama_batch_ext_add_token(batch, seq, id),
                TokenOrRows::Rows(data) => {
                    let embd = sys::llama_embd {
                        data,
                        n_rows: 1,
                        n_embd: n_embd_inp,
                    };
                    sys::llama_batch_ext_add_embd(batch, seq, embd)
                }
            };
            if idx < 0 {
                bail!("the batch rejected an entry with code {idx}");
            }
            sys::llama_batch_ext_set_pos(batch, idx, pos.as_ptr());
        }
        Ok(())
    }
}

/// True when the model reads more than one rotary position for one embedding.
fn multi_axis_rope(model: *const sys::llama_model) -> bool {
    let kind = unsafe { sys::llama_model_rope_type(model) };
    kind == sys::LLAMA_ROPE_TYPE_MROPE || kind == sys::LLAMA_ROPE_TYPE_IMROPE
}

/// The positions of one media row, in the order of the rotary sections.
///
/// One value serves every section of a model with one rotary position. A model
/// with multi-axis rotary positions reads four sections. An image row then
/// takes its place in the picture grid, and any other media row takes the
/// running position.
fn media_pos(
    chunk: *const sys::mtmd_input_chunk,
    kind: sys::mtmd_input_chunk_type,
    base: sys::llama_pos,
    row: usize,
    mrope: bool,
) -> [sys::llama_pos; 4] {
    let at = base + row as sys::llama_pos;
    if !mrope {
        return [at, 0, 0, 0];
    }
    if kind == sys::MTMD_INPUT_CHUNK_TYPE_IMAGE {
        let tokens = unsafe { sys::mtmd_input_chunk_get_tokens_image(chunk) };
        if !tokens.is_null() {
            let grid = unsafe { sys::mtmd_image_tokens_get_decoder_pos(tokens, base, row) };
            return [
                grid.t as sys::llama_pos,
                grid.y as sys::llama_pos,
                grid.x as sys::llama_pos,
                grid.z as sys::llama_pos,
            ];
        }
    }
    [at, at, at, at]
}

enum TokenOrRows {
    Token(sys::llama_token),
    Rows(*const f32),
}

/// Copy the pooled vector of one sequence.
unsafe fn pooled_seq(ctx: &LlamaContext<'_>, seq: sys::llama_seq_id) -> Result<Vec<f32>> {
    let out = unsafe { sys::llama_get_embeddings_seq(ctx.as_ptr(), seq) };
    if out.is_null() {
        bail!("sequence {seq} returned no pooled embedding");
    }
    let n_out = unsafe { sys::llama_model_n_embd_out(ctx.model.as_ptr()) };
    if n_out <= 0 {
        bail!("the model reports no embedding size");
    }
    Ok(unsafe { std::slice::from_raw_parts(out, n_out as usize) }.to_vec())
}

/// The time of one frame or clip as `MM:SS`.
///
/// The seconds round down, so a part at 3.75 s reads `00:03`, the second that
/// holds that part. The minutes pass 59 for a long video, because a video here
/// can hold more than one hour of media.
pub fn mmss(at: f64) -> String {
    let seconds = at.max(0.0).floor() as u64;
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

/// The text of one window: one `MM:SS` marker before each media part.
///
/// The marker names the start time of its part, so the model reads the order
/// of the events and not only the pictures. The count of the markers always
/// matches the count of the media parts.
pub fn window_text(times: &[f64]) -> String {
    times
        .iter()
        .map(|at| format!("{} {}", mmss(*at), mtmd_default_marker()))
        .collect::<Vec<String>>()
        .join(" ")
}

/// Scale a vector to unit length.
pub fn unit(raw: &[f32]) -> Result<Vec<f32>> {
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 {
        bail!("zero-length embedding");
    }
    Ok(raw.iter().map(|x| x / norm).collect())
}

/// Cosine similarity of two unit vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The marker text of one window names each part with `MM:SS`.
    #[test]
    fn the_window_text_names_every_part() {
        // The example of the spec: a 2 s window at 3 s, two frames and a clip.
        assert_eq!(
            window_text(&[3.0, 4.0, 3.0]),
            "00:03 <__media__> 00:04 <__media__> 00:03 <__media__>"
        );
    }

    /// One marker for one media part, and no marker for no part.
    #[test]
    fn the_marker_count_matches_the_part_count() {
        assert_eq!(window_text(&[7.0]).matches(mtmd_default_marker()).count(), 1);
        assert_eq!(
            window_text(&[7.0, 7.0, 7.0])
                .matches(mtmd_default_marker())
                .count(),
            3
        );
        assert_eq!(window_text(&[]), "");
    }

    /// The seconds round down, so a part reads the second that holds it.
    #[test]
    fn a_part_reads_the_second_that_holds_it() {
        assert_eq!(mmss(0.0), "00:00");
        assert_eq!(mmss(3.0), "00:03");
        assert_eq!(mmss(3.75), "00:03");
        assert_eq!(mmss(59.999), "00:59");
        assert_eq!(mmss(-1.0), "00:00");
    }

    /// An hour of video keeps a readable time.
    #[test]
    fn a_long_video_still_reads_as_minutes_and_seconds() {
        assert_eq!(mmss(60.0), "01:00");
        assert_eq!(mmss(600.0), "10:00");
        assert_eq!(mmss(3600.0), "60:00");
        assert_eq!(mmss(3661.0), "61:01");
    }

    /// The console, the page, and the JSON field use these two names.
    #[test]
    fn a_device_round_trips_through_its_name() {
        assert_eq!(Device::Gpu.as_str(), "GPU");
        assert_eq!(Device::Cpu.as_str(), "CPU");
        assert_eq!(Device::parse("GPU"), Some(Device::Gpu));
        assert_eq!(Device::parse("gpu"), Some(Device::Gpu));
        assert_eq!(Device::parse("Cpu"), Some(Device::Cpu));
        assert_eq!(Device::parse(" cpu "), Some(Device::Cpu));
        assert_eq!(Device::parse("npu"), None);
        assert_eq!(Device::parse(""), None);
    }

    /// The GPU mode keeps every layer on the card. The CPU mode keeps none.
    #[test]
    fn the_gpu_mode_offloads_every_layer() {
        assert_eq!(
            model_params(Device::Gpu).n_gpu_layers(),
            N_GPU_LAYERS as i32
        );
        assert_eq!(model_params(Device::Cpu).n_gpu_layers(), 0);
    }

    /// The media encoder follows the same device as the text model.
    #[test]
    fn the_media_encoder_follows_the_device() {
        assert!(mtmd_params(Device::Gpu).use_gpu);
        assert!(!mtmd_params(Device::Cpu).use_gpu);
    }

    /// One backend for every test in this run.
    ///
    /// `LlamaBackend::init` works one time in a process, and its drop frees
    /// the backend. The static holds the one handle for the whole run.
    fn test_backend() -> &'static LlamaBackend {
        static BACKEND: std::sync::OnceLock<LlamaBackend> = std::sync::OnceLock::new();
        BACKEND.get_or_init(|| LlamaBackend::init().expect("the backend starts"))
    }

    /// The default build carries no CUDA kernels, so it has no card.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_cpu_build_has_no_card() {
        assert!(!gpu_available(test_backend()));
    }

    /// The card wins when the user names none, and a named device must exist.
    #[test]
    fn a_named_device_must_be_here() {
        let backend = test_backend();
        let card = gpu_available(backend);
        assert_eq!(
            choose_device(backend, None).unwrap(),
            if card { Device::Gpu } else { Device::Cpu }
        );
        assert_eq!(
            choose_device(backend, Some(Device::Cpu)).unwrap(),
            Device::Cpu
        );
        if card {
            assert_eq!(
                choose_device(backend, Some(Device::Gpu)).unwrap(),
                Device::Gpu
            );
        } else {
            assert!(
                choose_device(backend, Some(Device::Gpu)).is_err(),
                "a card that is not here is an error, not a fall back"
            );
        }
    }

    /// A failed load on the card falls back to the processor.
    ///
    /// Google tries the preferred backend and falls back when init fails. The
    /// caller reports the device that answered, so the user learns the truth.
    #[test]
    fn a_failed_card_load_falls_back_to_the_processor() {
        let tried = std::cell::RefCell::new(Vec::new());
        let answer = try_devices(Device::Gpu, |device| {
            tried.borrow_mut().push(device);
            if device == Device::Gpu {
                bail!("the card is out of memory")
            }
            Ok("the processor model")
        })
        .unwrap();
        assert_eq!(*tried.borrow(), vec![Device::Gpu, Device::Cpu]);
        assert_eq!(answer.0, Device::Cpu);
        assert_eq!(answer.1, "the processor model");
    }

    /// A card that loads keeps its place. The processor never loads.
    #[test]
    fn a_card_that_loads_is_kept() {
        let tried = std::cell::RefCell::new(Vec::new());
        let answer = try_devices(Device::Gpu, |device| {
            tried.borrow_mut().push(device);
            Ok("the card model")
        })
        .unwrap();
        assert_eq!(*tried.borrow(), vec![Device::Gpu]);
        assert_eq!(answer.0, Device::Gpu);
        assert_eq!(answer.1, "the card model");
    }

    /// The processor is the last device. A failure there is an error.
    #[test]
    fn a_failed_processor_load_is_an_error() {
        let answer: Result<(Device, &str)> = try_devices(Device::Cpu, |_| bail!("no room"));
        assert!(answer.is_err());
    }
}
