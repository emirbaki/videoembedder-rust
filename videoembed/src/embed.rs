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
const N_UBATCH: u32 = 2048;

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
        let (model, mtmd) = Self::open(models, &backend, device)?;
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
    /// until [`Embedder::swap`], so a failure here changes nothing.
    pub fn prepare(
        &self,
        models: &Path,
        device: Device,
    ) -> Result<(Device, LlamaModel, MtmdContext)> {
        let device = choose_device(&self.backend, Some(device))?;
        let (model, mtmd) = Self::open(models, &self.backend, device)?;
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

    /// Embed one image in a caller-owned context.
    pub fn embed_image_in(&self, ctx: &mut LlamaContext<'_>, frame: &Path) -> Result<Vec<f32>> {
        ctx.clear_kv_cache();
        let path = frame.to_str().context("frame path is not utf8")?;
        let bitmap = MtmdBitmap::from_file(&self.mtmd, path, false)?;
        self.embed_joint(ctx, &[bitmap])
    }

    /// Embed one audio clip in a caller-owned context.
    pub fn embed_audio_in(&self, ctx: &mut LlamaContext<'_>, clip: &Path) -> Result<Vec<f32>> {
        ctx.clear_kv_cache();
        let path = clip.to_str().context("audio path is not utf8")?;
        let bitmap = MtmdBitmap::from_file(&self.mtmd, path, false)?;
        self.embed_joint(ctx, &[bitmap])
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
        ctx.clear_kv_cache();
        let mut bitmaps = Vec::with_capacity(media.len());
        for path in media {
            let path = path.to_str().context("media path is not utf8")?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
        }
        self.embed_joint(ctx, &bitmaps)
    }

    /// One batch holds every token and every media row. One decode follows.
    fn embed_joint(&self, ctx: &mut LlamaContext<'_>, bitmaps: &[MtmdBitmap]) -> Result<Vec<f32>> {
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let input = MtmdInputText {
            text: mtmd_default_marker().repeat(refs.len()),
            add_special: true,
            parse_special: true,
        };
        let chunks = self.mtmd.tokenize(input, &refs)?;

        let n_embd_inp = unsafe { sys::llama_model_n_embd_inp(self.model.as_ptr()) };
        if n_embd_inp <= 0 {
            bail!("the model reports no input embedding size");
        }
        let n_embd_inp = n_embd_inp as usize;

        let batch = unsafe { sys::llama_batch_ext_init(ctx.as_ptr()) };
        if batch.is_null() {
            bail!("llama_batch_ext_init returned null");
        }
        let filled = self.fill_batch(ctx, &chunks, batch, n_embd_inp);
        let raw = match filled {
            Ok(()) => unsafe { pooled(ctx) },
            Err(err) => Err(err),
        };
        unsafe { sys::llama_batch_ext_free(batch) };
        unit(&raw?)
    }

    /// Add every chunk to the batch. Text adds tokens. Media adds embedding rows.
    fn fill_batch(
        &self,
        ctx: &LlamaContext<'_>,
        chunks: &MtmdInputChunks,
        batch: *mut sys::llama_batch_ext,
        n_embd_inp: usize,
    ) -> Result<()> {
        let n_chunks = unsafe { sys::mtmd_input_chunks_size(chunks.as_ptr()) };
        let mut pos: sys::llama_pos = 0;

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
                    self.push(batch, TokenOrRows::Token(id), n_embd_inp, &mut pos)?;
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
                for t in 0..n_tokens {
                    let row = unsafe { data.add(t * n_embd_inp) };
                    self.push(batch, TokenOrRows::Rows(row), n_embd_inp, &mut pos)?;
                }
                // Media chunks can carry more positions than tokens. Text does not.
                let n_pos = unsafe { sys::mtmd_input_chunk_get_n_pos(chunk) };
                if n_pos > n_tokens as sys::llama_pos {
                    pos = pos + (n_pos - n_tokens as sys::llama_pos);
                }
            }
        }

        let rc = unsafe { sys::llama_process(ctx.as_ptr(), sys::LLAMA_PROCESS_TYPE_DECODE, batch) };
        if rc != 0 {
            bail!("llama_process failed with code {rc}");
        }
        Ok(())
    }

    fn push(
        &self,
        batch: *mut sys::llama_batch_ext,
        item: TokenOrRows,
        n_embd_inp: usize,
        pos: &mut sys::llama_pos,
    ) -> Result<()> {
        unsafe {
            let idx = match item {
                TokenOrRows::Token(id) => sys::llama_batch_ext_add_token(batch, 0, id),
                TokenOrRows::Rows(data) => {
                    let embd = sys::llama_embd {
                        data,
                        n_rows: 1,
                        n_embd: n_embd_inp,
                    };
                    sys::llama_batch_ext_add_embd(batch, 0, embd)
                }
            };
            if idx < 0 {
                bail!("the batch rejected an entry with code {idx}");
            }
            sys::llama_batch_ext_set_pos(batch, idx, pos);
            *pos += 1;
        }
        Ok(())
    }
}

enum TokenOrRows {
    Token(sys::llama_token),
    Rows(*const f32),
}

/// Copy the pooled vector of sequence 0.
unsafe fn pooled(ctx: &LlamaContext<'_>) -> Result<Vec<f32>> {
    let out = unsafe { sys::llama_get_embeddings_seq(ctx.as_ptr(), 0) };
    if out.is_null() {
        bail!("the context returned no pooled embedding");
    }
    let n_out = unsafe { sys::llama_model_n_embd_out(ctx.model.as_ptr()) };
    if n_out <= 0 {
        bail!("the model reports no embedding size");
    }
    Ok(unsafe { std::slice::from_raw_parts(out, n_out as usize) }.to_vec())
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
}
