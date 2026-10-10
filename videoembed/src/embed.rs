//! The embedding model. Text and window embedding live here.
//!
//! A window holds frames and one audio clip. The model must see them together,
//! in one batch, so that one pooling step covers the whole window.

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

/// The model parameters. A `cuda` build offloads every layer.
fn model_params() -> LlamaModelParams {
    let base = LlamaModelParams::default();
    if cfg!(feature = "cuda") {
        base.with_n_gpu_layers(N_GPU_LAYERS)
    } else {
        base
    }
}

/// The multimodal parameters. A `cuda` build runs the media encoder on the GPU.
fn mtmd_params() -> MtmdContextParams {
    MtmdContextParams {
        use_gpu: cfg!(feature = "cuda"),
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

pub struct Embedder {
    backend: LlamaBackend,
    model: LlamaModel,
    mtmd: MtmdContext,
}

impl Embedder {
    /// Load the text model and the multimodal projection.
    pub fn load(models: &Path) -> Result<Self> {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        let backend = LlamaBackend::init()?;
        let model = LlamaModel::load_from_file(
            &backend,
            &models.join("embeddinggemma-2-Q8_0.gguf"),
            &model_params(),
        )?;
        let mmproj = models.join("mmproj-embeddinggemma-2-Q8_0.gguf");
        let mtmd = MtmdContext::init_from_file(
            mmproj.to_str().context("model path is not utf8")?,
            &model,
            &mtmd_params(),
        )?;
        Ok(Self {
            backend,
            model,
            mtmd,
        })
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

    /// Make one context. A caller that embeds many windows keeps this context
    /// and passes it to `embed_window_in`, so the GPU key/value cache stays put.
    pub fn new_context(&self) -> Result<LlamaContext<'_>> {
        Ok(self.model.new_context(&self.backend, context_params())?)
    }

    /// Embed one window. Frames and audio go into one batch.
    ///
    /// This call makes its own context. Use it for one window. A loop uses
    /// `new_context` and `embed_window_in` instead.
    pub fn embed_window(&self, frames: &[PathBuf], audio: &Path) -> Result<Vec<f32>> {
        let mut ctx = self.new_context()?;
        self.embed_window_in(&mut ctx, frames, audio)
    }

    /// Embed one window in a caller-owned context.
    ///
    /// The context holds the key/value cache of every earlier call. This method
    /// clears that cache first, so one window never sees another window.
    pub fn embed_window_in(
        &self,
        ctx: &mut LlamaContext<'_>,
        frames: &[PathBuf],
        audio: &Path,
    ) -> Result<Vec<f32>> {
        ctx.clear_kv_cache();
        let mut bitmaps = Vec::with_capacity(frames.len() + 1);
        for frame in frames {
            let path = frame.to_str().context("frame path is not utf8")?;
            bitmaps.push(MtmdBitmap::from_file(&self.mtmd, path, false)?);
        }
        // A few windows decode to an empty audio clip. Then keep the frames only.
        let audio_path = audio.to_str().context("audio path is not utf8")?;
        match MtmdBitmap::from_file(&self.mtmd, audio_path, false) {
            Ok(bitmap) => bitmaps.push(bitmap),
            Err(_) => println!("  (no audio for this window)"),
        }
        self.embed_joint(ctx, &bitmaps)
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
