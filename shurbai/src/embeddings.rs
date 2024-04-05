use anyhow::{bail, Result};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::AddBos,
};

use crate::{types::ModelState, TokenCallback};

pub fn generate_embeddings(
    model: &ModelState,
    backend: &LlamaBackend,
    prompt: &String,
) -> Result<Vec<f32>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get() as u32)
        .with_embeddings(true);
    let mut ctx = model
        .model
        .new_context(&backend, ctx_params)
        .expect("Failed to create context");
    let tokens = model
        .model
        .str_to_token(prompt, AddBos::Always)
        .expect("Failed to tokenize");
    let n_ctx = ctx.n_ctx() as usize;

    if n_ctx < tokens.len() {
        bail!("One of the provided prompts exceeds the size of the context window");
    }
    let max_seq_id_batch = 0;
    let mut batch = LlamaBatch::new(n_ctx, 1);
    batch.add_sequence(&tokens, max_seq_id_batch, false)?;
    Ok(batch_decode(&mut ctx, &mut batch, true))
}

fn batch_decode(ctx: &mut LlamaContext, batch: &mut LlamaBatch, normalise: bool) -> Vec<f32> {
    ctx.clear_kv_cache();
    ctx.decode(batch).expect("Failed to decode");

    let embedding = ctx.embeddings_seq_ith(0).expect("Failed to get embeddings");
    let output_embeddings = if normalise {
        normalize(embedding)
    } else {
        embedding.to_vec()
    };
    batch.clear();
    output_embeddings
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}
