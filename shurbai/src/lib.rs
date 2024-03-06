//! This is an translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::Ok;
use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::ggml_time_us;

use std::num::NonZeroU32;
use types::{LlamaResult, ModelConfig, ModelManager, ModelState};

use std::collections::HashMap;
use std::time::Duration;

pub type TokenCallback = Box<dyn Fn(String, bool)>;

pub mod types;


pub fn load_model(
    path: String,
    model_config: ModelConfig,
    llama_backend: &LlamaBackend,
) -> Result<LlamaModel> {
    let init_params = {
        #[cfg(feature = "cublas")]
        if model_config.use_gpu.unwrap_or(true) {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(feature = "cublas"))]
        LlamaModelParams::default().with_use_mlock(true)
    };
    let params = init_params.with_use_mlock(true);
    let model = LlamaModel::load_from_file(llama_backend, path.clone(), &params)
        .with_context(|| format!("failed to load model from {}", path))?;
    Ok(model)
}

pub fn load_models(models: Vec<types::ModelDefinition>) -> Result<ModelManager> {
    let llama_backend = LlamaBackend::init().expect("failed to initialize llama_backend");
    //let arc_llama_backend = Arc::new(llama_backend);
    let mut loaded_models = HashMap::new();
    for model in models {
        let llama_model = load_model(model.path, model.config.clone(), &llama_backend)
            .expect("failed to load model");
        let model_state = types::ModelState {
            model: llama_model,
            config: model.config,
            chat_template: model.chat_template,
        };
        let name = model.name.clone();
        loaded_models.insert(name, model_state);
    }
    let model_manager = ModelManager {
        models: loaded_models,
        backend: llama_backend,
    };
    Ok(model_manager)
}

/// Generate a llama response
/// # Arguments
/// * `model` - The llama model
/// * `ctx` - The llama context
/// * `tokens_list` - The list of tokens
/// * `n_len` - The length of the sequence
/// * `token_callback` - The token callback
/// # Returns
/// * The llama result
pub fn generate(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tokens_list: Vec<LlamaToken>, // Do we need this argument? what are logits?
    n_len: i32,
    token_callback: Option<TokenCallback>,
    stops: Option<&Vec<String>>
) -> Result<LlamaResult> {
    let mut batch = LlamaBatch::new(512, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .expect("llama_decode() failed");

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let t_main_start = ggml_time_us();
    let mut generated_tokens = Vec::new();
    let mut generated_tokens_data = Vec::new();

    loop {
        let mut is_last = n_cur == n_len; // Keep track of it here for the callback and use loop to save cycles
        let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
        let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        let new_token_id = ctx.sample_token_greedy(candidates_p);
        if new_token_id == model.token_eos() {
            println!("EOS token found");
            is_last = true;
        }
        let token_str = model.token_to_str(new_token_id).expect("That UTF8 shit"); // We should make EOS a blank string
        println!("{}", token_str);
        if let Some(stops) = stops {
            if stops.iter().any(|stop| token_str.eq(stop)) {
                is_last = true;
            }
        }
        if is_last{
            if let Some(ref token_callback) = token_callback {
                token_callback("".to_string(), is_last);
            }
            break;
        } //TODO: This will be re done
        generated_tokens.push(new_token_id);
        generated_tokens_data.push(token_str.clone()); //TODO: make that suck less

        if let Some(ref token_callback) = token_callback {
            token_callback(token_str, is_last);
        }
        if is_last {
            break;
        }
        batch.clear();
        batch.add(new_token_id, n_cur, &[0], true)?;

        n_cur += 1;
        ctx.decode(&mut batch).with_context(|| "failed to eval")?;
        n_decode += 1;
    }

    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    println!("{}", ctx.timings());
    let llama_result = LlamaResult {
        n_tokens: n_cur,
        n_decode: n_decode,
        duration: duration,
        generated_tokens: generated_tokens,
        generated_tokens_data: generated_tokens_data,
    };
    Ok(llama_result)
}

pub fn pretty_generate(
    model: &ModelState,
    backend: &LlamaBackend,
    prompt: &String,
    n_len: i32,
    stops: &Vec<String>,
    token_callback: Option<TokenCallback>
) -> Result<LlamaResult> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_seed(0);

    let mut ctx = model.model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // tokenize the prompt
    let tokens_list = model.model
        .str_to_token(&prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {}", prompt))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if n_kv_req > n_cxt {
        bail!(
            "n_kv_req > n_ctx, the required kv cache size is not big enough either reduce n_len or increase n_ctx"
        )
    }
    let r = generate(&model.model, &mut ctx, tokens_list, n_len, token_callback, Some(stops)).expect("failed to generate");
    Ok(r)
}



