use core::time;
use std::convert::Infallible;
use std::pin::Pin;
use std::{io::Write, sync::Arc};

use axum::body::Body;
use axum::http::request;
use axum::response::Response;
use axum_streams::StreamBodyAs;
use futures::Stream;
use futures::{stream, StreamExt, TryStreamExt};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use shurbai::{pretty_generate, types::ModelManager, TokenCallback};

use tokio::sync::{mpsc, Mutex};

use crate::{
    prompt,
    types::{
        ChatCompletionsRequest, ChatCompletionsResponse, ChoiceDelta, ChoiceObject, Message,
        ModelResponse, ModelsResponseOutput, StatusMessage, StreamChoice, StreamResponseChunk,
    },
};

pub async fn index() -> (StatusCode, Json<StatusMessage>) {
    let r = StatusMessage {
        message: "pong".to_string(),
    };
    (StatusCode::OK, Json(r))
}

pub async fn list_models(
    State(model_manager): State<Arc<ModelManager>>,
) -> (StatusCode, Json<ModelsResponseOutput>) {
    let mut models = Vec::new();
    for (name, model) in model_manager.models.iter() {
        models.push(ModelResponse {
            id: name.clone(),
            name: name.clone(),
            max_tokens: Some(model.config.num_ctx.unwrap_or(2048)),
            owned_by: "Shurburt".to_string(),

            object: "model".to_string(),
        });
    }
    let r = ModelsResponseOutput { data: models };
    (StatusCode::OK, Json(r))
}

pub async fn chat_completion(
    State(model_manager): State<Arc<ModelManager>>,
    Json(request_body): Json<ChatCompletionsRequest>,
) -> impl IntoResponse {
    let model_name = request_body.model.clone();
    if !model_manager.models.contains_key(&model_name) {
        //return (StatusCode::NOT_FOUND, Json(None));
        panic!("Model not found: {}", model_name) // make this good later
    }
    let model = model_manager.models.get(&model_name).unwrap();
    let prompt = prompt::generate_chat_prompt(&request_body.messages, &model.chat_template)
        .expect("failed to generate prompt");
    println!("Generated prompt: {:?}", prompt);
    if request_body.stream.unwrap_or(false) {
        return chat_stream(model_name, &model_manager, prompt, &request_body);
    }
    return chat_stream(model_name, &model_manager, prompt, &request_body);
    //return chat_no_stream(model, &model_manager, prompt, &request_body)
}
fn chat_stream(
    model_name: String,
    model_manager: &Arc<ModelManager>,
    prompt: String,
    request: &ChatCompletionsRequest,
) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::channel::<StreamResponseChunk>(32);
    let tx = Arc::new(Mutex::new(tx)); // Wrap tx in Arc<Mutex<T>>
    let model_manager_clone = Arc::clone(model_manager);
    let request_body_clone = request.clone();
    tokio::spawn(async move {
        let model = model_manager_clone.models.get(&model_name).expect("Model not found");
        let request_body = request_body_clone;
        let g = pretty_generate(
            model,
            &model_manager_clone.backend,
            &prompt,
            request_body.max_tokens.unwrap_or(32),
            &model.chat_template.stops,
            Some(Box::new(move |s, is_last| {
                // Modify the closure to take two arguments
                let response_chunk = StreamResponseChunk {
                    id: "Teehee".to_string(),
                    created: 0,
                    object: "stream".to_string(),
                    model: "phi".to_string(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: ChoiceDelta {
                            role: Some("assistant".to_string()),
                            content: Some(s.clone()),
                            tool_call: None,
                        },
                        logprobs: None,
                        finish_reason: Some(if is_last {
                            "complete".to_string()
                        } else {
                            "incomplete".to_string()
                        }),
                    }],
                };
                let tx_clone = tx.clone();
                task::spawn(async move {
                    let _ = tx_clone.lock().await.send(response_chunk).await;
                });
                std::io::stdout().flush().unwrap();
            })),
        )
        .expect("failed to generate response");
        println!("Generated response: {:?}", g.generated_tokens_data);
    });
    let rx_stream = ReceiverStream::new(rx);
    StreamBodyAs::json_nl(rx_stream)
}

fn chat_no_stream(
    model: &shurbai::types::ModelState,
    model_manager: &Arc<ModelManager>,
    prompt: String,
    request_body: &ChatCompletionsRequest,
) -> impl IntoResponse {
    let g = pretty_generate(
        model,
        &model_manager.backend,
        &prompt,
        request_body.max_tokens.unwrap_or(32),
        &model.chat_template.stops,
        None,
    )
    .expect("failed to generate response");
    println!("Generated response: {:?}", g.generated_tokens_data);
    let r = ChatCompletionsResponse {
        choices: vec![ChoiceObject {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: g.generated_tokens_data.concat(),
                function_call: None,
            },
            logprobs: serde_json::Value::Null,
            finish_reason: "complete".to_string(),
        }],
    };
    let json = serde_json::to_string(&r).unwrap();
    // Create a Response<Body> object
    let response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(json))
        .unwrap();
    response
}
