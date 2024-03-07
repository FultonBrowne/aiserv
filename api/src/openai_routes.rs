use std::{sync::Arc, vec};

use axum_streams::StreamBodyAs;

use tokio::task;
use tokio_stream::wrappers::ReceiverStream;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use shurbai::{pretty_generate, types::ModelManager};

use tokio::sync::{mpsc, Mutex};

use crate::{
    prompt, tools::{self}, types::{
        ChatCompletionsRequest, ChatCompletionsResponse, ChoiceDelta, ChoiceObject, FunctionCall, Message, ModelResponse, ModelsResponseOutput, StatusMessage, StreamChoice, StreamResponseChunk
    }
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
    println!("Request body: {:}", serde_json::to_string_pretty(&request_body).unwrap());
    let model_name = request_body.model.clone();
    if !model_manager.models.contains_key(&model_name) {
        //return (StatusCode::NOT_FOUND, Json(None));
        panic!("Model not found: {}", model_name) // make this good later
    }
    let mut function_calls = vec![];
    if request_body.function_call.is_some() {
        function_calls.extend(tools::predict_tool_calls());
    }
    let model = model_manager.models.get(&model_name).unwrap();
    let prompt = prompt::generate_chat_prompt(&request_body.messages, &model.chat_template)
        .expect("failed to generate prompt");
    let json_format = !request_body.response_format.is_none();
    if request_body.stream.unwrap_or(false) {
        return chat_stream(
            model_name,
            &model_manager,
            prompt,
            &request_body,
            json_format,
            function_calls,
        )
        .into_response();
    }
    return chat_no_stream(model, &model_manager, prompt, &request_body, json_format, function_calls)
        .into_response();
}
fn chat_stream(
    model_name: String,
    model_manager: &Arc<ModelManager>,
    prompt: String,
    request: &ChatCompletionsRequest,
    json_format: bool,
    function_calls: Vec<FunctionCall>,
) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<String>(32);
    let tx = Arc::new(Mutex::new(tx)); // Wrap tx in Arc<Mutex<T>>
    let model_manager_clone = Arc::clone(model_manager);
    let request_body_clone = request.clone();

    tokio::spawn(async move {
        let tx_clone = tx.clone();
        let model_name_clone = model_name.clone();
        task::spawn(async move {
            let init_assistant = StreamResponseChunk {
                id: "chatcmpl-123".to_string(),
                created: 1694268190, // TODO: make it real
                system_fingerprint: "fp_44709d6fcb".to_string(),
                object: "chat.completion.chunk".to_string(),
                model: model_name_clone,
                choices: vec![StreamChoice {
                    index: 0,
                    delta: ChoiceDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                        function_call: function_calls.get(0).cloned()
                    },
                    logprobs: None,
                    finish_reason: None,
                }],
            };
            let init_str = serde_json::to_string(&init_assistant).unwrap();
            let _ = tx_clone
                .lock()
                .await
                .send(format!("data: {} \n\n", init_str))
                .await;
        });
        let model = model_manager_clone
            .models
            .get(&model_name)
            .expect("Model not found");
        let request_body = request_body_clone;
        let g = pretty_generate(
            model,
            &model_manager_clone.backend,
            &prompt,
            request_body.max_tokens.unwrap_or(32),
            &model.chat_template.stops,
            Some(Box::new(move |s, is_last| {
                // Modify the closure to take two arguments
                let model_name_clone = model_name.clone();
                let response_chunk = StreamResponseChunk {
                    id: "chatcmpl-123".to_string(),
                    created: 1694268190, // TODO: make it real
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    model: model_name_clone,
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: ChoiceDelta {
                            role: None,
                            content: Some(s.clone()),
                            function_call: None,
                        },
                        logprobs: None,
                        finish_reason: if is_last {
                            Some("complete".to_string())
                        } else {
                            None
                        },
                    }],
                };
                let json_str =
                    serde_json::to_string(&response_chunk).expect("failed to serialize response");
                let tx_clone = tx.clone();
                task::spawn(async move {
                    let _ = tx_clone
                        .lock()
                        .await
                        .send(format!("data: {} \n\n", json_str))
                        .await;
                    if is_last {
                        let _ = tx_clone
                            .lock()
                            .await
                            .send("data: [DONE]\n\n".to_string())
                            .await;
                    }
                });
            })),
            Some(json_format),
        )
        .expect("failed to generate response");
        println!("Generated response: {:?}", g.generated_tokens_data);
    });
    let rx_stream = ReceiverStream::new(rx);
    StreamBodyAs::text(rx_stream)
}

fn chat_no_stream(
    model: &shurbai::types::ModelState,
    model_manager: &Arc<ModelManager>,
    prompt: String,
    request_body: &ChatCompletionsRequest,
    json_format: bool,
    _function_calls: Vec<FunctionCall>,
) -> impl IntoResponse {
    let g = pretty_generate(
        model,
        &model_manager.backend,
        &prompt,
        request_body.max_tokens.unwrap_or(32),
        &model.chat_template.stops,
        None,
        Some(json_format),
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
    let response = Json(r);
    response
}
