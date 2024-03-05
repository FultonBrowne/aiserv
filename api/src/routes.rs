use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use shurbai::{generate_no_callback, types::ModelManager};

use crate::{prompt, types::{ChatCompletionsRequest, ChatCompletionsResponse, ChoiceObject, Message, ModelResponse, ModelsResponseOutput, StatusMessage}};

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
    Json(request_body):Json<ChatCompletionsRequest>) -> (StatusCode, Json<Option<ChatCompletionsResponse>>
    ) {
    if request_body.stream.unwrap_or(false) {
        return (StatusCode::NOT_IMPLEMENTED, Json(None))
    }
    let model_name = request_body.model.clone();
    if !model_manager.models.contains_key(&model_name) {
        return (StatusCode::NOT_FOUND, Json(None))
    }
    let model = model_manager.models.get(&model_name).unwrap();
    let prompt = prompt::generate_chat_prompt(request_body.messages, &model.chat_template).expect("failed to generate prompt");

    let g = generate_no_callback(model, &model_manager.backend, &prompt,
                                                request_body.max_tokens.unwrap_or(32)).expect("failed to generate response");
    println!("Generated response: {:?}", g.generated_tokens_data);
    let r = ChatCompletionsResponse {
        choices: vec![
            ChoiceObject {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: g.generated_tokens_data.concat(),
                    function_call: None,
                },
                logprobs: serde_json::Value::Null,
                finish_reason: "complete".to_string(),
            }
            ],
    };
    (StatusCode::OK, Json(Some(r)))
}
