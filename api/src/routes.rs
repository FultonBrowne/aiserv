use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use shurbai::{
    pretty_generate,
    types::{LlamaResult, ModelManager},
};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    get_model, prompt,
    tools::predict_tool_calls,
    types::{
        ChatGenerateCall, ChatGenerateResponse, ChatGenerateResponseChuck, ErrorResponse,
        GenerateCall, GenerateResponse, GeneratreResponseChuck, ListModelsResponse,
        ModelListObject, ServerMetadata, ToolCall,
    },
    utils::{self, has_model},
};

pub async fn generate(
    State(model_manager): State<Arc<ModelManager>>,
    Json(request_body): Json<GenerateCall>,
) -> impl IntoResponse {
    if !has_model(model_manager.as_ref(), &request_body.model) {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new("Model not found")),
        )
            .into_response();
    }

    let max_tokens = request_body
        .generate_params
        .as_ref()
        .map_or(512, |p| p.max_tokens.unwrap());

    if request_body.stream.unwrap_or(false) {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        task::spawn(async move {
            let model_state = get_model!(&model_manager, &request_body.model); // I don't like this, but we need it for the threading
            let tx_arc = Arc::new(tokio::sync::Mutex::new(tx));
            pretty_generate(
                model_state,
                &model_manager.backend,
                &request_body.prompt,
                max_tokens,
                &model_state.chat_template.stops,
                Some(Box::new(move |s, is_last| {
                    utils::send_to_stream(
                        Arc::clone(&tx_arc),
                        &GeneratreResponseChuck {
                            meta: ServerMetadata::new(),
                            token_str: s,
                            model: request_body.model.clone(),
                            halt_reason: if is_last {
                                Some("End of stream".to_string())
                            } else {
                                None
                            },
                        },
                    );
                })),
                Some(false),
            )
            .expect("Failed to generate"); //TODO: At some point lets return the full info to the user
        });
        let rx_stream = ReceiverStream::new(rx);
        return StreamBodyAs::json_nl(rx_stream).into_response();
    }

    let model_state = get_model!(&model_manager, &request_body.model);
    let response = pretty_generate(
        model_state,
        &model_manager.backend,
        &request_body.prompt,
        max_tokens,
        &model_state.chat_template.stops,
        None,
        Some(false),
    )
    .expect("Failed to generate");
    println!("response {:?}", response.generated_tokens_data);
    let obj = GenerateResponse {
        meta: ServerMetadata::new(),
        response: response.generated_tokens_data.concat(),
        model: request_body.model.clone(),
        took: response.duration.as_nanos(),
        halt_reason: None,
    };
    (StatusCode::OK, Json(obj)).into_response()
}

pub async fn chat_generate(
    State(model_manager): State<Arc<ModelManager>>,
    Json(request_body): Json<ChatGenerateCall>,
) -> impl IntoResponse {
    if !has_model(model_manager.as_ref(), &request_body.model) {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new("Model not found")),
        )
            .into_response();
    }
    let mut tool_calls = Vec::<ToolCall>::new();
    let tool_calls_only = &request_body.tool_call_only.unwrap_or(false);
    let has_tools = request_body.tools.is_some() && request_body.tools.as_ref().unwrap().len() > 0;
    let model_state = get_model!(&model_manager, &request_body.model);
    if has_tools {
        predict_tool_calls(
            model_state,
            &model_manager,
            &mut request_body.messages.clone(),
            &mut tool_calls,
            &request_body.tools.unwrap(),
        );
    } else if tool_calls_only.clone() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "No tools provided but tool_call_only is set to true",
            )),
        )
            .into_response();
    }

    let prompt = prompt::generate_chat_prompt(&request_body.messages, &model_state.chat_template)
        .expect("Failed to generate prompt");

    if request_body.stream.unwrap_or(false) {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let model_manager = model_manager.clone();
        let tx_arc = Arc::new(tokio::sync::Mutex::new(tx));
        println!("{:?}", tool_calls);
        if !tool_calls.is_empty() {
            println!("Sending tool calls");
            utils::send_to_stream(
                tx_arc.clone(),
                &ChatGenerateResponseChuck {
                    meta: ServerMetadata::new(),
                    token_str: "".to_string(),
                    role: "assistant".to_string(),
                    model: request_body.model.clone(),
                    halt_reason: None,
                    tool_calls: Some(tool_calls),
                },
            );
        }
        if !tool_calls_only {
            let prompt = prompt.clone();
            task::spawn(async move {
                let model_state = get_model!(&model_manager, &request_body.model); // I don't like this, but we need it for the threading
                pretty_generate(
                    model_state,
                    &model_manager.backend,
                    &prompt,
                    512,
                    &model_state.chat_template.stops,
                    Some(Box::new(move |s, is_last| {
                        utils::send_to_stream(
                            Arc::clone(&tx_arc),
                            &ChatGenerateResponseChuck {
                                meta: ServerMetadata::new(),
                                token_str: s,
                                role: "assistant".to_string(),
                                model: request_body.model.clone(),
                                halt_reason: if is_last {
                                    Some("End of stream".to_string())
                                } else {
                                    None
                                },
                                tool_calls: None,
                            },
                        );
                    })),
                    Some(false),
                )
                .expect("Failed to generate"); //TODO: At some point lets return the full info to the user
            });
        }
        let rx_stream = ReceiverStream::new(rx);
        return StreamBodyAs::json_nl(rx_stream).into_response();
    }

    let response = if tool_calls_only.clone() {
        // TODO: dont't clone the bool lazy ass
        pretty_generate(
            model_state,
            &model_manager.backend,
            &prompt,
            512,
            &model_state.chat_template.stops,
            None,
            Some(false),
        )
        .expect("Failed to generate")
    } else {
        LlamaResult::default()
    };

    let obj = ChatGenerateResponse {
        meta: ServerMetadata::new(),
        response: response.generated_tokens_data.concat(),
        model: request_body.model.clone(),
        took: response.duration.as_nanos(),
        tool_calls: Some(tool_calls),
        halt_reason: None,
    };
    return (StatusCode::OK, Json(obj)).into_response();
}

pub async fn list_models(State(model_manager): State<Arc<ModelManager>>) -> impl IntoResponse {
    let models = model_manager
        .models
        .iter()
        .map(|(name, _model)| ModelListObject {
            name: name.to_string(),
            type_str: "model".to_string(),
        })
        .collect();
    let r = ListModelsResponse {
        meta: ServerMetadata::new(),
        models,
    };
    (StatusCode::OK, Json(r))
}
