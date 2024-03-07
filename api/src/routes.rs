use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use shurbai::{pretty_generate, types::ModelManager};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    get_model,
    types::{
        ChatCompletionsRequest, ErrorResponse, GenerateCall, GenerateResponse,
        GeneratreResponseChuck, ListModelsResponse, ModelListObject, ServerMetadata,
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

    if request_body.stream.unwrap_or(false) {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        task::spawn(async move {
            let model_state = get_model!(&model_manager, &request_body.model); // I don't like this, but we need it for the threading
            let tx_arc = Arc::new(tokio::sync::Mutex::new(tx));
            pretty_generate(
                model_state,
                &model_manager.backend,
                &request_body.prompt,
                512,
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
        512,
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
    Json(request_body): Json<ChatCompletionsRequest>,
) -> impl IntoResponse {
}

pub async fn list_models(State(model_manager): State<Arc<ModelManager>>) -> impl IntoResponse {
    let models = model_manager
        .models
        .iter()
        .map(|(name, _model)| ModelListObject {
            name: name.to_string(),
            type_str: "model".clone().to_string(),
        })
        .collect();
    let r = ListModelsResponse {
        meta: ServerMetadata::new(),
        models,
    };
    (StatusCode::OK, Json(r))
}
