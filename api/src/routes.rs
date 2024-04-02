use std::sync::{Arc, Mutex};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use shurbai::{
    pretty_generate,
    types::{LlamaResult, ModelManager},
};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    get_model, prompt, tools,
    types::{
        ChatGenerateCall, ChatGenerateResponse, ChatGenerateResponseChuck, ErrorResponse,
        GenerateCall, GenerateResponse, GeneratreResponseChuck, ListModelsResponse, Message,
        ModelListObject, ServerMetadata, XmlState,
    },
    utils::{self, has_model, process_xml_token},
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
    if !has_model(&model_manager.as_ref(), &request_body.model) {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new("Model not found")),
        )
            .into_response();
    }
    let mut messages = request_body.messages.clone();
    let has_tools = request_body.tools.is_some(); // Emoty tools should be handled client side
    if has_tools {
        let defs_text = tools::build_tool_defs_text(request_body.tools.unwrap());
        messages.insert(0, Message::new("tool", &defs_text));
    }
    let model_state = get_model!(&model_manager, &request_body.model);

    let prompt = prompt::generate_chat_prompt(&messages, &model_state.chat_template)
        .expect("Failed to generate prompt");

    if request_body.stream.unwrap_or(false) {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let model_manager = model_manager.clone();
        let prompt = prompt.clone();
        task::spawn(async move {
            let model_name = request_body.model.clone();
            let model_state = get_model!(&model_manager, &model_name); // I don't like this, but we need it for the threading
            let tx_arc = Arc::new(tokio::sync::Mutex::new(tx));
            let tx_arc_ref = Arc::clone(&tx_arc);
            let xml_state = Arc::new(Mutex::new(XmlState::new()));
            let r = pretty_generate(
                model_state,
                &model_manager.backend,
                &prompt,
                512,
                &model_state.chat_template.stops,
                Some(Box::new(move |s, is_last| {
                    let tx_arc = Arc::clone(&tx_arc_ref);
                    let mut xml_state = xml_state.lock().unwrap();
                    if has_tools && !is_last {
                        let t = process_xml_token(&mut xml_state, &s);
                        if t.is_some() {
                            // Pull out tool calls
                            let ts = t.unwrap();
                            print!("ts {:?}", ts);
                            let tool_call = tools::parse_tool_call(&ts);
                            let tool_calls = if tool_call.is_some() {
                                vec![tool_call.unwrap()]
                            } else {
                                Vec::new()
                            };
                            // send them
                            utils::send_to_stream(
                                Arc::clone(&tx_arc),
                                &ChatGenerateResponseChuck::new_tool_call(&model_name, tool_calls),
                            );
                        }
                    }
                    if !xml_state.halt_output && &s != ">" {
                        //Temp hack to avoid sending the last token
                        let block = if !is_last {
                            ChatGenerateResponseChuck::new_token(&model_name, &s)
                        } else {
                            ChatGenerateResponseChuck::new_halt(&s, "done")
                        };
                        utils::send_to_stream(Arc::clone(&tx_arc), &block);
                    }
                })),
                Some(false),
            )
            .expect("Failed to generate"); //TODO: At some point lets return the full info to the user
        });
        let rx_stream = ReceiverStream::new(rx);
        return StreamBodyAs::json_nl(rx_stream).into_response();
    }

    let response = pretty_generate(
        model_state,
        &model_manager.backend,
        &prompt,
        512,
        &model_state.chat_template.stops,
        None,
        Some(false),
    )
    .expect("Failed to generate");

    let obj = ChatGenerateResponse {
        meta: ServerMetadata::new(),
        response: response.generated_tokens_data.concat(),
        model: request_body.model.clone(),
        took: response.duration.as_nanos(),
        tool_calls: None, //TODO: Need to re do the parsing here
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
