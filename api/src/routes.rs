use axum::{http::StatusCode, Json};

use crate::types::Message;

pub async fn index() -> (StatusCode, Json<Message>) {
    let r = Message {
        message: "pong".to_string()
    };
    (StatusCode::OK, Json(r))
}