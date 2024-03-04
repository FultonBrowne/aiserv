use axum::{http::StatusCode, Json};

use crate::types::StatusMessage;

pub async fn index() -> (StatusCode, Json<StatusMessage>) {
    let r = StatusMessage {
        message: "pong".to_string()
    };
    (StatusCode::OK, Json(r))
}