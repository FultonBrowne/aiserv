mod routes;
mod types;
mod prompt;
use std::sync::Arc;

use axum::{routing::{get, post}, Router};
use shurbai::load_models;

use crate::types::Config;

#[tokio::main]
async fn main() {
    // read the config file
    let config_data = std::fs::read_to_string("./config.json").expect("failed to read config file");
    let config: Config = serde_json::from_str(&config_data)
        .expect("failed to parse and/or assign default Json and config");
    println!("Loaded config.json");
    let model_manager = Arc::new(load_models(config.models).expect("failed to load models"));
    // build our application with a single route
    let app = Router::new()
        .route("/", get(routes::index))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completion))
        .with_state(model_manager);

    // run our app with hyper, listening globally on port 8080
    let address = "0.0.0.0:8080";
    println!("vctr listening on {}", address);
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
