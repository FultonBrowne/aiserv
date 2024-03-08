mod prompt;
mod routes;
mod tools;
mod types;
mod utils;
use std::sync::Arc;

use crate::types::Config;
use axum::{
    routing::{get, post},
    Router,
};
use shurbai::load_models;

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
        .route("/models", get(routes::list_models))
        .route("/generate", post(routes::generate))
        .route("/generate/chat", post(routes::chat_generate))
        .with_state(model_manager);

    // run our app with hyper, listening globally on port 8080
    let address = "0.0.0.0:8080";
    let art = "

        ███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗
        ██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝
        █████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝
        ██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝
        ██║     ██║  ██║██║██████╔╝██║  ██║   ██║
        ╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝
        ";
    println!("{}", art);
    println!("       Developed by Shurburt LLC. (c) 2024");
    println!("       AI server listening on {}", address);
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
