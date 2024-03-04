mod types;
mod routes;
use axum::{
    routing::get, Router
};

#[tokio::main]
async fn main() {
    // build our application with a single route
    let app = Router::new().route("/", get(routes::index));

    // run our app with hyper, listening globally on port 8080
    let address = "0.0.0.0:8080";
    println!("vctr listening on {}", address);
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}