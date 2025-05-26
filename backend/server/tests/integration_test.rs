use serde_json::Value;
use std::fs;

const SERVER_URL: &str = "http://127.0.0.1:3000";

#[tokio::test]
async fn test_image_classification_preserves_functionality() {
    // This characterization test ensures image classification continues working
    // as we refactor the architecture. Tests the core contract:
    // JPEG in -> JSON classification response out
    
    let client = reqwest::Client::new();
    
    // Read test image
    let image_data = fs::read("tests/fixtures/3.jpeg")
        .expect("Failed to read test image");
    
    // Send POST request to /infer endpoint
    let response = client
        .post(format!("{}/infer", SERVER_URL))
        .header("Content-Type", "image/jpeg")
        .body(image_data)
        .send()
        .await;
    
    // Server must be running for tests to pass
    let resp = response.expect("Failed to connect to server - is it running on port 3000?");
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("content-type").unwrap(), "application/json");
    
    let json: Value = resp.json().await.expect("Response should be valid JSON");
    
    // Response should be object with output, metadata, and model_info
    assert!(json.is_object());
    let obj = json.as_object().unwrap();
    
    // Should have the three required top-level fields
    assert!(obj.contains_key("output"));
    assert!(obj.contains_key("metadata"));
    assert!(obj.contains_key("model_info"));
    
    // Output should contain the actual inference result
    let output = &obj["output"];
    assert!(!output.is_null());
}

#[tokio::test]
async fn test_server_cors_headers() {
    let client = reqwest::Client::new();
    
    let response = client
        .request(reqwest::Method::OPTIONS, format!("{}/infer", SERVER_URL))
        .send()
        .await;
    
    let resp = response.expect("Failed to connect to server - is it running on port 3000?");
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("access-control-allow-origin").unwrap(), "*");
    assert!(resp.headers().get("access-control-allow-methods").is_some());
    assert!(resp.headers().get("access-control-allow-headers").is_some());
}

#[tokio::test]
async fn test_unsupported_content_type() {
    let client = reqwest::Client::new();
    
    let response = client
        .post(format!("{}/infer", SERVER_URL))
        .header("Content-Type", "application/xml")
        .body("<xml>not supported</xml>")
        .send()
        .await;
    
    let resp = response.expect("Failed to connect to server - is it running on port 3000?");
    assert_eq!(resp.status(), 415); // Unsupported Media Type
}

#[tokio::test]
async fn test_missing_content_type() {
    let client = reqwest::Client::new();
    
    let response = client
        .post(format!("{}/infer", SERVER_URL))
        .body("some data")
        .send()
        .await;
    
    let resp = response.expect("Failed to connect to server - is it running on port 3000?");
    assert_eq!(resp.status(), 400); // Bad Request
}

#[tokio::test]
async fn test_logs_endpoint() {
    let client = reqwest::Client::new();
    
    let response = client
        .get(format!("{}/logs", SERVER_URL))
        .send()
        .await;
    
    let resp = response.expect("Failed to connect to server - is it running on port 3000?");
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("content-type").unwrap(), "text/event-stream");
}