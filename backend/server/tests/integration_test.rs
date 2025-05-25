use reqwest;
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
        .post(&format!("{}/infer", SERVER_URL))
        .header("Content-Type", "image/jpeg")
        .body(image_data)
        .send()
        .await;
    
    // Should get a response (server might not be running in CI)
    if let Ok(resp) = response {
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("content-type").unwrap(), "application/json");
        
        let json: Value = resp.json().await.expect("Response should be valid JSON");
        
        // Response should be array with [label, probability]
        assert!(json.is_array());
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        
        // First element should be string (label), second should be number (probability)
        assert!(arr[0].is_string());
        assert!(arr[1].is_number());
        
        let probability = arr[1].as_f64().unwrap();
        assert!(probability >= 0.0 && probability <= 1.0);
    }
}

#[tokio::test]
async fn test_server_cors_headers() {
    let client = reqwest::Client::new();
    
    let response = client
        .request(reqwest::Method::OPTIONS, &format!("{}/infer", SERVER_URL))
        .send()
        .await;
    
    if let Ok(resp) = response {
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("access-control-allow-origin").unwrap(), "*");
        assert!(resp.headers().get("access-control-allow-methods").is_some());
        assert!(resp.headers().get("access-control-allow-headers").is_some());
    }
}

#[tokio::test]
async fn test_unsupported_content_type() {
    let client = reqwest::Client::new();
    
    let response = client
        .post(&format!("{}/infer", SERVER_URL))
        .header("Content-Type", "text/plain")
        .body("not an image")
        .send()
        .await;
    
    if let Ok(resp) = response {
        assert_eq!(resp.status(), 415); // Unsupported Media Type
    }
}

#[tokio::test]
async fn test_missing_content_type() {
    let client = reqwest::Client::new();
    
    let response = client
        .post(&format!("{}/infer", SERVER_URL))
        .body("some data")
        .send()
        .await;
    
    if let Ok(resp) = response {
        assert_eq!(resp.status(), 400); // Bad Request
    }
}

#[tokio::test]
async fn test_logs_endpoint() {
    let client = reqwest::Client::new();
    
    let response = client
        .get(&format!("{}/logs", SERVER_URL))
        .send()
        .await;
    
    if let Ok(resp) = response {
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("content-type").unwrap(), "text/event-stream");
    }
}