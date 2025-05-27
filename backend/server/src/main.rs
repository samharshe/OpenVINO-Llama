
use std::sync::Arc;

use bytes::Buf;
use http_body_util::BodyExt;
use hyper::{
    body::{
        Incoming as Body,
        Frame
    },
    header, server::conn::http1,
    service::service_fn, Method,
    Request,
    Response,
    StatusCode
};
use hyper_util::rt::{TokioIo, TokioTimer};
use server::model_config::{ImageModelConfig, ModelConfig, ModelType};
use server::config::AppConfig;
use tokio::{
    net::TcpListener,
    sync::{
        mpsc::{unbounded_channel, UnboundedSender},
        oneshot,
        broadcast,
    },
    task::spawn_blocking,
};
use server::utils::{full, BoxBody, InferenceRequest, Result};
use wasmtime::{Config, Engine, Module};
use futures::stream::{StreamExt};
use http_body_util::StreamBody;
use tokio_stream::wrappers::BroadcastStream;

static NOT_FOUND: &[u8] = b"Not Found\n";
static MISSING_CONTENT_TYPE: &[u8] = b"Missing Content-Type.\n";
static INTERNAL_SERVER_ERROR: &[u8] = b"Internal Server Error\n";

async fn infer(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<InferenceRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>>
{
    log_sender.send("[server/main.rs] Received inference request.".to_string()).ok();

    if let Some(content_type) = request.headers().get(header::CONTENT_TYPE) {
        let content_type_str = content_type.to_str().unwrap_or("");
        
        // Determine model type based on content type
        let model_type = match content_type_str {
            "image/jpeg" => {
                log_sender.send("[server/main.rs] Detected image/jpeg content type.".to_string()).ok();
                ModelType::Image
            },
            "text/plain" => {
                log_sender.send("[server/main.rs] Detected text/plain content type.".to_string()).ok();
                ModelType::Text
            },
            _ => {
                log_sender.send(format!("[server/main.rs] Unsupported media type: {}", content_type_str)).ok();
                return Ok(Response::builder()
                    .status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
                    .body(full("Unsupported media type. Expected image/jpeg or text/plain."))
                    .unwrap());
            }
        };
        
        let mut body = request.collect().await?.aggregate();
        let bytes = body.copy_to_bytes(body.remaining());

        log_sender.send(format!("[server/main.rs] Forwarding data to {} model for processing.", 
            match model_type {
                ModelType::Image => "image",
                ModelType::Text => "text",
                _ => "unknown"
            }
        )).ok();
        
        let (sender, receiver) = oneshot::channel();
        inference_thread_sender.send(InferenceRequest {
            data: bytes.to_vec(),
            model_type,
            responder: sender,
        })?;

        log_sender.send("[server/main.rs] Passed raw data to inferencer. Waiting for inference result.".to_string()).ok();
        log_sender.send("[server/main.rs] (No logs during inference because this is performed by WASM module.)".to_string()).ok();
        return match receiver.await {
            Ok(response) => {
                let json = serde_json::to_string(&response).unwrap_or_else(|_| {
                    log_sender.send("[server/main.rs] Error serializing inference response.".to_string()).ok();
                    "{}".to_string()
                });
                log_sender.send("[server/main.rs] Inference successful. Sending response.".to_string()).ok();
                Ok(Response::builder().header(header::CONTENT_TYPE, "application/json").body(full(json)).unwrap())
            },
            Err(_) => {
                log_sender.send("[server/main.rs] Inference task failed or channel closed.".to_string()).ok();
                Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(full(INTERNAL_SERVER_ERROR))
                .unwrap())
            },
        };
    }

    log_sender.send("[server/main.rs] Inference request missing Content-Type header.".to_string()).ok();
    Ok(Response::builder().status(StatusCode::BAD_REQUEST).body(full(MISSING_CONTENT_TYPE)).unwrap())
}

async fn logs(
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>> {
    let rx = log_sender.subscribe();

    let stream = BroadcastStream::new(rx)
        .filter_map(|msg| {
            use futures::future::ready;

            match msg {
                Ok(data) => {
                    ready(Some(Ok(Frame::data(bytes::Bytes::from(format!("data: {}\n\n", data))))))
                },
                Err(e) => {
                    eprintln!("SSE stream error: {}", e);
                    ready(None)
                }
            }
        });

    let body = StreamBody::new(stream);

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(header::ACCESS_CONTROL_ALLOW_METHODS, "GET, OPTIONS")
        .header(header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type, x-session-id")
        .body(BoxBody::new(body))?;

    Ok(response)
}

async fn serve(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<InferenceRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>> {
    if request.method() == Method::OPTIONS {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .header(header::ACCESS_CONTROL_ALLOW_METHODS, "POST, OPTIONS, GET")
            .header(header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type, x-session-id")
            .body(full(""))
            .unwrap());
    }

    let mut response = match (request.method(), request.uri().path()) {
        (&Method::GET, "/logs") => logs(log_sender.clone()).await?,
        (&Method::POST, "/infer") => infer(request, inference_thread_sender, log_sender.clone()).await?,
        _ => {
            log_sender.send(format!("[server/main.rs] Unhandled request: {} {}", request.method(), request.uri().path())).ok();
            Response::builder().status(StatusCode::NOT_FOUND).body(full(NOT_FOUND)).unwrap()
        },
    };

    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    headers.insert(header::ACCESS_CONTROL_ALLOW_METHODS, "POST, OPTIONS, GET".parse().unwrap());
    headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type, x-session-id".parse().unwrap());

    Ok(response)
}

#[tokio::main]
pub async fn main() -> anyhow::Result<()>
{
    // Load configuration
    let config = AppConfig::load();
    
    let addr = config.socket_addr();
    let listener = TcpListener::bind(addr).await?;
    println!("Server listening on {}", addr);
    
    let (tx, mut rx) = unbounded_channel::<InferenceRequest>();

    let (log_tx, _log_rx) = broadcast::channel::<String>(16);
    let log_tx_inference = log_tx.clone();

    // Clone config for the inference thread
    let config_inference = config.clone();
    
    tokio::spawn(async move {
        log_tx_inference.send("Inference thread is active and working.".to_string()).ok();
        let engine = Arc::new(Engine::new(&Config::new()).unwrap());
        let module =
            Arc::new(Module::from_file(&engine, &config_inference.wasm_module_path).unwrap());
        
        // Create model instances for each type
        let image_model = Arc::new(ImageModelConfig::new(
            engine.clone(),
            module.clone(),
            log_tx_inference.clone(),
            config_inference.image_model.name.clone(),
            config_inference.image_model.version.clone()
        ));
        
        // TODO: Create TextModelConfig when implemented
        // let text_model = Arc::new(TextModelConfig::new(
        //     engine.clone(),
        //     module.clone(),
        //     log_tx_inference.clone(),
        //     config_inference.text_model.name.clone(),
        //     config_inference.text_model.version.clone()
        // ));
        
        while let Some(request) = rx.recv().await {
            match request.model_type {
                ModelType::Image => {
                    let model = Arc::clone(&image_model);
                    spawn_blocking(move || -> anyhow::Result<()> {
                        let result = model.infer(&request.data)
                            .expect("Inference failed");
                        request.responder.send(result).unwrap();
                        Ok(())
                    });
                },
                ModelType::Text => {
                    // TODO: Use text model when implemented
                    log_tx_inference.send("[server/main.rs] Text model not yet implemented.".to_string()).ok();
                    request.responder.send(serde_json::json!({
                        "error": "Text model not yet implemented"
                    })).unwrap();
                },
                _ => {
                    log_tx_inference.send("[server/main.rs] Unsupported model type in inference thread.".to_string()).ok();
                    request.responder.send(serde_json::json!({
                        "error": "Unsupported model type"
                    })).unwrap();
                }
            }
        }
    });

    loop {
        let (tcp, _) = listener.accept().await?;
        let io = TokioIo::new(tcp);
        let tx = tx.clone();
        let log_tx_clone = log_tx.clone();
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .timer(TokioTimer::new())
                .serve_connection(io, service_fn(move |req| serve(req, tx.clone(), log_tx_clone.clone())))
                .await
            {
                println!("Error serving connection: {:?}", err);
            }
        });
    }
}