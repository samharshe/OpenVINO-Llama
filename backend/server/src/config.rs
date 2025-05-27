use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub wasm_module_path: PathBuf,
    pub image_model: ImageModelSpec,
    pub text_model: TextModelSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageModelSpec {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextModelSpec {
    pub name: String,
    pub version: String,
    pub max_input_length: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
            },
            wasm_module_path: PathBuf::from("../target/wasm32-wasip1/debug/inferencer.wasm"),
            image_model: ImageModelSpec {
                name: "mobilenet_v3_large".to_string(),
                version: "1.0".to_string(),
            },
            text_model: TextModelSpec {
                name: "llama2_7b".to_string(),
                version: "2.0".to_string(),
                max_input_length: 10240,
            },
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        // For now, just use defaults
        // Future: Add environment variable overrides
        // Future: Add JSON file loading
        Self::default()
    }
    
    pub fn socket_addr(&self) -> std::net::SocketAddr {
        format!("{}:{}", self.server.host, self.server.port)
            .parse()
            .expect("Invalid socket address")
    }
}