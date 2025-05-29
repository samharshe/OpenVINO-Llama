use std::{fs, sync::Mutex};
use log::{error, info};

use inferencer::{MobilnetModel, TextModel, registry::{ModelRegistry, RegisteredModel, ModelMetadata}, preprocessing, text_preprocessing};

// Removed hardcoded ImageNet labels - models should handle their own output formatting

static MODEL_REGISTRY: Mutex<Option<ModelRegistry>> = Mutex::new(None);

fn main() {
    // Initialize logger for WASM
    #[cfg(target_arch = "wasm32")]
    wasm_logger::init(wasm_logger::Config::default());
    
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    
    info!("Inferencer initialized");
}

fn ensure_registry() -> Result<u32, String> {
    let mut registry_guard = MODEL_REGISTRY.lock().unwrap();
    
    if registry_guard.is_none() {
        *registry_guard = Some(ModelRegistry::new());
        info!("Model registry initialized");
    }
    
    let registry = registry_guard.as_mut().unwrap();
    
    // Load default model if no models are registered
    if registry.list_models().is_empty() {
        let xml = fs::read_to_string("fixture/model.xml")
            .map_err(|e| format!("Failed to read model.xml: {}", e))?;
        let weights = fs::read("fixture/model.bin")
            .map_err(|e| format!("Failed to read model.bin: {}", e))?;
        
        let model = MobilnetModel::from_buffer_result(xml.into_bytes(), weights)?;
        let metadata = ModelMetadata {
            name: "mobilenet_v3_large".to_string(),
            version: "1.0".to_string(),
            model_type: "image".to_string(),
        };
        
        let id = registry.register_model(RegisteredModel::ImageNet(model), metadata);
        info!("Default image model loaded with ID: {}", id);
        
        // Load text model (unconditionally, fail if not available)
        info!("Loading TinyLlama text model...");
        
        let tokenizer_json = std::fs::read("fixture/text_model/tokenizer.json")
            .map_err(|e| format!("Failed to read tokenizer.json: {}", e))?;
        
        let xml = std::fs::read("fixture/text_model/openvino_model.xml")
            .map_err(|e| format!("Failed to read text model XML: {}", e))?;
        
        let weights = std::fs::read("fixture/text_model/openvino_model.bin")
            .map_err(|e| format!("Failed to read text model weights: {}", e))?;
        
        info!("Loaded TinyLlama model files - XML: {} bytes, weights: {} bytes", 
             xml.len(), weights.len());
        
        let (text_model, tokenizer) = TextModel::from_buffer_with_tokenizer(
            xml, 
            weights, 
            tokenizer_json
        ).map_err(|e| format!("Failed to create text model: {}", e))?;
        
        let text_metadata = ModelMetadata {
            name: "tinyllama-1.1b-chat".to_string(),
            version: "v1.0-int4".to_string(),
            model_type: "text".to_string(),
        };
        
        let text_id = registry.register_model(
            RegisteredModel::Text { model: text_model, tokenizer },
            text_metadata
        );
        
        info!("TinyLlama model loaded successfully with ID: {}", text_id);
        
        
        Ok(1) // Return ID of first model (image model)
    } else {
        Ok(1) // Return ID of first model
    }
}

#[no_mangle]
pub extern "C" fn register_model(config_ptr: i32, config_len: i32) -> i32 {
    if config_ptr == 0 || config_len <= 0 {
        error!("Invalid parameters: config_ptr={}, config_len={}", config_ptr, config_len);
        return -1;
    }
    
    let config_bytes = unsafe { std::slice::from_raw_parts(config_ptr as *const u8, config_len as usize) };
    let config_str = match std::str::from_utf8(config_bytes) {
        Ok(s) => s,
        Err(e) => {
            error!("Invalid UTF-8 in config: {}", e);
            return -1;
        }
    };
    
    // Parse JSON config
    let _config: serde_json::Value = match serde_json::from_str(config_str) {
        Ok(v) => v,
        Err(e) => {
            error!("Invalid JSON config: {}", e);
            return -1;
        }
    };
    
    // For now, only support loading the default model
    match ensure_registry() {
        Ok(id) => id as i32,
        Err(e) => {
            error!("Failed to register model: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn load_model(_xml_ptr: i32, _xml_len: i32, _weights_ptr: i32, _weights_len: i32) -> i32 {
    // Backward compatibility - register model with default metadata
    ensure_registry().map(|_| 0).unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn infer(data_ptr: i32, data_len: i32, result_ptr: i32) -> i32 {
    // Auto-detect input type and route to appropriate model
    let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len as usize) };
    
    // Detect input type
    let model_id = if preprocessing::is_jpeg(data) || preprocessing::is_tensor(data) {
        1 // Image model
    } else if text_preprocessing::is_text(data) {
        2 // Text model (if registered)
    } else {
        error!("Unable to detect input type");
        return -1;
    };
    
    info!("Auto-detected model ID: {} for input", model_id);
    infer_with_model(data_ptr, data_len, result_ptr, model_id)
}

#[no_mangle]
pub extern "C" fn infer_with_model(data_ptr: i32, data_len: i32, result_ptr: i32, model_id: i32) -> i32 {
    // Basic validation
    if data_ptr == 0 || data_len <= 0 || result_ptr == 0 {
        error!("Invalid parameters: data_ptr={}, data_len={}, result_ptr={}, model_id={}", 
               data_ptr, data_len, result_ptr, model_id);
        return -1;
    }
    
    let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len as usize) };
    
    // Ensure registry and get model
    if let Err(e) = ensure_registry() {
        error!("Failed to ensure registry: {}", e);
        return -2;
    }
    
    let registry_guard = MODEL_REGISTRY.lock().unwrap();
    let registry = registry_guard.as_ref().unwrap();
    
    let (model, metadata) = match registry.get_model(model_id as u32) {
        Some(entry) => entry,
        None => {
            error!("Model with ID {} not found", model_id);
            return -3;
        }
    };
    
    // Run inference - the model returns the complete response format
    let response = match model.infer(data, metadata) {
        Ok(json_result) => json_result,
        Err(e) => {
            error!("Inference failed: {}", e);
            return -4;
        }
    };
    
    let json_str = response.to_string();
    let json_bytes = json_str.as_bytes();
    
    unsafe {
        let result_u8_ptr = result_ptr as *mut u8;
        // Write length first, then the JSON string
        std::ptr::write(result_u8_ptr as *mut u32, json_bytes.len() as u32);
        std::ptr::copy_nonoverlapping(json_bytes.as_ptr(), result_u8_ptr.add(4), json_bytes.len());
    }
    
    0 // Success
}

#[no_mangle]
pub extern "C" fn register_text_model(
    xml_ptr: i32, 
    xml_len: i32, 
    weights_ptr: i32, 
    weights_len: i32,
    tokenizer_ptr: i32,
    tokenizer_len: i32,
    metadata_ptr: i32,
    metadata_len: i32
) -> i32 {
    // Validate parameters
    if xml_ptr == 0 || xml_len <= 0 || weights_ptr == 0 || weights_len <= 0 ||
       tokenizer_ptr == 0 || tokenizer_len <= 0 || metadata_ptr == 0 || metadata_len <= 0 {
        error!("Invalid parameters for text model registration");
        return -1;
    }
    
    // Get data slices
    let xml = unsafe { std::slice::from_raw_parts(xml_ptr as *const u8, xml_len as usize) };
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr as *const u8, weights_len as usize) };
    let tokenizer_json = unsafe { std::slice::from_raw_parts(tokenizer_ptr as *const u8, tokenizer_len as usize) };
    let metadata_bytes = unsafe { std::slice::from_raw_parts(metadata_ptr as *const u8, metadata_len as usize) };
    
    // Parse metadata
    let metadata: ModelMetadata = match serde_json::from_slice(metadata_bytes) {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to parse metadata: {}", e);
            return -2;
        }
    };
    
    // Create text model with tokenizer
    let (model, tokenizer) = match TextModel::from_buffer_with_tokenizer(
        xml.to_vec(), 
        weights.to_vec(), 
        tokenizer_json.to_vec()
    ) {
        Ok((m, t)) => (m, t),
        Err(e) => {
            error!("Failed to create text model: {}", e);
            return -3;
        }
    };
    
    // Register in the registry
    let mut registry_guard = MODEL_REGISTRY.lock().unwrap();
    if registry_guard.is_none() {
        *registry_guard = Some(ModelRegistry::new());
    }
    let registry = registry_guard.as_mut().unwrap();
    
    let id = registry.register_model(
        RegisteredModel::Text { model, tokenizer },
        metadata
    );
    
    info!("Text model registered with ID: {}", id);
    id as i32
}
