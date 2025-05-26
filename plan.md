# Implementation Plan: OpenVINO-Llama Universal ML Inference Demo

## Overview
Transform existing v0.1 image classification prototype into a clean, flexible universal ML inference demo supporting vision, text, and multimodal models. Build a tidy system that showcases architectural flexibility while preserving all working functionality.

## Project Scope
- **Demo Focus**: Clean, extensible demonstration system (not production service)
- **Timeline**: 3-5 days of intense, focused work
- **Environment**: Local development only (localhost)
- **Frontend**: Simple webapp in `/old_frontend` that sends HTTP requests
- **Models**: Support any OpenVINO IR format model of each type

## Core Principles

### Testing Philosophy: Test-Supported Refactoring
- **Characterization Tests**: Write tests for behavior we want to preserve, not implementation
- **Fearless Refactoring**: Change internal structure while maintaining functionality
- **Selective Preservation**: Only test functionality we want to keep
- **Contract Testing**: Test external interfaces, not internal details

**Key Strategy:**
1. Start with working v0.1 prototype
2. Write tests for essential functionality (image in → classification out)
3. Refactor architecture aggressively while keeping tests green
4. Tests ensure demo always works throughout development

### Design Principles
- **Clean & Extensible**: Code should be extremely easy to understand and extend
- **Flexible Interfaces**: Support any model of each type without code changes
- **Always Working**: Demo never breaks - every change preserves existing functionality
- **Minimal Complexity**: Skip production concerns (caching, optimization, security hardening)

## Technical Requirements

### Must Have
- **Multi-Model Support**: Vision, text, and multimodal models
- **Model Flexibility**: Easy to swap any OpenVINO model without code changes
- **Clean API**: Well-designed endpoints for each model type
- **Frontend Integration**: Webapp works with all model types
- **Model Registry**: JSON-based configuration for easy model management

### Nice to Have
- **Streaming**: Real-time text generation
- **Multimodal**: Combined image+text inference
- **Model Management**: Runtime model switching via API

### Explicitly Out of Scope
- Production optimization (memory usage, caching, connection pooling)
- Security hardening (rate limiting, authentication, etc.)
- A/B testing or fine-tuning capabilities
- Performance benchmarking or monitoring dashboards
- Advanced error recovery or graceful degradation

## 3-5 Day Implementation Plan

### Day 1: Foundation & Architecture
**Goal**: Clean, testable foundation with flexible model interface

#### Morning: Assessment & Testing ✅ COMPLETED
- [x] Load v0.1 prototype and understand current structure
- [x] Write characterization tests for essential functionality
- [x] Document current API contracts and data flows
- [x] Ensure existing image classification works end-to-end
- [x] Restructured response format to JSON with output/metadata/model_info

#### Afternoon: Architecture Abstraction ✅ COMPLETED
- [x] Design `ModelConfig` trait system for flexibility
- [x] Implement ImageModelConfig wrapper (with hardcoded values)
- [x] Implement TextModelConfig placeholder
- [x] Extract model configuration from hard-coded values (name/version now configurable)
- [x] Update server routing to use trait instead of hardcoded inference
- [x] Clean up redundant code and fix all linter errors

**Tests Pass**: Image classification still works through new architecture (test already validates JSON format)

### Day 2: Multi-Model Support
**Goal**: Add text models alongside existing image classification

#### Morning: WASM Abstraction & Preprocessing ✅ COMPLETED
- [x] Abstract WASM interface to be model-agnostic
  - [x] Created model registry system in WASM (`registry.rs`)
  - [x] Removed hardcoded ImageNet labels from main.rs
  - [x] Added complete ImageNet labels module (`imagenet_labels.rs`)
  - [x] Implemented new WASM exports: `register_model()` and `infer_with_model()`
  - [x] Model now returns human-readable labels ("golden retriever" not "unknown class")
- [x] Create abstract preprocessing pipeline interface for different model types
  - [x] Moved JPEG preprocessing from server main.rs into ImageModelConfig
  - [x] ImageModelConfig now handles full pipeline: validate → preprocess → infer → format
  - [x] Added proper error conversion from ValidationError to InferenceError
- [x] Update server to pass raw bytes instead of preprocessed tensors
  - [x] Server now forwards raw JPEG data to ModelConfig.infer()
  - [x] Removed tensor module import from main.rs
  - [x] All tests pass - system works end-to-end

#### Afternoon: Text Infrastructure ⚠️ NEXT TASK
**CRITICAL STATUS UPDATE**: Research phase completed. Ready for implementation.

**IMMEDIATE NEXT STEPS**:
1. **[NOW]** Make WASM interface accept multiple named tensors (see Section: "Text Model WASM Requirements" below)
2. **[THEN]** Implement text preprocessing in TextModelConfig (see Section: "Text Preprocessing Implementation" below)
3. **[THEN]** Add `/infer/text` endpoint with proper Content-Type detection
4. **[THEN]** Test both model types work independently

**RESEARCH FINDINGS**: 
- Text models require multiple input tensors (input_ids, attention_mask, token_type_ids)
- Rust tokenizers crate is compatible if using exact same tokenizer files
- OpenVINO expects int32/int64 tensor arrays for token IDs
- Current WASM interface only supports single tensor input (BLOCKING ISSUE)

**Tests Pass**: Both vision and text inference work independently

### Day 3: Frontend Integration & Polish
**Goal**: Complete demo with working frontend

#### Morning: API Refinement
- [ ] Standardize response formats across all endpoints
- [ ] Add proper error handling and validation
- [ ] Implement model management endpoints (`GET /models`, etc.)
- [ ] Add health check and status endpoints

#### Afternoon: Frontend Integration
- [ ] Update webapp to work with new multi-model API
- [ ] Add model selection/switching in UI
- [ ] Test all model types through frontend
- [ ] Polish error handling and user experience

**Tests Pass**: Complete demo works end-to-end through frontend

## Success Criteria

### Functional Requirements
- [ ] Image classification works (preserve v0.1 functionality)
- [ ] Text inference works with any compatible model
- [ ] Models can be swapped without code changes
- [ ] Frontend works with all model types
- [ ] System never breaks during development

### Code Quality
- [ ] Code is clean and easy to understand
- [ ] Architecture is flexible and extensible
- [ ] Adding new model types is straightforward
- [ ] Tests provide confidence for refactoring

### Demo Quality
- [ ] System showcases architectural flexibility
- [ ] Multiple model types work seamlessly
- [ ] Frontend provides good user experience
- [ ] Documentation explains how to extend

## Testing Strategy

### Characterization Tests (Day 1)
```rust
#[test]
fn image_classification_preserves_functionality() {
    // Send JPEG → get classification result
    // Don't test internal implementation
}

#[test]
fn server_health_check_works() {
    // Basic server functionality
}
```

### Integration Tests (Throughout)
```rust
#[test]
fn text_inference_works_end_to_end() {
    // Text prompt → generated response
}

#[test]
fn model_switching_preserves_functionality() {
    // Switch models → same functionality
}
```

### Frontend Tests (Day 3)
- Manual testing through webapp
- All model types accessible via UI
- Error cases handled gracefully

## Risk Mitigation

### Technical Risks
- **Architecture Lock-in**: Use characterization tests to enable aggressive refactoring
- **Model Compatibility**: Test with multiple models early
- **Frontend Integration**: Test UI integration frequently

### Process Risks
- **Scope Creep**: Stick to demo requirements, avoid production features
- **Perfectionism**: Ship working demo over perfect code
- **Time Management**: Focus on core functionality first, extensions second

## Daily Workflow
1. **Start**: Ensure all existing tests pass
2. **Develop**: Make changes while keeping tests green
3. **Test**: Validate new functionality works
4. **Integrate**: Test full system including frontend
5. **Document**: Update docs and plan next day

This plan prioritizes building a clean, flexible demo that showcases the architecture while maintaining working functionality throughout development.

---

## ModelConfig Trait Specification

**CRITICAL**: This section defines the exact interface and behavior required for the flexible model architecture. Follow this specification exactly.

### Overview
The `ModelConfig` trait abstracts different model types (image, text) behind a unified interface. Each model handles its own preprocessing, inference, and postprocessing while the server routes requests based on data type detection.

### Core Trait Definition

```rust
pub trait ModelConfig: Send + Sync {
    /// Validates input data format and size
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError>;
    
    /// Performs complete inference pipeline: preprocess -> infer -> postprocess
    fn infer(&self, data: &[u8]) -> Result<serde_json::Value, InferenceError>;
    
    /// Returns model metadata for responses
    fn model_info(&self) -> ModelInfo;
}
```

### Required Error Types

```rust
#[derive(Debug)]
pub enum ValidationError {
    InvalidFormat,      // Wrong data format (not JPEG, not text, etc.)
    InvalidSize,        // Data too large/small
    InvalidDimensions,  // Wrong tensor dimensions
    MalformedData,      // Corrupted or incomplete data
}

#[derive(Debug)]
pub enum InferenceError {
    PreprocessingFailed(String),
    ModelLoadFailed(String),
    InferenceFailed(String),
    PostprocessingFailed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Image,
    Text,
    Multimodal,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
}
```

### Response Format Contract

**MANDATORY**: All models MUST return JSON in this exact structure:

```json
{
  "output": <model-specific-result>,
  "metadata": {
    // model-specific metadata - no required fields
  },
  "model_info": {
    "name": <string>,
    "version": <string>,
    "model_type": "Image" | "Text" | "Multimodal"
  }
}
```

**Note**: The `metadata` object is model-specific and has no required fields. Each model type defines its own relevant metadata.

### Implementation Requirements

#### ImageModelConfig
- **Input validation**: Must detect JPEG magic bytes (`FF D8 FF`)
- **Preprocessing**: JPEG → BGR tensor (224x224x3, normalized to [0,1])
- **Output format**: 
  ```json
  {
    "output": "golden retriever",  // human-readable class name
    "metadata": {
      "probability": 0.8234,
      "class_index": 207
    },
    "model_info": {
      "name": "mobilenet_v3_large",
      "version": "1.0",
      "model_type": "Image"
    }
  }
  ```

#### TextModelConfig  
- **Input validation**: Must validate UTF-8 text, max length limits
- **Preprocessing**: Text → tokens → tensor
- **Output format**:
  ```json
  {
    "output": "The quick brown fox jumps over the lazy dog.",
    "metadata": {
      "token_count": 12,
      "inference_time_ms": 45,
      "temperature": 0.7
    },
    "model_info": {
      "name": "llama2_7b",
      "version": "2.0", 
      "model_type": "Text"
    }
  }
  ```

### Server Integration Requirements

#### Data Type Detection
The server MUST route requests to the correct model based on Content-Type:

```rust
fn detect_model_type(content_type: &str, data: &[u8]) -> Option<ModelType> {
    match content_type {
        "image/jpeg" => Some(ModelType::Image),
        "text/plain" => Some(ModelType::Text),
        _ => None
    }
}
```

#### Model Registry JSON Format
Create `models.json` configuration file:

```json
{
  "models": {
    "default_image": {
      "type": "image",
      "config": {
        "model_path": "fixture/model.xml",
        "weights_path": "fixture/model.bin",
        "input_size": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      },
      "info": {
        "name": "mobilenet_v3_large",
        "version": "1.0",
        "model_type": "Image"
      }
    },
    "default_text": {
      "type": "text", 
      "config": {
        "model_path": "models/llama2.xml",
        "weights_path": "models/llama2.bin",
        "max_length": 512,
        "temperature": 0.7
      },
      "info": {
        "name": "llama2_7b",
        "version": "2.0",
        "model_type": "Text"
      }
    }
  }
}
```

### Implementation Steps Progress

**CURRENT STATE** (Day 2 Afternoon): 
- ModelConfig trait is fully integrated into server routing
- WASM interface has been abstracted with model registry system
- Preprocessing pipeline has been abstracted - models handle their own preprocessing
- Server passes raw data to models, no longer does preprocessing
- All tests pass, system is ready for text model implementation

**COMPLETED TODAY**:
1. ✅ Created `ModelRegistry` in WASM for managing multiple models
2. ✅ Removed hardcoded ImageNet labels from main.rs
3. ✅ Added complete `imagenet_labels.rs` module with all 1000 labels
4. ✅ Implemented `register_model()` and `infer_with_model()` WASM exports
5. ✅ Fixed "unknown class" issue - model now returns proper labels
6. ✅ Moved preprocessing into ImageModelConfig - full pipeline in model
7. ✅ Server now passes raw JPEG bytes instead of preprocessed tensors

**NEXT STEPS (Day 2 Afternoon)**: 
1. **NOW**: Implement text preprocessing pipeline (tokenization)
2. **THEN**: Create real TextModelConfig implementation (replace mock)
3. **THEN**: Add `/infer/text` endpoint for text models
4. **THEN**: Test both model types work independently

### Testing Requirements

**CRITICAL**: The existing integration test `test_image_classification_preserves_functionality()` already validates the JSON response format and MUST remain green throughout the refactor.

Current test status:
- [x] Image classification test validates JSON format ✅
- [ ] **TODAY**: Add new tests for text model type
- [x] Error handling tests for invalid inputs ✅
- [ ] **TODAY**: Test model registry loading

### Current System Status

**Day 1 Complete** ✅: ModelConfig trait foundation is fully integrated and working!
**Day 2 Morning Complete** ✅: WASM abstraction and preprocessing pipeline done!

**COMPLETED FEATURES** ✅:
- Response format returns proper JSON structure with output/metadata/model_info
- ModelConfig trait system fully defined with all error types
- ImageModelConfig handles full inference pipeline: validate → preprocess → infer → format
- TextModelConfig placeholder returns mock responses (ready for real implementation)
- Server routing uses ModelConfig trait for all inference
- Server passes raw data to models (no preprocessing in server)
- WASM model registry system implemented with dynamic model management
- ImageNet labels in dedicated module with proper label lookup
- New WASM exports for model registration and inference
- All tests pass, system fully functional

**REMAINING TASKS** (Day 2 Afternoon):
- Text models only return mock responses (need real implementation)
- No text preprocessing pipeline (tokenization) yet
- No `/infer/text` endpoint yet
- Server-side JSON model registry not implemented (low priority for demo)

**WHAT'S HAPPENING NOW**: 
Day 2 morning tasks complete. Ready to implement text model support:
1. Implement text preprocessing pipeline (tokenization)
2. Create real TextModelConfig with actual inference
3. Add `/infer/text` endpoint
4. Test both model types work independently

### Critical Implementation Notes

1. **Thread Safety**: All models must be `Send + Sync`
2. **Error Handling**: Use `Result` types consistently, never panic
3. **Memory Management**: Keep it simple, no explicit cleanup
4. **Configuration**: All model parameters come from JSON registry
5. **Preprocessing**: Each model handles its own preprocessing completely
6. **One Model Per Thread**: No concurrent inference on same model instance

### Success Criteria

**Day 1** ✅ **COMPLETED**:
- [x] `cargo test` passes with current architecture ✅
- [x] Image classification returns new JSON format ✅ 
- [x] Text inference returns placeholder responses (trait level) ✅
- [x] Extract hardcoded values to make ImageModelConfig configurable ✅
- [x] Server uses ModelConfig trait instead of hardcoded logic ✅
- [x] All linter errors fixed, code is clean ✅
- [x] Server routes based on Content-Type detection ✅
- [x] All error cases return proper HTTP status codes ✅

**Day 2 Goals**:
- [x] Abstract WASM interface to be model-agnostic ✅
  - [x] Model registry in WASM
  - [x] Remove hardcoded labels
  - [x] Dynamic model management
- [x] Create abstract preprocessing pipeline (move JPEG processing into ImageModelConfig) ✅
- [x] Update server to pass raw bytes instead of preprocessed tensors ✅
- [ ] **[CRITICAL]** Make WASM accept multiple input tensors (BLOCKING)
- [ ] Implement real text model preprocessing and inference
- [ ] Add `/infer/text` endpoint for text models
- [ ] Models load from JSON configuration (server-side) - deferred as low priority

---

# DETAILED IMPLEMENTATION GUIDE FOR NEXT DEVELOPER

## CRITICAL CONTEXT: WHERE WE ARE

### Current State Summary
**Date**: Day 2 Afternoon
**Status**: Morning tasks complete, afternoon ready to start
**What works**: Image classification fully functional with new architecture
**What's next**: Text model support implementation

### Architecture Overview
The system has two layers:
1. **Server layer** (Rust): HTTP server, routing, ModelConfig trait implementations
2. **WASM layer** (Rust→WASM): OpenVINO inference engine, model registry

**Current flow**: HTTP request → Server detects type → ModelConfig.infer() → WASM inference → JSON response

### What Just Got Fixed
1. **Preprocessing abstraction**: Server no longer does preprocessing, ModelConfig implementations handle their own
2. **WASM registry**: WASM can now manage multiple models with metadata
3. **Clean architecture**: Server passes raw data, models handle validation/preprocessing/inference

## IMMEDIATE BLOCKING ISSUE: WASM INTERFACE LIMITATION

### The Problem
Current WASM interface only accepts **single tensor input**:
```rust
// In inferencer/src/main.rs - CURRENT LIMITATION
#[no_mangle]
pub extern "C" fn infer_with_model(
    data_ptr: i32, 
    data_len: i32, 
    result_ptr: i32, 
    model_id: i32
) -> i32
```

**BUT**: Text models need **multiple named tensors**:
- `input_ids`: [1, sequence_length] int32
- `attention_mask`: [1, sequence_length] int32  
- `token_type_ids`: [1, sequence_length] int32 (optional)

### Required WASM Changes
**File**: `backend/inferencer/src/main.rs`

**STEP 1**: Create multi-tensor input structure
```rust
#[derive(Serialize, Deserialize)]
pub struct MultiTensorInput {
    pub tensors: HashMap<String, TensorData>,
}

#[derive(Serialize, Deserialize)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data_type: String, // "int32", "float32", etc.
    pub data: Vec<u8>,
}
```

**STEP 2**: Replace single-tensor inference with multi-tensor
```rust
#[no_mangle]
pub extern "C" fn infer_multi_tensor(
    input_json_ptr: i32,
    input_json_len: i32, 
    result_ptr: i32,
    model_id: i32
) -> i32 {
    // Deserialize MultiTensorInput from JSON
    // Set multiple input tensors on OpenVINO model
    // Run inference
    // Return JSON result
}
```

**STEP 3**: Update server to call new WASM function
**File**: `backend/server/src/runtime.rs`
- Change `WasmInstance::infer()` to accept `MultiTensorInput`
- Serialize to JSON, pass to WASM
- Keep backward compatibility for image models

## TEXT PREPROCESSING IMPLEMENTATION

### Required Dependencies
**File**: `backend/server/Cargo.toml`
```toml
[dependencies]
tokenizers = "0.15"  # HuggingFace tokenizers for Rust
```

### TextModelConfig Implementation
**File**: `backend/server/src/model_config.rs`

**CRITICAL DECISIONS MADE**:
1. **Use Rust tokenizers crate** (not OpenVINO tokenizers) for simplicity
2. **Start with BERT-style models** (classification, not generation)
3. **Fixed sequence length** of 512 tokens for demo
4. **Mock model initially** - focus on getting pipeline working

**Implementation template**:
```rust
pub struct TextModelConfig {
    engine: Arc<Engine>,
    module: Arc<Module>,
    log_sender: broadcast::Sender<String>,
    tokenizer: Tokenizer,  // From tokenizers crate
    max_length: usize,
    name: String,
    version: String,
}

impl TextModelConfig {
    pub fn new(
        engine: Arc<Engine>,
        module: Arc<Module>, 
        log_sender: broadcast::Sender<String>,
        tokenizer_path: &str,  // Path to tokenizer.json
        max_length: usize,
        name: String,
        version: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        Ok(Self {
            engine, module, log_sender, tokenizer, max_length, name, version
        })
    }
}

impl ModelConfig for TextModelConfig {
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        // Validate UTF-8 text
        std::str::from_utf8(data)
            .map_err(|_| ValidationError::InvalidFormat)?;
        
        // Check length limits
        if data.len() > 10240 {  // 10KB max for demo
            return Err(ValidationError::InvalidSize);
        }
        
        Ok(())
    }

    fn infer(&self, data: &[u8]) -> Result<serde_json::Value, InferenceError> {
        // 1. Validate input
        self.validate_input(data)?;
        
        // 2. Convert to text
        let text = std::str::from_utf8(data)
            .map_err(|e| InferenceError::PreprocessingFailed(e.to_string()))?;
        
        // 3. Tokenize to multiple tensors
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| InferenceError::PreprocessingFailed(e.to_string()))?;
        
        // 4. Create multi-tensor input
        let multi_tensor = self.create_multi_tensor_input(&encoding)?;
        
        // 5. Create WASM instance and run inference  
        let mut wasm_instance = WasmInstance::new(self.engine.clone(), self.module.clone())
            .map_err(|e| InferenceError::ModelLoadFailed(e.to_string()))?;
        
        // 6. Call new multi-tensor inference
        let result = wasm_instance.infer_multi_tensor(multi_tensor)
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;
        
        Ok(result)
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            version: self.version.clone(),
            model_type: ModelType::Text,
        }
    }
}

impl TextModelConfig {
    fn create_multi_tensor_input(&self, encoding: &Encoding) -> Result<MultiTensorInput, InferenceError> {
        let input_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        
        // Pad to max_length
        let mut padded_ids = input_ids.clone();
        let mut padded_mask = attention_mask.clone();
        
        padded_ids.resize(self.max_length, 0);  // PAD token = 0
        padded_mask.resize(self.max_length, 0);
        
        let mut tensors = HashMap::new();
        
        // input_ids tensor
        tensors.insert("input_ids".to_string(), TensorData {
            shape: vec![1, self.max_length],
            data_type: "int32".to_string(),
            data: bytemuck::cast_slice(&padded_ids).to_vec(),
        });
        
        // attention_mask tensor  
        tensors.insert("attention_mask".to_string(), TensorData {
            shape: vec![1, self.max_length],
            data_type: "int32".to_string(),
            data: bytemuck::cast_slice(&padded_mask).to_vec(),
        });
        
        Ok(MultiTensorInput { tensors })
    }
}
```

### Server Routing Updates
**File**: `backend/server/src/main.rs`

**Update infer function**:
```rust
async fn infer(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<InferenceRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>> {
    if let Some(content_type) = request.headers().get(header::CONTENT_TYPE) {
        let content_type_str = content_type.to_str().unwrap_or("");
        
        // Route based on content type
        match content_type_str {
            "image/jpeg" => {
                // Existing image logic - no changes needed
            },
            "text/plain" => {
                // NEW: Text processing logic
                let mut body = request.collect().await?.aggregate();
                let bytes = body.copy_to_bytes(body.remaining());
                
                log_sender.send("[server/main.rs] Processing text for inference.".to_string()).ok();
                let (sender, receiver) = oneshot::channel();
                inference_thread_sender.send(InferenceRequest {
                    data: bytes.to_vec(),
                    model_type: ModelType::Text,  // NEW: Add model_type to InferenceRequest
                    responder: sender,
                })?;
                
                // Wait for response (same as image)
            },
            _ => {
                // Unsupported media type error
            }
        }
    }
}
```

**Update InferenceRequest structure**:
**File**: `backend/server/src/utils.rs`
```rust
#[derive(Debug)]
pub struct InferenceRequest {
    pub data: Vec<u8>,
    pub model_type: ModelType,  // NEW: Route to correct model
    pub responder: oneshot::Sender<Value>,
}
```

**Update inference thread**:
**File**: `backend/server/src/main.rs` (in main function)
```rust
// Create both model configs
let image_model = Arc::new(ImageModelConfig::new(/*...*/));
let text_model = Arc::new(TextModelConfig::new(/*...*/));

while let Some(request) = rx.recv().await {
    let image_model = Arc::clone(&image_model);
    let text_model = Arc::clone(&text_model);
    
    spawn_blocking(move || -> anyhow::Result<()> {
        let result = match request.model_type {
            ModelType::Image => image_model.infer(&request.data),
            ModelType::Text => text_model.infer(&request.data),
            _ => Err(InferenceError::PreprocessingFailed("Unsupported model type".to_string())),
        };
        
        let response = result.expect("Inference failed");
        request.responder.send(response).unwrap();
        Ok(())
    });
}
```

## TESTING STRATEGY

### Required Test Files
**Create**: `backend/server/tests/text_test_input.txt`
```
This is a test sentence for text classification.
```

### Test Implementation
**File**: `backend/server/tests/integration_test.rs`
```rust
#[tokio::test]
async fn test_text_classification_preserves_functionality() {
    let client = reqwest::Client::new();
    
    let test_text = "This is a test sentence.";
    
    let response = client
        .post("http://127.0.0.1:3000/infer")
        .header("Content-Type", "text/plain")
        .body(test_text)
        .send()
        .await
        .expect("Failed to send request");
    
    assert_eq!(response.status(), 200);
    
    let json: serde_json::Value = response.json().await.expect("Failed to parse JSON");
    
    // Validate required JSON structure
    assert!(json.get("output").is_some());
    assert!(json.get("metadata").is_some()); 
    assert!(json.get("model_info").is_some());
    
    let model_info = json.get("model_info").unwrap();
    assert_eq!(model_info.get("model_type").unwrap(), "Text");
}
```

## CRITICAL FILES TO MODIFY

**PRIORITY 1** (Required for basic functionality):
1. `backend/inferencer/src/main.rs` - Add multi-tensor WASM interface
2. `backend/server/src/model_config.rs` - Implement TextModelConfig
3. `backend/server/src/runtime.rs` - Update WasmInstance for multi-tensor
4. `backend/server/src/utils.rs` - Add model_type to InferenceRequest
5. `backend/server/src/main.rs` - Update routing and inference thread

**PRIORITY 2** (For proper testing):
6. `backend/server/tests/integration_test.rs` - Add text model tests
7. `backend/server/Cargo.toml` - Add tokenizers dependency

## TOKENIZER FILES NEEDED

For testing, download a simple BERT tokenizer:
```bash
# In backend/server/fixture/ directory
wget https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json
```

Or use mock tokenizer for initial implementation (recommended).

## POTENTIAL GOTCHAS & DEBUGGING

### 1. **WASM Compilation Issues**
- WASM target must support serde JSON serialization
- May need to add features to Cargo.toml: `serde = { version = "1.0", features = ["derive"] }`

### 2. **Tokenizer Compatibility**
- Must use exact same tokenizer as original model training
- Test with known inputs first: "Hello world" should tokenize to predictable IDs

### 3. **Memory Management**
- Text tensors are larger than image tensors (sequence_length * vocab_size)
- Watch for memory leaks across FFI boundary

### 4. **Error Handling**
- Tokenization can fail on malformed text
- WASM serialization/deserialization can fail
- Add detailed error logging for debugging

## SUCCESS CRITERIA

**Minimum viable implementation**:
1. ✅ WASM accepts multi-tensor input (even if text model is mocked)
2. ✅ Server routes text/plain to TextModelConfig
3. ✅ TextModelConfig tokenizes input and calls WASM
4. ✅ System returns JSON with correct structure
5. ✅ All existing image tests still pass
6. ✅ New text test passes

**Ready for Day 3**:
- Both image and text inference work independently
- Clean error handling for both model types
- Frontend can call both endpoints
- System is ready for UI integration

## IMPLEMENTATION ORDER

**DO NOT DEVIATE FROM THIS ORDER** (learned from previous refactoring):

1. **First**: Make WASM multi-tensor interface work with mock text model
2. **Second**: Implement TextModelConfig with mock inference 
3. **Third**: Add server routing for text/plain
4. **Fourth**: Test end-to-end with mock data
5. **Fifth**: Add real tokenization and test again
6. **Last**: Add real OpenVINO text model (if time permits)

**Rationale**: Get the architecture working first with mocks, then add complexity incrementally while keeping tests green.

## DOCUMENTATION REFERENCES

All required documentation is in `/docs/` directory:
- **OpenVINO Tokenizers**: How tokenization works with OpenVINO
- **BERT Demo**: Exact tensor format requirements
- **openvino-rs**: Rust integration patterns
- **GenAI Framework**: Text generation APIs (for future)

**Key insight from research**: Rust tokenizers crate IS compatible with OpenVINO if using same tokenizer files. Start with this approach for simplicity.

---

# END OF IMPLEMENTATION GUIDE

**FINAL NOTE**: The architecture is solid. The preprocessing pipeline is clean. The WASM interface needs to support multiple tensors, then text models are straightforward. Do not overthink it. Follow the plan exactly and the demo will work perfectly.