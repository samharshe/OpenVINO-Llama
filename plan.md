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

#### Afternoon: Preprocessing Architecture Refactor ⚠️ CRITICAL CHANGE
**MAJOR ARCHITECTURE DECISION**: Moving ALL preprocessing to WASM layer for cleaner design.

**Why This Refactor**:
- **Current state**: ImageModelConfig does preprocessing (JPEG→tensor) in server layer
- **Issue**: Server layer has model-specific logic, violating single responsibility
- **Solution**: Move ALL preprocessing (image AND text) to WASM layer
- **Benefit**: Server becomes pure router, models are fully self-contained

**REFACTORING PLAN**:
1. **[FIRST]** Migrate image preprocessing from server to WASM
2. **[THEN]** Implement text preprocessing in WASM (following same pattern)
3. **[THEN]** Server only routes raw bytes based on Content-Type
4. **[THEN]** Test both model types work with new architecture

**KEY INSIGHTS**: 
- Models should be self-contained (raw input → processed output)
- Server should only route, not process
- WASM can handle both JPEG decoding and tokenization
- More elegant and maintainable architecture

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

**NEXT STEPS (Day 2 Afternoon - UPDATED)**: 
1. **NOW**: Migrate image preprocessing from server to WASM
2. **THEN**: Implement text model in WASM with tokenization
3. **THEN**: Update server to pass raw bytes for both model types
4. **THEN**: Test both model types work with new architecture

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

**REMAINING TASKS** (Day 2 Afternoon - UPDATED):
- Image preprocessing still in server layer (needs migration to WASM)
- Text models not implemented yet
- Server should become pure router (no preprocessing)
- Server-side JSON model registry not implemented (low priority for demo)

**WHAT'S HAPPENING NOW**: 
Day 2 morning tasks complete. Architecture refactor in progress:
1. First migrate image preprocessing to WASM (maintain working system)
2. Then add text model with preprocessing in WASM
3. Server becomes pure router for both model types
4. Test both model types work with clean architecture

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
- [x] ~~Make WASM accept multiple input tensors~~ NOT NEEDED - Option B chosen ✅
- [ ] Implement real text model preprocessing and inference
- [ ] Server routing already supports text/plain ✅
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
1. **Preprocessing abstraction**: ModelConfig implementations handle their own preprocessing (BUT still in server layer)
2. **WASM registry**: WASM can now manage multiple models with metadata
3. **Partially clean architecture**: Server passes raw data to ModelConfig, but preprocessing still happens in server

### What We're Fixing Now
**Problem**: Preprocessing is still in the server layer (ImageModelConfig does JPEG→tensor conversion)
**Solution**: Move ALL preprocessing to WASM layer for truly clean architecture

## PREPROCESSING REFACTOR: MOVING TO WASM LAYER

### Current Architecture (Suboptimal)
- **Server**: Raw data → Preprocessing (JPEG→tensor, text→tokens) → WASM
- **WASM**: Preprocessed data → Inference → Results
- **Problem**: Server has model-specific preprocessing logic

### Target Architecture (Clean)
- **Server**: Raw data → WASM (pure routing)
- **WASM**: Raw data → Preprocessing → Inference → Results
- **Benefit**: Models are fully self-contained

### Image Preprocessing Migration Plan

#### Phase 1: Add JPEG Processing to Inferencer ✅ Maintains Working System
1. Add `image` crate dependency to inferencer Cargo.toml
2. Create preprocessing module in inferencer
3. Update MobilnetModel to accept raw JPEG (keep tensor path for compatibility)
4. **System still works**: Server continues current flow

#### Phase 2: Update WASM Interface ✅ Maintains Working System
1. Modify `infer_with_model` to detect input type (JPEG vs tensor)
2. Route to appropriate processing based on magic bytes
3. **System still works**: Dual support for both paths

#### Phase 3: Simplify Server Layer ✅ Maintains Working System
1. Update ImageModelConfig to pass raw JPEG to WASM
2. Remove preprocessing from server
3. **System still works**: Now using cleaner architecture

#### Phase 4: Add Text Support Following Same Pattern
1. Text model in WASM handles raw text input
2. Tokenization happens inside WASM
3. Server just routes based on Content-Type

### WASM Text Model Implementation
**File**: `backend/inferencer/src/lib.rs` (new text model)

```rust
impl TextModel {
    fn prepare_tensors(&self, token_ids: &[i32]) -> Result<(), String> {
        let seq_len = token_ids.len();
        
        // Create input_ids tensor from provided data
        let input_ids = Tensor::new(&self.core, &[1, seq_len], token_ids)?;
        
        // Construct attention_mask (all 1s for real tokens)
        let attention_mask: Vec<i32> = vec![1; seq_len];
        let attention_tensor = Tensor::new(&self.core, &[1, seq_len], &attention_mask)?;
        
        // Set tensors on model
        self.request.set_tensor("input_ids", input_ids)?;
        self.request.set_tensor("attention_mask", attention_tensor)?;
        
        Ok(())
    }
}

## TEXT MODEL IMPLEMENTATION (IN WASM)

### Key Decision
With the preprocessing refactor, text models will:
1. Receive raw UTF-8 text bytes from server
2. Handle tokenization inside WASM (using tokenizers or similar)
3. Construct all required tensors internally
4. Run inference and return results

### Benefits
- Consistent with image model approach
- Server remains a pure router
- Models are fully self-contained
- Better encapsulation and portability

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

**PRIORITY 1** (Image preprocessing migration):
1. `backend/inferencer/Cargo.toml` - Add image crate dependency
2. `backend/inferencer/src/lib.rs` - Add preprocessing module and update MobilnetModel
3. `backend/inferencer/src/main.rs` - Update to detect raw JPEG vs tensor
4. `backend/server/src/model_config.rs` - Simplify ImageModelConfig to pass raw JPEG

**PRIORITY 2** (Text model addition):
5. `backend/inferencer/src/lib.rs` - Add text model with tokenization
6. `backend/server/src/utils.rs` - Add model_type to InferenceRequest
7. `backend/server/src/main.rs` - Update routing for model types
8. `backend/inferencer/src/registry.rs` - Register text model type

## TOKENIZER FILES NEEDED

For autoregressive generation, download a GPT-2 or similar tokenizer:
```bash
# In backend/server/fixture/ directory
wget https://huggingface.co/gpt2/resolve/main/tokenizer.json
wget https://huggingface.co/gpt2/resolve/main/tokenizer_config.json
```

Real tokenizer is required - mock would produce nonsense output.

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
1. ✅ TextModelConfig tokenizes text using `tokenizers` crate
2. ✅ WASM text model constructs attention_mask internally
3. ✅ Server routes text/plain to TextModelConfig
4. ✅ System returns JSON with correct structure
5. ✅ All existing image tests still pass
6. ✅ New text test passes

**Ready for Day 3**:
- Both image and text inference work independently
- Clean error handling for both model types
- Frontend can call both endpoints
- System is ready for UI integration

## IMPLEMENTATION ORDER

**DO NOT DEVIATE FROM THIS ORDER** (updated for preprocessing refactor):

1. **First**: Add image crate dependency to inferencer Cargo.toml
2. **Second**: Create preprocessing module in inferencer with JPEG support
3. **Third**: Update MobilnetModel to handle raw JPEG input
4. **Fourth**: Update server ImageModelConfig to pass raw JPEG
5. **Fifth**: Add text model to WASM with tokenization
6. **Last**: Clean up unused preprocessing code from server

**Rationale**: Migrate preprocessing incrementally while keeping system working at each step.

## DOCUMENTATION REFERENCES

All required documentation is in `/docs/` directory:
- **OpenVINO Tokenizers**: How tokenization works with OpenVINO
- **BERT Demo**: Exact tensor format requirements
- **openvino-rs**: Rust integration patterns
- **GenAI Framework**: Text generation APIs (for future)

**Key insights from research**: 
- Rust tokenizers crate IS compatible with OpenVINO models
- Autoregressive models only need input_ids and attention_mask (not token_type_ids)
- Tensors can be constructed inside WASM layer, avoiding interface changes
- Option B approach is cleaner and simpler than modifying WASM interface

---

# END OF IMPLEMENTATION GUIDE

**FINAL NOTE**: The architecture is solid. The preprocessing pipeline is clean. No WASM interface changes needed with Option B. Text models construct their own tensors internally. Follow the updated plan and the demo will work perfectly.