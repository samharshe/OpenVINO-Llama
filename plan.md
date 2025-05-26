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

#### Afternoon: Text Infrastructure
- [ ] Implement text preprocessing pipeline (tokenization)
- [ ] Create `TextConfig` and text model loading
- [ ] Add `/infer/text` endpoint alongside `/infer/image`
- [ ] Ensure both model types can coexist

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
- [ ] Implement real text model preprocessing and inference
- [ ] Add `/infer/text` endpoint for text models
- [ ] Models load from JSON configuration (server-side) - deferred as low priority