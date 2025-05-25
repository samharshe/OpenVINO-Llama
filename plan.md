# Implementation Plan: OpenVINO-Llama Universal ML Inference Demo

## Overview
Transform existing v0.1 image classification prototype into a clean, flexible universal ML inference demo supporting vision, text, and multimodal models. Build a tidy system that showcases architectural flexibility while preserving all working functionality.

## Project Scope
- **Demo Focus**: Clean, extensible demonstration system (not production service)
- **Timeline**: 3-5 days of intense, focused work
- **Environment**: Local development only (localhost)
- **Frontend**: Simple webapp in `/webapp` that sends HTTP requests
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
- [x] **ALREADY DONE**: Restructured response format to JSON with output/metadata/model_info

#### Afternoon: Architecture Abstraction **← CURRENT WORK**
- [ ] **IN PROGRESS**: Design `ModelConfig` trait system for flexibility
- [ ] Extract model configuration from hard-coded values  
- [ ] Create abstract preprocessing pipeline
- [ ] Refactor WASM interface to be model-agnostic

**Tests Pass**: Image classification still works through new architecture (test already validates JSON format)

### Day 2: Multi-Model Support
**Goal**: Add text models alongside existing image classification

#### Morning: Text Infrastructure
- [ ] Implement text preprocessing pipeline (tokenization)
- [ ] Create `TextConfig` and text model loading
- [ ] Add `/infer/text` endpoint alongside `/infer/image`
- [ ] Ensure both model types can coexist

#### Afternoon: Model Registry
- [ ] Design JSON-based model registry format
- [ ] Implement dynamic model loading system
- [ ] Add model discovery and validation
- [ ] Create model switching without restart

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

### Day 4: Extension & Documentation (Optional)
**Goal**: Prepare for future extensions

#### Morning: Multimodal Support
- [ ] Add combined image+text preprocessing
- [ ] Implement `/infer/multimodal` endpoint
- [ ] Test multimodal workflow through frontend

#### Afternoon: Documentation & Polish
- [ ] Update API documentation
- [ ] Add examples for adding new models
- [ ] Code cleanup and refactoring for clarity
- [ ] Comprehensive testing of all features

**Tests Pass**: All features work, code is clean and documented

### Day 5: Advanced Features (If Time)
**Goal**: Showcase system flexibility

#### Optional Features
- [ ] Streaming text generation with Server-Sent Events
- [ ] Model hot-swapping via drag-and-drop
- [ ] Advanced model configuration options
- [ ] Performance timing and metadata in responses

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

### Implementation Steps **← CURRENT AFTERNOON WORK**

**Starting Point**: We have a working image classification system that returns proper JSON format with tests passing.

1. **IN PROGRESS**: Define trait and error types in `src/model_config.rs`
2. **TODAY**: Implement ImageModelConfig that wraps existing WASM inference
3. **TODAY**: Implement TextModelConfig as placeholder (returns mock responses)
4. **TODAY**: Create ModelRegistry to load from JSON config
5. **TODAY**: Update server routing to use trait instead of hardcoded inference
6. **OPTIONAL**: Add model selection endpoint `GET /models`

**GOAL**: Complete ModelConfig trait system and multi-model support by end of afternoon while keeping all tests green.

### Testing Requirements

**CRITICAL**: The existing integration test `test_image_classification_preserves_functionality()` already validates the JSON response format and MUST remain green throughout the refactor.

Current test status:
- [x] Image classification test validates JSON format ✅
- [ ] **TODAY**: Add new tests for text model type
- [x] Error handling tests for invalid inputs ✅
- [ ] **TODAY**: Test model registry loading

### Current System Status

**RESPONSE FORMAT ALREADY UPDATED** ✅: The system already returns the new JSON structure:
```json
{
  "output": "golden retriever",  // human-readable class name  
  "metadata": {
    "probability": 0.8234
  },
  "model_info": {
    "name": "mobilenet_v3_large"
  }
}
```
**Integration test confirms this format and MUST remain green during refactor.**

### Critical Implementation Notes

1. **Thread Safety**: All models must be `Send + Sync`
2. **Error Handling**: Use `Result` types consistently, never panic
3. **Memory Management**: Keep it simple, no explicit cleanup
4. **Configuration**: All model parameters come from JSON registry
5. **Preprocessing**: Each model handles its own preprocessing completely
6. **One Model Per Thread**: No concurrent inference on same model instance

### Success Criteria

- [x] `cargo test` passes with current architecture ✅
- [x] Image classification returns new JSON format ✅ 
- [ ] **TODAY'S GOAL**: Text inference returns placeholder responses
- [ ] **TODAY'S GOAL**: Models load from JSON configuration  
- [x] Server routes based on Content-Type detection ✅
- [x] All error cases return proper HTTP status codes ✅