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

#### Morning: Assessment & Testing
- [x] Load v0.1 prototype and understand current structure
- [x] Write characterization tests for essential functionality
- [x] Document current API contracts and data flows
- [x] Ensure existing image classification works end-to-end

#### Afternoon: Architecture Abstraction  
- [ ] Design `ModelConfig` trait system for flexibility
- [ ] Extract model configuration from hard-coded values
- [ ] Create abstract preprocessing pipeline
- [ ] Refactor WASM interface to be model-agnostic

**Tests Pass**: Image classification still works through new architecture

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