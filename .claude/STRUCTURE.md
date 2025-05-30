# Repository Structure

## Overview
This repository implements a web-based demo for machine learning inference using OpenVINO via wasi-nn in WebAssembly. It consists of:
- Frontend: Web interface for user interaction
- Backend: HTTP server and ML inferencer
- Models: Currently supports image classification, with text generation in development

## Directory Structure

### `/frontend/`
- `index.html` - Main web interface
- `app.js` - Frontend JavaScript logic
- `styles.css` - UI styling
- `files/` - Sample images for testing

### `/backend/`
- `Cargo.toml` - Root workspace configuration
- `Makefile` - Build automation

#### `/backend/server/`
HTTP server that handles requests and routes them to the inferencer.
- `src/main.rs` - Server entry point
- `src/config.rs` - Server configuration
- `src/model_config.rs` - Model configuration structures
- `src/runtime.rs` - Runtime management
- `src/tensor.rs` - Tensor operations
- `src/utils.rs` - Utility functions
- `fixture/` - Test models and data
  - `model.bin`, `model.xml` - Image model files
  - `text_model/` - Text model files (OpenVINO format)
- `tests/` - Integration tests

#### `/backend/inferencer/`
ML inference engine using OpenVINO via wasi-nn.
- `src/main.rs` - Inferencer entry point
- `src/lib.rs` - Core library
- `src/registry.rs` - Model registry
- `src/preprocessing.rs` - Image preprocessing
- `src/text_preprocessing.rs` - Text preprocessing
- `src/imagenet_labels.rs` - ImageNet class labels

### `/ov_ir/`
- `gpt2-8bit.tflite` - TFLite GPT-2 model (8-bit quantized)
- `tflite.ipynb` - Jupyter notebook for TFLite to OpenVINO conversion

## Request Flow

1. **User Interaction** → Frontend (`index.html` + `app.js`)
   - User uploads image or enters text
   - Frontend sends HTTP request to backend server

2. **HTTP Server** → Backend Server (`/backend/server/`)
   - Receives request at server endpoint
   - Validates input and prepares for inference
   - Routes to appropriate model handler

3. **Inference** → Inferencer (`/backend/inferencer/`)
   - Loads appropriate model (image or text)
   - Preprocesses input data
   - Runs inference using OpenVINO via wasi-nn
   - Postprocesses results

4. **Response** → Server → Frontend
   - Server formats inference results
   - Sends HTTP response back to frontend
   - Frontend displays results to user

## Key Technical Constraints
- **Memory**: Limited to <4GB due to Wasm environment
- **Tensor Size**: Maximum 32-bit tensors (wasi-nn limitation)
- **Runtime**: OpenVINO inference via wasi-nn in WebAssembly

## Current Status
- ✅ Image classification: Fully functional
- 🚧 Text generation: Under development (encountering internal server errors)
  - GPT-2 model available in TFLite format
  - Conversion to OpenVINO IR in progress
  - Integration with inferencer pending