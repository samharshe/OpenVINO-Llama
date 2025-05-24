# OpenVINO-Llama: Universal ML Inference System Documentation

## Overview

A flexible machine learning inference system supporting multiple model types (vision, text, multimodal) with two main components:

1. **Inferencer** - A WebAssembly module providing ML inference via OpenVINO
2. **Server** - An HTTP server hosting the WASM inferencer with configurable preprocessing pipelines

### Architecture

```
[HTTP Client] → [Server] → [Model-Specific Preprocessing] → [WASM Inferencer] → [OpenVINO] → [Model]
```

### Supported Model Types

- **Vision Models**: Image classification, object detection, segmentation
- **Language Models**: Text generation, completion, embedding
- **Multimodal Models**: Vision-language, text-to-image
- **Custom Models**: Extensible via model configuration system

### Data Flow (Configurable)

1. Client sends request to model-specific endpoint (e.g., `/infer/image`, `/infer/text`)
2. Server applies model-appropriate preprocessing (images, tokenization, etc.)
3. Server calls WASM module with preprocessed tensor data
4. WASM module runs OpenVINO inference via WASI-NN
5. Server applies model-specific post-processing and returns structured response

## Component 1: Inferencer (WASM)

### Purpose
WebAssembly module for universal ML inference using OpenVINO through WASI-NN interface. Supports any model type through configurable model definitions.

### Model Configuration System
```rust
pub trait ModelConfig {
    fn model_type(&self) -> ModelType;           // Vision, Text, Multimodal
    fn input_specs(&self) -> Vec<TensorSpec>;    // Dynamic input specifications
    fn output_specs(&self) -> Vec<TensorSpec>;   // Dynamic output specifications
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError>;
    fn tensor_type(&self) -> wasi_nn::TensorType;
}

pub enum ModelType {
    Vision(VisionConfig),
    Text(TextConfig),
    Multimodal(MultimodalConfig),
}
```

### Model Configurations

#### Vision Models
```rust
pub struct VisionConfig {
    pub input_shape: [u32; 4],      // [N, C, H, W]
    pub output_classes: usize,       // Number of output classes
    pub task_type: VisionTask,       // Classification, Detection, Segmentation
}
```

#### Text Models
```rust
pub struct TextConfig {
    pub vocab_size: usize,           // Tokenizer vocabulary size
    pub max_sequence_length: usize,  // Maximum input tokens
    pub output_type: TextOutput,     // Generation, Embedding, Classification
}
```

#### Multimodal Models
```rust
pub struct MultimodalConfig {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub fusion_type: FusionType,     // Early, Late, Cross-attention
}
```

### FFI Interface
```c
// Load model with configuration
int load_model(int model_config_ptr, int xml_ptr, int xml_len, int weights_ptr, int weights_len);

// Run inference with dynamic tensor shapes
int infer(int input_tensors_ptr, int num_tensors, int result_ptr);

// Get model metadata
int get_model_info(int model_id, int info_ptr);
```

### Build
```bash
rustup target add wasm32-wasip1
cargo build --target wasm32-wasip1 --release
```

## Component 2: Server (HTTP)

### Purpose
HTTP server providing REST API for multi-model ML inference, with model-specific preprocessing pipelines and response formatting.

### Core Architecture
- **Hyper + Tokio**: Async HTTP server
- **Wasmtime**: WebAssembly runtime with WASI support
- **Pluggable Preprocessing**: Model-type specific input processing
- **Model Registry**: Dynamic model loading and switching

### Preprocessing Pipelines

#### Vision Pipeline
```rust
async fn preprocess_image(
    image_data: Vec<u8>, 
    config: &VisionConfig
) -> Result<Vec<u8>>
```
- JPEG/PNG decoding
- Resize to model requirements
- Normalization and format conversion
- Tensor layout conversion (HWC → NCHW)

#### Text Pipeline
```rust
async fn preprocess_text(
    text: String, 
    config: &TextConfig
) -> Result<Vec<u8>>
```
- Tokenization with model-specific vocabulary
- Sequence padding/truncation
- Attention mask generation
- Token ID tensor conversion

#### Multimodal Pipeline
```rust
async fn preprocess_multimodal(
    image_data: Vec<u8>,
    text: String,
    config: &MultimodalConfig
) -> Result<Vec<Vec<u8>>>
```
- Combined image and text preprocessing
- Cross-modal alignment
- Multiple tensor output for different modalities

### Build
```bash
# Install system dependencies
brew install opencv  # macOS
sudo apt-get install libopencv-dev clang  # Ubuntu

# Install tokenizer library
cargo add tokenizers

cargo build --release
```

## API Reference

### Model-Specific Endpoints

#### POST /infer/image
**Request:**
- Content-Type: `image/jpeg` or `image/png`
- Body: Raw image bytes

**Response:**
```json
{
  "model_type": "vision",
  "task": "classification",
  "results": [
    {"label": "cat", "confidence": 0.85, "class_id": 285}
  ]
}
```

#### POST /infer/text
**Request:**
- Content-Type: `application/json`
- Body: 
```json
{
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "model_type": "text",
  "task": "generation",
  "results": {
    "generated_text": "The capital of France is Paris, a beautiful city...",
    "tokens_generated": 12,
    "finish_reason": "length"
  }
}
```

#### POST /infer/multimodal
**Request:**
- Content-Type: `multipart/form-data`
- Fields: `image` (file), `text` (string)

**Response:**
```json
{
  "model_type": "multimodal",
  "task": "vision_language",
  "results": {
    "description": "A cat sitting on a wooden table",
    "confidence": 0.92
  }
}
```

### Model Management

#### GET /models
List available models and their configurations

#### POST /models/load
Load a new model from files

#### PUT /models/{model_id}/activate
Switch active model for an endpoint

### Streaming Support

#### GET /infer/text/stream
Server-Sent Events for streaming text generation

### CORS
- Origin: `http://localhost:8000`
- Methods: `POST, GET, PUT, OPTIONS`

## Data Formats

### Dynamic Tensor Specifications
```rust
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<i32>,        // -1 for dynamic dimensions
    pub data_type: DataType,    // f32, i32, i64
    pub layout: TensorLayout,   // NCHW, NHWC, Sequence
}
```

### Vision Tensors
- **Format**: NCHW or NHWC (configurable)
- **Dimensions**: Variable (224x224, 512x512, etc.)
- **Types**: f32, uint8
- **Batch Support**: Dynamic batch sizes

### Text Tensors
- **Format**: Sequence layout [batch, sequence_length]
- **Types**: i32 (token IDs), i64 (attention masks)
- **Dynamic Length**: Variable sequence lengths with padding

### FFI Return Codes
- `0`: Success
- `1`: Model already loaded
- `-1`: Invalid parameters
- `-2`: Model load failure
- `-3`: Invalid tensor format
- `-4`: Inference failure
- `-5`: Unsupported model type

## Model Configuration

### Model Registry Format
```json
{
  "models": {
    "mobilenet_v2": {
      "type": "vision",
      "task": "classification",
      "files": {
        "model": "models/mobilenet_v2.xml",
        "weights": "models/mobilenet_v2.bin"
      },
      "config": {
        "input_shape": [1, 3, 224, 224],
        "output_classes": 1001,
        "preprocessing": {
          "resize": [224, 224],
          "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        }
      }
    },
    "llama_7b": {
      "type": "text",
      "task": "generation",
      "files": {
        "model": "models/llama_7b.xml",
        "weights": "models/llama_7b.bin",
        "tokenizer": "models/tokenizer.json"
      },
      "config": {
        "vocab_size": 32000,
        "max_sequence_length": 2048,
        "context_window": 4096
      }
    }
  }
}
```

### File Structure
```
project/
├── src/main.rs              # Server code
├── models/                  # Model files
│   ├── registry.json        # Model configurations
│   ├── mobilenet_v2.xml     # Vision model
│   ├── mobilenet_v2.bin
│   ├── llama_7b.xml         # Text model
│   ├── llama_7b.bin
│   └── tokenizer.json       # Tokenizer config
└── target/
    └── wasm32-wasip1/
        └── release/
            └── inferencer.wasm
```

## Deployment

### Runtime Requirements
- **OpenVINO**: Runtime libraries (CPU/GPU support)
- **Tokenizers**: For text preprocessing
- **OpenCV**: For image preprocessing (optional, vision models only)

### Configuration
- Server listens on `127.0.0.1:3000`
- Model registry: `models/registry.json`
- WASM module: Auto-detected in target directory

### Environment Variables
```bash
export MODEL_REGISTRY_PATH=./models/registry.json
export DEFAULT_VISION_MODEL=mobilenet_v2
export DEFAULT_TEXT_MODEL=llama_7b
export MAX_BATCH_SIZE=8
export MAX_SEQUENCE_LENGTH=2048
```

## Key Features

- **Multi-Model Support**: Vision, text, and multimodal models
- **Dynamic Configuration**: Runtime model switching
- **Flexible Preprocessing**: Model-specific input pipelines
- **Streaming Support**: Real-time text generation
- **Security**: WASM sandboxed execution
- **Performance**: Optimized OpenVINO backend
- **Scalability**: Stateless design with configurable batching