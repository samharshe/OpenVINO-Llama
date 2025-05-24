# Cascadia Demo: Inferencer and Server Documentation

## Overview

The Cascadia Demo is a machine learning inference system consisting of two main components:

1. **@cascadia-demo/inferencer** - A WebAssembly (WASM) module that provides ML inference capabilities using OpenVINO
2. **@cascadia-demo/server** - An HTTP server that hosts the WASM inferencer and provides a REST API for image classification

### High-Level Architecture

```
[HTTP Client] ’ [Server] ’ [Image Preprocessing] ’ [WASM Inferencer] ’ [OpenVINO] ’ [Model]
     “              “              “                     “              “         “
[JSON Response]  [Server]  [Result Processing]  [WASM FFI]  [WASI-NN]  [Inference]
```

### Data Flow

1. **Image Input**: Client sends JPEG image via HTTP POST to `/infer`
2. **Preprocessing**: Server uses OpenCV to:
   - Decode JPEG
   - Resize to 224x224 pixels
   - Convert to float32 format
   - Reorder to NCHW format (channels first)
3. **WASM Invocation**: Server calls WASM module with preprocessed tensor data
4. **Inference**: WASM module uses WASI-NN to run OpenVINO inference
5. **Response**: Server returns JSON with classification label and confidence score

### Key Design Decisions

- **WASM Isolation**: ML inference runs in sandboxed WASM environment for security
- **OpenVINO Backend**: Uses Intel's OpenVINO for CPU-optimized inference
- **C FFI Interface**: WASM module exports C-compatible functions for cross-language compatibility
- **Async Architecture**: Server uses Tokio for handling concurrent requests
- **Real-time Logging**: Server-Sent Events provide live logging for debugging

## Component 1: @cascadia-demo/inferencer

### Purpose
A WebAssembly module that provides machine learning inference capabilities. Designed to run ImageNet-style classification models (specifically optimized for MobileNet architectures) using OpenVINO through the WASI-NN interface.

### Core Architecture

#### Model Abstraction Layer
The inferencer implements a generic model system with trait-based configuration:

```rust
pub trait ModelConfig {
    fn output_size(&self) -> usize;        // Number of output classes
    fn input_dims(&self) -> &[u32];        // Input tensor dimensions [N,C,H,W]
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError>;
    fn tensor_type(&self) -> wasi_nn::TensorType;
}
```

#### Model Configurations
- **ImageNetConfig**: For ImageNet classification (1001 classes, 224x224x3 input)
- **TextModelConfig**: Extensible for text models (configurable vocab/sequence length)

#### FFI Interface
Exports C-compatible functions for external integration:

```c
// Load model from memory buffers
int load_model(int xml_ptr, int xml_len, int weights_ptr, int weights_len);

// Run inference on tensor data
int infer(int tensor_ptr, int tensor_len, int result_ptr);
```

### Key Components

#### lib.rs:main:4-17
**InferenceResult & ValidationError Types**
- `InferenceResult(usize, f32)`: Classification result with label index and confidence
- `ValidationError`: Enum for input validation failures (dimensions, data size, format)

#### lib.rs:main:82-119  
**Core Model Structure**
```rust
pub struct Model<C: ModelConfig> {
    context_ptr: u32,     // WASI-NN execution context
    _graph_ptr: u32,      // WASI-NN graph handle
    config: C,            // Model configuration
}
```

#### lib.rs:main:92-119
**Model Loading Process**
1. Load OpenVINO IR format (XML + binary weights) via WASI-NN
2. Initialize execution context
3. Store model configuration for validation

#### lib.rs:main:121-168
**Inference Pipeline**
1. Set input tensor in WASI-NN context
2. Execute compute operation
3. Retrieve output probabilities
4. Find highest confidence classification
5. Return result with class index and probability

#### main.rs:main:6-48
**Model Lifecycle Management**
- Global `OnceLock<MobilnetModel>` for singleton pattern
- Filesystem loading fallback (`load_model_fs`)
- Thread-safe model initialization

#### main.rs:main:51-80
**FFI Load Function**
- Validates memory pointers and lengths
- Creates model from raw XML/weights buffers
- Returns status codes: 0=success, 1=already loaded, -1=error

#### main.rs:main:83-132
**FFI Inference Function** 
- Validates input tensor pointer/length
- Auto-loads model from filesystem if not initialized
- Executes inference and writes results to provided memory location
- Returns detailed error codes for different failure modes

### Dependencies

#### Core Dependencies
- **wasi-nn 0.1.0**: WebAssembly System Interface for Neural Networks
- **log 0.4**: Structured logging framework
- **wasm-logger 0.2**: WASM-compatible logger implementation  
- **env_logger 0.10**: Environment-based logger for native compilation

### Build Targets
- **WASM32-WASI**: Primary target for server integration
- **Native**: For standalone testing and development

### Error Handling Strategy
- Result-based error propagation for Rust APIs
- C-style return codes for FFI interface
- Comprehensive logging at all error points
- Graceful degradation with fallback model loading

## Component 2: @cascadia-demo/server

### Purpose
An HTTP server that provides a REST API for image classification by hosting and orchestrating the WASM inferencer module. Handles image preprocessing, WASM runtime management, and client communication.

### Core Architecture

#### Async HTTP Server Stack
- **Hyper 1.6**: High-performance HTTP implementation
- **Tokio**: Async runtime for concurrent request handling
- **Multi-threaded**: Separate inference thread pool to prevent blocking

#### WASM Runtime Integration
- **Wasmtime**: WebAssembly runtime engine
- **WASI Support**: Full WASI preview1 implementation
- **WASI-NN**: Neural network system interface with OpenVINO backend

### Key Components

#### main.rs:main:42-101
**Inference Request Handler**
```rust
async fn infer(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<InferenceRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>>
```

**Process Flow:**
1. Validate Content-Type header (must be `image/jpeg`)
2. Extract JPEG bytes from request body
3. Convert JPEG to raw BGR tensor using OpenCV
4. Send inference request to dedicated thread pool
5. Wait for WASM inference result
6. Return JSON response with classification

#### main.rs:main:103-134
**Real-time Logging Endpoint**
- Server-Sent Events (SSE) stream at `/logs`
- Broadcasts server and inference logs in real-time
- CORS-enabled for web frontend integration
- Auto-reconnecting stream with error handling

#### main.rs:main:136-166
**Request Routing & CORS**
- `OPTIONS` preflight handling for CORS
- Route mapping: `GET /logs`, `POST /infer`
- CORS headers for `localhost:8000` origin
- Standardized error responses

#### main.rs:main:169-210
**Server Lifecycle**
- TCP listener on `127.0.0.1:3000`
- Dedicated inference thread with WASM runtime
- Per-connection task spawning
- Shared channels for communication

#### runtime.rs:main:10-35
**WASM Context Setup**
```rust
struct Context {
    wasi: WasiP1Ctx,      // WASI preview1 context
    wasi_nn: WasiNnCtx,   // Neural network context
}
```

**Configuration:**
- Inherits stdio for debugging
- Preopens `fixture/` directory for model files
- Initializes OpenVINO backend with model preloading
- Sets up in-memory model registry

#### runtime.rs:main:44-64
**WASM Instance Management**
```rust
impl WasmInstance {
    pub fn new(engine: Arc<Engine>, module: Arc<Module>) -> anyhow::Result<WasmInstance>
}
```

**Setup Process:**
1. Create Wasmtime store with custom context
2. Configure WASI and WASI-NN linkers
3. Instantiate WASM module
4. Extract memory export for direct access

#### runtime.rs:main:66-82
**Inference Execution**
- Direct memory manipulation for tensor data transfer
- FFI call to WASM `infer` function
- Result extraction from fixed memory location
- Type conversion (u32 label, f32 confidence)

#### tensor.rs:main:4-55
**Image Preprocessing Pipeline**
```rust
pub fn jpeg_to_raw_bgr(jpeg_bytes: Vec<u8>, log_sender: &LogSender<String>) -> anyhow::Result<Vec<u8>>
```

**Steps:**
1. **Decode**: JPEG bytes ’ OpenCV Mat (BGR format)
2. **Resize**: Any size ’ 224x224 pixels (bilinear interpolation)  
3. **Convert**: uint8 ’ float32 with normalization
4. **Reorder**: HWC (height-width-channels) ’ NCHW format
5. **Serialize**: f32 array ’ raw bytes for WASM transfer

#### utils.rs:main:15-23
**Data Structures**
```rust
pub struct InferenceResult(pub u32, pub f32);  // (label_index, confidence)
pub struct InferenceRequest {
    pub tensor_bytes: Vec<u8>,                  // Preprocessed tensor data
    pub responder: oneshot::Sender<InferenceResult>, // Response channel
}
```

### Dependencies

#### HTTP & Async Runtime
- **tokio 1.45.0**: Async runtime with full feature set
- **hyper 1.6**: HTTP server implementation
- **hyper-util 0.1.11**: HTTP utilities and adapters
- **http-body-util 0.1.3**: Body handling utilities
- **futures 0.3**: Future combinators and utilities
- **tokio-stream 0.1**: Stream utilities for SSE

#### Image Processing  
- **opencv 0.94.4**: Computer vision library for image preprocessing
- **image 0.25.6**: Image format support (JPEG decoding)

#### WASM Runtime
- **wasmtime 32.0.0**: WebAssembly runtime engine
- **wasmtime-wasi 32.0.0**: WASI system interface implementation
- **wasmtime-wasi-nn 32.0.0**: Neural network interface with OpenVINO backend

#### Utilities
- **bytes 1.10.1**: Efficient byte buffer operations
- **serde 1.0.219**: Serialization framework for JSON responses
- **serde_json 1.0.140**: JSON serialization implementation
- **anyhow 1.0.98**: Error handling and context
- **tracing 0.1**: Structured logging framework
- **tracing-subscriber 0.3**: Log output formatting

### Network Configuration
- **Listen Address**: `127.0.0.1:3000`
- **CORS Origin**: `http://localhost:8000`
- **Allowed Methods**: `POST, OPTIONS, GET`
- **Content-Type**: `image/jpeg` for inference, `application/json` for responses

### Error Handling
- HTTP status codes for different error conditions
- Detailed error logging with request context  
- Graceful degradation for WASM runtime failures
- Channel-based error propagation between async tasks

## Dependencies and External Requirements

### System Dependencies

#### OpenVINO Runtime
- **Purpose**: CPU-optimized neural network inference engine
- **Integration**: Via WASI-NN interface in WASM context
- **Models**: Supports OpenVINO IR format (XML + binary weights)
- **Backend**: Configured for CPU execution target

#### OpenCV
- **Version**: 0.94.4 with specific feature flags
- **Features**: `clang-runtime`, `imgcodecs`, `imgproc`
- **Purpose**: JPEG decoding, image resizing, color space conversion
- **Format Support**: JPEG input, BGR color space processing

### Rust Toolchain Requirements

#### WASM Target
- **Target**: `wasm32-wasip1` (WebAssembly System Interface Preview 1)
- **Toolchain**: Recent Rust with WASM support
- **Build**: `cargo build --target wasm32-wasip1`

#### Native Target  
- **Target**: Host architecture (x86_64, ARM64)
- **Purpose**: Server compilation and development
- **Requirements**: Standard Rust toolchain 2021 edition

### Model Files

#### Required Files
Located in `fixture/` directory:
- **model.xml**: OpenVINO IR model definition
- **model.bin**: Model weights in binary format
- **tensor.bgr**: Example tensor data for testing

#### Model Format
- **Type**: ImageNet classification model
- **Input**: 224x224x3 RGB images
- **Output**: 1001 class probabilities (ImageNet + background)
- **Optimization**: Intel OpenVINO optimized for CPU inference

### Runtime Dependencies

#### WASM Runtime
- **Engine**: Wasmtime 32.0.0
- **WASI**: Preview 1 implementation
- **Memory**: Linear memory for tensor data transfer
- **Security**: Sandboxed execution environment

#### Network Requirements
- **Ports**: TCP 3000 for HTTP server
- **Protocols**: HTTP/1.1, Server-Sent Events
- **CORS**: Configured for localhost:8000 origin

## API Contracts and Data Formats

### HTTP Endpoints

#### POST /infer
**Purpose**: Perform image classification inference

**Request:**
- **Method**: POST
- **Content-Type**: `image/jpeg`
- **Body**: Raw JPEG image bytes
- **Size Limit**: Bounded by available memory

**Response:**
- **Content-Type**: `application/json`
- **Format**: 
  ```json
  [label_index, confidence]
  ```
- **Example**:
  ```json
  [285, 0.8547]
  ```

**Status Codes:**
- `200 OK`: Successful inference
- `400 Bad Request`: Missing Content-Type header
- `415 Unsupported Media Type`: Non-JPEG content
- `500 Internal Server Error`: Inference or processing failure

#### GET /logs
**Purpose**: Real-time server and inference logging

**Request:**
- **Method**: GET
- **Headers**: Standard HTTP headers

**Response:**
- **Content-Type**: `text/event-stream`
- **Format**: Server-Sent Events
- **Pattern**:
  ```
  data: [component/file.rs] Log message content.
  
  ```
- **Connection**: Keep-alive, auto-reconnecting

**CORS Headers** (All Endpoints):
```
Access-Control-Allow-Origin: http://localhost:8000
Access-Control-Allow-Methods: POST, OPTIONS, GET  
Access-Control-Allow-Headers: Content-Type, x-session-id
```

### Internal Data Formats

#### Tensor Format
**Layout**: NCHW (channels-first)
- **N**: Batch size (always 1)
- **C**: Channels (3 for RGB)
- **H**: Height (224 pixels)
- **W**: Width (224 pixels)

**Data Type**: `f32` (32-bit floating point)
**Total Size**: 1 × 3 × 224 × 224 × 4 bytes = 602,112 bytes

**Channel Order**: 
1. Channel 0: Blue values [0..224×224-1]
2. Channel 1: Green values [224×224..2×224×224-1]  
3. Channel 2: Red values [2×224×224..3×224×224-1]

#### FFI Interface (WASM)

**load_model Function:**
```c
int load_model(int xml_ptr, int xml_len, int weights_ptr, int weights_len);
```
- **Returns**: 0=success, 1=already loaded, -1=error
- **Memory**: Pointers to XML and binary weight data

**infer Function:**
```c
int infer(int tensor_ptr, int tensor_len, int result_ptr);
```
- **Returns**: 0=success, -1=invalid params, -2=model load fail, -3=invalid tensor, -4=inference fail
- **Input**: Pointer to 602,112 byte tensor data
- **Output**: 8 bytes at result_ptr (4-byte u32 label + 4-byte f32 confidence)

### Error Response Format

**HTTP Error Responses:**
- Plain text error messages
- Standard HTTP status codes
- CORS headers included

**WASM Error Codes:**
- Negative values indicate different failure modes
- Logged with detailed context information
- Propagated through async channels to HTTP layer

## Build and Deployment Process

### Build Requirements

#### Environment Setup
```bash
# Install Rust with WASM target
rustup target add wasm32-wasip1

# Install required system dependencies (Ubuntu/Debian)
sudo apt-get install libopencv-dev clang

# Install required system dependencies (macOS)
brew install opencv
```

#### Workspace Structure
```
cascadia-demo/
   inferencer/           # WASM inference module
      Cargo.toml
      src/
          lib.rs       # Core inference logic
          main.rs      # FFI interface
   server/              # HTTP server
      Cargo.toml
      src/
          main.rs      # Server and routing
          runtime.rs   # WASM runtime management
          tensor.rs    # Image preprocessing
          utils.rs     # Shared types
   fixture/             # Model files
       model.xml        # OpenVINO IR definition
       model.bin        # Model weights
       tensor.bgr       # Test data
```

### Build Process

#### 1. Build Inferencer (WASM)
```bash
cd cascadia-demo/inferencer
cargo build --target wasm32-wasip1 --release
```
**Output**: `../target/wasm32-wasip1/release/inferencer.wasm`

#### 2. Build Server (Native)  
```bash
cd cascadia-demo/server
cargo build --release
```
**Output**: `../target/release/server`

### Deployment Configuration

#### Model Files
Ensure model files are present in server working directory:
```
server_binary_location/
   fixture/
       model.xml
       model.bin
       tensor.bgr
```

#### WASM Module Location
Server expects WASM module at:
```
../target/wasm32-wasip1/debug/inferencer.wasm
```

### Runtime Requirements

#### System Libraries
- **OpenVINO**: Runtime libraries for neural network inference
- **OpenCV**: Image processing libraries
- **WASM Runtime**: Wasmtime provides embedded runtime

#### Network Configuration
- **Port**: 3000 (configurable in main.rs:171)
- **Interface**: 127.0.0.1 (localhost only)
- **CORS**: Configured for localhost:8000 frontend

#### Performance Considerations
- **Concurrency**: Multi-threaded request handling
- **Memory**: WASM module instances per inference request
- **CPU**: OpenVINO optimized for CPU inference
- **I/O**: Async file operations for model loading

### Production Deployment Notes

#### Security
- WASM provides sandboxed execution environment
- Network binding limited to localhost
- File system access restricted to model directory

#### Scaling
- Stateless design enables horizontal scaling
- Each request creates isolated WASM instance
- No shared state between inference requests

#### Monitoring
- Real-time logging via `/logs` endpoint
- Structured logging with component identification
- Error tracking with detailed context

#### Resource Management
- WASM instances auto-cleanup after inference
- Model loading cached globally per process
- OpenCV operations optimized for batch processing

---

This documentation provides the complete specification needed to re-implement both the inferencer WASM module and HTTP server with identical functionality and architecture. The implementation details, dependency requirements, and API contracts ensure compatibility with the existing system design.