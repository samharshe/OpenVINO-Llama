## Diagnostic Plan: wasi_nn::load Error Investigation

### Phase 1: Understand the Working Path
**Goal**: Trace exactly how the backend's working image inference loads models

1. **Trace InMemoryRegistry Loading** ✅ COMPLETED
   - Find where `registry.load()` is called in runtime.rs
   - Understand what `as_dir_loadable()` does with the OpenVINO backend
   - See if models are pre-processed before WASM even starts

   **FINDINGS**:
   - `registry.load()` is called in `runtime.rs:24-25` when `preload_model=true`
   - `as_dir_loadable()` returns a directory-scanning interface for OpenVINO
   - Models ARE pre-loaded into registry before WASM starts:
     ```
     WasmInstance::new() → Context::new() → registry.load() → scan ./fixture → pre-load models
     ```
   - The registry provides an optimization - models are loaded once at startup

2. **Check if Backend Ever Calls wasi_nn::load** ✅ COMPLETED
   - Search for successful wasi_nn::load calls in the inferencer
   - Determine if the working path uses a different API
   - Check if graph pointers are pre-created outside WASM

   **FINDINGS**:
   - YES! The inferencer DOES call `wasi_nn::load` directly in `lib.rs:67`
   - Same exact pattern as our test: `wasi_nn::load(&[&xml, &weights], GRAPH_ENCODING_OPENVINO, EXECUTION_TARGET_CPU)`
   - This means direct loading IS supported and works in production
   - **KEY INSIGHT**: The working inferencer succeeds with the same call that fails for us

3. **Examine OpenVINO Backend Configuration** ✅ COMPLETED
   - Look for OpenVINO initialization settings
   - Check for environment variables or config files
   - See if there's special setup before model loading

   **FINDINGS**:
   - OpenVINO requires runtime libraries installed on the system
   - On macOS: `/opt/homebrew/lib/libopenvino_c.dylib` (via Homebrew)
   - Requires `DYLD_LIBRARY_PATH=/opt/homebrew/lib` environment variable
   - The backend server also fails without this setup

### Phase 2: Compare Execution Contexts
**Goal**: Find differences between working inferencer and our test

1. **WASM Module Differences** ✅ COMPLETED
   - Compare how inferencer/main.rs is compiled vs our test
   - Check for different WASM flags or features
   - Look for initialization code we might be missing

   **FINDINGS**:
   - Same compilation: both use `cargo build --target wasm32-wasip1`
   - Same dependencies: wasi-nn 0.1.0, wasm-logger 0.2
   - Same initialization: wasm_logger::init()
   - No special WASM flags or missing init code

2. **Runtime Environment Differences** ✅ COMPLETED
   - Compare WasiNnCtx creation in both cases
   - Check if preload_model=true does something special
   - See if there are multiple OpenVINO backend instances

   **FINDINGS**:
   - **CRITICAL DIFFERENCE FOUND**: The working runtime calls `registry.load()`
   - Our test skipped this step, which initializes the OpenVINO backend
   - When we added `registry.load()`, we got: "Unable to find the `openvino_c` library"
   - **KEY INSIGHT**: The server ALSO fails with the same error!
   - This means the "working" system must have OpenVINO runtime libraries installed

3. **Timing and Sequencing** ✅ COMPLETED
   - Determine if models must be loaded before WASM starts
   - Check if there's a required initialization sequence
   - See if the registry holds pre-initialized graph pointers

   **FINDINGS**:
   - `registry.load()` MUST be called before WasiNnCtx creation
   - This initializes the OpenVINO backend properly
   - Without it, wasi_nn::load fails with "Unknown error" code 5

### Phase 3: Isolate the Failure Point
**Goal**: Narrow down exactly what's failing

1. **Test Minimal OpenVINO Operations** ✅ COMPLETED
   - Try loading an empty/minimal model
   - Test if OpenVINO backend is even initialized
   - Check if the error is from OpenVINO or wasi-nn layer

   **FINDINGS**:
   - Error was from OpenVINO backend not being initialized
   - With proper initialization, both models load successfully
   - Image model: 143KB XML, 13MB BIN → graph_ptr: 0
   - Text model: 5.4MB XML, 124MB BIN → graph_ptr: 1

### Root Cause Analysis ✅ SOLVED

The "Unknown error" code 5 was caused by TWO missing pieces:

1. **Missing `registry.load()` call**: This initializes the OpenVINO backend
2. **Missing OpenVINO libraries**: Requires system-installed OpenVINO with proper library path

### Solution Summary

To make wasi_nn::load work:

1. **Install OpenVINO**: `brew install openvino`
2. **Set library path**: `export DYLD_LIBRARY_PATH=/opt/homebrew/lib`
3. **Call registry.load()**: Initialize backend before creating WasiNnCtx
4. **Then call wasi_nn::load**: Works exactly as in the inferencer

### Current Status

✅ **PROBLEM SOLVED**: Both image and text models now load successfully!
- Image model loads with graph_ptr: 0
- Text model loads with graph_ptr: 1
- The approach is validated and working

**Remaining Issue**: Inference has a tensor size mismatch (256 vs 64) but that's a separate problem from the loading issue.

### Key Learnings

1. **OpenVINO requires system libraries**: Not bundled with the Rust crate
2. **registry.load() is critical**: Initializes the backend properly
3. **Direct wasi_nn::load works**: Once the backend is initialized
4. **Environment matters**: DYLD_LIBRARY_PATH required on macOS