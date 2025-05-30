#!/bin/bash
# Test script for text model inference

echo "=== Testing Text Model Inference ==="
cd /Users/sharshe/Documents/CL/OpenVINO-Llama/backend/server

# Set OpenVINO library path
export DYLD_LIBRARY_PATH=/opt/homebrew/lib

# Build the inferencer WASM with our test
cd ../gpt2_rust
cargo build --target wasm32-wasip1
cp target/wasm32-wasip1/debug/test_gpt2_rust.wasm ../server/

# Run wasmtime directly with our test
cd ../server
wasmtime \
  --dir=fixture::fixture \
  test_gpt2_rust.wasm \
  --invoke test_wasi_nn_inference 1