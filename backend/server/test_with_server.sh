#!/bin/bash
set -e

# Source environment for OpenVINO
source ~/.zshrc

echo "Building inferencer WASM..."
cd .. && make inferencer-build && cd server

echo "Starting server..."
cargo run &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
sleep 3

# Run tests
echo "Running tests..."
cargo test
TEST_RESULT=$?

# Kill server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true

echo "Tests complete!"
exit $TEST_RESULT