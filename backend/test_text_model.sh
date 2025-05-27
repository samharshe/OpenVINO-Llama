#!/bin/bash

# This script tests text model loading with detailed logging

echo "Starting test server with detailed logging..."
cd server

# Set up environment variables for OpenVINO
source ~/.zshrc

# Enable debug logging
export RUST_LOG=debug

# Start the server
cargo run &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to start..."
sleep 5

# Send a simple text inference request
echo "Sending text inference request..."
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: text/plain" \
  -d "Hello, world!" \
  -v

# Kill the server
echo -e "\n\nStopping server..."
kill $SERVER_PID 2>/dev/null || true

echo "Test complete."