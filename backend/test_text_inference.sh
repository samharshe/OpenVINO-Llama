#!/bin/bash

# Start server in background
cd server
cargo run &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to start
sleep 3

# Test text inference
echo "Testing text inference..."
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: text/plain" \
  -d "Hello, this is a test of the text inference system." \
  -v

# Kill server
kill $SERVER_PID
echo "Server stopped"