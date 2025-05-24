#!/bin/bash
set -eo pipefail

source ~/.zshrc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "starting development servers..."

REQUIRED_DIRS=("backend/server" "backend/inferencer" "frontend")
for dir in "${REQUIRED_DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "Error: Required directory '$dir' not found in $SCRIPT_DIR" >&2
    exit 1
  fi
done

# Initialize PIDs as empty
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo -e "\nstopping servers..."
  if [[ -n "$BACKEND_PID" ]]; then
    kill $BACKEND_PID 2>/dev/null || true
  fi
  if [[ -n "$FRONTEND_PID" ]]; then
    kill $FRONTEND_PID 2>/dev/null || true
  fi
  wait
  echo "servers stopped."
}
trap cleanup INT TERM

echo "starting Rust backend..."
cd "$SCRIPT_DIR/backend"
if ! make inferencer-build; then
  echo "error: failed to build inferencer" >&2
  exit 1
fi

cd server
cargo run &
BACKEND_PID=$!
if ! kill -0 $BACKEND_PID 2>/dev/null; then
  echo "error: failed to start backend server" >&2
  exit 1
fi

echo "starting frontend server..."
cd "$SCRIPT_DIR/old_frontend"
python3 -m http.server 8000 &
FRONTEND_PID=$!
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
  echo "error: failed to start frontend server" >&2
  kill $BACKEND_PID 2>/dev/null || true
  exit 1
fi

echo -e "\n\033[1mbackend running on http://localhost:3000\033[0m"
echo -e "\033[1mfrontend running on http://localhost:8000\033[0m"
echo "press Ctrl+C to stop both servers"

wait