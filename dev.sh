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

echo "starting Rust backend..."
cd "$SCRIPT_DIR/backend"
make inferencer-build
cd server
cargo run &
BACKEND_PID=$!

echo "starting frontend server..."
cd "$SCRIPT_DIR/old_frontend"
python3 -m http.server 8000 &
FRONTEND_PID=$!

echo -e "\n\033[1mBackend running on http://localhost:3000\033[0m"
echo -e "\033[1mFrontend running on http://localhost:8000\033[0m"
echo "Press Ctrl+C to stop both servers"

cleanup() {
  echo -e "\nstopping servers..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait
  echo "servers stopped."
}
trap cleanup INT TERM

wait