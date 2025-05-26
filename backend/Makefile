define USAGE
REQUIREMENTS:
- Cargo

USAGE:
> make [
	sense: see usage.
	clean: remove the build environment.
	serve: run in dev move.
	build: build the production executable.
]
endef

PWD = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

clean:
	@cargo clean

inferencer-build: 
	cargo build --package inferencer --target wasm32-wasip1

server-build:
	cargo build --package server

format:
	@cargo fix
	@cargo +nightly fmt

lint:
	@cargo clippy --all-targets --all-features -- -D warnings

test:
	cd server && cargo test

test-with-server: inferencer-build
	@echo "Starting server with tests..."
	@cd server && \
	bash -c 'source ~/.zshrc && cargo run & SERVER_PID=$$! && \
	sleep 3 && \
	echo "Running tests..." && \
	cargo test; \
	TEST_RESULT=$$?; \
	echo "Stopping server..." && \
	kill $$SERVER_PID 2>/dev/null || true; \
	exit $$TEST_RESULT'