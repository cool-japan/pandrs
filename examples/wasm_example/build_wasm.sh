#!/bin/bash

# Ensure wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack is not installed. Installing..."
    cargo install wasm-pack
fi

# Build the WebAssembly package
echo "Building WebAssembly package..."
wasm-pack build --target web --out-dir examples/wasm_example/pkg --features wasm

echo "Build completed! WebAssembly package is available in examples/wasm_example/pkg/"
echo "To run the example, serve the directory with a local web server, for example:"
echo "cd examples/wasm_example && python -m http.server 8080"
echo "Then open http://localhost:8080 in your browser."