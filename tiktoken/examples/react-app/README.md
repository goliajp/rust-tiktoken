# tiktoken-wasm React Demo

A React app demonstrating the tiktoken WASM tokenizer.

## Prerequisites

Build the WASM package first:

```bash
cd ../../tiktoken-wasm
wasm-pack build --target web --release
```

## Run

```bash
bun install
bun dev
```

## Features

- Encode / decode text with any tiktoken encoding
- Real-time token count and encode timing
- Roundtrip verification (encode → decode → compare)
- Cost estimation for OpenAI, Claude, and Gemini models
- Model pricing info display
