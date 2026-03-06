//! WebAssembly bindings for the tiktoken BPE tokenizer.
//!
//! Provides browser-compatible wrappers around the core `tiktoken` crate,
//! enabling high-performance token encoding, decoding, counting, and
//! cost estimation directly in JavaScript/TypeScript applications.
//!
//! All encoding instances are cached globally via `OnceLock`, so repeated
//! calls to `getEncoding()` with the same name return the same underlying data.

use wasm_bindgen::prelude::*;

/// WASM wrapper around a tiktoken encoding instance.
///
/// Created via [`get_encoding`] or [`encoding_for_model`].
/// Call `.free()` when done to release WASM memory.
#[wasm_bindgen]
pub struct Encoding {
    /// encoding name (e.g. "cl100k_base") — always a static string
    name: &'static str,
    /// reference to the globally cached CoreBpe instance
    bpe: &'static tiktoken::CoreBpe,
}

#[wasm_bindgen]
impl Encoding {
    /// Encode text into token ids (returns `Uint32Array` in JS).
    ///
    /// Special tokens like `<|endoftext|>` are treated as ordinary text.
    /// Use `encodeWithSpecialTokens()` to recognize them.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode(text)
    }

    /// Encode text into token ids, recognizing special tokens.
    ///
    /// Special tokens (e.g. `<|endoftext|>`) are encoded as their designated ids
    /// instead of being split into sub-word pieces.
    #[wasm_bindgen(js_name = encodeWithSpecialTokens)]
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    /// Decode token ids back to a UTF-8 string.
    ///
    /// Uses lossy UTF-8 conversion — invalid byte sequences are replaced with U+FFFD.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes = self.bpe.decode(tokens);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Count tokens without building the full token id array.
    ///
    /// Faster than `encode(text).length` for cases where you only need the count.
    pub fn count(&self, text: &str) -> usize {
        self.bpe.count(text)
    }

    /// Get the encoding name (e.g. `"cl100k_base"`).
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.to_string()
    }
}

/// Get an encoding by name.
///
/// Supported: `"cl100k_base"`, `"o200k_base"`, `"p50k_base"`, `"p50k_edit"`, `"r50k_base"`.
///
/// Throws `Error` for unknown encoding names.
#[wasm_bindgen(js_name = getEncoding)]
pub fn get_encoding(name: &str) -> Result<Encoding, JsError> {
    let bpe = tiktoken::get_encoding(name)
        .ok_or_else(|| JsError::new(&format!("unknown encoding: {name}")))?;
    let static_name: &'static str = match name {
        "cl100k_base" => "cl100k_base",
        "o200k_base" => "o200k_base",
        "p50k_base" => "p50k_base",
        "p50k_edit" => "p50k_edit",
        "r50k_base" => "r50k_base",
        _ => return Err(JsError::new(&format!("unknown encoding: {name}"))),
    };
    Ok(Encoding {
        name: static_name,
        bpe,
    })
}

/// Get an encoding for an OpenAI model name (e.g. `"gpt-4o"`, `"o3-mini"`).
///
/// Automatically resolves the model name to the correct encoding.
/// Throws `Error` for unknown model names.
#[wasm_bindgen(js_name = encodingForModel)]
pub fn encoding_for_model(model: &str) -> Result<Encoding, JsError> {
    let name = tiktoken::model_to_encoding(model)
        .ok_or_else(|| JsError::new(&format!("unknown model: {model}")))?;
    let bpe = tiktoken::get_encoding(name)
        .ok_or_else(|| JsError::new(&format!("unknown encoding: {name}")))?;
    Ok(Encoding { name, bpe })
}

/// Estimate cost in USD for a given model, input token count, and output token count.
///
/// Supports OpenAI, Anthropic Claude, and Google Gemini models.
/// Throws `Error` for unknown model ids.
#[wasm_bindgen(js_name = estimateCost)]
pub fn estimate_cost(
    model_id: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> Result<f64, JsError> {
    tiktoken::pricing::estimate_cost(model_id, input_tokens as u64, output_tokens as u64)
        .ok_or_else(|| JsError::new(&format!("unknown model: {model_id}")))
}

/// Get model pricing and metadata as a JS object.
///
/// Returns an object with: `id`, `provider`, `input_per_1m`, `output_per_1m`,
/// `cached_input_per_1m`, `context_window`, `max_output`.
///
/// Throws `Error` for unknown model ids.
#[wasm_bindgen(js_name = getModelInfo)]
pub fn get_model_info(model_id: &str) -> Result<JsValue, JsError> {
    let model = tiktoken::pricing::get_model(model_id)
        .ok_or_else(|| JsError::new(&format!("unknown model: {model_id}")))?;

    let info = ModelInfo {
        id: model.id.to_string(),
        provider: model.provider.to_string(),
        input_per_1m: model.pricing.input_per_1m,
        output_per_1m: model.pricing.output_per_1m,
        cached_input_per_1m: model.pricing.cached_input_per_1m,
        context_window: model.context_window,
        max_output: model.max_output,
    };

    serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
}

/// Internal struct for JSON serialization of model info to JS.
#[derive(serde::Serialize)]
struct ModelInfo {
    id: String,
    provider: String,
    input_per_1m: f64,
    output_per_1m: f64,
    cached_input_per_1m: Option<f64>,
    context_window: u32,
    max_output: u32,
}
