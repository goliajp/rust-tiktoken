//! WebAssembly bindings for the tiktoken BPE tokenizer.
//!
//! Provides browser-compatible wrappers around the core `tiktoken` crate,
//! enabling high-performance token encoding, decoding, counting, and
//! cost estimation directly in JavaScript/TypeScript applications.
//!
//! All encoding instances are cached globally via `OnceLock`, so repeated
//! calls to `getEncoding()` with the same name return the same underlying data.

use wasm_bindgen::prelude::{wasm_bindgen, JsError};

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

    /// Count tokens, recognizing special tokens.
    ///
    /// Like `count()` but special tokens (e.g. `<|endoftext|>`) are counted
    /// as single tokens instead of being split into sub-word pieces.
    #[wasm_bindgen(js_name = countWithSpecialTokens)]
    pub fn count_with_special_tokens(&self, text: &str) -> usize {
        self.bpe.count_with_special_tokens(text)
    }

    /// Get the number of regular (non-special) tokens in the vocabulary.
    #[wasm_bindgen(js_name = vocabSize, getter)]
    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }

    /// Get the number of special tokens in the vocabulary.
    #[wasm_bindgen(js_name = numSpecialTokens, getter)]
    pub fn num_special_tokens(&self) -> usize {
        self.bpe.num_special_tokens()
    }

    /// Get the encoding name (e.g. `"cl100k_base"`).
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.to_string()
    }
}

/// List all available encoding names.
///
/// Returns an array of strings: `["cl100k_base", "o200k_base", ...]`
#[wasm_bindgen(js_name = listEncodings)]
pub fn list_encodings() -> Vec<String> {
    tiktoken::list_encodings()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Get an encoding by name.
///
/// Supported encodings:
/// - `"cl100k_base"` — GPT-4, GPT-3.5-turbo
/// - `"o200k_base"` — GPT-4o, GPT-4.1, o1, o3
/// - `"p50k_base"` — text-davinci-002/003
/// - `"p50k_edit"` — text-davinci-edit
/// - `"r50k_base"` — GPT-3 (davinci, curie, etc.)
/// - `"llama3"` — Meta Llama 3/4
/// - `"deepseek_v3"` — DeepSeek V3/R1
/// - `"qwen2"` — Qwen 2/2.5/3
/// - `"mistral_v3"` — Mistral/Codestral/Pixtral
///
/// Throws `Error` for unknown encoding names.
#[wasm_bindgen(js_name = getEncoding)]
pub fn get_encoding(name: &str) -> Result<Encoding, JsError> {
    // look up the static name from tiktoken's canonical list (single source of truth)
    let static_name = tiktoken::list_encodings()
        .iter()
        .find(|&&n| n == name)
        .ok_or_else(|| JsError::new(&format!("unknown encoding: {name}")))?;
    let bpe = tiktoken::get_encoding(name)
        .ok_or_else(|| JsError::new(&format!("unknown encoding: {name}")))?;
    Ok(Encoding {
        name: static_name,
        bpe,
    })
}

/// Get an encoding for a model name (e.g. `"gpt-4o"`, `"o3-mini"`, `"llama-4"`, `"deepseek-r1"`).
///
/// Supports models from OpenAI, Meta, DeepSeek, Qwen, and Mistral.
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

/// Map a model name to its encoding name without loading the encoding.
///
/// Returns the encoding name string (e.g. `"o200k_base"`) or `null` for unknown models.
#[wasm_bindgen(js_name = modelToEncoding)]
pub fn model_to_encoding(model: &str) -> Option<String> {
    tiktoken::model_to_encoding(model).map(|s| s.to_string())
}

/// Estimate cost in USD for a given model, input token count, and output token count.
///
/// Supports OpenAI, Anthropic Claude, Google Gemini, Meta Llama, DeepSeek, Qwen, and Mistral models.
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

/// Get model pricing and metadata.
///
/// Returns a typed object with: `id`, `provider`, `inputPer1m`, `outputPer1m`,
/// `cachedInputPer1m`, `contextWindow`, `maxOutput`.
///
/// Throws `Error` for unknown model ids.
#[wasm_bindgen(js_name = getModelInfo)]
pub fn get_model_info(model_id: &str) -> Result<ModelInfo, JsError> {
    let model = tiktoken::pricing::get_model(model_id)
        .ok_or_else(|| JsError::new(&format!("unknown model: {model_id}")))?;
    Ok(convert_model(model))
}

/// List all supported models with pricing info.
///
/// Returns an array of `ModelInfo` objects.
#[wasm_bindgen(js_name = allModels)]
pub fn all_models() -> Vec<ModelInfo> {
    tiktoken::pricing::all_models()
        .iter()
        .map(convert_model)
        .collect()
}

/// List models filtered by provider name.
///
/// Provider names: `"OpenAI"`, `"Anthropic"`, `"Google"`, `"Meta"`, `"DeepSeek"`, `"Alibaba"`, `"Mistral"`.
/// Returns an empty array for unknown providers.
#[wasm_bindgen(js_name = modelsByProvider)]
pub fn models_by_provider(provider: &str) -> Vec<ModelInfo> {
    let Some(provider) = parse_provider(provider) else {
        return Vec::new();
    };

    tiktoken::pricing::models_by_provider(provider)
        .iter()
        .map(|m| convert_model(m))
        .collect()
}

fn convert_model(m: &tiktoken::pricing::Model) -> ModelInfo {
    ModelInfo {
        id: m.id,
        provider: m.provider.to_string(),
        input_per_1m: m.pricing.input_per_1m,
        output_per_1m: m.pricing.output_per_1m,
        cached_input_per_1m: m.pricing.cached_input_per_1m,
        context_window: m.context_window,
        max_output: m.max_output,
    }
}

fn parse_provider(s: &str) -> Option<tiktoken::pricing::Provider> {
    match s {
        "OpenAI" => Some(tiktoken::pricing::Provider::OpenAI),
        "Anthropic" => Some(tiktoken::pricing::Provider::Anthropic),
        "Google" => Some(tiktoken::pricing::Provider::Google),
        "Meta" => Some(tiktoken::pricing::Provider::Meta),
        "DeepSeek" => Some(tiktoken::pricing::Provider::DeepSeek),
        "Alibaba" => Some(tiktoken::pricing::Provider::Alibaba),
        "Mistral" => Some(tiktoken::pricing::Provider::Mistral),
        _ => None,
    }
}

/// Model pricing and metadata.
#[wasm_bindgen]
#[derive(Clone)]
pub struct ModelInfo {
    id: &'static str,
    provider: String,
    input_per_1m: f64,
    output_per_1m: f64,
    cached_input_per_1m: Option<f64>,
    context_window: u32,
    max_output: u32,
}

#[wasm_bindgen]
impl ModelInfo {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.to_string()
    }
    #[wasm_bindgen(getter)]
    pub fn provider(&self) -> String {
        self.provider.clone()
    }
    #[wasm_bindgen(getter, js_name = inputPer1m)]
    pub fn input_per_1m(&self) -> f64 {
        self.input_per_1m
    }
    #[wasm_bindgen(getter, js_name = outputPer1m)]
    pub fn output_per_1m(&self) -> f64 {
        self.output_per_1m
    }
    #[wasm_bindgen(getter, js_name = cachedInputPer1m)]
    pub fn cached_input_per_1m(&self) -> Option<f64> {
        self.cached_input_per_1m
    }
    #[wasm_bindgen(getter, js_name = contextWindow)]
    pub fn context_window(&self) -> u32 {
        self.context_window
    }
    #[wasm_bindgen(getter, js_name = maxOutput)]
    pub fn max_output(&self) -> u32 {
        self.max_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_encodings_roundtrip() {
        for &name in tiktoken::list_encodings() {
            let enc = get_encoding(name).unwrap();
            let text = "hello world 你好 🚀";
            let tokens = enc.encode(text);
            let decoded = enc.decode(&tokens);
            assert_eq!(decoded, text, "roundtrip failed for {name}");
        }
    }

    #[test]
    fn encoding_for_known_models() {
        let models = [
            "gpt-4o", "gpt-4", "gpt-3.5-turbo", "llama-4", "deepseek-r1", "qwen3", "mistral-large",
        ];
        for model in models {
            let enc = encoding_for_model(model);
            assert!(enc.is_ok(), "encoding_for_model failed for {model}");
        }
    }

    #[test]
    fn list_encodings_count() {
        let names = list_encodings();
        assert_eq!(names.len(), 9);
    }

    #[test]
    fn all_models_count() {
        let models = all_models();
        assert_eq!(models.len(), tiktoken::pricing::all_models().len());
    }

    #[test]
    fn models_by_valid_provider() {
        let openai = models_by_provider("OpenAI");
        assert!(!openai.is_empty());
        for m in &openai {
            assert_eq!(m.provider, "OpenAI");
        }
    }

    #[test]
    fn models_by_invalid_provider() {
        let unknown = models_by_provider("NonExistent");
        assert!(unknown.is_empty());
    }

    #[test]
    fn estimate_cost_known_model() {
        let cost = estimate_cost("gpt-4o", 1000, 1000).unwrap();
        assert!(cost > 0.0);
    }

    #[test]
    fn estimate_cost_unknown_model() {
        assert!(estimate_cost("fake-model", 1000, 1000).is_err());
    }

    #[test]
    fn get_model_info_known() {
        let info = get_model_info("gpt-4o").unwrap();
        assert_eq!(info.id(), "gpt-4o");
        assert_eq!(info.provider(), "OpenAI");
        assert!(info.context_window() > 0);
    }

    #[test]
    fn get_model_info_unknown() {
        assert!(get_model_info("fake-model").is_err());
    }

    #[test]
    fn unknown_encoding_error() {
        assert!(get_encoding("nonexistent").is_err());
    }

    #[test]
    fn unknown_model_encoding_error() {
        assert!(encoding_for_model("nonexistent-model-xyz").is_err());
    }

    #[test]
    fn model_to_encoding_known() {
        let name = model_to_encoding("gpt-4o");
        assert_eq!(name.as_deref(), Some("o200k_base"));
    }

    #[test]
    fn model_to_encoding_unknown() {
        assert!(model_to_encoding("fake-model").is_none());
    }

    #[test]
    fn parse_provider_all_variants() {
        assert!(parse_provider("OpenAI").is_some());
        assert!(parse_provider("Anthropic").is_some());
        assert!(parse_provider("Google").is_some());
        assert!(parse_provider("Meta").is_some());
        assert!(parse_provider("DeepSeek").is_some());
        assert!(parse_provider("Alibaba").is_some());
        assert!(parse_provider("Mistral").is_some());
        assert!(parse_provider("Unknown").is_none());
    }
}
