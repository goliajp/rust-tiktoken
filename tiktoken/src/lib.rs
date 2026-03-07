//! High-performance pure-Rust BPE tokenizer compatible with OpenAI's tiktoken
//! and all mainstream LLM tokenizers.
//!
//! Supports 9 encodings across 5 providers: OpenAI (`cl100k_base`, `o200k_base`,
//! `p50k_base`, `p50k_edit`, `r50k_base`), Meta (`llama3`), DeepSeek (`deepseek_v3`),
//! Alibaba (`qwen2`), and Mistral (`mistral_v3`).
//!
//! Includes token encoding, decoding, counting, and multi-provider pricing.
//!
//! # Quick Start
//!
//! ```
//! // by encoding name
//! let enc = tiktoken::get_encoding("cl100k_base").unwrap();
//! let tokens = enc.encode("hello world");
//! let text = enc.decode_to_string(&tokens).unwrap();
//! assert_eq!(text, "hello world");
//!
//! // by model name
//! let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
//! let count = enc.count("hello world");
//! assert_eq!(count, 2);
//! ```

mod bpe;
pub mod encoding;
mod merge;
mod pretokenize;
pub mod pricing;
mod vocab;

pub use bpe::CoreBpe;

use std::sync::OnceLock;

static CL100K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static O200K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static P50K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static P50K_EDIT: OnceLock<CoreBpe> = OnceLock::new();
static R50K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static LLAMA3: OnceLock<CoreBpe> = OnceLock::new();
static DEEPSEEK_V3: OnceLock<CoreBpe> = OnceLock::new();
static QWEN2: OnceLock<CoreBpe> = OnceLock::new();
static MISTRAL_V3: OnceLock<CoreBpe> = OnceLock::new();

/// All available encoding names.
///
/// Returns the list of encoding names that can be passed to [`get_encoding`].
///
/// # Examples
///
/// ```
/// let names = tiktoken::list_encodings();
/// assert!(names.contains(&"cl100k_base"));
/// assert!(names.contains(&"llama3"));
/// assert_eq!(names.len(), 9);
/// ```
pub fn list_encodings() -> &'static [&'static str] {
    &[
        "cl100k_base",
        "o200k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
        "llama3",
        "deepseek_v3",
        "qwen2",
        "mistral_v3",
    ]
}

/// Get a cached tokenizer by encoding name.
///
/// Supported encodings:
/// - OpenAI: `cl100k_base`, `o200k_base`, `p50k_base`, `p50k_edit`, `r50k_base`
/// - Meta: `llama3`
/// - DeepSeek: `deepseek_v3`
/// - Alibaba: `qwen2`
/// - Mistral: `mistral_v3`
pub fn get_encoding(name: &str) -> Option<&'static CoreBpe> {
    match name {
        "cl100k_base" => Some(CL100K_BASE.get_or_init(encoding::cl100k_base)),
        "o200k_base" => Some(O200K_BASE.get_or_init(encoding::o200k_base)),
        "p50k_base" => Some(P50K_BASE.get_or_init(encoding::p50k_base)),
        "p50k_edit" => Some(P50K_EDIT.get_or_init(encoding::p50k_edit)),
        "r50k_base" => Some(R50K_BASE.get_or_init(encoding::r50k_base)),
        "llama3" => Some(LLAMA3.get_or_init(encoding::llama3)),
        "deepseek_v3" => Some(DEEPSEEK_V3.get_or_init(encoding::deepseek_v3)),
        "qwen2" => Some(QWEN2.get_or_init(encoding::qwen2)),
        "mistral_v3" => Some(MISTRAL_V3.get_or_init(encoding::mistral_v3)),
        _ => None,
    }
}

/// Get a cached tokenizer by OpenAI model name.
///
/// Maps model name prefixes to their encoding.
/// Returns `None` for unknown or non-OpenAI models.
pub fn encoding_for_model(model: &str) -> Option<&'static CoreBpe> {
    model_to_encoding(model).and_then(get_encoding)
}

/// Map a model name to its encoding name.
///
/// Returns the encoding name (e.g. `"o200k_base"`) for the given model,
/// or `None` for unknown models. Supports OpenAI, Meta, DeepSeek, Qwen, and Mistral models.
pub fn model_to_encoding(model: &str) -> Option<&'static str> {
    // order matters: more specific prefixes must come before less specific ones.
    // e.g. "gpt-4o" must be checked before "gpt-4" since starts_with("gpt-4")
    // would also match "gpt-4o".

    // o200k_base models (newest first)
    if model.starts_with("o4-mini")
        || model.starts_with("o3")
        || model.starts_with("o1")
        || model.starts_with("gpt-4.1")
        || model.starts_with("gpt-4o")
        || model.starts_with("chatgpt-4o")
    {
        return Some("o200k_base");
    }

    // cl100k_base models
    if model.starts_with("gpt-4")
        || model.starts_with("gpt-3.5")
        || model.starts_with("text-embedding-ada")
        || model.starts_with("text-embedding-3")
    {
        return Some("cl100k_base");
    }

    // p50k_base models
    if model.starts_with("text-davinci-003")
        || model.starts_with("text-davinci-002")
        || model.starts_with("code-davinci")
        || model.starts_with("code-cushman")
    {
        return Some("p50k_base");
    }

    // r50k_base models
    if model.starts_with("text-davinci-001")
        || model.starts_with("text-curie")
        || model.starts_with("text-babbage")
        || model.starts_with("text-ada")
        || model.starts_with("davinci")
        || model.starts_with("curie")
        || model.starts_with("babbage")
        || model.starts_with("ada")
    {
        return Some("r50k_base");
    }

    // llama models (llama3 encoding covers all llama 3.x and 4.x)
    if model.starts_with("llama-")
        || model.starts_with("llama3")
        || model.starts_with("llama4")
        || model.starts_with("Llama-")
        || model.starts_with("Meta-Llama-")
    {
        return Some("llama3");
    }

    // deepseek models
    if model.starts_with("deepseek") || model.starts_with("DeepSeek") {
        return Some("deepseek_v3");
    }

    // qwen models
    if model.starts_with("qwen") || model.starts_with("Qwen") {
        return Some("qwen2");
    }

    // mistral / mixtral / codestral / pixtral models
    if model.starts_with("mistral")
        || model.starts_with("Mistral")
        || model.starts_with("mixtral")
        || model.starts_with("Mixtral")
        || model.starts_with("codestral")
        || model.starts_with("Codestral")
        || model.starts_with("pixtral")
        || model.starts_with("Pixtral")
    {
        return Some("mistral_v3");
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // encoding lookup

    #[test]
    fn test_get_encoding_known() {
        for name in [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "llama3",
            "deepseek_v3",
            "qwen2",
            "mistral_v3",
        ] {
            assert!(get_encoding(name).is_some(), "missing encoding: {name}");
        }
    }

    #[test]
    fn test_get_encoding_unknown() {
        assert!(get_encoding("nonexistent").is_none());
    }

    // model mapping

    #[test]
    fn test_encoding_for_latest_openai_models() {
        for model in [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o4-mini",
        ] {
            let enc = encoding_for_model(model);
            assert!(enc.is_some(), "no encoding for {model}");
        }
    }

    #[test]
    fn test_encoding_for_gpt4_models() {
        for model in ["gpt-4", "gpt-4-turbo", "gpt-4-0613"] {
            assert!(
                encoding_for_model(model).is_some(),
                "no encoding for {model}"
            );
        }
    }

    #[test]
    fn test_encoding_for_gpt35() {
        assert!(encoding_for_model("gpt-3.5-turbo").is_some());
    }

    #[test]
    fn test_encoding_for_model_unknown() {
        assert!(encoding_for_model("unknown-model").is_none());
    }

    // encode/decode roundtrip

    #[test]
    fn test_cl100k_roundtrip() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "hello world";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_o200k_roundtrip() {
        let enc = get_encoding("o200k_base").unwrap();
        let text = "hello world, 你好世界 🚀";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_p50k_roundtrip() {
        let enc = get_encoding("p50k_base").unwrap();
        let decoded = enc.decode(&enc.encode("hello world"));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), "hello world");
    }

    #[test]
    fn test_r50k_roundtrip() {
        let enc = get_encoding("r50k_base").unwrap();
        let decoded = enc.decode(&enc.encode("hello world"));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), "hello world");
    }

    #[test]
    fn test_unicode_roundtrip() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "café résumé naïve über 日本語 한국어 العربية";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    // count

    #[test]
    fn test_count_equals_encode_len() {
        let enc = get_encoding("cl100k_base").unwrap();
        for text in [
            "hello world",
            "The quick brown fox.",
            "你好世界",
            "",
            "a",
            "  \n\n  ",
        ] {
            assert_eq!(
                enc.count(text),
                enc.encode(text).len(),
                "mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn test_o200k_count_equals_encode_len() {
        let enc = get_encoding("o200k_base").unwrap();
        for text in ["hello world", "OpenAI GPT-4o is great", ""] {
            assert_eq!(
                enc.count(text),
                enc.encode(text).len(),
                "mismatch for {text:?}"
            );
        }
    }

    // special tokens

    #[test]
    fn test_count_with_special_tokens_cl100k() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "hello<|endoftext|>world";
        assert_eq!(
            enc.count_with_special_tokens(text),
            enc.encode_with_special_tokens(text).len()
        );
    }

    #[test]
    fn test_special_tokens_cl100k() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "hello<|endoftext|>world";
        let with = enc.encode_with_special_tokens(text);
        assert!(with.contains(&100257));
        let without = enc.encode(text);
        assert!(!without.contains(&100257));
    }

    // edge cases

    #[test]
    fn test_empty_input() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert!(enc.encode("").is_empty());
        assert_eq!(enc.count(""), 0);
    }

    #[test]
    fn test_cached_instance_is_same() {
        let a = get_encoding("cl100k_base").unwrap() as *const CoreBpe;
        let b = get_encoding("cl100k_base").unwrap() as *const CoreBpe;
        assert_eq!(a, b);
    }

    #[test]
    fn test_long_text_roundtrip() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "word ".repeat(10000);
        let decoded = enc.decode(&enc.encode(&text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_whitespace_roundtrip() {
        let enc = get_encoding("cl100k_base").unwrap();
        for text in [" ", "  ", "\n", "\t", "  \n  \n  "] {
            let decoded = enc.decode(&enc.encode(text));
            assert_eq!(
                std::str::from_utf8(&decoded).unwrap(),
                text,
                "failed for {text:?}"
            );
        }
    }

    #[test]
    fn test_single_characters() {
        let enc = get_encoding("cl100k_base").unwrap();
        for ch in ['a', 'Z', '0', '!', '@', '#'] {
            let text = ch.to_string();
            let decoded = enc.decode(&enc.encode(&text));
            assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
        }
    }

    #[test]
    fn test_encoding_deterministic() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "deterministic check";
        assert_eq!(enc.encode(text), enc.encode(text));
    }

    // exact token sequence tests verified against Python tiktoken 0.12.0
    #[test]
    fn test_exact_tokens_hello_world() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode("hello world"), vec![15339, 1917]);
    }

    #[test]
    fn test_exact_tokens_spaces_before_word() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode("  hello"), vec![220, 24748]);
    }

    #[test]
    fn test_exact_tokens_three_spaces() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode("   hello"), vec![256, 24748]);
    }

    #[test]
    fn test_exact_tokens_trailing_spaces() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode("hello   "), vec![15339, 262]);
    }

    #[test]
    fn test_exact_tokens_mixed_whitespace() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode("hello\t  world"), vec![15339, 3762, 1917]);
    }

    #[test]
    fn test_exact_tokens_unicode() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode("你好世界"), vec![57668, 53901, 3574, 244, 98220]);
    }

    #[test]
    fn test_exact_tokens_empty() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.encode(""), Vec::<u32>::new());
    }

    // decode_to_string

    #[test]
    fn test_decode_to_string_valid() {
        let enc = get_encoding("cl100k_base").unwrap();
        let tokens = enc.encode("hello world");
        assert_eq!(enc.decode_to_string(&tokens).unwrap(), "hello world");
    }

    #[test]
    fn test_decode_to_string_empty() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.decode_to_string(&[]).unwrap(), "");
    }

    #[test]
    fn test_decode_to_string_unicode() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "日本語テスト 🎉";
        let tokens = enc.encode(text);
        assert_eq!(enc.decode_to_string(&tokens).unwrap(), text);
    }

    // model_to_encoding (now public)

    #[test]
    fn test_model_to_encoding_o200k() {
        for model in [
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o1",
            "o3",
            "o3-pro",
            "o4-mini",
            "chatgpt-4o",
        ] {
            assert_eq!(
                model_to_encoding(model),
                Some("o200k_base"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_cl100k() {
        for model in [
            "gpt-4",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
            "text-embedding-3-small",
        ] {
            assert_eq!(
                model_to_encoding(model),
                Some("cl100k_base"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_p50k() {
        for model in [
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
            "code-cushman-001",
        ] {
            assert_eq!(
                model_to_encoding(model),
                Some("p50k_base"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_r50k() {
        for model in ["text-davinci-001", "davinci", "curie", "babbage", "ada"] {
            assert_eq!(
                model_to_encoding(model),
                Some("r50k_base"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_llama3() {
        for model in [
            "llama-3.1-70b",
            "llama3-8b",
            "Meta-Llama-3.1-8B",
            "llama-4-scout",
            "llama-4-maverick",
        ] {
            assert_eq!(
                model_to_encoding(model),
                Some("llama3"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_deepseek() {
        for model in ["deepseek-v3", "DeepSeek-R1", "deepseek-chat"] {
            assert_eq!(
                model_to_encoding(model),
                Some("deepseek_v3"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_qwen() {
        for model in [
            "qwen2.5-72b",
            "Qwen2.5-7B",
            "qwen3-32b",
            "qwen3-max",
            "qwen3-coder",
        ] {
            assert_eq!(
                model_to_encoding(model),
                Some("qwen2"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_mistral() {
        for model in [
            "mistral-small-latest",
            "Mistral-Small-24B",
            "mixtral-8x7b",
            "codestral",
            "Codestral",
            "pixtral-large",
            "Pixtral-Large",
        ] {
            assert_eq!(
                model_to_encoding(model),
                Some("mistral_v3"),
                "wrong encoding for {model}"
            );
        }
    }

    #[test]
    fn test_model_to_encoding_unknown() {
        assert_eq!(model_to_encoding("unknown-model"), None);
    }

    // vocab_size / num_special_tokens

    #[test]
    fn test_vocab_sizes() {
        let cases: &[(&str, usize)] = &[
            ("cl100k_base", 100256),
            ("o200k_base", 199998),
            ("p50k_base", 50280),
            ("r50k_base", 50256),
            ("llama3", 128000),
            ("deepseek_v3", 128000),
            ("qwen2", 151643),
            ("mistral_v3", 131072),
        ];
        for &(name, expected) in cases {
            let enc = get_encoding(name).unwrap();
            assert_eq!(enc.vocab_size(), expected, "vocab_size mismatch for {name}");
        }
    }

    #[test]
    fn test_special_token_counts() {
        let enc = get_encoding("cl100k_base").unwrap();
        assert_eq!(enc.num_special_tokens(), 5);

        let enc = get_encoding("p50k_edit").unwrap();
        assert_eq!(enc.num_special_tokens(), 4); // endoftext + 3 fim tokens

        let enc = get_encoding("llama3").unwrap();
        assert_eq!(enc.num_special_tokens(), 8);
    }

    // regression: gpt-4o must resolve to o200k, not cl100k (prefix order matters)
    #[test]
    fn test_model_to_encoding_gpt4o_vs_gpt4() {
        assert_eq!(model_to_encoding("gpt-4o"), Some("o200k_base"));
        assert_eq!(model_to_encoding("gpt-4o-mini"), Some("o200k_base"));
        assert_eq!(model_to_encoding("gpt-4"), Some("cl100k_base"));
        assert_eq!(model_to_encoding("gpt-4-turbo"), Some("cl100k_base"));
    }

    // new encoding edge cases

    #[test]
    fn test_llama3_special_tokens() {
        let enc = get_encoding("llama3").unwrap();
        let text = "hello<|begin_of_text|>world";
        let with = enc.encode_with_special_tokens(text);
        assert!(with.contains(&128000));
        let without = enc.encode(text);
        assert!(!without.contains(&128000));
    }

    #[test]
    fn test_deepseek_special_tokens() {
        let enc = get_encoding("deepseek_v3").unwrap();
        let text = "hello<|EOT|>world";
        let with = enc.encode_with_special_tokens(text);
        assert!(with.contains(&128805));
    }

    #[test]
    fn test_qwen2_special_tokens() {
        let enc = get_encoding("qwen2").unwrap();
        let text = "hello<|endoftext|>world";
        let with = enc.encode_with_special_tokens(text);
        assert!(with.contains(&151643));
    }

    #[test]
    fn test_mistral_special_tokens() {
        let enc = get_encoding("mistral_v3").unwrap();
        let text = "hello[INST]world";
        let with = enc.encode_with_special_tokens(text);
        assert!(with.contains(&3));
    }

    #[test]
    fn test_deepseek_zwj_roundtrip() {
        let enc = get_encoding("deepseek_v3").unwrap();
        // ZWJ emoji sequence
        let text = "\u{200d}\u{200d}test";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_all_encodings_empty_roundtrip() {
        for name in [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "llama3",
            "deepseek_v3",
            "qwen2",
            "mistral_v3",
        ] {
            let enc = get_encoding(name).unwrap();
            assert!(enc.encode("").is_empty(), "non-empty for {name}");
            assert_eq!(enc.count(""), 0, "non-zero count for {name}");
            assert!(enc.decode(&[]).is_empty(), "non-empty decode for {name}");
        }
    }

    #[test]
    fn test_all_encodings_single_byte_roundtrip() {
        for name in [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "r50k_base",
            "llama3",
            "deepseek_v3",
            "qwen2",
            "mistral_v3",
        ] {
            let enc = get_encoding(name).unwrap();
            for b in 0u8..=127 {
                let text = String::from(b as char);
                let decoded = enc.decode(&enc.encode(&text));
                assert_eq!(
                    &decoded[..],
                    text.as_bytes(),
                    "byte {b} roundtrip failed for {name}"
                );
            }
        }
    }

    #[test]
    fn test_count_with_special_tokens_equals_encode_with_special_tokens() {
        for name in ["cl100k_base", "o200k_base", "llama3", "qwen2"] {
            let enc = get_encoding(name).unwrap();
            let text = "hello world test text";
            assert_eq!(
                enc.count_with_special_tokens(text),
                enc.encode_with_special_tokens(text).len(),
                "mismatch for {name}"
            );
        }
    }

    // count_with_special_tokens across all encodings with their specific tokens

    #[test]
    fn test_count_with_special_tokens_all_encodings() {
        let cases: &[(&str, &str)] = &[
            ("cl100k_base", "<|endoftext|>"),
            ("o200k_base", "<|endoftext|>"),
            ("p50k_edit", "<|fim_prefix|>"),
            ("llama3", "<|begin_of_text|>"),
            ("deepseek_v3", "<|EOT|>"),
            ("qwen2", "<|endoftext|>"),
            ("mistral_v3", "[INST]"),
        ];
        for &(name, special) in cases {
            let enc = get_encoding(name).unwrap();
            let text = format!("hello{special}world");
            assert_eq!(
                enc.count_with_special_tokens(&text),
                enc.encode_with_special_tokens(&text).len(),
                "count_with_special_tokens mismatch for {name}"
            );
        }
    }

    #[test]
    fn test_count_with_special_tokens_adjacent() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "<|endoftext|><|endoftext|>";
        assert_eq!(
            enc.count_with_special_tokens(text),
            enc.encode_with_special_tokens(text).len()
        );
    }

    // special token roundtrips for new encodings

    #[test]
    fn test_llama3_special_token_roundtrip() {
        let enc = get_encoding("llama3").unwrap();
        let text = "start<|begin_of_text|>middle<|eot_id|>end";
        let tokens = enc.encode_with_special_tokens(text);
        assert_eq!(enc.decode_to_string(&tokens).unwrap(), text);
    }

    #[test]
    fn test_qwen2_special_token_roundtrip() {
        let enc = get_encoding("qwen2").unwrap();
        let text = "<|im_start|>user\nhello<|im_end|>";
        let tokens = enc.encode_with_special_tokens(text);
        assert_eq!(enc.decode_to_string(&tokens).unwrap(), text);
    }

    #[test]
    fn test_mistral_special_token_roundtrip() {
        let enc = get_encoding("mistral_v3").unwrap();
        let text = "[INST]hello[/INST]";
        let tokens = enc.encode_with_special_tokens(text);
        assert_eq!(enc.decode_to_string(&tokens).unwrap(), text);
    }

    // decode unknown token id: should silently skip
    #[test]
    fn test_decode_unknown_token_id() {
        let enc = get_encoding("cl100k_base").unwrap();
        let result = enc.decode(&[u32::MAX]);
        assert!(
            result.is_empty(),
            "unknown token should be silently skipped"
        );
    }

    #[test]
    fn test_decode_mixed_known_and_unknown() {
        let enc = get_encoding("cl100k_base").unwrap();
        let tokens = enc.encode("hello");
        let mut with_unknown = tokens.clone();
        with_unknown.push(u32::MAX);
        with_unknown.extend_from_slice(&enc.encode(" world"));
        let decoded = enc.decode_to_string(&with_unknown).unwrap();
        assert_eq!(decoded, "hello world");
    }

    // p50k_edit roundtrip (uses different special tokens from p50k_base)

    #[test]
    fn test_p50k_edit_roundtrip() {
        let enc = get_encoding("p50k_edit").unwrap();
        let text = "hello world, p50k_edit encoding";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_p50k_edit_special_tokens() {
        let enc = get_encoding("p50k_edit").unwrap();
        let text = "prefix<|fim_prefix|>middle<|fim_middle|>suffix<|fim_suffix|>end";
        let tokens = enc.encode_with_special_tokens(text);
        assert!(tokens.contains(&50281)); // <|fim_prefix|>
        assert!(tokens.contains(&50282)); // <|fim_middle|>
        assert!(tokens.contains(&50283)); // <|fim_suffix|>
    }

    // o200k special tokens

    #[test]
    fn test_o200k_special_tokens() {
        let enc = get_encoding("o200k_base").unwrap();
        let text = "hello<|endoftext|>world";
        let with = enc.encode_with_special_tokens(text);
        assert!(with.contains(&199999)); // o200k endoftext id
        let without = enc.encode(text);
        assert!(!without.contains(&199999));
    }

    // decode special tokens

    #[test]
    fn test_decode_special_token_cl100k() {
        let enc = get_encoding("cl100k_base").unwrap();
        let decoded = enc.decode(&[100257]); // <|endoftext|>
        assert_eq!(&decoded, b"<|endoftext|>");
    }

    #[test]
    fn test_decode_special_token_roundtrip() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "hello<|endoftext|>world";
        let tokens = enc.encode_with_special_tokens(text);
        let decoded = enc.decode_to_string(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    // new encoding roundtrips

    #[test]
    fn test_llama3_roundtrip() {
        let enc = get_encoding("llama3").unwrap();
        let text = "Hello, 世界! 🚀 test";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_deepseek_roundtrip() {
        let enc = get_encoding("deepseek_v3").unwrap();
        let text = "Hello, 世界! 🚀 test";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_qwen2_roundtrip() {
        let enc = get_encoding("qwen2").unwrap();
        let text = "Hello, 世界! 🚀 test";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_mistral_roundtrip() {
        let enc = get_encoding("mistral_v3").unwrap();
        let text = "Hello, 世界! 🚀 test";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    // count consistency across all encodings

    #[test]
    fn test_count_consistency_all_encodings() {
        let text = "Hello, 世界! This is a test with mixed content 🚀 and numbers 12345.";
        for name in [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "llama3",
            "deepseek_v3",
            "qwen2",
            "mistral_v3",
        ] {
            let enc = get_encoding(name).unwrap();
            assert_eq!(
                enc.count(text),
                enc.encode(text).len(),
                "count != encode().len() for {name}"
            );
        }
    }
}
