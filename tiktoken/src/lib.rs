//! High-performance pure-Rust BPE tokenizer compatible with OpenAI's tiktoken.
//!
//! Supports all tiktoken encodings (`cl100k_base`, `o200k_base`, `p50k_base`, `p50k_edit`,
//! `r50k_base`) with token encoding, decoding, counting, and multi-provider pricing.
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
mod encoding;
pub mod pricing;

pub use bpe::CoreBpe;

use std::sync::OnceLock;

static CL100K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static O200K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static P50K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static P50K_EDIT: OnceLock<CoreBpe> = OnceLock::new();
static R50K_BASE: OnceLock<CoreBpe> = OnceLock::new();

/// Get a cached tokenizer by encoding name.
///
/// Supported: `cl100k_base`, `o200k_base`, `p50k_base`, `p50k_edit`, `r50k_base`.
pub fn get_encoding(name: &str) -> Option<&'static CoreBpe> {
    match name {
        "cl100k_base" => Some(CL100K_BASE.get_or_init(encoding::cl100k_base)),
        "o200k_base" => Some(O200K_BASE.get_or_init(encoding::o200k_base)),
        "p50k_base" => Some(P50K_BASE.get_or_init(encoding::p50k_base)),
        "p50k_edit" => Some(P50K_EDIT.get_or_init(encoding::p50k_edit)),
        "r50k_base" => Some(R50K_BASE.get_or_init(encoding::r50k_base)),
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
/// or `None` for unknown models.
pub fn model_to_encoding(model: &str) -> Option<&'static str> {
    // o200k_base models (newest first)
    if model.starts_with("o4-mini")
        || model.starts_with("o3")
        || model.starts_with("o1")
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
        for model in ["gpt-4o", "o1", "o3", "o4-mini", "chatgpt-4o"] {
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
    fn test_model_to_encoding_unknown() {
        assert_eq!(model_to_encoding("unknown-model"), None);
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
