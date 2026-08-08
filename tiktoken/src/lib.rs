//! High-performance pure-Rust BPE tokenizer compatible with OpenAI's tiktoken
//! and all mainstream LLM tokenizers.
//!
//! Supports 17 encodings across 8 providers: OpenAI (`cl100k_base`, `o200k_base`,
//! `o200k_harmony`, `p50k_base`, `p50k_edit`, `r50k_base`, `gpt2`), Meta (`llama3`),
//! DeepSeek (`deepseek_v3`, `deepseek_v4`), Alibaba (`qwen2`), Mistral
//! (`mistral_v3`), Moonshot (`kimi_k2`, `kimi_k3`), Zhipu (`glm4`, `glm5`),
//! and MiniMax (`minimax_m2`).
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
static O200K_HARMONY: OnceLock<CoreBpe> = OnceLock::new();
static P50K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static P50K_EDIT: OnceLock<CoreBpe> = OnceLock::new();
static R50K_BASE: OnceLock<CoreBpe> = OnceLock::new();
static LLAMA3: OnceLock<CoreBpe> = OnceLock::new();
static DEEPSEEK_V3: OnceLock<CoreBpe> = OnceLock::new();
static QWEN2: OnceLock<CoreBpe> = OnceLock::new();
static MISTRAL_V3: OnceLock<CoreBpe> = OnceLock::new();
static KIMI_K2: OnceLock<CoreBpe> = OnceLock::new();
static KIMI_K3: OnceLock<CoreBpe> = OnceLock::new();
static GLM4: OnceLock<CoreBpe> = OnceLock::new();
static GLM5: OnceLock<CoreBpe> = OnceLock::new();
static MINIMAX_M2: OnceLock<CoreBpe> = OnceLock::new();
static DEEPSEEK_V4: OnceLock<CoreBpe> = OnceLock::new();

/// All available encoding names.
///
/// Returns the list of encoding names that can be passed to [`get_encoding`].
///
/// # Examples
///
/// ```
/// let names = tiktoken::list_encodings();
/// assert!(names.contains(&"cl100k_base"));
/// assert!(names.contains(&"o200k_harmony"));
/// assert!(names.contains(&"gpt2"));
/// assert!(names.contains(&"kimi_k3"));
/// assert!(names.contains(&"glm5"));
/// assert_eq!(names.len(), 17);
/// ```
pub fn list_encodings() -> &'static [&'static str] {
    &[
        "cl100k_base",
        "o200k_base",
        "o200k_harmony",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
        "gpt2",
        "llama3",
        "deepseek_v3",
        "deepseek_v4",
        "qwen2",
        "mistral_v3",
        "kimi_k2",
        "kimi_k3",
        "glm4",
        "glm5",
        "minimax_m2",
    ]
}

/// Get a cached tokenizer by encoding name.
///
/// Supported encodings:
/// - OpenAI: `cl100k_base`, `o200k_base`, `o200k_harmony`, `p50k_base`, `p50k_edit`, `r50k_base`, `gpt2`
/// - Meta: `llama3`
/// - DeepSeek: `deepseek_v3`, `deepseek_v4`
/// - Alibaba: `qwen2`
/// - Mistral: `mistral_v3`
/// - Moonshot: `kimi_k2`, `kimi_k3`
/// - Zhipu: `glm4`, `glm5`
/// - MiniMax: `minimax_m2`
///
/// Note: `gpt2` is a name-level alias for `r50k_base`; both return the same cached instance.
pub fn get_encoding(name: &str) -> Option<&'static CoreBpe> {
    match name {
        "cl100k_base" => Some(CL100K_BASE.get_or_init(encoding::cl100k_base)),
        "o200k_base" => Some(O200K_BASE.get_or_init(encoding::o200k_base)),
        "o200k_harmony" => Some(O200K_HARMONY.get_or_init(encoding::o200k_harmony)),
        "p50k_base" => Some(P50K_BASE.get_or_init(encoding::p50k_base)),
        "p50k_edit" => Some(P50K_EDIT.get_or_init(encoding::p50k_edit)),
        // gpt2 shares r50k_base's cache slot — same vocab, same regex, same special token.
        "r50k_base" | "gpt2" => Some(R50K_BASE.get_or_init(encoding::r50k_base)),
        "llama3" => Some(LLAMA3.get_or_init(encoding::llama3)),
        "deepseek_v3" => Some(DEEPSEEK_V3.get_or_init(encoding::deepseek_v3)),
        "deepseek_v4" => Some(DEEPSEEK_V4.get_or_init(encoding::deepseek_v4)),
        "qwen2" => Some(QWEN2.get_or_init(encoding::qwen2)),
        "mistral_v3" => Some(MISTRAL_V3.get_or_init(encoding::mistral_v3)),
        "kimi_k2" => Some(KIMI_K2.get_or_init(encoding::kimi_k2)),
        "kimi_k3" => Some(KIMI_K3.get_or_init(encoding::kimi_k3)),
        "glm4" => Some(GLM4.get_or_init(encoding::glm4)),
        "glm5" => Some(GLM5.get_or_init(encoding::glm5)),
        "minimax_m2" => Some(MINIMAX_M2.get_or_init(encoding::minimax_m2)),
        _ => None,
    }
}

/// Get a cached tokenizer by model name.
///
/// Supports OpenAI, Meta, DeepSeek, Qwen, Mistral, Moonshot (Kimi), Zhipu (GLM), and MiniMax models.
/// Maps model name prefixes to their encoding.
/// Returns `None` for unknown models.
pub fn encoding_for_model(model: &str) -> Option<&'static CoreBpe> {
    model_to_encoding(model).and_then(get_encoding)
}

/// Map a model name to its encoding name.
///
/// Returns the encoding name (e.g. `"o200k_base"`) for the given model,
/// or `None` for unknown models. Supports OpenAI, Meta, DeepSeek, Qwen, Mistral, Moonshot (Kimi), Zhipu (GLM), and MiniMax models.
pub fn model_to_encoding(model: &str) -> Option<&'static str> {
    // Strip the `ft:` prefix used for fine-tuned model IDs
    // (e.g. `ft:gpt-4o:my-org::abc123` → `gpt-4o:my-org::abc123`). Upstream
    // enumerates a fixed set of `ft:` prefixes; stripping generalizes to any
    // fine-tunable base model, so this is a superset of upstream behavior.
    let model = model.strip_prefix("ft:").unwrap_or(model);

    // Exact matches win over prefixes — several legacy ids would otherwise be
    // captured by a shorter prefix and resolve to the wrong encoding
    // (`davinci-codex` is p50k_base, not the r50k_base that `davinci` implies;
    // `code-davinci-edit-001` is p50k_edit, not the p50k_base of `code-davinci`).
    EXACT_MODEL_ENCODINGS
        .iter()
        .find(|&&(m, _)| m == model)
        .map(|&(_, enc)| enc)
        .or_else(|| {
            MODEL_PREFIX_ENCODINGS
                .iter()
                .find(|&&(p, _)| model.starts_with(p))
                .map(|&(_, enc)| enc)
        })
}

/// Exact model id → encoding. Mirrors upstream `MODEL_TO_ENCODING`, which the
/// `model_map` test pins against the reference registry.
///
/// `gpt-2` / `gpt2` map to `r50k_base`, the name this crate caches them under;
/// upstream calls the identical encoding `gpt2` (see [`get_encoding`]).
const EXACT_MODEL_ENCODINGS: &[(&str, &str)] = &[
    // chat / reasoning
    ("gpt-5", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("gpt-3.5-turbo", "cl100k_base"),
    ("gpt-35-turbo", "cl100k_base"),
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4-mini", "o200k_base"),
    // base / legacy completion
    ("davinci-002", "cl100k_base"),
    ("babbage-002", "cl100k_base"),
    ("davinci", "r50k_base"),
    ("curie", "r50k_base"),
    ("babbage", "r50k_base"),
    ("ada", "r50k_base"),
    ("gpt-2", "r50k_base"),
    ("gpt2", "r50k_base"),
    // codex era — these predate the `code-*` prefixes and disagree with them
    ("davinci-codex", "p50k_base"),
    ("cushman-codex", "p50k_base"),
    ("code-davinci-001", "p50k_base"),
    ("code-davinci-002", "p50k_base"),
    ("code-cushman-001", "p50k_base"),
    ("code-cushman-002", "p50k_base"),
    // edit models (FIM special tokens)
    ("text-davinci-edit-001", "p50k_edit"),
    ("code-davinci-edit-001", "p50k_edit"),
    // instruct-era completion
    ("text-davinci-003", "p50k_base"),
    ("text-davinci-002", "p50k_base"),
    ("text-davinci-001", "r50k_base"),
    ("text-curie-001", "r50k_base"),
    ("text-babbage-001", "r50k_base"),
    ("text-ada-001", "r50k_base"),
    // DeepSeek API aliases: both point at V4 since 2026-07-24
    ("deepseek-chat", "deepseek_v4"),
    ("deepseek-reasoner", "deepseek_v4"),
    // Moonshot API alias for the current flagship
    ("kimi-latest", "kimi_k3"),
    // embeddings
    ("text-embedding-3-small", "cl100k_base"),
    ("text-embedding-3-large", "cl100k_base"),
    ("text-embedding-ada-002", "cl100k_base"),
    // first-generation search / similarity embeddings
    ("text-search-davinci-doc-001", "r50k_base"),
    ("text-search-curie-doc-001", "r50k_base"),
    ("text-search-babbage-doc-001", "r50k_base"),
    ("text-search-ada-doc-001", "r50k_base"),
    ("text-similarity-davinci-001", "r50k_base"),
    ("text-similarity-curie-001", "r50k_base"),
    ("text-similarity-babbage-001", "r50k_base"),
    ("text-similarity-ada-001", "r50k_base"),
    ("code-search-babbage-code-001", "r50k_base"),
    ("code-search-ada-code-001", "r50k_base"),
];

/// Model id prefix → encoding, scanned in order: more specific prefixes first,
/// since `starts_with("gpt-4")` would otherwise swallow `gpt-4o` and `gpt-4.1`.
///
/// Upstream pins dated suffixes (`gpt-5-`, `gpt-4.1-`); this table uses the
/// undated stem (`gpt-5`) so point releases resolve without a code change —
/// `gpt-5.6-sol` and `gpt-5.4-mini` both reach `o200k_base` here.
const MODEL_PREFIX_ENCODINGS: &[(&str, &str)] = &[
    // OpenAI — o200k_harmony must precede o200k_base so `gpt-oss` is not
    // shadowed, and the o200k block must precede the cl100k `gpt-4` entry.
    ("gpt-oss", "o200k_harmony"),
    ("o4-mini", "o200k_base"),
    ("o3", "o200k_base"),
    ("o1", "o200k_base"),
    ("chatgpt-4o", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("gpt-4.5", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("gpt-35-turbo", "cl100k_base"),
    ("davinci-002", "cl100k_base"),
    ("babbage-002", "cl100k_base"),
    ("text-embedding-3", "cl100k_base"),
    ("text-embedding-ada", "cl100k_base"),
    ("text-davinci-003", "p50k_base"),
    ("text-davinci-002", "p50k_base"),
    ("code-davinci", "p50k_base"),
    ("code-cushman", "p50k_base"),
    ("text-davinci-001", "r50k_base"),
    ("text-curie", "r50k_base"),
    ("text-babbage", "r50k_base"),
    ("text-ada", "r50k_base"),
    ("davinci", "r50k_base"),
    ("curie", "r50k_base"),
    ("babbage", "r50k_base"),
    ("ada", "r50k_base"),
    ("gpt-2", "r50k_base"),
    ("gpt2", "r50k_base"),
    // Meta — the llama3 encoding covers Llama 3.x and 4.x
    ("llama-", "llama3"),
    ("llama3", "llama3"),
    ("llama4", "llama3"),
    ("Llama-", "llama3"),
    ("Meta-Llama-", "llama3"),
    // DeepSeek — v4 prefixes must precede the generation catch-alls
    ("deepseek-v4", "deepseek_v4"),
    ("DeepSeek-V4", "deepseek_v4"),
    ("deepseek", "deepseek_v3"),
    ("DeepSeek", "deepseek_v3"),
    // Moonshot — k3 before k2 is not required (distinct prefixes), but keep
    // the newest generation first for readability
    ("kimi-k3", "kimi_k3"),
    ("Kimi-K3", "kimi_k3"),
    ("kimi-k2", "kimi_k2"),
    ("Kimi-K2", "kimi_k2"),
    ("kimi", "kimi_k3"),
    ("Kimi", "kimi_k3"),
    // Zhipu
    ("glm-5", "glm5"),
    ("GLM-5", "glm5"),
    ("glm-4", "glm4"),
    ("GLM-4", "glm4"),
    ("glm", "glm4"),
    ("GLM", "glm4"),
    // MiniMax — the M2 tokenizer covers the whole M2.x family
    ("minimax", "minimax_m2"),
    ("MiniMax", "minimax_m2"),
    // Alibaba
    ("qwen", "qwen2"),
    ("Qwen", "qwen2"),
    // Mistral
    ("mistral", "mistral_v3"),
    ("Mistral", "mistral_v3"),
    ("mixtral", "mistral_v3"),
    ("Mixtral", "mistral_v3"),
    ("codestral", "mistral_v3"),
    ("Codestral", "mistral_v3"),
    ("pixtral", "mistral_v3"),
    ("Pixtral", "mistral_v3"),
];

#[cfg(test)]
mod tests {
    use super::*;

    // encoding lookup

    #[test]
    fn test_get_encoding_known() {
        for name in [
            "cl100k_base",
            "o200k_base",
            "o200k_harmony",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "llama3",
            "deepseek_v3",
            "deepseek_v4",
            "qwen2",
            "mistral_v3",
            "kimi_k2",
            "kimi_k3",
            "glm4",
            "glm5",
            "minimax_m2",
        ] {
            assert!(get_encoding(name).is_some(), "missing encoding: {name}");
        }
    }

    #[test]
    fn test_get_encoding_unknown() {
        assert!(get_encoding("nonexistent").is_none());
    }

    #[test]
    fn test_o200k_harmony_roundtrip() {
        let enc = get_encoding("o200k_harmony").unwrap();
        let text = "hello world, 你好世界 🚀";
        let decoded = enc.decode(&enc.encode(text));
        assert_eq!(std::str::from_utf8(&decoded).unwrap(), text);
    }

    #[test]
    fn test_o200k_harmony_matches_base_for_plain_text() {
        // Same merge ranks + regex → ordinary text encodes identically.
        let base = get_encoding("o200k_base").unwrap();
        let harmony = get_encoding("o200k_harmony").unwrap();
        for text in ["hello world", "the quick brown fox", "你好世界 🚀"] {
            assert_eq!(base.encode(text), harmony.encode(text), "{text}");
        }
    }

    #[test]
    fn test_encoding_for_gpt_oss() {
        assert_eq!(model_to_encoding("gpt-oss-20b"), Some("o200k_harmony"));
        assert_eq!(model_to_encoding("gpt-oss-120b"), Some("o200k_harmony"));
        assert_ne!(model_to_encoding("gpt-oss-20b"), Some("o200k_base"));
    }

    #[test]
    fn test_gpt2_in_registry() {
        assert!(get_encoding("gpt2").is_some());
        assert!(list_encodings().contains(&"gpt2"));
    }

    #[test]
    fn test_gpt2_shares_r50k_base_instance() {
        // gpt2 and r50k_base hit the same OnceLock slot, so the returned references
        // point to the exact same CoreBpe instance.
        let a = get_encoding("gpt2").unwrap() as *const _;
        let b = get_encoding("r50k_base").unwrap() as *const _;
        assert_eq!(a, b);
    }

    #[test]
    fn test_gpt2_encodes_like_r50k_base() {
        let g = get_encoding("gpt2").unwrap();
        let r = get_encoding("r50k_base").unwrap();
        for text in [
            "hello world",
            "the quick brown fox",
            "let me think about this",
        ] {
            assert_eq!(g.encode(text), r.encode(text), "{text}");
        }
    }

    #[test]
    fn test_encoding_for_gpt2_models() {
        assert_eq!(model_to_encoding("gpt2"), Some("r50k_base"));
        assert_eq!(model_to_encoding("gpt-2"), Some("r50k_base"));
        assert!(encoding_for_model("gpt2").is_some());
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
    fn test_encoding_for_gpt5_family() {
        for m in ["gpt-5", "gpt-5-turbo", "gpt-4.5", "gpt-4.5-preview"] {
            assert_eq!(model_to_encoding(m), Some("o200k_base"), "{m}");
        }
    }

    #[test]
    fn test_encoding_for_gpt5_point_releases() {
        // The 2026 GPT-5.x point releases and the Sol/Terra/Luna tier names must
        // resolve without a per-SKU entry.
        for m in [
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.4-pro",
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.5-cyber",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.3-codex",
            "gpt-5.3-chat-latest",
        ] {
            assert_eq!(model_to_encoding(m), Some("o200k_base"), "{m}");
        }
    }

    #[test]
    fn test_encoding_for_chinese_model_families() {
        // Moonshot: K2 and K3 share a vocabulary but have distinct special
        // tables; kimi-latest is the API alias for the current flagship.
        assert_eq!(model_to_encoding("kimi-k3"), Some("kimi_k3"));
        assert_eq!(model_to_encoding("kimi-k2-0711-preview"), Some("kimi_k2"));
        assert_eq!(model_to_encoding("kimi-k2.6"), Some("kimi_k2"));
        assert_eq!(model_to_encoding("kimi-latest"), Some("kimi_k3"));
        assert_eq!(model_to_encoding("Kimi-K3"), Some("kimi_k3"));
        // Zhipu: generation split — 4.x and 5.x are independently trained vocabularies.
        assert_eq!(model_to_encoding("glm-5.2"), Some("glm5"));
        assert_eq!(model_to_encoding("glm-5"), Some("glm5"));
        assert_eq!(model_to_encoding("glm-4.7"), Some("glm4"));
        assert_eq!(model_to_encoding("glm-4.5-air"), Some("glm4"));
        assert_eq!(model_to_encoding("GLM-4.6"), Some("glm4"));
        // MiniMax: one tokenizer across the whole M2 family.
        assert_eq!(model_to_encoding("minimax-m2.7"), Some("minimax_m2"));
        assert_eq!(model_to_encoding("MiniMax-M2.1"), Some("minimax_m2"));
        // DeepSeek: v4 prefix wins over the generation catch-all; the API
        // aliases point at V4 since 2026-07-24.
        assert_eq!(model_to_encoding("deepseek-v4-flash"), Some("deepseek_v4"));
        assert_eq!(model_to_encoding("DeepSeek-V4-Pro"), Some("deepseek_v4"));
        assert_eq!(model_to_encoding("deepseek-chat"), Some("deepseek_v4"));
        assert_eq!(model_to_encoding("deepseek-reasoner"), Some("deepseek_v4"));
        assert_eq!(model_to_encoding("deepseek-v3"), Some("deepseek_v3"));
        assert_eq!(model_to_encoding("deepseek-r1"), Some("deepseek_v3"));
    }

    #[test]
    fn test_kimi_generations_share_vocab_but_not_specials() {
        let k2 = get_encoding("kimi_k2").unwrap();
        let k3 = get_encoding("kimi_k3").unwrap();
        // plain text encodes identically (shared merges)
        for text in ["hello world", "你好世界", "let x = 42;"] {
            assert_eq!(k2.encode(text), k3.encode(text), "{text}");
        }
        // id 163586 means <|im_end|> in K2 but <|end_of_msg|> in K3
        assert_eq!(k2.encode_with_special_tokens("<|im_end|>"), vec![163586]);
        assert_eq!(
            k3.encode_with_special_tokens("<|end_of_msg|>"),
            vec![163586]
        );
    }

    #[test]
    fn test_deepseek_v4_extends_v3_specials() {
        let v4 = get_encoding("deepseek_v4").unwrap();
        // inherited from V3
        assert_eq!(v4.encode_with_special_tokens("<｜User｜>"), vec![128803]);
        // V4-only
        assert_eq!(v4.encode_with_special_tokens("<think>"), vec![128821]);
        assert_eq!(
            v4.encode_with_special_tokens("<|place_holder_mm_span_0021|>"),
            vec![128847]
        );
        // V3 does not know the V4-only ids
        let v3 = get_encoding("deepseek_v3").unwrap();
        assert_ne!(v3.encode_with_special_tokens("<think>"), vec![128821]);
    }

    #[test]
    fn test_encoding_for_legacy_exact_ids() {
        // Regression: a bare prefix scan routed these to the wrong encoding —
        // `davinci-codex` matched `davinci` (r50k_base) and `code-davinci-edit-001`
        // matched `code-davinci` (p50k_base). Exact ids must win over prefixes.
        assert_eq!(model_to_encoding("davinci-codex"), Some("p50k_base"));
        assert_eq!(model_to_encoding("cushman-codex"), Some("p50k_base"));
        assert_eq!(
            model_to_encoding("text-davinci-edit-001"),
            Some("p50k_edit")
        );
        assert_eq!(
            model_to_encoding("code-davinci-edit-001"),
            Some("p50k_edit")
        );
    }

    #[test]
    fn test_encoding_for_first_gen_embedding_models() {
        // Regression: these returned None — no prefix in the old scan covered the
        // `text-search-*` / `text-similarity-*` / `code-search-*` families.
        for m in [
            "text-search-davinci-doc-001",
            "text-search-curie-doc-001",
            "text-search-babbage-doc-001",
            "text-search-ada-doc-001",
            "text-similarity-davinci-001",
            "text-similarity-curie-001",
            "text-similarity-babbage-001",
            "text-similarity-ada-001",
            "code-search-babbage-code-001",
            "code-search-ada-code-001",
        ] {
            assert_eq!(model_to_encoding(m), Some("r50k_base"), "{m}");
        }
    }

    #[test]
    fn test_encoding_for_davinci_babbage_002() {
        // Regression: these were incorrectly routed to r50k_base by
        // starts_with("davinci")/("babbage"). They use cl100k_base upstream.
        assert_eq!(model_to_encoding("davinci-002"), Some("cl100k_base"));
        assert_eq!(model_to_encoding("babbage-002"), Some("cl100k_base"));
    }

    #[test]
    fn test_encoding_for_finetuned_models() {
        assert_eq!(
            model_to_encoding("ft:gpt-4o:my-org::abc123"),
            Some("o200k_base")
        );
        assert_eq!(model_to_encoding("ft:gpt-4:org::xyz"), Some("cl100k_base"));
        assert_eq!(
            model_to_encoding("ft:gpt-3.5-turbo:org::xyz"),
            Some("cl100k_base")
        );
    }

    #[test]
    fn test_encoding_for_azure_gpt35() {
        // Azure uses `gpt-35-turbo` instead of `gpt-3.5-turbo`.
        assert_eq!(model_to_encoding("gpt-35-turbo"), Some("cl100k_base"));
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
