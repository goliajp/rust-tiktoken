//! Encoding definitions and data parsing for tiktoken-compatible BPE vocabularies.
//!
//! Each encoding consists of:
//! - A `.tiktoken` data file (base64-encoded token → rank mapping, embedded at compile time)
//! - A regex pattern that splits input text into pieces before BPE processing
//! - A set of special tokens (e.g. `<|endoftext|>`) with designated token ids
//!
//! Pattern source: <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>

use base64::Engine;
use rustc_hash::FxHashMap;

use crate::bpe::CoreBpe;

// embedded encoding data files — parsed once on first use, then cached via OnceLock in lib.rs
const CL100K_BASE_DATA: &[u8] = include_bytes!("encodings/cl100k_base.tiktoken");
const O200K_BASE_DATA: &[u8] = include_bytes!("encodings/o200k_base.tiktoken");
const P50K_BASE_DATA: &[u8] = include_bytes!("encodings/p50k_base.tiktoken");
const R50K_BASE_DATA: &[u8] = include_bytes!("encodings/r50k_base.tiktoken");

// cl100k pattern: handles English contractions, Unicode letters/numbers, punctuation, whitespace.
// original tiktoken uses `\s+(?!\S)|\s+` but we use plain `\s+` and emulate the negative
// lookahead in bpe.rs::adjust_whitespace_end — this lets us use the `regex` crate's DFA engine
// instead of a slower backtracking engine like fancy-regex or pcre2.
const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

// o200k pattern: similar to cl100k but with finer Unicode category distinctions
// (Lu/Lt/Lm/Lo/M vs plain \p{L}), supporting better CamelCase and mixed-script splitting
const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// p50k/r50k pattern: simpler, older pattern used by GPT-3 era models
const P50K_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// Parse a `.tiktoken` file (base64-encoded lines of `<token_b64> <rank>`) into a rank map.
///
/// Each line is: `<base64-encoded token bytes> <integer rank>`
/// The rank determines merge priority in the BPE algorithm (lower = merged first).
fn parse_tiktoken_data(data: &[u8]) -> FxHashMap<Vec<u8>, u32> {
    let engine = base64::engine::general_purpose::STANDARD;
    let content = std::str::from_utf8(data).expect("tiktoken data must be valid UTF-8");

    let mut ranks = FxHashMap::default();
    ranks.reserve(data.len() / 20); // rough estimate: ~20 bytes per line

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let token_b64 = parts.next().expect("missing token");
        let rank_str = parts.next().expect("missing rank");
        let token_bytes = engine
            .decode(token_b64)
            .expect("invalid base64 in tiktoken data");
        let rank: u32 = rank_str.parse().expect("invalid rank in tiktoken data");
        ranks.insert(token_bytes, rank);
    }

    ranks
}

/// Build a special token map from `(text, id)` pairs.
fn special_tokens(pairs: &[(&str, u32)]) -> FxHashMap<Vec<u8>, u32> {
    pairs
        .iter()
        .map(|&(s, v)| (s.as_bytes().to_vec(), v))
        .collect()
}

/// Construct the cl100k_base encoding (GPT-4, GPT-3.5 Turbo, embeddings).
/// Vocabulary size: 100,256 regular tokens + 5 special tokens.
pub fn cl100k_base() -> CoreBpe {
    let encoder = parse_tiktoken_data(CL100K_BASE_DATA);
    let special = special_tokens(&[
        ("<|endoftext|>", 100257),
        ("<|fim_prefix|>", 100258),
        ("<|fim_middle|>", 100259),
        ("<|fim_suffix|>", 100260),
        ("<|endofprompt|>", 100276),
    ]);
    CoreBpe::new(encoder, special, CL100K_PATTERN)
}

/// Construct the p50k_base encoding (text-davinci-002/003, code-davinci, code-cushman).
/// Vocabulary size: 50,256 regular tokens + 1 special token.
pub fn p50k_base() -> CoreBpe {
    let encoder = parse_tiktoken_data(P50K_BASE_DATA);
    let special = special_tokens(&[("<|endoftext|>", 50256)]);
    CoreBpe::new(encoder, special, P50K_PATTERN)
}

/// Construct the p50k_edit encoding (text-davinci-edit, code-davinci-edit).
/// Same merge ranks as p50k_base but with additional FIM (fill-in-middle) special tokens.
pub fn p50k_edit() -> CoreBpe {
    let encoder = parse_tiktoken_data(P50K_BASE_DATA);
    let special = special_tokens(&[
        ("<|endoftext|>", 50256),
        ("<|fim_prefix|>", 50281),
        ("<|fim_middle|>", 50282),
        ("<|fim_suffix|>", 50283),
    ]);
    CoreBpe::new(encoder, special, P50K_PATTERN)
}

/// Construct the o200k_base encoding (GPT-4o, o1, o3, o4-mini).
/// Vocabulary size: 199,998 regular tokens + 2 special tokens.
pub fn o200k_base() -> CoreBpe {
    let encoder = parse_tiktoken_data(O200K_BASE_DATA);
    let special = special_tokens(&[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)]);
    CoreBpe::new(encoder, special, O200K_PATTERN)
}

/// Construct the r50k_base encoding (GPT-3 era: davinci, curie, babbage, ada).
/// Vocabulary size: 50,256 regular tokens + 1 special token.
/// Uses the same merge ranks and regex pattern as p50k_base.
pub fn r50k_base() -> CoreBpe {
    let encoder = parse_tiktoken_data(R50K_BASE_DATA);
    let special = special_tokens(&[("<|endoftext|>", 50256)]);
    CoreBpe::new(encoder, special, P50K_PATTERN)
}
