//! Encoding definitions and data parsing for tiktoken-compatible BPE vocabularies.
//!
//! Each encoding consists of:
//! - A `.tiktoken.zst` data file (zstd-compressed, base64-encoded token → rank lines, embedded at compile time)
//! - A regex pattern that splits input text into pieces before BPE processing
//! - A set of special tokens (e.g. `<|endoftext|>`) with designated token ids
//!
//! Pattern source: <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>

use base64::Engine;
use rustc_hash::FxHashMap;

use crate::bpe::CoreBpe;

// embedded encoding data files — zstd-compressed, decompressed on first use via OnceLock in lib.rs
const CL100K_BASE_DATA: &[u8] = include_bytes!("encodings/cl100k_base.tiktoken.zst");
const O200K_BASE_DATA: &[u8] = include_bytes!("encodings/o200k_base.tiktoken.zst");
const P50K_BASE_DATA: &[u8] = include_bytes!("encodings/p50k_base.tiktoken.zst");
const R50K_BASE_DATA: &[u8] = include_bytes!("encodings/r50k_base.tiktoken.zst");
const LLAMA3_DATA: &[u8] = include_bytes!("encodings/llama3.tiktoken.zst");
const DEEPSEEK_V3_DATA: &[u8] = include_bytes!("encodings/deepseek_v3.tiktoken.zst");
const QWEN2_DATA: &[u8] = include_bytes!("encodings/qwen2.tiktoken.zst");
const MISTRAL_V3_DATA: &[u8] = include_bytes!("encodings/mistral_v3.tiktoken.zst");

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

// llama3 pattern: same structure as cl100k (contractions, letters, numbers, punctuation, whitespace)
// original uses `\s+(?!\S)|\s+` — we emulate the lookahead in pretokenize.rs
const LLAMA3_PATTERN: &str = CL100K_PATTERN;

// deepseek v3 pattern: 3 sequential splits combined into one alternation.
// priority: numbers (1-3 digits) > CJK/Japanese > general pattern
// final catch-all `[\s\S]` ensures format chars (ZWJ etc.) are not skipped,
// matching HF's Split/Isolated behavior where non-matching text is kept.
const DEEPSEEK_V3_PATTERN: &str = concat!(
    r"\p{N}{1,3}",
    r"|[一-龥\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+",
    r"|[!-/:-@\[-`{-~][A-Za-z]+",
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+",
    r"| ?[\p{P}\p{S}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+",
    r"|[\s\S]",
);

// qwen2 pattern: similar to cl100k but \p{N} matches single digits (not 1-3)
const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

// mistral v3 (tekken) pattern: same as cl100k
const MISTRAL_V3_PATTERN: &str = CL100K_PATTERN;

/// Parse a zstd-compressed `.tiktoken` file into a rank map.
///
/// The compressed data is first decompressed, then parsed line by line.
/// Each line is: `<base64-encoded token bytes> <integer rank>`
/// The rank determines merge priority in the BPE algorithm (lower = merged first).
pub(crate) fn parse_tiktoken_data(compressed: &[u8]) -> FxHashMap<Vec<u8>, u32> {
    let mut decoder =
        ruzstd::decoding::StreamingDecoder::new(compressed).expect("zstd decompression failed");
    let mut data = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut data).expect("zstd decompression failed");
    parse_tiktoken_lines(&data)
}

/// Parse raw (uncompressed) `.tiktoken` lines into a rank map.
fn parse_tiktoken_lines(data: &[u8]) -> FxHashMap<Vec<u8>, u32> {
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

/// Construct the llama3 encoding (Llama 3 / 3.1 / 3.2 / 3.3).
/// Vocabulary size: 128,000 regular tokens + 256 special tokens.
pub fn llama3() -> CoreBpe {
    let encoder = parse_tiktoken_data(LLAMA3_DATA);
    let special = special_tokens(&[
        ("<|begin_of_text|>", 128000),
        ("<|end_of_text|>", 128001),
        ("<|finetune_right_pad_id|>", 128004),
        ("<|start_header_id|>", 128006),
        ("<|end_header_id|>", 128007),
        ("<|eom_id|>", 128008),
        ("<|eot_id|>", 128009),
        ("<|python_tag|>", 128010),
    ]);
    CoreBpe::new(encoder, special, LLAMA3_PATTERN)
}

/// Construct the deepseek_v3 encoding (DeepSeek V3, R1).
/// Vocabulary size: 128,000 regular tokens + 804 special tokens.
pub fn deepseek_v3() -> CoreBpe {
    let encoder = parse_tiktoken_data(DEEPSEEK_V3_DATA);
    let special = special_tokens(&[
        ("<｜begin▁of▁sentence｜>", 0),
        ("<｜end▁of▁sentence｜>", 1),
        ("<｜▁pad▁｜>", 2),
        ("<|EOT|>", 128805),
    ]);
    CoreBpe::new(encoder, special, DEEPSEEK_V3_PATTERN)
}

/// Construct the qwen2 encoding (Qwen 2.5 / 3).
/// Vocabulary size: 151,643 regular tokens + 14 special tokens.
pub fn qwen2() -> CoreBpe {
    let encoder = parse_tiktoken_data(QWEN2_DATA);
    let special = special_tokens(&[
        ("<|endoftext|>", 151643),
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
        ("<|object_ref_start|>", 151646),
        ("<|object_ref_end|>", 151647),
        ("<|box_start|>", 151648),
        ("<|box_end|>", 151649),
        ("<|quad_start|>", 151650),
        ("<|quad_end|>", 151651),
        ("<|vision_start|>", 151652),
        ("<|vision_end|>", 151653),
        ("<|vision_pad|>", 151654),
        ("<|image_pad|>", 151655),
        ("<|video_pad|>", 151656),
    ]);
    CoreBpe::new(encoder, special, QWEN2_PATTERN)
}

/// Construct the mistral_v3 encoding (Mistral, Mixtral with Tekken tokenizer).
/// Vocabulary size: 131,072 regular tokens + 1000 special tokens.
pub fn mistral_v3() -> CoreBpe {
    let encoder = parse_tiktoken_data(MISTRAL_V3_DATA);
    let special = special_tokens(&[
        ("<unk>", 0),
        ("<s>", 1),
        ("</s>", 2),
        ("[INST]", 3),
        ("[/INST]", 4),
        ("[AVAILABLE_TOOLS]", 5),
        ("[/AVAILABLE_TOOLS]", 6),
        ("[TOOL_RESULTS]", 7),
        ("[/TOOL_RESULTS]", 8),
        ("[TOOL_CALLS]", 9),
        ("[IMG]", 10),
        ("[IMG_BREAK]", 12),
        ("[IMG_END]", 13),
        ("[PREFIX]", 14),
        ("[MIDDLE]", 15),
        ("[SUFFIX]", 16),
    ]);
    CoreBpe::new(encoder, special, MISTRAL_V3_PATTERN)
}

/// Expose cl100k rank map for internal tests (e.g. Vocab equivalence)
#[cfg(test)]
pub(crate) fn parse_tiktoken_data_for_test() -> FxHashMap<Vec<u8>, u32> {
    parse_tiktoken_data(CL100K_BASE_DATA)
}
