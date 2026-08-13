//! Encoding definitions and data parsing for tiktoken-compatible BPE vocabularies.
//!
//! Each encoding consists of:
//! - A `.tkv.zst` data file (see [`parse_tkv`] for the format), embedded at compile time
//! - A regex pattern that splits input text into pieces before BPE processing
//! - A set of special tokens (e.g. `<|endoftext|>`) with designated token ids
//!
//! Pattern source: <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>

use rustc_hash::FxHashMap;

use crate::bpe::CoreBpe;
use crate::pretokenize::{FastPath, WhitespaceRules};

// Embedded vocabulary frames, decompressed on first use via the OnceLocks in
// lib.rs. Built from the `tests/vocab-oracle/*.tiktoken.zst` reference files by
// `src/encodings/build_tkv.py`; the `vocab_oracle` tests below diff the two.
const CL100K_BASE_TKV: &[u8] = include_bytes!("encodings/cl100k_base.tkv.zst");
const O200K_BASE_TKV: &[u8] = include_bytes!("encodings/o200k_base.tkv.zst");
const R50K_BASE_TKV: &[u8] = include_bytes!("encodings/r50k_base.tkv.zst");
const DEEPSEEK_V3_TKV: &[u8] = include_bytes!("encodings/deepseek_v3.tkv.zst");
const QWEN2_TKV: &[u8] = include_bytes!("encodings/qwen2.tkv.zst");
const MISTRAL_V3_TKV: &[u8] = include_bytes!("encodings/mistral_v3.tkv.zst");
const KIMI_K2_TKV: &[u8] = include_bytes!("encodings/kimi_k2.tkv.zst");
const GLM4_TKV: &[u8] = include_bytes!("encodings/glm4.tkv.zst");
const MINIMAX_M2_TKV: &[u8] = include_bytes!("encodings/minimax_m2.tkv.zst");

// Three vocabularies are exact rank-aligned extensions of another one: every
// token at rank `i` of the base is the token at rank `i` of the derived
// vocabulary. Their files hold only the tail, so the shared prefix is stored
// once. Reconstructing them costs nothing extra at run time — those leading
// entries have to be built either way.
const LLAMA3_TAIL_TKV: &[u8] = include_bytes!("encodings/llama3.tkv.zst");
const GLM5_TAIL_TKV: &[u8] = include_bytes!("encodings/glm5.tkv.zst");
const P50K_BASE_TAIL_TKV: &[u8] = include_bytes!("encodings/p50k_base.tkv.zst");

pub(crate) const CL100K_BASE_CHAIN: &[&[u8]] = &[CL100K_BASE_TKV];
pub(crate) const O200K_BASE_CHAIN: &[&[u8]] = &[O200K_BASE_TKV];
pub(crate) const R50K_BASE_CHAIN: &[&[u8]] = &[R50K_BASE_TKV];
pub(crate) const DEEPSEEK_V3_CHAIN: &[&[u8]] = &[DEEPSEEK_V3_TKV];
pub(crate) const QWEN2_CHAIN: &[&[u8]] = &[QWEN2_TKV];
pub(crate) const MISTRAL_V3_CHAIN: &[&[u8]] = &[MISTRAL_V3_TKV];
pub(crate) const KIMI_K2_CHAIN: &[&[u8]] = &[KIMI_K2_TKV];
pub(crate) const GLM4_CHAIN: &[&[u8]] = &[GLM4_TKV];
pub(crate) const MINIMAX_M2_CHAIN: &[&[u8]] = &[MINIMAX_M2_TKV];
pub(crate) const LLAMA3_CHAIN: &[&[u8]] = &[CL100K_BASE_TKV, LLAMA3_TAIL_TKV];
pub(crate) const GLM5_CHAIN: &[&[u8]] = &[GLM4_TKV, GLM5_TAIL_TKV];
pub(crate) const P50K_BASE_CHAIN: &[&[u8]] = &[R50K_BASE_TKV, P50K_BASE_TAIL_TKV];

// cl100k pattern: handles English contractions, Unicode letters/numbers, punctuation, whitespace.
// original tiktoken uses `\s+(?!\S)|\s+` but we use plain `\s+` and emulate the negative
// lookahead in bpe.rs::adjust_whitespace_end — this lets us use the `regex` crate's DFA engine
// instead of a slower backtracking engine like fancy-regex or pcre2.
pub(crate) const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

// o200k pattern: similar to cl100k but with finer Unicode category distinctions
// (Lu/Lt/Lm/Lo/M vs plain \p{L}), supporting better CamelCase and mixed-script
// splitting. Note the punctuation rule's `[\r\n/]*` tail: unlike cl100k, o200k
// admits `/` there and the vocabulary leans on it — ".\n/" is a single token.
pub(crate) const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// p50k/r50k pattern: simpler, older pattern used by GPT-3 era models
pub(crate) const P50K_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

// llama3 pattern: same structure as cl100k (contractions, letters, numbers, punctuation, whitespace)
// original uses `\s+(?!\S)|\s+` — we emulate the lookahead in pretokenize.rs
const LLAMA3_PATTERN: &str = CL100K_PATTERN;

// deepseek v3 pattern: 3 sequential splits combined into one alternation.
// priority: numbers (1-3 digits) > CJK/Japanese > general pattern
// final catch-all `[\s\S]` ensures format chars (ZWJ etc.) are not skipped,
// matching HF's Split/Isolated behavior where non-matching text is kept.
pub(crate) const DEEPSEEK_V3_PATTERN: &str = concat!(
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
pub(crate) const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

// kimi pattern (Kimi K2 / K3, from moonshotai's tokenization_kimi.py): a
// dedicated leading `[\p{Han}]+` branch, then o200k-style case-splitting rules
// whose letter classes use set intersection to exclude Han (so CJK never mixes
// into a case-split word), then the digit/punct/whitespace rules shared with
// o200k. The original ends `\s+(?!\S)|\s+`; the lookahead is emulated via
// `WhitespaceRules::NewlineFirst` like the other patterns.
pub(crate) const KIMI_PATTERN: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// glm4 / glm5 (Zhipu GLM-4.x / GLM-5.x): the tokenizer.json split regex is
// exactly the cl100k pattern; the two generations differ only in vocabulary
// (151,329 vs 154,820 base tokens, independently trained merges).
const GLM_PATTERN: &str = CL100K_PATTERN;

// minimax_m2 (MiniMax M2 family): o200k's letter/digit/whitespace rules, but
// the punctuation rule's trailing class is `[\r\n/]*` (admitting `/`, like
// Tekken) rather than o200k's `[\r\n]*`.
pub(crate) const MINIMAX_M2_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// mistral v3 (tekken) pattern. Case-splitting like o200k, but with three
// deliberate differences that a cl100k/o200k stand-in gets wrong:
//   - no contraction rule at all (no `(?i:'s|'t|…)` alternative or suffix)
//   - `\p{N}` matches a single digit, not `\p{N}{1,3}`
//   - the punctuation rule's trailing class is `[\r\n/]*`, admitting `/`
// Source: the `Split` pre-tokenizer regex in Tekken's tokenizer.json
// (e.g. mistralai/Mistral-Nemo-Base-2407).
pub(crate) const MISTRAL_V3_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

/// Parse a chain of zstd-compressed `.tkv.zst` frames into a rank map.
///
/// A chain is one frame for a self-contained vocabulary, or a base frame
/// followed by a tail frame for one of the three rank-aligned extensions
/// (llama3 over cl100k_base, glm5 over glm4, p50k_base over r50k_base).
///
/// Frame layout, after zstd decompression:
///
/// ```text
/// "TKV1"      4 B    magic
/// n_tokens    4 B    u32 LE
/// rank0       4 B    u32 LE   rank of this frame's first token
/// lengths     varint x n_tokens, in rank order
/// body        token bytes, grouped by length class (ascending),
///             each class in rank order
/// ```
///
/// Ranks are consecutive within a frame, so the rank of the `i`-th token is
/// `rank0 + i` and no rank column is stored. The body is grouped by length
/// rather than laid out in rank order because the length block is read first
/// and already says which class each token comes from — the regrouping is free
/// and compresses better. Files are produced by `src/encodings/build_tkv.py`.
///
/// The data is embedded at compile time, so malformed input is a build defect,
/// not a runtime condition: this panics rather than returning an error.
pub(crate) fn parse_tkv(chain: &[&[u8]]) -> FxHashMap<Vec<u8>, u32> {
    let mut ranks = FxHashMap::default();
    for frame in chain {
        let mut decoder =
            ruzstd::decoding::StreamingDecoder::new(*frame).expect("zstd decompression failed");
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut data).expect("zstd decompression failed");
        insert_frame(&data, &mut ranks);
    }
    ranks
}

/// Decode one decompressed TKV1 frame into `ranks`.
fn insert_frame(data: &[u8], ranks: &mut FxHashMap<Vec<u8>, u32>) {
    assert_eq!(&data[..4], b"TKV1", "bad vocabulary frame magic");
    let n_tokens = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let rank0 = u32::from_le_bytes(data[8..12].try_into().unwrap());

    let mut pos = 12;
    let mut lengths = Vec::with_capacity(n_tokens);
    let mut max_len = 0usize;
    for _ in 0..n_tokens {
        let mut len = 0usize;
        let mut shift = 0;
        loop {
            let byte = data[pos];
            pos += 1;
            len |= ((byte & 0x7F) as usize) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        max_len = max_len.max(len);
        lengths.push(len);
    }

    // where each length class starts in the body
    let mut cursors = vec![0usize; max_len + 1];
    for &len in &lengths {
        cursors[len] += 1;
    }
    let mut at = pos;
    for (len, cursor) in cursors.iter_mut().enumerate() {
        let count = *cursor;
        *cursor = at;
        at += len * count;
    }
    assert_eq!(at, data.len(), "vocabulary frame body length mismatch");

    ranks.reserve(n_tokens);
    for (i, &len) in lengths.iter().enumerate() {
        let start = cursors[len];
        cursors[len] = start + len;
        ranks.insert(data[start..start + len].to_vec(), rank0 + i as u32);
    }
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
    let encoder = parse_tkv(CL100K_BASE_CHAIN);
    let special = special_tokens(&[
        ("<|endoftext|>", 100257),
        ("<|fim_prefix|>", 100258),
        ("<|fim_middle|>", 100259),
        ("<|fim_suffix|>", 100260),
        ("<|endofprompt|>", 100276),
    ]);
    CoreBpe::new(
        encoder,
        special,
        CL100K_PATTERN,
        FastPath::Cl100k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the p50k_base encoding (text-davinci-002/003, code-davinci, code-cushman).
/// Vocabulary size: 50,256 regular tokens + 1 special token.
pub fn p50k_base() -> CoreBpe {
    let encoder = parse_tkv(P50K_BASE_CHAIN);
    let special = special_tokens(&[("<|endoftext|>", 50256)]);
    CoreBpe::new(
        encoder,
        special,
        P50K_PATTERN,
        FastPath::None,
        WhitespaceRules::Generic,
    )
}

/// Construct the p50k_edit encoding (text-davinci-edit, code-davinci-edit).
/// Same merge ranks as p50k_base but with additional FIM (fill-in-middle) special tokens.
pub fn p50k_edit() -> CoreBpe {
    let encoder = parse_tkv(P50K_BASE_CHAIN);
    let special = special_tokens(&[
        ("<|endoftext|>", 50256),
        ("<|fim_prefix|>", 50281),
        ("<|fim_middle|>", 50282),
        ("<|fim_suffix|>", 50283),
    ]);
    CoreBpe::new(
        encoder,
        special,
        P50K_PATTERN,
        FastPath::None,
        WhitespaceRules::Generic,
    )
}

/// Construct the o200k_base encoding (GPT-4o, o1, o3, o4-mini).
/// Vocabulary size: 199,998 regular tokens + 2 special tokens.
pub fn o200k_base() -> CoreBpe {
    let encoder = parse_tkv(O200K_BASE_CHAIN);
    let special = special_tokens(&[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)]);
    CoreBpe::new(
        encoder,
        special,
        O200K_PATTERN,
        FastPath::O200k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the o200k_harmony encoding (gpt-oss family / harmony chat format).
///
/// Shares merge ranks and regex with [`o200k_base`]; the only delta is the
/// special-token table — 15 named tokens (199998..=200012) plus 1075 reserved
/// placeholders (`<|reserved_200013|>`..=`<|reserved_201087|>`).
///
/// Note: `<|reserved_200018|>` shadows `<|endofprompt|>` from o200k_base
/// at the same id; this mirrors the upstream Python implementation.
pub fn o200k_harmony() -> CoreBpe {
    let encoder = parse_tkv(O200K_BASE_CHAIN);
    let mut special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    for (name, id) in [
        ("<|startoftext|>", 199998_u32),
        ("<|endoftext|>", 199999),
        ("<|reserved_200000|>", 200000),
        ("<|reserved_200001|>", 200001),
        ("<|return|>", 200002),
        ("<|constrain|>", 200003),
        ("<|reserved_200004|>", 200004),
        ("<|channel|>", 200005),
        ("<|start|>", 200006),
        ("<|end|>", 200007),
        ("<|message|>", 200008),
        ("<|reserved_200009|>", 200009),
        ("<|reserved_200010|>", 200010),
        ("<|reserved_200011|>", 200011),
        ("<|call|>", 200012),
    ] {
        special.insert(name.as_bytes().to_vec(), id);
    }
    for id in 200013..=201087_u32 {
        special.insert(format!("<|reserved_{id}|>").into_bytes(), id);
    }
    CoreBpe::new(
        encoder,
        special,
        O200K_PATTERN,
        FastPath::O200k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the r50k_base encoding (GPT-3 era: davinci, curie, babbage, ada).
/// Vocabulary size: 50,256 regular tokens + 1 special token.
/// Uses the same merge ranks and regex pattern as p50k_base.
pub fn r50k_base() -> CoreBpe {
    let encoder = parse_tkv(R50K_BASE_CHAIN);
    let special = special_tokens(&[("<|endoftext|>", 50256)]);
    CoreBpe::new(
        encoder,
        special,
        P50K_PATTERN,
        FastPath::None,
        WhitespaceRules::Generic,
    )
}

/// Construct the `gpt2` encoding (GPT-2 BPE).
///
/// Byte-for-byte identical to [`r50k_base`] — same merge ranks, regex, and
/// single special token (`<|endoftext|>` at 50256). Exposed as a distinct
/// name for parity with upstream `openai/tiktoken`; the runtime shares
/// `r50k_base`'s cached instance.
pub fn gpt2() -> CoreBpe {
    r50k_base()
}

/// Construct the llama3 encoding (Llama 3 / 3.1 / 3.2 / 3.3).
/// Vocabulary size: 128,000 regular tokens + 256 special tokens.
pub fn llama3() -> CoreBpe {
    let encoder = parse_tkv(LLAMA3_CHAIN);
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
    CoreBpe::new(
        encoder,
        special,
        LLAMA3_PATTERN,
        FastPath::Cl100k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the deepseek_v3 encoding (DeepSeek V3, R1).
///
/// Vocabulary size: 128,000 regular tokens + 818 added tokens — 3 sentence
/// markers, 800 placeholders (`<｜place▁holder▁no▁0｜>`..=`no▁799｜>`, ids
/// 128000..=128799), and 15 named tokens (ids 128800..=128814).
///
/// Note the named tokens mostly use fullwidth pipes (`｜`, U+FF5C), not ASCII
/// `|`; `<|EOT|>` is the one exception and really is ASCII.
pub fn deepseek_v3() -> CoreBpe {
    let encoder = parse_tkv(DEEPSEEK_V3_CHAIN);
    CoreBpe::new(
        encoder,
        deepseek_v3_special_tokens(),
        DEEPSEEK_V3_PATTERN,
        FastPath::Deepseek,
        WhitespaceRules::NewlineFirstSplitOnNumCjk,
    )
}

/// DeepSeek V3's 818-entry added-token table, shared as the base of
/// [`deepseek_v4`]'s table.
fn deepseek_v3_special_tokens() -> FxHashMap<Vec<u8>, u32> {
    let mut special = special_tokens(&[
        ("<｜begin▁of▁sentence｜>", 0),
        ("<｜end▁of▁sentence｜>", 1),
        ("<｜▁pad▁｜>", 2),
        ("<｜fim▁hole｜>", 128800),
        ("<｜fim▁begin｜>", 128801),
        ("<｜fim▁end｜>", 128802),
        ("<｜User｜>", 128803),
        ("<｜Assistant｜>", 128804),
        ("<|EOT|>", 128805),
        ("<｜tool▁calls▁begin｜>", 128806),
        ("<｜tool▁calls▁end｜>", 128807),
        ("<｜tool▁call▁begin｜>", 128808),
        ("<｜tool▁call▁end｜>", 128809),
        ("<｜tool▁outputs▁begin｜>", 128810),
        ("<｜tool▁outputs▁end｜>", 128811),
        ("<｜tool▁output▁begin｜>", 128812),
        ("<｜tool▁output▁end｜>", 128813),
        ("<｜tool▁sep｜>", 128814),
    ]);
    for id in 128000..=128799_u32 {
        let n = id - 128000;
        special.insert(format!("<｜place▁holder▁no▁{n}｜>").into_bytes(), id);
    }
    special
}

/// Construct the qwen2 encoding (Qwen 2.5 / 3).
/// Vocabulary size: 151,643 regular tokens + 22 added tokens (ids
/// 151643..=151664, covering the chat, vision, tool-call and FIM markers).
pub fn qwen2() -> CoreBpe {
    let encoder = parse_tkv(QWEN2_CHAIN);
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
        ("<tool_call>", 151657),
        ("</tool_call>", 151658),
        ("<|fim_prefix|>", 151659),
        ("<|fim_middle|>", 151660),
        ("<|fim_suffix|>", 151661),
        ("<|fim_pad|>", 151662),
        ("<|repo_name|>", 151663),
        ("<|file_sep|>", 151664),
    ]);
    CoreBpe::new(
        encoder,
        special,
        QWEN2_PATTERN,
        FastPath::Qwen2,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the mistral_v3 encoding (Mistral, Mixtral with Tekken tokenizer).
/// Vocabulary size: 131,072 regular tokens + 1000 special tokens.
pub fn mistral_v3() -> CoreBpe {
    let encoder = parse_tkv(MISTRAL_V3_CHAIN);
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
    CoreBpe::new(
        encoder,
        special,
        MISTRAL_V3_PATTERN,
        FastPath::Tekken,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the deepseek_v4 encoding (DeepSeek V4 Pro / Flash, 2026).
///
/// Same 128,000-token vocabulary, merges, and split pattern as
/// [`deepseek_v3`]; the delta is the added-token table, which grows from
/// V3's 818 entries to 1,283 — 50 new named tokens (`<think>`, the DSML
/// markup markers, vision/grounding tags) plus 415 multimodal span
/// placeholders (`<|place_holder_mm_span_0021|>`..=`_0435|>`).
pub fn deepseek_v4() -> CoreBpe {
    let encoder = parse_tkv(DEEPSEEK_V3_CHAIN);
    let mut special = deepseek_v3_special_tokens();
    for (name, id) in [
        ("<｜begin▁of▁repo▁name｜>", 128815_u32),
        ("<｜end▁of▁repo▁name｜>", 128816),
        ("<｜begin▁of▁file▁name｜>", 128817),
        ("<｜end▁of▁file▁name｜>", 128818),
        ("<｜begin▁of▁file｜>", 128819),
        ("<｜end▁of▁file｜>", 128820),
        ("<think>", 128821),
        ("</think>", 128822),
        ("<｜place▁holder▁for▁copy｜>", 128823),
        ("<｜place▁holder▁for▁pointer▁replace｜>", 128824),
        ("｜DSML｜", 128825),
        ("<｜begin▁sys｜>", 128826),
        ("<｜end▁sys｜>", 128827),
        ("<｜latest_reminder｜>", 128828),
        ("<｜action｜>", 128829),
        ("<｜query｜>", 128830),
        ("<｜authority｜>", 128831),
        ("<｜domain｜>", 128832),
        ("<｜task｜>", 128833),
        ("<｜political｜>", 128834),
        ("<｜entity｜>", 128835),
        ("<｜title｜>", 128836),
        ("<｜safety｜>", 128837),
        ("<｜answer｜>", 128838),
        ("<｜search｜>", 128839),
        ("<dsml:", 128840),
        ("</dsml:", 128841),
        ("<｜search▁begin｜>", 128842),
        ("<｜search▁end｜>", 128843),
        ("<｜extracted_url｜>", 128844),
        ("<｜read_url｜>", 128845),
        ("<｜end_of_query｜>", 128846),
        ("<｜rl_image_pad｜>", 129262),
        ("<｜rl_image_start｜>", 129263),
        ("<｜image2｜>", 129264),
        ("<｜/table>｜", 129265),
        ("<｜table｜>", 129266),
        ("<｜/td｜>", 129267),
        ("<｜td｜>", 129268),
        ("<｜/tr｜>", 129269),
        ("<｜tr｜>", 129270),
        ("<｜/polygon｜>", 129271),
        ("<｜polygon｜>", 129272),
        ("<｜/point｜>", 129273),
        ("<｜point｜>", 129274),
        ("<｜/box｜>", 129275),
        ("<｜box｜>", 129276),
        ("<｜/ref｜>", 129277),
        ("<｜ref｜>", 129278),
        ("<｜image｜>", 129279),
    ] {
        special.insert(name.as_bytes().to_vec(), id);
    }
    // 415 multimodal span placeholders: number 0021..=0435 map to contiguous
    // ids 128847..=129261 (id = 128847 + (n - 21)).
    for n in 21..=435_u32 {
        special.insert(
            format!("<|place_holder_mm_span_{n:04}|>").into_bytes(),
            128847 + (n - 21),
        );
    }
    CoreBpe::new(
        encoder,
        special,
        DEEPSEEK_V3_PATTERN,
        FastPath::Deepseek,
        WhitespaceRules::NewlineFirstSplitOnNumCjk,
    )
}

/// The Kimi K2 / K3 shared base vocabulary is 163,584 tokens (byte-identical
/// `tiktoken.model` across both generations); each generation defines its own
/// special-token ids in the 163584..163839 reserved range.
///
/// Construct the kimi_k2 encoding (Kimi K2 / K2.5 / K2.6, Moonshot).
pub fn kimi_k2() -> CoreBpe {
    let encoder = parse_tkv(KIMI_K2_CHAIN);
    let special = special_tokens(&[
        ("[BOS]", 163584),
        ("[EOS]", 163585),
        ("<|im_end|>", 163586),
        ("<|im_user|>", 163587),
        ("<|im_assistant|>", 163588),
        ("<|start_header_id|>", 163590),
        ("<|end_header_id|>", 163591),
        ("[EOT]", 163593),
        ("<|im_system|>", 163594),
        ("<|tool_calls_section_begin|>", 163595),
        ("<|tool_calls_section_end|>", 163596),
        ("<|tool_call_begin|>", 163597),
        ("<|tool_call_argument_begin|>", 163598),
        ("<|tool_call_end|>", 163599),
        ("<|im_middle|>", 163601),
        ("[UNK]", 163838),
        ("[PAD]", 163839),
    ]);
    CoreBpe::new(
        encoder,
        special,
        KIMI_PATTERN,
        FastPath::Kimi,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the kimi_k3 encoding (Kimi K3, Moonshot 2026).
///
/// Shares [`kimi_k2`]'s merge ranks and regex; only the special-token table
/// differs (K3 renames the chat markers and adds media tokens).
pub fn kimi_k3() -> CoreBpe {
    let encoder = parse_tkv(KIMI_K2_CHAIN);
    let special = special_tokens(&[
        ("[BOS]", 163584),
        ("[EOS]", 163585),
        ("<|end_of_msg|>", 163586),
        ("<|open|>", 163587),
        ("<|close|>", 163588),
        ("<|sep|>", 163589),
        ("[start_header_id]", 163590),
        ("[end_header_id]", 163591),
        ("[EOT]", 163593),
        ("<|media_begin|>", 163602),
        ("<|media_content|>", 163603),
        ("<|media_end|>", 163604),
        ("<|media_pad|>", 163605),
        ("<osagent_mode>", 163649),
        ("[UNK]", 163838),
        ("[PAD]", 163839),
    ]);
    CoreBpe::new(
        encoder,
        special,
        KIMI_PATTERN,
        FastPath::Kimi,
        WhitespaceRules::NewlineFirst,
    )
}

/// Build the GLM special-token table: both generations use the same 36 names
/// at contiguous ids starting right after the base vocabulary.
fn glm_special_tokens(base: u32) -> FxHashMap<Vec<u8>, u32> {
    const NAMES: [&str; 36] = [
        "<|endoftext|>",
        "[MASK]",
        "[gMASK]",
        "[sMASK]",
        "<sop>",
        "<eop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
        "<|begin_of_transcription|>",
        "<|end_of_transcription|>",
        "<|code_prefix|>",
        "<|code_middle|>",
        "<|code_suffix|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "/nothink",
        "<|begin_of_box|>",
        "<|end_of_box|>",
        "<|image|>",
        "<|video|>",
    ];
    NAMES
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_bytes().to_vec(), base + i as u32))
        .collect()
}

/// Construct the glm4 encoding (Zhipu GLM-4.5 / 4.6 / 4.7).
/// Vocabulary size: 151,329 regular tokens + 36 special tokens.
pub fn glm4() -> CoreBpe {
    let encoder = parse_tkv(GLM4_CHAIN);
    CoreBpe::new(
        encoder,
        glm_special_tokens(151_329),
        GLM_PATTERN,
        FastPath::Cl100k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the glm5 encoding (Zhipu GLM-5 / 5.2).
/// Vocabulary size: 154,820 regular tokens + 36 special tokens
/// (independently trained merges — not an extension of glm4's).
pub fn glm5() -> CoreBpe {
    let encoder = parse_tkv(GLM5_CHAIN);
    CoreBpe::new(
        encoder,
        glm_special_tokens(154_820),
        GLM_PATTERN,
        FastPath::Cl100k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the minimax_m2 encoding (MiniMax M2 / M2.1 / M2.5 / M2.7).
/// Vocabulary size: 200,000 regular tokens + 54 special tokens
/// (byte-identical tokenizer across the whole M2 family).
pub fn minimax_m2() -> CoreBpe {
    let encoder = parse_tkv(MINIMAX_M2_CHAIN);
    let special = special_tokens(&[
        ("]!p~[", 200000),
        ("<fim_prefix>", 200001),
        ("<fim_middle>", 200002),
        ("<fim_suffix>", 200003),
        ("<fim_pad>", 200004),
        ("<reponame>", 200005),
        ("<filename>", 200006),
        ("<gh_stars>", 200007),
        ("<issue_start>", 200008),
        ("<issue_comment>", 200009),
        ("<issue_closed>", 200010),
        ("<jupyter_start>", 200011),
        ("<jupyter_text>", 200012),
        ("<jupyter_code>", 200013),
        ("<jupyter_output>", 200014),
        ("<empty_output>", 200015),
        ("<commit_before>", 200016),
        ("<commit_msg>", 200017),
        ("<commit_after>", 200018),
        ("]~b]", 200019),
        ("[e~[", 200020),
        ("]!d~[", 200021),
        ("<function_call>", 200022),
        ("<code_interpreter>", 200023),
        ("]<]speech[>[", 200024),
        ("]<]image[>[", 200025),
        ("]<]video[>[", 200026),
        ("]<]start of speech[>[", 200027),
        ("]<]end of speech[>[", 200028),
        ("]<]start of image[>[", 200029),
        ("]<]end of image[>[", 200030),
        ("]<]start of video[>[", 200031),
        ("]<]end of video[>[", 200032),
        ("]<]vision pad[>[", 200033),
        ("]~!b[", 200034),
        ("<jupyter_error>", 200035),
        ("<add_file>", 200036),
        ("<delete_file>", 200037),
        ("<rename_file>", 200038),
        ("<edit_file>", 200039),
        ("<commit_message>", 200040),
        ("<empty_source_file>", 200041),
        ("<repo_struct>", 200042),
        ("<code_context>", 200043),
        ("<file_content>", 200044),
        ("<source_files>", 200045),
        ("<pr_start>", 200046),
        ("<review_comment>", 200047),
        ("<filepath>", 200048),
        ("<file_sep>", 200049),
        ("<think>", 200050),
        ("</think>", 200051),
        ("<minimax:tool_call>", 200052),
        ("</minimax:tool_call>", 200053),
    ]);
    CoreBpe::new(
        encoder,
        special,
        MINIMAX_M2_PATTERN,
        FastPath::MiniMax,
        WhitespaceRules::NewlineFirst,
    )
}

/// Expose cl100k rank map for internal tests (e.g. Vocab equivalence)
#[cfg(test)]
pub(crate) fn parse_tiktoken_data_for_test() -> FxHashMap<Vec<u8>, u32> {
    parse_tkv(CL100K_BASE_CHAIN)
}

/// Differential guard for the shipped vocabulary data.
///
/// The `.tkv.zst` files are a compact re-encoding of the `.tiktoken.zst`
/// reference files under `tests/vocab-oracle/` (base64 token + explicit rank,
/// one per line — the form upstream publishes). Those references stay in the
/// repository and out of the published package, so a re-encoding bug cannot
/// hide: every shipped frame chain is decoded here and diffed against them,
/// entry by entry.
#[cfg(test)]
mod vocab_oracle {
    use super::*;
    use base64::Engine;

    /// Every shipped vocabulary, as (name, frame chain).
    const CHAINS: &[(&str, &[&[u8]])] = &[
        ("cl100k_base", CL100K_BASE_CHAIN),
        ("o200k_base", O200K_BASE_CHAIN),
        ("r50k_base", R50K_BASE_CHAIN),
        ("p50k_base", P50K_BASE_CHAIN),
        ("llama3", LLAMA3_CHAIN),
        ("deepseek_v3", DEEPSEEK_V3_CHAIN),
        ("qwen2", QWEN2_CHAIN),
        ("mistral_v3", MISTRAL_V3_CHAIN),
        ("kimi_k2", KIMI_K2_CHAIN),
        ("glm4", GLM4_CHAIN),
        ("glm5", GLM5_CHAIN),
        ("minimax_m2", MINIMAX_M2_CHAIN),
    ];

    /// Parse `tests/vocab-oracle/<name>.tiktoken.zst`: one
    /// `<base64(token bytes)> <rank>` line per entry.
    fn oracle(name: &str) -> FxHashMap<Vec<u8>, u32> {
        let path = format!(
            "{}/tests/vocab-oracle/{name}.tiktoken.zst",
            env!("CARGO_MANIFEST_DIR")
        );
        let compressed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut decoder = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice()).unwrap();
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut data).unwrap();

        let engine = base64::engine::general_purpose::STANDARD;
        let text = std::str::from_utf8(&data).unwrap();
        let mut ranks = FxHashMap::default();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (token, rank) = line.trim().split_once(' ').expect("malformed oracle line");
            ranks.insert(engine.decode(token).unwrap(), rank.parse().unwrap());
        }
        ranks
    }

    #[test]
    fn shipped_vocabularies_match_the_oracle() {
        for &(name, chain) in CHAINS {
            let shipped = parse_tkv(chain);
            let reference = oracle(name);
            assert_eq!(
                shipped.len(),
                reference.len(),
                "[{name}] token count: {} shipped vs {} in oracle",
                shipped.len(),
                reference.len()
            );
            for (token, rank) in &reference {
                assert_eq!(
                    shipped.get(token),
                    Some(rank),
                    "[{name}] rank {rank} token {token:?} missing or misranked"
                );
            }
        }
    }

    /// The three derived vocabularies must stay exact rank-aligned extensions of
    /// their base — the property that lets their files hold only the tail.
    #[test]
    fn derived_vocabularies_extend_their_base() {
        for (derived, base) in [
            ("llama3", "cl100k_base"),
            ("glm5", "glm4"),
            ("p50k_base", "r50k_base"),
        ] {
            let derived = oracle(derived);
            for (token, rank) in oracle(base) {
                assert_eq!(derived.get(&token), Some(&rank), "token {token:?}");
            }
        }
    }
}
