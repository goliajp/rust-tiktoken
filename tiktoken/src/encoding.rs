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
// lib.rs. The data lives in the tiktoken-vocab-* crates, one per vendor, built
// from the `tests/vocab-oracle/*.tiktoken.zst` reference files by
// `vocabs/tools/build_tkv.py`; the `vocab_oracle` tests below diff the two.
//
// Each file is gated on its `vocab-*` feature so that a build only carries the
// vocabularies it names — see the feature table in Cargo.toml.
#[cfg(feature = "vocab-cl100k_base")]
const CL100K_BASE_TKV: &[u8] = tiktoken_vocab_openai::CL100K_BASE;
#[cfg(feature = "vocab-o200k_base")]
const O200K_BASE_TKV: &[u8] = tiktoken_vocab_openai::O200K_BASE;
#[cfg(feature = "vocab-r50k_base")]
const R50K_BASE_TKV: &[u8] = tiktoken_vocab_openai::R50K_BASE;
#[cfg(feature = "vocab-deepseek_v3")]
const DEEPSEEK_V3_TKV: &[u8] = tiktoken_vocab_deepseek::DEEPSEEK_V3;
#[cfg(feature = "vocab-qwen2")]
const QWEN2_TKV: &[u8] = tiktoken_vocab_qwen::QWEN2;
#[cfg(feature = "vocab-mistral_v3")]
const MISTRAL_V3_TKV: &[u8] = tiktoken_vocab_mistral::MISTRAL_V3;
#[cfg(feature = "vocab-kimi_k2")]
const KIMI_K2_TKV: &[u8] = tiktoken_vocab_moonshot::KIMI_K2;
#[cfg(feature = "vocab-glm4")]
const GLM4_TKV: &[u8] = tiktoken_vocab_zhipu::GLM4;
#[cfg(feature = "vocab-minimax_m2")]
const MINIMAX_M2_TKV: &[u8] = tiktoken_vocab_minimax::MINIMAX_M2;
#[cfg(feature = "vocab-qwen3_5")]
const QWEN3_5_TKV: &[u8] = tiktoken_vocab_qwen::QWEN3_5;
#[cfg(feature = "vocab-llama4")]
const LLAMA4_TKV: &[u8] = tiktoken_vocab_meta::LLAMA4;
#[cfg(feature = "vocab-mistral_v15")]
const MISTRAL_V15_TKV: &[u8] = tiktoken_vocab_mistral::MISTRAL_V15;

// Three vocabularies are exact rank-aligned extensions of another one: every
// token at rank `i` of the base is the token at rank `i` of the derived
// vocabulary. Their files hold only the tail, so the shared prefix is stored
// once. Reconstructing them costs nothing extra at run time — those leading
// entries have to be built either way. Their features pull in the base's
// feature, which is why the chains below can reference it unconditionally.
#[cfg(feature = "vocab-llama3")]
const LLAMA3_TAIL_TKV: &[u8] = tiktoken_vocab_meta::LLAMA3_TAIL;
#[cfg(feature = "vocab-glm5")]
const GLM5_TAIL_TKV: &[u8] = tiktoken_vocab_zhipu::GLM5_TAIL;
#[cfg(feature = "vocab-p50k_base")]
const P50K_BASE_TAIL_TKV: &[u8] = tiktoken_vocab_openai::P50K_BASE_TAIL;

#[cfg(feature = "vocab-cl100k_base")]
pub(crate) const CL100K_BASE_CHAIN: &[&[u8]] = &[CL100K_BASE_TKV];
#[cfg(feature = "vocab-o200k_base")]
pub(crate) const O200K_BASE_CHAIN: &[&[u8]] = &[O200K_BASE_TKV];
#[cfg(feature = "vocab-r50k_base")]
pub(crate) const R50K_BASE_CHAIN: &[&[u8]] = &[R50K_BASE_TKV];
#[cfg(feature = "vocab-deepseek_v3")]
pub(crate) const DEEPSEEK_V3_CHAIN: &[&[u8]] = &[DEEPSEEK_V3_TKV];
#[cfg(feature = "vocab-qwen2")]
pub(crate) const QWEN2_CHAIN: &[&[u8]] = &[QWEN2_TKV];
#[cfg(feature = "vocab-mistral_v3")]
pub(crate) const MISTRAL_V3_CHAIN: &[&[u8]] = &[MISTRAL_V3_TKV];
#[cfg(feature = "vocab-kimi_k2")]
pub(crate) const KIMI_K2_CHAIN: &[&[u8]] = &[KIMI_K2_TKV];
#[cfg(feature = "vocab-glm4")]
pub(crate) const GLM4_CHAIN: &[&[u8]] = &[GLM4_TKV];
#[cfg(feature = "vocab-minimax_m2")]
pub(crate) const MINIMAX_M2_CHAIN: &[&[u8]] = &[MINIMAX_M2_TKV];
#[cfg(feature = "vocab-qwen3_5")]
pub(crate) const QWEN3_5_CHAIN: &[&[u8]] = &[QWEN3_5_TKV];
#[cfg(feature = "vocab-llama4")]
pub(crate) const LLAMA4_CHAIN: &[&[u8]] = &[LLAMA4_TKV];
#[cfg(feature = "vocab-mistral_v15")]
pub(crate) const MISTRAL_V15_CHAIN: &[&[u8]] = &[MISTRAL_V15_TKV];
#[cfg(feature = "vocab-llama3")]
pub(crate) const LLAMA3_CHAIN: &[&[u8]] = &[CL100K_BASE_TKV, LLAMA3_TAIL_TKV];
#[cfg(feature = "vocab-glm5")]
pub(crate) const GLM5_CHAIN: &[&[u8]] = &[GLM4_TKV, GLM5_TAIL_TKV];
#[cfg(feature = "vocab-p50k_base")]
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

// qwen3_5 pattern (Qwen 3.5 and later): qwen2's, except that the letter rule
// also takes combining marks (`[\p{L}\p{M}]+`) and the punctuation rule
// excludes them. The two differ only on text with combining marks.
pub(crate) const QWEN3_5_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

// llama4 pattern (Llama 4): identical to minimax_m2's, o200k with a `[\r\n/]*`
// punctuation tail.
const LLAMA4_PATTERN: &str = MINIMAX_M2_PATTERN;

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

// mistral v15 (Tekken v15, Mistral Large 4) pattern: v3's, except digits. The
// upstream rule `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}` groups a digit
// run in threes from the right; the lookahead is out of the `regex` crate's
// reach, so `FastPath::Tekken15` cuts digit runs before the regex runs, and
// the regex only needs to classify them.
pub(crate) const MISTRAL_V15_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"|\p{N}+",
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
/// and compresses better. Files are produced by `vocabs/tools/build_tkv.py`.
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

mod deepseek;
mod meta;
mod minimax;
mod mistral;
mod moonshot;
mod openai;
mod qwen;
mod zhipu;

pub use deepseek::*;
pub use meta::*;
pub use minimax::*;
pub use mistral::*;
pub use moonshot::*;
pub use openai::*;
pub use qwen::*;
pub use zhipu::*;

/// Build a special token map from `(text, id)` pairs.
fn special_tokens(pairs: &[(&str, u32)]) -> FxHashMap<Vec<u8>, u32> {
    pairs
        .iter()
        .map(|&(s, v)| (s.as_bytes().to_vec(), v))
        .collect()
}

/// Expose cl100k rank map for internal tests (e.g. Vocab equivalence)
#[cfg(all(test, feature = "vocab-cl100k_base"))]
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

    /// Every vocabulary this build carries, as (name, frame chain).
    // the pushes are cfg-gated, so a collect() would not express this
    #[allow(clippy::vec_init_then_push)]
    fn chains() -> Vec<(&'static str, &'static [&'static [u8]])> {
        #[allow(unused_mut)]
        let mut chains: Vec<(&'static str, &'static [&'static [u8]])> = Vec::new();
        #[cfg(feature = "vocab-cl100k_base")]
        chains.push(("cl100k_base", CL100K_BASE_CHAIN));
        #[cfg(feature = "vocab-o200k_base")]
        chains.push(("o200k_base", O200K_BASE_CHAIN));
        #[cfg(feature = "vocab-r50k_base")]
        chains.push(("r50k_base", R50K_BASE_CHAIN));
        #[cfg(feature = "vocab-p50k_base")]
        chains.push(("p50k_base", P50K_BASE_CHAIN));
        #[cfg(feature = "vocab-llama3")]
        chains.push(("llama3", LLAMA3_CHAIN));
        #[cfg(feature = "vocab-deepseek_v3")]
        chains.push(("deepseek_v3", DEEPSEEK_V3_CHAIN));
        #[cfg(feature = "vocab-qwen2")]
        chains.push(("qwen2", QWEN2_CHAIN));
        #[cfg(feature = "vocab-mistral_v3")]
        chains.push(("mistral_v3", MISTRAL_V3_CHAIN));
        #[cfg(feature = "vocab-kimi_k2")]
        chains.push(("kimi_k2", KIMI_K2_CHAIN));
        #[cfg(feature = "vocab-glm4")]
        chains.push(("glm4", GLM4_CHAIN));
        #[cfg(feature = "vocab-glm5")]
        chains.push(("glm5", GLM5_CHAIN));
        #[cfg(feature = "vocab-minimax_m2")]
        chains.push(("minimax_m2", MINIMAX_M2_CHAIN));
        #[cfg(feature = "vocab-qwen3_5")]
        chains.push(("qwen3_5", QWEN3_5_CHAIN));
        #[cfg(feature = "vocab-llama4")]
        chains.push(("llama4", LLAMA4_CHAIN));
        #[cfg(feature = "vocab-mistral_v15")]
        chains.push(("mistral_v15", MISTRAL_V15_CHAIN));
        chains
    }

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
        for (name, chain) in chains() {
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
    /// their base — the property that lets their files hold only the tail. Read
    /// straight from the oracles, so this holds regardless of which features are on.
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
