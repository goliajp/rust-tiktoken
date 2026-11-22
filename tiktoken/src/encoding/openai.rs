//! OpenAI encodings: cl100k_base, o200k_base / o200k_harmony, and the GPT-3 era
//! p50k_base / p50k_edit / r50k_base (alias gpt2).

use super::*;

/// Construct the cl100k_base encoding (GPT-4, GPT-3.5 Turbo, embeddings).
/// Vocabulary size: 100,256 regular tokens + 5 special tokens.
#[cfg(feature = "vocab-cl100k_base")]
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
#[cfg(feature = "vocab-p50k_base")]
pub fn p50k_base() -> CoreBpe {
    let vocab = shared_vocab("p50k_base", P50K_BASE_CHAIN);
    let special = special_tokens(&[("<|endoftext|>", 50256)]);
    CoreBpe::with_vocab(
        vocab,
        special,
        P50K_PATTERN,
        FastPath::P50k,
        WhitespaceRules::Generic,
    )
}

/// Construct the p50k_edit encoding (text-davinci-edit, code-davinci-edit).
/// Same merge ranks as p50k_base but with additional FIM (fill-in-middle) special tokens.
#[cfg(feature = "vocab-p50k_base")]
pub fn p50k_edit() -> CoreBpe {
    let vocab = shared_vocab("p50k_base", P50K_BASE_CHAIN);
    let special = special_tokens(&[
        ("<|endoftext|>", 50256),
        ("<|fim_prefix|>", 50281),
        ("<|fim_middle|>", 50282),
        ("<|fim_suffix|>", 50283),
    ]);
    CoreBpe::with_vocab(
        vocab,
        special,
        P50K_PATTERN,
        FastPath::P50k,
        WhitespaceRules::Generic,
    )
}

/// Construct the o200k_base encoding (GPT-4o, o1, o3, o4-mini).
/// Vocabulary size: 199,998 regular tokens + 2 special tokens.
#[cfg(feature = "vocab-o200k_base")]
pub fn o200k_base() -> CoreBpe {
    let vocab = shared_vocab("o200k_base", O200K_BASE_CHAIN);
    let special = special_tokens(&[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)]);
    CoreBpe::with_vocab(
        vocab,
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
#[cfg(feature = "vocab-o200k_base")]
pub fn o200k_harmony() -> CoreBpe {
    let vocab = shared_vocab("o200k_base", O200K_BASE_CHAIN);
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
    CoreBpe::with_vocab(
        vocab,
        special,
        O200K_PATTERN,
        FastPath::O200k,
        WhitespaceRules::NewlineFirst,
    )
}

/// Construct the r50k_base encoding (GPT-3 era: davinci, curie, babbage, ada).
/// Vocabulary size: 50,256 regular tokens + 1 special token.
/// Uses the same merge ranks and regex pattern as p50k_base.
#[cfg(feature = "vocab-r50k_base")]
pub fn r50k_base() -> CoreBpe {
    let encoder = parse_tkv(R50K_BASE_CHAIN);
    let special = special_tokens(&[("<|endoftext|>", 50256)]);
    CoreBpe::new(
        encoder,
        special,
        P50K_PATTERN,
        FastPath::P50k,
        WhitespaceRules::Generic,
    )
}

/// Construct the `gpt2` encoding (GPT-2 BPE).
///
/// Byte-for-byte identical to [`r50k_base`] — same merge ranks, regex, and
/// single special token (`<|endoftext|>` at 50256). Exposed as a distinct
/// name for parity with upstream `openai/tiktoken`; the runtime shares
/// `r50k_base`'s cached instance.
#[cfg(feature = "vocab-r50k_base")]
pub fn gpt2() -> CoreBpe {
    r50k_base()
}
