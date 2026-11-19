//! Zhipu GLM encodings: glm4 and glm5.

use super::*;

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
#[cfg(feature = "vocab-glm4")]
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
#[cfg(feature = "vocab-glm5")]
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
