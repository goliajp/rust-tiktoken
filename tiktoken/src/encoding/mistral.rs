//! Mistral encodings: mistral_v3 (Tekken, 131k vocabulary) and mistral_v15
//! (Tekken v15, 204.8k vocabulary).

use super::*;

/// Construct the mistral_v3 encoding (Mistral, Mixtral with Tekken tokenizer).
/// Vocabulary size: 131,072 regular tokens + 1000 special tokens.
#[cfg(feature = "vocab-mistral_v3")]
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

/// Construct the mistral_v15 encoding (Tekken v15: Mistral Large 4).
///
/// Vocabulary size: 204,800 including 1,000 control tokens at ids 0..1000,
/// retrained rather than extended from [`mistral_v3`]'s. Other v15 models
/// (Mistral Small 4) keep the 131k vocabulary and use `mistral_v3`.
#[cfg(feature = "vocab-mistral_v15")]
pub fn mistral_v15() -> CoreBpe {
    let encoder = parse_tkv(MISTRAL_V15_CHAIN);
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
        ("[SYSTEM_PROMPT]", 17),
        ("[/SYSTEM_PROMPT]", 18),
        ("[TOOL_CONTENT]", 19),
        ("[AUDIO]", 24),
        ("[BEGIN_AUDIO]", 25),
        ("[ARGS]", 32),
        ("[CALL_ID]", 33),
        ("[THINK]", 34),
        ("[/THINK]", 35),
        ("[MODEL_SETTINGS]", 36),
        ("[/MODEL_SETTINGS]", 37),
    ]);
    CoreBpe::new(
        encoder,
        special,
        MISTRAL_V15_PATTERN,
        FastPath::Tekken15,
        WhitespaceRules::NewlineFirst,
    )
}
