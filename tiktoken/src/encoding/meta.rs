//! Meta encodings: llama3 (Llama 3.x) and llama4 (Llama 4).

use super::*;

/// Construct the llama3 encoding (Llama 3 / 3.1 / 3.2 / 3.3).
/// Vocabulary size: 128,000 regular tokens + 256 special tokens.
#[cfg(feature = "vocab-llama3")]
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

/// Construct the llama4 encoding (Llama 4 Scout / Maverick, and Meta's Muse
/// open-weight models, which share its vocabulary).
///
/// Vocabulary size: 200,000 regular tokens, unrelated to [`llama3`]'s. Of the
/// 1,135 added tokens (200000..=201134) the named ones are registered here;
/// the remainder are reserved placeholders.
#[cfg(feature = "vocab-llama4")]
pub fn llama4() -> CoreBpe {
    let encoder = parse_tkv(LLAMA4_CHAIN);
    let special = special_tokens(&[
        ("<|begin_of_text|>", 200000),
        ("<|end_of_text|>", 200001),
        ("<|fim_prefix|>", 200002),
        ("<|fim_middle|>", 200003),
        ("<|fim_suffix|>", 200004),
        ("<|header_start|>", 200005),
        ("<|header_end|>", 200006),
        ("<|eom|>", 200007),
        ("<|eot|>", 200008),
        ("<|step|>", 200009),
        ("<|python_start|>", 200016),
        ("<|python_end|>", 200017),
        ("<|finetune_right_pad|>", 200018),
        ("<|image_start|>", 200080),
        ("<|image_end|>", 200081),
        ("<|tile_x_separator|>", 200084),
        ("<|tile_y_separator|>", 200085),
        ("<|image|>", 200090),
        ("<|patch|>", 200092),
        ("<|finetune_right_pad_id|>", 201134),
    ]);
    CoreBpe::new(
        encoder,
        special,
        LLAMA4_PATTERN,
        FastPath::MiniMax,
        WhitespaceRules::NewlineFirst,
    )
}
