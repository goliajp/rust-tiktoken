//! Alibaba Qwen encodings: qwen2 (Qwen 2 through Qwen 3) and qwen3_5 (Qwen 3.5
//! and later).

use super::*;

/// Construct the qwen2 encoding (Qwen 2.5 / 3).
/// Vocabulary size: 151,643 regular tokens + 22 added tokens (ids
/// 151643..=151664, covering the chat, vision, tool-call and FIM markers).
#[cfg(feature = "vocab-qwen2")]
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

/// Construct the qwen3_5 encoding (Qwen 3.5 through 3.8).
///
/// Vocabulary size: 248,044 regular tokens, a vocabulary trained afresh rather
/// than extended from [`qwen2`]'s. Special tokens are Qwen 3.8's 33, a superset
/// of Qwen 3.5's 26 (3.8 adds the audio and TTS markers at 248070..=248076).
///
/// Assumes NFC-normalized input, as [`qwen2`] does: Qwen's tokenizer
/// normalizes to NFC before splitting, and this crate does not normalize.
#[cfg(feature = "vocab-qwen3_5")]
pub fn qwen3_5() -> CoreBpe {
    let encoder = parse_tkv(QWEN3_5_CHAIN);
    let special = special_tokens(&[
        ("<|endoftext|>", 248044),
        ("<|im_start|>", 248045),
        ("<|im_end|>", 248046),
        ("<|object_ref_start|>", 248047),
        ("<|object_ref_end|>", 248048),
        ("<|box_start|>", 248049),
        ("<|box_end|>", 248050),
        ("<|quad_start|>", 248051),
        ("<|quad_end|>", 248052),
        ("<|vision_start|>", 248053),
        ("<|vision_end|>", 248054),
        ("<|vision_pad|>", 248055),
        ("<|image_pad|>", 248056),
        ("<|video_pad|>", 248057),
        ("<tool_call>", 248058),
        ("</tool_call>", 248059),
        ("<|fim_prefix|>", 248060),
        ("<|fim_middle|>", 248061),
        ("<|fim_suffix|>", 248062),
        ("<|fim_pad|>", 248063),
        ("<|repo_name|>", 248064),
        ("<|file_sep|>", 248065),
        ("<tool_response>", 248066),
        ("</tool_response>", 248067),
        ("<think>", 248068),
        ("</think>", 248069),
        ("<|audio_start|>", 248070),
        ("<|audio_end|>", 248071),
        ("<tts_pad>", 248072),
        ("<tts_text_bos>", 248073),
        ("<tts_text_eod>", 248074),
        ("<tts_text_bos_single>", 248075),
        ("<|audio_pad|>", 248076),
    ]);
    // qwen2's scanner is exact here too: it only commits on chars whose class
    // it is certain of, none of which are combining marks, and defers the rest
    // to this pattern.
    CoreBpe::new(
        encoder,
        special,
        QWEN3_5_PATTERN,
        FastPath::Qwen2,
        WhitespaceRules::NewlineFirst,
    )
}
