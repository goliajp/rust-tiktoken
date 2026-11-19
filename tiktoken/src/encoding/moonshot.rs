//! Moonshot Kimi encodings: kimi_k2 and kimi_k3.

use super::*;

/// The Kimi K2 / K3 shared base vocabulary is 163,584 tokens (byte-identical
/// `tiktoken.model` across both generations); each generation defines its own
/// special-token ids in the 163584..163839 reserved range.
///
/// Construct the kimi_k2 encoding (Kimi K2 / K2.5 / K2.6, Moonshot).
#[cfg(feature = "vocab-kimi_k2")]
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
#[cfg(feature = "vocab-kimi_k2")]
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
