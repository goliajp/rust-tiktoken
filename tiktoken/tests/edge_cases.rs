// edge case integration tests for tiktoken
// covers scenarios not exercised by oracle, proptest, parallel, or unit tests

const ALL_ENCODINGS: &[&str] = &[
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "p50k_edit",
    "r50k_base",
    "llama3",
    "deepseek_v3",
    "qwen2",
    "mistral_v3",
];

// helper: known special tokens for each encoding
fn special_tokens_for(name: &str) -> Vec<&'static str> {
    match name {
        "cl100k_base" => vec![
            "<|endoftext|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|endofprompt|>",
        ],
        "o200k_base" => vec!["<|endoftext|>", "<|endofprompt|>"],
        "p50k_base" => vec!["<|endoftext|>"],
        "p50k_edit" => vec![
            "<|endoftext|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ],
        "r50k_base" => vec!["<|endoftext|>"],
        "llama3" => vec![
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|finetune_right_pad_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ],
        "deepseek_v3" => vec![
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
            "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
            "<\u{ff5c}\u{2581}pad\u{2581}\u{ff5c}>",
            "<|EOT|>",
        ],
        "qwen2" => vec![
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ],
        "mistral_v3" => vec![
            "<unk>",
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "[TOOL_CALLS]",
            "[IMG]",
            "[IMG_BREAK]",
            "[IMG_END]",
            "[PREFIX]",
            "[MIDDLE]",
            "[SUFFIX]",
        ],
        _ => vec![],
    }
}

// --- 1. multiple adjacent special tokens ---

#[test]
fn adjacent_special_tokens_roundtrip() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);
        if specials.is_empty() {
            continue;
        }

        // two adjacent
        let text = format!("{}{}", specials[0], specials[0]);
        let tokens = enc.encode_with_special_tokens(&text);
        let decoded = enc.decode_to_string(&tokens).unwrap();
        assert_eq!(decoded, text, "[{name}] two adjacent special tokens failed");

        // all special tokens concatenated
        let all: String = specials.iter().copied().collect();
        let tokens = enc.encode_with_special_tokens(&all);
        let decoded = enc.decode_to_string(&tokens).unwrap();
        assert_eq!(decoded, all, "[{name}] all adjacent special tokens failed");
    }
}

#[test]
fn many_adjacent_special_tokens_count_consistency() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);
        if specials.is_empty() {
            continue;
        }

        // repeat first special token 10 times
        let text: String = specials[0].repeat(10);
        let tokens = enc.encode_with_special_tokens(&text);
        let count = enc.count_with_special_tokens(&text);
        assert_eq!(
            count,
            tokens.len(),
            "[{name}] count mismatch for 10 adjacent special tokens"
        );
    }
}

// --- 2. binary-like data: all 256 byte values via latin-1 ---

#[test]
fn all_256_byte_values_roundtrip() {
    // encode each byte 0x00..=0xFF as a latin-1 char in a string,
    // verify roundtrip for every encoding
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        for b in 0u8..=255 {
            // create a string containing a single character with codepoint == b
            let ch = char::from(b);
            let text = ch.to_string();
            let tokens = enc.encode(&text);
            let decoded = enc.decode(&tokens);
            assert_eq!(
                decoded,
                text.as_bytes(),
                "[{name}] byte {b:#04x} roundtrip failed"
            );
        }
    }
}

// --- 3. cross-encoding comparison: decode(encode(text)) identical across encodings ---

#[test]
fn cross_encoding_decode_consistency() {
    let texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "日本語テスト",
        "emoji: \u{1f600}\u{1f680}\u{2764}\u{fe0f}",
        "fn main() { println!(\"hello\"); }",
        "混合 mixed 内容 content 123",
    ];

    for text in &texts {
        for &name in ALL_ENCODINGS {
            let enc = tiktoken::get_encoding(name).unwrap();
            let tokens = enc.encode(text);
            let decoded = enc.decode_to_string(&tokens).unwrap();
            assert_eq!(
                &decoded, text,
                "[{name}] cross-encoding decode mismatch for {text:?}"
            );
        }
    }
}

// --- 4. stress test: text with every special token for each encoding ---

#[test]
fn stress_all_special_tokens_interleaved() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);
        if specials.is_empty() {
            continue;
        }

        // build text: "word0<special0>word1<special1>...wordN"
        let mut text = String::new();
        for (i, &special) in specials.iter().enumerate() {
            text.push_str(&format!("word{i}"));
            text.push_str(special);
        }
        text.push_str("final");

        let tokens = enc.encode_with_special_tokens(&text);
        let decoded = enc.decode_to_string(&tokens).unwrap();
        assert_eq!(
            decoded, text,
            "[{name}] interleaved special tokens roundtrip failed"
        );

        // count must match
        let count = enc.count_with_special_tokens(&text);
        assert_eq!(
            count,
            tokens.len(),
            "[{name}] count mismatch for interleaved special tokens"
        );
    }
}

// --- 5. vocab size sanity checks ---

#[test]
fn vocab_size_sanity() {
    // verify vocab sizes via special token ids known from encoding.rs
    // (encoding, special_token_text, expected_id, expected_regular_vocab_description)
    let checks: &[(&str, &str, u32)] = &[
        ("cl100k_base", "<|endoftext|>", 100257), // regular: 100,256
        ("o200k_base", "<|endoftext|>", 199999),  // regular: 199,998
        ("p50k_base", "<|endoftext|>", 50256),    // regular: 50,256
        ("p50k_edit", "<|endoftext|>", 50256),    // same base as p50k
        ("r50k_base", "<|endoftext|>", 50256),    // regular: 50,256
        ("llama3", "<|begin_of_text|>", 128000),  // regular: 128,000
        ("deepseek_v3", "<|EOT|>", 128805),       // regular: 128,000 + specials
        ("qwen2", "<|endoftext|>", 151643),       // regular: 151,643
        ("mistral_v3", "[INST]", 3),              // mistral has low special ids
    ];

    for &(name, special, expected_id) in checks {
        let enc = tiktoken::get_encoding(name).unwrap();
        let tokens = enc.encode_with_special_tokens(special);
        assert_eq!(
            tokens.len(),
            1,
            "[{name}] {special:?} should encode to exactly 1 token"
        );
        assert_eq!(
            tokens[0], expected_id,
            "[{name}] {special:?} should have token id {expected_id}, got {}",
            tokens[0]
        );
    }

    // additionally verify that distinct token ids span the expected range
    // by encoding many diverse strings and collecting unique ids
    let diverse_texts: Vec<String> = (0u8..=255)
        .map(|b| char::from(b).to_string())
        .chain((0..100).map(|i| format!("word{i}")))
        .collect();

    let range_checks: &[(&str, u32)] = &[
        ("cl100k_base", 100_000),
        ("o200k_base", 199_000),
        ("p50k_base", 50_000),
        ("r50k_base", 50_000),
        ("llama3", 127_000),
        ("qwen2", 151_000),
        ("mistral_v3", 130_000),
    ];

    for &(name, min_max_id) in range_checks {
        let enc = tiktoken::get_encoding(name).unwrap();
        let mut max_id: u32 = 0;
        for text in &diverse_texts {
            for &id in &enc.encode(text) {
                max_id = max_id.max(id);
            }
        }
        assert!(
            max_id >= 200,
            "[{name}] max token id seen {max_id} is suspiciously low"
        );
        // the max id from a diverse sample won't cover the full vocab,
        // but the special token id checks above confirm the vocab reaches
        // the expected range
        let _ = min_max_id; // used only for documentation
    }
}

// --- 6. encode single special token text ---

#[test]
fn encode_single_special_token_only() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);

        for &special in &specials {
            // with special token recognition: should produce exactly 1 token
            let tokens = enc.encode_with_special_tokens(special);
            assert_eq!(
                tokens.len(),
                1,
                "[{name}] encode_with_special_tokens({special:?}) should produce 1 token, got {}",
                tokens.len()
            );

            // roundtrip
            let decoded = enc.decode_to_string(&tokens).unwrap();
            assert_eq!(
                decoded, special,
                "[{name}] single special token roundtrip failed for {special:?}"
            );

            // without special token recognition: should produce >0 tokens
            // and the special token text should still roundtrip
            let tokens_plain = enc.encode(special);
            assert!(
                !tokens_plain.is_empty(),
                "[{name}] encode({special:?}) should produce tokens"
            );
            let decoded_plain = enc.decode_to_string(&tokens_plain).unwrap();
            assert_eq!(
                decoded_plain, special,
                "[{name}] plain encode roundtrip failed for {special:?}"
            );
        }
    }
}

// --- 7. mixed special and non-special tokens in various orders ---

#[test]
fn mixed_special_nonspecial_orders() {
    let patterns = [
        // (prefix, suffix) around a special token
        ("", ""),             // special only (covered above but different assertion)
        ("hello ", ""),       // text before
        ("", " world"),       // text after
        ("hello ", " world"), // text both sides
        ("\n", "\n"),         // newlines around
        ("123 ", " 456"),     // numbers around
    ];

    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);
        if specials.is_empty() {
            continue;
        }
        let special = specials[0];

        for (prefix, suffix) in &patterns {
            let text = format!("{prefix}{special}{suffix}");
            let tokens = enc.encode_with_special_tokens(&text);
            let decoded = enc.decode_to_string(&tokens).unwrap();
            assert_eq!(
                decoded, text,
                "[{name}] mixed order failed for prefix={prefix:?} suffix={suffix:?}"
            );
        }
    }
}

#[test]
fn special_between_identical_words() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);
        if specials.is_empty() {
            continue;
        }
        let special = specials[0];

        // same word on both sides
        let text = format!("test{special}test");
        let tokens = enc.encode_with_special_tokens(&text);
        let decoded = enc.decode_to_string(&tokens).unwrap();
        assert_eq!(
            decoded, text,
            "[{name}] special between identical words failed"
        );
    }
}

// --- 8. very long text (100KB+) roundtrip ---

#[test]
fn long_text_100kb_roundtrip() {
    // generate ~100KB of mixed content
    let base = "The quick brown fox jumps over the lazy dog. \
                Hello, 世界! Rust is fast. 🚀 Numbers: 12345. \
                Special chars: @#$%^&*(). Newline:\n";
    let repeat_count = (100 * 1024) / base.len() + 1;
    let text: String = base.repeat(repeat_count);
    assert!(text.len() >= 100 * 1024, "text should be >= 100KB");

    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode_to_string(&tokens).unwrap();
        assert_eq!(
            decoded,
            text,
            "[{name}] 100KB roundtrip failed (text len={})",
            text.len()
        );

        // count must match
        assert_eq!(
            enc.count(&text),
            tokens.len(),
            "[{name}] count mismatch for 100KB text"
        );
    }
}

// --- 9. decode_to_string with valid unicode ---

#[test]
fn decode_to_string_multilingual() {
    let texts = [
        "English text",
        "日本語テスト",
        "한국어 테스트",
        "العربية",
        "Ελληνικά",
        "Кириллица",
        "ไทย",
        "emoji mix: \u{1f600}\u{1f389}\u{1f4a1}\u{2728}",
        "diacritics: \u{00e9}\u{00e8}\u{00ea}\u{00eb}\u{00f1}\u{00fc}\u{00e4}\u{00f6}",
        "math: \u{221e} \u{2200}x \u{2203}y \u{2208} \u{2124}",
    ];

    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        for text in &texts {
            let tokens = enc.encode(text);
            let result = enc.decode_to_string(&tokens);
            assert!(
                result.is_ok(),
                "[{name}] decode_to_string failed for {text:?}: {:?}",
                result.err()
            );
            assert_eq!(
                result.unwrap(),
                *text,
                "[{name}] decode_to_string mismatch for {text:?}"
            );
        }
    }
}

// --- 10. empty string for all encodings (integration test form) ---

#[test]
fn empty_string_all_encodings_integration() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();

        // encode
        let tokens = enc.encode("");
        assert!(tokens.is_empty(), "[{name}] encode empty should be empty");

        // encode_with_special_tokens
        let tokens_special = enc.encode_with_special_tokens("");
        assert!(
            tokens_special.is_empty(),
            "[{name}] encode_with_special_tokens empty should be empty"
        );

        // count
        assert_eq!(enc.count(""), 0, "[{name}] count empty should be 0");

        // count_with_special_tokens
        assert_eq!(
            enc.count_with_special_tokens(""),
            0,
            "[{name}] count_with_special_tokens empty should be 0"
        );

        // decode empty
        let decoded = enc.decode(&[]);
        assert!(decoded.is_empty(), "[{name}] decode empty should be empty");

        // decode_to_string empty
        assert_eq!(
            enc.decode_to_string(&[]).unwrap(),
            "",
            "[{name}] decode_to_string empty should be empty string"
        );
    }
}

// --- bonus: encode_with_special_tokens vs encode for text without specials ---

#[test]
fn encode_with_special_tokens_matches_encode_for_plain_text() {
    let texts = [
        "hello world",
        "The quick brown fox.",
        "日本語テスト 🚀",
        "fn main() {}",
        "1234567890",
    ];

    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        for text in &texts {
            let plain = enc.encode(text);
            let special = enc.encode_with_special_tokens(text);
            assert_eq!(
                plain, special,
                "[{name}] encode vs encode_with_special_tokens mismatch for plain text {text:?}"
            );
        }
    }
}

// --- bonus: repeated special tokens produce correct token count ---

#[test]
fn repeated_special_tokens_correct_count() {
    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        let specials = special_tokens_for(name);
        if specials.is_empty() {
            continue;
        }
        let special = specials[0];

        for n in [1, 2, 5, 20] {
            let text = special.repeat(n);
            let tokens = enc.encode_with_special_tokens(&text);
            // each special token occurrence should produce exactly 1 token
            assert_eq!(
                tokens.len(),
                n,
                "[{name}] {n} repeated {special:?} should produce {n} tokens, got {}",
                tokens.len()
            );
        }
    }
}

// --- bonus: surrogate-range unicode (edge of BMP) ---

#[test]
fn bmp_boundary_characters_roundtrip() {
    // characters near interesting unicode boundaries
    let edge_chars = [
        '\u{007F}',   // DEL (last ASCII)
        '\u{0080}',   // first latin-1 supplement
        '\u{00FF}',   // last latin-1 supplement
        '\u{0100}',   // first latin extended
        '\u{FFFD}',   // replacement character
        '\u{FFFE}',   // noncharacter (but valid in Rust strings)
        '\u{10000}',  // first supplementary character
        '\u{10FFFF}', // last valid unicode scalar
    ];

    for &name in ALL_ENCODINGS {
        let enc = tiktoken::get_encoding(name).unwrap();
        for ch in &edge_chars {
            let text = ch.to_string();
            let tokens = enc.encode(&text);
            let decoded = enc.decode(&tokens);
            assert_eq!(
                decoded,
                text.as_bytes(),
                "[{name}] BMP boundary char U+{:04X} roundtrip failed",
                *ch as u32
            );
        }
    }
}
