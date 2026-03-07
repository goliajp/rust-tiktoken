// property-based tests: verify invariants hold for 100k random inputs

use proptest::prelude::*;

// strategy: mix of ascii, unicode, whitespace, emoji, CJK, control chars
fn text_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // pure ascii
        "[a-zA-Z0-9 .,!?'\\-]{0,500}",
        // unicode mix: latin, CJK, emoji, whitespace
        "[ -~\u{00A0}-\u{00FF}\u{4E00}-\u{9FFF}\u{3000}-\u{303F}\u{1F600}-\u{1F64F}\n\r\t]{0,300}",
        // arbitrary valid utf-8
        "\\PC{0,200}",
        // whitespace-heavy
        "[ \t\n\r\u{3000}]{0,100}",
        // code-like
        "(fn |def |class |import |const |let |var )[a-z_]{1,20}\\([a-z, ]{0,30}\\)\\s*\\{?\\s*\n",
    ]
}

const ENCODINGS: &[&str] = &[
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(25000))]

    #[test]
    fn roundtrip_cl100k(text in text_strategy()) {
        let enc = tiktoken::get_encoding("cl100k_base").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_o200k(text in text_strategy()) {
        let enc = tiktoken::get_encoding("o200k_base").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_p50k(text in text_strategy()) {
        let enc = tiktoken::get_encoding("p50k_base").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_p50k_edit(text in text_strategy()) {
        let enc = tiktoken::get_encoding("p50k_edit").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_r50k(text in text_strategy()) {
        let enc = tiktoken::get_encoding("r50k_base").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn count_equals_encode_len(text in text_strategy()) {
        for &name in ENCODINGS {
            let enc = tiktoken::get_encoding(name).unwrap();
            let tokens = enc.encode(&text);
            let count = enc.count(&text);
            prop_assert_eq!(count, tokens.len(), "count mismatch for {}", name);
        }
    }

    #[test]
    fn roundtrip_llama3(text in text_strategy()) {
        let enc = tiktoken::get_encoding("llama3").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_deepseek_v3(text in text_strategy()) {
        let enc = tiktoken::get_encoding("deepseek_v3").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_qwen2(text in text_strategy()) {
        let enc = tiktoken::get_encoding("qwen2").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn roundtrip_mistral_v3(text in text_strategy()) {
        let enc = tiktoken::get_encoding("mistral_v3").unwrap();
        let tokens = enc.encode(&text);
        let decoded = enc.decode(&tokens);
        prop_assert_eq!(&decoded[..], text.as_bytes(), "roundtrip failed");
    }

    #[test]
    fn encode_with_special_tokens_roundtrip(text in text_strategy()) {
        // encode_with_special_tokens should also roundtrip correctly
        // (special token literals rarely appear in random text, but the
        // code path through special_regex must still be exercised)
        for &name in ENCODINGS {
            let enc = tiktoken::get_encoding(name).unwrap();
            let tokens = enc.encode_with_special_tokens(&text);
            let decoded = enc.decode(&tokens);
            prop_assert_eq!(
                &decoded[..],
                text.as_bytes(),
                "encode_with_special_tokens roundtrip failed for {}",
                name
            );
        }
    }

    #[test]
    fn count_with_special_tokens_equals_encode_with_special_tokens_len(text in text_strategy()) {
        for &name in ENCODINGS {
            let enc = tiktoken::get_encoding(name).unwrap();
            let tokens = enc.encode_with_special_tokens(&text);
            let count = enc.count_with_special_tokens(&text);
            prop_assert_eq!(
                count,
                tokens.len(),
                "count_with_special_tokens mismatch for {}",
                name
            );
        }
    }
}
