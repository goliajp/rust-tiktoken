#![cfg(feature = "parallel")]

#[test]
fn parallel_matches_sequential_cl100k() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "Hello world\nThis is a test\nWith multiple lines\n".repeat(100);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_matches_sequential_o200k() {
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let text = "Hello world! 你好世界 🚀 test 123\n".repeat(200);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_short_text_fallback() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(enc.encode_parallel("short"), enc.encode("short"));
}

#[test]
fn parallel_empty_text() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(enc.encode_parallel(""), enc.encode(""));
}

#[test]
fn parallel_all_encodings() {
    let text = "The quick brown fox jumps over the lazy dog. 你好世界！\n".repeat(100);
    for name in [
        "cl100k_base",
        "o200k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
        "llama3",
        "deepseek_v3",
        "qwen2",
        "mistral_v3",
    ] {
        let enc = tiktoken::get_encoding(name).unwrap();
        assert_eq!(
            enc.encode_parallel(&text),
            enc.encode(&text),
            "parallel mismatch for {name}"
        );
    }
}

#[test]
fn parallel_unicode_heavy() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "café résumé naïve über 日本語 한국어 العربية 🎉🚀💡 ".repeat(100);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_exactly_4096_bytes() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "x".repeat(4096);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_just_under_threshold() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "x".repeat(4095);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_just_over_threshold() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "x".repeat(4097);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_single_large_word() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "a".repeat(5000);
    assert_eq!(enc.encode_parallel(&text), enc.encode(&text));
}

#[test]
fn parallel_deterministic() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "Hello world! 你好 🚀\n".repeat(200);
    let r1 = enc.encode_parallel(&text);
    let r2 = enc.encode_parallel(&text);
    assert_eq!(r1, r2);
}
