// run once with: cargo test -p tiktoken --test generate_oracle -- --ignored
// generates tests/fixtures/*.json from current v2 behavior

use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct OracleCase {
    text: String,
    tokens: Vec<u32>,
    count: usize,
}

fn corpus() -> Vec<String> {
    let mut texts: Vec<String> = vec![
        // empty / trivial
        "".into(),
        " ".into(),
        "  ".into(),
        "   ".into(),
        "\n".into(),
        "\t".into(),
        "\r\n".into(),
        "  \n  \n  ".into(),
        // single characters
        "a".into(),
        "Z".into(),
        "0".into(),
        "!".into(),
        "@".into(),
        "#".into(),
        // short english
        "hello".into(),
        "hello world".into(),
        "Hello World".into(),
        "Hello, world!".into(),
        "The quick brown fox jumps over the lazy dog.".into(),
        // whitespace variations
        "  hello".into(),
        "   hello".into(),
        "hello   ".into(),
        "hello\t  world".into(),
        "hello\nworld".into(),
        "hello  \n  world".into(),
        // contractions (important for cl100k/o200k patterns)
        "I'm".into(),
        "don't".into(),
        "they're".into(),
        "we've".into(),
        "she'll".into(),
        "it'd".into(),
        // numbers
        "123".into(),
        "1234567890".into(),
        "3.14159".into(),
        "1,000,000".into(),
        // CJK
        "你好".into(),
        "你好世界".into(),
        "こんにちは".into(),
        "こんにちは世界".into(),
        "안녕하세요".into(),
        "世界你好世界".into(),
        // mixed script
        "Hello 你好 World".into(),
        "Hello 你好 🌍".into(),
        "café résumé naïve über".into(),
        "日本語テスト 🎉".into(),
        "café résumé naïve über 日本語 한국어 العربية".into(),
        // emoji
        "🎉".into(),
        "🚀💡🎯".into(),
        "👨‍👩‍👧‍👦".into(),
        "🇯🇵".into(),
        // code
        "fn main() { println!(\"Hello\"); }".into(),
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)".into(),
        "const x = { key: \"value\", arr: [1, 2, 3] };".into(),
        // special token text (not as special tokens, just as plain text)
        "<|endoftext|>".into(),
        "test<|endoftext|>test".into(),
        "<|fim_prefix|>hello<|fim_suffix|>".into(),
        // punctuation heavy
        "Hello!!! What?? Yes... No---maybe".into(),
        "a@b.com http://example.com/path?q=1&r=2".into(),
        // long repetitive
        "a".repeat(100),
        "hello ".repeat(100),
        "word ".repeat(1000),
        // long mixed
        "The quick brown fox. ".repeat(200),
        "你好世界！".repeat(200),
        // arabic
        "مرحبا بالعالم".into(),
        // korean sentence
        "인공지능 기술이 빠르게 발전하고 있습니다".into(),
        // tabs and mixed whitespace
        "\t\t\thello\t\t\t".into(),
        "line1\nline2\nline3".into(),
        "line1\r\nline2\r\nline3".into(),
        // numbers in different contexts
        "The year is 2024 and pi is 3.14159.".into(),
        "100% of $1,000.00 = $1,000.00".into(),
    ];

    // add some boundary-length texts
    texts.push("x".to_string());
    texts.push("x".repeat(2));
    texts.push("x".repeat(3));
    texts.push("x".repeat(10));
    texts.push("x".repeat(50));
    texts.push("x".repeat(255));
    texts.push("x".repeat(256));
    texts.push("x".repeat(257));

    texts
}

fn generate_for_encoding(name: &str, fixture_dir: &Path) {
    let enc = tiktoken::get_encoding(name).unwrap_or_else(|| panic!("unknown encoding: {name}"));
    let cases: Vec<OracleCase> = corpus()
        .into_iter()
        .map(|text| {
            let tokens = enc.encode(&text);
            let count = enc.count(&text);
            assert_eq!(
                count,
                tokens.len(),
                "count != encode().len() for {name}: {text:?}"
            );
            OracleCase {
                text,
                tokens,
                count,
            }
        })
        .collect();

    let path = fixture_dir.join(format!("{name}.json"));
    let json = serde_json::to_string_pretty(&cases).unwrap();
    std::fs::write(&path, &json).unwrap();
    eprintln!("wrote {} cases to {}", cases.len(), path.display());
}

#[test]
#[ignore] // run manually: cargo test --test generate_oracle -- --ignored
fn generate_oracle_fixtures() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::create_dir_all(&fixture_dir).unwrap();

    for name in [
        "cl100k_base",
        "o200k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    ] {
        generate_for_encoding(name, &fixture_dir);
    }
}
