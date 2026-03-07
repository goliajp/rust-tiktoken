// oracle tests for HuggingFace-sourced encodings
// verifies roundtrip and count consistency; compares token ids with HF reference

use serde::Deserialize;

#[derive(Deserialize)]
struct HfOracleCase {
    text: String,
    tokens: Vec<u32>,
    count: usize,
}

fn load_cases(name: &str) -> Vec<HfOracleCase> {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let data = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing fixture: {path}"));
    serde_json::from_str(&data).unwrap_or_else(|_| panic!("invalid fixture: {path}"))
}

fn verify_roundtrip_and_count(name: &str) {
    let enc = tiktoken::get_encoding(name).unwrap_or_else(|| panic!("unknown encoding: {name}"));
    let cases = load_cases(name);

    for (i, case) in cases.iter().enumerate() {
        let tokens = enc.encode(&case.text);
        let count = enc.count(&case.text);

        // count must equal encode length
        assert_eq!(
            count,
            tokens.len(),
            "[{name}] case {i} count({}) != encode().len({})",
            count,
            tokens.len()
        );

        // roundtrip: decode(encode(text)) == text bytes
        let decoded = enc.decode(&tokens);
        assert_eq!(
            decoded,
            case.text.as_bytes(),
            "[{name}] case {i} roundtrip failed for text: {:?}",
            case.text
        );

        // token ids must match HF reference
        assert_eq!(
            tokens, case.tokens,
            "[{name}] case {i} token mismatch for text: {:?}\n  rust: {:?}\n  hf:   {:?}",
            case.text, tokens, case.tokens
        );

        // count must match HF reference
        assert_eq!(
            count, case.count,
            "[{name}] case {i} count mismatch for text: {:?}",
            case.text
        );
    }
}

#[test]
fn oracle_llama3() {
    verify_roundtrip_and_count("llama3");
}

#[test]
fn oracle_deepseek_v3() {
    verify_roundtrip_and_count("deepseek_v3");
}

#[test]
fn oracle_qwen2() {
    verify_roundtrip_and_count("qwen2");
}

#[test]
fn oracle_mistral_v3() {
    verify_roundtrip_and_count("mistral_v3");
}
