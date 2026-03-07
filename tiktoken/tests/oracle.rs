// oracle tests: verify v3 behavior matches v2 snapshots
// run with: cargo test -p tiktoken --test oracle

use serde::Deserialize;

#[derive(Deserialize)]
struct OracleCase {
    text: String,
    tokens: Vec<u32>,
    count: usize,
}

fn load_cases(encoding: &str) -> Vec<OracleCase> {
    let path = format!(
        "{}/tests/fixtures/{encoding}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing fixture: {path}"));
    serde_json::from_str(&data).unwrap_or_else(|_| panic!("invalid fixture: {path}"))
}

fn verify_encoding(name: &str) {
    let enc = tiktoken::get_encoding(name).unwrap_or_else(|| panic!("unknown encoding: {name}"));
    let cases = load_cases(name);

    for (i, case) in cases.iter().enumerate() {
        let tokens = enc.encode(&case.text);
        assert_eq!(
            tokens, case.tokens,
            "[{name}] case {i} encode mismatch for text: {:?}",
            case.text
        );

        let count = enc.count(&case.text);
        assert_eq!(
            count, case.count,
            "[{name}] case {i} count mismatch for text: {:?}",
            case.text
        );

        // roundtrip: decode(encode(text)) == text bytes
        let decoded = enc.decode(&tokens);
        assert_eq!(
            decoded,
            case.text.as_bytes(),
            "[{name}] case {i} roundtrip failed for text: {:?}",
            case.text
        );
    }
}

#[test]
fn oracle_cl100k_base() {
    verify_encoding("cl100k_base");
}

#[test]
fn oracle_o200k_base() {
    verify_encoding("o200k_base");
}

#[test]
fn oracle_p50k_base() {
    verify_encoding("p50k_base");
}

#[test]
fn oracle_p50k_edit() {
    verify_encoding("p50k_edit");
}

#[test]
fn oracle_r50k_base() {
    verify_encoding("r50k_base");
}
