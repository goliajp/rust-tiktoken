//! Full-corpus differential check against canonical `openai/tiktoken`.
//!
//! The committed fixtures in `tests/fixtures/` are the CI regression guard; this
//! is the wider net used when auditing parity — it runs an arbitrarily large
//! corpus (fuzz included) through both implementations and reports every
//! divergence. Issue #5 was 2,504 divergences over a 10,491-case corpus.
//!
//! Generate the corpus, then run the test:
//!
//! ```text
//! pip install tiktoken
//! python tiktoken/tests/canonical_corpus.py /tmp/canonical.json
//! CANONICAL_JSON=/tmp/canonical.json cargo test -p tiktoken --test canonical_parity -- --ignored --nocapture
//! ```
//!
//! Ignored by default: it needs a corpus file the repo does not ship (the full
//! fuzz corpus is several MB per encoding).

use std::collections::BTreeMap;

#[test]
#[ignore = "needs CANONICAL_JSON; see module docs"]
fn matches_canonical_tiktoken() {
    let path = std::env::var("CANONICAL_JSON")
        .expect("set CANONICAL_JSON to the corpus produced by canonical_corpus.py");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let data: BTreeMap<String, Vec<(String, Vec<u32>)>> =
        serde_json::from_str(&raw).expect("invalid corpus json");

    // How many divergences to print per encoding before going quiet. Raise it
    // via MAX_REPORT when classifying a failure mode rather than spot-checking.
    let max_report: usize = std::env::var("MAX_REPORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let mut total = 0usize;
    let mut failed = 0usize;

    for (enc_name, cases) in &data {
        let Some(enc) = tiktoken::get_encoding(enc_name) else {
            panic!("corpus names an encoding this crate does not have: {enc_name}");
        };
        let mut enc_failed = 0usize;
        for (text, expected) in cases {
            total += 1;
            let got = enc.encode(text);
            if &got != expected {
                failed += 1;
                enc_failed += 1;
                if enc_failed <= max_report {
                    eprintln!("MISMATCH [{enc_name}] text={text:?}");
                    eprintln!("   canonical: {expected:?}");
                    eprintln!("   ours:      {got:?}");
                }
            }
        }
        eprintln!("{enc_name}: {enc_failed} / {} mismatched", cases.len());
    }

    eprintln!("TOTAL: {failed} / {total} mismatched");
    assert_eq!(
        failed, 0,
        "{failed} of {total} cases diverge from canonical"
    );
}
