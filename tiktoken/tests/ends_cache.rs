//! Token ends of long pieces come from the shared merge table, which keeps
//! them only from the first encode with ends on. Whatever the table held
//! before, and after it is cleared, the ends must be the summed token lengths.
//! One test in its own process: it clears the shared table.

fn summed(enc: &tiktoken::CoreBpe, ids: &[u32]) -> Vec<usize> {
    let mut at = 0;
    ids.iter()
        .map(|&id| {
            at += enc.token_bytes(id).unwrap().len();
            at
        })
        .collect()
}

#[test]
fn ends_survive_every_table_state() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/corpus");
    let texts: Vec<String> = ["zh_xiyouji.md", "ja_wagahai.txt", "en_pride.txt"]
        .iter()
        .map(|f| std::fs::read_to_string(format!("{dir}/{f}")).unwrap())
        .collect();
    for name in ["o200k_base", "cl100k_base", "p50k_base", "qwen3_5"] {
        let enc = tiktoken::get_encoding(name).unwrap();
        for round in 0..4 {
            for text in &texts {
                // plain encodes first fill the table without ends
                if round % 2 == 0 {
                    let _ = enc.encode(text);
                }
                for _ in 0..2 {
                    let (ids, ends) = enc.encode_with_ends(text);
                    assert_eq!(ids, enc.encode(text), "{name} round {round}");
                    assert_eq!(ends, summed(enc, &ids), "{name} round {round}");
                }
            }
            if round == 1 {
                // SAFETY: this is the only test in this process
                unsafe { enc.__reset_shared_cache() };
            }
        }
    }
}
