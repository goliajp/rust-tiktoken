use crate::CoreBpe;

/// The ends by summing token lengths, the definition the API documents.
fn summed(enc: &CoreBpe, ids: &[u32]) -> Vec<usize> {
    let mut at = 0;
    ids.iter()
        .map(|&id| {
            at += enc.token_bytes(id).unwrap().len();
            at
        })
        .collect()
}

fn check(name: &str, enc: &CoreBpe, text: &str) {
    let (ids, ends) = enc.encode_with_ends(text);
    assert_eq!(ids, enc.encode(text), "{name}: ids of {:?}", head(text));
    assert_eq!(ends, summed(enc, &ids), "{name}: ends of {:?}", head(text));
    assert_eq!(ends.last().copied().unwrap_or(0), text.len(), "{name}");
}

fn head(text: &str) -> &str {
    let mut n = text.len().min(80);
    while !text.is_char_boundary(n) {
        n -= 1;
    }
    &text[..n]
}

fn corpora() -> Vec<String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/corpus");
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap())
        .collect();
    assert!(out.len() >= 3);
    out.sort();
    out
}

const ALPHABET: &[&str] = &[
    "a",
    "Z",
    "the",
    " ",
    " ",
    "  ",
    "\n",
    "\n\n",
    "\r\n",
    "\t",
    "'s",
    "'LL",
    ".",
    ",",
    "...",
    "/",
    "-",
    "12",
    "345678",
    "\u{a0}",
    "\u{3000}",
    "中",
    "文",
    "，",
    "。",
    "の",
    "한",
    "ب",
    "٣",
    "\u{301}",
    "²",
    "１",
    "😀",
    "é",
    "ʰ",
    "<|endoftext|>",
    "qzxj",
    "Ω",
    "\u{200d}",
    "🇯🇵",
];

fn random_text(mut seed: u64, words: usize) -> String {
    let mut out = String::new();
    for _ in 0..words {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        out.push_str(ALPHABET[(seed % ALPHABET.len() as u64) as usize]);
    }
    out
}

#[test]
fn ends_match_summed_token_lengths() {
    let texts = corpora();
    for &name in crate::list_encodings() {
        let enc = crate::get_encoding(name).unwrap();
        for text in &texts {
            check(name, enc, text);
            // every short length, and long prefixes through the windows'
            // tail copy
            for len in (0..200).chain((200..text.len().min(3000)).step_by(37)) {
                if text.is_char_boundary(len) {
                    check(name, enc, &text[..len]);
                }
            }
        }
        for seed in 1..200u64 {
            check(
                name,
                enc,
                &random_text(seed * 0x9E37_79B9, 1 + seed as usize % 400),
            );
        }
    }
}

// texts that every vocabulary splits into byte tokens inside a character
#[test]
fn ends_inside_a_character() {
    let text = "\u{10FFFD}\u{E0001}ꙮ𓀀".repeat(40);
    for &name in crate::list_encodings() {
        let enc = crate::get_encoding(name).unwrap();
        check(name, enc, &text);
        let (_, ends) = enc.encode_with_ends(&text);
        assert!(ends.iter().any(|&e| !text.is_char_boundary(e)), "{name}");
    }
}

#[test]
fn appends_with_ends_in_each_text() {
    let enc = crate::get_encoding("o200k_base").unwrap();
    let (mut ids, mut ends) = (vec![7], vec![]);
    enc.encode_with_ends_into("hello world", &mut ids, &mut ends);
    enc.encode_with_ends_into("hello world", &mut ids, &mut ends);
    assert_eq!(
        ids[1..],
        [enc.encode("hello world"), enc.encode("hello world")].concat()
    );
    assert_eq!(ends, [5, 11, 5, 11]);
    let (ids, ends) = enc.encode_with_ends("");
    assert!(ids.is_empty() && ends.is_empty());
}

// long texts take the parallel path when threads are available, and the
// chunks' ends must be offsets in the whole text
#[test]
fn long_texts() {
    let texts = corpora();
    let long = texts.concat().repeat(2);
    assert!(long.len() >= 64 * 1024);
    for name in [
        "cl100k_base",
        "o200k_base",
        "p50k_base",
        "deepseek_v3",
        "qwen3_5",
    ] {
        let enc = crate::get_encoding(name).unwrap();
        check(name, enc, &long);
        #[cfg(not(target_arch = "wasm32"))]
        if crate::pool::threads() > 1 {
            // the pool may be busy with another test for a while
            let (mut ids, mut ends) = (vec![], vec![]);
            let ran = (0..10_000).any(|_| {
                std::thread::yield_now();
                super::parallel(&enc.engine(), &long, &mut ids, &mut ends)
            });
            assert!(ran, "{name}");
            assert_eq!(ids, enc.encode(&long), "{name}");
            assert_eq!(ends, summed(enc, &ids), "{name}");
        }
    }
}
