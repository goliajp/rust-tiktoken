use crate::CoreBpe;
use crate::pretokenize::PreTokenizer;

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

/// The ids of `text` merged piece by piece, past every shared table.
fn reference_ids(enc: &CoreBpe, text: &str) -> Vec<u32> {
    let mut want = vec![];
    let mut pos = 0;
    while let Some((s, e)) = enc.pre_tokenizer().next_match(text, pos) {
        crate::merge::bpe_encode(&text.as_bytes()[s..e], enc.vocab(), &mut want);
        pos = e;
    }
    want
}

/// Text that no other test makes: clauses of random chars from scripts that
/// merge from atoms and wall chars, Latin words, chars that every
/// vocabulary splits into byte tokens, and text that looks like special
/// tokens, so most of its pieces are met for the first time.
fn fresh_text(seed: u64, bytes: usize) -> String {
    const BLOCKS: &[(u32, u32)] = &[
        (0x4E00, 0x9FA5),
        (0x3040, 0x30FF),
        (0xAC00, 0xD7A3),
        (0x0600, 0x06FF),
        (0x0400, 0x04FF),
        (0xFF01, 0xFF65),
        (0x1F300, 0x1F64F),
        (0x80, 0x24F),
        (0x10FF00, 0x10FFFD),
        (0xE0001, 0xE007F),
    ];
    const SPECIAL: &[&str] = &[
        "<|endoftext|>",
        "<|im_start|>",
        "<|fim_prefix|>",
        "<|begin_of_text|>",
        "<｜end▁of▁sentence｜>",
        "[INST]",
    ];
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let mut out = String::new();
    while out.len() < bytes {
        match next(8) {
            0..=3 => {
                // a clause of one script, long enough for the long table
                let (lo, hi) = BLOCKS[next(BLOCKS.len() as u64) as usize];
                for _ in 0..1 + next(40) {
                    if let Some(c) = char::from_u32(lo + next(u64::from(hi - lo + 1)) as u32) {
                        out.push(c);
                    }
                }
            }
            4 | 5 => {
                out.push(' ');
                for _ in 0..1 + next(14) {
                    out.push((b'a' + next(26) as u8) as char);
                }
            }
            6 => out.push_str(SPECIAL[next(SPECIAL.len() as u64) as usize]),
            _ => out.push_str([" ", "\n", "，", "。", "、", ". ", "\n\n", "  "][next(8) as usize]),
        }
    }
    out
}

/// Ids equal to the reference merge and ends equal to the summed token
/// lengths, at first meeting and again once the shared tables hold the
/// pieces.
fn check_twice(name: &str, enc: &CoreBpe, text: &str) {
    let want = reference_ids(enc, text);
    for pass in 0..2 {
        let (ids, ends) = enc.encode_with_ends(text);
        assert_eq!(ids, want, "{name} pass {pass}: ids of {:?}", head(text));
        assert_eq!(
            ends,
            summed(enc, &ids),
            "{name} pass {pass}: ends of {:?}",
            head(text)
        );
    }
}

#[test]
fn ends_of_first_met_pieces() {
    for &name in crate::list_encodings() {
        let enc = crate::get_encoding(name).unwrap();
        for seed in 1..60u64 {
            let text = fresh_text(
                seed ^ (name.len() as u64 * 7919),
                1 + (seed as usize * 97) % 3000,
            );
            check_twice(name, enc, &text);
        }
    }
}

// 32 KiB and more go to the worker threads, where each chunk's pieces are
// first met on another thread
#[test]
fn ends_of_first_met_pieces_in_long_texts() {
    for &name in crate::list_encodings() {
        let enc = crate::get_encoding(name).unwrap();
        let text = fresh_text(0xC0FFEE ^ name.len() as u64, 48 * 1024);
        assert!(text.len() >= crate::bpe::PARALLEL_MIN);
        check_twice(name, enc, &text);
    }
}
