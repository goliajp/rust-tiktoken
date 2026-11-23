//! Every position the cut rules accept must split the single-threaded scan of
//! every encoding cleanly: a piece starts there, and the two halves scanned on
//! their own give the same pieces.

use std::sync::LazyLock;

use super::*;
use crate::pretokenize::{PreTokenizer, ScalarPreTokenizer};

fn pieces(pt: &ScalarPreTokenizer, text: &str) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut pos = 0;
    while let Some(r) = pt.next_match(text, pos) {
        out.push(r);
        pos = r.1;
    }
    out
}

/// The texts of every oracle fixture, deduplicated.
static FIXTURE_TEXTS: LazyLock<Vec<String>> = LazyLock::new(|| {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let mut texts = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            let data = std::fs::read_to_string(&path).unwrap();
            let cases: Vec<serde_json::Value> = serde_json::from_str(&data).unwrap();
            texts.extend(cases.iter().map(|c| c["text"].as_str().unwrap().to_owned()));
        }
    }
    texts.sort();
    texts.dedup();
    assert!(texts.len() > 3000, "{} fixture texts", texts.len());
    texts
});

/// Chars that sit on or next to the boundaries of some pattern's rules.
const ALPHABET: &[char] = &[
    'a', 'Z', 'q', 's', 't', 'e', '0', '7', ' ', ' ', ' ', '\n', '\n', '\r', '\t', '\x0b', '\x0c',
    '\'', '.', ',', '/', '-', '{', '\u{a0}', '\u{85}', '\u{2028}', '\u{3000}', '中', '文', '一',
    '龥', '龦', '々', '，', '。', '、', '！', '？', '；', '：', '「', 'の', 'カ', '한', 'ب',
    '\u{64b}', '٣', '\u{301}', 'ǅ', 'ʰ', '²', 'Ⅷ', 'Ａ', '１', '😀', '\u{200d}', 'é', 'Σ',
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn random_texts(seed: u64, count: usize, max_len: usize) -> Vec<String> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            let len = (rng.next() as usize) % max_len;
            (0..len)
                .map(|_| ALPHABET[(rng.next() as usize) % ALPHABET.len()])
                .collect()
        })
        .collect()
}

fn check_cuts(name: &str, pt: &ScalarPreTokenizer, text: &str) -> usize {
    let b = text.as_bytes();
    let whole = pieces(pt, text);
    let mut checked = 0;
    for i in 1..b.len() {
        if !is_cut(b, i) {
            continue;
        }
        let split = whole.partition_point(|&(s, _)| s < i);
        assert!(
            split < whole.len() && whole[split].0 == i,
            "[{name}] cut {i} is not a piece start in {text:?}"
        );
        let head = pieces(pt, &text[..i]);
        let tail: Vec<_> = pieces(pt, &text[i..])
            .into_iter()
            .map(|(s, e)| (s + i, e + i))
            .collect();
        assert_eq!(head, whole[..split], "[{name}] head of cut {i} in {text:?}");
        assert_eq!(tail, whole[split..], "[{name}] tail of cut {i} in {text:?}");
        checked += 1;
    }
    checked
}

fn for_each_encoding(mut f: impl FnMut(&str, &ScalarPreTokenizer)) {
    for &name in crate::list_encodings() {
        f(name, crate::get_encoding(name).unwrap().pre_tokenizer());
    }
}

#[test]
fn cuts_split_fixture_texts_cleanly() {
    for_each_encoding(|name, pt| {
        let checked: usize = FIXTURE_TEXTS.iter().map(|t| check_cuts(name, pt, t)).sum();
        assert!(checked > 1000, "[{name}] only {checked} cuts");
    });
}

#[test]
fn cuts_split_random_texts_cleanly() {
    let texts = random_texts(0x9e37_79b9_7f4a_7c15, 3000, 120);
    for_each_encoding(|name, pt| {
        let checked: usize = texts.iter().map(|t| check_cuts(name, pt, t)).sum();
        assert!(checked > 1000, "[{name}] only {checked} cuts");
    });
}

#[test]
fn every_rule_fires() {
    let b = "ab cd\nx.\nYz中，".as_bytes();
    let hits: Vec<usize> = (1..b.len()).filter(|&i| is_cut(b, i)).collect();
    // after "ab", after "cd", before "x", before "Y", before "，"
    assert_eq!(hits, [2, 5, 6, 9, 14]);
    // a blank char before the newline, or a han char outside the range
    assert!(!is_cut(b" \nA", 2));
    assert!(!is_cut("龦，".as_bytes(), 3));
}

#[test]
fn cuts_cover_the_text() {
    let text = "word ".repeat(10_000);
    let c = cuts(text.as_bytes(), 4096, 8);
    assert_eq!(c.len(), 9);
    assert_eq!((c[0], c[8]), (0, text.len()));
    assert!(c.windows(2).all(|w| w[0] < w[1]));
    assert!(c[1..8].iter().all(|&i| is_cut(text.as_bytes(), i)));
    // nothing to cut at
    let digits = "1".repeat(50_000);
    assert_eq!(cuts(digits.as_bytes(), 4096, 8), [0, 50_000]);
    assert_eq!(cuts(b"", 4096, 8), [0, 0]);
}
