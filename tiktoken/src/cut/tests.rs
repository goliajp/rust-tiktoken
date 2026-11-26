//! Every position the cut rules accept must split the single-threaded scan of
//! every encoding cleanly: a piece starts there, the two halves scanned on
//! their own give the same pieces, and their ids add up to the whole text's.

use std::sync::LazyLock;

use super::*;
use crate::CoreBpe;
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

/// Real text: English, Chinese and Japanese prose, and this crate's Rust
/// source with space and with tab indentation. The prose is public domain:
/// the start of chapter 2 of *Pride and Prejudice*, chapter 1 of *Journey to
/// the West* from Wikisource, and the start of 夏目漱石『吾輩は猫である』 from
/// Aozora Bunko with its ruby removed.
static CORPORA: LazyLock<Vec<(String, String)>> = LazyLock::new(|| {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut out = Vec::new();
    for name in ["en_pride.txt", "zh_xiyouji.md", "ja_wagahai.txt"] {
        let path = format!("{root}/tests/fixtures/corpus/{name}");
        out.push((name.to_owned(), std::fs::read_to_string(path).unwrap()));
    }
    let mut code = String::new();
    for file in [
        "bpe.rs",
        "split.rs",
        "encode_loop.rs",
        "cut/mod.rs",
        "cut/simd.rs",
    ] {
        code.push_str(&std::fs::read_to_string(format!("{root}/src/{file}")).unwrap());
    }
    out.push(("code, tabs".to_owned(), code.replace("    ", "\t")));
    out.push(("code".to_owned(), code));
    out
});

/// Chars that sit on or next to the boundaries of some pattern's rules.
const ALPHABET: &[char] = &[
    'a', 'Z', 'q', 's', 't', 'e', '0', '7', ' ', ' ', ' ', ' ', '\n', '\n', '\r', '\t', '\t',
    '\x0b', '\x0c', '\x01', '\'', '.', ',', '/', '-', '{', ';', ')', '\u{a0}', '\u{85}',
    '\u{2028}', '\u{3000}', '中', '文', '一', '龥', '龦', '々', '〇', '，', '。', '、', '！', '？',
    '；', '：', '「', '」', '『', '』', '（', '）', '・', '゛', '\u{3099}', 'の', 'ん', 'カ', 'ー',
    'ヶ', 'ゝ', '한', 'ب', '\u{64b}', '٣', '\u{301}', 'ǅ', 'ʰ', '²', 'Ⅷ', 'Ａ', '１', '😀',
    '\u{200d}', 'é', 'Σ',
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

/// Cut positions by the scalar rule, every byte tried.
fn all_cuts(b: &[u8], rules: Rules) -> Vec<usize> {
    (1..b.len()).filter(|&i| is_cut(b, i, rules)).collect()
}

/// Each cut on its own: a piece start, and both halves scan as the whole.
fn check_each_cut(name: &str, pt: &ScalarPreTokenizer, text: &str) -> usize {
    let whole = pieces(pt, text);
    let cuts = all_cuts(text.as_bytes(), pt.cut_rules());
    for &i in &cuts {
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
    }
    cuts.len()
}

/// Every cut at once, pieces and ids: the segments between cuts encode to
/// the whole text's ids.
fn check_segments(name: &str, enc: &CoreBpe, text: &str) -> usize {
    let pt = enc.pre_tokenizer();
    let cuts = all_cuts(text.as_bytes(), pt.cut_rules());
    let bounds: Vec<usize> = std::iter::once(0)
        .chain(cuts.iter().copied())
        .chain(std::iter::once(text.len()))
        .collect();
    let mut joined_pieces = Vec::new();
    let mut joined_ids = Vec::new();
    for w in bounds.windows(2) {
        let seg = &text[w[0]..w[1]];
        joined_pieces.extend(
            pieces(pt, seg)
                .into_iter()
                .map(|(s, e)| (s + w[0], e + w[0])),
        );
        joined_ids.extend(enc.encode(seg));
    }
    assert_eq!(joined_pieces, pieces(pt, text), "[{name}] pieces");
    assert_eq!(joined_ids, enc.encode(text), "[{name}] ids");
    cuts.len()
}

fn for_each_encoding(mut f: impl FnMut(&str, &CoreBpe)) {
    for &name in crate::list_encodings() {
        f(name, crate::get_encoding(name).unwrap());
    }
}

/// `text` in pieces of about `len` bytes, cut at char boundaries.
fn windows(text: &str, len: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut s = 0;
    while s < text.len() {
        let mut e = (s + len).min(text.len());
        while !text.is_char_boundary(e) {
            e += 1;
        }
        out.push(&text[s..e]);
        s = e;
    }
    out
}

#[test]
fn cuts_split_fixture_texts_cleanly() {
    for_each_encoding(|name, enc| {
        let pt = enc.pre_tokenizer();
        let checked: usize = FIXTURE_TEXTS
            .iter()
            .map(|t| check_each_cut(name, pt, t))
            .sum();
        assert!(checked > 1000, "[{name}] only {checked} cuts");
    });
}

#[test]
fn cuts_split_random_texts_cleanly() {
    let texts = random_texts(0x9e37_79b9_7f4a_7c15, 4000, 120);
    for_each_encoding(|name, enc| {
        let pt = enc.pre_tokenizer();
        let checked: usize = texts.iter().map(|t| check_each_cut(name, pt, t)).sum();
        assert!(checked > 1000, "[{name}] only {checked} cuts");
    });
}

#[test]
fn cuts_split_real_texts_cleanly() {
    for_each_encoding(|name, enc| {
        let pt = enc.pre_tokenizer();
        for (corpus, text) in CORPORA.iter() {
            let label = format!("{name}, {corpus}");
            let checked: usize = windows(text, 600)
                .iter()
                .map(|w| check_each_cut(&label, pt, w))
                .sum();
            assert!(checked > 100, "[{label}] only {checked} cuts");
        }
    });
}

#[test]
fn segments_between_all_cuts_encode_to_the_whole() {
    let random = random_texts(0x2545_f491_4f6c_dd1d, 300, 2000);
    for_each_encoding(|name, enc| {
        for (corpus, text) in CORPORA.iter() {
            check_segments(&format!("{name}, {corpus}"), enc, text);
        }
        for t in FIXTURE_TEXTS.iter().chain(&random) {
            check_segments(name, enc, t);
        }
    });
}

#[test]
fn every_rule_fires() {
    let rules = Rules::of(FastPath::Cl100k);
    let hits = |s: &str| all_cuts(s.as_bytes(), rules);
    // R1, R2 and R4 after letters and punctuation
    assert_eq!(hits("ab cd\nx.\nYz, w"), [2, 5, 6, 9, 12]);
    // R3: Han and kana before marks, not the middle dot or a han char out of range
    assert_eq!(hits("中，の。カ」"), [3, 9, 15]);
    assert!(hits("・。龦，゛、").is_empty());
    // R4 after a non-ASCII char, not after a non-ASCII blank
    assert_eq!(hits("é x"), [2]);
    assert!(hits("\u{3000} \u{a0}\tx").contains(&6));
    assert!(!hits("\u{3000} \u{a0}\tx").contains(&3));
    // R5: the last blank before a non-blank ASCII char
    assert_eq!(hits("{\n    let"), [5]);
    assert_eq!(hits("\n\t\tx"), [2]);
    // not before a digit for deepseek, nor after a control char
    let deepseek = Rules::of(FastPath::Deepseek);
    assert_eq!(all_cuts(b"=\n  1", deepseek), Vec::<usize>::new());
    assert_eq!(all_cuts(b"=\n  1", rules), [3]);
    assert!(all_cuts(b"\x01 x", rules).is_empty());
}

#[test]
fn vector_scans_find_what_the_rule_finds() {
    let rules = Rules::of(FastPath::O200k);
    let mut texts = random_texts(0x1234_5678_9abc_def1, 400, 300);
    texts.extend(CORPORA.iter().map(|(_, t)| windows(t, 4000)[0].to_owned()));
    // long runs with no candidate, then one at every offset of a block
    for k in 0..70 {
        texts.push(format!("{}a {}", "x".repeat(k), "y".repeat(100)));
    }
    for t in &texts {
        let b = t.as_bytes();
        let cuts = all_cuts(b, rules);
        for i in 0..=b.len() + 1 {
            let want = cuts.iter().copied().find(|&c| c >= i);
            assert_eq!(next(b, i, rules), want, "next from {i} in {t:?}");
            let want = cuts.iter().copied().rfind(|&c| c <= i);
            assert_eq!(prev(b, i, rules), want, "prev to {i} in {t:?}");
        }
        #[cfg(target_arch = "x86_64")]
        for i in 1..b.len() {
            let want = cuts.iter().copied().find(|&c| c >= i);
            let got = simd::next_sse2(b, i, rules)
                .map_or_else(|k| (k..b.len()).find(|&k| is_cut(b, k, rules)), Some);
            assert_eq!(got, want, "sse2 next from {i} in {t:?}");
            let j = i.min(b.len() - 1);
            let want = cuts.iter().copied().rfind(|&c| c <= j);
            let got = simd::prev_sse2(b, j, rules)
                .map_or_else(|k| (1..=k).rev().find(|&k| is_cut(b, k, rules)), Some);
            assert_eq!(got, want, "sse2 prev to {j} in {t:?}");
        }
        for i in 1..b.len() {
            if is_cut(b, i, rules) {
                assert!(is_candidate(b, i), "cut {i} is no candidate in {t:?}");
            }
        }
    }
}

#[test]
fn api_agrees_with_encode() {
    let enc = crate::get_encoding("o200k_base").unwrap();
    let text = &CORPORA[2].1;
    let mut from = 1;
    let mut seen = 0;
    while let Some(b) = enc.next_cut(text, from) {
        if seen % 97 == 0 {
            let mut ids = enc.encode(&text[..b]);
            ids.extend(enc.encode(&text[b..]));
            assert_eq!(ids, enc.encode(text));
            assert_eq!(
                enc.count(&text[..b]) + enc.count(&text[b..]),
                enc.count(text)
            );
        }
        assert_eq!(enc.prev_cut(text, b), Some(b));
        seen += 1;
        from = b + 1;
    }
    assert!(seen > 100, "{seen} cuts");
    assert_eq!(enc.next_cut("", 0), None);
    assert_eq!(enc.prev_cut("a", 5), None);
}
