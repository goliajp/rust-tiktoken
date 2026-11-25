// The parallel path must give exactly the single-threaded result. Short texts
// go through it with small chunks (`__encode_chunks`), so every encoding's
// fixtures exercise many cuts; a chunk size larger than the text is the
// single-threaded reference.

use std::sync::LazyLock;

const SERIAL: usize = usize::MAX;
const PARALLEL_MIN: usize = 32 * 1024;

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
    texts
});

const ALPHABET: &[char] = &[
    'a', 'Z', 's', 't', '0', '7', ' ', ' ', ' ', '\n', '\n', '\r', '\t', '\'', '.', ',', '/', '-',
    '\u{a0}', '\u{3000}', '中', '文', '，', '。', '！', 'の', '한', 'ب', '٣', '\u{301}', '²', '１',
    '😀', 'é',
];

fn random_text(mut seed: u64, len: usize) -> String {
    let mut out = String::with_capacity(len * 2);
    while out.len() < len {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // words of 1-8 chars keep cut candidates frequent
        let word = 1 + (seed % 8) as usize;
        for k in 0..word {
            out.push(ALPHABET[((seed >> (8 + 5 * k)) as usize) % ALPHABET.len()]);
        }
        out.push(if seed & 0x10_0000 == 0 { ' ' } else { 'x' });
    }
    out
}

fn check(name: &str, text: &str, chunk: usize) {
    let enc = tiktoken::get_encoding(name).unwrap();
    let want = enc.__encode_chunks(text, SERIAL);
    assert_eq!(
        enc.__encode_chunks(text, chunk),
        want,
        "[{name}] encode, chunk {chunk}, text {text:?}"
    );
    assert_eq!(
        enc.__count_chunks(text, chunk),
        want.len(),
        "[{name}] count, chunk {chunk}"
    );
}

#[test]
fn fixtures_match_serial_for_every_encoding() {
    let all = FIXTURE_TEXTS.concat();
    for &name in tiktoken::list_encodings() {
        for text in FIXTURE_TEXTS.iter() {
            check(name, text, 16);
        }
        for chunk in [64, 1000, 8192] {
            check(name, &all, chunk);
        }
    }
}

#[test]
fn random_texts_match_serial_for_every_encoding() {
    for &name in tiktoken::list_encodings() {
        for seed in 1..20u64 {
            let text = random_text(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15), 3000);
            check(name, &text, 37);
        }
    }
}

#[test]
fn long_texts_take_the_parallel_path() {
    let text = random_text(7, 300_000);
    assert!(text.len() >= PARALLEL_MIN);
    for &name in tiktoken::list_encodings() {
        let enc = tiktoken::get_encoding(name).unwrap();
        let want = enc.__encode_chunks(&text, SERIAL);
        assert_eq!(enc.encode(&text), want, "[{name}] encode");
        assert_eq!(enc.encode_parallel(&text), want, "[{name}] encode_parallel");
        assert_eq!(enc.count(&text), want.len(), "[{name}] count");
    }
}

#[test]
fn special_tokens_around_long_segments() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let a = random_text(11, 100_000);
    let b = random_text(12, 70_000);
    let text = format!("{a}<|endoftext|>{b}");
    let mut want = enc.__encode_chunks(&a, SERIAL);
    want.push(100257);
    want.extend(enc.__encode_chunks(&b, SERIAL));
    assert_eq!(enc.encode_with_special_tokens(&text), want);
    assert_eq!(enc.count_with_special_tokens(&text), want.len());
}

#[test]
fn concurrent_callers_get_their_own_results() {
    // only one of them gets the pool at a time; the rest run on their thread
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let texts: Vec<String> = (0..8).map(|i| random_text(100 + i, 120_000)).collect();
    let want: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| enc.__encode_chunks(t, SERIAL))
        .collect();
    std::thread::scope(|s| {
        for (text, want) in texts.iter().zip(&want) {
            s.spawn(move || {
                for _ in 0..5 {
                    assert_eq!(&enc.encode(text), want);
                }
            });
        }
    });
}

#[test]
fn text_without_cut_points() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "1".repeat(PARALLEL_MIN + 1000);
    assert_eq!(enc.encode(&text), enc.__encode_chunks(&text, SERIAL));
}

#[test]
fn empty_and_short_texts() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(enc.encode_parallel(""), Vec::<u32>::new());
    assert_eq!(enc.__encode_chunks("", 16), Vec::<u32>::new());
    assert_eq!(enc.encode_parallel("short"), enc.encode("short"));
}

#[test]
fn encode_into_appends_on_both_paths() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let long = "The quick brown fox, 狐狸 jumps. ".repeat(3000);
    assert!(long.len() >= PARALLEL_MIN);
    let mut ids = vec![7, 8, 9];
    let mut want = ids.clone();
    for text in ["hello world", long.as_str(), "", "a"] {
        enc.encode_into(text, &mut ids);
        want.extend(enc.encode(text));
        assert_eq!(ids, want);
    }
}
