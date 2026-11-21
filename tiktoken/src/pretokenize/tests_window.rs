//! The window rules against the scalar scanners (which `tests.rs` checks
//! against the regex reference), for every family that has window rules.

use super::tests::{FAMILIES, Family, check, pieces};
use super::window::{CaseRules, Cl100kRules};
use super::windowed::{PieceSink, split};
use super::*;

struct Collect<'a> {
    text: &'a [u8],
    pos: usize,
    out: Vec<(usize, usize)>,
}

impl PieceSink for Collect<'_> {
    fn window(&mut self) {}

    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        assert!(end > start, "empty piece");
        assert!(buf.len() >= start + crate::vocab::KEY_READ);
        let len = end - start;
        assert_eq!(&buf[start..end], &self.text[self.pos..self.pos + len]);
        self.out.push((self.pos, self.pos + len));
        self.pos += len;
    }
}

fn window_pieces(pt: &ScalarPreTokenizer, text: &str) -> Vec<(usize, usize)> {
    let mut c = Collect {
        text: text.as_bytes(),
        pos: 0,
        out: Vec::new(),
    };
    match pt.window_kind().unwrap() {
        WindowKind::Cl100k => split::<Cl100kRules<3, false>, _>(pt, text, &mut c),
        WindowKind::Qwen2 => split::<Cl100kRules<1, false>, _>(pt, text, &mut c),
        WindowKind::Qwen35 => split::<Cl100kRules<1, true>, _>(pt, text, &mut c),
        WindowKind::O200k => split::<CaseRules<true, 3>, _>(pt, text, &mut c),
        WindowKind::Tekken => split::<CaseRules<false, 1>, _>(pt, text, &mut c),
    }
    assert_eq!(c.pos, text.len());
    c.out
}

fn windowed() -> impl Iterator<Item = &'static Family> {
    FAMILIES.iter().filter(|f| f.pt.window_kind().is_some())
}

fn check_window(f: &Family, text: &str) -> Result<(), String> {
    let got = window_pieces(f.pt, text);
    let want = pieces(f.pt, text);
    if got == want {
        return Ok(());
    }
    let first = got
        .iter()
        .zip(&want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    let show = |v: &[(usize, usize)]| -> Vec<String> {
        v.iter()
            .skip(first.saturating_sub(2))
            .take(6)
            .map(|&(a, b)| format!("{a}:{:?}", &text[a..b]))
            .collect()
    };
    Err(format!(
        "{} ({:?}) on {text:?}\n window: {:?}\n scalar: {:?}",
        f.name,
        f.family,
        show(&got),
        show(&want)
    ))
}

fn check_all(text: &str) {
    for f in windowed() {
        if let Err(e) = check_window(f, text) {
            panic!("{e}");
        }
    }
}

/// Fillers that put a text at every offset from a window start, with
/// different kinds of piece before it.
const FILLERS: [&str; 4] = ["x ", "ab, ", "\n\t", "12 é"];

fn filler(kind: usize, len: usize) -> String {
    FILLERS[kind % FILLERS.len()]
        .chars()
        .cycle()
        .take(len)
        .collect()
}

/// `text` alone, and placed so that it ends at the top of the first window
/// of a classified chunk (and of the tail copy), with filler after it.
fn placements(text: &str, out: &mut Vec<String>) {
    out.clear();
    out.push(text.to_string());
    for j in 0..3 {
        let lead = 64usize.saturating_sub(text.len() + j);
        for kind in 0..2 {
            let head = filler(kind, lead);
            out.push(format!("{head}{text}{}", filler(kind + 1, 160)));
            out.push(format!("{head}{text}{}", filler(kind + 3, 40)));
        }
    }
}

fn alphabet(family: FastPath) -> [char; 14] {
    match family {
        FastPath::O200k | FastPath::MiniMax | FastPath::Tekken => [
            ' ', '\n', 'e', 'R', 's', 'S', '\'', '1', '.', '/', 'É', 'é', '世', '\u{3000}',
        ],
        _ => [
            ' ', '\n', '\r', 'e', 'R', 's', '\'', '1', '.', '’', 'é', '\u{3000}', '\u{301}', '²',
        ],
    }
}

fn sweep(max_len: u32, families: &[&'static Family]) {
    let mut texts = Vec::new();
    for f in families {
        let chars = alphabet(f.family);
        let k = chars.len();
        let mut text = String::new();
        for len in 0..=max_len {
            for mut code in 0..k.pow(len) {
                text.clear();
                for _ in 0..len {
                    text.push(chars[code % k]);
                    code /= k;
                }
                if len <= 4 {
                    check(f, &text).unwrap();
                }
                placements(&text, &mut texts);
                for t in &texts {
                    if let Err(e) = check_window(f, t) {
                        panic!("{e}");
                    }
                }
            }
        }
    }
}

#[test]
fn all_short_strings_window_matches_scalar() {
    let fams: Vec<_> = windowed().collect();
    sweep(4, &fams);
}

#[test]
#[ignore = "minutes in release; run before changing the window rules"]
fn all_strings_up_to_six_chars_window_matches_scalar() {
    let fams: Vec<_> = windowed().collect();
    std::thread::scope(|s| {
        for f in fams {
            s.spawn(move || sweep(6, &[f]));
        }
    });
}

#[test]
fn hand_picked_texts_at_every_offset() {
    let texts = [
        "Hello, world! I'm don't they're we've she'll it'd I'M DON'T",
        "  hello  \n\n  x \t\ty  \r\n\r\n  .\n/ x/\n/\n",
        "fn main() {\n    let x = 1234567;\n    println!(\"{x}\");\n}\n",
        "“Well,” she said — ‘it’s fine’… naïve café",
        "HTMLParser parserHTML ABCdefGHI don'tX DON'Tx x's's 'll 're",
        "使用API。 ２３ ٣٤ a\u{301}b \u{3000}\u{3000}x \u{85}\u{a0}y",
        "!!\u{301}abc ǅungla ʰa Ⅷx ² 😀\u{200d}🎉",
    ];
    for t in texts {
        for pad in 0..=130 {
            for kind in 0..FILLERS.len() {
                let text = format!("{}{t}{}", filler(kind, pad), filler(kind + 1, pad / 3));
                check_all(&text);
            }
        }
    }
}

/// Chunks are decided by several chains that start at a letter followed by
/// whitespace near the middle of the chunk; put that start, and tricky text
/// right after it, at every offset around the place the search begins.
#[test]
fn chain_starts_at_every_offset() {
    // no letter in the filler, so the first candidate is the one placed
    let head = "1.2; (3) 45 ";
    let texts = [
        " don't  \n\n  x \t\ty  \r\n x' 'll",
        "\n\n\r\n  HTMLParser parserHTML 1234567 é",
        "\r“quoted” ‘it’s’ 世界 \u{3000}\u{3000}x ٣٤ a\u{301}b",
        "   \t  \n",
    ];
    for t in texts {
        for at in 1920..2120 {
            let mut text: String = head.chars().cycle().take(at - 1).collect();
            text.push('a');
            text.push_str(t);
            text.push_str(&filler(at, 3000));
            check_all(&text);
        }
    }
}

/// Long random texts, so pieces straddle windows and 4096-byte chunks.
#[test]
fn long_random_texts() {
    const CHARS: &[&str] = &[
        "a", "b", "Z", "Q", "s", "t", "e", "l", "'", "0", "7", " ", " ", " ", "\t", "\n", "\r",
        ".", ",", "/", "-", "(", "’", "“", "é", "É", "世", "\u{3000}", "\u{301}", "²", "😀",
    ];
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    for round in 0..40 {
        let mut text = String::new();
        let len = 200 + round * 300;
        while text.len() < len {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let i = (s % CHARS.len() as u64) as usize;
            let reps = 1 + (s >> 40) % if i < 10 { 9 } else { 3 };
            for _ in 0..reps {
                text.push_str(CHARS[i]);
            }
        }
        check_all(&text);
    }
}

/// Dense CJK goes to the scalar scanner a chunk (or the tail) at a time;
/// switching back and forth must not lose or split a piece.
#[test]
fn dense_and_sparse_chunks_alternate() {
    let zh = "你好，世界！这是一个测试。漢字かなカナ한글 ";
    let en = "The quick brown fox doesn't jump; it's 1234 “quoted” text.\n  ";
    for (a, b) in [(1, 1), (90, 3), (3, 90), (150, 70), (70, 150)] {
        for pad in [0, 1, 7, 63, 130] {
            let text = format!(
                "{}{}{}{}",
                filler(0, pad),
                zh.repeat(a),
                en.repeat(b),
                zh.repeat(a / 7 + 1)
            );
            check_all(&text);
        }
    }
}

/// Every fixture text through every family's window rules.
#[test]
fn fixture_texts_window_matches_scalar() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let mut texts = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for case in cases {
            check_all(case["text"].as_str().unwrap());
            texts += 1;
        }
    }
    assert!(texts > 50_000, "only {texts} fixture texts");
}

const TRICKY: &str = concat!(
    "[ \t\n\r\u{b}\u{c}'/aAsStTlLdDmMrReEvVxX09.,!?\\-(_",
    "\u{a0}\u{85}\u{2028}\u{3000}",
    "世界你龥\u{3400}\u{20000}ぁあーアヿ゠・한글",
    "مرح\u{64b}\u{650}١٢",
    "ǅʰ²Ⅷ\u{301}\u{3099}ＡａＺ１、。’“—",
    "😀\u{200d}🎉]{0,300}"
);

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    #[test]
    fn prop_tricky_alphabet_padded(text in TRICKY, pad in 0usize..=130, kind in 0usize..4) {
        let t = format!("{}{text}", filler(kind, pad));
        for f in windowed() {
            check_window(f, &t).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }

    #[test]
    fn prop_ascii_padded(text in "[ -~ \t\r\n]{0,400}", pad in 0usize..=130) {
        let t = format!("{}{text}", filler(pad, pad));
        for f in windowed() {
            check_window(f, &t).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }
}
