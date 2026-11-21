//! The scalar scanners against the regex reference, for every family.

use std::sync::LazyLock;

use super::reference::{Reference, adjust_whitespace_end};
use super::*;

/// One pre-tokenizer per family with its reference, taken from the shipped
/// encodings so the tests use exactly the pattern/family pairs that ship.
pub(super) struct Family {
    pub(super) name: &'static str,
    pub(super) family: FastPath,
    pub(super) pt: &'static ScalarPreTokenizer,
    pub(super) reference: Reference,
}

pub(super) static FAMILIES: LazyLock<Vec<Family>> = LazyLock::new(|| {
    let mut out: Vec<Family> = Vec::new();
    for &name in crate::list_encodings() {
        let pt = crate::get_encoding(name).unwrap().pre_tokenizer();
        let family = pt.family();
        if let Some(f) = out.iter().find(|f| f.family == family) {
            assert_eq!(f.pt.pattern, pt.pattern, "{name} vs {}", f.name);
            assert_eq!(f.pt.ws, pt.ws, "{name} vs {}", f.name);
            continue;
        }
        out.push(Family {
            name,
            family,
            pt,
            reference: Reference::new(&pt.pattern, family, pt.ws),
        });
    }
    assert_eq!(out.len(), 10);
    out
});

pub(super) fn pieces(pt: &ScalarPreTokenizer, text: &str) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut pos = 0;
    while let Some((start, end)) = pt.next_match(text, pos) {
        assert!(end > start, "empty piece at {start} in {text:?}");
        out.push((start, end));
        pos = end;
    }
    out
}

pub(super) fn check(f: &Family, text: &str) -> Result<(), String> {
    let got = pieces(f.pt, text);
    let want = f.reference.matches(text);
    if got == want {
        return Ok(());
    }
    let show = |v: &[(usize, usize)]| -> Vec<String> {
        v.iter()
            .map(|&(a, b)| format!("{:?}", &text[a..b]))
            .collect()
    };
    Err(format!(
        "{} ({:?}) on {text:?}\n scalar: {:?}\n regex:  {:?}",
        f.name,
        f.family,
        show(&got),
        show(&want)
    ))
}

fn check_all(text: &str) {
    for f in FAMILIES.iter() {
        if let Err(e) = check(f, text) {
            panic!("{e}");
        }
    }
}

#[test]
fn hand_picked_texts_match_reference() {
    let texts = [
        "",
        "a",
        "Hello, world!",
        "你好世界",
        "fn main() { }",
        "  hello  ",
        "line1\nline2\n",
        "café résumé",
        "100% of $1,000",
        "a@b.com",
        "   \t\n   ",
        "   \n  \t  ",
        "hello world! 你好 🚀 test 123",
        "word\n\nnext",
        "\r\n@rem",
        "a\n\n\nb",
        "a \n\n b",
        "\n  ",
        "\n  x",
        "I'm don't they're we've she'll it'd I'M DON'T",
        "Hello, world! CamelCase mixedScript123 HTMLParser parserHTML",
        "0²",
        "x²³⁴5",
        "!!\u{301}abc",
        "a\u{301}b \u{301}x \u{301}\u{301}",
        "使用API。",
        "ǅungla ʰa Ⅷx ²",
        "مرحبا بالعالم ١٢٣٤٥",
        "\u{3000}\u{3000}x \u{a0}\u{a0}y \u{85}z \u{2028}\u{2028}",
        "\u{1F468}\u{200D}\u{1F469} ok",
        "ＡＢＣａｂｃ１２３",
        ".\n/ x/\n/\n",
        "1234567 12３4",
        "  1 \u{4e00}  \u{3040}",
        "世界、。！？（）「」・…",
        "か\u{3099}々〇\u{20000}好",
        "\x0b\x0c \x0b",
    ];
    for t in texts {
        check_all(t);
    }
}

#[test]
fn hand_picked_cjk_texts_match_reference() {
    let texts = [
        "世界",
        "你好，世界！",
        "、你好",
        "　世界",
        "ハロー・ワールド",
        "世A",
        "世AB",
        "A世",
        "ＡB世A",
        "abc世界",
        "世界abc",
        "カタカナー",
        "パーティー",
        "が",
        "안녕하세요 세계",
        "ｱｲｳｴｵﾞ",
        "世's",
        "世界。。。",
        "……你好……",
        "「引用」",
        "第１２３号",
        "３．１４",
        "〇一二",
        "深圳市－广州市",
        "日本語テスト123テスト",
        "。\n、",
        "税込１，０００円",
        "「こんにちは」と言った",
    ];
    for t in texts {
        check_all(t);
    }
}

/// Every text of every encoding's fixture file, through every family.
#[test]
fn fixture_texts_match_reference() {
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

// the old lookahead emulation, pinned so the reference stays what it was

use WhitespaceRules::{Generic, NewlineFirst};

#[test]
fn reference_whitespace_adjustment() {
    assert_eq!(adjust_whitespace_end(b"a b", 0, 1, Generic), 1);
    assert_eq!(adjust_whitespace_end(b"  ", 0, 2, Generic), 2);
    assert_eq!(adjust_whitespace_end(b"hello world", 0, 5, Generic), 5);
    assert_eq!(adjust_whitespace_end(b"  x", 0, 2, Generic), 1);
    assert_eq!(adjust_whitespace_end(b"   ", 0, 2, Generic), 2);
    let s = "\u{3000}\u{3000}x".as_bytes();
    assert_eq!(adjust_whitespace_end(s, 0, 6, Generic), 3);
    let s = "\u{3000}\u{3000}\u{3000}".as_bytes();
    assert_eq!(adjust_whitespace_end(s, 0, 6, Generic), 6);
    let s = "\u{3000}x".as_bytes();
    assert_eq!(adjust_whitespace_end(s, 0, 3, Generic), 3);
    assert_eq!(adjust_whitespace_end(b"\n\nx", 0, 2, NewlineFirst), 2);
    assert_eq!(adjust_whitespace_end(b"\n\nx", 0, 2, Generic), 1);
    assert_eq!(adjust_whitespace_end(b"\r\n@", 0, 2, NewlineFirst), 2);
    assert_eq!(adjust_whitespace_end(b"\r\n@", 0, 2, Generic), 1);
    assert_eq!(adjust_whitespace_end(b"  x", 0, 2, NewlineFirst), 1);
    assert_eq!(adjust_whitespace_end(b"\n  x", 1, 3, NewlineFirst), 2);
}

/// How many random cases each property sweep generates.
///
/// The committed `proptest-regressions/` seeds replay before any random
/// case, so every input that ever failed is still checked at any setting —
/// what this scales is how much *new* ground a run covers. The default is a
/// smoke sample so a push is answered in minutes; the full sweep is a release
/// gate, run with `PROPTEST_CASES=20000` (see the Release section of the
/// top-level README).
fn proptest_cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}

/// Chars where the rules interact: every whitespace kind, `'` and `/`, both
/// cases, digits and punctuation, the Unicode spaces, Han, kana, Hangul,
/// Arabic letters, marks and digits, Lt, Lm, No, Nl, fullwidth forms, emoji
/// and ZWJ.
const TRICKY: &str = concat!(
    "[ \t\n\r\u{b}\u{c}'/aAsStTlLdDmMrReEvVxX09.,!?\\-(_",
    "\u{a0}\u{85}\u{2028}\u{3000}",
    "世界你龥\u{3400}\u{20000}ぁあーアヿ゠・한글",
    "مرح\u{64b}\u{650}١٢",
    "ǅʰ²Ⅷ\u{301}\u{3099}ＡａＺ１、。",
    "😀\u{200d}🎉]{0,300}"
);

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(proptest_cases()))]

    #[test]
    fn prop_any_text(text in ".*") {
        for f in FAMILIES.iter() {
            check(f, &text).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }

    #[test]
    fn prop_ascii(text in "[ -~ \t\r\n]*") {
        for f in FAMILIES.iter() {
            check(f, &text).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }

    #[test]
    fn prop_newlines_and_slashes(text in "[/\r\n \tabcAB0.!]*") {
        for f in FAMILIES.iter() {
            check(f, &text).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }

    #[test]
    fn prop_cjk(
        text in "[世界你好日本語謎アイウエオぁあんーゟＡＢａｂ한글、。！？（）「」・…　 a-cA-C0-9'\r\n々〇\u{3099}\u{20000}é龥龦゠ヿ]*"
    ) {
        for f in FAMILIES.iter() {
            check(f, &text).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }

    #[test]
    fn prop_tricky_alphabet(text in TRICKY) {
        for f in FAMILIES.iter() {
            check(f, &text).map_err(proptest::test_runner::TestCaseError::fail)?;
        }
    }
}
