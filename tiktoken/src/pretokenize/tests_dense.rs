//! The scan of dense non-ASCII text against the scalar scanners, for every
//! family with window rules.

use super::tests::{FAMILIES, pieces};
use super::window::{CaseRules, Cl100kRules};
use super::windowed::{PieceSink, split};
use super::*;

struct Collect(Vec<(usize, usize)>);

impl PieceSink for Collect {
    fn window(&mut self) {}

    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        assert!(end > start && buf.len() >= start + crate::vocab::KEY_READ);
        self.0.push((start, end));
    }
}

fn split_pieces(pt: &ScalarPreTokenizer, text: &str) -> Vec<(usize, usize)> {
    let mut c = Collect(Vec::new());
    match pt.window_kind().unwrap() {
        WindowKind::Cl100k => split::<Cl100kRules<3, false>, _>(pt, text, &mut c),
        WindowKind::Qwen2 => split::<Cl100kRules<1, false>, _>(pt, text, &mut c),
        WindowKind::Qwen35 => split::<Cl100kRules<1, true>, _>(pt, text, &mut c),
        WindowKind::O200k => split::<CaseRules<true, 3>, _>(pt, text, &mut c),
        WindowKind::Tekken => split::<CaseRules<false, 1>, _>(pt, text, &mut c),
    }
    // pieces taken from the tail copy are offsets into it; rebase them
    let mut pos = 0;
    c.0.into_iter()
        .map(|(s, e)| {
            let len = e - s;
            let r = (pos, pos + len);
            pos += len;
            r
        })
        .collect()
}

fn check(text: &str) {
    for f in FAMILIES.iter().filter(|f| f.pt.window_kind().is_some()) {
        let got = split_pieces(f.pt, text);
        let want = pieces(f.pt, text);
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
            let show = |v: &[(usize, usize)]| -> Vec<String> {
                v.iter()
                    .skip(at.saturating_sub(2))
                    .take(5)
                    .map(|&(a, b)| format!("{a}:{:?}", &text[a..b]))
                    .collect()
            };
            panic!(
                "{} ({:?}), {} bytes\n dense:  {:?}\n scalar: {:?}",
                f.name,
                f.family,
                text.len(),
                show(&got),
                show(&want)
            );
        }
    }
}

/// Chars of every class the scanners tell apart, weighted towards CJK so
/// that most chunks count as dense.
const CHARS: &[&str] = &[
    "世", "界", "你", "好", "こ", "ん", "ー", "ア", "한", "글", "م", "ر", "\u{64b}", "\u{3099}",
    "゠", "・", "，", "。", "！", "\u{3000}", "\u{a0}", "\u{85}", "\u{2028}", "１", "٣", "Ⅷ", "²",
    "ǅ", "ʰ", "É", "é", "😀", "\u{200d}", " ", " ", "\n", "\r", "\t", "'", "s", "S", "t", "l", "L",
    "d", "r", "e", "v", "m", "x", "A", "0", "7", ".", "/", "-", "\u{301}",
];

fn random_text(seed: u64, len: usize) -> String {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut text = String::new();
    while text.len() < len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let i = (s % CHARS.len() as u64) as usize;
        // CJK chars often, and runs long enough to cross a span
        let i = if (s >> 20).is_multiple_of(3) {
            (s >> 24) as usize % 8
        } else {
            i
        };
        let reps = match (s >> 32) % 64 {
            0 => 200 + (s >> 40) % 400,
            1..=8 => 2 + (s >> 40) % 30,
            _ => 1,
        };
        for _ in 0..reps {
            text.push_str(CHARS[i]);
        }
    }
    text
}

#[test]
fn random_dense_texts_match_scalar() {
    for seed in 0..300 {
        let len = 100 + (seed as usize * 97) % 5000;
        check(&random_text(seed, len));
    }
}

/// Runs of every kind of char, short and across the end of a classified
/// chunk, with every kind of char after them.
#[test]
fn runs_with_every_kind_of_end() {
    let runs = [
        "世", " ", "\u{3000}", "\n", " \n", "１", "ア", "ǅ", "'", "！", "ʰ",
    ];
    let tails = [
        "", "s", "'s", "'re", " ", "\n", "x", "世", "１", "!", "\u{301}",
    ];
    for run in runs {
        for tail in tails {
            for lead in [0, 1, 2, 3, 5] {
                for len in [6, 170, 1360, 1366, 1400] {
                    let text = format!(
                        "{}{}{tail}{}",
                        "界".repeat(lead + 90),
                        run.repeat(len - lead),
                        "你好".repeat(40)
                    );
                    check(&text);
                }
            }
        }
    }
}
