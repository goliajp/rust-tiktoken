//! Every string up to a fixed length over a small per-family alphabet, scalar
//! scanner against the regex reference. The alphabets hold one char of each
//! class the family's rules distinguish, so rule interactions are covered
//! without relying on any derivation of the scanners.

use super::tests::{FAMILIES, check};
use super::*;

const BASE: [char; 12] = [
    ' ', '\n', 'a', 'S', 'l', '\'', '1', '.', '\u{301}', '世', '\u{3000}', '/',
];

fn alphabet(family: FastPath) -> Vec<char> {
    let extra: [char; 2] = match family {
        FastPath::Cl100k | FastPath::Qwen2 | FastPath::Qwen35 | FastPath::P50k => ['\t', '²'],
        FastPath::Tekken15 => ['²', 'ǅ'],
        FastPath::Kimi => ['ア', 'ǅ'],
        FastPath::Deepseek => ['ア', '\r'],
        FastPath::O200k | FastPath::MiniMax | FastPath::Tekken => ['ǅ', 'ʰ'],
    };
    BASE.iter().chain(extra.iter()).copied().collect()
}

fn sweep(max_len: u32) {
    for f in FAMILIES.iter() {
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
                if let Err(e) = check(f, &text) {
                    panic!("{e}");
                }
            }
        }
    }
}

#[test]
fn all_short_strings_match_reference() {
    sweep(4);
}

#[test]
#[ignore = "about a minute in release; run before changing a scanner"]
fn all_strings_up_to_six_chars_match_reference() {
    sweep(6);
}
