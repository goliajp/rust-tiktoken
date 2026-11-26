//! Cuts: byte offsets where a text can be split in two without changing its
//! tokens.
//!
//! `b` is a cut of `t` when the pieces of `t[..b]` followed by those of
//! `t[b..]` (shifted by `b`) are exactly the pieces of `t`. BPE runs inside
//! each piece, so the ids then add up too. Two facts about the scanners make
//! the rules below short:
//!
//! - A scan reads forward from a piece start and never behind it, with one
//!   exception (mistral_v15's digit rule looks at the char before a digit).
//!   So once `b` is a piece start of `t`, `t[b..]` scans like the rest of `t`
//!   unless `t[b]` is a digit.
//! - Without lookahead, a match inside `t[..b]` is also a match in `t`, and
//!   the preferred one stays preferred, so `t[..b]` scans like the start of
//!   `t`. The only lookahead is `\s+(?!\S)`, which at `b` sees the end of
//!   `t[..b]` but `t[b]` in `t`; the two agree when `t[b]` is whitespace or
//!   `t[b-1]` is not.
//!
//! Every rule therefore shows that `b` is a piece start of `t` and that
//! `t[b]` is not a digit, and that `t[b-1]` is not whitespace or `t[b]` is.
//! Each looks only at the few bytes around `b`, inside `t`, so a cut of a
//! text is a cut of every text that contains it.
//!
//! **R1.** `t[b-1]` is an ASCII letter and `t[b]` is `\n`. No rule of any
//! pattern takes a letter and then a newline: the letter rules end at a
//! non-letter (their contraction suffixes start with `'`), and the
//! punctuation rules' newline tails only follow punctuation.
//!
//! **R2.** `t[b-2]` is ASCII and not whitespace, `t[b-1]` is `\n` and `t[b]`
//! an ASCII letter or digit. The newline ends a punctuation tail, forms
//! `\s*[\r\n]+` or (p50k) a lone `\s+`; no letter prefix class admits `\r` or
//! `\n`. Requiring a non-blank char before it keeps p50k's `\s+(?!\S)` from
//! claiming a longer run when the text ends there. mistral_v15's digit rule
//! treats the newline like the start of the text.
//!
//! **R3.** `t[b-1]` is a Han char in U+4E00..=U+9FA5 or a kana letter
//! (U+3041..=U+30FF with category `Lo` or `Lm`, so not `・` or `゛`), and
//! `t[b]` one of `，。、！？；：「」『』（）`. Those chars are in every
//! letter class (deepseek's CJK run, kimi's `[\p{Han}]+`, the `\p{L}`
//! classes elsewhere) and the marks are in none, nor in deepseek's CJK run
//! range; they are punctuation everywhere, and no rule takes a letter and then
//! punctuation.
//!
//! **R4.** `t[b]` is a space or a tab and `t[b-1]` is an ASCII graphic char or
//! a non-ASCII char that is not whitespace. Whitespace appears inside a match
//! only in the whitespace rules (`\s*[\r\n]+`, `\s+`) and as the single
//! leading char of a word or punctuation rule; neither puts it after a
//! non-whitespace char of the same match, and the newline tails after
//! punctuation take only `\r` and `\n`.
//!
//! **R5.** `t[b]` is a space or a tab, `t[b-1]` is whitespace and `t[b+1]` an
//! ASCII graphic char: the last blank of a whitespace run, before the first
//! char of a word, number or punctuation run; indentation in code. Let the
//! run start at `w`. The pieces that start in `w..b` end by `b`: the word
//! and punctuation rules need a non-blank second char, which `t[w+1..=b]`
//! is not; `\s*[\r\n]+` ends at the run's last newline, before `b` because
//! `t[b]` is not one; `\s+(?!\S)` backs off from the run's end until the next
//! char is whitespace, which first holds at `b`. A punctuation tail that
//! reaches into the run takes only newlines. deepseek_v3 splits on digits
//! and on CJK before its other rules, so a digit after the run ends its
//! lookahead like the end of the text and `\s+(?!\S)` takes the whole run:
//! for deepseek `t[b+1]` must not be a digit ([`Rules::indent_before_digit`]).
//! An ASCII graphic `t[b+1]` is never CJK.
//!
//! The tests in `tests.rs` check each rule against every encoding's scanner
//! on random and real texts, piece by piece and id by id.

#[cfg(test)]
mod tests;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod simd;

use crate::pretokenize::FastPath;
use crate::unicode::{GROUP_MASK, LM, LO, WS, category, decode_before};

/// What differs between patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rules {
    /// R5 also holds before a digit.
    pub(crate) indent_before_digit: bool,
}

impl Rules {
    pub(crate) const fn of(family: FastPath) -> Self {
        Rules {
            indent_before_digit: !matches!(family, FastPath::Deepseek),
        }
    }
}

/// Whether one of the rules allows a cut before `b[i]`, `0 < i < b.len()`.
#[inline]
pub(crate) fn is_cut(b: &[u8], i: usize, rules: Rules) -> bool {
    let (prev, cur) = (b[i - 1], b[i]);
    match cur {
        b' ' | b'\t' => {
            let blank_before = if prev < 0x80 {
                !prev.is_ascii_graphic()
            } else {
                category(decode_before(b, i)) & WS != 0
            };
            if !blank_before {
                return true; // R4
            }
            // R5; an ASCII control char before the blank is neither
            b.get(i + 1).is_some_and(|&n| {
                n.is_ascii_graphic() && (rules.indent_before_digit || !n.is_ascii_digit())
            }) && is_blank_byte(prev)
        }
        b'\n' => prev.is_ascii_alphabetic(),
        _ if prev == b'\n' => {
            cur.is_ascii_alphanumeric()
                && i >= 2
                && b[i - 2].is_ascii()
                && !b[i - 2].is_ascii_whitespace()
                && b[i - 2] != 0x0B
        }
        0xE3 | 0xEF => i >= 3 && cjk_mark_at(b, i) && cjk_letter_before(b, i),
        _ => false,
    }
}

/// `b[i - 1]` is the last byte of a whitespace char: ASCII blanks, or the
/// continuation byte of a non-ASCII one (the caller checked its category).
#[inline]
fn is_blank_byte(prev: u8) -> bool {
    prev >= 0x80 || matches!(prev, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

fn cjk_mark_at(b: &[u8], i: usize) -> bool {
    matches!(
        b.get(i..i + 3),
        Some(
            [0xEF, 0xBC, 0x8C]          // ，
                | [0xE3, 0x80, 0x82]    // 。
                | [0xE3, 0x80, 0x81]    // 、
                | [0xEF, 0xBC, 0x81]    // ！
                | [0xEF, 0xBC, 0x9F]    // ？
                | [0xEF, 0xBC, 0x9B]    // ；
                | [0xEF, 0xBC, 0x9A]    // ：
                | [0xE3, 0x80, 0x8C]    // 「
                | [0xE3, 0x80, 0x8D]    // 」
                | [0xE3, 0x80, 0x8E]    // 『
                | [0xE3, 0x80, 0x8F]    // 』
                | [0xEF, 0xBC, 0x88]    // （
                | [0xEF, 0xBC, 0x89] // ）
        )
    )
}

/// The char before `i` is a three-byte Han char in U+4E00..=U+9FA5 or a kana
/// letter.
fn cjk_letter_before(b: &[u8], i: usize) -> bool {
    let [b0, b1, b2] = [b[i - 3], b[i - 2], b[i - 1]];
    if b0 & 0xF0 != 0xE0 || b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 {
        return false;
    }
    let cp = (u32::from(b0 & 0x0F) << 12) | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F);
    (0x4E00..=0x9FA5).contains(&cp)
        || ((0x3041..=0x30FF).contains(&cp) && matches!(category(cp) & GROUP_MASK, LO | LM))
}

/// The first cut at or after `from`.
pub(crate) fn next(b: &[u8], from: usize, rules: Rules) -> Option<usize> {
    let i = from.max(1);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let i = match simd::next(b, i, rules) {
        Ok(cut) => return Some(cut),
        Err(i) => i,
    };
    (i..b.len()).find(|&i| is_cut(b, i, rules))
}

/// The last cut at or before `to`.
pub(crate) fn prev(b: &[u8], to: usize, rules: Rules) -> Option<usize> {
    if b.len() < 2 || to == 0 {
        return None;
    }
    let j = to.min(b.len() - 1);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let j = match simd::prev(b, j, rules) {
        Ok(cut) => return Some(cut),
        Err(j) => j,
    };
    (1..=j).rev().find(|&i| is_cut(b, i, rules))
}

/// Whether byte `i` can start a cut or follows a newline: what the vector
/// scans look for before calling [`is_cut`].
#[cfg(test)]
pub(crate) fn is_candidate(b: &[u8], i: usize) -> bool {
    let (prev, cur) = (b[i - 1], b[i]);
    matches!(cur, b' ' | b'\t' | b'\n')
        || prev == b'\n'
        || (cur == 0xE3 && b.get(i + 1) == Some(&0x80))
        || (cur == 0xEF && b.get(i + 1) == Some(&0xBC))
}
