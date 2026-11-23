//! Complete scanner for the case-splitting patterns (o200k_base, minimax_m2,
//! llama4, mistral_v3, mistral_v15, kimi_k2, kimi_k3):
//!
//! ```text
//! [\p{Han}]+                                    (kimi only)
//! | [^\r\n\p{L}\p{N}]?U*W+C?                    (rule A)
//! | [^\r\n\p{L}\p{N}]?U+W*C?                    (rule B)
//! | \p{N}{1,MAX} | ?[^\s\p{L}\p{N}]+[\r\n]*  (or [\r\n/]*) | \s*[\r\n]+ | \s+(?!\S) | \s+
//! ```
//!
//! `U` is `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`, `W` is `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`
//! (both without Han for kimi) and `C` the optional contraction suffix.

use crate::unicode::{C_HAN, C_LOWER, C_NUM, C_PREFIX, C_UPPER, FamilyMap, class_at};

use super::match_contraction;
use super::scalar::{class_run, digit_run, is_ws_at, no_boundary, punct_piece, whitespace_piece};

/// Rule A's letters from `p`: `U*` takes the longest run of upper-class
/// chars; if a lower-class char follows, `W+` runs from there. Otherwise the
/// engine backtracks `U*` until `W+` can take one char, and the only run
/// members `W` accepts are the chars in both classes, so the letters end
/// after the run's last such char. `None` when there is none.
#[inline]
fn rule_a(map: &FamilyMap, b: &[u8], p: usize) -> Option<usize> {
    let n = b.len();
    let mut q = p;
    let mut last_both = None;
    while q < n {
        let (f, len) = class_at(map, b, q);
        if f & C_UPPER == 0 {
            if f & C_LOWER != 0 {
                return Some(class_run(map, b, q + len, C_LOWER));
            }
            break;
        }
        q += len;
        if f & C_LOWER != 0 {
            last_both = Some(q);
        }
    }
    last_both
}

/// Rule B's letters from `p`: `U+W*`.
#[inline]
fn rule_b(map: &FamilyMap, b: &[u8], p: usize) -> Option<usize> {
    let q = class_run(map, b, p, C_UPPER);
    (q > p).then(|| class_run(map, b, q, C_LOWER))
}

/// End of the letters of the word rules at `pos`, in the pattern's order of
/// preference: rule A with the leading char, rule A without it, then the same
/// for rule B. Only a `\p{M}` can be both a leading char and a letter.
#[inline]
fn word_letters(map: &FamilyMap, b: &[u8], pos: usize, f0: u8, l0: usize) -> Option<usize> {
    let prefix = f0 & C_PREFIX != 0;
    if prefix && let Some(e) = rule_a(map, b, pos + l0) {
        return Some(e);
    }
    if f0 & (C_UPPER | C_LOWER) != 0
        && let Some(e) = rule_a(map, b, pos)
    {
        return Some(e);
    }
    if prefix && let Some(e) = rule_b(map, b, pos + l0) {
        return Some(e);
    }
    if f0 & C_UPPER != 0 {
        return rule_b(map, b, pos);
    }
    None
}

#[inline(never)]
pub(super) fn case_piece<
    const CONTRACTIONS: bool,
    const MAX_DIGITS: usize,
    const SLASH: bool,
    const HAN_APART: bool,
>(
    map: &FamilyMap,
    b: &[u8],
    pos: usize,
) -> usize {
    let n = b.len();
    let (f0, l0) = class_at(map, b, pos);
    if HAN_APART && f0 & C_HAN != 0 {
        return class_run(map, b, pos + l0, C_HAN);
    }
    if let Some(mut e) = word_letters(map, b, pos, f0, l0) {
        if CONTRACTIONS
            && e < n
            && b[e] == b'\''
            && let Some(len) = match_contraction(b, e)
        {
            e += len;
        }
        return e;
    }
    if f0 & C_NUM != 0 {
        return digit_run::<MAX_DIGITS>(map, b, pos);
    }
    if let Some(e) = punct_piece::<SLASH>(map, b, pos, f0, l0) {
        return e;
    }
    debug_assert!(is_ws_at(map, b, pos));
    whitespace_piece::<true>(b, pos, no_boundary)
}
