//! Complete scanner for the deepseek_v3 / deepseek_v4 pattern:
//!
//! ```text
//! \p{N}{1,3} | [一-龥\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+ | [!-/:-@\[-`{-~][A-Za-z]+
//! | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+ | [\s\S]
//! ```
//!
//! Upstream runs the first two alternatives as earlier split stages, so a
//! whitespace run followed by a digit or one of those CJK chars ends its
//! slice and keeps its last char.

use crate::unicode::{
    C_LETTER, C_NUM, C_PREFIX, C_WS, GROUP_MASK, MAP_DEEPSEEK, NUM, category, class_at, decode,
};

use super::scalar::{class_run, digit_run, punct_piece, whitespace_piece};

/// The CJK alternative's class, by code point.
#[inline]
pub(super) fn is_deepseek_cjk(c: u32) -> bool {
    matches!(c, 0x4E00..=0x9FA5 | 0x3040..=0x30FF)
}

/// Whether the char at `j` starts a slice of its own upstream.
fn split_boundary(b: &[u8], j: usize) -> bool {
    let (cp, _) = decode(b, j);
    category(cp) & GROUP_MASK == NUM || is_deepseek_cjk(cp)
}

#[inline(never)]
pub(super) fn deepseek_piece(b: &[u8], pos: usize) -> usize {
    let map = &MAP_DEEPSEEK;
    let n = b.len();
    let (cp0, l0) = decode(b, pos);
    let f0 = map[category(cp0) as usize];
    if f0 & C_NUM != 0 {
        return digit_run::<3>(map, b, pos);
    }
    if is_deepseek_cjk(cp0) {
        let mut j = pos + l0;
        while j < n {
            let (cp, len) = decode(b, j);
            if !is_deepseek_cjk(cp) {
                break;
            }
            j += len;
        }
        return j;
    }
    if b[pos].is_ascii_punctuation() && b.get(pos + 1).is_some_and(u8::is_ascii_alphabetic) {
        let mut j = pos + 2;
        while j < n && b[j].is_ascii_alphabetic() {
            j += 1;
        }
        return j;
    }
    let p = pos + l0;
    if f0 & C_PREFIX != 0 && p < n {
        let (f1, l1) = class_at(map, b, p);
        if f1 & C_LETTER != 0 {
            return class_run(map, b, p + l1, C_LETTER);
        }
    }
    if f0 & C_LETTER != 0 {
        return class_run(map, b, p, C_LETTER);
    }
    if let Some(e) = punct_piece::<false>(map, b, pos, f0, l0) {
        return e;
    }
    if f0 & C_WS != 0 {
        return whitespace_piece::<true>(b, pos, split_boundary);
    }
    p
}
