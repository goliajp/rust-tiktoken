//! Complete scanner for the p50k/r50k pattern:
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.

use crate::unicode::{C_LETTER, C_NUM, C_PUNCT, C_WS, MAP_BASE, class_at};

use super::scalar::{class_run, no_boundary, whitespace_piece};

/// The case-sensitive contraction at `i` (which holds `'`), as a length.
#[inline]
fn contraction(b: &[u8], i: usize) -> Option<usize> {
    match b.get(i + 1)? {
        b's' | b't' | b'm' | b'd' => Some(2),
        b'r' | b'v' if b.get(i + 2) == Some(&b'e') => Some(3),
        b'l' if b.get(i + 2) == Some(&b'l') => Some(3),
        _ => None,
    }
}

#[inline(never)]
pub(super) fn p50k_piece(b: &[u8], pos: usize) -> usize {
    let map = &MAP_BASE;
    if b[pos] == b'\''
        && let Some(len) = contraction(b, pos)
    {
        return pos + len;
    }
    let (mut f, mut l) = class_at(map, b, pos);
    let mut start = pos;
    if b[pos] == b' ' && pos + 1 < b.len() {
        let (f1, l1) = class_at(map, b, pos + 1);
        if f1 & C_WS == 0 {
            (f, l, start) = (f1, l1, pos + 1);
        }
    }
    for class in [C_LETTER, C_NUM, C_PUNCT] {
        if f & class != 0 {
            return class_run(map, b, start + l, class);
        }
    }
    whitespace_piece::<false>(b, pos, no_boundary)
}
