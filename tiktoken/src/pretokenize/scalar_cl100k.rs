//! Complete scanner for the cl100k-shaped patterns:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?L+|\p{N}{1,MAX}| ?P+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//! where `L` is `\p{L}` (`[\p{L}\p{M}]` for qwen3_5) and `P` the rest of the
//! non-space chars. The family map supplies `L` and `P`.

use crate::unicode::{C_LETTER, C_NUM, C_PREFIX, FamilyMap, class_at};

use super::match_contraction;
use super::scalar::{class_run, digit_run, is_ws_at, no_boundary, punct_piece, whitespace_piece};

#[inline(never)]
pub(super) fn cl100k_piece<const MAX_DIGITS: usize>(
    map: &FamilyMap,
    b: &[u8],
    pos: usize,
) -> usize {
    let n = b.len();
    if b[pos] == b'\''
        && let Some(len) = match_contraction(b, pos)
    {
        return pos + len;
    }
    let (f0, l0) = class_at(map, b, pos);
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
    if f0 & C_NUM != 0 {
        return digit_run::<MAX_DIGITS>(map, b, pos);
    }
    if let Some(e) = punct_piece::<false>(map, b, pos, f0, l0) {
        return e;
    }
    debug_assert!(is_ws_at(map, b, pos));
    whitespace_piece::<true>(b, pos, no_boundary)
}
