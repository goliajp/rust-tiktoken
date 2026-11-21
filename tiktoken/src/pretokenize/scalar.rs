//! Rules shared by the complete scalar scanners. Each scanner takes a piece
//! start `pos < b.len()` and returns the piece end; every pattern matches any
//! char, so a piece always starts at `pos`.

use crate::unicode::{C_NUM, C_PUNCT, C_WS, FamilyMap, WS, category, class_at, decode};

use super::is_ascii_ws;

/// The whitespace rules at `pos`, which holds a `\s` char.
///
/// `\s*[\r\n]+` (when `NEWLINE_FIRST`) takes the run up to and including its
/// last `\r`/`\n`. Otherwise `\s+(?!\S)|\s+` takes the whole run, except that
/// a run of two or more chars followed by a non-space gives its last char
/// back. `boundary` reports chars after which the run keeps that char anyway.
#[inline(never)]
pub(super) fn whitespace_piece<const NEWLINE_FIRST: bool>(
    b: &[u8],
    pos: usize,
    boundary: impl Fn(&[u8], usize) -> bool,
) -> usize {
    let n = b.len();
    let mut j = pos;
    let mut nl_end = 0;
    let mut last = pos;
    while j < n {
        let c = b[j];
        if c < 0x80 {
            if !is_ascii_ws(c) {
                break;
            }
            if c == b'\r' || c == b'\n' {
                nl_end = j + 1;
            }
            last = j;
            j += 1;
        } else {
            let (cp, len) = decode(b, j);
            if category(cp) & WS == 0 {
                break;
            }
            last = j;
            j += len;
        }
    }
    if NEWLINE_FIRST && nl_end != 0 {
        return nl_end;
    }
    if j < n && last > pos && !boundary(b, j) {
        last
    } else {
        j
    }
}

/// No char after a whitespace run is a boundary.
#[inline(always)]
pub(super) fn no_boundary(_: &[u8], _: usize) -> bool {
    false
}

/// End of the run of chars with any of the `class` bits, from `j`.
#[inline]
pub(super) fn class_run(map: &FamilyMap, b: &[u8], mut j: usize, class: u8) -> usize {
    while j < b.len() {
        let (f, len) = class_at(map, b, j);
        if f & class == 0 {
            break;
        }
        j += len;
    }
    j
}

/// `\p{N}{1,MAX}` from `j`, which holds a `\p{N}` char.
#[inline]
pub(super) fn digit_run<const MAX: usize>(map: &FamilyMap, b: &[u8], mut j: usize) -> usize {
    let mut k = 0;
    while k < MAX && j < b.len() {
        let (f, len) = class_at(map, b, j);
        if f & C_NUM == 0 {
            break;
        }
        j += len;
        k += 1;
    }
    j
}

/// The punctuation rule ` ?P+T*` at `pos`, where `P` is the family's
/// punctuation class and `T` is `[\r\n]` (or `[\r\n/]` with `SLASH`).
/// `None` when the rule does not match.
#[inline]
pub(super) fn punct_piece<const SLASH: bool>(
    map: &FamilyMap,
    b: &[u8],
    pos: usize,
    f0: u8,
    l0: usize,
) -> Option<usize> {
    let start = if f0 & C_PUNCT != 0 {
        pos + l0
    } else if b[pos] == b' ' && pos + 1 < b.len() {
        let (f1, l1) = class_at(map, b, pos + 1);
        if f1 & C_PUNCT == 0 {
            return None;
        }
        pos + 1 + l1
    } else {
        return None;
    };
    let e = class_run(map, b, start, C_PUNCT);
    Some(super::take_line_tail::<SLASH>(b, e))
}

/// Whether the char at `j` is `\s` (for debug assertions).
pub(super) fn is_ws_at(map: &FamilyMap, b: &[u8], j: usize) -> bool {
    class_at(map, b, j).0 & C_WS != 0
}
