//! Text dense in non-ASCII chars (CJK, Hangul, Arabic and the like).
//!
//! Most pieces of such text are one run of letters, with an optional leading
//! char. Those are scanned here, with one bit per code point below 0x10000
//! ([`BmpBits`]) telling whether a 2- or 3-byte char continues the run: the
//! next position follows from the lead byte alone, so it never waits on the
//! table. Every other piece goes to the complete scalar scanner.

use super::windowed::PieceSink;
use super::{FastPath, PreTokenizer, ScalarPreTokenizer};
use crate::unicode::{
    BOTH_CASES, BmpBits, C_LETTER, C_LOWER, C_PREFIX, C_PUNCT, C_UPPER, FamilyMap, LETTERS,
    LETTERS_QWEN35, MAP_BASE, MAP_QWEN35, class_at,
};

/// Pieces of `text` from `pos` until one ends at or past `until`, like the
/// scalar scanner. `buf` holds the same bytes as `text`, with
/// [`KEY_READ`](crate::vocab::KEY_READ) readable bytes from `until`.
#[inline(never)]
pub(super) fn span<S: PieceSink>(
    pre: &ScalarPreTokenizer,
    text: &str,
    buf: &[u8],
    pos: usize,
    until: usize,
    sink: &mut S,
) -> usize {
    let at = Span {
        pre,
        text,
        buf,
        pos,
        until,
    };
    match pre.fast {
        FastPath::Cl100k | FastPath::Qwen2 => run::<_, false>(at, sink, &MAP_BASE, &LETTERS),
        FastPath::Qwen35 => run::<_, false>(at, sink, &MAP_QWEN35, &LETTERS_QWEN35),
        FastPath::O200k | FastPath::MiniMax | FastPath::Tekken => {
            run::<_, true>(at, sink, &MAP_BASE, &BOTH_CASES)
        }
        f => unreachable!("no window rules for {f:?}"),
    }
}

/// The arguments of [`span`].
#[derive(Clone, Copy)]
struct Span<'a> {
    pre: &'a ScalarPreTokenizer,
    text: &'a str,
    buf: &'a [u8],
    pos: usize,
    until: usize,
}

/// [`span`] with the family's class map and bits of run chars.
#[inline(always)]
fn run<S: PieceSink, const CASE: bool>(
    at: Span<'_>,
    sink: &mut S,
    map: &FamilyMap,
    bits: &BmpBits,
) -> usize {
    let Span {
        pre,
        text,
        buf,
        mut pos,
        until,
    } = at;
    let b = text.as_bytes();
    while pos < until {
        let end = match word_end::<CASE>(b, map, bits, pos) {
            Some(e) => e,
            None => pre.next_match(text, pos).map_or(b.len(), |(_, e)| e),
        };
        sink.window();
        sink.dense_piece(buf, pos, end);
        pos = end;
    }
    pos
}

/// Chars whose runs the word rules take whole: letters (cl100k family), or
/// chars in both case classes (case-split family: CJK, kana, Hangul, Arabic
/// letters, modifier letters and marks). A mark can also lead a word, but
/// taken as the leading char or as the first char of the run, the piece
/// reaches the same end.
#[inline(always)]
fn run_class<const CASE: bool>(f: u8) -> bool {
    if CASE {
        f & (C_UPPER | C_LOWER) == C_UPPER | C_LOWER
    } else {
        f & C_LETTER != 0
    }
}

/// End of the piece at `k` when it is a run of [`run_class`] chars, with an
/// optional leading char, that ends at a char no word rule takes, or a run of
/// punctuation that no word follows; `None` when the complete scanner has to
/// decide.
#[inline(always)]
fn word_end<const CASE: bool>(
    b: &[u8],
    map: &FamilyMap,
    bits: &BmpBits,
    k: usize,
) -> Option<usize> {
    let n = b.len();
    let (f0, l0) = class_at(map, b, k);
    let mut j = if run_class::<CASE>(f0) {
        k + l0
    } else if f0 & C_PREFIX != 0 && k + l0 < n {
        let (f1, l1) = class_at(map, b, k + l0);
        // a leading contraction comes first in the cl100k pattern
        if run_class::<CASE>(f1) && (CASE || b[k] != b'\'') {
            k + l0 + l1
        } else if f1 & (C_LETTER | C_UPPER | C_LOWER) != 0 {
            return None;
        } else if f0 & C_PUNCT != 0 {
            return punct_end(b, map, k + l0);
        } else if b[k] == b' ' && f1 & C_PUNCT != 0 {
            // the optional space before a punctuation run
            return punct_end(b, map, k + l0 + l1);
        } else {
            return None;
        }
    } else {
        return None;
    };
    while j < n {
        // valid utf-8 has every byte of a char its lead byte announces; a
        // word of `bits` covers the 64 code points that share all but the
        // last byte's payload
        let c = b[j];
        if c & 0xF0 == 0xE0 {
            let word = bits[(c as usize & 0x0F) << 6 | (b[j + 1] as usize & 0x3F)];
            if word >> (b[j + 2] & 0x3F) & 1 == 0 {
                break;
            }
            j += 3;
        } else if c & 0xE0 == 0xC0 {
            if bits[c as usize & 0x1F] >> (b[j + 1] & 0x3F) & 1 == 0 {
                break;
            }
            j += 2;
        } else {
            let (f, l) = class_at(map, b, j);
            if !run_class::<CASE>(f) {
                break;
            }
            j += l;
        }
    }
    if CASE && j < n {
        // a case-split word goes on over a cased letter, and may take a
        // contraction suffix
        let (t, _) = class_at(map, b, j);
        if t & (C_UPPER | C_LOWER) != 0 || b[j] == b'\'' {
            return None;
        }
    }
    Some(j)
}

/// End of a punctuation run whose first char ends before `j`, and whose
/// second char (at `j`, if any) starts no word: the punctuation rule takes
/// the whole run. `None` when a newline follows it, which the rule's tail
/// may take.
#[inline(never)]
fn punct_end(b: &[u8], map: &FamilyMap, mut j: usize) -> Option<usize> {
    let n = b.len();
    while j < n {
        let (f, l) = class_at(map, b, j);
        if f & C_PUNCT == 0 {
            break;
        }
        j += l;
    }
    if j < n && matches!(b[j], b'\r' | b'\n') {
        return None;
    }
    Some(j)
}
