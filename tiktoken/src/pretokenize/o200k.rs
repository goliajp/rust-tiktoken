//! ASCII scanner for the o200k-shaped patterns: case-splitting letter rules,
//! used by o200k, Tekken, MiniMax / Llama 4 and Kimi.

use super::*;

/// ASCII pre-tokenizer for the case-splitting patterns: o200k, Mistral's
/// Tekken, MiniMax and Kimi.
///
/// The letter rules differ from cl100k: both split on case
/// (`[\p{Lu}…]*[\p{Ll}…]+` then `[\p{Lu}…]+[\p{Ll}…]*`, CamelCase-aware).
/// o200k/MiniMax/Kimi attach an optional contraction suffix to the word and
/// use `\p{N}{1,3}`; Tekken has no contraction rule and uses `\p{N}`. On ASCII
/// Kimi differs from o200k only in its punctuation tail.
///
/// `None` hands the piece to the complete scanner: a non-ASCII byte at a
/// decision point, or a whitespace run.
#[inline(always)]
pub(super) fn o200k_like_ascii_next<
    const CONTRACTIONS: bool,
    const MAX_DIGITS: usize,
    const SLASH_TAIL: bool,
>(
    b: &[u8],
    i: usize,
) -> Option<(usize, usize)> {
    let n = b.len();
    if i >= n {
        return None;
    }
    let c0 = b[i];
    if c0 >= 0x80 {
        return None;
    }

    // Determine the letter start `p`: either c0 itself (a letter), or one
    // leading non-CRLF non-alnum char followed by an ASCII letter.
    let p = if c0.is_ascii_alphabetic() {
        i
    } else if c0 != b'\r' && c0 != b'\n' && !c0.is_ascii_digit() {
        // eligible leading char (punct/space). The word rules apply only if a
        // letter follows; otherwise it's a digit/punct/ws piece.
        match b.get(i + 1) {
            Some(&c1) if c1 < 0x80 && c1.is_ascii_alphabetic() => i + 1,
            Some(&c1) if c1 >= 0x80 => return None,
            _ => return ascii_num_punct::<MAX_DIGITS, SLASH_TAIL>(b, i),
        }
    } else {
        // digit, or \r\n
        return ascii_num_punct::<MAX_DIGITS, SLASH_TAIL>(b, i);
    };

    // Scan the uppercase run from `p`; a non-ASCII byte at any decision point
    // hands the whole piece over to the complete scanner.
    let mut q = p;
    while q < n && b[q] < 0x80 && b[q].is_ascii_uppercase() {
        q += 1;
    }
    if q < n && b[q] >= 0x80 {
        return None;
    }

    let letters_end = if q > p {
        // started with uppercase(s)
        if q < n && b[q].is_ascii_lowercase() {
            // Rule A: [A-Z]*[a-z]+  (greedy uppercase, then lowercase run)
            let mut r = q;
            while r < n && b[r] < 0x80 && b[r].is_ascii_lowercase() {
                r += 1;
            }
            if r < n && b[r] >= 0x80 {
                return None;
            }
            r
        } else {
            // Rule B: [A-Z]+  (no trailing lowercase)
            q
        }
    } else {
        // b[p] is lowercase (it's a letter but not uppercase): Rule A lowercase+
        let mut r = p;
        while r < n && b[r] < 0x80 && b[r].is_ascii_lowercase() {
            r += 1;
        }
        if r < n && b[r] >= 0x80 {
            return None;
        }
        r
    };

    // Optional contraction suffix attached to the word: (?i:'s|'t|…)?
    // Tekken has no contraction rule at all, so it never extends the word here.
    let mut end = letters_end;
    if CONTRACTIONS
        && end < n
        && b[end] == b'\''
        && let Some(len) = match_contraction(b, end)
    {
        end += len;
    }
    Some((i, end))
}
