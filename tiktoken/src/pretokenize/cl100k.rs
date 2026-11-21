//! ASCII scanner for the cl100k-shaped patterns (cl100k_base, llama3,
//! glm4/glm5, and qwen2 / qwen3_5 with a one-digit cap).

use super::*;

/// ASCII pre-tokenizer for the cl100k pattern (and qwen2 / qwen3_5, which
/// agree with it on ASCII except `MAX_DIGITS = 1` instead of 3).
///
/// Returns `Some((pos, end))` for a piece it can resolve entirely within ASCII,
/// or `None` to hand the piece to the complete scanner (a non-ASCII byte at a
/// decision point, or a whitespace run). Alternatives are tried in the
/// pattern's leftmost-first order.
#[inline(always)]
pub(super) fn cl100k_ascii_next<const MAX_DIGITS: usize>(
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

    // Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d). On no contraction, fall through; the
    // quote may act as a rule-2 leading char or a rule-4 punctuation run.
    if c0 == b'\''
        && let Some(len) = match_contraction(b, i)
    {
        return Some((i, i + len));
    }

    // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
    // case A: one leading non-CRLF non-alnum char, then letters
    if c0 != b'\r'
        && c0 != b'\n'
        && !c0.is_ascii_alphabetic()
        && !c0.is_ascii_digit()
        && let Some(&c1) = b.get(i + 1)
    {
        if c1 < 0x80 && c1.is_ascii_alphabetic() {
            let mut j = i + 2;
            while j < n && b[j] < 0x80 && b[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j < n && b[j] >= 0x80 {
                return None;
            }
            return Some((i, j));
        }
        if c1 >= 0x80 {
            return None;
        }
    }
    // case B: no leading char, c0 is a letter
    if c0.is_ascii_alphabetic() {
        let mut j = i + 1;
        while j < n && b[j] < 0x80 && b[j].is_ascii_alphabetic() {
            j += 1;
        }
        if j < n && b[j] >= 0x80 {
            return None;
        }
        return Some((i, j));
    }

    // Rules 3 & 4: digits, punctuation. Rules 5/6 (whitespace) → defer.
    ascii_num_punct::<MAX_DIGITS, false>(b, i)
}
