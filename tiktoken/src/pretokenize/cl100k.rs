//! ASCII-first scanners for the cl100k-shaped patterns (cl100k_base, llama3,
//! glm4/glm5, and qwen2 / qwen3_5 with a one-digit cap).

use super::*;

/// A space at `i` followed by a non-ASCII char: the punct rule applies only
/// if that char is certainly punctuation. Out of line with the other CJK arms.
#[cold]
#[inline(never)]
pub(super) fn space_cjk_punct<const SLASH_TAIL: bool>(
    b: &[u8],
    i: usize,
) -> Option<(usize, usize)> {
    if cjk_class(decode_char(b, i + 1).0) != CjkClass::Punct {
        return None;
    }
    let e = scan_punct_run_mixed(b, i + 1)?;
    Some((i, take_line_tail::<SLASH_TAIL>(b, e)))
}

/// cl100k-family piece starting on a non-ASCII char. Same leftmost-first rule
/// order as the ASCII path: rule 2 (optional leading char + letters), then
/// rule 4 (punct run). Out of line so CJK support does not bloat the inlined
/// ASCII scanner.
#[cold]
#[inline(never)]
pub(super) fn cl100k_cjk_next(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let n = b.len();
    let (ch, len) = decode_char(b, i);
    match cjk_class(ch) {
        // rule 2, no leading char
        CjkClass::Han | CjkClass::Caseless | CjkClass::Upper | CjkClass::Lower => {
            scan_letter_run_mixed(b, i + len).map(|e| (i, e))
        }
        // 、 or 　 can be rule 2's leading char `[^\r\n\p{L}\p{N}]` when
        // letters follow; a lone 、 falls to the punct rule, a lone 　 to the
        // whitespace rules (defer).
        CjkClass::Punct | CjkClass::Ws => {
            let j = i + len;
            let next_is_letter = j < n && {
                let c1 = b[j];
                if c1 < 0x80 {
                    c1.is_ascii_alphabetic()
                } else {
                    matches!(
                        cjk_class(decode_char(b, j).0),
                        CjkClass::Han | CjkClass::Caseless | CjkClass::Upper | CjkClass::Lower
                    )
                }
            };
            if next_is_letter {
                return scan_letter_run_mixed(b, j).map(|e| (i, e));
            }
            if cjk_class(ch) == CjkClass::Ws {
                return None; // whitespace rules → regex
            }
            let e = scan_punct_run_mixed(b, i + len)?;
            Some((i, take_line_tail::<false>(b, e)))
        }
        CjkClass::Num | CjkClass::Other => None,
    }
}

/// cl100k-family: an eligible ASCII leading char at `i` with a non-ASCII char
/// after it. Rule 2 if that char is a letter; otherwise the digit/punct rules
/// via [`ascii_num_punct`], whose own cold escapes finish any CJK punct run.
#[cold]
#[inline(never)]
pub(super) fn cl100k_cjk_after_lead<const MAX_DIGITS: usize>(
    b: &[u8],
    i: usize,
) -> Option<(usize, usize)> {
    if matches!(
        cjk_class(decode_char(b, i + 1).0),
        CjkClass::Han | CjkClass::Caseless | CjkClass::Upper | CjkClass::Lower
    ) {
        return scan_letter_run_mixed(b, i + 1).map(|e| (i, e));
    }
    ascii_num_punct::<MAX_DIGITS, false>(b, i)
}

/// ASCII fast-path pre-tokenizer for the cl100k pattern (and qwen2, which is
/// identical except `MAX_DIGITS = 1` instead of 3).
///
/// Returns `Some((pos, end))` for a piece it can resolve entirely within ASCII,
/// or `None` to defer to the regex (non-ASCII byte at a decision point, or a
/// whitespace-run start whose `\s*[\r\n]+|\s+` + lookahead semantics we don't
/// replicate here). Alternatives are tried in the regex's leftmost-first order.
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
        return cl100k_cjk_next(b, i);
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
                return scan_letter_run_mixed(b, j).map(|e| (i, e));
            }
            return Some((i, j));
        }
        if c1 >= 0x80 {
            return cl100k_cjk_after_lead::<MAX_DIGITS>(b, i);
        }
    }
    // case B: no leading char, c0 is a letter
    if c0.is_ascii_alphabetic() {
        let mut j = i + 1;
        while j < n && b[j] < 0x80 && b[j].is_ascii_alphabetic() {
            j += 1;
        }
        if j < n && b[j] >= 0x80 {
            return scan_letter_run_mixed(b, j).map(|e| (i, e));
        }
        return Some((i, j));
    }

    // Rules 3 & 4: digits, punctuation. Rules 5/6 (whitespace) → defer.
    ascii_num_punct::<MAX_DIGITS, false>(b, i)
}
