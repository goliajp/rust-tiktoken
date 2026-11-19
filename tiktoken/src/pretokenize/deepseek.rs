//! Scanner and split-boundary helpers for the deepseek_v3 / deepseek_v4 pattern.

use super::*;

/// Rule 2 of `DEEPSEEK_V3_PATTERN`, by codepoint: `[一-龥\x{3040}-\x{309F}\x{30A0}-\x{30FF}]`.
#[inline]
pub(super) fn is_deepseek_cjk(c: u32) -> bool {
    matches!(c, 0x4E00..=0x9FA5 | 0x3040..=0x30FF)
}

/// Whether `c` would have been isolated by an earlier stage of the deepseek_v3
/// split pipeline: `\p{N}{1,3}` (stage 1) or `[一-龥\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+`
/// (stage 2). Kept in sync with `DEEPSEEK_V3_PATTERN`'s first two alternatives.
#[inline]
pub(super) fn is_deepseek_split_boundary(c: char) -> bool {
    c.is_numeric() || matches!(c, '一'..='龥' | '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}')
}

/// ASCII fast-path pre-tokenizer for the deepseek_v3 pattern.
///
/// Pattern (leftmost-first): `\p{N}{1,3}` | CJK/kana+ | `[ascii-punct][A-Za-z]+`
/// | `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` | ` ?[\p{P}\p{S}]+[\r\n]*` | `\s*[\r\n]+`
/// | `\s+` | `[\s\S]`. ASCII `[\p{P}\p{S}]` is exactly `u8::is_ascii_punctuation()`.
///
/// Conservative: resolves digits, letters, punct+letters, punct runs, and the
/// common space-led letter/punct pieces; defers to the regex on any non-ASCII
/// byte, whitespace/control start, or catch-all case (deferral is always safe).
#[inline]
pub(super) fn deepseek_ascii_next(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let n = b.len();
    if i >= n {
        return None;
    }
    let c0 = b[i];
    if c0 >= 0x80 {
        // Rule 2: `[一-龥\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+`. Unlike the
        // category-based rules, these are exact ranges, so membership is
        // certain in both directions and the run never needs to defer.
        let (ch, len) = decode_char(b, i);
        if is_deepseek_cjk(ch) {
            let mut j = i + len;
            while j < n && b[j] >= 0x80 {
                let (c2, l2) = decode_char(b, j);
                if is_deepseek_cjk(c2) {
                    j += l2;
                } else {
                    break;
                }
            }
            return Some((i, j));
        }
        return None; // other non-ASCII → defer
    }

    // Rule 1: \p{N}{1,3}
    if c0.is_ascii_digit() {
        let mut j = i;
        let mut k = 0;
        while j < n && k < 3 && b[j].is_ascii_digit() {
            j += 1;
            k += 1;
        }
        if k < 3 && j < n && b[j] >= 0x80 {
            return None; // a Unicode \p{N} could extend the run
        }
        return Some((i, j));
    }

    // Rule 3: [ascii-punct][A-Za-z]+ (one punct glued to a letter run).
    // Rule 5: ` ?[\p{P}\p{S}]+[\r\n]*` (here with no leading space — c0 is punct).
    if c0.is_ascii_punctuation() {
        if let Some(&c1) = b.get(i + 1)
            && c1 < 0x80
            && c1.is_ascii_alphabetic()
        {
            // Rule 3 — note its letters are [A-Za-z], so a non-ASCII byte simply
            // ends the run (no defer needed).
            let mut j = i + 1;
            while j < n && b[j].is_ascii_alphabetic() {
                j += 1;
            }
            return Some((i, j));
        }
        // Rule 5: punctuation/symbol run, then trailing newlines.
        let mut k = i;
        while k < n && b[k] < 0x80 && b[k].is_ascii_punctuation() {
            k += 1;
        }
        if k < n && b[k] >= 0x80 {
            return None; // a Unicode \p{P}/\p{S} could extend the run
        }
        k = take_line_tail::<false>(b, k);
        return Some((i, k));
    }

    // Rule 4 (no leading char): [\p{L}\p{M}]+ — for ASCII, a letter run.
    if c0.is_ascii_alphabetic() {
        let mut j = i;
        while j < n && b[j].is_ascii_alphabetic() {
            j += 1;
        }
        if j < n && b[j] >= 0x80 {
            return None; // a Unicode letter/mark could extend the run
        }
        return Some((i, j));
    }

    // Leading space: Rule 4 (space + letters) or Rule 5 (space + punct run).
    if c0 == b' ' {
        match b.get(i + 1) {
            Some(&c1) if c1 >= 0x80 => return None, // unicode letter/punct ambiguous
            Some(&c1) if c1.is_ascii_alphabetic() => {
                let mut j = i + 1;
                while j < n && b[j].is_ascii_alphabetic() {
                    j += 1;
                }
                if j < n && b[j] >= 0x80 {
                    return None;
                }
                return Some((i, j));
            }
            Some(&c1) if c1.is_ascii_punctuation() => {
                let mut k = i + 1;
                while k < n && b[k] < 0x80 && b[k].is_ascii_punctuation() {
                    k += 1;
                }
                if k < n && b[k] >= 0x80 {
                    return None;
                }
                k = take_line_tail::<false>(b, k);
                return Some((i, k));
            }
            // space followed by digit/space/eof → whitespace rules → defer
            _ => return None,
        }
    }

    // other whitespace, control chars, catch-all → defer to regex
    None
}
