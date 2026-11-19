//! Emulation of the `\s+(?!\S)` lookahead, which the `regex` crate cannot express.

use super::*;

/// Emulates `\s+(?!\S)|\s+` from original tiktoken patterns.
/// Pure byte-level fast path for ASCII whitespace, char-level fallback for Unicode.
///
/// `ws` gates the emulation: only the generic whitespace branch carries the
/// lookahead. For newline-branch patterns a match ending in `\r`/`\n` came from
/// a rule that has none (`\s*[\r\n]+`, or the `[\r\n]*` tail of the punctuation
/// rule) and is returned untouched. For
/// [`WhitespaceRules::NewlineFirstSplitOnNumCjk`] a following digit or CJK char
/// is an upstream split boundary and counts as end-of-input.
#[inline]
pub(super) fn adjust_whitespace_end(
    bytes: &[u8],
    start: usize,
    end: usize,
    ws: WhitespaceRules,
) -> usize {
    if end - start <= 1 || end >= bytes.len() {
        return end;
    }

    // Newline-branch patterns: a match ending in \r/\n never carries the
    // lookahead, so it keeps its full extent (canonical "\n\n" / "\r\n" tokens).
    if ws != WhitespaceRules::Generic && matches!(bytes[end - 1], b'\r' | b'\n') {
        return end;
    }

    // fast reject: if first byte is printable ASCII (0x21..0x7E), not whitespace
    let first = bytes[start];
    if first > 0x20 && first < 0x7F {
        return end;
    }

    // deepseek_v3: an upstream split boundary right after the run terminates the
    // slice the lookahead would have seen, so the run keeps its full extent.
    if ws == WhitespaceRules::NewlineFirstSplitOnNumCjk
        && let Some(next) = bytes[end..].iter().next()
        && (next.is_ascii_digit() || *next >= 0x80)
        && let Some(c) = std::str::from_utf8(&bytes[end..])
            .ok()
            .and_then(|s| s.chars().next())
        && is_deepseek_split_boundary(c)
    {
        return end;
    }

    // ASCII fast path
    // safety: end < bytes.len() is guaranteed by the early return above
    let piece = &bytes[start..end];
    if piece.iter().all(|&b| is_ascii_ws(b)) {
        let next = bytes[end];
        if is_ascii_ws(next) {
            return end;
        }
        return end - 1;
    }

    // unicode slow path
    // safety: regex::Match boundaries are always valid UTF-8 since input is &str
    let matched = std::str::from_utf8(&bytes[start..end]).unwrap();
    if !matched.chars().all(|c| c.is_whitespace()) {
        return end;
    }
    let tail = std::str::from_utf8(&bytes[end..]).unwrap();
    let next_char = match tail.chars().next() {
        Some(c) => c,
        None => return end,
    };
    if next_char.is_whitespace() {
        return end;
    }
    let last_len = matched.chars().next_back().unwrap().len_utf8();
    // don't trim if it would make the piece empty (single multi-byte whitespace char)
    if end - last_len <= start {
        return end;
    }
    end - last_len
}
