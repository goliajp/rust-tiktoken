// pre-tokenization: split text into pieces before BPE

use regex::Regex;

/// Trait for splitting text into pieces before BPE encoding.
pub trait PreTokenizer: Send + Sync {
    /// Find the next match starting at or after `pos`.
    /// Returns `(start, end)` byte offsets into `text`.
    /// The `end` is already adjusted for whitespace lookahead emulation.
    fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)>;
}

/// Regex-based pre-tokenizer wrapping the existing regex + whitespace adjustment logic.
pub struct RegexPreTokenizer {
    regex: Regex,
}

impl RegexPreTokenizer {
    pub(crate) fn new(pattern: &str) -> Self {
        Self {
            regex: Regex::new(pattern).expect("invalid regex pattern"),
        }
    }
}

impl PreTokenizer for RegexPreTokenizer {
    #[inline]
    fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)> {
        let mat = self.regex.find_at(text, pos)?;
        let start = mat.start();
        let end = adjust_whitespace_end(text.as_bytes(), start, mat.end());
        Some((start, end))
    }
}

/// Emulates `\s+(?!\S)|\s+` from original tiktoken patterns.
/// Pure byte-level fast path for ASCII whitespace, char-level fallback for Unicode.
#[inline]
fn adjust_whitespace_end(bytes: &[u8], start: usize, end: usize) -> usize {
    if end - start <= 1 || end >= bytes.len() {
        return end;
    }

    // fast reject: if first byte is printable ASCII (0x21..0x7E), not whitespace
    let first = bytes[start];
    if first > 0x20 && first < 0x7F {
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

#[inline(always)]
const fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_matches(pt: &dyn PreTokenizer, text: &str) -> Vec<(usize, usize)> {
        let mut result = vec![];
        let mut pos = 0;
        while let Some((start, end)) = pt.next_match(text, pos) {
            result.push((start, end));
            pos = end;
        }
        result
    }

    // patterns from encoding.rs
    const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";
    const O200K_PATTERN: &str = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|\p{N}{1,3}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"|\s*[\r\n]+",
        r"|\s+",
    );
    const P50K_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

    // verify RegexPreTokenizer matches the v2 behavior by comparing
    // piece-by-piece with v2's regex.find_at + adjust_whitespace_end
    fn v2_collect_matches(pattern: &str, text: &str) -> Vec<(usize, usize)> {
        let regex = Regex::new(pattern).unwrap();
        let bytes = text.as_bytes();
        let mut result = vec![];
        let mut pos = 0;
        while pos < text.len() {
            let mat = match regex.find_at(text, pos) {
                Some(m) => m,
                None => break,
            };
            let start = mat.start();
            let end = adjust_whitespace_end(bytes, start, mat.end());
            result.push((start, end));
            pos = end;
        }
        result
    }

    #[test]
    fn test_cl100k_english() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        let v2 = v2_collect_matches(CL100K_PATTERN, "Hello, world!");
        let v3 = collect_matches(&pt, "Hello, world!");
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_cl100k_cjk() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        let text = "你好世界";
        let v2 = v2_collect_matches(CL100K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_cl100k_contractions() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        let text = "I'm don't they're we've she'll it'd";
        let v2 = v2_collect_matches(CL100K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_o200k_english() {
        let pt = RegexPreTokenizer::new(O200K_PATTERN);
        let text = "Hello, world! CamelCase mixedScript123";
        let v2 = v2_collect_matches(O200K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_p50k_english() {
        let pt = RegexPreTokenizer::new(P50K_PATTERN);
        let text = "Hello world, I'm testing!";
        let v2 = v2_collect_matches(P50K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_empty_input() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        assert_eq!(collect_matches(&pt, ""), vec![]);
    }

    #[test]
    fn test_only_whitespace() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        let text = "   \n  \t  ";
        let v2 = v2_collect_matches(CL100K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_emoji() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        let text = "🎉🚀💡";
        let v2 = v2_collect_matches(CL100K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_mixed_script() {
        let pt = RegexPreTokenizer::new(CL100K_PATTERN);
        let text = "Hello 你好 World 🌍";
        let v2 = v2_collect_matches(CL100K_PATTERN, text);
        let v3 = collect_matches(&pt, text);
        assert_eq!(v2, v3);
    }

    // whitespace adjustment tests (migrated from v2 bpe.rs)

    #[test]
    fn test_adjust_whitespace_single_byte() {
        assert_eq!(adjust_whitespace_end(b"a b", 0, 1), 1);
    }

    #[test]
    fn test_adjust_whitespace_at_end_of_input() {
        assert_eq!(adjust_whitespace_end(b"  ", 0, 2), 2);
    }

    #[test]
    fn test_adjust_whitespace_non_ws_piece() {
        assert_eq!(adjust_whitespace_end(b"hello world", 0, 5), 5);
    }

    #[test]
    fn test_adjust_whitespace_trim_before_nonws() {
        let bytes = b"  x";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2), 1);
    }

    #[test]
    fn test_adjust_whitespace_no_trim_before_ws() {
        let bytes = b"   ";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2), 2);
    }

    #[test]
    fn test_adjust_whitespace_unicode_slow_path() {
        let input = "\u{3000}\u{3000}x";
        let bytes = input.as_bytes();
        assert_eq!(adjust_whitespace_end(bytes, 0, 6), 3);
    }

    #[test]
    fn test_adjust_whitespace_unicode_followed_by_unicode_ws() {
        let input = "\u{3000}\u{3000}\u{3000}";
        let bytes = input.as_bytes();
        assert_eq!(adjust_whitespace_end(bytes, 0, 6), 6);
    }

    #[test]
    fn test_adjust_whitespace_single_multibyte_ws_before_nonws() {
        // U+3000 (ideographic space, 3 bytes) followed by 'x'
        // trimming the last char would make the piece empty, so it should NOT trim
        let input = "\u{3000}x";
        let bytes = input.as_bytes();
        // piece is bytes[0..3] (the ideographic space), next char is 'x' (non-ws)
        // without the protection, this would trim to bytes[0..0] which is empty
        assert_eq!(adjust_whitespace_end(bytes, 0, 3), 3);
    }

    // comprehensive comparison: run many inputs through all patterns
    #[test]
    fn test_all_patterns_match_v2() {
        let texts = vec![
            "Hello, world!",
            "你好世界",
            "fn main() { }",
            "  hello  ",
            "line1\nline2\n",
            "café résumé",
            "100% of $1,000",
            "a@b.com",
            "   \t\n   ",
            "",
            "a",
            "hello world! 你好 🚀 test 123",
        ];

        for &pattern in &[CL100K_PATTERN, O200K_PATTERN, P50K_PATTERN] {
            let pt = RegexPreTokenizer::new(pattern);
            for text in &texts {
                let v2 = v2_collect_matches(pattern, text);
                let v3 = collect_matches(&pt, text);
                assert_eq!(v2, v3, "mismatch for pattern / text: {text:?}");
            }
        }
    }
}
