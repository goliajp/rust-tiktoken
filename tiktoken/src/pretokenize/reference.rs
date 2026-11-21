//! The regex-based pre-tokenizer the scalar scanners replaced, kept as the
//! reference implementation for the differential tests: the `regex` crate
//! finds each match and [`adjust_whitespace_end`] emulates the `\s+(?!\S)`
//! lookahead the crate cannot express.

use std::sync::LazyLock;

use regex::Regex;

use super::{FastPath, WhitespaceRules, is_ascii_ws};

pub(super) struct Reference {
    regex: Regex,
    ws: WhitespaceRules,
    tekken15: bool,
}

impl Reference {
    pub(super) fn new(pattern: &str, family: FastPath, ws: WhitespaceRules) -> Self {
        Self {
            regex: Regex::new(pattern).unwrap(),
            ws,
            tekken15: family == FastPath::Tekken15,
        }
    }

    pub(super) fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)> {
        if self.tekken15
            && let Some(end) = tekken15_digit_group_end(text, pos)
        {
            return Some((pos, end));
        }
        let mat = self.regex.find_at(text, pos)?;
        let start = mat.start();
        let end = adjust_whitespace_end(text.as_bytes(), start, mat.end(), self.ws);
        Some((start, end))
    }

    pub(super) fn matches(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = vec![];
        let mut pos = 0;
        while pos < text.len() {
            let Some((start, end)) = self.next_match(text, pos) else {
                break;
            };
            out.push((start, end));
            pos = end;
        }
        out
    }
}

/// `\p{N}` in the regex crate's Unicode version. The replaced code asked
/// `char::is_numeric`, whose version follows the toolchain and can be newer;
/// the scanners and this reference both follow the pattern's version.
fn is_numeric(c: char) -> bool {
    static NUM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{N}$").unwrap());
    NUM.is_match(c.encode_utf8(&mut [0; 4]))
}

/// Whether `c` would have been isolated by an earlier stage of the deepseek_v3
/// split pipeline: `\p{N}{1,3}` (stage 1) or the CJK/kana runs (stage 2).
fn is_deepseek_split_boundary(c: char) -> bool {
    is_numeric(c) || matches!(c, '一'..='龥' | '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}')
}

/// Mistral v15's digit groups.
fn tekken15_digit_group_end(text: &str, pos: usize) -> Option<usize> {
    let b = text.as_bytes();
    let first = *b.get(pos)?;
    if first < 0x80 {
        if !first.is_ascii_digit() {
            return None;
        }
    } else if !text[pos..].chars().next().is_some_and(is_numeric) {
        return None;
    }
    let mid_run = text[..pos].chars().next_back().is_some_and(is_numeric);
    let take = if mid_run {
        3
    } else {
        let run = text[pos..].chars().take_while(|&c| is_numeric(c)).count();
        match run % 3 {
            0 => 3,
            r => r,
        }
    };
    let len: usize = text[pos..].chars().take(take).map(char::len_utf8).sum();
    Some(pos + len)
}

/// Emulates `\s+(?!\S)|\s+` from original tiktoken patterns.
///
/// `ws` gates the emulation: only the generic whitespace branch carries the
/// lookahead. For newline-branch patterns a match ending in `\r`/`\n` came from
/// a rule that has none (`\s*[\r\n]+`, or the `[\r\n]*` tail of the punctuation
/// rule) and is returned untouched. For
/// [`WhitespaceRules::NewlineFirstSplitOnNumCjk`] a following digit or CJK char
/// is an upstream split boundary and counts as end-of-input.
pub(super) fn adjust_whitespace_end(
    bytes: &[u8],
    start: usize,
    end: usize,
    ws: WhitespaceRules,
) -> usize {
    if end - start <= 1 || end >= bytes.len() {
        return end;
    }
    if ws != WhitespaceRules::Generic && matches!(bytes[end - 1], b'\r' | b'\n') {
        return end;
    }
    let first = bytes[start];
    if first > 0x20 && first < 0x7F {
        return end;
    }
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
    let piece = &bytes[start..end];
    if piece.iter().all(|&b| is_ascii_ws(b)) {
        let next = bytes[end];
        if is_ascii_ws(next) {
            return end;
        }
        return end - 1;
    }
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
    if end - last_len <= start {
        return end;
    }
    end - last_len
}
