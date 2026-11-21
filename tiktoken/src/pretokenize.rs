//! Pre-tokenization: split text into pieces before BPE encoding.
//!
//! Each pattern family has a complete scalar scanner (`scalar_*.rs`) that
//! reproduces the family's split pattern, with Unicode classes from the
//! generated tables in `crate::unicode`. ASCII scanners (`cl100k.rs`,
//! `o200k.rs`, `deepseek.rs`) are tried first and resolve the common pieces
//! inline; whatever they cannot decide goes to the complete scanner.

mod bitops;
mod chains;
mod cl100k;
pub(crate) mod classify;
#[cfg(target_arch = "aarch64")]
mod classify_neon;
#[cfg(any(
    test,
    not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
mod classify_portable;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod classify_wasm;
#[cfg(target_arch = "x86_64")]
mod classify_x86;
mod deepseek;
mod digits;
mod o200k;
mod scalar;
mod scalar_case;
mod scalar_cl100k;
mod scalar_deepseek;
mod scalar_p50k;
pub(crate) mod window;
mod window_case;
pub(crate) mod windowed;

#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_exhaustive;
#[cfg(test)]
mod tests_window;

use cl100k::*;
use deepseek::*;
use digits::*;
use o200k::*;
use scalar_case::case_piece;
use scalar_cl100k::cl100k_piece;
use scalar_deepseek::deepseek_piece;
use scalar_p50k::p50k_piece;

use crate::unicode::{MAP_BASE, MAP_KIMI, MAP_QWEN35, decode};

/// Trait for splitting text into pieces before BPE encoding.
pub trait PreTokenizer: Send + Sync {
    /// The piece starting at `pos`, as `(start, end)` byte offsets into
    /// `text`; `None` at the end of the text.
    fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)>;
}

/// Which split pattern a [`ScalarPreTokenizer`] implements. Chosen by the
/// caller in `encoding.rs`, which owns the pattern definitions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FastPath {
    /// p50k_base / r50k_base pattern.
    P50k,
    /// cl100k_base / llama3 / glm pattern.
    Cl100k,
    /// o200k_base / o200k_harmony pattern.
    O200k,
    /// qwen2 pattern: identical to cl100k except `\p{N}` matches a single digit
    /// (not 1-3).
    Qwen2,
    /// qwen3_5 pattern: qwen2's, with `[\p{L}\p{M}]+` words and no marks in
    /// the punctuation rule.
    Qwen35,
    /// deepseek_v3 pattern (digits, CJK, punct+letters, letters, punct runs).
    Deepseek,
    /// mistral_v3 (Tekken) pattern: o200k-style case splitting, but with no
    /// contraction rule, single-digit `\p{N}`, and a `[\r\n/]*` punctuation tail.
    Tekken,
    /// mistral_v15 (Tekken v15) pattern: Tekken's rules, except that digit runs
    /// split into right-aligned groups of three (see `digits.rs`).
    Tekken15,
    /// minimax_m2 / llama4 pattern: o200k's letter/digit rules (contractions
    /// included) with Tekken's `[\r\n/]*` punctuation tail.
    MiniMax,
    /// kimi_k2 / kimi_k3 pattern: a leading `[\p{Han}]+` branch, o200k's word
    /// rules without Han, and the plain `[\r\n]*` punctuation tail.
    Kimi,
}

/// Which whitespace rules a pattern uses. The scanners derive this from
/// [`FastPath`]; `encoding.rs` still names it next to each pattern because
/// the regex reference used by the tests needs it to emulate `\s+(?!\S)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WhitespaceRules {
    /// The pattern's only whitespace handling is the generic `\s+(?!\S)|\s+`
    /// (p50k_base / r50k_base).
    Generic,
    /// The pattern has a dedicated `\s*[\r\n]+` branch ordered *before* the
    /// generic `\s+(?!\S)|\s+` (cl100k, o200k, qwen2, ...). A run containing
    /// a newline is claimed by that branch, which has no lookahead.
    NewlineFirst,
    /// deepseek_v3: as [`Self::NewlineFirst`], but upstream runs `\p{N}{1,3}`
    /// and the CJK/kana runs as earlier split stages, so a digit or CJK/kana
    /// char after a whitespace run acts as end-of-input for the lookahead.
    NewlineFirstSplitOnNumCjk,
}

/// The pre-tokenizer of every encoding: ASCII scanner first, complete scalar
/// scanner for the rest.
pub struct ScalarPreTokenizer {
    fast: FastPath,
    /// The regex this pre-tokenizer reproduces, kept for the differential
    /// tests.
    #[cfg(test)]
    pub(crate) pattern: String,
    #[cfg(test)]
    pub(crate) ws: WhitespaceRules,
}

impl ScalarPreTokenizer {
    pub(crate) fn new(pattern: &str, fast: FastPath, ws: WhitespaceRules) -> Self {
        #[cfg(not(test))]
        let _ = (pattern, ws);
        Self {
            fast,
            #[cfg(test)]
            pattern: pattern.to_string(),
            #[cfg(test)]
            ws,
        }
    }

    #[cfg(test)]
    pub(crate) fn family(&self) -> FastPath {
        self.fast
    }

    /// The window rules of this pattern, if it has them.
    #[inline(always)]
    pub(crate) fn window_kind(&self) -> Option<WindowKind> {
        match self.fast {
            FastPath::Cl100k => Some(WindowKind::Cl100k),
            FastPath::Qwen2 => Some(WindowKind::Qwen2),
            FastPath::Qwen35 => Some(WindowKind::Qwen35),
            FastPath::O200k | FastPath::MiniMax => Some(WindowKind::O200k),
            FastPath::Tekken => Some(WindowKind::Tekken),
            _ => None,
        }
    }
}

/// Patterns with window rules (`window.rs`, `window_case.rs`). The others
/// look behind a piece start (mistral_v15's digit groups, deepseek's
/// whitespace) or have no window rules yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WindowKind {
    Cl100k,
    Qwen2,
    Qwen35,
    O200k,
    Tekken,
}

impl PreTokenizer for ScalarPreTokenizer {
    #[inline(always)]
    fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)> {
        let bytes = text.as_bytes();
        let fast = match self.fast {
            FastPath::Cl100k => cl100k_ascii_next::<3>(bytes, pos),
            FastPath::Qwen2 | FastPath::Qwen35 => cl100k_ascii_next::<1>(bytes, pos),
            FastPath::O200k => o200k_like_ascii_next::<true, 3, true>(bytes, pos),
            FastPath::Tekken => o200k_like_ascii_next::<false, 1, true>(bytes, pos),
            FastPath::Tekken15 => return tekken15_next(bytes, pos),
            FastPath::MiniMax => o200k_like_ascii_next::<true, 3, true>(bytes, pos),
            FastPath::Kimi => o200k_like_ascii_next::<true, 3, false>(bytes, pos),
            FastPath::Deepseek => deepseek_ascii_next(bytes, pos),
            FastPath::P50k => None,
        };
        if let Some(r) = fast {
            return Some(r);
        }
        if pos >= bytes.len() {
            return None;
        }
        Some((pos, complete_piece(self.fast, bytes, pos)))
    }
}

/// The complete scanner of `family` at `pos < b.len()`.
#[inline(never)]
pub(crate) fn complete_piece(family: FastPath, b: &[u8], pos: usize) -> usize {
    match family {
        FastPath::P50k => p50k_piece(b, pos),
        FastPath::Cl100k => cl100k_piece::<3>(&MAP_BASE, b, pos),
        FastPath::Qwen2 => cl100k_piece::<1>(&MAP_BASE, b, pos),
        FastPath::Qwen35 => cl100k_piece::<1>(&MAP_QWEN35, b, pos),
        FastPath::O200k => case_piece::<true, 3, true, false>(&MAP_BASE, b, pos),
        FastPath::Tekken | FastPath::Tekken15 => {
            case_piece::<false, 1, true, false>(&MAP_BASE, b, pos)
        }
        FastPath::MiniMax => case_piece::<true, 3, true, false>(&MAP_BASE, b, pos),
        FastPath::Kimi => case_piece::<true, 3, false, true>(&MAP_KIMI, b, pos),
        FastPath::Deepseek => deepseek_piece(b, pos),
    }
}

/// Kept out of line: inlining the digit rule into `next_match` grew it
/// enough to cost the other encodings' `encode` loop ~20% on ASCII text.
#[inline(never)]
fn tekken15_next(b: &[u8], pos: usize) -> Option<(usize, usize)> {
    if let Some(end) = tekken15_digit_group_end(b, pos) {
        return Some((pos, end));
    }
    if let Some(r) = o200k_like_ascii_next::<false, 1, true>(b, pos) {
        return Some(r);
    }
    if pos >= b.len() {
        return None;
    }
    Some((pos, complete_piece(FastPath::Tekken15, b, pos)))
}

/// Consume the trailing line-tail class of the punctuation rule starting at `k`,
/// returning the new offset. Most patterns spell it `[\r\n]*`; Mistral's Tekken
/// pattern spells it `[\r\n/]*`, so `SLASH` admits `/` as well.
#[inline]
fn take_line_tail<const SLASH: bool>(b: &[u8], mut k: usize) -> usize {
    while k < b.len() && (b[k] == b'\r' || b[k] == b'\n' || (SLASH && b[k] == b'/')) {
        k += 1;
    }
    k
}

/// Shared ASCII handler for the digit and punctuation rules, parameterized at
/// compile time so every [`FastPath`] keeps fully specialized codegen: the
/// digit rule's repeat cap is `MAX_DIGITS` (`\p{N}{1,3}` → 3 for cl100k/o200k,
/// `\p{N}` → 1 for qwen2/Tekken), and the punctuation rule
/// (` ?[^\s\p{L}\p{N}]+[\r\n]*`) admits `/` in its trailing class when
/// `SLASH_TAIL` is set (Tekken's `[\r\n/]*`).
///
/// Returns `Some((i, end))` on a match, or `None` for the complete scanner
/// (the start is whitespace, or a non-ASCII byte could extend the run under
/// Unicode semantics). Caller guarantees `i < n` and `b[i] < 0x80`.
#[inline(always)]
fn ascii_num_punct<const MAX_DIGITS: usize, const SLASH_TAIL: bool>(
    b: &[u8],
    i: usize,
) -> Option<(usize, usize)> {
    let n = b.len();
    let c0 = b[i];

    // Rule: \p{N}{1,MAX_DIGITS}
    if c0.is_ascii_digit() {
        let mut j = i;
        let mut k = 0;
        while j < n && k < MAX_DIGITS && b[j] < 0x80 && b[j].is_ascii_digit() {
            j += 1;
            k += 1;
        }
        // Fewer than max digits and a non-ASCII byte next: it may be a Unicode
        // \p{N} (superscripts, other-number) that extends the run.
        // At the cap the rule stops regardless, so it's safe to return.
        if k < MAX_DIGITS && j < n && b[j] >= 0x80 {
            return None;
        }
        return Some((i, j));
    }

    // Rule: ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let mut j = i;
    if c0 == b' ' {
        // optional single leading space, but only if a punct run follows
        match b.get(i + 1) {
            Some(&c1)
                if c1 < 0x80
                    && !is_ascii_ws(c1)
                    && !c1.is_ascii_alphabetic()
                    && !c1.is_ascii_digit() =>
            {
                j = i + 1;
            }
            // space not followed by ASCII punct → whitespace rules or a
            // non-ASCII char → complete scanner
            _ => return None,
        }
    }
    let cj = b[j];
    if cj < 0x80 && !is_ascii_ws(cj) && !cj.is_ascii_alphabetic() && !cj.is_ascii_digit() {
        let mut k = j;
        while k < n
            && b[k] < 0x80
            && !is_ascii_ws(b[k])
            && !b[k].is_ascii_alphabetic()
            && !b[k].is_ascii_digit()
        {
            k += 1;
        }
        if k < n && b[k] >= 0x80 {
            return None;
        }
        k = take_line_tail::<SLASH_TAIL>(b, k);
        return Some((i, k));
    }

    // whitespace run (or other) → complete scanner
    None
}

/// Match a contraction at `b[i] == '\''`, returning its byte length (2 or 3) or
/// `None`. Case-insensitive, matching `(?i:'s|'t|'re|'ve|'m|'ll|'d)`. Shared by
/// both patterns (standalone alternative in cl100k, word suffix in o200k).
#[inline]
fn match_contraction(b: &[u8], i: usize) -> Option<usize> {
    let c1 = b.get(i + 1).copied()?.to_ascii_lowercase();
    match c1 {
        b's' | b't' | b'm' | b'd' => Some(2),
        b'r' if b.get(i + 2).map(|c| c.to_ascii_lowercase()) == Some(b'e') => Some(3),
        b'v' if b.get(i + 2).map(|c| c.to_ascii_lowercase()) == Some(b'e') => Some(3),
        b'l' if b.get(i + 2).map(|c| c.to_ascii_lowercase()) == Some(b'l') => Some(3),
        _ => None,
    }
}

#[inline(always)]
const fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}
