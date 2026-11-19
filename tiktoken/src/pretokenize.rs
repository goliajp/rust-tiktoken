//! Pre-tokenization: split text into pieces before BPE encoding.
//!
//! The [`PreTokenizer`] trait defines a regex-based splitter that partitions
//! input text into byte-range pieces. [`RegexPreTokenizer`] implements this
//! using the `regex` crate (DFA-based), with a custom whitespace lookahead
//! emulation (`adjust_whitespace_end`) that avoids `fancy-regex` entirely.

use regex::Regex;

mod cl100k;
mod deepseek;
mod digits;
mod o200k;
mod whitespace;

use cl100k::*;
use deepseek::*;
use digits::*;
use o200k::*;
use whitespace::*;

/// Trait for splitting text into pieces before BPE encoding.
pub trait PreTokenizer: Send + Sync {
    /// Find the next match starting at or after `pos`.
    /// Returns `(start, end)` byte offsets into `text`.
    /// The `end` is already adjusted for whitespace lookahead emulation.
    fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)>;
}

/// Selects which ASCII fast-path scanner (if any) a [`RegexPreTokenizer`] tries
/// before falling back to the regex. Chosen by the caller in `encoding.rs`,
/// which owns the pattern definitions — the pre-tokenizer itself stays unaware
/// of any specific encoding's pattern string.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FastPath {
    /// No fast path: always use the regex (for patterns without a scanner).
    None,
    /// cl100k_base / llama3 pattern.
    Cl100k,
    /// o200k_base / o200k_harmony pattern.
    O200k,
    /// qwen2 pattern: identical to cl100k except `\p{N}` matches a single digit
    /// (not 1-3), so it reuses the cl100k scanner with a max-digit cap of 1.
    Qwen2,
    /// deepseek_v3 pattern (digits, CJK, punct+letters, letters, punct runs).
    Deepseek,
    /// mistral_v3 (Tekken) pattern: o200k-style case splitting, but with no
    /// contraction rule, single-digit `\p{N}`, and a `[\r\n/]*` punctuation tail.
    Tekken,
    /// mistral_v15 (Tekken v15) pattern: Tekken's rules, except that digit runs
    /// split into right-aligned groups of three (see `digits.rs`).
    Tekken15,
    /// minimax_m2 pattern: o200k's letter/digit rules (contractions included)
    /// with Tekken's `[\r\n/]*` punctuation tail.
    MiniMax,
    /// kimi_k2 / kimi_k3 pattern: o200k's ASCII behaviour but with the plain
    /// `[\r\n]*` punctuation tail — Moonshot's pat_str does not admit `/`
    /// there. (Its `[\p{Han}]+` branch and Han-excluded letter classes only
    /// matter for non-ASCII input, where this scanner defers to the regex.)
    Kimi,
}

/// Which whitespace rules a pattern uses, deciding whether the `\s+(?!\S)`
/// lookahead emulation may trim a match. Like [`FastPath`], this is chosen by
/// the caller in `encoding.rs`, which owns the pattern definitions.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhitespaceRules {
    /// The pattern's only whitespace handling is the generic `\s+(?!\S)|\s+`
    /// (p50k_base / r50k_base). Every all-whitespace match carries the
    /// lookahead, so all of them are subject to the trim.
    Generic,
    /// The pattern has a dedicated `\s*[\r\n]+` branch ordered *before* the
    /// generic `\s+(?!\S)|\s+` (cl100k, o200k, qwen2, deepseek_v3).
    ///
    /// Under leftmost-first alternation an all-whitespace run that contains a
    /// newline is always claimed by that branch, which has no lookahead — so a
    /// match ending in `\r`/`\n` must never be trimmed. Trimming it would split
    /// canonical multi-newline tokens (`"\n\n"`, `"\r\n"`) into single ones.
    NewlineFirst,
    /// deepseek_v3: as [`Self::NewlineFirst`], but the pattern is the last stage
    /// of a sequential HuggingFace `Split` pipeline whose earlier stages isolate
    /// `\p{N}{1,3}` runs and CJK/kana runs. This crate folds those stages into
    /// one alternation, so the lookahead — which upstream only ever sees a
    /// single stage-boundary-delimited slice — would otherwise peek past a
    /// boundary. A digit or CJK/kana char after a whitespace run starts a new
    /// upstream slice and so acts as end-of-input: no trim.
    NewlineFirstSplitOnNumCjk,
}

/// Regex-based pre-tokenizer wrapping the existing regex + whitespace adjustment logic.
pub struct RegexPreTokenizer {
    regex: Regex,
    /// Which ASCII fast-path scanner (if any) to try before the regex.
    fast: FastPath,
    /// Which whitespace rules the pattern uses (gates the lookahead trim).
    ws: WhitespaceRules,
}

impl RegexPreTokenizer {
    pub(crate) fn new(pattern: &str, fast: FastPath, ws: WhitespaceRules) -> Self {
        Self {
            regex: Regex::new(pattern).expect("invalid regex pattern"),
            fast,
            ws,
        }
    }
}

impl PreTokenizer for RegexPreTokenizer {
    #[inline]
    fn next_match(&self, text: &str, pos: usize) -> Option<(usize, usize)> {
        let bytes = text.as_bytes();
        let fast = match self.fast {
            FastPath::Cl100k => cl100k_ascii_next::<3>(bytes, pos),
            FastPath::Qwen2 => cl100k_ascii_next::<1>(bytes, pos),
            FastPath::O200k => o200k_like_ascii_next::<true, 3, true, false>(bytes, pos),
            FastPath::Tekken => o200k_like_ascii_next::<false, 1, true, false>(bytes, pos),
            FastPath::Tekken15 => return self.tekken15_next(text, pos),
            FastPath::MiniMax => o200k_like_ascii_next::<true, 3, true, false>(bytes, pos),
            FastPath::Kimi => o200k_like_ascii_next::<true, 3, false, true>(bytes, pos),
            FastPath::Deepseek => deepseek_ascii_next(bytes, pos),
            FastPath::None => None,
        };
        if let Some(r) = fast {
            return Some(r);
        }
        self.regex_next(text, pos)
    }
}

impl RegexPreTokenizer {
    #[inline]
    fn regex_next(&self, text: &str, pos: usize) -> Option<(usize, usize)> {
        let mat = self.regex.find_at(text, pos)?;
        let start = mat.start();
        let end = adjust_whitespace_end(text.as_bytes(), start, mat.end(), self.ws);
        Some((start, end))
    }

    /// Kept out of line: inlining the digit rule into `next_match` grew it
    /// enough to cost the other encodings' `encode` loop ~20% on ASCII text.
    #[inline(never)]
    fn tekken15_next(&self, text: &str, pos: usize) -> Option<(usize, usize)> {
        if let Some(end) = tekken15_digit_group_end(text, pos) {
            return Some((pos, end));
        }
        o200k_like_ascii_next::<false, 1, true, false>(text.as_bytes(), pos)
            .or_else(|| self.regex_next(text, pos))
    }
}

/// Character-class certainty for the CJK extension of the fast-path scanners.
///
/// Each variant is a *certainty claim* about how the patterns' Unicode classes
/// treat the char; `Other` means "this table cannot be certain" and always
/// defers the piece to the regex, so an omission here costs speed, never
/// correctness. The claims are pinned char-by-char against the regex crate's
/// own Unicode tables by `cjk_class_matches_regex_tables`, so the table cannot
/// drift from the engine that defines correctness.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CjkClass {
    /// `\p{Han}` (and therefore `\p{L}`/`Lo`): CJK Unified Ideographs + Ext A.
    Han,
    /// `\p{L}` of category `Lo`/`Lm`, not Han: kana, Hangul syllables,
    /// halfwidth katakana. Caseless, so o200k's case-split classes contain it
    /// on both sides.
    Caseless,
    /// `\p{Lu}`: fullwidth Ａ-Ｚ.
    Upper,
    /// `\p{Ll}`: fullwidth ａ-ｚ.
    Lower,
    /// Matches `[^\s\p{L}\p{N}]`: CJK and fullwidth punctuation/symbols.
    Punct,
    /// `\p{N}`: fullwidth digits, ideographic numerals.
    Num,
    /// Whitespace (ideographic space).
    Ws,
    /// Unknown to this table — defer to the regex.
    Other,
}

#[inline]
fn cjk_class(c: u32) -> CjkClass {
    use CjkClass::*;
    match c {
        0x4E00..=0x9FFF | 0x3400..=0x4DBF => Han,
        0x3041..=0x3096 | 0x309D..=0x309F => Caseless, // hiragana + iteration marks
        0x30A1..=0x30FA | 0x30FC..=0x30FF => Caseless, // katakana + ー ヽ ヾ ヿ
        0xAC00..=0xD7A3 => Caseless,                   // hangul syllables
        0xFF66..=0xFF9F => Caseless,                   // halfwidth katakana (incl. ｰ ﾞ ﾟ, all Lo/Lm)
        0xFF21..=0xFF3A => Upper,                      // fullwidth Ａ-Ｚ
        0xFF41..=0xFF5A => Lower,                      // fullwidth ａ-ｚ
        // CJK punctuation. 3005 々 / 3006 〆 (letters), 3007 〇 (number) and the
        // mark/numeral stretches of the block are deliberately absent.
        0x3001..=0x3004 | 0x3008..=0x3020 | 0x3030 | 0x3036 | 0x303D => Punct,
        0x30A0 | 0x30FB => Punct, // ゠ ・ (the two non-letters inside katakana)
        0xFF01..=0xFF0F | 0xFF1A..=0xFF20 | 0xFF3B..=0xFF40 | 0xFF5B..=0xFF65 => Punct,
        0x2014 | 0x2018..=0x201D | 0x2025..=0x2026 => Punct, // — quotes ‥ …
        0xFF10..=0xFF19 | 0x3007 | 0x3021..=0x3029 | 0x3038..=0x303A => Num,
        0x3000 => Ws,
        _ => Other,
    }
}

/// Decode one char at byte offset `i`. Input comes from `&str`, so the UTF-8
/// is valid by construction and no error path exists.
#[inline]
fn decode_char(b: &[u8], i: usize) -> (u32, usize) {
    let c0 = b[i];
    if c0 < 0x80 {
        (c0 as u32, 1)
    } else if c0 < 0xE0 {
        ((((c0 & 0x1F) as u32) << 6) | (b[i + 1] & 0x3F) as u32, 2)
    } else if c0 < 0xF0 {
        (
            (((c0 & 0x0F) as u32) << 12)
                | (((b[i + 1] & 0x3F) as u32) << 6)
                | (b[i + 2] & 0x3F) as u32,
            3,
        )
    } else {
        (
            (((c0 & 0x07) as u32) << 18)
                | (((b[i + 1] & 0x3F) as u32) << 12)
                | (((b[i + 2] & 0x3F) as u32) << 6)
                | (b[i + 3] & 0x3F) as u32,
            4,
        )
    }
}

/// Scan a `\p{L}+` run (cl100k-family letter rule) from `j`, ASCII and CJK
/// alike. `Some(end)` when the run ends at a char that is *certainly* not a
/// letter; `None` (defer) on the first char the table cannot place.
///
/// Cold: the inlined scanners handle pure-ASCII runs with their own open-coded
/// loops and only branch here when a non-ASCII byte actually appears, so CJK
/// support costs the ASCII hot path nothing but a taken-once branch.
#[cold]
#[inline(never)]
fn scan_letter_run_mixed(b: &[u8], mut j: usize) -> Option<usize> {
    let n = b.len();
    while j < n {
        let c = b[j];
        if c < 0x80 {
            if c.is_ascii_alphabetic() {
                j += 1;
                continue;
            }
            return Some(j);
        }
        let (ch, len) = decode_char(b, j);
        match cjk_class(ch) {
            CjkClass::Han | CjkClass::Caseless | CjkClass::Upper | CjkClass::Lower => j += len,
            CjkClass::Punct | CjkClass::Num | CjkClass::Ws => return Some(j),
            CjkClass::Other => return None,
        }
    }
    Some(j)
}

/// Scan a `[^\s\p{L}\p{N}]+` run from `j`, ASCII and CJK alike. Same
/// certainty contract and cold placement as [`scan_letter_run_mixed`].
#[cold]
#[inline(never)]
fn scan_punct_run_mixed(b: &[u8], mut j: usize) -> Option<usize> {
    let n = b.len();
    while j < n {
        let c = b[j];
        if c < 0x80 {
            if !is_ascii_ws(c) && !c.is_ascii_alphanumeric() {
                j += 1;
                continue;
            }
            return Some(j);
        }
        let (ch, len) = decode_char(b, j);
        match cjk_class(ch) {
            CjkClass::Punct => j += len,
            CjkClass::Han
            | CjkClass::Caseless
            | CjkClass::Upper
            | CjkClass::Lower
            | CjkClass::Num
            | CjkClass::Ws => return Some(j),
            CjkClass::Other => return None,
        }
    }
    Some(j)
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
/// Returns `Some((i, end))` on a match, or `None` to defer to the regex (the
/// start is whitespace, or a non-ASCII byte could extend the run under Unicode
/// semantics). Caller guarantees `i < n` and `b[i] < 0x80`.
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
        // \p{N} (superscripts, other-number) the regex would fold in — defer.
        // At the cap the regex stops regardless, so it's safe to return.
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
            // space + CJK punctuation starts the run just as well (a CJK
            // letter would have been claimed by the word rules in the caller)
            Some(&c1) if c1 >= 0x80 => {
                return space_cjk_punct::<SLASH_TAIL>(b, i);
            }
            // space not followed by punct → whitespace rules → defer
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
            let k = scan_punct_run_mixed(b, k)?;
            return Some((i, take_line_tail::<SLASH_TAIL>(b, k)));
        }
        k = take_line_tail::<SLASH_TAIL>(b, k);
        return Some((i, k));
    }

    // whitespace run (or other) → defer to regex
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

    // Single source of truth: the real production patterns. Importing them here
    // (rather than copying) guarantees the fast-path equivalence proptests below
    // validate against exactly the patterns used in production.
    use crate::encoding::{
        CL100K_PATTERN, DEEPSEEK_V3_PATTERN, KIMI_PATTERN, MISTRAL_V3_PATTERN, O200K_PATTERN,
        P50K_PATTERN, QWEN2_PATTERN,
    };

    /// A production pattern bundled with the [`FastPath`] and
    /// [`WhitespaceRules`] `encoding.rs` pairs it with. Keeping the three
    /// together stops tests from drifting to a combination that never ships.
    #[derive(Clone, Copy)]
    struct Spec {
        pattern: &'static str,
        fast: FastPath,
        ws: WhitespaceRules,
    }

    const CL100K: Spec = Spec {
        pattern: CL100K_PATTERN,
        fast: FastPath::Cl100k,
        ws: WhitespaceRules::NewlineFirst,
    };
    const O200K: Spec = Spec {
        pattern: O200K_PATTERN,
        fast: FastPath::O200k,
        ws: WhitespaceRules::NewlineFirst,
    };
    const QWEN2: Spec = Spec {
        pattern: QWEN2_PATTERN,
        fast: FastPath::Qwen2,
        ws: WhitespaceRules::NewlineFirst,
    };
    const DEEPSEEK: Spec = Spec {
        pattern: DEEPSEEK_V3_PATTERN,
        fast: FastPath::Deepseek,
        ws: WhitespaceRules::NewlineFirst,
    };
    const MISTRAL: Spec = Spec {
        pattern: MISTRAL_V3_PATTERN,
        fast: FastPath::Tekken,
        ws: WhitespaceRules::NewlineFirst,
    };
    const KIMI: Spec = Spec {
        pattern: KIMI_PATTERN,
        fast: FastPath::Kimi,
        ws: WhitespaceRules::NewlineFirst,
    };
    const P50K: Spec = Spec {
        pattern: P50K_PATTERN,
        fast: FastPath::None,
        ws: WhitespaceRules::Generic,
    };

    impl Spec {
        fn tokenizer(self) -> RegexPreTokenizer {
            RegexPreTokenizer::new(self.pattern, self.fast, self.ws)
        }
    }

    // Reference implementation: pure regex + whitespace adjustment, with no
    // ASCII fast path. The fast paths must be byte-for-byte equivalent to it.
    fn reference_matches(spec: Spec, text: &str) -> Vec<(usize, usize)> {
        let regex = Regex::new(spec.pattern).unwrap();
        let bytes = text.as_bytes();
        let mut result = vec![];
        let mut pos = 0;
        while pos < text.len() {
            let mat = match regex.find_at(text, pos) {
                Some(m) => m,
                None => break,
            };
            let start = mat.start();
            let end = adjust_whitespace_end(bytes, start, mat.end(), spec.ws);
            result.push((start, end));
            pos = end;
        }
        result
    }

    fn assert_fast_matches_reference(spec: Spec, text: &str) {
        let pt = spec.tokenizer();
        assert_eq!(
            reference_matches(spec, text),
            collect_matches(&pt, text),
            "fast/regex mismatch for {text:?}"
        );
    }

    #[test]
    fn test_cl100k_english() {
        assert_fast_matches_reference(CL100K, "Hello, world!");
    }

    #[test]
    fn test_cl100k_cjk() {
        assert_fast_matches_reference(CL100K, "你好世界");
    }

    #[test]
    fn test_cl100k_contractions() {
        assert_fast_matches_reference(CL100K, "I'm don't they're we've she'll it'd");
    }

    #[test]
    fn test_o200k_english() {
        assert_fast_matches_reference(O200K, "Hello, world! CamelCase mixedScript123");
    }

    #[test]
    fn test_p50k_english() {
        assert_fast_matches_reference(P50K, "Hello world, I'm testing!");
    }

    #[test]
    fn test_empty_input() {
        let pt = CL100K.tokenizer();
        assert_eq!(collect_matches(&pt, ""), vec![]);
    }

    #[test]
    fn test_only_whitespace() {
        assert_fast_matches_reference(CL100K, "   \n  \t  ");
    }

    #[test]
    fn test_emoji() {
        assert_fast_matches_reference(CL100K, "🎉🚀💡");
    }

    #[test]
    fn test_mixed_script() {
        assert_fast_matches_reference(CL100K, "Hello 你好 World 🌍");
    }

    // whitespace adjustment tests (migrated from v2 bpe.rs)

    use WhitespaceRules::{Generic, NewlineFirst};

    #[test]
    fn test_adjust_whitespace_single_byte() {
        assert_eq!(adjust_whitespace_end(b"a b", 0, 1, Generic), 1);
    }

    #[test]
    fn test_adjust_whitespace_at_end_of_input() {
        assert_eq!(adjust_whitespace_end(b"  ", 0, 2, Generic), 2);
    }

    #[test]
    fn test_adjust_whitespace_non_ws_piece() {
        assert_eq!(adjust_whitespace_end(b"hello world", 0, 5, Generic), 5);
    }

    #[test]
    fn test_adjust_whitespace_trim_before_nonws() {
        let bytes = b"  x";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, Generic), 1);
    }

    #[test]
    fn test_adjust_whitespace_no_trim_before_ws() {
        let bytes = b"   ";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, Generic), 2);
    }

    #[test]
    fn test_adjust_whitespace_unicode_slow_path() {
        let input = "\u{3000}\u{3000}x";
        let bytes = input.as_bytes();
        assert_eq!(adjust_whitespace_end(bytes, 0, 6, Generic), 3);
    }

    #[test]
    fn test_adjust_whitespace_unicode_followed_by_unicode_ws() {
        let input = "\u{3000}\u{3000}\u{3000}";
        let bytes = input.as_bytes();
        assert_eq!(adjust_whitespace_end(bytes, 0, 6, Generic), 6);
    }

    #[test]
    fn test_adjust_whitespace_single_multibyte_ws_before_nonws() {
        // U+3000 (ideographic space, 3 bytes) followed by 'x'
        // trimming the last char would make the piece empty, so it should NOT trim
        let input = "\u{3000}x";
        let bytes = input.as_bytes();
        // piece is bytes[0..3] (the ideographic space), next char is 'x' (non-ws)
        // without the protection, this would trim to bytes[0..0] which is empty
        assert_eq!(adjust_whitespace_end(bytes, 0, 3, Generic), 3);
    }

    // Newline-branch gating (issue #5): a match ending in \r/\n comes from
    // `\s*[\r\n]+`, which carries no lookahead, so it must keep its full extent.

    #[test]
    fn test_adjust_whitespace_newline_branch_keeps_double_newline() {
        let bytes = b"\n\nx";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, NewlineFirst), 2);
        // the generic-only patterns (p50k/r50k) still trim — canonical behavior
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, Generic), 1);
    }

    #[test]
    fn test_adjust_whitespace_newline_branch_keeps_crlf() {
        let bytes = b"\r\n@";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, NewlineFirst), 2);
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, Generic), 1);
    }

    #[test]
    fn test_adjust_whitespace_newline_branch_still_trims_spaces() {
        // no newline at the end → generic `\s+` branch → lookahead applies
        let bytes = b"  x";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2, NewlineFirst), 1);
    }

    #[test]
    fn test_adjust_whitespace_newline_branch_trims_trailing_spaces_after_newline() {
        // "\n  " + "x": the `\s*[\r\n]+` branch stops after "\n", so the piece
        // under adjustment here is the following "  " run, which does trim.
        let bytes = b"\n  x";
        assert_eq!(adjust_whitespace_end(bytes, 1, 3, NewlineFirst), 2);
    }

    // comprehensive comparison: fast path vs pure-regex reference, all patterns
    #[test]
    fn test_all_patterns_match_reference() {
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
            "word\n\nnext",
            "\r\n@rem",
            "a\n\n\nb",
            "a \n\n b",
        ];

        for spec in [CL100K, O200K, QWEN2, DEEPSEEK, MISTRAL, KIMI, P50K] {
            for text in &texts {
                assert_fast_matches_reference(spec, text);
            }
        }
    }

    /// The CJK table's claims, pinned char-by-char against the regex crate's
    /// own Unicode tables over the entire codepoint space. `Other` claims
    /// nothing and needs no check; every other variant is a certainty claim
    /// the fast paths rely on for both run membership *and* run termination.
    #[test]
    fn cjk_class_matches_regex_tables() {
        let letter = Regex::new(r"^\p{L}$").unwrap();
        let han = Regex::new(r"^\p{Han}$").unwrap();
        let caseless = Regex::new(r"^[\p{Lo}\p{Lm}]$").unwrap();
        let upper = Regex::new(r"^\p{Lu}$").unwrap();
        let lower = Regex::new(r"^\p{Ll}$").unwrap();
        let num = Regex::new(r"^\p{N}$").unwrap();
        let ws = Regex::new(r"^\s$").unwrap();
        let punct = Regex::new(r"^[^\s\p{L}\p{N}]$").unwrap();
        let mut buf = [0u8; 4];
        for cp in 0x80..=0x10FFFF_u32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let s: &str = c.encode_utf8(&mut buf);
            match cjk_class(cp) {
                CjkClass::Han => {
                    assert!(
                        han.is_match(s) && caseless.is_match(s),
                        "U+{cp:04X} claimed Han"
                    );
                }
                CjkClass::Caseless => {
                    assert!(
                        caseless.is_match(s) && !han.is_match(s),
                        "U+{cp:04X} claimed caseless letter"
                    );
                }
                CjkClass::Upper => assert!(upper.is_match(s), "U+{cp:04X} claimed Lu"),
                CjkClass::Lower => assert!(lower.is_match(s), "U+{cp:04X} claimed Ll"),
                CjkClass::Num => assert!(num.is_match(s), "U+{cp:04X} claimed N"),
                CjkClass::Ws => assert!(ws.is_match(s), "U+{cp:04X} claimed whitespace"),
                CjkClass::Punct => {
                    assert!(
                        punct.is_match(s),
                        "U+{cp:04X} claimed [^\\s\\p{{L}}\\p{{N}}]"
                    );
                }
                CjkClass::Other => {}
            }
            // the deepseek ranges are their own rule; letter-hood is irrelevant,
            // but they must at least stay valid chars (surrogates are skipped)
            let _ = is_deepseek_cjk(cp);
            // every claimed letter must also be \p{L}
            if matches!(
                cjk_class(cp),
                CjkClass::Han | CjkClass::Caseless | CjkClass::Upper | CjkClass::Lower
            ) {
                assert!(
                    letter.is_match(s),
                    "U+{cp:04X} claimed letter but is not \\p{{L}}"
                );
            }
        }
    }

    // Hand-picked CJK shapes, including the ones where o200k's case classes
    // interact with caseless letters (the `[U]*[L]+` backtracking cases) and
    // the chars sitting right outside every table range.
    #[test]
    fn test_cjk_pieces_match_reference() {
        let texts = [
            "世界",
            "你好，世界！",
            "、你好",
            "　世界",
            "ハロー・ワールド",
            "世A",
            "世AB",
            "A世",
            "ＡB世A",
            "abc世界",
            "世界abc",
            "カタカナー",
            "パーティー",
            "が",         // precomposed
            "か\u{3099}", // combining dakuten → Other → defer
            "안녕하세요 세계",
            "ｱｲｳｴｵﾞ",
            "ＡＢＣａｂｃ",
            "世's",
            "世界。。。",
            "……你好……",
            "「引用」",
            "（括号）",
            "第１２３号",
            "３．１４",
            "一二三四五六七八九十",
            "〇一二",      // 〇 is \p{N}
            "々仕事",      // 々 is a letter but outside the table → defer
            "\u{20000}好", // Ext-B Han → defer
            "深圳市－广州市",
            "ＦＵＬＬｗｉｄｔｈ",
            "ｶﾞｷﾞｸﾞ",
            "日本語テスト123テスト",
            "。\n、",
            "税込１，０００円",
            "「こんにちは」と言った",
        ];
        for spec in [CL100K, O200K, QWEN2, DEEPSEEK, MISTRAL, KIMI, P50K] {
            for text in &texts {
                assert_fast_matches_reference(spec, text);
            }
        }
    }

    // ASCII fast-path equivalence: the cl100k fast path (now built into
    // RegexPreTokenizer) must produce byte-for-byte identical pieces to the
    // pure-regex reference for ANY input.
    /// How many random cases each property sweep generates.
    ///
    /// The committed `proptest-regressions/` seeds replay before any random
    /// case, so every input that ever failed is still checked at any setting —
    /// what this scales is how much *new* ground a run covers. The default is a
    /// smoke sample so a push is answered in minutes; the full sweep is a
    /// release gate, run with `PROPTEST_CASES=20000` (see the Release section
    /// of the top-level README).
    fn proptest_cases() -> u32 {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(proptest_cases()))]

        #[test]
        fn prop_cl100k_fast_matches_regex(text in ".*") {
            let pt = CL100K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(CL100K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // ASCII-heavy generator to stress the fast path specifically.
        #[test]
        fn prop_cl100k_fast_matches_regex_ascii(text in "[ -~ \t\r\n]*") {
            let pt = CL100K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(CL100K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // Newline-dense generator: the alphabet the issue #5 regression lives in.
        #[test]
        fn prop_cl100k_fast_matches_regex_newlines(text in "[\r\n \tabc.!]*") {
            let pt = CL100K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(CL100K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_o200k_fast_matches_regex(text in ".*") {
            let pt = O200K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(O200K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_o200k_fast_matches_regex_ascii(text in "[ -~ \t\r\n]*") {
            let pt = O200K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(O200K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_o200k_fast_matches_regex_newlines(text in "[\r\n \tabc.!]*") {
            let pt = O200K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(O200K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_qwen2_fast_matches_regex(text in ".*") {
            let pt = QWEN2.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(QWEN2, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_qwen2_fast_matches_regex_ascii(text in "[ -~ \t\r\n]*") {
            let pt = QWEN2.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(QWEN2, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_deepseek_fast_matches_regex(text in ".*") {
            let pt = DEEPSEEK.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(DEEPSEEK, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_deepseek_fast_matches_regex_ascii(text in "[ -~ \t\r\n]*") {
            let pt = DEEPSEEK.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(DEEPSEEK, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_mistral_fast_matches_regex(text in ".*") {
            let pt = MISTRAL.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(MISTRAL, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_mistral_fast_matches_regex_ascii(text in "[ -~ \t\r\n]*") {
            let pt = MISTRAL.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(MISTRAL, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // Slash-dense generator: o200k's `[\r\n/]*` punctuation tail — the shape
        // whose absence from the fixture corpus let a missing `/` survive.
        #[test]
        fn prop_o200k_fast_matches_regex_slashes(text in "[/\r\n .!abcAB0]*") {
            let pt = O200K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(O200K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // Slash-dense generator: the `[\r\n/]*` punctuation tail shared with o200k.
        #[test]
        fn prop_mistral_fast_matches_regex_slashes(text in "[/\r\n .!abcAB0]*") {
            let pt = MISTRAL.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(MISTRAL, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_kimi_fast_matches_regex(text in ".*") {
            let pt = KIMI.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(KIMI, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // Kimi keeps o200k's ASCII rules but NOT its `[\r\n/]*` tail — this
        // generator is what separates the two scanners.
        #[test]
        fn prop_kimi_fast_matches_regex_slashes(text in "[/\r\n .!abcAB0]*") {
            let pt = KIMI.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(KIMI, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_p50k_fast_matches_regex(text in "[ -~ \t\r\n]*") {
            let pt = P50K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(P50K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // CJK-dense generators. The alphabet deliberately mixes run members
        // (Han, kana, hangul, fullwidth letters), certain terminators (CJK
        // punctuation, fullwidth digits, ideographic space), chars right
        // outside the table (々 〇 combining marks, Ext-B Han) that must
        // defer, and ASCII to hit every boundary between the two worlds.
        #[test]
        fn prop_cl100k_fast_matches_regex_cjk(
            text in "[世界你好日本語謎アイウエオぁあんーゟＡＢａｂ한글、。！？（）「」・…　 a-cA-C0-9'\r\n々〇\u{3099}\u{20000}é]*"
        ) {
            let pt = CL100K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(CL100K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_o200k_fast_matches_regex_cjk(
            text in "[世界你好日本語謎アイウエオぁあんーゟＡＢａｂ한글、。！？（）「」・…　 a-cA-C0-9'\r\n々〇\u{3099}\u{20000}é]*"
        ) {
            let pt = O200K.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(O200K, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_qwen2_fast_matches_regex_cjk(
            text in "[世界你好ぁーア１２、。！　 a-cA-C0-9'\r\n々〇é]*"
        ) {
            let pt = QWEN2.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(QWEN2, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_deepseek_fast_matches_regex_cjk(
            text in "[世界你好龥龦ぁゟ゠アヿー、。１ a-c0-9\r\n々é]*"
        ) {
            let pt = DEEPSEEK.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(DEEPSEEK, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        #[test]
        fn prop_mistral_fast_matches_regex_cjk(
            text in "[世界アＡａ、。！　 a-cA-C0-9/\r\né]*"
        ) {
            let pt = MISTRAL.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(MISTRAL, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }

        // Kimi: Han runs are their own branch and Han must never join a word.
        #[test]
        fn prop_kimi_fast_matches_regex_cjk(
            text in "[世界你好龥アイぁーＡａ한、。！　 a-cA-C0-9'\r\n々\u{20000}é]*"
        ) {
            let pt = KIMI.tokenizer();
            let fast = collect_matches(&pt, &text);
            let reference = reference_matches(KIMI, &text);
            proptest::prop_assert_eq!(fast, reference, "fast/regex mismatch for {:?}", text);
        }
    }
}
