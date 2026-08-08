//! Pre-tokenization: split text into pieces before BPE encoding.
//!
//! The [`PreTokenizer`] trait defines a regex-based splitter that partitions
//! input text into byte-range pieces. [`RegexPreTokenizer`] implements this
//! using the `regex` crate (DFA-based), with a custom whitespace lookahead
//! emulation (`adjust_whitespace_end`) that avoids `fancy-regex` entirely.

use regex::Regex;

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

/// Rule 2 of `DEEPSEEK_V3_PATTERN`, by codepoint: `[一-龥\x{3040}-\x{309F}\x{30A0}-\x{30FF}]`.
#[inline]
fn is_deepseek_cjk(c: u32) -> bool {
    matches!(c, 0x4E00..=0x9FA5 | 0x3040..=0x30FF)
}

/// Whether `c` would have been isolated by an earlier stage of the deepseek_v3
/// split pipeline: `\p{N}{1,3}` (stage 1) or `[一-龥\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+`
/// (stage 2). Kept in sync with `DEEPSEEK_V3_PATTERN`'s first two alternatives.
#[inline]
fn is_deepseek_split_boundary(c: char) -> bool {
    c.is_numeric() || matches!(c, '一'..='龥' | '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}')
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
            FastPath::MiniMax => o200k_like_ascii_next::<true, 3, true, false>(bytes, pos),
            FastPath::Kimi => o200k_like_ascii_next::<true, 3, false, true>(bytes, pos),
            FastPath::Deepseek => deepseek_ascii_next(bytes, pos),
            FastPath::None => None,
        };
        if let Some(r) = fast {
            return Some(r);
        }
        let mat = self.regex.find_at(text, pos)?;
        let start = mat.start();
        let end = adjust_whitespace_end(bytes, start, mat.end(), self.ws);
        Some((start, end))
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

/// A space at `i` followed by a non-ASCII char: the punct rule applies only
/// if that char is certainly punctuation. Out of line with the other CJK arms.
#[cold]
#[inline(never)]
fn space_cjk_punct<const SLASH_TAIL: bool>(b: &[u8], i: usize) -> Option<(usize, usize)> {
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
fn cl100k_cjk_next(b: &[u8], i: usize) -> Option<(usize, usize)> {
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
fn cl100k_cjk_after_lead<const MAX_DIGITS: usize>(b: &[u8], i: usize) -> Option<(usize, usize)> {
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
fn cl100k_ascii_next<const MAX_DIGITS: usize>(b: &[u8], i: usize) -> Option<(usize, usize)> {
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

/// Membership of one char in the o200k-family case-split letter classes.
///
/// The "upper" class is `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`, the "lower" class
/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]` — `Lo`/`Lm` (Han, kana, hangul) sit in *both*,
/// which is what `Both` encodes. With `HAN_APART` (kimi), Han is excluded from
/// both classes (`&&[^\p{Han}]`) and certainly ends a word run instead.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LetterKind {
    Upper,
    Lower,
    Both,
    End,
    Defer,
}

#[inline(always)]
fn o200k_letter_kind<const HAN_APART: bool>(b: &[u8], j: usize) -> (LetterKind, usize) {
    let c = b[j];
    if c < 0x80 {
        if c.is_ascii_uppercase() {
            return (LetterKind::Upper, 1);
        }
        if c.is_ascii_lowercase() {
            return (LetterKind::Lower, 1);
        }
        return (LetterKind::End, 1);
    }
    o200k_letter_kind_cjk::<HAN_APART>(b, j)
}

/// [`o200k_letter_kind`] for the non-ASCII case, out of line.
#[cold]
#[inline(never)]
fn o200k_letter_kind_cjk<const HAN_APART: bool>(b: &[u8], j: usize) -> (LetterKind, usize) {
    let (ch, len) = decode_char(b, j);
    let kind = match cjk_class(ch) {
        CjkClass::Han => {
            if HAN_APART {
                LetterKind::End
            } else {
                LetterKind::Both
            }
        }
        CjkClass::Caseless => LetterKind::Both,
        CjkClass::Upper => LetterKind::Upper,
        CjkClass::Lower => LetterKind::Lower,
        CjkClass::Punct | CjkClass::Num | CjkClass::Ws => LetterKind::End,
        CjkClass::Other => LetterKind::Defer,
    };
    (kind, len)
}

/// How an o200k-family piece that starts on a non-ASCII char proceeds.
enum CjkStart {
    /// A complete piece (kimi's `[\p{Han}]+` branch).
    Piece(usize),
    /// Defer to the regex.
    Defer,
    /// Not a word start — try the digit/punct rules.
    Punct,
    /// The word rules apply; letters begin at this byte offset.
    Letters(usize),
}

/// Classify a non-ASCII piece start for [`o200k_like_ascii_next`], out of
/// line so CJK support does not bloat the inlined ASCII scanner.
#[cold]
#[inline(never)]
fn o200k_cjk_start<const HAN_APART: bool>(b: &[u8], i: usize) -> CjkStart {
    let n = b.len();
    let (ch, len) = decode_char(b, i);
    let cls = cjk_class(ch);
    // Kimi's `[\p{Han}]+` branch is ordered before the word rules: a Han run
    // is a piece of its own, with no leading char and no suffix.
    if HAN_APART && cls == CjkClass::Han {
        let mut j = i + len;
        while j < n {
            if b[j] < 0x80 {
                break;
            }
            let (c2, l2) = decode_char(b, j);
            match cjk_class(c2) {
                CjkClass::Han => j += l2,
                // Ext-B and friends are also \p{Han} but unknown to the
                // table — the run might continue, so defer.
                CjkClass::Other => return CjkStart::Defer,
                _ => break,
            }
        }
        return CjkStart::Piece(j);
    }
    match cls {
        // a letter with no leading char
        CjkClass::Han | CjkClass::Caseless | CjkClass::Upper | CjkClass::Lower => {
            CjkStart::Letters(i)
        }
        // 、 or 　 as the leading char `[^\r\n\p{L}\p{N}]` when a word
        // follows; otherwise 、 falls to the punct rule, 　 to whitespace.
        CjkClass::Punct | CjkClass::Ws => {
            let j = i + len;
            let follows_word = j < n && {
                let (k1, _) = o200k_letter_kind::<HAN_APART>(b, j);
                matches!(k1, LetterKind::Upper | LetterKind::Lower | LetterKind::Both)
            };
            if follows_word {
                CjkStart::Letters(j)
            } else if cls == CjkClass::Ws {
                CjkStart::Defer
            } else {
                CjkStart::Punct
            }
        }
        CjkClass::Num | CjkClass::Other => CjkStart::Defer,
    }
}

/// Fast-path pre-tokenizer for the case-splitting patterns: o200k, Mistral's
/// Tekken, MiniMax and Kimi.
///
/// The letter rules differ from cl100k: both split on case
/// (`[\p{Lu}…]*[\p{Ll}…]+` then `[\p{Lu}…]+[\p{Ll}…]*`, CamelCase-aware),
/// with the caseless `Lo`/`Lm` letters (Han, kana, hangul) a member of both
/// classes — see [`o200k_letter_kind`].
///
/// The variants differ in places passed in by the caller: o200k/MiniMax attach
/// an optional contraction suffix to the word and use `\p{N}{1,3}`; Tekken has
/// no contraction rule and uses `\p{N}`; Kimi (`HAN_APART`) tries a dedicated
/// `[\p{Han}]+` branch first and excludes Han from both letter classes.
#[inline(always)]
fn o200k_like_ascii_next<
    const CONTRACTIONS: bool,
    const MAX_DIGITS: usize,
    const SLASH_TAIL: bool,
    const HAN_APART: bool,
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
        return o200k_cjk_next::<CONTRACTIONS, MAX_DIGITS, SLASH_TAIL, HAN_APART>(b, i);
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
            Some(&c1) if c1 >= 0x80 => {
                return o200k_cjk_after_lead::<CONTRACTIONS, MAX_DIGITS, SLASH_TAIL, HAN_APART>(
                    b, i,
                );
            }
            _ => return ascii_num_punct::<MAX_DIGITS, SLASH_TAIL>(b, i),
        }
    } else {
        // digit, or \r\n
        return ascii_num_punct::<MAX_DIGITS, SLASH_TAIL>(b, i);
    };

    // Scan the uppercase run from `p`; a non-ASCII byte at any decision point
    // hands the whole word over to the CJK-aware cold scanner.
    let mut q = p;
    while q < n && b[q] < 0x80 && b[q].is_ascii_uppercase() {
        q += 1;
    }
    if q < n && b[q] >= 0x80 {
        return o200k_word_mixed::<CONTRACTIONS, HAN_APART>(b, i, p);
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
                return o200k_word_mixed::<CONTRACTIONS, HAN_APART>(b, i, p);
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
            return o200k_word_mixed::<CONTRACTIONS, HAN_APART>(b, i, p);
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

/// o200k-family piece starting on a non-ASCII char, out of line.
#[cold]
#[inline(never)]
fn o200k_cjk_next<
    const CONTRACTIONS: bool,
    const MAX_DIGITS: usize,
    const SLASH_TAIL: bool,
    const HAN_APART: bool,
>(
    b: &[u8],
    i: usize,
) -> Option<(usize, usize)> {
    match o200k_cjk_start::<HAN_APART>(b, i) {
        CjkStart::Piece(e) => Some((i, e)),
        CjkStart::Defer => None,
        CjkStart::Punct => ascii_num_punct::<MAX_DIGITS, SLASH_TAIL>(b, i),
        CjkStart::Letters(p) => o200k_word_mixed::<CONTRACTIONS, HAN_APART>(b, i, p),
    }
}

/// o200k-family: an eligible ASCII leading char at `i` with a non-ASCII char
/// after it. The word rules if that char is a word member; otherwise the
/// digit/punct rules via [`ascii_num_punct`].
#[cold]
#[inline(never)]
fn o200k_cjk_after_lead<
    const CONTRACTIONS: bool,
    const MAX_DIGITS: usize,
    const SLASH_TAIL: bool,
    const HAN_APART: bool,
>(
    b: &[u8],
    i: usize,
) -> Option<(usize, usize)> {
    if matches!(
        o200k_letter_kind::<HAN_APART>(b, i + 1).0,
        LetterKind::Upper | LetterKind::Lower | LetterKind::Both
    ) {
        return o200k_word_mixed::<CONTRACTIONS, HAN_APART>(b, i, i + 1);
    }
    ascii_num_punct::<MAX_DIGITS, SLASH_TAIL>(b, i)
}

/// The o200k-family word scan over mixed ASCII/CJK letters, out of line. The
/// piece starts at `i`, its letters at `p`.
///
/// The upper-ish run `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*` takes Upper and Both
/// members. `last_both_end` emulates rule A's backtracking: when the run is
/// followed by a non-letter, `[U]*[L]+` hands characters back until `[L]+` can
/// take one — and the only run members `[L]+` can take are Both-class chars,
/// so the match ends exactly after the run's last Both-class char (or the rule
/// fails and rule B keeps the whole run).
#[cold]
#[inline(never)]
fn o200k_word_mixed<const CONTRACTIONS: bool, const HAN_APART: bool>(
    b: &[u8],
    i: usize,
    p: usize,
) -> Option<(usize, usize)> {
    let n = b.len();
    let mut q = p;
    let mut last_both_end: Option<usize> = None;
    while q < n {
        let (kind, len) = o200k_letter_kind::<HAN_APART>(b, q);
        match kind {
            LetterKind::Upper => q += len,
            LetterKind::Both => {
                q += len;
                last_both_end = Some(q);
            }
            LetterKind::Lower | LetterKind::End => break,
            LetterKind::Defer => return None,
        }
    }

    let next_kind = if q < n {
        o200k_letter_kind::<HAN_APART>(b, q).0
    } else {
        LetterKind::End
    };
    let letters_end = if q > p {
        match next_kind {
            // Rule A: greedy upper part, then the lower-ish run
            LetterKind::Lower => scan_lower_run::<HAN_APART>(b, q)?,
            // Rule A via backtracking if the run holds a Both char, else rule B
            LetterKind::End => last_both_end.unwrap_or(q),
            _ => return None,
        }
    } else {
        // no upper part: p is lower-ish, rule A with `[U]*` empty
        scan_lower_run::<HAN_APART>(b, p)?
    };

    // Optional contraction suffix attached to the word: (?i:'s|'t|…)?
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

/// The lower-ish run `[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`: Lower and Both members.
/// Greedy with no backtracking (it is the branch's last letter element).
#[inline]
fn scan_lower_run<const HAN_APART: bool>(b: &[u8], mut r: usize) -> Option<usize> {
    let n = b.len();
    while r < n {
        let (kind, len) = o200k_letter_kind::<HAN_APART>(b, r);
        match kind {
            LetterKind::Lower | LetterKind::Both => r += len,
            LetterKind::Upper | LetterKind::End => return Some(r),
            LetterKind::Defer => return None,
        }
    }
    Some(r)
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
fn deepseek_ascii_next(b: &[u8], i: usize) -> Option<(usize, usize)> {
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
fn adjust_whitespace_end(bytes: &[u8], start: usize, end: usize, ws: WhitespaceRules) -> usize {
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
    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(20000))]

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
