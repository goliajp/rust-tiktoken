//! ASCII-first scanners for the o200k-shaped patterns: case-splitting letter
//! rules, used by o200k, Tekken, MiniMax / Llama 4 and Kimi.

use super::*;

/// Membership of one char in the o200k-family case-split letter classes.
///
/// The "upper" class is `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`, the "lower" class
/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]` — `Lo`/`Lm` (Han, kana, hangul) sit in *both*,
/// which is what `Both` encodes. With `HAN_APART` (kimi), Han is excluded from
/// both classes (`&&[^\p{Han}]`) and certainly ends a word run instead.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LetterKind {
    Upper,
    Lower,
    Both,
    End,
    Defer,
}

#[inline(always)]
pub(super) fn o200k_letter_kind<const HAN_APART: bool>(b: &[u8], j: usize) -> (LetterKind, usize) {
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
pub(super) fn o200k_letter_kind_cjk<const HAN_APART: bool>(
    b: &[u8],
    j: usize,
) -> (LetterKind, usize) {
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
pub(super) enum CjkStart {
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
pub(super) fn o200k_cjk_start<const HAN_APART: bool>(b: &[u8], i: usize) -> CjkStart {
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
pub(super) fn o200k_like_ascii_next<
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
pub(super) fn o200k_cjk_next<
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
pub(super) fn o200k_cjk_after_lead<
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
pub(super) fn o200k_word_mixed<const CONTRACTIONS: bool, const HAN_APART: bool>(
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
pub(super) fn scan_lower_run<const HAN_APART: bool>(b: &[u8], mut r: usize) -> Option<usize> {
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
