//! Encoding a long text as independent chunks, and where to cut it.
//!
//! A cut at `b` is valid when the single-threaded scan of the whole text has
//! a piece starting at `b`, and scanning `text[..b]` and `text[b..]` on their
//! own gives the same pieces as the whole scan does on either side. Every
//! split pattern decides a piece from the text after its start, except two
//! lookarounds: `\s+(?!\S)` at a piece end and mistral_v15's digit rule,
//! which looks at the previous char. The three rules below hold for the
//! split pattern of every encoding:
//!
//! 1. `text[b-1]` is an ASCII letter and `text[b]` a space or `\n`. A piece
//!    holding a letter comes from a letter or contraction rule, and none of
//!    them takes whitespace, so the piece ends at `b`; whitespace at `b`
//!    answers `(?!\S)` the same way the end of the text does.
//! 2. `text[b-2]` is ASCII and not whitespace, `text[b-1]` is `\n` and
//!    `text[b]` an ASCII letter or digit. The newline ends a punctuation
//!    tail, forms `\s*[\r\n]+` or (p50k) a lone `\s+`; no letter prefix class
//!    admits `\r` or `\n`. Requiring a non-blank char before it keeps p50k's
//!    `\s+(?!\S)` from claiming a longer run when the chunk ends there.
//! 3. `text[b-3..b]` is a Han char in U+4E00..=U+9FA5 and `text[b]` one of
//!    `，。、！？；：`. Han only occurs in letter runs (deepseek's CJK run,
//!    kimi's `[\p{Han}]+`), and these marks are in no letter class.
//!
//! The char before `b` is a letter or a newline under all three, which the
//! digit rule treats like the start of the text.

use crate::encode_loop::Engine;
use crate::pool;

/// Nominal chunk length in bytes.
pub(crate) const CHUNK: usize = 8 * 1024;

/// Encode `text` as chunks on the pool, appending to `result`; falls back to
/// this thread when parallelism is 1, the text has no cut points, or the pool
/// is busy. At most `max_chunks` chunks, three per thread by default.
#[inline(never)]
pub(crate) fn encode(
    engine: &Engine<'_>,
    text: &str,
    chunk: usize,
    max_chunks: Option<usize>,
    result: &mut Vec<u32>,
) {
    if let Some((threads, cuts)) = plan(text, chunk, max_chunks) {
        let parts = pool::run(cuts.len() - 1, threads, |i| {
            let part = &text[cuts[i]..cuts[i + 1]];
            let mut out = Vec::with_capacity(part.len() / 3);
            engine.encode_into(part, &mut out);
            out
        });
        if let Some(parts) = parts {
            pool::concat_into(&parts, threads, result);
            return;
        }
    }
    result.reserve(text.len() / 3);
    engine.encode_into(text, result);
}

/// The counting twin of [`encode`].
#[inline(never)]
pub(crate) fn count(
    engine: &Engine<'_>,
    text: &str,
    chunk: usize,
    max_chunks: Option<usize>,
) -> usize {
    if let Some((threads, cuts)) = plan(text, chunk, max_chunks) {
        let parts = pool::run(cuts.len() - 1, threads, |i| {
            engine.count(&text[cuts[i]..cuts[i + 1]])
        });
        if let Some(parts) = parts {
            return parts.iter().sum();
        }
    }
    engine.count(text)
}

/// Thread count and chunk boundaries, or `None` to stay on this thread.
fn plan(text: &str, chunk: usize, max_chunks: Option<usize>) -> Option<(usize, Vec<usize>)> {
    let threads = pool::threads();
    if threads <= 1 {
        return None;
    }
    let cuts = cuts(text.as_bytes(), chunk, max_chunks.unwrap_or(threads * 3));
    (cuts.len() > 2).then_some((threads, cuts))
}

/// Chunk boundaries of `b`: `0`, the cuts in increasing order, `b.len()`.
/// Aims for `min(len / chunk, max_chunks)` chunks; a nominal cut moves
/// forward to the first valid position within half a chunk, and is dropped if
/// there is none.
pub(crate) fn cuts(b: &[u8], chunk: usize, max_chunks: usize) -> Vec<usize> {
    let n = b.len();
    let chunks = n.div_ceil(chunk.max(1)).min(max_chunks).max(1);
    let step = n / chunks;
    let mut out = Vec::with_capacity(chunks + 1);
    out.push(0);
    for k in 1..chunks {
        let from = (k * step).max(out[out.len() - 1] + 1);
        let to = (from + step / 2).min(n);
        if let Some(cut) = (from..to).find(|&i| is_cut(b, i)) {
            out.push(cut);
        }
    }
    out.push(n);
    out
}

/// Whether one of the three rules allows a cut before `b[i]`, `0 < i < len`.
#[inline]
pub(crate) fn is_cut(b: &[u8], i: usize) -> bool {
    let (prev, cur) = (b[i - 1], b[i]);
    if prev.is_ascii_alphabetic() {
        return cur == b' ' || cur == b'\n';
    }
    if prev == b'\n' {
        return cur.is_ascii_alphanumeric()
            && i >= 2
            && b[i - 2].is_ascii()
            && !b[i - 2].is_ascii_whitespace()
            && b[i - 2] != 0x0B;
    }
    cur >= 0xE3 && i >= 3 && cjk_mark_at(b, i) && han_before(b, i)
}

fn cjk_mark_at(b: &[u8], i: usize) -> bool {
    matches!(
        b.get(i..i + 3),
        Some(
            [0xEF, 0xBC, 0x8C]          // ，
                | [0xE3, 0x80, 0x82]    // 。
                | [0xE3, 0x80, 0x81]    // 、
                | [0xEF, 0xBC, 0x81]    // ！
                | [0xEF, 0xBC, 0x9F]    // ？
                | [0xEF, 0xBC, 0x9B]    // ；
                | [0xEF, 0xBC, 0x9A] // ：
        )
    )
}

fn han_before(b: &[u8], i: usize) -> bool {
    let [b0, b1, b2] = [b[i - 3], b[i - 2], b[i - 1]];
    if b0 & 0xF0 != 0xE0 || b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 {
        return false;
    }
    let cp = (u32::from(b0 & 0x0F) << 12) | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F);
    (0x4E00..=0x9FA5).contains(&cp)
}

#[cfg(test)]
#[path = "split_tests.rs"]
mod tests;
