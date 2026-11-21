//! Split a text into pieces with the window rules.
//!
//! The body of the text is classified [`CHUNK`] bytes at a time, from the
//! current piece start, and its windows decided (see `chains.rs`) while a
//! whole window fits in the chunk; the next chunk opens at the first undecided
//! start. The
//! last bytes go through a zero-padded stack copy, so that every piece start
//! handed to the sink has [`KEY_READ`] readable bytes after it.

use super::chains::decide_chunk;
use super::classify::{BODY_WORDS, Bitmaps, CHUNK, classify};
use super::window::{Masks, WindowRules};
use super::{PreTokenizer, ScalarPreTokenizer, dense};
use crate::vocab::KEY_READ;

/// Receives the pieces in order.
pub(crate) trait PieceSink {
    /// Called before each window, which emits at most 64 pieces.
    fn window(&mut self);
    /// The piece `buf[start..end]`; `buf` has at least `start + KEY_READ`
    /// bytes, those past the piece being arbitrary.
    fn piece(&mut self, buf: &[u8], start: usize, end: usize);
    /// A piece of dense non-ASCII text, which is seldom a token by itself;
    /// `window` was called right before it.
    #[inline(always)]
    fn dense_piece(&mut self, buf: &[u8], start: usize, end: usize) {
        self.piece(buf, start, end);
    }

    /// The pieces `buf[start + offs[i]..start + offs[i + 1]]` in order, with
    /// [`KEY_READ`] readable bytes from each start.
    #[inline(always)]
    fn pieces(&mut self, buf: &[u8], start: usize, offs: &[u16]) {
        let mut i = 0;
        while i + 1 < offs.len() {
            self.window();
            let end = (i + 64).min(offs.len() - 1);
            for j in i..end {
                self.piece(buf, start + offs[j] as usize, start + offs[j + 1] as usize);
            }
            i = end;
        }
    }
}

/// Texts shorter than this go through the tail copy alone.
const TAIL_MAX: usize = 2 * 64 + KEY_READ;
const TAIL_BUF: usize = 3 * 64;

/// Above this many non-ASCII bytes per [`DENSE_PROBE`] bytes at the start of
/// a chunk (or of the tail), it is scanned piece by piece (`dense.rs`): dense
/// CJK runs are longer than a window, and their per-char classification
/// costs more than the scan.
const DENSE_LIMIT: u32 = 32;
const DENSE_PROBE: usize = 256;

#[inline(always)]
pub(crate) fn split<R: WindowRules, S: PieceSink>(
    pre: &ScalarPreTokenizer,
    text: &str,
    sink: &mut S,
) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut pos = 0;
    if n >= TAIL_MAX {
        let mut maps = Bitmaps::<BODY_WORDS>::new();
        while n - pos >= TAIL_MAX {
            let blocks = ((n - pos - KEY_READ) / 64).min(CHUNK / 64);
            let limit = 64 * blocks;
            if dense(&bytes[pos..pos + limit]) {
                pos = dense::span(pre, text, bytes, pos, pos + limit, sink);
                continue;
            }
            let cont = classify_for::<R, BODY_WORDS>(bytes, pos, blocks, &mut maps);
            let stop = limit - 63;
            pos += if cont {
                decide_chunk::<R, true, S, BODY_WORDS>(
                    pre, &maps, text, bytes, pos, limit, stop, sink,
                )
            } else {
                decide_chunk::<R, false, S, BODY_WORDS>(
                    pre, &maps, text, bytes, pos, limit, stop, sink,
                )
            };
        }
    }
    let tl = n - pos;
    if tl == 0 {
        return;
    }
    let mut buf = [0u8; TAIL_BUF];
    buf[..tl].copy_from_slice(&bytes[pos..]);
    let tail = &text[pos..];
    if dense(&buf[..tl]) {
        dense::span(pre, tail, &buf, 0, tl, sink);
        return;
    }
    let mut maps = Bitmaps::<4>::new();
    let blocks = tl.div_ceil(64);
    let cont = classify_for::<R, 4>(&buf, 0, blocks, &mut maps);
    if cont {
        windows::<R, true, S, 4>(pre, &maps, tail, &buf, 0, tl, tl, true, sink);
    } else {
        windows::<R, false, S, 4>(pre, &maps, tail, &buf, 0, tl, tl, true, sink);
    }
}

/// Decide windows from `start` while the window start is below `start +
/// stop`, the classified bytes ending at `start + limit`. Returns how far the
/// pieces reach past `start`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn windows<R: WindowRules, const CONT: bool, S: PieceSink, const W: usize>(
    pre: &ScalarPreTokenizer,
    maps: &Bitmaps<W>,
    text: &str,
    buf: &[u8],
    start: usize,
    limit: usize,
    stop: usize,
    eot: bool,
    sink: &mut S,
) -> usize {
    let mut r = 0;
    while r < stop {
        sink.window();
        let (m, reaches) = load_for::<R, W>(maps, r, limit);
        let base = start + r;
        let (s, cut) = R::decide::<CONT>(&m, eot && reaches, &buf[base..]);
        if cut == 0 {
            let end = scalar_end(pre, text, base);
            sink.piece(buf, base, end);
            r = end - start;
            continue;
        }
        let mut bits = if cut == 64 {
            s
        } else {
            s & ((1u64 << cut) - 1)
        };
        while bits != 0 {
            let a = bits.trailing_zeros();
            bits &= bits - 1;
            let b = if bits == 0 {
                cut
            } else {
                bits.trailing_zeros()
            };
            sink.piece(buf, base + a as usize, base + b as usize);
        }
        r += cut as usize;
    }
    r
}

/// End of the piece at `pos` by the scalar scanner.
#[inline(never)]
pub(super) fn scalar_end(pre: &ScalarPreTokenizer, text: &str, pos: usize) -> usize {
    pre.next_match(text, pos).map_or(text.len(), |(_, end)| end)
}

#[inline(always)]
fn classify_for<R: WindowRules, const W: usize>(
    text: &[u8],
    start: usize,
    blocks: usize,
    maps: &mut Bitmaps<W>,
) -> bool {
    if R::CASE {
        classify::<true, W>(R::classes(), text, start, blocks, maps)
    } else {
        classify::<false, W>(R::classes(), text, start, blocks, maps)
    }
}

#[inline(always)]
fn load_for<R: WindowRules, const W: usize>(
    maps: &Bitmaps<W>,
    r: usize,
    limit: usize,
) -> (Masks, bool) {
    if R::CASE {
        Masks::load::<true, W>(maps, r, limit)
    } else {
        Masks::load::<false, W>(maps, r, limit)
    }
}

#[inline(always)]
fn dense(chunk: &[u8]) -> bool {
    let probe = &chunk[..chunk.len().min(DENSE_PROBE)];
    // most text is ASCII: one pass of ors before counting anything
    let any = probe
        .as_chunks::<8>()
        .0
        .iter()
        .fold(0, |a, w| a | u64::from_le_bytes(*w));
    if any & 0x8080_8080_8080_8080 == 0 {
        return false;
    }
    let hi: u32 = probe
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| (u64::from_le_bytes(*w) & 0x8080_8080_8080_8080).count_ones())
        .sum();
    hi as usize * DENSE_PROBE > DENSE_LIMIT as usize * probe.len()
}
