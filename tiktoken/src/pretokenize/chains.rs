//! Window decisions for one classified chunk, as independent chains.
//!
//! Each window opens at the previous window's cut, so a single chain of
//! windows runs at the latency of one decision per window. The chunk is
//! instead cut at piece starts that need no decision (an ASCII letter
//! followed by a space or a newline: no rule of any pattern takes a letter
//! and the whitespace after it into one piece) and each part is decided by
//! its own chain, the chains interleaved so that their windows overlap. The
//! chains record piece starts in one bitmap, which is then walked in order.

use std::mem::MaybeUninit;

use super::ScalarPreTokenizer;
use super::classify::{Bitmaps, M_CONT, M_L, M_LW, M_NL, M_SP};
use super::window::{Masks, WindowRules};
use super::windowed::{PieceSink, scalar_end};

/// Chains per chunk.
const CHAINS: usize = 2;
/// Words searched for a chain boundary from its nominal position.
const SEARCH_WORDS: usize = 4;

/// Bit `i` of word `i / 64` set: byte `i` from the chunk start begins a piece.
pub(super) struct Starts<const W: usize>([u64; W]);

impl<const W: usize> Starts<W> {
    #[inline(always)]
    pub(super) fn new() -> Self {
        Self([0; W])
    }

    /// Or in `s`, bit 0 standing for byte `at`. Needs `at / 64 + 1 < W`.
    #[inline(always)]
    fn or_at(&mut self, at: usize, s: u64) {
        let (k, sh) = (at >> 6, (at & 63) as u32);
        debug_assert!(k + 1 < W);
        // SAFETY: the caller keeps `at` below `64 * (W - 1)`
        unsafe {
            *self.0.get_unchecked_mut(k) |= s << sh;
            // two shifts: a shift by 64 is not a no-op
            *self.0.get_unchecked_mut(k + 1) |= (s >> 1) >> (63 - sh);
        }
    }
}

/// The first byte `b` in `from + 1..to`, searching from word `at / 64`, that
/// follows an ASCII letter and is a space or a newline.
#[inline(always)]
fn boundary<const W: usize>(
    case: bool,
    maps: &Bitmaps<W>,
    at: usize,
    from: usize,
    to: usize,
) -> Option<usize> {
    let k0 = at / 64;
    for k in k0..(k0 + SEARCH_WORDS).min(to.div_ceil(64)) {
        let mut l = maps.window(M_L, 64 * k);
        if case {
            l |= maps.window(M_LW, 64 * k);
        }
        // a letter byte that is not a continuation byte is ASCII: a lead
        // byte is never followed by a space
        let ascii_l = l & !maps.window(M_CONT, 64 * k);
        let gap = maps.window(M_SP, 64 * k) | maps.window(M_NL, 64 * k);
        let mut c = gap & (ascii_l << 1);
        while c != 0 {
            let b = 64 * k + c.trailing_zeros() as usize;
            c &= c - 1;
            if b > from && b < to {
                return Some(b);
            }
        }
    }
    None
}

/// Decide the windows of a chunk classified in `maps` (bytes `start..start +
/// limit` of `buf`), from the chunk start while a window start is below
/// `stop`, and hand the pieces to `sink`. Returns how far the pieces reach
/// past `start`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn decide_chunk<R: WindowRules, const CONT: bool, S: PieceSink, const W: usize>(
    pre: &ScalarPreTokenizer,
    maps: &Bitmaps<W>,
    text: &str,
    buf: &[u8],
    start: usize,
    limit: usize,
    stop: usize,
    sink: &mut S,
) -> usize {
    // chain i decides from r[i] while r[i] < run[i], keeping starts below
    // end[i]; the last chain runs to `stop` and keeps everything
    let mut r = [0usize; CHAINS];
    let mut run = [0usize; CHAINS];
    let mut end = [usize::MAX; CHAINS];
    let mut n = 1;
    for i in 1..CHAINS {
        let at = stop * i / CHAINS;
        if let Some(b) = boundary::<W>(R::CASE, maps, at, r[n - 1], stop) {
            run[n - 1] = b;
            end[n - 1] = b;
            r[n] = b;
            n += 1;
        }
    }
    run[n - 1] = stop;
    let mut starts = Starts::<W>::new();
    loop {
        let mut active = false;
        for i in 0..CHAINS {
            if r[i] < run[i] {
                active = true;
                r[i] = step::<R, CONT, W>(
                    pre,
                    maps,
                    text,
                    buf,
                    start,
                    limit,
                    r[i],
                    end[i],
                    &mut starts,
                );
            }
        }
        if !active {
            break;
        }
    }
    let reach = r[n - 1];
    walk(&starts, buf, start, reach, sink);
    reach
}

/// One window of a chain at `r`; returns the next window start.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn step<R: WindowRules, const CONT: bool, const W: usize>(
    pre: &ScalarPreTokenizer,
    maps: &Bitmaps<W>,
    text: &str,
    buf: &[u8],
    start: usize,
    limit: usize,
    r: usize,
    end: usize,
    starts: &mut Starts<W>,
) -> usize {
    debug_assert!(r + 64 <= limit);
    let m = if R::CASE {
        Masks::load_body::<true, W>(maps, r)
    } else {
        Masks::load_body::<false, W>(maps, r)
    };
    let base = start + r;
    debug_assert!(base < buf.len());
    // SAFETY: the window lies inside the classified bytes of `buf`
    let win = unsafe { buf.get_unchecked(base..) };
    let (s, cut) = R::decide::<CONT>(&m, false, win);
    if cut == 0 {
        starts.or_at(r, 1);
        let e = scalar_end(pre, text, base) - start;
        debug_assert!(e <= end);
        return e;
    }
    let keep = (end - r).min(cut as usize);
    starts.or_at(
        r,
        if keep >= 64 {
            s
        } else {
            s & ((1u64 << keep) - 1)
        },
    );
    r + cut as usize
}

/// Piece starts handed to the sink at a time. The buffer lives on the stack,
/// and a small one keeps a fresh thread's first call from touching more
/// stack pages.
const POS_BATCH: usize = 512;
/// [`POS_BATCH`] plus one word's starts and the overshoot of [`flatten`].
const POS_CAP: usize = POS_BATCH + 64 + 8;

/// Appends the offsets of the set bits of `word` (word `k` of the bitmap) to
/// `pos` after `n` entries; returns the new count. Writes in groups of eight,
/// so the loop branches on the count rather than once per bit.
#[inline(always)]
fn flatten(word: u64, k: usize, pos: *mut u16, n: usize) -> usize {
    debug_assert!(n <= POS_BATCH);
    let mut m = word;
    let cnt = m.count_ones() as usize;
    let base = 64 * k as u16;
    let put = |j: usize, m: &mut u64| {
        // a spent word writes base + 64, past the count and ignored
        debug_assert!(n + j < POS_CAP);
        // SAFETY: n <= POS_BATCH, and a word writes at most 64 + 8 entries
        unsafe { pos.add(n + j).write(base + m.trailing_zeros() as u16) };
        *m &= m.wrapping_sub(1);
    };
    for j in 0..8 {
        put(j, &mut m);
    }
    if cnt > 8 {
        for j in 8..16 {
            put(j, &mut m);
        }
        let mut j = 16;
        while m != 0 {
            put(j, &mut m);
            j += 1;
        }
    }
    n + cnt
}

/// Hand the pieces between consecutive starts below `reach` to `sink`, the
/// last one ending at `reach`.
#[inline(always)]
fn walk<S: PieceSink, const W: usize>(
    starts: &Starts<W>,
    buf: &[u8],
    start: usize,
    reach: usize,
    sink: &mut S,
) {
    let mut pos = MaybeUninit::<[u16; POS_CAP]>::uninit();
    let pos = pos.as_mut_ptr().cast::<u16>();
    let mut n = 0;
    for (k, &word) in starts.0.iter().enumerate().take(reach.div_ceil(64)) {
        n = flatten(word, k, pos, n);
        if n >= POS_BATCH {
            // SAFETY: entries 0..n are written
            let offs = unsafe { std::slice::from_raw_parts(pos, n) };
            sink.pieces(buf, start, offs);
            // SAFETY: as above; the last start opens the next batch
            unsafe { pos.write(pos.add(n - 1).read()) };
            n = 1;
        }
    }
    // SAFETY: entries 0..n are written, and bit 0 put the chunk start first
    let offs = unsafe { std::slice::from_raw_parts(pos, n) };
    debug_assert!(n >= 1);
    sink.pieces(buf, start, offs);
    sink.window();
    sink.piece(buf, start + offs[n - 1] as usize, start + reach);
}
