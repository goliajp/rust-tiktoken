//! Window decisions for one classified chunk, as two independent chains.
//!
//! Each window opens at the previous window's cut, so a single chain of
//! windows runs at the latency of one decision per window. The chunk is
//! instead cut near its middle at a piece start that needs no decision (an
//! ASCII letter followed by a space or a newline: no rule of any pattern
//! takes a letter and the whitespace after it into one piece), and each half
//! is decided by its own chain, a window of one after a window of the other,
//! so that the two overlap. The chains record piece starts in one bitmap,
//! which is then walked in order.

use std::mem::MaybeUninit;

use super::ScalarPreTokenizer;
use super::classify::{Bitmaps, M_CONT, M_L, M_LW, M_NL, M_SP};
use super::window::{Masks, WindowRules};
use super::windowed::{PieceSink, scalar_end};

/// Words searched for the second chain's start from the middle.
const SEARCH_WORDS: usize = 4;

/// Bit `i` of word `i / 64` set: byte `i` from the chunk start begins a piece.
/// Only the first `len` words are in use (and initialized).
pub(super) struct Starts<const W: usize> {
    words: MaybeUninit<[u64; W]>,
    len: usize,
}

impl<const W: usize> Starts<W> {
    /// `len` zeroed words.
    #[inline(always)]
    pub(super) fn new(len: usize) -> Self {
        debug_assert!(len <= W);
        let mut words = MaybeUninit::<[u64; W]>::uninit();
        // SAFETY: the first `len` of the W words
        unsafe { words.as_mut_ptr().cast::<u64>().write_bytes(0, len) };
        Self { words, len }
    }

    /// The words in use.
    #[inline(always)]
    fn words(&self) -> &[u64] {
        // SAFETY: `new` initialized the first `len` words
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u64>(), self.len) }
    }

    /// Or in `s`, bit 0 standing for byte `at`. Needs `at / 64 + 1 < len`.
    #[inline(always)]
    fn or_at(&mut self, at: usize, s: u64) {
        let (k, sh) = (at >> 6, (at & 63) as u32);
        debug_assert!(k + 1 < self.len);
        // SAFETY: the caller keeps `at` below `64 * (len - 1)`
        unsafe {
            let w = self.words.as_mut_ptr().cast::<u64>();
            *w.add(k) |= s << sh;
            // two shifts: a shift by 64 is not a no-op
            *w.add(k + 1) |= (s >> 1) >> (63 - sh);
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
    // chain a decides from 0 while its window start is below `mid`, keeping
    // starts below `end_a`, and chain b from `mid` while below `stop`,
    // keeping everything; with no start found near the middle, chain a does
    // it all
    let (mid, end_a) = match boundary::<W>(R::CASE, maps, stop / 2, 0, stop) {
        Some(b) => (b, b),
        None => (stop, usize::MAX),
    };
    // window bits stay below `limit`, and `or_at` writes one word past
    let mut starts = Starts::<W>::new(limit / 64 + 1);
    let mut step = |r: usize, end: usize| {
        step::<R, CONT, W>(pre, maps, text, buf, start, limit, r, end, &mut starts)
    };
    // plain locals, not arrays indexed in a loop: a chain's next start must
    // stay in a register, or every window waits on a store and a reload
    let (mut ra, mut rb) = (0, mid);
    while ra < mid && rb < stop {
        ra = step(ra, end_a);
        rb = step(rb, usize::MAX);
    }
    while ra < mid {
        ra = step(ra, end_a);
    }
    while rb < stop {
        rb = step(rb, usize::MAX);
    }
    let reach = if mid < stop { rb } else { ra };
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
    for (k, &word) in starts.words().iter().enumerate().take(reach.div_ceil(64)) {
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
