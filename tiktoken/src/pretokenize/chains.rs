//! Window decisions for one classified chunk, as three independent chains.
//!
//! Each window opens at the previous window's cut, so a single chain of
//! windows runs at the latency of one decision per window. The chunk is
//! instead cut about a third and two thirds of the way in, at piece starts
//! that need no decision (an ASCII letter followed by a space or a newline:
//! no rule of any pattern takes a letter and the whitespace after it into one
//! piece), and each part is decided by its own chain, a window of each in
//! turn, so that they overlap. The chains record piece starts in one bitmap,
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
    // window bits stay below `limit`, and `or_at` writes one word past
    let mut starts = Starts::<W>::new(limit / 64 + 1);
    let mut step = |r: usize, end: usize| {
        step::<R, CONT, W>(pre, maps, text, buf, start, limit, r, end, &mut starts)
    };
    // chain k decides from `m[k]` while its window start is below `m[k +
    // 1]`, keeping starts below `e[k]`; the last chain with a start keeps
    // everything. Where no start is found, the chain before does that part
    // too. ASCII chunks are cut a third and two thirds of the way in; chunks
    // with other bytes only in the middle: their longer rules in three
    // chains ran out of registers, and the spills cost more than the
    // overlap gained
    let first = if CONT { stop / 2 } else { stop / 3 };
    let m1 = boundary::<W>(R::CASE, maps, first, 0, stop).unwrap_or(stop);
    let m2 = if !CONT && m1 < stop {
        boundary::<W>(R::CASE, maps, (2 * stop / 3).max(m1), m1, stop).unwrap_or(stop)
    } else {
        stop
    };
    let e0 = if m1 < stop { m1 } else { usize::MAX };
    let e1 = if m2 < stop { m2 } else { usize::MAX };
    // plain locals, not arrays indexed in a loop: a chain's next start must
    // stay in a register, or every window waits on a store and a reload
    let (mut r0, mut r1, mut r2) = (0, m1, m2);
    if !CONT {
        while r0 < m1 && r1 < m2 && r2 < stop {
            r0 = step(r0, e0);
            r1 = step(r1, e1);
            r2 = step(r2, usize::MAX);
        }
    }
    while r0 < m1 && r1 < m2 {
        r0 = step(r0, e0);
        r1 = step(r1, e1);
    }
    while r0 < m1 {
        r0 = step(r0, e0);
    }
    while r1 < m2 {
        r1 = step(r1, e1);
    }
    while r2 < stop {
        r2 = step(r2, usize::MAX);
    }
    let reach = if m2 < stop {
        r2
    } else if m1 < stop {
        r1
    } else {
        r0
    };
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
    // most windows of a chunk with some non-ASCII bytes have none of their
    // own, and the rules without continuation bytes are much shorter
    let (s, cut) = if CONT && m.cont & m.v != 0 {
        R::decide::<true>(&m, false, win)
    } else {
        R::decide::<false>(&m, false, win)
    };
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

/// Offsets of the set bits of each byte value, ascending, then zeros.
static BYTE_BITS: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut v = 0;
    while v < 256 {
        let (mut i, mut k) = (0, 0);
        while i < 8 {
            if v >> i & 1 == 1 {
                t[v][k] = i as u16;
                k += 1;
            }
            i += 1;
        }
        v += 1;
    }
    t
};

/// Appends the offsets of the set bits of `word` (word `k` of the bitmap) to
/// `pos` after `n` entries; returns the new count. Each byte of the word
/// writes eight entries from a table at its own place, found from a prefix
/// sum of the byte counts, so nothing waits on the previous bit or byte.
#[inline(always)]
fn flatten(word: u64, k: usize, pos: *mut u16, n: usize) -> usize {
    debug_assert!(n <= POS_BATCH);
    // bit counts per byte, then twice the count of all bytes below each
    // byte: the byte offset of its entries, at most 112
    let mut c = word - ((word >> 1) & 0x5555_5555_5555_5555);
    c = (c & 0x3333_3333_3333_3333) + ((c >> 2) & 0x3333_3333_3333_3333);
    c = (c + (c >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    let below2 = c.wrapping_mul(0x0202_0202_0202_0200);
    let total = ((below2 >> 57) + (c >> 56)) as usize;
    // SAFETY: n <= POS_BATCH
    let out = unsafe { pos.add(n).cast::<u8>() };
    let base = (64 * k) as u16;
    for j in 0..8 {
        let b = (word >> (8 * j)) as u8;
        let at = ((below2 >> (8 * j)) & 0xFF) as usize;
        let v = BYTE_BITS[b as usize].map(|x| x + (base + 8 * j as u16));
        // entries past the byte's count are overwritten by the next byte or
        // left past the word's count
        debug_assert!(n + at / 2 + 8 <= POS_CAP);
        // SAFETY: n <= POS_BATCH and at / 2 <= 56, so the eight entries end
        // inside POS_CAP
        unsafe { out.add(at).cast::<[u16; 8]>().write_unaligned(v) };
    }
    n + total
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_lists_every_set_bit() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for k in 0..4000usize {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let word = match k % 4 {
                0 => s,
                1 => s & s.rotate_left(7),
                2 => u64::MAX << (k % 64),
                _ => s & s.rotate_left(3) & s.rotate_left(11),
            };
            let mut buf = [0u16; POS_CAP];
            let n0 = k % 9;
            let n = flatten(word, k % 64, buf.as_mut_ptr(), n0);
            let want: Vec<u16> = (0..64)
                .filter(|i| word >> i & 1 == 1)
                .map(|i| (64 * (k % 64) + i) as u16)
                .collect();
            assert_eq!(&buf[n0..n], &want[..], "{word:#x}");
        }
    }
}
