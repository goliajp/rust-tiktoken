//! The merge of many short pieces at once, by token id.
//!
//! One piece's merge is a chain of rounds, each of which needs the previous
//! round's lookups to pick its pair. Merged alone, the chain either stalls on
//! every lookup or, written with branches, mispredicts the chosen pair on
//! most rounds of a piece the predictor has not seen. Here eight pieces are
//! in flight at once, each round is free of data-dependent branches, and a
//! lane takes the next piece as soon as its own is done, so the rounds of
//! different pieces overlap.
//!
//! A lane holds its parts by byte position: a bit mask of part starts, the
//! id of the part at each start, and the rank of each part joined with the
//! next. Position 0 is a sentinel part before the piece and position
//! `n + 1` one after it; their id is [`NO_ID`], which no pair holds, so
//! the neighbours of a merge need no bounds checks. Ranks of joined parts
//! come from [`Splits`] by the two parts' ids, which matches the
//! substring rule exactly (see `vocab::splits`).

use crate::vocab::{NO_ID, Splits, Vocab};

/// Longest piece a lane takes.
pub(crate) const LANE_MAX: usize = 15;

const LANES: usize = 8;

#[cfg(target_arch = "aarch64")]
mod ranks {
    use std::arch::aarch64::*;

    /// Sixteen ranks, written one at a time and read four at a time.
    #[derive(Clone, Copy)]
    #[repr(C, align(64))]
    pub(super) struct Ranks([u32; 16]);

    impl Ranks {
        #[inline(always)]
        pub(super) fn new(r: &[u32; 16]) -> Self {
            Self(*r)
        }

        /// The leftmost lowest rank and its position.
        #[inline(always)]
        pub(super) fn min_pos(&self) -> (usize, u32) {
            let p = self.0.as_ptr();
            // SAFETY: four loads of four lanes from a 16-lane array
            unsafe {
                let (a, b, c, d) = (
                    vld1q_u32(p),
                    vld1q_u32(p.add(4)),
                    vld1q_u32(p.add(8)),
                    vld1q_u32(p.add(12)),
                );
                let m = vminvq_u32(vminq_u32(vminq_u32(a, b), vminq_u32(c, d)));
                let mv = vdupq_n_u32(m);
                let ab = vcombine_u16(vmovn_u32(vceqq_u32(a, mv)), vmovn_u32(vceqq_u32(b, mv)));
                let cd = vcombine_u16(vmovn_u32(vceqq_u32(c, mv)), vmovn_u32(vceqq_u32(d, mv)));
                let bytes = vcombine_u8(vmovn_u16(ab), vmovn_u16(cd));
                // four bits per lane
                let nib = vshrn_n_u16(vreinterpretq_u16_u8(bytes), 4);
                let mask = vget_lane_u64(vreinterpret_u64_u8(nib), 0);
                ((mask.trailing_zeros() / 4) as usize & 15, m)
            }
        }

        /// Set rank `j` to none, then `k` to `rk`, then `p` to `rp`.
        #[inline(always)]
        pub(super) fn set(&mut self, j: usize, k: usize, rk: u32, p: usize, rp: u32) {
            self.0[j & 15] = u32::MAX;
            self.0[k & 15] = rk;
            self.0[p & 15] = rp;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod ranks {
    /// Sixteen ranks; the portable form of the vector one.
    #[derive(Clone, Copy)]
    pub(super) struct Ranks([u32; 16]);

    impl Ranks {
        #[inline(always)]
        pub(super) fn new(r: &[u32; 16]) -> Self {
            Self(*r)
        }

        #[inline(always)]
        pub(super) fn min_pos(&self) -> (usize, u32) {
            let m = self.0.iter().fold(u32::MAX, |a, &b| a.min(b));
            let mask = self
                .0
                .iter()
                .enumerate()
                .fold(0u32, |acc, (i, &r)| acc | ((r == m) as u32) << i);
            (mask.trailing_zeros() as usize & 15, m)
        }

        #[inline(always)]
        pub(super) fn set(&mut self, j: usize, k: usize, rk: u32, p: usize, rp: u32) {
            self.0[j] = u32::MAX;
            self.0[k] = rk;
            self.0[p] = rp;
        }
    }
}

use ranks::Ranks;

#[derive(Clone, Copy)]
struct Lane {
    ranks: Ranks,
    // id of the part starting at each position; NO_ID at the sentinels
    ids: [u32; 32],
    // bit i: a part starts at position i (bit 0 and bit n + 1: sentinels)
    starts: u32,
    n: u32,
}

impl Lane {
    #[inline(always)]
    fn new(piece: &[u8], vocab: &Vocab) -> Self {
        let n = piece.len();
        debug_assert!((2..=LANE_MAX).contains(&n));
        let mut b = [0u8; LANE_MAX + 2];
        b[..n].copy_from_slice(piece);
        let mut ids = [NO_ID; 32];
        let mut rank = [u32::MAX; 16];
        // byte i is the part at position i + 1
        for i in 0..LANE_MAX {
            let id = vocab.rank_single(b[i]);
            ids[i + 1] = if i < n { id } else { NO_ID };
            let r = vocab.rank_pair(b[i], b[i + 1]);
            rank[i + 1] = if i + 1 < n { r } else { u32::MAX };
        }
        Self {
            ranks: Ranks::new(&rank),
            ids,
            starts: (1u32 << (n + 2)) - 1,
            n: n as u32,
        }
    }

    /// One merge; false when the piece has none left.
    #[inline(always)]
    fn step(&mut self, splits: &Splits) -> bool {
        let (k, m) = self.ranks.min_pos();
        let live = m != u32::MAX;
        let above = !0u32 << (k + 1);
        // the part after k joins it; then its successor and k's predecessor
        // are k's neighbours
        let j = (self.starts & above).trailing_zeros() as usize & 31;
        self.starts ^= (live as u32) << j;
        let nj = (self.starts & above).trailing_zeros() as usize & 31;
        let p = 31 - ((self.starts & ((1 << k) - 1)) | 1).leading_zeros() as usize;
        self.ids[k] = m;
        let rk = splits.get(m, self.ids[nj]);
        let rp = splits.get(self.ids[p], m);
        self.ranks.set(j & 15, k, rk, p, rp);
        live
    }

    /// Write the part ids to `out`; returns their count.
    #[inline(always)]
    fn out(&self, out: &mut [u32; 16]) -> usize {
        let real = self.starts & ((1 << (self.n + 1)) - 2);
        let mut c = 0;
        for (i, &id) in self.ids.iter().enumerate().take(16).skip(1) {
            out[c] = id;
            c += (real >> i & 1) as usize;
        }
        c
    }
}

/// Merge pieces of `2..=LANE_MAX` bytes: the ids of `pieces[i]` go to
/// `out[i][..counts[i]]`. The vocabulary has every byte as a token.
pub(crate) fn merge_by_id(
    pieces: &[&[u8]],
    vocab: &Vocab,
    splits: &Splits,
    out: &mut [[u32; 16]],
    counts: &mut [u8],
) {
    let total = pieces.len();
    let first = total.min(LANES);
    let idle = Lane {
        ranks: Ranks::new(&[u32::MAX; 16]),
        ids: [NO_ID; 32],
        starts: 0,
        n: 0,
    };
    let mut lanes = [idle; LANES];
    let mut which = [usize::MAX; LANES];
    for w in 0..first {
        lanes[w] = Lane::new(pieces[w], vocab);
        which[w] = w;
    }
    let (mut next, mut live) = (first, first);
    while live > 0 {
        for w in 0..LANES {
            if which[w] == usize::MAX || lanes[w].step(splits) {
                continue;
            }
            let i = which[w];
            counts[i] = lanes[w].out(&mut out[i]) as u8;
            if next < total {
                lanes[w] = Lane::new(pieces[next], vocab);
                which[w] = next;
                next += 1;
            } else {
                which[w] = usize::MAX;
                live -= 1;
            }
        }
    }
}
