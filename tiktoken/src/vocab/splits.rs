//! Merge results by token id: for each token `T` that merges from its own
//! bytes, the pair of tokens `(A, B)` that its last merge step joins, mapped
//! to `T`.
//!
//! This answers the merge's question exactly. Whenever two adjacent parts
//! `A` and `B` of any piece concatenate to a token `T`, the merge of `T`'s
//! bytes alone ends by joining `A` and `B`: every part inside `A` and `B`
//! formed in rank order from their own bytes, and a step of `T`'s own merge
//! that crossed the `A | B` boundary first would have happened in the piece
//! as well, so `A` and `B` would not both be parts. One pair per token thus
//! suffices, and a pair missing from the table is a pair that does not
//! merge.
//!
//! The table is built from the vocabulary in order of token length: a
//! token's own merge only ever forms shorter tokens, whose pairs are in the
//! table by then, and it stops with two parts exactly when the token merges
//! from its bytes. Lookups are a two-choice cuckoo hash of single slots, so
//! a lookup is two independent loads and no branch.

use super::{ABSENT, Vocab};
use crate::merge::lanes::{LANE_MAX, merge_by_id};

// tokens merged together while building
const BUILD_BATCH: usize = 256;

/// Ids at or above this do not fit a slot; it also marks "no part".
pub(crate) const NO_ID: u32 = (1 << 18) - 1;

const M1: u64 = 0x9E37_79B9_7F4A_7C15;
const M2: u64 = 0xC2B2_AE3D_27D4_EB4F;

pub(crate) struct Splits {
    slots: Box<[u64]>,
    shift: u32,
}

/// The pair's tag: a marker bit, so that no tag is zero, and the two ids.
/// A slot holds `tag << 24 | merged id`; an empty slot is zero.
#[inline(always)]
fn tag(a: u32, b: u32) -> u64 {
    1 << 39 | (a as u64) << 18 | b as u64
}

impl Splits {
    /// Build for `vocab`, or `None` when an id does not fit a slot or a byte
    /// is not a token (a part of such a byte has no id to look up by).
    pub(crate) fn build(vocab: &Vocab) -> Option<Self> {
        let n = vocab.decoder.len();
        if n >= NO_ID as usize || !vocab.has_every_byte() {
            return None;
        }
        // ranks of tokens of 3 bytes or more, by length; 2-byte tokens only
        // ever merge from two single bytes, which the pair table answers
        let mut by_len: Vec<(u16, u32)> = (0..n as u32)
            .filter_map(|r| vocab.try_decode(r).map(|t| (t.len() as u16, r)))
            .filter(|&(len, _)| len >= 3)
            .collect();
        by_len.sort_unstable();
        let mut entries = by_len.len();
        loop {
            if let Some(table) = Self::fill(vocab, &by_len, entries) {
                return Some(table);
            }
            // a cuckoo cycle: start over twice as large
            entries *= 2;
        }
    }

    /// The table for tokens `by_len` (ascending length) in room for
    /// `entries`, or `None` on a cuckoo cycle.
    fn fill(vocab: &Vocab, by_len: &[(u16, u32)], entries: usize) -> Option<Self> {
        let mut table = Self::empty(entries);
        let mut scratch = Scratch::default();
        let mut pieces: Vec<&[u8]> = Vec::new();
        let mut out = vec![[0u32; 16]; BUILD_BATCH];
        let mut counts = [0u8; BUILD_BATCH];
        let mut at = 0;
        while at < by_len.len() {
            let len = by_len[at].0 as usize;
            let end = at + by_len[at..].partition_point(|&(l, _)| l as usize == len);
            let group = &by_len[at..end];
            at = end;
            if len > LANE_MAX {
                for &(_, r) in group {
                    let t = vocab.try_decode(r).unwrap_or_default();
                    if let Some((a, b)) = scratch.last_pair(t, vocab, &table)
                        && !table.insert(tag(a, b) << 24 | r as u64)
                    {
                        return None;
                    }
                }
                continue;
            }
            // tokens of one length merge many at a time; their own pairs go
            // in only after the whole length is done, which changes nothing,
            // as none of them forms inside another of the same length
            let mut found = Vec::new();
            for chunk in group.chunks(BUILD_BATCH) {
                pieces.clear();
                pieces.extend(
                    chunk
                        .iter()
                        .map(|&(_, r)| vocab.try_decode(r).unwrap_or_default()),
                );
                merge_by_id(&pieces, vocab, &table, &mut out, &mut counts);
                for (i, &(_, r)) in chunk.iter().enumerate() {
                    if counts[i] == 2 {
                        found.push(tag(out[i][0], out[i][1]) << 24 | r as u64);
                    }
                }
            }
            for slot in found {
                if !table.insert(slot) {
                    return None;
                }
            }
        }
        Some(table)
    }

    fn empty(entries: usize) -> Self {
        // at most half full: two-choice cuckoo hashing with one slot per
        // place fails often beyond that
        let slots = (entries * 2).max(16).next_power_of_two();
        Self {
            slots: vec![0; slots].into_boxed_slice(),
            shift: 64 - slots.trailing_zeros(),
        }
    }

    #[inline(always)]
    fn homes(&self, t: u64) -> (usize, usize) {
        (
            (t.wrapping_mul(M1) >> self.shift) as usize,
            (t.wrapping_mul(M2) >> self.shift) as usize,
        )
    }

    /// Place a slot, moving others to their other bucket; false on a cycle.
    fn insert(&mut self, mut slot: u64) -> bool {
        let (mut at, _) = self.homes(slot >> 24);
        for _ in 0..500 {
            std::mem::swap(&mut self.slots[at], &mut slot);
            if slot == 0 {
                return true;
            }
            let (h1, h2) = self.homes(slot >> 24);
            at = if at == h1 { h2 } else { h1 };
        }
        false
    }

    /// The token that parts `a` and `b` merge into, or [`ABSENT`].
    #[inline(always)]
    pub(crate) fn get(&self, a: u32, b: u32) -> u32 {
        let t = tag(a, b);
        let (h1, h2) = self.homes(t);
        // SAFETY: `homes` maps into the power-of-two slot array
        let (s0, s1) = unsafe { (*self.slots.get_unchecked(h1), *self.slots.get_unchecked(h2)) };
        let pick = |s: u64| std::hint::select_unpredictable(s >> 24 == t, s, 0);
        let v = pick(s0) | pick(s1);
        std::hint::select_unpredictable(v != 0, v as u32 & 0xFF_FFFF, ABSENT)
    }

    /// Bytes held by the table.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.slots.len() * size_of::<u64>()
    }
}

/// Buffers of the build's merges, reused across tokens.
#[derive(Default)]
struct Scratch {
    ids: Vec<u32>,
}

impl Scratch {
    /// Merge `t` by id with the pairs of all shorter tokens; the two parts
    /// it stops at, if it stops at two.
    fn last_pair(&mut self, t: &[u8], vocab: &Vocab, table: &Splits) -> Option<(u32, u32)> {
        let ids = &mut self.ids;
        ids.clear();
        ids.extend(t.iter().map(|&b| vocab.rank_single(b)));
        // rank of each part joined with its successor; byte pairs from the
        // pair table, which holds no token as long as `t`
        let mut ranks: Vec<u32> = t.windows(2).map(|w| vocab.rank_pair(w[0], w[1])).collect();
        while let Some((k, &r)) = ranks
            .iter()
            .enumerate()
            .min_by_key(|&(i, &r)| (r, i))
            .filter(|(_, r)| **r != ABSENT)
        {
            ids[k] = r;
            ids.remove(k + 1);
            ranks.remove(k);
            if k + 1 < ids.len() {
                ranks[k] = table.get(ids[k], ids[k + 1]);
            }
            if k > 0 {
                ranks[k - 1] = table.get(ids[k - 1], ids[k]);
            }
        }
        (ids.len() == 2).then(|| (ids[0], ids[1]))
    }
}
