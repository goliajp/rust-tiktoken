//! Open-addressed table from packed keys to token ids, shared by all threads.
//!
//! Every vocabulary token of up to 15 bytes has a slot. Slots come in pairs
//! that share a 64-byte cache line; a key hashes to a line and takes the
//! first free slot from there on. Tokens are inserted in ascending rank, so
//! the frequent low-rank tokens land in their home line and the encode loop
//! answers most pieces by comparing the two slots of one line; anything else
//! goes through [`FastTable::get`], which walks the probe chain.
//!
//! The slot fields are atomics so that entries can later be published into a
//! live table: a writer fills `hi` and `ids` before storing `lo` with
//! `Release`, and readers load `lo` with `Acquire`. On aarch64 that load is an
//! `ldapr`, which measured no slower than a plain `ldr` in this probe loop.

use std::sync::atomic::{AtomicU64, Ordering};

use super::key::hash;

#[repr(C)]
struct FastSlot {
    lo: AtomicU64,
    // zero only in an empty slot
    hi: AtomicU64,
    // first id in the low half, id count in the high half
    w0: AtomicU64,
    // second and third ids, for entries that map to more than one token
    w1: AtomicU64,
}

#[repr(C, align(64))]
struct FastLine([FastSlot; 2]);

impl FastSlot {
    #[inline(always)]
    fn matches(&self, lo: u64, hi: u64) -> bool {
        (self.lo.load(Ordering::Acquire) == lo) & (self.hi.load(Ordering::Relaxed) == hi)
    }

    #[inline(always)]
    fn id(&self) -> u32 {
        self.w0.load(Ordering::Relaxed) as u32
    }
}

pub(crate) struct FastTable {
    lines: Box<[FastLine]>,
    shift: u32,
}

impl FastTable {
    /// Build from `(lo, hi, id)` triples, which the caller orders by rank.
    pub(crate) fn build(keys: &[(u64, u64, u32)]) -> Self {
        let n = keys.len();
        let mut slots = (n * 2).max(4).next_power_of_two();
        // wasm builds trade probe length for half the memory
        if cfg!(target_arch = "wasm32") && slots / 2 > n + n / 4 {
            slots /= 2;
        }
        let size = slots / 2;
        let empty = || FastSlot {
            lo: AtomicU64::new(0),
            hi: AtomicU64::new(0),
            w0: AtomicU64::new(0),
            w1: AtomicU64::new(0),
        };
        let mut lines: Box<[FastLine]> = (0..size).map(|_| FastLine([empty(), empty()])).collect();
        let shift = 64 - size.trailing_zeros();
        let mask = size - 1;
        for &(lo, hi, id) in keys {
            debug_assert!(hi != 0);
            let mut idx = (hash(lo, hi) >> shift) as usize;
            let slot = loop {
                let [a, b] = &mut lines[idx].0;
                if *a.hi.get_mut() == 0 {
                    break a;
                }
                if *b.hi.get_mut() == 0 {
                    break b;
                }
                idx = (idx + 1) & mask;
            };
            *slot.lo.get_mut() = lo;
            *slot.hi.get_mut() = hi;
            *slot.w0.get_mut() = id as u64 | (1u64 << 32);
        }
        Self { lines, shift }
    }

    #[inline(always)]
    fn home_index(&self, lo: u64, hi: u64) -> usize {
        (hash(lo, hi) >> self.shift) as usize
    }

    /// The id stored in the key's home line, if the key is there.
    #[inline(always)]
    pub(crate) fn home(&self, lo: u64, hi: u64) -> Option<u32> {
        let [a, b] = &self.lines[self.home_index(lo, hi)].0;
        if a.matches(lo, hi) {
            Some(a.id())
        } else if b.matches(lo, hi) {
            Some(b.id())
        } else {
            None
        }
    }

    /// The key's id, wherever it sits on its probe chain.
    #[inline]
    pub(crate) fn get(&self, lo: u64, hi: u64) -> Option<u32> {
        let mask = self.lines.len() - 1;
        let mut idx = self.home_index(lo, hi);
        loop {
            for slot in &self.lines[idx].0 {
                if slot.matches(lo, hi) {
                    return Some(slot.id());
                }
                if slot.hi.load(Ordering::Relaxed) == 0 {
                    return None;
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Whether the key sits in its home line; `None` if absent.
    #[cfg(test)]
    pub(crate) fn is_home(&self, lo: u64, hi: u64) -> Option<bool> {
        self.get(lo, hi)?;
        Some(self.home(lo, hi).is_some())
    }

    /// Number of slots.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lines.len() * 2
    }
}
