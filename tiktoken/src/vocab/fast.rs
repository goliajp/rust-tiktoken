//! Open-addressed table from packed keys to token ids, shared by all threads.
//!
//! Every vocabulary token of up to 15 bytes has a slot. Tokens are inserted
//! in ascending rank, so the frequent low-rank tokens own their home slot and
//! the encode loop can answer most pieces with one slot read; anything else
//! goes through [`FastTable::get`], which walks the probe chain.
//!
//! The slot fields are atomics so that entries can later be published into a
//! live table: a writer fills `hi` and `ids` before storing `lo` with
//! `Release`, and readers load `lo` with `Acquire`. On aarch64 that load is an
//! `ldapr`, which measured no slower than a plain `ldr` in this probe loop.

use std::sync::atomic::{AtomicU64, Ordering};

use super::key::hash;

#[repr(C, align(32))]
struct FastSlot {
    lo: AtomicU64,
    // zero only in an empty slot
    hi: AtomicU64,
    // first id in the low half, id count in the high half
    w0: AtomicU64,
    // second and third ids, for entries that map to more than one token
    w1: AtomicU64,
}

pub(crate) struct FastTable {
    slots: Box<[FastSlot]>,
    shift: u32,
}

impl FastTable {
    /// Build from `(lo, hi, id)` triples, which the caller orders by rank.
    pub(crate) fn build(keys: &[(u64, u64, u32)]) -> Self {
        let n = keys.len();
        let mut size = (n * 2).max(2).next_power_of_two();
        // wasm builds trade probe length for half the memory
        if cfg!(target_arch = "wasm32") && size / 2 > n + n / 4 {
            size /= 2;
        }
        let mut slots: Box<[FastSlot]> = (0..size)
            .map(|_| FastSlot {
                lo: AtomicU64::new(0),
                hi: AtomicU64::new(0),
                w0: AtomicU64::new(0),
                w1: AtomicU64::new(0),
            })
            .collect();
        let shift = 64 - size.trailing_zeros();
        let mask = size - 1;
        for &(lo, hi, id) in keys {
            debug_assert!(hi != 0);
            let mut idx = (hash(lo, hi) >> shift) as usize;
            while *slots[idx].hi.get_mut() != 0 {
                idx = (idx + 1) & mask;
            }
            let slot = &mut slots[idx];
            *slot.lo.get_mut() = lo;
            *slot.hi.get_mut() = hi;
            *slot.w0.get_mut() = id as u64 | (1u64 << 32);
        }
        Self { slots, shift }
    }

    #[inline(always)]
    fn home_index(&self, lo: u64, hi: u64) -> usize {
        (hash(lo, hi) >> self.shift) as usize
    }

    /// The id stored in the key's home slot, if the key is there.
    #[inline(always)]
    pub(crate) fn home(&self, lo: u64, hi: u64) -> Option<u32> {
        let slot = &self.slots[self.home_index(lo, hi)];
        let hit = (slot.lo.load(Ordering::Acquire) == lo) & (slot.hi.load(Ordering::Relaxed) == hi);
        let w0 = slot.w0.load(Ordering::Relaxed);
        hit.then_some(w0 as u32)
    }

    /// The key's id, wherever it sits on its probe chain.
    #[inline]
    pub(crate) fn get(&self, lo: u64, hi: u64) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut idx = self.home_index(lo, hi);
        loop {
            let slot = &self.slots[idx];
            let slot_hi = slot.hi.load(Ordering::Relaxed);
            if slot.lo.load(Ordering::Acquire) == lo && slot_hi == hi {
                return Some(slot.w0.load(Ordering::Relaxed) as u32);
            }
            if slot_hi == 0 {
                return None;
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Whether the key sits in its home slot; `None` if absent.
    #[cfg(test)]
    pub(crate) fn is_home(&self, lo: u64, hi: u64) -> Option<bool> {
        self.get(lo, hi)?;
        Some(self.home(lo, hi).is_some())
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }
}
