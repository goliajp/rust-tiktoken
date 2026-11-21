//! Open-addressed table from packed keys to token ids, shared by all threads.
//!
//! Every vocabulary token of up to 15 bytes has a slot. Slots come in pairs
//! that share a 64-byte cache line; a key hashes to a line and takes the
//! first free slot from there on. Tokens are inserted in ascending rank, so
//! the frequent low-rank tokens land in their home line and the encode loop
//! answers most pieces by comparing the two slots of one line; anything else
//! goes through [`FastTable::get`] or [`FastTable::entry`], which walk the
//! probe chain.
//!
//! Derived entries: a piece that is not a token but merges into 2..=15 ids
//! is published into a free slot by whichever thread merges it first, up to
//! `derived_cap` entries. Their stored `hi` carries [`MULTI`], so vocabulary
//! lookups (exact `hi` compare) never see them. Up to three ids sit in the
//! slot; more go to a shared append-only arena.
//!
//! Publication: a writer claims an empty slot by swapping `hi` from 0 to
//! `BUSY_HI`, writes the ids and the real `hi`, and stores `lo` last with
//! `Release`. Readers load `lo` with `Acquire` and match it first. An
//! unpublished slot still holds `OPEN_LO` in `lo`, which no query has (queries
//! come from utf-8 text, which never contains 0xff), so a reader cannot match
//! it before the rest of the slot is visible. Derived entries only fill slots
//! that were empty after the vocabulary was built, so they never sit between a
//! vocabulary key and its home line, and a reader that stops at a slot still
//! being claimed loses nothing but a derived hit.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::key::hash;

const OPEN_LO: u64 = u64::MAX;
// a real hi has a length of at most 15 in its top byte, plus at most MULTI
const BUSY_HI: u64 = u64::MAX;
/// Set in the stored `hi` of a derived entry.
const MULTI: u64 = 1 << 63;
// w0 layout: first id (or arena offset) | id count << 32 | flags
const COUNT_SHIFT: u32 = 32;
const DERIVED: u64 = 1 << 48;
const INLINE: usize = 3;
/// Most ids a derived entry may hold (a 15-byte piece merges into at most 15).
const DERIVED_MAX_IDS: usize = 15;

#[repr(C)]
struct FastSlot {
    lo: AtomicU64,
    // zero only in an empty slot
    hi: AtomicU64,
    w0: AtomicU64,
    // second and third ids of a derived entry with up to three ids
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

/// The ids of a table entry, as read from its slot.
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    w0: u64,
    w1: u64,
}

impl Entry {
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        ((self.w0 >> COUNT_SHIFT) & 0xFF) as usize
    }

    #[inline(always)]
    fn inline_ids(&self) -> [u32; 3] {
        [self.w0 as u32, self.w1 as u32, (self.w1 >> 32) as u32]
    }
}

pub(crate) struct FastTable {
    lines: Box<[FastLine]>,
    shift: u32,
    derived: AtomicU32,
    derived_cap: u32,
    arena: Box<[AtomicU32]>,
    arena_len: AtomicU32,
}

fn zeroed_atomics(n: usize) -> Box<[AtomicU32]> {
    let words = vec![0u32; n].into_boxed_slice();
    // SAFETY: AtomicU32 has the same size, alignment and bit validity as u32
    unsafe { Box::from_raw(Box::into_raw(words) as *mut [AtomicU32]) }
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
            lo: AtomicU64::new(OPEN_LO),
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
            *slot.w0.get_mut() = id as u64 | (1u64 << COUNT_SHIFT);
        }
        // stop at three quarters full so probe chains stay short
        let derived_cap = (slots / 4).min((slots * 3 / 4).saturating_sub(n)) as u32;
        Self {
            lines,
            shift,
            derived: AtomicU32::new(0),
            derived_cap,
            arena: zeroed_atomics(derived_cap as usize * 4),
            arena_len: AtomicU32::new(0),
        }
    }

    #[inline(always)]
    fn home_index(&self, lo: u64, hi: u64) -> usize {
        (hash(lo, hi) >> self.shift) as usize
    }

    /// The id stored in the key's home line, if the key is a vocabulary
    /// token sitting there.
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

    /// The entry in the key's home line, vocabulary token or derived. The
    /// key must come from utf-8 text.
    #[inline(always)]
    pub(crate) fn home_entry(&self, lo: u64, hi: u64) -> Option<Entry> {
        debug_assert!(lo != OPEN_LO);
        for slot in &self.lines[self.home_index(lo, hi)].0 {
            if slot.lo.load(Ordering::Acquire) == lo
                && slot.hi.load(Ordering::Relaxed) & !MULTI == hi
            {
                let e = Entry {
                    w0: slot.w0.load(Ordering::Relaxed),
                    w1: slot.w1.load(Ordering::Relaxed),
                };
                return Some(e);
            }
        }
        None
    }

    /// Write the entry's ids at `out`, and up to two arbitrary words after
    /// them.
    ///
    /// # Safety
    ///
    /// `out` has room for `max(3, e.len())` ids.
    #[inline(always)]
    pub(crate) unsafe fn write_ids(&self, e: Entry, out: *mut u32) {
        let n = e.len();
        // SAFETY: the caller guarantees the room
        unsafe {
            if n <= INLINE {
                out.cast::<[u32; 3]>().write_unaligned(e.inline_ids());
            } else {
                let off = e.w0 as u32 as usize;
                // the slot's Acquire load of lo made these writes visible
                for (k, w) in self.arena[off..off + n].iter().enumerate() {
                    out.add(k).write(w.load(Ordering::Relaxed));
                }
            }
        }
    }

    /// The key's id if it is a vocabulary token, wherever it sits on its
    /// probe chain.
    #[inline]
    pub(crate) fn get(&self, lo: u64, hi: u64) -> Option<u32> {
        self.get_hashed(lo, hi, hash(lo, hi))
    }

    /// [`get`](Self::get) with the key's hash already computed.
    #[inline]
    pub(crate) fn get_hashed(&self, lo: u64, hi: u64, h: u64) -> Option<u32> {
        let mask = self.lines.len() - 1;
        let mut idx = (h >> self.shift) as usize;
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

    /// The key's entry, vocabulary token or derived. The key must come from
    /// utf-8 text.
    #[inline]
    pub(crate) fn entry(&self, lo: u64, hi: u64) -> Option<Entry> {
        debug_assert!(lo != OPEN_LO);
        let mask = self.lines.len() - 1;
        let mut idx = self.home_index(lo, hi);
        loop {
            for slot in &self.lines[idx].0 {
                if slot.lo.load(Ordering::Acquire) == lo {
                    let h = slot.hi.load(Ordering::Relaxed);
                    if h & !MULTI == hi {
                        return Some(Entry {
                            w0: slot.w0.load(Ordering::Relaxed),
                            w1: slot.w1.load(Ordering::Relaxed),
                        });
                    }
                }
                if slot.hi.load(Ordering::Relaxed) == 0 {
                    return None;
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Append the entry's ids to `out`.
    #[inline]
    pub(crate) fn extend(&self, e: Entry, out: &mut Vec<u32>) {
        let n = e.len();
        if n <= INLINE {
            let ids = [e.w0 as u32, e.w1 as u32, (e.w1 >> 32) as u32];
            out.extend_from_slice(&ids[..n]);
        } else {
            let off = e.w0 as u32 as usize;
            // the slot's Acquire load of lo made these writes visible
            out.extend(
                self.arena[off..off + n]
                    .iter()
                    .map(|w| w.load(Ordering::Relaxed)),
            );
        }
    }

    /// Publish a derived entry for a piece that is not a token. Returns false
    /// when the table no longer accepts entries; the caller then keeps the
    /// result elsewhere. Two threads publishing the same key may both succeed;
    /// the duplicate is unreachable and harmless.
    pub(crate) fn insert_derived(&self, lo: u64, hi: u64, ids: &[u32]) -> bool {
        let n = ids.len();
        debug_assert!((2..=DERIVED_MAX_IDS).contains(&n) && lo != OPEN_LO && hi & MULTI == 0);
        if self.derived.load(Ordering::Relaxed) >= self.derived_cap {
            return false;
        }
        let (w0, w1) = if n <= INLINE {
            let w1 = ids[1] as u64 | ids.get(2).map_or(0, |&id| (id as u64) << 32);
            (ids[0] as u64, w1)
        } else {
            let off = self.arena_len.fetch_add(n as u32, Ordering::Relaxed) as usize;
            if off + n > self.arena.len() {
                return false;
            }
            for (w, &id) in self.arena[off..off + n].iter().zip(ids) {
                w.store(id, Ordering::Relaxed);
            }
            (off as u64, 0)
        };
        let w0 = w0 | ((n as u64) << COUNT_SHIFT) | DERIVED;
        let mask = self.lines.len() - 1;
        let mut idx = self.home_index(lo, hi);
        loop {
            for slot in &self.lines[idx].0 {
                let h = slot.hi.load(Ordering::Relaxed);
                if h == 0 {
                    if slot
                        .hi
                        .compare_exchange(0, BUSY_HI, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                    {
                        slot.w0.store(w0, Ordering::Relaxed);
                        slot.w1.store(w1, Ordering::Relaxed);
                        slot.hi.store(hi | MULTI, Ordering::Relaxed);
                        slot.lo.store(lo, Ordering::Release);
                        self.derived.fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                } else if h == hi | MULTI && slot.lo.load(Ordering::Acquire) == lo {
                    return true;
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Whether [`insert_derived`](Self::insert_derived) may still succeed.
    #[inline]
    pub(crate) fn accepts_derived(&self) -> bool {
        self.derived.load(Ordering::Relaxed) < self.derived_cap
    }

    /// Remove every derived entry, restoring the table as built.
    ///
    /// Caller guarantees no other thread uses the table meanwhile.
    pub(crate) fn clear_derived(&self) {
        for line in self.lines.iter() {
            for slot in &line.0 {
                if slot.w0.load(Ordering::Relaxed) & DERIVED != 0 {
                    slot.lo.store(OPEN_LO, Ordering::Relaxed);
                    slot.hi.store(0, Ordering::Relaxed);
                    slot.w0.store(0, Ordering::Relaxed);
                    slot.w1.store(0, Ordering::Relaxed);
                }
            }
        }
        self.derived.store(0, Ordering::Relaxed);
        self.arena_len.store(0, Ordering::Relaxed);
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

    #[cfg(test)]
    pub(crate) fn derived_cap(&self) -> u32 {
        self.derived_cap
    }
}
