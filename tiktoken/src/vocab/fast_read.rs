//! Reading the ids of a [`FastTable`] entry.

use std::sync::atomic::Ordering;

use super::{Entry, FastTable, INLINE, MULTI, OPEN_LO};

impl Entry {
    #[inline(always)]
    fn inline_ids(&self) -> [u32; 3] {
        [self.w0 as u32, self.w1 as u32, (self.w1 >> 32) as u32]
    }
}

impl FastTable {
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

    /// Append the entry's ids to `out`.
    #[inline]
    pub(crate) fn extend(&self, e: Entry, out: &mut Vec<u32>) {
        let n = e.len();
        if n <= INLINE {
            out.extend_from_slice(&e.inline_ids()[..n]);
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
}
