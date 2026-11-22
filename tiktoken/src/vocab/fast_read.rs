//! Reading the ids of a [`FastTable`] entry.

use std::sync::atomic::Ordering;

use super::{Entry, FastTable, INLINE};

/// Copy `n >= 1` ids from `src` to `dst` with a few overlapping block moves
/// rather than a loop of one id at a time, whose exit mispredicts on every
/// piece. Reads exactly `src[..n]` and writes exactly `dst[..n]`.
///
/// # Safety
///
/// `src[..n]` is readable and not written concurrently, and `dst` has room
/// for `n` ids.
#[inline(always)]
pub(crate) unsafe fn copy_ids(src: *const u32, dst: *mut u32, n: usize) {
    debug_assert!(n >= 1);
    // SAFETY: every block lies in src[..n] and dst[..n]
    unsafe {
        if n >= 8 {
            let mut k = 0;
            while k + 8 < n {
                let v = src.add(k).cast::<[u32; 8]>().read_unaligned();
                dst.add(k).cast::<[u32; 8]>().write_unaligned(v);
                k += 8;
            }
            let v = src.add(n - 8).cast::<[u32; 8]>().read_unaligned();
            dst.add(n - 8).cast::<[u32; 8]>().write_unaligned(v);
        } else if n >= 4 {
            let a = src.cast::<[u32; 4]>().read_unaligned();
            let b = src.add(n - 4).cast::<[u32; 4]>().read_unaligned();
            dst.cast::<[u32; 4]>().write_unaligned(a);
            dst.add(n - 4).cast::<[u32; 4]>().write_unaligned(b);
        } else {
            let a = src.read();
            let b = src.add(n / 2).read();
            let c = src.add(n - 1).read();
            dst.write(a);
            dst.add(n / 2).write(b);
            dst.add(n - 1).write(c);
        }
    }
}

impl Entry {
    #[inline(always)]
    fn inline_ids(&self) -> [u32; 3] {
        [self.w0 as u32, self.w1 as u32, (self.w1 >> 32) as u32]
    }
}

impl FastTable {
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
                let ids = &self.arena[off..off + n];
                // the slot's Acquire load of lo made these writes visible,
                // and published ids are not written again until a clear,
                // which excludes readers
                copy_ids(ids.as_ptr().cast::<u32>(), out, n);
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
