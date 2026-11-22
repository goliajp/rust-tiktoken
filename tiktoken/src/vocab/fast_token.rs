//! Vocabulary-token lookups for the merge, by value.

use std::sync::atomic::Ordering;

use super::{FastLine, FastTable, line_of, next_line};

/// Vocabulary-token lookups of a [`FastTable`] by value, for the merge's
/// rank lookups: the fields stay in registers across a merge.
#[derive(Clone, Copy)]
pub(crate) struct TokenProbe<'a> {
    lines: &'a [FastLine],
    shift: u32,
}

impl TokenProbe<'_> {
    /// The key's id if it is a vocabulary token, wherever it sits on its
    /// probe chain; `h` is the key's hash.
    #[inline(always)]
    pub(crate) fn get(self, lo: u64, hi: u64, h: u64) -> Option<u32> {
        let size = self.lines.len();
        let mut idx = line_of(h, self.shift, size);
        loop {
            // SAFETY: `line_of` and `next_line` stay below the line count
            let [a, b] = &unsafe { self.lines.get_unchecked(idx) }.0;
            // relaxed loads suffice: a token's slot never changes, a
            // derived entry's `hi` carries MULTI and never equals a token's,
            // and a slot claimed after the build lies past the end of every
            // token's chain
            for slot in [a, b] {
                let h = slot.hi.load(Ordering::Relaxed);
                if h == hi && slot.lo.load(Ordering::Relaxed) == lo {
                    return Some(slot.id());
                }
                if h == 0 {
                    return None;
                }
            }
            idx = next_line(idx, size);
        }
    }
}

impl FastTable {
    #[inline(always)]
    pub(crate) fn token_probe(&self) -> TokenProbe<'_> {
        TokenProbe {
            lines: &self.lines,
            shift: self.shift,
        }
    }
}
