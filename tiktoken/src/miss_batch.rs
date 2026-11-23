//! Short pieces that miss the shared table, collected so that their merges
//! run together (see `merge_lanes`) instead of one at a time.
//!
//! A sink that meets such a piece records its key and the output position
//! its ids belong at, and goes on with the next piece; the ids written after
//! it are not moved until the batch is resolved. Resolving merges every
//! collected piece, publishes the results in the shared table as the
//! one-at-a-time path does, and splices the ids into place, last piece
//! first. A text without misses never creates a batch.

use crate::encode_loop::Engine;
use crate::merge::{self, lanes};
use crate::piece_cache;
use crate::vocab::{ABSENT, FAST_KEY_MAX, Vocab};

/// Distinct pieces resolved together.
const BATCH: usize = 32;
/// Distinct pieces a batch should average; fewer, and the lanes cannot
/// make up for the bookkeeping.
const FILLED: u32 = 12;
/// Occurrences of them in the output.
const OCCURS: usize = 2 * BATCH;

pub(crate) struct Misses {
    // distinct pieces, by key
    n: usize,
    keys: [(u64, u64); BATCH],
    // occurrences: the piece, and the output position of its ids before
    // the earlier occurrences' ids
    occurs: usize,
    piece: [u8; OCCURS],
    at: [usize; OCCURS],
    // set once misses repeat too soon for batches to fill: text with few
    // distinct misses (code) merges them one at a time again
    off: bool,
    // batches resolved so far, and their distinct pieces
    batches: u32,
    merged: u32,
}

impl Misses {
    fn new() -> Box<Self> {
        Box::new(Self {
            n: 0,
            keys: [(0, 0); BATCH],
            occurs: 0,
            piece: [0; OCCURS],
            at: [0; OCCURS],
            off: false,
            batches: 0,
            merged: 0,
        })
    }

    /// Record a piece whose ids belong after `at` ids; true when the batch
    /// must be resolved before the next piece: it is full, or the piece came
    /// again, as words of code often do a few lines apart, and later copies
    /// should find its result in the shared table instead of waiting too.
    #[inline]
    pub(crate) fn push(&mut self, key: (u64, u64), at: usize) -> bool {
        let seen = self.keys[..self.n].iter().position(|&k| k == key);
        let i = seen.unwrap_or_else(|| {
            self.keys[self.n] = key;
            self.n += 1;
            self.n - 1
        });
        self.piece[self.occurs] = i as u8;
        self.at[self.occurs] = at;
        self.occurs += 1;
        let flush = seen.is_some() || self.n == BATCH || self.occurs == OCCURS;
        if flush {
            self.batches += 1;
            self.merged += self.n as u32;
            self.off = self.batches >= 4 && self.merged < FILLED * self.batches;
        }
        flush
    }

    /// Merge and publish every piece, writing piece `i`'s ids to `ids[i]`.
    fn resolve(&self, engine: &Engine<'_>, ids: &mut [[u32; 16]; BATCH], counts: &mut [u8; BATCH]) {
        let vocab = engine.vocab;
        let n = self.n;
        let mut bytes = [[0u8; 16]; BATCH];
        let mut pieces: [&[u8]; BATCH] = [&[]; BATCH];
        for (b, &(lo, hi)) in bytes.iter_mut().zip(&self.keys[..n]) {
            b[..8].copy_from_slice(&lo.to_le_bytes());
            b[8..].copy_from_slice(&hi.to_le_bytes());
        }
        for (i, b) in bytes[..n].iter().enumerate() {
            pieces[i] = &b[..(self.keys[i].1 >> 56) as usize];
        }
        let pieces = &pieces[..n];
        match vocab.splits() {
            Some(splits) => lanes::merge_by_id(pieces, vocab, splits, ids, counts),
            None => {
                for (i, p) in pieces.iter().enumerate() {
                    counts[i] = merge::merge_small_bytes(p, vocab, &mut ids[i]) as u8;
                    assert!(
                        !ids[i][..counts[i] as usize].contains(&ABSENT),
                        "single byte not in vocab"
                    );
                }
            }
        }
        for i in 0..n {
            let (lo, hi) = self.keys[i];
            let got = &ids[i][..counts[i] as usize];
            if !vocab.insert_derived(lo, hi, got) {
                // full now: keep it where the next occurrence will look
                piece_cache::with_ids(
                    engine.cache_id,
                    pieces[i],
                    |buf| {
                        buf[..got.len()].copy_from_slice(got);
                        got.len()
                    },
                    |_| (),
                );
            }
        }
    }

    /// Resolve the batch into `out`, which holds `len` ids without the
    /// batch's; returns the length with them.
    #[cold]
    #[inline(never)]
    pub(crate) fn flush_into(
        &mut self,
        engine: &Engine<'_>,
        out: &mut Vec<u32>,
        len: usize,
    ) -> usize {
        let mut ids = [[0u32; 16]; BATCH];
        let mut counts = [0u8; BATCH];
        self.resolve(engine, &mut ids, &mut counts);
        let n = self.occurs;
        self.n = 0;
        self.occurs = 0;
        let total: usize = self.piece[..n]
            .iter()
            .map(|&k| counts[k as usize] as usize)
            .sum();
        // SAFETY: `len` ids were written below the capacity
        unsafe { out.set_len(len) };
        out.reserve(total);
        let p = out.as_mut_ptr();
        // last piece first: the ids after piece i move right by the counts
        // of pieces 0..=i
        let mut end = len;
        let mut shift = total;
        for i in (0..n).rev() {
            let at = self.at[i];
            let k = self.piece[i] as usize;
            let c = counts[k] as usize;
            // SAFETY: every range lies below len + total, within the
            // capacity reserved above; `copy` allows the overlap
            unsafe {
                std::ptr::copy(p.add(at), p.add(at + shift), end - at);
                shift -= c;
                std::ptr::copy_nonoverlapping(ids[k].as_ptr(), p.add(at + shift), c);
            }
            end = at;
        }
        // SAFETY: the ids up to len + total are written
        unsafe { out.set_len(len + total) };
        len + total
    }

    /// Resolve the batch; returns its id count.
    #[cold]
    #[inline(never)]
    pub(crate) fn flush_count(&mut self, engine: &Engine<'_>) -> usize {
        let mut ids = [[0u32; 16]; BATCH];
        let mut counts = [0u8; BATCH];
        self.resolve(engine, &mut ids, &mut counts);
        let n = self.occurs;
        self.n = 0;
        self.occurs = 0;
        self.piece[..n]
            .iter()
            .map(|&k| counts[k as usize] as usize)
            .sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.occurs == 0
    }
}

/// A sink's batch of misses, created once a text has shown a few.
pub(crate) struct Batch {
    seen: u32,
    misses: Option<Box<Misses>>,
}

/// Misses of a text merged one at a time before batching starts: short
/// texts with a miss or two do not pay for a batch.
const EAGER: u32 = 8;

impl Batch {
    pub(crate) const fn new() -> Self {
        Self {
            seen: 0,
            misses: None,
        }
    }

    /// The batch to leave a short piece the shared table lacks in, or
    /// `None` to merge it now.
    #[inline]
    pub(crate) fn take(&mut self, vocab: &Vocab, piece: &[u8]) -> Option<&mut Misses> {
        self.seen += 1;
        if self.seen <= EAGER || !batches(vocab, piece) {
            return None;
        }
        let m = self.misses.get_or_insert_with(Misses::new);
        (!m.off).then_some(&mut **m)
    }

    /// The batch, if pieces wait in it.
    #[inline]
    pub(crate) fn pending(&mut self) -> Option<&mut Misses> {
        self.misses.as_deref_mut().filter(|m| !m.is_empty())
    }
}

/// Whether a short piece that the shared table lacks can wait for a batch:
/// a vocabulary that merges by id, a shared table that still takes its
/// result, and a piece the lanes take.
#[inline]
pub(crate) fn batches(vocab: &Vocab, piece: &[u8]) -> bool {
    cfg!(not(target_arch = "wasm32"))
        && vocab.merges_by_id()
        && vocab.accepts_derived()
        && (2..=lanes::LANE_MAX.min(FAST_KEY_MAX)).contains(&piece.len())
        && !merge::mostly_multibyte(piece)
}

#[cfg(test)]
#[path = "miss_batch_tests.rs"]
mod tests;
