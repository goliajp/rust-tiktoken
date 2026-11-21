//! Vocabulary storage tuned for the encode loop and the BPE merge.
//!
//! - **up to 15 bytes** — [`FastTable`]: a packed two-word key, one hash, and
//!   in the common case one 32-byte slot read. The encode loop probes only
//!   the home slot inline and leaves the rest of the chain to [`Vocab::get`].
//! - **longer keys** — [`LongTable`]: arena references with a hash tag.
//! - **1- and 2-byte keys** — direct-indexed rank tables as well, for the
//!   merge's initial adjacent-pair scan, which needs no hash at all.
//!
//! Decoding indexes a contiguous arena by rank.

mod fast;
mod filter;
mod key;
mod long;

pub(crate) use fast::Entry;
use fast::FastTable;
use filter::KeyFilter;
pub(crate) use key::{FAST_KEY_MAX, KEY_READ, key_in, key_of, key_padded};
use long::LongTable;

/// Rank sentinel for "absent" in the direct-indexed tables.
pub(crate) const ABSENT: u32 = u32::MAX;

pub struct Vocab {
    // all token bytes, contiguous; referenced by `decoder` and `long`
    arena: Box<[u8]>,
    // rank by byte, for 1-byte keys; ABSENT if missing
    single: Box<[u32; 256]>,
    // rank by (first byte << 8 | second byte), for 2-byte keys
    pair: Box<[u32]>,
    fast: FastTable,
    // in front of the merge's substring probes into `fast`
    filter: KeyFilter,
    long: LongTable,
    // indexed by rank: (offset, len) into the arena, for decode
    decoder: Box<[(u32, u16)]>,
}

impl Vocab {
    /// Build a vocabulary from (token_bytes, rank) pairs.
    pub(crate) fn from_entries(mut entries: Vec<(Vec<u8>, u32)>) -> Self {
        // ascending rank: frequent tokens claim their home slot first
        entries.sort_unstable_by_key(|&(_, rank)| rank);
        let max_rank = entries.last().map_or(0, |&(_, r)| r);

        let total_bytes: usize = entries.iter().map(|(k, _)| k.len()).sum();
        let mut arena = Vec::with_capacity(total_bytes);
        // u32::MAX offset = sentinel for "rank not present in vocabulary"
        let mut decoder = vec![(u32::MAX, 0u16); max_rank as usize + 1];
        let mut single = Box::new([ABSENT; 256]);
        let mut pair = vec![ABSENT; 1 << 16].into_boxed_slice();
        let mut fast_keys = Vec::with_capacity(entries.len());
        let mut long_items = Vec::new();

        for (token, rank) in &entries {
            debug_assert!(
                arena.len() <= u32::MAX as usize,
                "arena offset overflow: {} bytes exceeds u32 range",
                arena.len()
            );
            debug_assert!(
                token.len() <= u16::MAX as usize,
                "token length {} exceeds u16 range",
                token.len()
            );
            let offset = arena.len() as u32;
            arena.extend_from_slice(token);
            debug_assert!(
                decoder[*rank as usize].0 == u32::MAX,
                "duplicate rank {rank} in vocabulary entries"
            );
            decoder[*rank as usize] = (offset, token.len() as u16);
            match token.len() {
                0 => continue,
                1 => single[token[0] as usize] = *rank,
                2 => pair[(token[0] as usize) << 8 | token[1] as usize] = *rank,
                _ => {}
            }
            if token.len() <= FAST_KEY_MAX {
                let (lo, hi) = key_of(token);
                fast_keys.push((lo, hi, *rank));
            } else {
                long_items.push((*rank, offset, token.len() as u32));
            }
        }

        let arena = arena.into_boxed_slice();
        let long = LongTable::build(&arena, &long_items);
        let filter = KeyFilter::build(fast_keys.iter().map(|&(lo, hi, _)| key::hash(lo, hi)));
        Self {
            fast: FastTable::build(&fast_keys),
            filter,
            long,
            arena,
            single,
            pair,
            decoder: decoder.into_boxed_slice(),
        }
    }

    /// Look up the rank for a token byte sequence.
    #[inline]
    pub(crate) fn get(&self, token: &[u8]) -> Option<u32> {
        let rank = match token.len() {
            0 => return None,
            1 => self.single[token[0] as usize],
            2 => self.pair[(token[0] as usize) << 8 | token[1] as usize],
            len if len <= FAST_KEY_MAX => {
                let (lo, hi) = key_of(token);
                return self.fast.get(lo, hi);
            }
            _ => return self.long.get(&self.arena, token),
        };
        (rank != ABSENT).then_some(rank)
    }

    /// Rank of the packed key, looking only at its home slot. `None` does not
    /// mean absent: the caller falls back to [`get`](Self::get).
    #[inline(always)]
    pub(crate) fn get_home(&self, lo: u64, hi: u64) -> Option<u32> {
        self.fast.home(lo, hi)
    }

    /// Rank of `buf[at..at + len]` or [`ABSENT`], where `buf` stays readable
    /// for [`KEY_READ`] bytes from `at` (the merge pads its scratch copy).
    #[inline]
    pub(crate) fn rank_padded(&self, buf: &[u8], at: usize, len: usize) -> u32 {
        if len <= FAST_KEY_MAX {
            let (lo, hi) = key_padded(buf, at, len);
            let h = key::hash(lo, hi);
            if !self.filter.may_contain(h) {
                return ABSENT;
            }
            self.fast.get_hashed(lo, hi, h).unwrap_or(ABSENT)
        } else {
            self.long
                .get(&self.arena, &buf[at..at + len])
                .unwrap_or(ABSENT)
        }
    }

    /// Entry of a piece's packed key: a token's id or the ids of a derived
    /// entry. The key must come from utf-8 text.
    #[inline]
    pub(crate) fn entry(&self, lo: u64, hi: u64) -> Option<Entry> {
        self.fast.entry(lo, hi)
    }

    /// Append an entry's ids to `out`.
    #[inline]
    pub(crate) fn extend_entry(&self, e: Entry, out: &mut Vec<u32>) {
        self.fast.extend(e, out);
    }

    /// Share the merge result of a piece that is not a token with every
    /// thread; false when the shared table is full.
    #[inline]
    pub(crate) fn insert_derived(&self, lo: u64, hi: u64, ids: &[u32]) -> bool {
        self.fast.insert_derived(lo, hi, ids)
    }

    /// Whether the shared table still takes derived entries.
    #[inline]
    pub(crate) fn accepts_derived(&self) -> bool {
        self.fast.accepts_derived()
    }

    /// Drop all derived entries. Caller guarantees no concurrent use.
    pub(crate) fn clear_derived(&self) {
        self.fast.clear_derived();
    }

    /// Rank of a 1-byte key, or [`ABSENT`].
    #[inline(always)]
    pub(crate) fn rank_single(&self, b: u8) -> u32 {
        self.single[b as usize]
    }

    /// Look up the rank of a 2-byte key without slicing: the merge loop's
    /// initial adjacent-pair scan, which is the single hottest lookup class.
    #[inline]
    pub(crate) fn rank_pair(&self, a: u8, b: u8) -> u32 {
        self.pair[(a as usize) << 8 | b as usize]
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn get_pair(&self, a: u8, b: u8) -> Option<u32> {
        let rank = self.pair[(a as usize) << 8 | b as usize];
        (rank != ABSENT).then_some(rank)
    }

    /// Check if a token byte sequence exists in the vocabulary.
    #[inline]
    #[cfg(test)]
    pub(crate) fn contains_key(&self, token: &[u8]) -> bool {
        self.get(token).is_some()
    }

    /// Get the token bytes for a given rank.
    ///
    /// Panics if rank is out of range.
    #[inline]
    #[cfg(test)]
    fn decode(&self, rank: u32) -> &[u8] {
        let (offset, len) = self.decoder[rank as usize];
        &self.arena[offset as usize..(offset as usize + len as usize)]
    }

    /// Number of entries in the vocabulary.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.decoder
            .iter()
            .filter(|&&(offset, _)| offset != u32::MAX)
            .count()
    }

    /// Try to get the token bytes for a given rank.
    /// Returns `None` if the rank is out of range or not present in the vocabulary.
    #[inline]
    pub(crate) fn try_decode(&self, rank: u32) -> Option<&[u8]> {
        let idx = rank as usize;
        if idx >= self.decoder.len() {
            return None;
        }
        let (offset, len) = self.decoder[idx];
        // u32::MAX offset = sentinel for "rank not present"
        if offset == u32::MAX {
            return None;
        }
        Some(&self.arena[offset as usize..(offset as usize + len as usize)])
    }
}

#[cfg(test)]
mod tests;
