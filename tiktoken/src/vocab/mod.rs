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

mod atoms;
mod decoder;
mod fast;
mod filter;
mod frags;
mod key;
mod long;
mod long_derived;
mod splits;
mod walls;

pub(crate) use atoms::{Atoms, char_len};
pub(crate) use fast::{Entry, Home, HomeProbe, copy_ids};
use fast::{FastTable, TokenProbe};
use filter::{FilterProbe, KeyFilter};
pub(crate) use key::{FAST_KEY_MAX, KEY_READ, key_in, key_of, key_padded};
use long::LongTable;
use long_derived::LongDerived;
pub(crate) use long_derived::{LONG_KEY_MAX, LongIds};
pub(crate) use splits::{NO_ID, Splits};
use std::sync::OnceLock;
pub(crate) use walls::{Start, cp_at};

/// Rank sentinel for "absent" in the direct-indexed tables.
pub(crate) const ABSENT: u32 = u32::MAX;

/// The rank lookups of a merge, with the tables' fields held by value so
/// that they stay in registers across the merge's rounds.
#[derive(Clone, Copy)]
pub(crate) struct RankProbe<'a> {
    vocab: &'a Vocab,
    filter: FilterProbe<'a>,
    tokens: TokenProbe<'a>,
    pair: &'a [u32; 1 << 16],
    // opaque, so that the multiply takes it from a register rather than
    // building the constant for every key
    mul: u64,
}

impl RankProbe<'_> {
    /// Same as [`Vocab::rank_padded`].
    #[inline(always)]
    pub(crate) fn rank_padded(self, buf: &[u8], at: usize, len: usize) -> u32 {
        if len > FAST_KEY_MAX {
            return self.vocab.rank_long(&buf[at..at + len]);
        }
        self.rank_key(key_padded(buf, at, len))
    }

    /// Rank of `t` (2 or more bytes) or [`ABSENT`], reading no byte past
    /// its end.
    #[inline(always)]
    pub(crate) fn rank_of(self, t: &[u8]) -> u32 {
        match t.len() {
            2 => self.rank_pair(t[0], t[1]),
            len if len <= FAST_KEY_MAX => self.rank_key(key_of(t)),
            _ => self.vocab.rank_long(t),
        }
    }

    /// Same as [`Vocab::rank_pair`].
    #[inline(always)]
    pub(crate) fn rank_pair(self, a: u8, b: u8) -> u32 {
        self.pair[(a as usize) << 8 | b as usize]
    }

    #[inline(always)]
    fn rank_key(self, (lo, hi): (u64, u64)) -> u32 {
        let h = key::hash_with(lo, hi, self.mul);
        if !self.filter.may_contain(h) {
            return ABSENT;
        }
        self.tokens.get(lo, hi, h).unwrap_or(ABSENT)
    }
}

/// See [`Vocab::long_piece`].
pub(crate) struct LongPiece<'p> {
    piece: &'p [u8],
    hash: u64,
}

pub struct Vocab {
    // all token bytes, contiguous; referenced by `decoder` and `long`
    arena: Box<[u8]>,
    // rank by byte, for 1-byte keys; ABSENT if missing
    single: Box<[u32; 256]>,
    // every byte is a token, so a merge result never holds ABSENT
    every_byte: bool,
    // rank by (first byte << 8 | second byte), for 2-byte keys
    pair: Box<[u32]>,
    fast: FastTable,
    // in front of the merge's substring probes into `fast`
    filter: KeyFilter,
    long: LongTable,
    // merge results of longer pieces, created on first use
    long_derived: OnceLock<LongDerived>,
    // chars a merge may start from whole, built on first use
    atoms: OnceLock<Atoms>,
    // merge results by token id, built on the first batch of misses
    splits: OnceLock<Option<Splits>>,
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

        // the per-token buffers go before the tables are allocated, so that a
        // wasm heap, which never shrinks, can reuse their space
        drop(entries);
        let arena = arena.into_boxed_slice();
        let long = LongTable::build(&arena, &long_items);
        let filter = KeyFilter::build(fast_keys.iter().map(|&(lo, hi, _)| key::hash(lo, hi)));
        Self {
            fast: FastTable::build(&fast_keys),
            filter,
            long,
            long_derived: OnceLock::new(),
            atoms: OnceLock::new(),
            splits: OnceLock::new(),
            arena,
            every_byte: !single.contains(&ABSENT),
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

    /// [`get_home`](Self::get_home) as a value to keep in registers.
    #[inline(always)]
    pub(crate) fn home_probe(&self) -> HomeProbe<'_> {
        self.fast.home_probe()
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

    /// The merge's rank lookups.
    #[inline(always)]
    pub(crate) fn rank_probe(&self) -> RankProbe<'_> {
        RankProbe {
            vocab: self,
            filter: self.filter.probe(),
            tokens: self.fast.token_probe(),
            pair: self.pair.as_ref().try_into().unwrap(),
            mul: std::hint::black_box(key::HASH_MUL),
        }
    }

    /// Rank of a key longer than [`FAST_KEY_MAX`] bytes, or [`ABSENT`].
    #[inline(never)]
    fn rank_long(&self, t: &[u8]) -> u32 {
        self.long.get(&self.arena, t).unwrap_or(ABSENT)
    }

    /// Entry of a piece's packed key: a token's id or the ids of a derived
    /// entry. The key must come from utf-8 text.
    #[inline]
    pub(crate) fn entry(&self, lo: u64, hi: u64) -> Option<Entry> {
        self.fast.entry(lo, hi)
    }

    /// Write an entry's ids at `out`, and up to two arbitrary words after
    /// them.
    ///
    /// # Safety
    ///
    /// `out` has room for `max(3, e.len())` ids.
    #[inline(always)]
    pub(crate) unsafe fn write_entry(&self, e: Entry, out: *mut u32) {
        // SAFETY: passed on from the caller
        unsafe { self.fast.write_ids(e, out) }
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

    /// The shared merge result of a piece longer than [`FAST_KEY_MAX`].
    #[inline]
    pub(crate) fn long_ids(&self, piece: &[u8]) -> Option<LongIds<'_>> {
        self.long_derived.get()?.get(piece)
    }

    /// A piece of 16 to [`LONG_KEY_MAX`] bytes, hashed once for the lookups
    /// and the insert of its first occurrence.
    #[inline]
    pub(crate) fn long_piece<'p>(&self, piece: &'p [u8]) -> LongPiece<'p> {
        LongPiece {
            piece,
            hash: key::hash_long(piece),
        }
    }

    /// [`long_ids`](Self::long_ids) of a hashed piece.
    #[inline]
    pub(crate) fn long_ids_of(&self, p: &LongPiece<'_>) -> Option<LongIds<'_>> {
        self.long_derived.get()?.get_hashed(p.piece, p.hash)
    }

    /// The id of a hashed piece if it is a vocabulary token.
    #[inline]
    pub(crate) fn long_token(&self, p: &LongPiece<'_>) -> Option<u32> {
        self.long.get_hashed(&self.arena, p.piece, p.hash)
    }

    /// The chars a merge may start from whole.
    pub(crate) fn atoms(&self) -> &Atoms {
        self.atoms.get_or_init(|| {
            let tokens =
                (0..self.decoder.len() as u32).filter_map(|r| self.try_decode(r).map(|t| (r, t)));
            Atoms::build(tokens, |t| self.get(t))
        })
    }

    /// The merge results by token id, or `None` for a vocabulary they do
    /// not fit (see [`Splits::build`]).
    #[inline]
    pub(crate) fn splits(&self) -> Option<&Splits> {
        self.splits.get_or_init(|| Splits::build(self)).as_ref()
    }

    /// Whether merges may go by token id: every byte is a token and every id
    /// fits [`Splits`].
    #[inline]
    pub(crate) fn merges_by_id(&self) -> bool {
        self.every_byte && self.decoder.len() < NO_ID as usize
    }

    /// Whether the shared table still takes derived entries.
    #[inline]
    pub(crate) fn accepts_derived(&self) -> bool {
        self.fast.accepts_derived()
    }

    /// Drop all derived entries. Caller guarantees no concurrent use.
    pub(crate) fn clear_derived(&self) {
        self.fast.clear_derived();
        if let Some(t) = self.long_derived.get() {
            t.clear();
        }
    }

    /// Whether every single byte is a token.
    #[inline(always)]
    pub(crate) fn has_every_byte(&self) -> bool {
        self.every_byte
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

    /// One past the highest rank.
    #[cfg(test)]
    pub(crate) fn decoder_len(&self) -> usize {
        self.decoder.len()
    }

    /// Number of entries in the vocabulary.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.decoder
            .iter()
            .filter(|&&(offset, _)| offset != u32::MAX)
            .count()
    }
}

#[cfg(test)]
mod tests;
