//! Vocabulary storage tuned for the BPE merge's access pattern.
//!
//! The merge loop is lookup-bound: on Unicode-dense input it performs ~2.4
//! vocabulary probes per emitted token, 77% of them misses, and 96.7% of the
//! keys are 8 bytes or shorter (59% are exactly 2 — the initial adjacent-pair
//! scan). See `PERF-2026-08-08-unicode-decomposition.md`. The layout serves
//! those classes directly:
//!
//! - **2-byte keys** — a direct-indexed table of 65,536 ranks. No hash, no
//!   probe, no byte comparison; one access into 256 KB.
//! - **1-byte keys** — a direct-indexed table of 256 ranks.
//! - **3–8-byte keys** — open addressing with the key bytes inlined into the
//!   16-byte slot. A probe is one memory access and one `u64` compare; a miss
//!   terminates at the first empty slot without ever touching the arena.
//! - **longer keys** — open addressing with an arena reference plus an 8-bit
//!   hash tag in the slot, so a mismatched slot is rejected without loading
//!   the arena bytes.
//!
//! Decoding is unchanged: direct indexing by rank into a contiguous arena.

use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// Longest key stored inline in a slot. Above this, the slot holds an arena
/// offset and a hash tag instead of the bytes themselves.
const INLINE_MAX: usize = 8;

/// Rank sentinel for "absent" in the direct-indexed tables.
const ABSENT: u32 = u32::MAX;

pub struct Vocab {
    // all token bytes, contiguous; referenced by `decoder` and by spill slots
    arena: Box<[u8]>,
    // rank by first byte, for 1-byte keys; ABSENT if missing
    single: Box<[u32]>,
    // rank by (first byte << 8 | second byte), for 2-byte keys
    pair: Box<[u32]>,
    // open addressing, linear probing, for keys of 3+ bytes
    table: Box<[Slot]>,
    mask: usize,
    // indexed by rank: (offset, len) into the arena, for decode
    decoder: Box<[(u32, u16)]>,
}

/// One 16-byte slot.
///
/// `len == 0` marks an empty slot. For `3..=INLINE_MAX`, `key` holds the token
/// bytes little-endian, zero-padded — equality is a single integer compare.
/// For longer keys, `key`'s low 32 bits hold the arena offset and bits 32..40
/// an 8-bit tag from the unused high hash bits; the tag rejects most
/// mismatched slots without an arena load.
#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    key: u64,
    rank: u32,
    len: u32,
}

const EMPTY: Slot = Slot {
    key: 0,
    rank: 0,
    len: 0,
};

/// Load 3..=8 bytes as a zero-padded little-endian u64 with two overlapping
/// word reads instead of a variable-length copy.
#[inline]
fn load_inline_key(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    debug_assert!((3..=INLINE_MAX).contains(&len));
    if len >= 4 {
        let lo = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64;
        let hi = u32::from_le_bytes(bytes[len - 4..].try_into().unwrap()) as u64;
        // the reads overlap on 8 - len bytes; identical bits, so OR is exact
        lo | (hi << ((len - 4) * 8))
    } else {
        (bytes[0] as u64) | ((bytes[1] as u64) << 8) | ((bytes[2] as u64) << 16)
    }
}

/// Hash for inline keys: multiply–xorshift over the padded key with the
/// length folded in (tokens may contain NUL bytes, so "ab" and "ab\0" share a
/// padded key and must not share a bucket chain shape).
#[inline]
fn hash_inline(key: u64, len: usize) -> u64 {
    let x = key.wrapping_add((len as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    let h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^ (h >> 32)
}

/// Hash for spill (> 8-byte) keys.
#[inline]
fn hash_spill(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    bytes.hash(&mut hasher);
    let h = hasher.finish();
    h ^ (h >> 32)
}

#[inline]
fn spill_tag(hash: u64) -> u64 {
    (hash >> 56) & 0xFF
}

impl Vocab {
    /// Build a vocabulary from (token_bytes, rank) pairs.
    pub(crate) fn from_entries(entries: Vec<(Vec<u8>, u32)>) -> Self {
        if entries.is_empty() {
            return Self {
                arena: Box::new([]),
                single: Box::new([]),
                pair: Box::new([]),
                table: Box::new([]),
                mask: 0,
                decoder: Box::new([]),
            };
        }

        let max_rank = entries.iter().map(|(_, r)| *r).max().unwrap_or(0);

        // arena + decoder hold every token regardless of lookup class
        let total_bytes: usize = entries.iter().map(|(k, _)| k.len()).sum();
        let mut arena = Vec::with_capacity(total_bytes);
        let mut items: Vec<(u32, u32, u16)> = Vec::with_capacity(entries.len());
        // u32::MAX offset = sentinel for "rank not present in vocabulary"
        let mut decoder = vec![(u32::MAX, 0u16); max_rank as usize + 1];

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
            let len = token.len() as u16;
            arena.extend_from_slice(token);
            items.push((*rank, offset, len));
            debug_assert!(
                decoder[*rank as usize].0 == u32::MAX,
                "duplicate rank {rank} in vocabulary entries"
            );
            decoder[*rank as usize] = (offset, len);
        }

        let arena = arena.into_boxed_slice();

        let mut single = vec![ABSENT; 256];
        let mut pair = vec![ABSENT; 1 << 16];

        // sized on the full entry count even though 1- and 2-byte keys never
        // enter it — the resulting sub-50% load factor keeps probe chains short
        let table_size = (entries.len() * 2).next_power_of_two();
        let mask = table_size - 1;
        let mut table = vec![EMPTY; table_size];

        for &(rank, offset, len) in &items {
            let token = &arena[offset as usize..(offset as usize + len as usize)];
            match token.len() {
                0 => {}
                1 => single[token[0] as usize] = rank,
                2 => pair[(token[0] as usize) << 8 | token[1] as usize] = rank,
                l if l <= INLINE_MAX => {
                    let key = load_inline_key(token);
                    let mut idx = hash_inline(key, l) as usize & mask;
                    while table[idx].len != 0 {
                        idx = (idx + 1) & mask;
                    }
                    table[idx] = Slot {
                        key,
                        rank,
                        len: l as u32,
                    };
                }
                l => {
                    let hash = hash_spill(token);
                    let mut idx = hash as usize & mask;
                    while table[idx].len != 0 {
                        idx = (idx + 1) & mask;
                    }
                    table[idx] = Slot {
                        key: (offset as u64) | (spill_tag(hash) << 32),
                        rank,
                        len: l as u32,
                    };
                }
            }
        }

        Self {
            arena,
            single: single.into_boxed_slice(),
            pair: pair.into_boxed_slice(),
            table: table.into_boxed_slice(),
            mask,
            decoder: decoder.into_boxed_slice(),
        }
    }

    /// Look up the rank for a token byte sequence.
    #[inline]
    pub(crate) fn get(&self, token: &[u8]) -> Option<u32> {
        // empty vocab ⇔ empty table (a non-empty build always allocates it)
        if self.table.is_empty() {
            return None;
        }
        match token.len() {
            0 => None,
            1 => {
                let rank = self.single[token[0] as usize];
                (rank != ABSENT).then_some(rank)
            }
            2 => {
                let rank = self.pair[(token[0] as usize) << 8 | token[1] as usize];
                (rank != ABSENT).then_some(rank)
            }
            len if len <= INLINE_MAX => {
                let key = load_inline_key(token);
                let mut idx = hash_inline(key, len) as usize & self.mask;
                loop {
                    let slot = self.table[idx];
                    if slot.len == 0 {
                        return None;
                    }
                    if slot.len == len as u32 && slot.key == key {
                        return Some(slot.rank);
                    }
                    idx = (idx + 1) & self.mask;
                }
            }
            len => {
                let hash = hash_spill(token);
                let tag = spill_tag(hash);
                let mut idx = hash as usize & self.mask;
                loop {
                    let slot = self.table[idx];
                    if slot.len == 0 {
                        return None;
                    }
                    if slot.len == len as u32 && (slot.key >> 32) & 0xFF == tag {
                        let offset = (slot.key & 0xFFFF_FFFF) as usize;
                        if &self.arena[offset..offset + len] == token {
                            return Some(slot.rank);
                        }
                    }
                    idx = (idx + 1) & self.mask;
                }
            }
        }
    }

    /// Look up the rank of a 2-byte key without slicing: the merge loop's
    /// initial adjacent-pair scan, which is the single hottest lookup class.
    #[inline]
    pub(crate) fn get_pair(&self, a: u8, b: u8) -> Option<u32> {
        if self.pair.is_empty() {
            return None;
        }
        let rank = self.pair[(a as usize) << 8 | b as usize];
        (rank != ABSENT).then_some(rank)
    }

    /// Check if a token byte sequence exists in the vocabulary.
    #[inline]
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
mod tests {
    use super::*;

    fn sample_entries() -> Vec<(Vec<u8>, u32)> {
        vec![
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"ab".to_vec(), 2),
            (b"abc".to_vec(), 3),
        ]
    }

    #[test]
    fn test_build_and_lookup() {
        let vocab = Vocab::from_entries(sample_entries());
        assert_eq!(vocab.get(b"a"), Some(0));
        assert_eq!(vocab.get(b"b"), Some(1));
        assert_eq!(vocab.get(b"ab"), Some(2));
        assert_eq!(vocab.get(b"abc"), Some(3));
    }

    #[test]
    fn test_missing_key() {
        let vocab = Vocab::from_entries(sample_entries());
        assert_eq!(vocab.get(b"xyz"), None);
        assert_eq!(vocab.get(b"abcd"), None);
        assert_eq!(vocab.get(b""), None);
    }

    #[test]
    fn test_contains_key() {
        let vocab = Vocab::from_entries(sample_entries());
        assert!(vocab.contains_key(b"a"));
        assert!(vocab.contains_key(b"abc"));
        assert!(!vocab.contains_key(b"xyz"));
    }

    #[test]
    fn test_get_pair() {
        let vocab = Vocab::from_entries(sample_entries());
        assert_eq!(vocab.get_pair(b'a', b'b'), Some(2));
        assert_eq!(vocab.get_pair(b'b', b'a'), None);
        assert_eq!(vocab.get_pair(0, 0), None);
    }

    #[test]
    fn test_decode_roundtrip() {
        let entries = sample_entries();
        let vocab = Vocab::from_entries(entries.clone());
        for (token, rank) in &entries {
            assert_eq!(vocab.decode(*rank), token.as_slice());
        }
    }

    #[test]
    fn test_all_single_bytes() {
        let entries: Vec<_> = (0u8..=255).map(|b| (vec![b], b as u32)).collect();
        let vocab = Vocab::from_entries(entries);
        for b in 0u8..=255 {
            assert_eq!(vocab.get(&[b]), Some(b as u32));
            assert_eq!(vocab.decode(b as u32), &[b]);
        }
    }

    #[test]
    fn test_empty_vocab() {
        let vocab = Vocab::from_entries(vec![]);
        assert_eq!(vocab.get(b"anything"), None);
        assert!(!vocab.contains_key(b"x"));
        assert_eq!(vocab.get_pair(b'a', b'b'), None);
    }

    #[test]
    fn test_long_token() {
        let long = vec![0x42u8; 1000];
        let vocab = Vocab::from_entries(vec![(long.clone(), 99)]);
        assert_eq!(vocab.get(&long), Some(99));
        assert_eq!(vocab.decode(99), long.as_slice());
    }

    #[test]
    fn test_inline_boundary_lengths() {
        // exercise every routing class: 1, 2, 3, 4, 7, 8 (inline), 9 (spill)
        let entries: Vec<(Vec<u8>, u32)> = [1usize, 2, 3, 4, 7, 8, 9]
            .iter()
            .enumerate()
            .map(|(i, &l)| (vec![b'x'; l], i as u32))
            .collect();
        let vocab = Vocab::from_entries(entries.clone());
        for (token, rank) in &entries {
            assert_eq!(vocab.get(token), Some(*rank), "len={}", token.len());
        }
        assert_eq!(vocab.get(&[b'x'; 5]), None);
        assert_eq!(vocab.get(&[b'x'; 10]), None);
    }

    #[test]
    fn test_nul_padding_not_confused() {
        // "ab" (len 2) and "ab\0" (len 3) and "ab\0\0" (len 4) share padded
        // key bits; length must separate them in every class
        let entries = vec![
            (b"ab".to_vec(), 1),
            (b"ab\0".to_vec(), 2),
            (b"ab\0\0".to_vec(), 3),
        ];
        let vocab = Vocab::from_entries(entries);
        assert_eq!(vocab.get(b"ab"), Some(1));
        assert_eq!(vocab.get(b"ab\0"), Some(2));
        assert_eq!(vocab.get(b"ab\0\0"), Some(3));
        assert_eq!(vocab.get(b"ab\0\0\0"), None);
    }

    #[test]
    fn test_try_decode_out_of_range() {
        let vocab = Vocab::from_entries(sample_entries());
        assert!(vocab.try_decode(0).is_some());
        assert!(vocab.try_decode(3).is_some());
        assert!(vocab.try_decode(99999).is_none());
    }

    #[test]
    fn test_try_decode_sparse_rank_returns_none() {
        // rank 50 is between rank 0 ("hello") and rank 100 ("world"), but not present
        let vocab = Vocab::from_entries(vec![(b"hello".to_vec(), 0), (b"world".to_vec(), 100)]);
        assert_eq!(vocab.try_decode(0), Some(b"hello".as_slice()));
        assert_eq!(vocab.try_decode(100), Some(b"world".as_slice()));
        // rank 50 does not exist — must return None, not Some(b"")
        assert_eq!(vocab.try_decode(50), None);
        assert_eq!(vocab.try_decode(1), None);
        assert_eq!(vocab.try_decode(99), None);
    }

    #[test]
    fn test_matches_hashmap_cl100k() {
        let hashmap = crate::encoding::parse_tiktoken_data_for_test();
        let entries: Vec<_> = hashmap.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let vocab = Vocab::from_entries(entries);

        for (key, &expected_rank) in &hashmap {
            assert_eq!(
                vocab.get(key),
                Some(expected_rank),
                "mismatch for key len={}",
                key.len()
            );
        }
    }

    #[test]
    fn test_sparse_ranks() {
        let entries = vec![(b"hello".to_vec(), 100), (b"world".to_vec(), 50000)];
        let vocab = Vocab::from_entries(entries);
        assert_eq!(vocab.get(b"hello"), Some(100));
        assert_eq!(vocab.get(b"world"), Some(50000));
        assert_eq!(vocab.decode(100), b"hello");
        assert_eq!(vocab.decode(50000), b"world");
    }

    #[test]
    fn test_len() {
        let vocab = Vocab::from_entries(sample_entries());
        assert_eq!(vocab.len(), 4);
    }

    #[test]
    fn test_len_empty() {
        let vocab = Vocab::from_entries(vec![]);
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn test_len_sparse() {
        let vocab = Vocab::from_entries(vec![(b"hello".to_vec(), 0), (b"world".to_vec(), 100)]);
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_many_entries_no_false_positives() {
        let mut entries: Vec<(Vec<u8>, u32)> = Vec::new();
        for i in 0u32..10000 {
            entries.push((i.to_le_bytes().to_vec(), i));
        }
        let vocab = Vocab::from_entries(entries.clone());

        for (token, rank) in &entries {
            assert_eq!(vocab.get(token), Some(*rank));
        }

        for i in 10000u32..10100 {
            let token = i.to_le_bytes().to_vec();
            assert_eq!(vocab.get(&token), None);
        }
    }
}
