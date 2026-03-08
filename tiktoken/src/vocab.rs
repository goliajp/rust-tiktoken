//! Arena-based vocabulary storage for zero-allocation token lookup.
//!
//! All token byte sequences are stored contiguously in a single `Box<[u8]>` arena.
//! Encoding lookups use an open-addressing hash table with linear probing and
//! `FxHash` for fast, low-collision hashing. Decoding uses direct indexing by rank
//! into a pre-built `(offset, len)` table. This design replaces ~200k individual
//! `Vec<u8>` heap allocations with a single contiguous block.

use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// Arena-based BPE vocabulary with open-addressing hash table.
///
/// All token bytes are stored contiguously in a single allocation (`arena`).
/// Encoding uses an open-addressing hash table with linear probing and
/// byte-content comparison on collision. Decoding uses direct indexing by rank.
pub struct Vocab {
    arena: Box<[u8]>,
    // open-addressing hash table: each slot is (rank, offset, len) or EMPTY
    // indexed by hash(token) & mask, linear probing on collision
    table: Box<[Slot]>,
    mask: usize,
    // indexed by rank: (offset, len) for decode
    decoder: Box<[(u32, u16)]>,
}

#[derive(Clone, Copy)]
struct Slot {
    rank: u32,
    offset: u32,
    len: u16,
    occupied: bool,
}

impl Slot {
    const EMPTY: Self = Self {
        rank: 0,
        offset: 0,
        len: 0,
        occupied: false,
    };
}

#[inline]
fn fx_hash(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    bytes.hash(&mut hasher);
    hasher.finish()
}

impl Vocab {
    /// Build a vocabulary from (token_bytes, rank) pairs.
    pub(crate) fn from_entries(entries: Vec<(Vec<u8>, u32)>) -> Self {
        if entries.is_empty() {
            return Self {
                arena: Box::new([]),
                table: Box::new([]),
                mask: 0,
                decoder: Box::new([]),
            };
        }

        let max_rank = entries.iter().map(|(_, r)| *r).max().unwrap_or(0);

        // build arena
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

        // build hash table (load factor ~50% for good performance)
        let table_size = (entries.len() * 2).next_power_of_two();
        let mask = table_size - 1;
        let mut table = vec![Slot::EMPTY; table_size];

        for &(rank, offset, len) in &items {
            let token = &arena[offset as usize..(offset as usize + len as usize)];
            let mut idx = fx_hash(token) as usize & mask;
            loop {
                if !table[idx].occupied {
                    table[idx] = Slot {
                        rank,
                        offset,
                        len,
                        occupied: true,
                    };
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self {
            arena,
            table: table.into_boxed_slice(),
            mask,
            decoder: decoder.into_boxed_slice(),
        }
    }

    /// Look up the rank for a token byte sequence.
    #[inline]
    pub(crate) fn get(&self, token: &[u8]) -> Option<u32> {
        if self.table.is_empty() {
            return None;
        }
        let mut idx = fx_hash(token) as usize & self.mask;
        loop {
            let slot = &self.table[idx];
            if !slot.occupied {
                return None;
            }
            let stored =
                &self.arena[slot.offset as usize..(slot.offset as usize + slot.len as usize)];
            if stored == token {
                return Some(slot.rank);
            }
            idx = (idx + 1) & self.mask;
        }
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
    }

    #[test]
    fn test_long_token() {
        let long = vec![0x42u8; 1000];
        let vocab = Vocab::from_entries(vec![(long.clone(), 99)]);
        assert_eq!(vocab.get(&long), Some(99));
        assert_eq!(vocab.decode(99), long.as_slice());
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
