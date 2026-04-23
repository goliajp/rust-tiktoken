//! Heap-accelerated BPE merge algorithm.
//!
//! Implements byte-pair encoding merging using a min-heap (`BinaryHeap<Reverse>`)
//! combined with a doubly-linked list for O(n log n) complexity. The heap tracks
//! candidate merges by rank; the linked list enables O(1) neighbor updates when
//! a merge removes a position. Lazy deletion handles stale heap entries.

use crate::vocab::Vocab;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// BPE merge: find the optimal partition of `piece` into sub-tokens.
///
/// Returns a list of split points (byte offsets into `piece`), e.g.
/// `[0, 3, 5]` means the piece is split into `piece[0..3]` and `piece[3..5]`.
///
/// Uses a min-heap + doubly-linked list for O(n log n) merging,
/// compared to the v2 algorithm's O(n * m) linear scan.
pub fn byte_pair_merge(piece: &[u8], vocab: &Vocab) -> Vec<usize> {
    let n = piece.len();
    debug_assert!(
        n <= u32::MAX as usize,
        "piece length {} exceeds u32 index range",
        n
    );

    if n == 0 {
        return vec![0];
    }

    // fast path: 1 byte
    if n == 1 {
        return vec![0, 1];
    }

    // fast path: 2 bytes
    if n == 2 {
        if vocab.contains_key(piece) {
            return vec![0, 2];
        }
        return vec![0, 1, 2];
    }

    // doubly-linked list over byte positions 0..n
    // next[i] = next active position after i
    // prev[i] = previous active position before i
    let mut next: Vec<u32> = (1..=n as u32).collect();
    let mut prev: Vec<u32> = (0..n).map(|i| i.saturating_sub(1) as u32).collect();

    // rank_at[i] = rank of the pair (i, next[i]), or u32::MAX if not mergeable
    let mut rank_at: Vec<u32> = vec![u32::MAX; n];

    // min-heap: (rank, position)
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();

    // initialize: compute ranks for all adjacent pairs
    for i in 0..n - 1 {
        if let Some(rank) = vocab.get(&piece[i..i + 2]) {
            rank_at[i] = rank;
            heap.push(Reverse((rank, i as u32)));
        }
    }

    let mut active_count = n;

    while let Some(Reverse((rank, pos))) = heap.pop() {
        let pos = pos as usize;

        // lazy deletion: skip if this entry is stale
        if rank_at[pos] != rank {
            continue;
        }

        let next_pos = next[pos] as usize;
        if next_pos >= n {
            continue;
        }

        if active_count <= 1 {
            break;
        }

        // merge: remove next_pos from the linked list
        let after = next[next_pos] as usize;
        next[pos] = after as u32;
        if after < n {
            prev[after] = pos as u32;
        }
        rank_at[next_pos] = u32::MAX; // mark deleted
        active_count -= 1;

        // recompute rank for the merged pair at pos
        // pair at pos is now: piece[pos..after] + piece[after..next[after]]
        rank_at[pos] = u32::MAX;
        if after < n {
            let after_next = next[after] as usize;
            if after_next <= n
                && let Some(new_rank) = vocab.get(&piece[pos..after_next])
            {
                rank_at[pos] = new_rank;
                heap.push(Reverse((new_rank, pos as u32)));
            }
        }

        // recompute rank for predecessor
        if pos > 0 {
            let prev_pos = prev[pos] as usize;
            rank_at[prev_pos] = u32::MAX;
            let pos_next = next[pos] as usize;
            debug_assert!(pos_next <= n);
            if pos_next > pos
                && let Some(new_rank) = vocab.get(&piece[prev_pos..pos_next])
            {
                rank_at[prev_pos] = new_rank;
                heap.push(Reverse((new_rank, prev_pos as u32)));
            }
        }
    }

    // collect result: walk the linked list
    let mut parts = Vec::with_capacity(active_count + 1);
    let mut i = 0usize;
    while i < n {
        parts.push(i);
        i = next[i] as usize;
    }
    parts.push(n);
    parts
}

/// BPE-encode a piece, writing tokens directly to result.
///
/// # Panics
///
/// Panics if a single byte or merged sub-token is missing from `vocab`.
/// Callers must ensure the vocabulary contains all 256 single bytes.
pub fn bpe_encode(piece: &[u8], vocab: &Vocab, result: &mut Vec<u32>) {
    if piece.len() == 1 {
        result.push(vocab.get(piece).expect("single byte not in vocab"));
        return;
    }

    let parts = byte_pair_merge(piece, vocab);

    for i in 0..parts.len() - 1 {
        let key = &piece[parts[i]..parts[i + 1]];
        result.push(vocab.get(key).expect("merged token not in vocab"));
    }
}

/// Count tokens in a piece without allocating a token vector.
pub fn bpe_count(piece: &[u8], vocab: &Vocab) -> usize {
    if piece.len() == 1 {
        return 1;
    }
    byte_pair_merge(piece, vocab).len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    fn make_vocab(entries: Vec<(Vec<u8>, u32)>) -> Vocab {
        Vocab::from_entries(entries)
    }

    // v2 reference implementation for oracle comparison
    fn v2_byte_pair_merge(piece: &[u8], ranks: &FxHashMap<Vec<u8>, u32>) -> Vec<usize> {
        let n = piece.len() + 1;

        if n == 3 {
            if ranks.contains_key(piece) {
                return vec![0, piece.len()];
            }
            return vec![0, 1, piece.len()];
        }

        let mut parts: Vec<usize> = (0..n).collect();
        let mut rank_cache: Vec<u32> = (0..n)
            .map(|i| {
                if i + 2 < n {
                    ranks.get(&piece[i..i + 2]).copied().unwrap_or(u32::MAX)
                } else {
                    u32::MAX
                }
            })
            .collect();

        loop {
            if parts.len() <= 2 {
                break;
            }

            let mut min_rank = u32::MAX;
            let mut min_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..parts.len() - 1 {
                if rank_cache[i] < min_rank {
                    min_rank = rank_cache[i];
                    min_idx = i;
                }
            }

            if min_rank == u32::MAX {
                break;
            }

            parts.remove(min_idx + 1);
            rank_cache.remove(min_idx + 1);

            rank_cache[min_idx] = if min_idx + 2 < parts.len() {
                ranks
                    .get(&piece[parts[min_idx]..parts[min_idx + 2]])
                    .copied()
                    .unwrap_or(u32::MAX)
            } else {
                u32::MAX
            };

            if min_idx > 0 {
                rank_cache[min_idx - 1] = if min_idx + 1 < parts.len() {
                    ranks
                        .get(&piece[parts[min_idx - 1]..parts[min_idx + 1]])
                        .copied()
                        .unwrap_or(u32::MAX)
                } else {
                    u32::MAX
                };
            }
        }

        parts
    }

    #[test]
    fn test_empty_piece() {
        let vocab = make_vocab(vec![(b"x".to_vec(), 0)]);
        assert_eq!(byte_pair_merge(b"", &vocab), vec![0]);
    }

    #[test]
    fn test_single_byte() {
        let vocab = make_vocab(vec![(b"x".to_vec(), 0)]);
        assert_eq!(byte_pair_merge(b"x", &vocab), vec![0, 1]);
    }

    #[test]
    fn test_two_bytes_merged() {
        let vocab = make_vocab(vec![
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"ab".to_vec(), 2),
        ]);
        assert_eq!(byte_pair_merge(b"ab", &vocab), vec![0, 2]);
    }

    #[test]
    fn test_two_bytes_unmerged() {
        let vocab = make_vocab(vec![(b"a".to_vec(), 0), (b"b".to_vec(), 1)]);
        assert_eq!(byte_pair_merge(b"ab", &vocab), vec![0, 1, 2]);
    }

    #[test]
    fn test_picks_lowest_rank_first() {
        // de(3) < ef(4), so merge de first → [de, f]
        let vocab = make_vocab(vec![
            (b"d".to_vec(), 0),
            (b"e".to_vec(), 1),
            (b"f".to_vec(), 2),
            (b"de".to_vec(), 3),
            (b"ef".to_vec(), 4),
        ]);
        assert_eq!(byte_pair_merge(b"def", &vocab), vec![0, 2, 3]);
    }

    #[test]
    fn test_full_collapse() {
        // ab(5) is lowest rank, merge first → ab+c
        // abc(3) exists → full collapse
        let vocab = make_vocab(vec![
            (b"a".to_vec(), 10),
            (b"b".to_vec(), 20),
            (b"c".to_vec(), 30),
            (b"ab".to_vec(), 5),
            (b"abc".to_vec(), 3),
        ]);
        assert_eq!(byte_pair_merge(b"abc", &vocab), vec![0, 3]);
    }

    #[test]
    fn test_no_merges_possible() {
        let vocab = make_vocab(vec![
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"c".to_vec(), 2),
        ]);
        assert_eq!(byte_pair_merge(b"abc", &vocab), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_matches_v2_on_real_vocab() {
        let hashmap = crate::encoding::parse_tiktoken_data_for_test();
        let entries: Vec<_> = hashmap.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let vocab = Vocab::from_entries(entries);

        // test various pieces that would go through the BPE merge path
        let test_pieces: Vec<&[u8]> = vec![
            b"hello",
            b"world",
            b"tokenization",
            b"supercalifragilistic",
            b"\xe4\xbd\xa0\xe5\xa5\xbd", // 你好
            b"abc",
            b"xyz123",
            b"  hello  ",
            b"\n\n\n",
        ];

        for piece in test_pieces {
            let v2_result = v2_byte_pair_merge(piece, &hashmap);
            let v3_result = byte_pair_merge(piece, &vocab);
            assert_eq!(
                v2_result,
                v3_result,
                "mismatch for piece: {:?}",
                std::str::from_utf8(piece).unwrap_or("<non-utf8>")
            );
        }
    }

    #[test]
    fn test_bpe_encode_single_byte() {
        let vocab = make_vocab(vec![(b"x".to_vec(), 42)]);
        let mut result = Vec::new();
        bpe_encode(b"x", &vocab, &mut result);
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn test_bpe_count_matches_encode() {
        let vocab = make_vocab(vec![
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"c".to_vec(), 2),
            (b"ab".to_vec(), 3),
        ]);
        let piece = b"abc";
        let mut tokens = Vec::new();
        bpe_encode(piece, &vocab, &mut tokens);
        assert_eq!(bpe_count(piece, &vocab), tokens.len());
    }
}
