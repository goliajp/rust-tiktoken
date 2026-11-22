//! The BPE merge for one piece, with tiktoken's rule: adjacent parts merge
//! when their concatenation is a token, the lowest rank first, ties to the
//! leftmost. A merged part's id is the rank that formed it, so the result
//! needs no lookup per output token.
//!
//! Pieces of up to [`SMALL_MAX`] bytes merge in stack arrays: a linked list
//! of parts by byte position, the rank of each part joined with its
//! successor, and a scan for the leftmost minimum each round. Longer pieces
//! use a min-heap with lazy deletion. Pieces made mostly of multibyte chars
//! start from whole atoms and from the parts of wall chars, which never
//! merge with their neighbours (see the `atoms` and `walls` modules of
//! `vocab`).

use crate::vocab::{ABSENT, Atoms, KEY_READ, Start, Vocab, char_len, cp_at};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Longest piece merged in stack arrays.
pub(crate) const SMALL_MAX: usize = 128;
// pieces up to these lengths use arrays sized for them
const TINY_MAX: usize = 16;
const MID_MAX: usize = 48;

/// Leftmost minimum of `r`; `u32::MAX` when `r` holds nothing smaller.
#[inline(always)]
fn argmin(r: &[u32]) -> (usize, u32) {
    let min = r.iter().fold(u32::MAX, |a, &b| a.min(b));
    if min == u32::MAX {
        return (0, min);
    }
    (r.iter().position(|&x| x == min).unwrap_or(0), min)
}

// long pieces keep the minimum of each block of ranks, so a round scans the
// block minima and one block instead of every rank
const BLOCK: usize = 16;
const BLOCKS: usize = SMALL_MAX / BLOCK;

#[inline(always)]
fn block_min(rank: &[u32], b: usize) -> u32 {
    rank[b * BLOCK..(b + 1) * BLOCK]
        .iter()
        .fold(u32::MAX, |a, &x| a.min(x))
}

/// Merge `piece` (`2..=N` bytes) and write the ids of its parts to `ids`;
/// returns the part count. A byte missing from the vocabulary yields
/// [`ABSENT`] for its part.
#[inline(always)]
fn merge_n<const N: usize>(
    piece: &[u8],
    vocab: &Vocab,
    ids: &mut [u32],
    atoms: Option<&Atoms>,
) -> usize {
    let n = piece.len();
    debug_assert!((2..=N).contains(&n) && N >= KEY_READ && ids.len() >= n);
    // twice the piece's room, so every substring key loads with two word reads
    let mut buf = [[0u8; N]; 2];
    copy_piece(&mut buf.as_flattened_mut()[..n], piece);
    let buf = buf.as_flattened();
    // next[i]: start of the part after the one at byte i (n at the end);
    // prev[i]: start of the part before it
    let mut next = [0u8; N];
    let mut prev = [0u8; N];
    // rank[i]: rank of the part at i joined with its successor, MAX if none
    let mut rank = [u32::MAX; N];
    let probe = vocab.rank_probe();
    let span = |a: usize, b: usize| probe.rank_padded(buf, a, b - a);
    // wall[i]: the part at i belongs to a wall char and never merges
    let mut wall = [false; N];
    let walls = atoms.is_some();
    if let Some(atoms) = atoms {
        // atoms start whole, wall chars as their parts, other chars as bytes
        let table = atoms.walls();
        let mut last = usize::MAX;
        let mut part = |at: usize, len: usize, id: u32, w: bool, last: &mut usize| {
            ids[at] = id;
            prev[at] = *last as u8;
            next[at] = (at + len) as u8;
            wall[at] = w;
            if *last != usize::MAX {
                rank[*last] = if w || wall[*last] {
                    u32::MAX
                } else if at - *last == 1 && len == 1 {
                    probe.rank_pair(piece[*last], piece[at])
                } else {
                    span(*last, at + len)
                };
            }
            *last = at;
        };
        let mut i = 0;
        while i < n {
            let l = char_len(piece[i]);
            if l == 4 {
                if let Some(r) = atoms_at(atoms, piece, i, l) {
                    part(i, l, r, false, &mut last);
                    i += l;
                    continue;
                }
            } else if l > 1
                && let Some(cp) = cp_at(piece, i, l)
            {
                match table.start(piece, i, l, cp, |b| vocab.rank_single(b)) {
                    Start::Whole(r) => {
                        part(i, l, r, false, &mut last);
                        i += l;
                        continue;
                    }
                    Start::Wall(parts) => {
                        let mut at = i;
                        for &(len, id) in parts.iter().take_while(|p| p.0 != 0) {
                            part(at, len as usize, id, true, &mut last);
                            at += len as usize;
                        }
                        i += l;
                        continue;
                    }
                    Start::Bytes => {}
                }
            }
            part(i, 1, vocab.rank_single(piece[i]), false, &mut last);
            i += 1;
        }
    } else {
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = i.wrapping_sub(1) as u8;
            ids[i] = vocab.rank_single(piece[i]);
        }
        for i in 0..n - 1 {
            rank[i] = probe.rank_pair(piece[i], piece[i + 1]);
        }
    }
    let blocked = N > MID_MAX;
    let mut mins = [u32::MAX; BLOCKS];
    let n_blocks = (n - 1).div_ceil(BLOCK);
    if blocked {
        for (b, m) in mins[..n_blocks].iter_mut().enumerate() {
            *m = block_min(&rank, b);
        }
    }
    loop {
        let (k, r) = if blocked {
            let (b, r) = argmin(&mins[..n_blocks]);
            let at = b * BLOCK;
            (at + argmin(&rank[at..at + BLOCK]).0, r)
        } else {
            argmin(&rank[..n - 1])
        };
        if r == u32::MAX {
            break;
        }
        ids[k] = r;
        let j = next[k] as usize;
        rank[j] = u32::MAX;
        let nj = next[j] as usize;
        next[k] = nj as u8;
        rank[k] = if nj < n {
            prev[nj] = k as u8;
            if walls && wall[nj] {
                u32::MAX
            } else {
                span(k, next[nj] as usize)
            }
        } else {
            u32::MAX
        };
        let mut p = k;
        if k > 0 {
            p = prev[k] as usize;
            if !(walls && wall[p]) {
                rank[p] = span(p, nj);
            }
        }
        if blocked {
            // the changed ranks sit at p <= k < j
            let (lo, hi) = (p / BLOCK, j / BLOCK);
            for (b, m) in mins.iter_mut().enumerate().take(hi + 1).skip(lo) {
                *m = block_min(&rank, b);
            }
        }
    }
    // compact in place: the c-th live part starts at a byte >= c
    let (mut i, mut c) = (0, 0);
    while i < n {
        ids[c] = ids[i];
        c += 1;
        i = next[i] as usize;
    }
    c
}

/// `dst.copy_from_slice(src)` for up to 16 bytes, in two overlapping
/// word copies rather than a library call.
#[inline(always)]
fn copy_piece(dst: &mut [u8], src: &[u8]) {
    let n = src.len();
    debug_assert!(dst.len() == n);
    if n > 16 {
        dst.copy_from_slice(src);
    } else if n >= 8 {
        dst[..8].copy_from_slice(&src[..8]);
        dst[n - 8..].copy_from_slice(&src[n - 8..]);
    } else if n >= 4 {
        dst[..4].copy_from_slice(&src[..4]);
        dst[n - 4..].copy_from_slice(&src[n - 4..]);
    } else {
        for (d, &b) in dst.iter_mut().zip(src) {
            *d = b;
        }
    }
}

/// The atom at `piece[i..i + l]`, if the char there is one and whole.
#[inline(always)]
fn atoms_at(atoms: &Atoms, piece: &[u8], i: usize, l: usize) -> Option<u32> {
    if l < 2 || i + l > piece.len() {
        return None;
    }
    atoms.at(piece, i, l)
}

/// Whether most bytes of `piece` belong to multibyte chars. Starting from
/// atoms pays off on such pieces; on words with a curly quote or an accent
/// the extra lookups cost more than the merges they save.
#[inline(always)]
fn mostly_multibyte(piece: &[u8]) -> bool {
    // most words of Latin script end in ASCII: one compare turns them away
    if piece[piece.len() - 1] < 0x80 {
        return false;
    }
    let (words, rest) = piece.as_chunks::<8>();
    let high = words
        .iter()
        .map(|w| (u64::from_le_bytes(*w) & 0x8080_8080_8080_8080).count_ones() as usize)
        .sum::<usize>()
        + rest.iter().filter(|&&b| b >= 0x80).count();
    2 * high > piece.len()
}

/// Merge a piece of `1..=SMALL_MAX` bytes into `ids`; returns the id count.
#[inline]
pub(crate) fn merge_small(piece: &[u8], vocab: &Vocab, ids: &mut [u32]) -> usize {
    if mostly_multibyte(piece) {
        return merge_from_atoms(piece, vocab, ids);
    }
    merge_small_bytes(piece, vocab, ids)
}

/// [`merge_small`] from bytes only, never from atoms.
#[inline]
pub(crate) fn merge_small_bytes(piece: &[u8], vocab: &Vocab, ids: &mut [u32]) -> usize {
    match piece.len() {
        1 => {
            ids[0] = vocab.rank_single(piece[0]);
            1
        }
        n if n <= TINY_MAX => merge_n::<TINY_MAX>(piece, vocab, ids, None),
        n if n <= MID_MAX => merge_n::<MID_MAX>(piece, vocab, ids, None),
        _ => merge_n::<SMALL_MAX>(piece, vocab, ids, None),
    }
}

/// [`merge_small`] of a piece of at least 2 bytes, from its atoms.
#[inline(never)]
fn merge_from_atoms(piece: &[u8], vocab: &Vocab, ids: &mut [u32]) -> usize {
    let atoms = vocab.atoms();
    match chars::merge_chars(piece, vocab, atoms, ids) {
        Some(c) => c,
        None => merge_linked_from_atoms(piece, vocab, ids, atoms),
    }
}

/// [`merge_from_atoms`] for pieces with a run of parts too long for the
/// char-at-a-time merge.
#[cold]
#[inline(never)]
fn merge_linked_from_atoms(piece: &[u8], vocab: &Vocab, ids: &mut [u32], atoms: &Atoms) -> usize {
    match piece.len() {
        n if n <= TINY_MAX => merge_n::<TINY_MAX>(piece, vocab, ids, Some(atoms)),
        n if n <= MID_MAX => merge_n::<MID_MAX>(piece, vocab, ids, Some(atoms)),
        _ => merge_n::<SMALL_MAX>(piece, vocab, ids, Some(atoms)),
    }
}

/// Heap merge for pieces of any length of at least 2 bytes; appends the
/// part ids to `out`.
fn merge_heap(piece: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let n = piece.len();
    debug_assert!(n >= 2 && n <= u32::MAX as usize);
    let mut buf = Vec::with_capacity(n + KEY_READ);
    buf.extend_from_slice(piece);
    buf.resize(n + KEY_READ, 0);

    let mut next: Vec<u32> = (1..=n as u32).collect();
    let mut prev: Vec<u32> = (0..n).map(|i| i.saturating_sub(1) as u32).collect();
    let mut ids: Vec<u32> = piece.iter().map(|&b| vocab.rank_single(b)).collect();
    let mut rank_at = vec![u32::MAX; n];
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    for i in 0..n - 1 {
        let r = vocab.rank_pair(piece[i], piece[i + 1]);
        if r != ABSENT {
            rank_at[i] = r;
            heap.push(Reverse((r, i as u32)));
        }
    }

    while let Some(Reverse((rank, pos))) = heap.pop() {
        let pos = pos as usize;
        // lazy deletion: the pair changed since this entry was pushed
        if rank_at[pos] != rank {
            continue;
        }
        let j = next[pos] as usize;
        let after = next[j] as usize;
        ids[pos] = rank;
        next[pos] = after as u32;
        rank_at[j] = u32::MAX;
        rank_at[pos] = u32::MAX;
        if after < n {
            prev[after] = pos as u32;
            let end = next[after] as usize;
            let r = vocab.rank_padded(&buf, pos, end - pos);
            if r != ABSENT {
                rank_at[pos] = r;
                heap.push(Reverse((r, pos as u32)));
            }
        }
        if pos > 0 {
            let p = prev[pos] as usize;
            let r = vocab.rank_padded(&buf, p, after - p);
            rank_at[p] = r;
            if r != ABSENT {
                heap.push(Reverse((r, p as u32)));
            }
        }
    }

    let mut i = 0;
    while i < n {
        out.push(ids[i]);
        i = next[i] as usize;
    }
}

/// Part boundaries of `piece`: byte offsets from 0 to its length.
#[cfg(test)]
pub fn byte_pair_merge(piece: &[u8], vocab: &Vocab) -> Vec<usize> {
    let mut ids = Vec::new();
    if !piece.is_empty() {
        bpe_encode(piece, vocab, &mut ids);
    }
    let mut parts = vec![0];
    for &id in &ids {
        let len = vocab.try_decode(id).map_or(1, <[u8]>::len);
        parts.push(parts.last().unwrap() + len);
    }
    parts
}

/// BPE-encode a piece, appending its tokens to `out`.
///
/// # Panics
///
/// Panics if a byte of the piece is missing from the vocabulary.
pub fn bpe_encode(piece: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let from = out.len();
    if piece.len() <= SMALL_MAX {
        let mut ids = [0u32; SMALL_MAX];
        let k = merge_small(piece, vocab, &mut ids);
        out.extend_from_slice(&ids[..k]);
    } else {
        merge_heap(piece, vocab, out);
    }
    assert!(!out[from..].contains(&ABSENT), "single byte not in vocab");
}

/// Count the tokens of a piece without keeping them.
#[cfg(test)]
pub fn bpe_count(piece: &[u8], vocab: &Vocab) -> usize {
    if piece.len() <= SMALL_MAX {
        let mut ids = [0u32; SMALL_MAX];
        merge_small(piece, vocab, &mut ids)
    } else {
        let mut out = Vec::new();
        merge_heap(piece, vocab, &mut out);
        out.len()
    }
}

#[path = "merge_chars.rs"]
mod chars;

#[cfg(test)]
#[path = "merge_atoms_tests.rs"]
mod atoms_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::FxHashMap;

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

    #[cfg(feature = "vocab-cl100k_base")]
    #[test]
    fn test_matches_v2_on_real_vocab() {
        let hashmap = crate::encoding::parse_tiktoken_data_for_test();
        let entries: Vec<_> = hashmap.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let vocab = Vocab::from_entries(entries);

        // test various pieces that would go through the BPE merge path.
        // The last few exceed LINEAR_THRESHOLD (32 bytes) on purpose, so they
        // exercise the heap-based path rather than the short-piece linear scan.
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
            b"supercalifragilisticexpialidocious", // 34 bytes > threshold
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // 64 bytes
            b"0123456789012345678901234567890123456789", // 40 digit bytes
            b"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", // 40 punct bytes
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

    #[cfg(feature = "vocab-cl100k_base")]
    #[test]
    fn random_pieces_match_v2_at_every_length_class() {
        let hashmap = crate::encoding::parse_tiktoken_data_for_test();
        let entries: Vec<_> = hashmap.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let vocab = Vocab::from_entries(entries);
        let alphabets: [&[char]; 4] = [
            &['a', 'e', 'i', 'n', 's', 't', 'r', 'o', ' ', 'E', 'q'],
            &[
                '的', '一', '是', '了', '我', '不', '人', '在', '他', '有', '这', '中', '大', '来',
                '上', '国',
            ],
            &[
                'こ', 'ん', 'に', 'ち', 'は', 'ト', 'ー', 'ク', 'ン', '語', '彙',
            ],
            &[
                'م', 'ر', 'ح', 'ب', 'ا', 'ل', 'ع', '안', '녕', '하', '세', '요', '1', '!', '\n',
            ],
        ];
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for case in 0..6000 {
            let alpha = alphabets[case % alphabets.len()];
            let target = 1 + (next() % 200) as usize;
            let mut s = String::new();
            while s.len() < target {
                s.push(alpha[(next() % alpha.len() as u64) as usize]);
            }
            let piece = s.as_bytes();
            assert_eq!(
                byte_pair_merge(piece, &vocab),
                v2_byte_pair_merge(piece, &hashmap),
                "{s:?}"
            );
            let mut ids = vec![];
            bpe_encode(piece, &vocab, &mut ids);
            assert_eq!(bpe_count(piece, &vocab), ids.len());
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
