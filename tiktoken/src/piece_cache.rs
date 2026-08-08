//! Thread-local memoisation of whole pre-token pieces → token ids.
//!
//! Real text repeats its pieces: chat templates, quoted context, function
//! words, and — especially — CJK prose, where the same particles and stems
//! recur every sentence. Memoising the merge turns every repeat into one hash
//! and one byte-compare instead of a full BPE merge.
//!
//! `gpt-tokenizer` does the same with a 100k-entry LRU `Map` keyed by JS
//! strings, paying a string hash per probe. This cache is direct-mapped with
//! fixed-size byte keys, so a hit costs tens of nanoseconds, and it is
//! thread-local, so no locking taxes the miss path.
//!
//! Correctness containment:
//! - the key is the exact piece bytes plus a per-[`CoreBpe`](crate::CoreBpe)
//!   nonce, so two encodings never share entries — including a new instance
//!   allocated at a dropped instance's address;
//! - a slot is filled only by a completed merge, and eviction is overwrite;
//! - pieces longer than [`KEY_MAX`] bytes or merging into more than
//!   [`TOKENS_MAX`] tokens bypass storage entirely.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Longest piece the cache will key. Sized for CJK prose clauses — Japanese
/// in particular runs kana+kanji clauses of 20-30 chars between punctuation
/// (60-90 bytes); longer pieces are rare enough to just merge.
pub(crate) const KEY_MAX: usize = 96;
/// Most tokens a stored piece may merge into.
const TOKENS_MAX: usize = 48;
/// Direct-mapped slot count. Power of two. 4,096 slots × ~300 B ≈ 1.2 MB per
/// thread that tokenizes — chosen over 2,048 because a few hundred distinct
/// pieces already produce measurable conflict evictions at the smaller size
/// (+23% on the varied-Unicode corpus).
const SLOTS: usize = 4096;

/// Per-instance nonce source. Address reuse after a drop must not let a new
/// instance read a dead instance's entries, so identity is a counter, never a
/// pointer. Starts at 1; 0 marks an empty slot.
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn new_cache_id() -> u64 {
    NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy)]
struct Slot {
    owner: u64, // cache id, 0 = empty
    len: u8,
    n_tokens: u8,
    key: [u8; KEY_MAX],
    tokens: [u32; TOKENS_MAX],
}

const EMPTY: Slot = Slot {
    owner: 0,
    len: 0,
    n_tokens: 0,
    key: [0; KEY_MAX],
    tokens: [0; TOKENS_MAX],
};

thread_local! {
    static CACHE: RefCell<Box<[Slot; SLOTS]>> = RefCell::new(Box::new([EMPTY; SLOTS]));
}

#[inline]
fn slot_index(owner: u64, piece: &[u8]) -> usize {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    owner.hash(&mut h);
    piece.hash(&mut h);
    let x = h.finish();
    ((x ^ (x >> 32)) as usize) & (SLOTS - 1)
}

/// Count the piece's tokens through the cache. `compute` runs the real merge
/// into the provided buffer and returns the token count; it is called only on
/// a miss, and its result is stored when it fits.
///
/// Caller guarantees `piece.len() <= KEY_MAX`.
#[inline]
pub(crate) fn count_piece(
    owner: u64,
    piece: &[u8],
    compute: impl FnOnce(&mut [u32; KEY_MAX]) -> usize,
) -> usize {
    debug_assert!(piece.len() <= KEY_MAX);
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let slot = &mut cache[slot_index(owner, piece)];
        if slot.owner == owner
            && slot.len as usize == piece.len()
            && &slot.key[..piece.len()] == piece
        {
            return slot.n_tokens as usize;
        }
        let mut buf = [0u32; KEY_MAX];
        let n = compute(&mut buf);
        if n <= TOKENS_MAX {
            slot.owner = owner;
            slot.len = piece.len() as u8;
            slot.n_tokens = n as u8;
            slot.key[..piece.len()].copy_from_slice(piece);
            slot.tokens[..n].copy_from_slice(&buf[..n]);
        }
        n
    })
}

/// Append the piece's tokens to `out` through the cache. Same contract as
/// [`count_piece`].
#[inline]
pub(crate) fn encode_piece(
    owner: u64,
    piece: &[u8],
    out: &mut Vec<u32>,
    compute: impl FnOnce(&mut [u32; KEY_MAX]) -> usize,
) {
    debug_assert!(piece.len() <= KEY_MAX);
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let slot = &mut cache[slot_index(owner, piece)];
        if slot.owner == owner
            && slot.len as usize == piece.len()
            && &slot.key[..piece.len()] == piece
        {
            out.extend_from_slice(&slot.tokens[..slot.n_tokens as usize]);
            return;
        }
        let mut buf = [0u32; KEY_MAX];
        let n = compute(&mut buf);
        out.extend_from_slice(&buf[..n]);
        if n <= TOKENS_MAX {
            slot.owner = owner;
            slot.len = piece.len() as u8;
            slot.n_tokens = n as u8;
            slot.key[..piece.len()].copy_from_slice(piece);
            slot.tokens[..n].copy_from_slice(&buf[..n]);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_returns_stored_tokens_per_owner() {
        let a = new_cache_id();
        let b = new_cache_id();
        let mut calls = 0;
        // fill under owner a
        let n = count_piece(a, b"xyz", |buf| {
            calls += 1;
            buf[0] = 7;
            buf[1] = 8;
            2
        });
        assert_eq!((n, calls), (2, 1));
        // hit under a: compute not called
        let n = count_piece(a, b"xyz", |_| unreachable!("must hit"));
        assert_eq!(n, 2);
        let mut out = vec![1];
        encode_piece(a, b"xyz", &mut out, |_| unreachable!("must hit"));
        assert_eq!(out, vec![1, 7, 8]);
        // same bytes under owner b: distinct entry
        let n = count_piece(b, b"xyz", |buf| {
            buf[0] = 9;
            1
        });
        assert_eq!(n, 1);
        // a's entry may have been evicted by b's fill only if they collide;
        // either way the answer must come from a's own compute or slot
        let n = count_piece(a, b"xyz", |buf| {
            buf[0] = 7;
            buf[1] = 8;
            2
        });
        assert_eq!(n, 2);
    }

    #[test]
    fn oversized_results_are_not_stored() {
        let a = new_cache_id();
        let n = count_piece(a, b"big", |buf| {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = i as u32;
            }
            KEY_MAX // > TOKENS_MAX → must not be stored
        });
        assert_eq!(n, KEY_MAX);
        let mut called = false;
        let n = count_piece(a, b"big", |buf| {
            called = true;
            buf[0] = 1;
            KEY_MAX
        });
        assert_eq!((n, called), (KEY_MAX, true));
    }
}
