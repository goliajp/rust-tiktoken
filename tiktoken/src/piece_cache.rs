//! Thread-local memo of whole pieces → token ids, for what the shared table
//! does not hold: pieces longer than its 15-byte keys, and any piece once the
//! shared table stops accepting entries. CJK prose clauses are the main
//! users; they repeat from sentence to sentence.
//!
//! Layout: a direct-mapped array of 16-byte slots (hash, arena offset) and an
//! arena of records `[len | n << 16] [owner lo] [owner hi] [key bytes,
//! padded to 4] [ids]`. A hit compares the 64-bit hash, then the record's
//! owner, length and bytes exactly. When the arena is full the whole cache is
//! reset. Eviction is overwrite; a slot only ever points at a complete record.
//!
//! Entries are keyed by a per-[`CoreBpe`](crate::CoreBpe) nonce, never an
//! address, so a new instance at a dropped instance's address cannot read its
//! entries.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Longest piece the cache keys.
pub(crate) const KEY_MAX: usize = 255;
const SLOTS: usize = 8192;
// arena size in u32 words (1 MiB)
const ARENA_WORDS: usize = 1 << 18;
const HEADER_WORDS: usize = 3;

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

/// A fresh owner nonce. Starts at 1.
pub(crate) fn new_cache_id() -> u64 {
    NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy, Default)]
struct Slot {
    // 0 = empty
    hash: u64,
    off: u32,
}

struct Cache {
    slots: Vec<Slot>,
    arena: Vec<u32>,
}

thread_local! {
    static CACHE: RefCell<Cache> = const {
        RefCell::new(Cache { slots: Vec::new(), arena: Vec::new() })
    };
}

#[inline(always)]
fn read_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let p = a as u128 * b as u128;
    (p as u64) ^ ((p >> 64) as u64)
}

/// Hash of `(owner, piece)`, never 0. Pieces of 16 bytes or more are read
/// 16 bytes at a time, the last block overlapping.
#[inline]
fn hash(owner: u64, piece: &[u8]) -> u64 {
    const K0: u64 = 0x9E37_79B9_7F4A_7C15;
    const K1: u64 = 0xD6E8_FEB8_6659_FD93;
    let n = piece.len();
    let mut h = fold(owner ^ K0, (n as u64) ^ K1);
    if n >= 16 {
        let mut at = 0;
        while at + 16 < n {
            h = fold(h ^ read_u64(piece, at), read_u64(piece, at + 8) ^ K1);
            at += 16;
        }
        h = fold(h ^ read_u64(piece, n - 16), read_u64(piece, n - 8) ^ K1);
    } else {
        let mut w = [0u8; 16];
        w[..n].copy_from_slice(piece);
        h = fold(h ^ read_u64(&w, 0), read_u64(&w, 8) ^ K1);
    }
    h.wrapping_mul(K0) | 1
}

#[inline(always)]
fn as_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: any u32 is four valid bytes; the length is in bytes
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) }
}

impl Cache {
    #[inline]
    fn find(&self, owner: u64, piece: &[u8], h: u64) -> Option<&[u32]> {
        let slot = self.slots.get(h as usize & (SLOTS - 1))?;
        if slot.hash != h {
            return None;
        }
        let rec = &self.arena[slot.off as usize..];
        let (len, n) = ((rec[0] & 0xFFFF) as usize, (rec[0] >> 16) as usize);
        let stored = rec[1] as u64 | (rec[2] as u64) << 32;
        let key_words = len.div_ceil(4);
        let body = &rec[HEADER_WORDS..HEADER_WORDS + key_words + n];
        if len != piece.len() || stored != owner || &as_bytes(&body[..key_words])[..len] != piece {
            return None;
        }
        Some(&body[key_words..])
    }

    fn store(&mut self, owner: u64, piece: &[u8], h: u64, ids: &[u32]) {
        let key_words = piece.len().div_ceil(4);
        let need = HEADER_WORDS + key_words + ids.len();
        if self.slots.is_empty() {
            self.slots = vec![Slot::default(); SLOTS];
            // touch the first stretch of the arena now: a page fault costs
            // microseconds, and the first records of a text come in a burst
            self.arena.reserve(ARENA_WORDS / 16);
            self.arena
                .spare_capacity_mut()
                .fill(std::mem::MaybeUninit::new(0));
        }
        if self.arena.len() + need > ARENA_WORDS {
            self.arena.clear();
            self.slots.fill(Slot::default());
        }
        let off = self.arena.len();
        self.arena
            .push(piece.len() as u32 | (ids.len() as u32) << 16);
        self.arena.push(owner as u32);
        self.arena.push((owner >> 32) as u32);
        for chunk in piece.chunks(4) {
            let mut w = [0u8; 4];
            w[..chunk.len()].copy_from_slice(chunk);
            self.arena.push(u32::from_ne_bytes(w));
        }
        self.arena.extend_from_slice(ids);
        self.slots[h as usize & (SLOTS - 1)] = Slot {
            hash: h,
            off: off as u32,
        };
    }
}

/// Look up the piece; on a miss, `compute` merges it into the buffer and
/// returns the id count, and the result is stored. Calls `emit` with the ids
/// either way and returns what it returns.
///
/// Caller guarantees `piece.len() <= KEY_MAX`; a piece merges into at most
/// one id per byte.
#[inline]
pub(crate) fn with_ids<R>(
    owner: u64,
    piece: &[u8],
    compute: impl FnOnce(&mut [u32; KEY_MAX]) -> usize,
    emit: impl FnOnce(&[u32]) -> R,
) -> R {
    debug_assert!(!piece.is_empty() && piece.len() <= KEY_MAX);
    let h = hash(owner, piece);
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(ids) = cache.find(owner, piece, h) {
            return emit(ids);
        }
        let mut buf = [0u32; KEY_MAX];
        let n = compute(&mut buf);
        cache.store(owner, piece, h, &buf[..n]);
        emit(&buf[..n])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(owner: u64, piece: &[u8], ids: &[u32], calls: &mut u32) -> Vec<u32> {
        with_ids(
            owner,
            piece,
            |buf| {
                *calls += 1;
                buf[..ids.len()].copy_from_slice(ids);
                ids.len()
            },
            <[u32]>::to_vec,
        )
    }

    #[test]
    fn hit_returns_stored_ids_per_owner() {
        let (a, b) = (new_cache_id(), new_cache_id());
        let mut calls = 0;
        assert_eq!(count(a, b"xyz", &[7, 8], &mut calls), [7, 8]);
        assert_eq!(count(a, b"xyz", &[0], &mut calls), [7, 8]);
        assert_eq!(calls, 1);
        // same bytes under another owner: its own entry
        assert_eq!(count(b, b"xyz", &[9], &mut calls), [9]);
        assert_eq!(calls, 2);
    }

    #[test]
    fn long_keys_compare_exactly() {
        let a = new_cache_id();
        let mut calls = 0;
        let mut k1 = vec![b'q'; KEY_MAX];
        let ids: Vec<u32> = (0..KEY_MAX as u32).collect();
        assert_eq!(count(a, &k1, &ids, &mut calls), ids);
        assert_eq!(count(a, &k1, &[1], &mut calls), ids);
        k1[100] = b'r';
        assert_eq!(count(a, &k1, &[1], &mut calls), [1]);
        assert_eq!(count(a, &k1[..17], &[2], &mut calls), [2]);
        assert_eq!(count(a, &k1[..16], &[3], &mut calls), [3]);
        assert_eq!(calls, 4);
    }

    #[test]
    fn full_arena_resets_and_keeps_answering() {
        let a = new_cache_id();
        let mut calls = 0;
        let ids = [5u32; 200];
        // each record is about 270 words; overflow the arena several times
        for round in 0..3u32 {
            for i in 0..2000u32 {
                let key = format!("{i:0>250}");
                let got = count(a, key.as_bytes(), &ids, &mut calls);
                assert_eq!(got, ids, "round {round} key {i}");
            }
        }
        assert!(calls >= 2000);
    }
}
