//! Merge results of pieces longer than the packed key, shared by all threads.
//!
//! CJK prose clauses are the main entries: they are longer than 15 bytes,
//! rarely tokens, and repeat. A slot is one word, `tag << 32 | offset + 1`
//! (0 = empty), in a bucket of eight that shares a cache line; the record at
//! the arena offset is `[len | n << 16] [key bytes, padded to 4] [ids]`.
//! A hit compares the tag, then the length and bytes exactly.
//!
//! Publication: a writer reserves its record with one `fetch_add` on the
//! arena length, writes it, then claims an empty slot with a `Release`
//! compare-exchange of the whole slot word. Readers load the slot with
//! `Acquire`, so the record they then read is complete. Records never change
//! after that, until [`LongDerived::clear`], which needs exclusive use. A full
//! bucket or arena refuses further entries.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::key::hash_long;

/// Longest piece kept.
pub(crate) const LONG_KEY_MAX: usize = 255;
const BUCKET: usize = 8;
const BUCKETS: usize = 1 << 13;
const ARENA_WORDS: usize = 1 << 20;

#[repr(C, align(64))]
struct Bucket([AtomicU64; BUCKET]);

pub(crate) struct LongDerived {
    buckets: Box<[Bucket]>,
    arena: Box<[AtomicU32]>,
    /// On a line of its own: every insert writes it, and next to `buckets`
    /// and `arena`, which every lookup reads, it would invalidate that line
    /// in every other core.
    arena_len: Padded,
}

#[repr(align(128))]
struct Padded(AtomicU32);

/// Ids of an entry, as published.
pub(crate) struct LongIds<'a>(&'a [AtomicU32]);

impl LongIds<'_> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// The ids, which the slot's `Acquire` load made visible and which are
    /// not written again until a [`clear`](LongDerived::clear).
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u32 {
        self.0.as_ptr().cast::<u32>()
    }

    /// Append the ids to `out`.
    #[inline]
    pub(crate) fn extend_into(&self, out: &mut Vec<u32>) {
        out.extend(self.0.iter().map(|w| w.load(Ordering::Relaxed)));
    }
}

/// The slot word's tag bits: hash bits below those that pick the bucket (the
/// low bits of the hash are the weak ones).
#[inline(always)]
fn tag(h: u64) -> u64 {
    (h << BUCKETS.trailing_zeros()) & !0xFFFF_FFFF
}

impl LongDerived {
    pub(crate) fn new() -> Self {
        let buckets = (0..BUCKETS)
            .map(|_| Bucket(std::array::from_fn(|_| AtomicU64::new(0))))
            .collect();
        // zeroed pages are mapped lazily; only the part in use is touched
        let words = vec![0u32; ARENA_WORDS].into_boxed_slice();
        // SAFETY: AtomicU32 has the same size, alignment and bit validity as u32
        let arena = unsafe { Box::from_raw(Box::into_raw(words) as *mut [AtomicU32]) };
        Self {
            buckets,
            arena,
            arena_len: Padded(AtomicU32::new(0)),
        }
    }

    #[inline(always)]
    fn bucket(&self, h: u64) -> &Bucket {
        &self.buckets[(h >> (64 - BUCKETS.trailing_zeros())) as usize]
    }

    /// The ids stored for `piece` (more than 15 bytes), if any.
    #[inline]
    pub(crate) fn get(&self, piece: &[u8]) -> Option<LongIds<'_>> {
        let h = hash_long(piece);
        let tag = tag(h);
        for slot in &self.bucket(h).0 {
            let v = slot.load(Ordering::Acquire);
            if v == 0 {
                return None;
            }
            if v & !0xFFFF_FFFF == tag
                && let Some(ids) = self.record(v as u32 - 1, piece)
            {
                return Some(ids);
            }
        }
        None
    }

    /// The ids of the record at `off` if its key is `piece`.
    #[inline]
    fn record(&self, off: u32, piece: &[u8]) -> Option<LongIds<'_>> {
        let rec = &self.arena[off as usize..];
        let head = rec[0].load(Ordering::Relaxed);
        let (len, n) = ((head & 0xFFFF) as usize, (head >> 16) as usize);
        if len != piece.len() {
            return None;
        }
        let key_words = len.div_ceil(4);
        let key = &rec[1..1 + key_words];
        // SAFETY: the slot's Acquire load ordered the record's writes before
        // this read, and a published record is never written again
        let bytes = unsafe { std::slice::from_raw_parts(key.as_ptr().cast::<u8>(), len) };
        if !same_long(bytes, piece) {
            return None;
        }
        Some(LongIds(&rec[1 + key_words..1 + key_words + n]))
    }

    /// Publish the ids of `piece` (16 to [`LONG_KEY_MAX`] bytes). False when
    /// the table refuses it; the caller keeps the result elsewhere.
    pub(crate) fn insert(&self, piece: &[u8], ids: &[u32]) -> bool {
        debug_assert!(piece.len() > 15 && piece.len() <= LONG_KEY_MAX && ids.len() <= piece.len());
        let h = hash_long(piece);
        let bucket = self.bucket(h);
        if bucket.0[BUCKET - 1].load(Ordering::Relaxed) != 0 {
            return false;
        }
        let key_words = piece.len().div_ceil(4);
        let need = 1 + key_words + ids.len();
        let off = self.arena_len.0.fetch_add(need as u32, Ordering::Relaxed) as usize;
        if off + need > ARENA_WORDS {
            // keep the length from creeping towards overflow
            self.arena_len
                .0
                .store(ARENA_WORDS as u32, Ordering::Relaxed);
            return false;
        }
        let rec = &self.arena[off..off + need];
        rec[0].store(
            piece.len() as u32 | (ids.len() as u32) << 16,
            Ordering::Relaxed,
        );
        for (w, chunk) in rec[1..].iter().zip(piece.chunks(4)) {
            let mut b = [0u8; 4];
            b[..chunk.len()].copy_from_slice(chunk);
            w.store(u32::from_ne_bytes(b), Ordering::Relaxed);
        }
        for (w, &id) in rec[1 + key_words..].iter().zip(ids) {
            w.store(id, Ordering::Relaxed);
        }
        let v = tag(h) | (off as u64 + 1);
        for slot in &bucket.0 {
            if slot
                .compare_exchange(0, v, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
        false
    }

    /// Drop every entry. Caller guarantees no concurrent use.
    pub(crate) fn clear(&self) {
        if self.arena_len.0.load(Ordering::Relaxed) == 0 {
            return;
        }
        // write only the slots in use: an unwritten line stays shared in
        // every core's cache
        for b in self.buckets.iter() {
            for slot in &b.0 {
                if slot.load(Ordering::Relaxed) != 0 {
                    slot.store(0, Ordering::Relaxed);
                }
            }
        }
        self.arena_len.0.store(0, Ordering::Relaxed);
    }
}

/// `a == b` for two slices of the same length of at least 16 bytes, in
/// 16-byte steps and one overlapping last step rather than a library call.
#[inline(always)]
fn same_long(a: &[u8], b: &[u8]) -> bool {
    let n = b.len();
    debug_assert!(a.len() == n && n >= 16);
    let at = |s: &[u8], i: usize| u128::from_ne_bytes(s[i..i + 16].try_into().unwrap());
    let mut i = 0;
    while i + 16 < n {
        if at(a, i) != at(b, i) {
            return false;
        }
        i += 16;
    }
    at(a, n - 16) == at(b, n - 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_long_finds_any_differing_byte() {
        let a: Vec<u8> = (0..80u8).map(|i| i.wrapping_mul(37)).collect();
        for n in 16..=80 {
            assert!(same_long(&a[..n], &a[..n]));
            for i in 0..n {
                let mut b = a[..n].to_vec();
                b[i] ^= 0x40;
                assert!(!same_long(&a[..n], &b), "{n} {i}");
            }
        }
    }

    #[test]
    fn insert_get_exact_and_clear() {
        let t = LongDerived::new();
        let a = "分词器把文本切成很多片段".as_bytes();
        assert!(t.get(a).is_none());
        assert!(t.insert(a, &[1, 2, 3]));
        let got = t.get(a).unwrap();
        let mut v = vec![];
        got.extend_into(&mut v);
        assert_eq!(v, [1, 2, 3]);
        // same samples, different middle: a distinct key
        let mut b = a.to_vec();
        b[14] ^= 1;
        assert!(t.get(&b).is_none());
        assert!(t.get(&a[..a.len() - 3]).is_none());
        t.clear();
        assert!(t.get(a).is_none());
    }

    #[test]
    fn full_table_refuses_and_keeps_answering() {
        let t = LongDerived::new();
        let mut kept = vec![];
        for i in 0..200_000u32 {
            let key = format!("{i:0>40}");
            if t.insert(key.as_bytes(), &[i; 20]) {
                kept.push(i);
            }
        }
        assert!(
            kept.len() > 10_000 && kept.len() < 200_000,
            "{}",
            kept.len()
        );
        for &i in kept.iter().step_by(97) {
            let key = format!("{i:0>40}");
            let mut v = vec![];
            t.get(key.as_bytes()).unwrap().extend_into(&mut v);
            assert_eq!(v, [i; 20]);
        }
    }
}
