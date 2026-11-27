//! Merge results of pieces longer than the packed key, shared by all threads.
//!
//! CJK prose clauses are the main entries: they are longer than 15 bytes,
//! rarely tokens, and repeat. A slot is one word, `tag << 32 | offset + 1`
//! (0 = empty), in a bucket of eight that shares a cache line; the record at
//! the arena offset is `[len | n << 16] [key bytes, padded to 4] [ids]`.
//! A hit compares the tag, then the length and bytes exactly.
//!
//! Encoding with token ends also keeps, for a record it has seen, the end of
//! each id's token in the piece (at most 255 bytes): a byte in a second arena,
//! at the id's word offset, written there by the first such encode to meet
//! the record. Its first byte goes last, with `Release`; 0 means not yet.
//! Inserts never touch that arena, so plain encoding does not pay for it.
//!
//! Publication: a writer reserves its record with one `fetch_add` on the
//! arena length, writes it, then claims an empty slot with a `Release`
//! compare-exchange of the whole slot word. Readers load the slot with
//! `Acquire`, so the record they then read is complete. Records never change
//! after that, until [`LongDerived::clear`], which needs exclusive use. A full
//! bucket or arena refuses further entries.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use super::copy_ids;
use super::key::hash_long;

/// Longest piece kept.
pub(crate) const LONG_KEY_MAX: usize = 255;
const BUCKET: usize = 8;
// wasm linear memory has no lazily mapped zero pages: the whole arena counts
// from the first long piece on, for every encoding in use
#[cfg(not(target_arch = "wasm32"))]
const BUCKETS: usize = 1 << 13;
#[cfg(not(target_arch = "wasm32"))]
const ARENA_WORDS: usize = 1 << 20;
#[cfg(target_arch = "wasm32")]
const BUCKETS: usize = 1 << 10;
#[cfg(target_arch = "wasm32")]
const ARENA_WORDS: usize = 1 << 17;

#[repr(C, align(64))]
struct Bucket([AtomicU64; BUCKET]);

pub(crate) struct LongDerived {
    buckets: Box<[Bucket]>,
    arena: Box<[AtomicU32]>,
    /// Token ends, a byte per arena word plus 8 readable past the last;
    /// created by the first encode with ends.
    ends: OnceLock<Box<[AtomicU8]>>,
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
            ends: OnceLock::new(),
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
        self.get_hashed(piece, hash_long(piece))
    }

    /// [`get`](Self::get) with the piece's [`hash_long`] already computed.
    #[inline]
    pub(crate) fn get_hashed(&self, piece: &[u8], h: u64) -> Option<LongIds<'_>> {
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

    /// Publish the ids of `piece` (16 to [`LONG_KEY_MAX`] bytes), whose
    /// [`hash_long`] is `h`. False when the table refuses it; the caller
    /// keeps the result elsewhere.
    pub(crate) fn insert(&self, piece: &[u8], ids: &[u32], h: u64) -> bool {
        self.publish(piece, ids, h).is_some()
    }

    /// [`insert`](Self::insert), returning the published ids.
    pub(crate) fn insert_ids(&self, piece: &[u8], ids: &[u32], h: u64) -> Option<LongIds<'_>> {
        let at = self.publish(piece, ids, h)?;
        Some(LongIds(&self.arena[at..at + ids.len()]))
    }

    /// Publish a record; the arena word of its first id, or `None` when
    /// refused.
    #[inline(always)]
    fn publish(&self, piece: &[u8], ids: &[u32], h: u64) -> Option<usize> {
        debug_assert!(piece.len() > 15 && piece.len() <= LONG_KEY_MAX && ids.len() <= piece.len());
        debug_assert_eq!(h, hash_long(piece));
        let bucket = self.bucket(h);
        if bucket.0[BUCKET - 1].load(Ordering::Relaxed) != 0 {
            return None;
        }
        let key_words = piece.len().div_ceil(4);
        let need = 1 + key_words + ids.len();
        let off = self.arena_len.0.fetch_add(need as u32, Ordering::Relaxed) as usize;
        if off + need > ARENA_WORDS {
            // keep the length from creeping towards overflow
            self.arena_len
                .0
                .store(ARENA_WORDS as u32, Ordering::Relaxed);
            return None;
        }
        // SAFETY: the words `off..off + need` were reserved above, no reader
        // sees them before the slot's Release below, and they are in bounds;
        // the key's padding bytes are never read
        unsafe {
            let rec = self.arena.as_ptr().cast::<u32>().cast_mut().add(off);
            rec.write(piece.len() as u32 | (ids.len() as u32) << 16);
            copy_key(piece, rec.add(1).cast::<u8>());
            copy_ids(ids.as_ptr(), rec.add(1 + key_words), ids.len());
        }
        let v = tag(h) | (off as u64 + 1);
        for slot in &bucket.0 {
            if slot
                .compare_exchange(0, v, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return Some(off + 1 + key_words);
            }
        }
        None
    }

    /// Where the token ends of `ids`, which come from this table, are kept,
    /// if they are; 8 bytes past the last are readable.
    #[inline(always)]
    pub(crate) fn ends_of(&self, ids: &LongIds<'_>) -> Option<*const AtomicU8> {
        let e = self.ends.get()?;
        let p = &e[self.word_of(ids)];
        (p.load(Ordering::Acquire) != 0).then_some(std::ptr::from_ref(p))
    }

    /// Keep the token ends of `ids`, which come from this table: the end of
    /// each id's token in its piece.
    #[cold]
    pub(crate) fn keep_ends(&self, ids: &LongIds<'_>, ends: &[u8]) {
        debug_assert_eq!(ends.len(), ids.len());
        let e = self
            .ends
            .get_or_init(|| (0..ARENA_WORDS + 8).map(|_| AtomicU8::new(0)).collect());
        let at = &e[self.word_of(ids)..];
        // every encode that keeps them writes the same bytes
        for (b, &x) in at[1..ends.len()].iter().zip(&ends[1..]) {
            b.store(x, Ordering::Relaxed);
        }
        at[0].store(ends[0], Ordering::Release);
    }

    #[inline(always)]
    fn word_of(&self, ids: &LongIds<'_>) -> usize {
        // SAFETY: the ids lie in the arena
        unsafe { ids.as_ptr().offset_from(self.arena.as_ptr().cast::<u32>()) as usize }
    }

    /// Drop every entry. Caller guarantees no concurrent use.
    pub(crate) fn clear(&self) {
        let used = self.arena_len.0.load(Ordering::Relaxed) as usize;
        if used == 0 {
            return;
        }
        // later records reuse these words: their ends are not known
        if let Some(e) = self.ends.get() {
            // SAFETY: no other thread uses the table, the bytes are in an
            // `UnsafeCell`, and the range is in bounds; one fill rather than
            // a store per byte
            unsafe {
                std::ptr::write_bytes(e.as_ptr().cast::<u8>().cast_mut(), 0, used.min(ARENA_WORDS));
            }
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

impl super::Vocab {
    /// Where the token ends of `ids`, from [`long_ids`](Self::long_ids), are
    /// kept, if they are; 8 bytes past the last are readable.
    #[inline(always)]
    pub(crate) fn long_ends(&self, ids: &LongIds<'_>) -> Option<*const AtomicU8> {
        self.long_derived.get()?.ends_of(ids)
    }

    /// Keep the token ends of `ids`, from [`long_ids`](Self::long_ids).
    pub(crate) fn keep_long_ends(&self, ids: &LongIds<'_>, ends: &[u8]) {
        if let Some(t) = self.long_derived.get() {
            t.keep_ends(ids, ends);
        }
    }
}

/// Copy a key of at least 16 bytes to `dst` in 16-byte blocks, the last
/// one overlapping: most keys take one or two blocks.
///
/// # Safety
///
/// `dst` has room for `key.len()` bytes.
#[inline(always)]
unsafe fn copy_key(key: &[u8], dst: *mut u8) {
    let n = key.len();
    debug_assert!(n >= 16);
    let block = |i: usize| u128::from_ne_bytes(key[i..i + 16].try_into().unwrap());
    // SAFETY: every block lies in dst[..n]
    unsafe {
        let mut i = 0;
        while i + 16 < n {
            dst.add(i).cast::<u128>().write_unaligned(block(i));
            i += 16;
        }
        dst.add(n - 16)
            .cast::<u128>()
            .write_unaligned(block(n - 16));
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
        assert!(t.insert(a, &[1, 2, 3], hash_long(a)));
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
    fn ends_are_kept_until_a_clear() {
        let t = LongDerived::new();
        let a = "分词器把文本切成很多片段".as_bytes();
        assert!(t.insert(a, &[1, 2, 3], hash_long(a)));
        assert!(t.ends_of(&t.get(a).unwrap()).is_none());
        t.keep_ends(&t.get(a).unwrap(), &[9, 20, 36]);
        let p = t.ends_of(&t.get(a).unwrap()).unwrap();
        // SAFETY: 8 bytes past the last end are readable
        let got: Vec<u8> = (0..3)
            .map(|j| unsafe { (*p.add(j)).load(Ordering::Relaxed) })
            .collect();
        assert_eq!(got, [9, 20, 36]);
        t.clear();
        let b = "另一段文本切成了很多片段".as_bytes();
        assert!(t.insert(b, &[4, 5, 6], hash_long(b)));
        // the same words, and no ends of the record before the clear
        assert!(t.ends_of(&t.get(b).unwrap()).is_none());
    }

    #[test]
    fn full_table_refuses_and_keeps_answering() {
        let t = LongDerived::new();
        let mut kept = vec![];
        for i in 0..200_000u32 {
            let key = format!("{i:0>40}");
            if t.insert(key.as_bytes(), &[i; 20], hash_long(key.as_bytes())) {
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
