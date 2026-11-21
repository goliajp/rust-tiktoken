//! A small word-at-a-time hasher for the crate's internal maps (special
//! tokens, the vocabulary while it is being built). Not DoS resistant: the
//! keys come from the embedded vocabularies, never from callers.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0xf135_7aea_2e62_a9c5;

#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher {
    h: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.h = (self.h.rotate_left(5) ^ w).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks {
            self.add(u64::from_le_bytes(*c));
        }
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(w) ^ ((rest.len() as u64) << 59));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // the multiply leaves the low bits weak; hashbrown indexes with them
        let h = self.h;
        h ^ (h >> 29) ^ (h >> 47)
    }
}

pub(crate) type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn distinct_short_keys_hash_apart() {
        let b = BuildHasherDefault::<FxHasher>::default();
        let keys: [&[u8]; 6] = [b"", b"\0", b"\0\0", b"a", b"ab", b"abcdefgh\0"];
        let hs: Vec<u64> = keys.iter().map(|k| b.hash_one(k)).collect();
        for i in 0..hs.len() {
            for j in i + 1..hs.len() {
                assert_ne!(hs[i], hs[j], "{:?} vs {:?}", keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn map_round_trips() {
        let mut m: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for i in 0..10_000u32 {
            m.insert(i.to_le_bytes().to_vec(), i);
        }
        for i in 0..10_000u32 {
            assert_eq!(m[&i.to_le_bytes().to_vec()], i);
        }
    }
}
