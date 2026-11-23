//! The bytes fragments need outside a char, by the one or two bytes they
//! share with it: the exact test behind the byte-pair filter of `walls`.
//!
//! Lists are reached by direct index and compared a word at a time: a few
//! hundred fragments per vocabulary, most keys with one to three.

use std::collections::HashMap;

/// One fragment's bytes outside the char: up to the 8 nearest packed in a
/// word as they lie in the text, and the length of all of them.
struct Outside {
    word: u64,
    mask: u64,
    len: usize,
    // all the bytes, for fragments needing more than 8
    bytes: Box<[u8]>,
}

/// Which side of the char the outside bytes lie on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Side {
    /// Before the char: keys are its first one or two bytes.
    Before,
    /// After it: keys are its last one or two bytes.
    After,
}

pub(super) struct Frags {
    side: Side,
    // range of `list` by key slot
    index: Box<[(u16, u16)]>,
    list: Box<[Outside]>,
}

const SLOTS: usize = 256 + 64 * 64;

/// Slot of a 1- or 2-byte key; the second byte of a 2-byte key is a
/// continuation byte, and so is the first unless it leads a char, so the
/// low six bits of each tell keys apart within their side.
#[inline(always)]
fn slot(key: &[u8]) -> usize {
    match *key {
        [a] => a as usize,
        [a, b] => 256 + ((a as usize & 0x3F) << 6 | (b as usize & 0x3F)),
        _ => unreachable!("keys have one or two bytes"),
    }
}

/// `piece[a - 8..a]` as a little-endian word, zero below the piece's start.
#[inline(always)]
fn word_before(piece: &[u8], a: usize) -> u64 {
    if a >= 8 {
        return u64::from_le_bytes(piece[a - 8..a].try_into().unwrap());
    }
    if piece.len() >= 8 {
        let w = u64::from_le_bytes(piece[..8].try_into().unwrap());
        // a < 8: the first `a` bytes move to the top
        return if a == 0 { 0 } else { w << (8 * (8 - a)) };
    }
    piece[..a]
        .iter()
        .fold(0, |w, &b| w >> 8 | u64::from(b) << 56)
}

/// `piece[e..e + 8]` as a little-endian word, zero past the piece's end.
#[inline(always)]
fn word_after(piece: &[u8], e: usize) -> u64 {
    let n = piece.len();
    if e + 8 <= n {
        return u64::from_le_bytes(piece[e..e + 8].try_into().unwrap());
    }
    if n >= 8 {
        let w = u64::from_le_bytes(piece[n - 8..].try_into().unwrap());
        // e > n - 8: the last `n - e` bytes move to the bottom
        return if e == n { 0 } else { w >> (8 * (e + 8 - n)) };
    }
    piece[e..]
        .iter()
        .rev()
        .fold(0, |w, &b| w << 8 | u64::from(b))
}

impl Frags {
    /// From the fragments by the bytes they share with a char (keys of more
    /// than two bytes are left out, as the callers never ask for them).
    pub(super) fn build(src: &HashMap<Vec<u8>, Vec<Box<[u8]>>>, side: Side) -> Self {
        let mut by_slot: Vec<Vec<Box<[u8]>>> = (0..SLOTS).map(|_| Vec::new()).collect();
        for (k, ts) in src {
            if k.len() > 2 {
                continue;
            }
            debug_assert!(
                k.iter()
                    .skip(usize::from(side == Side::Before))
                    .all(|&b| b & 0xC0 == 0x80)
            );
            debug_assert!(side == Side::After || k[0] >= 0xC0);
            let v = &mut by_slot[slot(k)];
            for t in ts {
                v.push(match side {
                    Side::Before => t[..t.len() - k.len()].into(),
                    Side::After => t[k.len()..].into(),
                });
            }
            v.sort();
            v.dedup();
        }
        let mut index = vec![(0u16, 0u16); SLOTS].into_boxed_slice();
        let mut list = Vec::new();
        for (s, v) in by_slot.into_iter().enumerate() {
            let start = list.len();
            for bytes in v {
                let len = bytes.len();
                let near = len.min(8);
                // the `near` bytes next to the char, where the text's word
                // has them: at the top before it, at the bottom after it
                let (word, mask) = match side {
                    Side::Before => {
                        let w = bytes[len - near..]
                            .iter()
                            .fold(0u64, |w, &b| w >> 8 | u64::from(b) << 56);
                        (w, u64::MAX << (8 * (8 - near)))
                    }
                    Side::After => {
                        let w = bytes[..near]
                            .iter()
                            .rev()
                            .fold(0u64, |w, &b| w << 8 | u64::from(b));
                        (w, u64::MAX >> (8 * (8 - near)))
                    }
                };
                list.push(Outside {
                    word,
                    mask,
                    len,
                    bytes,
                });
            }
            index[s] = (start as u16, list.len() as u16);
        }
        assert!(list.len() <= u16::MAX as usize);
        Self {
            side,
            index,
            list: list.into_boxed_slice(),
        }
    }

    /// Whether a fragment keyed by `key` occurs at `at` in `piece`: its
    /// outside bytes end at `at` (before) or start there (after).
    #[inline(always)]
    pub(super) fn occurs(&self, key: &[u8], piece: &[u8], at: usize) -> bool {
        let (lo, hi) = self.index[slot(key)];
        if lo == hi {
            return false;
        }
        let w = match self.side {
            Side::Before => word_before(piece, at),
            Side::After => word_after(piece, at),
        };
        let room = match self.side {
            Side::Before => at,
            Side::After => piece.len() - at,
        };
        self.list[lo as usize..hi as usize].iter().any(|o| {
            o.len <= room
                && w & o.mask == o.word
                && (o.len <= 8
                    || match self.side {
                        Side::Before => piece[..at].ends_with(&o.bytes),
                        Side::After => piece[at..].starts_with(&o.bytes),
                    })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // keys of either side with outside byte strings of 1 to 18 bytes,
    // tested at every position of random pieces against the plain
    // comparison
    #[test]
    fn occurs_matches_slice_comparison() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        // a small alphabet, so that fragments occur often
        let alpha = [0x20u8, 0x41, 0x80, 0x9F, 0xBF, 0xC3, 0xE4, 0xE5];
        for side in [Side::Before, Side::After] {
            let keys: Vec<Vec<u8>> = match side {
                Side::Before => vec![vec![0xE4], vec![0xC3], vec![0xE4, 0x80], vec![0xE5, 0xBF]],
                Side::After => vec![vec![0x80], vec![0xBF], vec![0x80, 0x9F], vec![0xBF, 0xBF]],
            };
            let mut src: HashMap<Vec<u8>, Vec<Box<[u8]>>> = HashMap::new();
            let mut outside: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();
            for k in &keys {
                for _ in 0..6 {
                    let len = 1 + next() as usize % 18;
                    let o: Vec<u8> = (0..len).map(|_| alpha[next() as usize % 8]).collect();
                    let t = match side {
                        Side::Before => [&o[..], k].concat(),
                        Side::After => [k, &o[..]].concat(),
                    };
                    src.entry(k.clone()).or_default().push(t.into());
                    outside.entry(k.clone()).or_default().push(o);
                }
            }
            let f = Frags::build(&src, side);
            for _ in 0..3000 {
                let n = 1 + next() as usize % 24;
                let piece: Vec<u8> = (0..n).map(|_| alpha[next() as usize % 8]).collect();
                for at in 0..=n {
                    for k in &keys {
                        let want = outside[k].iter().any(|o| match side {
                            Side::Before => piece[..at].ends_with(o),
                            Side::After => piece[at..].starts_with(o),
                        });
                        assert_eq!(f.occurs(k, &piece, at), want, "{piece:x?} {at} {k:x?}");
                    }
                }
            }
        }
    }
}
