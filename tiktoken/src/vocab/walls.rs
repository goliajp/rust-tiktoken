//! What the merge may assume about each 2- and 3-byte char before it starts.
//!
//! A char that merges alone into two or more parts is a wall at an
//! occurrence where no fragment (a token covering part of the char and bytes
//! outside it, see [`Atoms`](super::Atoms)) occurs: no merge ever joins one
//! of its parts with a byte outside it, since that part would be such a
//! fragment or the whole char, which never forms. So the merge may start
//! from the char's own parts and never pair them with their neighbours; its
//! inner merges and those outside run as independent sequences, as for
//! atoms, with no condition on ranks.
//!
//! One word per code point from U+0080 to U+FFFF says whether the char is an
//! atom (with its rank), a wall and which parts it has, or neither, and
//! which of its 1- and 2-byte prefixes and suffixes some fragment shares.
//! Most occurrences then need one more test: a fragment that covers part of
//! a char and bytes before it contains the byte pair across the char's
//! start, and one that covers bytes after it the pair across its end; one
//! bit per byte pair says whether any fragment has such a pair there.

use crate::hash::FxHashMap;
use std::collections::HashMap;

const FIRST: u32 = 0x80;
const RANK_BITS: u32 = 24;
const RANK_MASK: u32 = (1 << RANK_BITS) - 1;
const KIND_SHIFT: u32 = 24;
// kinds
const PLAIN: u32 = 0;
const ATOM: u32 = 1;
const WALL: u32 = 2;
// wall parts: every byte, the first two joined, or the last two joined
const SHAPE_SHIFT: u32 = 26;
const BYTES: u32 = 0;
const JOIN_FIRST: u32 = 1;
const JOIN_LAST: u32 = 2;
// some fragment shares the char's first byte, first two, last two, last one
const HEAD1: u32 = 1 << 28;
const HEAD2: u32 = 1 << 29;
const TAIL2: u32 = 1 << 30;
const TAIL1: u32 = 1 << 31;
const HEADS: u32 = HEAD1 | HEAD2;
const TAILS: u32 = TAIL1 | TAIL2;

/// How the merge may start from one char.
pub(crate) enum Start {
    /// The char as one part with this id.
    Whole(u32),
    /// Its parts as `(length, id)`, ended by a zero length; they never
    /// merge with anything outside the char.
    Wall([(u8, u32); 3]),
    /// Its bytes, like any other.
    Bytes,
}

/// The table word of one char: an atom, a wall, or neither.
#[derive(Clone, Copy)]
pub(crate) struct CharWord(u32);

impl CharWord {
    /// Neither atom nor wall: the merge starts from the char's bytes.
    #[inline(always)]
    pub(crate) fn is_plain(self) -> bool {
        self.0 & (3 << KIND_SHIFT) == PLAIN << KIND_SHIFT
    }

    /// The atom's id.
    #[inline(always)]
    pub(crate) fn atom(self) -> Option<u32> {
        (self.0 & (3 << KIND_SHIFT) == ATOM << KIND_SHIFT).then_some(self.0 & RANK_MASK)
    }

    /// A wall's parts: the id of its first two bytes joined, if they are,
    /// and of its last two joined, if they are; otherwise its bytes.
    #[inline(always)]
    pub(crate) fn joined(self) -> (Option<u32>, Option<u32>) {
        let r = self.0 & RANK_MASK;
        match (self.0 >> SHAPE_SHIFT) & 3 {
            JOIN_FIRST => (Some(r), None),
            JOIN_LAST => (None, Some(r)),
            _ => (None, None),
        }
    }
}

/// Code point of the 2- or 3-byte char at `piece[a..a + l]`, or `None` if
/// the bytes there are not one.
#[inline(always)]
pub(crate) fn cp_at(piece: &[u8], a: usize, l: usize) -> Option<u32> {
    match (l, piece.get(a..a + l)?) {
        (2, &[b0, b1]) => {
            let ok = (b0 & 0xE0 == 0xC0) & (b0 >= 0xC2) & (b1 & 0xC0 == 0x80);
            ok.then_some((u32::from(b0 & 0x1F) << 6) | u32::from(b1 & 0x3F))
        }
        (3, &[b0, b1, b2]) => cp3(b0, b1, b2),
        _ => None,
    }
}

/// Code point of a 3-byte char, or `None` if the bytes are not one.
#[inline(always)]
pub(crate) fn cp3(b0: u8, b1: u8, b2: u8) -> Option<u32> {
    let cp = (u32::from(b0 & 0x0F) << 12) | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F);
    let ok = (b0 & 0xF0 == 0xE0) & (b1 & 0xC0 == 0x80) & (b2 & 0xC0 == 0x80) & (cp >= 0x800);
    ok.then_some(cp)
}

fn pack(b: &[u8]) -> u32 {
    let mut w = [0u8; 4];
    w[..b.len()].copy_from_slice(b);
    u32::from_le_bytes(w)
}

pub(crate) struct Walls {
    chars: Box<[u32]>,
    // byte pairs across a char edge inside some fragment: (lead byte of the
    // char it ends in, byte before) and (last byte it takes from a char,
    // byte after)
    head_pairs: Box<[u64; 1024]>,
    tail_pairs: Box<[u64; 1024]>,
    // the bytes fragments need outside the char, by the 1 or 2 bytes they
    // share with it (keys of `before` start with a lead byte, of `after`
    // with a continuation byte)
    before: FxHashMap<u32, Box<[Box<[u8]>]>>,
    after: FxHashMap<u32, Box<[Box<[u8]>]>>,
}

#[inline(always)]
fn bit(set: &[u64; 1024], a: u8, b: u8) -> bool {
    let i = (a as usize) << 8 | b as usize;
    set[i / 64] >> (i % 64) & 1 != 0
}

fn set_bit(set: &mut [u64; 1024], a: u8, b: u8) {
    let i = (a as usize) << 8 | b as usize;
    set[i / 64] |= 1 << (i % 64);
}

impl Walls {
    /// `heads` and `tails` map the bytes a fragment shares with a char to
    /// the fragments; `atoms` gives each atom of 2 or 3 bytes as (code
    /// point, rank); `get` looks up a rank.
    pub(crate) fn build(
        heads: &HashMap<Vec<u8>, Vec<Box<[u8]>>>,
        tails: &HashMap<Vec<u8>, Vec<Box<[u8]>>>,
        atoms: impl Iterator<Item = (u32, u32)>,
        get: &impl Fn(&[u8]) -> Option<u32>,
    ) -> Self {
        let mut head_pairs = Box::new([0u64; 1024]);
        let mut tail_pairs = Box::new([0u64; 1024]);
        for (k, ts) in heads {
            for t in ts {
                set_bit(&mut head_pairs, k[0], t[t.len() - k.len() - 1]);
            }
        }
        for (k, ts) in tails {
            for t in ts {
                set_bit(&mut tail_pairs, k[k.len() - 1], t[k.len()]);
            }
        }
        let index = |src: &HashMap<Vec<u8>, Vec<Box<[u8]>>>, head: bool| {
            let mut out = FxHashMap::default();
            for (k, ts) in src {
                if k.len() > 2 {
                    continue;
                }
                let mut v: Vec<Box<[u8]>> = ts
                    .iter()
                    .map(|t| {
                        if head {
                            t[..t.len() - k.len()].into()
                        } else {
                            t[k.len()..].into()
                        }
                    })
                    .collect();
                v.sort();
                v.dedup();
                out.insert(pack(k), v.into_boxed_slice());
            }
            out
        };
        let before = index(heads, true);
        let after = index(tails, false);
        // the 1- and 2-byte keys present, by their bytes
        let mut keys = vec![0u64; 1024 + 4].into_boxed_slice();
        for k in heads.keys().chain(tails.keys()) {
            let i = match k[..] {
                [a] => 65536 + a as usize,
                [a, b] => (a as usize) << 8 | b as usize,
                _ => continue,
            };
            keys[i / 64] |= 1 << (i % 64);
        }
        let has = |k: &[u8]| {
            let i = if k.len() == 1 {
                65536 + k[0] as usize
            } else {
                (k[0] as usize) << 8 | k[1] as usize
            };
            keys[i / 64] >> (i % 64) & 1 != 0
        };
        let mut chars = vec![0u32; (0x10000 - FIRST) as usize].into_boxed_slice();
        let mut atom_of = vec![u32::MAX; chars.len()];
        for (cp, rank) in atoms {
            debug_assert!(rank <= RANK_MASK);
            atom_of[(cp - FIRST) as usize] = rank;
        }
        for (cp, w) in (FIRST..0x10000).zip(chars.iter_mut()) {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let mut buf = [0u8; 3];
            let b = c.encode_utf8(&mut buf).as_bytes();
            let n = b.len();
            // head keys start with a lead byte and tail keys with a
            // continuation byte, so one set holds both
            let mut flags = 0;
            for (k, flag) in [(&b[..1], HEAD1), (&b[n - 1..], TAIL1)] {
                if has(k) {
                    flags |= flag;
                }
            }
            if n == 3 {
                for (k, flag) in [(&b[..2], HEAD2), (&b[1..], TAIL2)] {
                    if has(k) {
                        flags |= flag;
                    }
                }
            }
            let atom = atom_of[(cp - FIRST) as usize];
            if atom != u32::MAX {
                *w = atom | ATOM << KIND_SHIFT | flags;
                continue;
            }
            if get(b).is_some() {
                continue;
            }
            let (shape, rank) = if n == 2 {
                (BYTES, 0)
            } else {
                let r = |x: &[u8]| get(x).unwrap_or(u32::MAX);
                let (a, z) = (r(&b[..2]), r(&b[1..]));
                if a == u32::MAX && z == u32::MAX {
                    (BYTES, 0)
                } else if a <= z {
                    (JOIN_FIRST, a)
                } else {
                    (JOIN_LAST, z)
                }
            };
            debug_assert!(rank <= RANK_MASK);
            *w = rank | WALL << KIND_SHIFT | shape << SHAPE_SHIFT | flags;
        }
        Self {
            chars,
            head_pairs,
            tail_pairs,
            before,
            after,
        }
    }

    /// The table word of a 2- or 3-byte char by code point.
    #[inline(always)]
    pub(crate) fn word(&self, cp: u32) -> CharWord {
        debug_assert!((FIRST..0x10000).contains(&cp));
        // SAFETY: a 2- or 3-byte char's code point is in FIRST..0x10000
        CharWord(*unsafe { self.chars.get_unchecked((cp - FIRST) as usize) })
    }

    /// How the merge may start from the char with code point `cp` at
    /// `piece[a..a + l]`, with `single` giving byte ids.
    #[inline(always)]
    pub(crate) fn start(
        &self,
        piece: &[u8],
        a: usize,
        l: usize,
        cp: u32,
        single: impl Fn(u8) -> u32,
    ) -> Start {
        let w = self.word(cp);
        if w.is_plain() || !self.clear(piece, a, l, w) {
            return Start::Bytes;
        }
        if let Some(r) = w.atom() {
            return Start::Whole(r);
        }
        let b = &piece[a..a + l];
        let rank = w.0 & RANK_MASK;
        Start::Wall(match ((w.0 >> SHAPE_SHIFT) & 3, l) {
            (BYTES, 2) => [(1, single(b[0])), (1, single(b[1])), (0, 0)],
            (BYTES, _) => [(1, single(b[0])), (1, single(b[1])), (1, single(b[2]))],
            (JOIN_FIRST, _) => [(2, rank), (1, single(b[2])), (0, 0)],
            _ => [(1, single(b[0])), (2, rank), (0, 0)],
        })
    }

    /// Whether no fragment occurs over the char at `piece[a..a + l]` with
    /// table word `w`, so that the word applies there.
    #[inline(always)]
    pub(crate) fn clear(&self, piece: &[u8], a: usize, l: usize, w: CharWord) -> bool {
        w.0 & (HEADS | TAILS) == 0 || !self.crossed(piece, a, l, w.0)
    }

    /// Whether a fragment occurs over the char at `piece[a..a + l]` with
    /// table word `w`.
    #[inline(always)]
    fn crossed(&self, piece: &[u8], a: usize, l: usize, w: u32) -> bool {
        let e = a + l;
        let head = w & HEADS != 0 && a > 0 && bit(&self.head_pairs, piece[a], piece[a - 1]);
        let tail =
            w & TAILS != 0 && e < piece.len() && bit(&self.tail_pairs, piece[e - 1], piece[e]);
        (head || tail) && self.crossed_exactly(piece, a, e, w)
    }

    #[cold]
    #[inline(never)]
    fn crossed_exactly(&self, piece: &[u8], a: usize, e: usize, w: u32) -> bool {
        let ends = |key: &[u8]| {
            self.before
                .get(&pack(key))
                .is_some_and(|v| v.iter().any(|x| piece[..a].ends_with(x)))
        };
        let starts = |key: &[u8]| {
            self.after
                .get(&pack(key))
                .is_some_and(|v| v.iter().any(|y| piece[e..].starts_with(y)))
        };
        (w & HEAD1 != 0 && ends(&piece[a..a + 1]))
            || (w & HEAD2 != 0 && ends(&piece[a..a + 2]))
            || (w & TAIL1 != 0 && starts(&piece[e - 1..e]))
            || (w & TAIL2 != 0 && starts(&piece[e - 2..e]))
    }
}
