//! What the merge may assume about each 3-byte char before it starts.
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
//! One word per code point from U+0800 to U+FFFF says whether the char is an
//! atom, a wall and which parts it has, or neither, and which of its 1- and
//! 2-byte prefixes and suffixes some fragment shares, so that most
//! occurrences need no fragment check at all.

use crate::hash::FxHashMap;
use std::collections::HashMap;

const FIRST: u32 = 0x800;
const RANK_BITS: u32 = 24;
const RANK_MASK: u32 = (1 << RANK_BITS) - 1;
const KIND_SHIFT: u32 = 24;
// kinds
const PLAIN: u32 = 0;
const ATOM: u32 = 1;
const WALL: u32 = 2;
/// An atom whose fragments need the full check of [`Atoms::at`](super::Atoms::at).
const RULED: u32 = 3;
// wall parts: three bytes, the first two joined, or the last two joined
const SHAPE_SHIFT: u32 = 26;
const BYTES: u32 = 0;
const JOIN_FIRST: u32 = 1;
const JOIN_LAST: u32 = 2;
// some fragment shares the char's first byte, first two, last two, last one
const HEAD1: u32 = 1 << 28;
const HEAD2: u32 = 1 << 29;
const TAIL2: u32 = 1 << 30;
const TAIL1: u32 = 1 << 31;

/// What the table says about one char.
pub(crate) enum Char3 {
    Plain,
    Atom(u32),
    /// Look it up in the atoms' own table.
    Ruled,
    /// A wall where no fragment occurs; see [`Walls::parts`].
    Wall,
}

/// Code point of a 3-byte char, or `None` if the bytes are not one.
#[inline(always)]
pub(crate) fn cp3(b0: u8, b1: u8, b2: u8) -> Option<u32> {
    let cp = (u32::from(b0 & 0x0F) << 12) | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F);
    let ok = (b0 & 0xF0 == 0xE0) & (b1 & 0xC0 == 0x80) & (b2 & 0xC0 == 0x80) & (cp >= FIRST);
    ok.then_some(cp)
}

fn pack(b: &[u8]) -> u32 {
    let mut w = [0u8; 4];
    w[..b.len()].copy_from_slice(b);
    u32::from_le_bytes(w)
}

pub(crate) struct Walls {
    chars: Box<[u32]>,
    // fragments sharing one byte with a char, by that byte and the byte they
    // need next to the char: a bit per pair
    head1: Box<[u64; 1024]>,
    tail1: Box<[u64; 1024]>,
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

impl Walls {
    /// `heads` and `tails` map the bytes a fragment shares with a char to
    /// the fragments; `atoms` gives each atom of 3 bytes as (bytes, rank,
    /// ruled); `get` looks up a rank.
    pub(crate) fn build(
        heads: &HashMap<Vec<u8>, Vec<Box<[u8]>>>,
        tails: &HashMap<Vec<u8>, Vec<Box<[u8]>>>,
        atoms: impl Iterator<Item = (u32, u32, bool)>,
        get: &impl Fn(&[u8]) -> Option<u32>,
    ) -> Self {
        let mut head1 = Box::new([0u64; 1024]);
        let mut tail1 = Box::new([0u64; 1024]);
        let mut index = |src: &HashMap<Vec<u8>, Vec<Box<[u8]>>>, head: bool| {
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
                if k.len() == 1 {
                    for x in &v {
                        let (set, next) = if head {
                            (&mut head1, x[x.len() - 1])
                        } else {
                            (&mut tail1, x[0])
                        };
                        let i = (k[0] as usize) << 8 | next as usize;
                        set[i / 64] |= 1 << (i % 64);
                    }
                }
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
        for (cp, w) in (FIRST..0x10000).zip(chars.iter_mut()) {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let mut b = [0u8; 3];
            c.encode_utf8(&mut b);
            let r = |x: &[u8]| get(x).unwrap_or(u32::MAX);
            let (a, z) = (r(&b[..2]), r(&b[1..]));
            let shape = if a == u32::MAX && z == u32::MAX {
                Some((BYTES, 0))
            } else if get(&b).is_some() {
                None
            } else if a <= z {
                Some((JOIN_FIRST, a))
            } else {
                Some((JOIN_LAST, z))
            };
            let mut flags = 0;
            // head keys start with a lead byte and tail keys with a
            // continuation byte, so one set holds both
            for (k, flag) in [
                (&b[..1], HEAD1),
                (&b[..2], HEAD2),
                (&b[1..], TAIL2),
                (&b[2..], TAIL1),
            ] {
                if has(k) {
                    flags |= flag;
                }
            }
            if let Some((s, rank)) = shape {
                debug_assert!(rank <= RANK_MASK);
                *w = rank | WALL << KIND_SHIFT | s << SHAPE_SHIFT | flags;
            }
        }
        for (cp, rank, ruled) in atoms {
            debug_assert!(rank <= RANK_MASK);
            chars[(cp - FIRST) as usize] = if ruled {
                RULED << KIND_SHIFT
            } else {
                rank | ATOM << KIND_SHIFT
            };
        }
        Self {
            chars,
            head1,
            tail1,
            before,
            after,
        }
    }

    /// What the merge may assume about the char with code point `cp`, and
    /// its table word.
    #[inline(always)]
    pub(crate) fn get(&self, cp: u32) -> (Char3, u32) {
        let w = self.chars[(cp - FIRST) as usize];
        let kind = match (w >> KIND_SHIFT) & 3 {
            PLAIN => Char3::Plain,
            ATOM => Char3::Atom(w & RANK_MASK),
            WALL => Char3::Wall,
            _ => Char3::Ruled,
        };
        (kind, w)
    }

    /// The parts of a wall char at `piece[a..a + 3]` with table word `w`,
    /// as `(length, id)` with `single` giving byte ids, or `None` if a
    /// fragment occurs there.
    #[inline(always)]
    pub(crate) fn parts(
        &self,
        piece: &[u8],
        a: usize,
        w: u32,
        single: impl Fn(u8) -> u32,
    ) -> Option<[(u8, u32); 3]> {
        if w & (HEAD1 | HEAD2 | TAIL2 | TAIL1) != 0 && self.crossed(piece, a, w) {
            return None;
        }
        let b = &piece[a..a + 3];
        let rank = w & RANK_MASK;
        Some(match (w >> SHAPE_SHIFT) & 3 {
            BYTES => [(1, single(b[0])), (1, single(b[1])), (1, single(b[2]))],
            JOIN_FIRST => [(2, rank), (1, single(b[2])), (0, 0)],
            _ => [(1, single(b[0])), (2, rank), (0, 0)],
        })
    }

    #[inline]
    fn crossed(&self, piece: &[u8], a: usize, w: u32) -> bool {
        let e = a + 3;
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
        if a > 0 {
            let prev = piece[a - 1];
            if w & HEAD1 != 0 && bit(&self.head1, piece[a], prev) && ends(&piece[a..a + 1]) {
                return true;
            }
            if w & HEAD2 != 0 && ends(&piece[a..a + 2]) {
                return true;
            }
        }
        if e < piece.len() {
            let next = piece[e];
            if w & TAIL1 != 0 && bit(&self.tail1, piece[e - 1], next) && starts(&piece[e - 1..e]) {
                return true;
            }
            if w & TAIL2 != 0 && starts(&piece[a + 1..e]) {
                return true;
            }
        }
        false
    }
}
