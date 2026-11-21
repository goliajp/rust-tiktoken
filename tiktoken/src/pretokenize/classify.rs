//! Per-byte class bitmaps for the window scanners.
//!
//! A run of 64-byte blocks is classified into one bitmap per class, bit `i`
//! of a map standing for byte `i` of the run. ASCII bytes are classified by
//! the instruction-set specific code (`classify_*.rs`, with the portable
//! version below as the reference). Non-ASCII chars are then looked up in the
//! unicode tables one by one, and their class is written on every byte of the
//! char, so a multibyte char is a run of equal bits like any other.

use std::mem::MaybeUninit;

use crate::unicode::{GROUP_MASK, LL, LM, LO, LT, LU, MARK, NUM, WS, category, decode};

/// Bytes classified at a time in the body of a text.
pub(crate) const CHUNK: usize = 4096;
/// Words per map for a [`CHUNK`], plus the guard word read by the unaligned
/// window loads.
pub(crate) const BODY_WORDS: usize = CHUNK / 64 + 1;

/// Letters (cl100k family) or upper-case letters (case-split family).
pub(crate) const M_L: usize = 0;
/// Lower-case letters (case-split family only).
pub(crate) const M_LW: usize = 1;
/// ASCII digits.
pub(crate) const M_N: usize = 2;
/// `\s`.
pub(crate) const M_W: usize = 3;
/// `\r` and `\n`.
pub(crate) const M_NL: usize = 4;
/// `' '`.
pub(crate) const M_SP: usize = 5;
/// `'\''`.
pub(crate) const M_AP: usize = 6;
/// `'/'` (case-split family only).
pub(crate) const M_SL: usize = 7;
/// Bytes the window rules do not decide.
pub(crate) const M_HARD: usize = 8;
/// Utf-8 continuation bytes. The ASCII classifiers store the high-bit mask
/// here; [`fixup`] replaces it.
pub(crate) const M_CONT: usize = 9;
pub(crate) const MAPS: usize = 10;

/// Window class of a non-ASCII char: letter (cl100k) or upper (case-split).
pub(crate) const X_L: u8 = 1;
/// Lower-case letter (case-split family).
pub(crate) const X_LW: u8 = 2;
/// `\s`.
pub(crate) const X_W: u8 = 4;
/// Not decided by the window rules.
pub(crate) const X_HARD: u8 = 8;

/// Window classes of the non-ASCII chars, indexed by category byte.
pub(crate) type ClassTable = [u8; 256];

const fn class_table(case: bool, mark_letter: bool) -> ClassTable {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 256 {
        let g = c as u8 & GROUP_MASK;
        let mut x = 0;
        if c as u8 & WS != 0 {
            x |= X_W;
        }
        if g == NUM || (mark_letter && g == MARK) {
            x |= X_HARD;
        }
        if case {
            match g {
                LU | LT => x |= X_L,
                LL => x |= X_LW,
                // in both case classes; the window rules only know the two
                LM | LO | MARK => x |= X_HARD,
                _ => {}
            }
        } else if matches!(g, LU | LL | LT | LM | LO) {
            x |= X_L;
        }
        t[c] = x;
        c += 1;
    }
    t
}

/// cl100k_base, llama3, glm4, glm5, qwen2.
pub(crate) static CLASS_CL100K: ClassTable = class_table(false, false);
/// qwen3_5: a mark is both a word letter and a possible leading char.
pub(crate) static CLASS_QWEN35: ClassTable = class_table(false, true);
/// o200k_base, minimax_m2, llama4, mistral_v3.
pub(crate) static CLASS_CASE: ClassTable = class_table(true, false);

/// Bitmaps of `W - 1` blocks of 64 bytes and a guard word. Only the words a
/// classification wrote are ever read.
pub(crate) struct Bitmaps<const W: usize> {
    words: MaybeUninit<[[u64; W]; MAPS]>,
}

impl<const W: usize> Bitmaps<W> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            words: MaybeUninit::uninit(),
        }
    }

    #[inline(always)]
    fn ptr(&mut self, map: usize) -> *mut u64 {
        debug_assert!(map < MAPS);
        // SAFETY: in bounds of the array; no reference to uninit memory is made
        unsafe { self.words.as_mut_ptr().cast::<u64>().add(map * W) }
    }

    /// Word `k` of `map`.
    #[inline(always)]
    pub(crate) fn word(&self, map: usize, k: usize) -> u64 {
        debug_assert!(k < W);
        // SAFETY: callers only read words a classification wrote
        unsafe { self.words.as_ptr().cast::<u64>().add(map * W + k).read() }
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, map: usize, k: usize, v: u64) {
        debug_assert!(k < W);
        // SAFETY: in bounds
        unsafe { self.ptr(map).add(k).write(v) }
    }

    /// 64 bits of `map` from bit `r`, the top `r % 8` of them zero.
    #[inline(always)]
    pub(crate) fn window(&self, map: usize, r: usize) -> u64 {
        debug_assert!(r / 64 + 1 < W);
        // SAFETY: the 8 bytes from `r / 8` lie in words `r / 64` and
        // `r / 64 + 1`, both written (the second may be the guard word)
        let w = unsafe {
            self.words
                .as_ptr()
                .cast::<u64>()
                .add(map * W)
                .cast::<u8>()
                .add(r >> 3)
                .cast::<u64>()
                .read_unaligned()
        };
        u64::from_le(w) >> (r & 7)
    }

    #[inline(always)]
    pub(crate) fn store_word(&mut self, map: usize, k: usize, v: u64) {
        self.set(map, k, v.to_le());
    }

    #[inline(always)]
    fn load_word(&self, map: usize, k: usize) -> u64 {
        u64::from_le(self.word(map, k))
    }
}

/// ASCII masks of one 64-byte block, in map order. `CASE` selects the
/// case-split family's maps (upper and lower letters, slash); otherwise
/// [`M_L`] holds all letters and [`M_LW`] and [`M_SL`] are left zero.
pub(crate) type BlockMasks = [u64; MAPS];

/// ASCII masks of one block with the best classifier for the target.
#[inline(always)]
fn block<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
    #[cfg(target_arch = "aarch64")]
    {
        super::classify_neon::block::<CASE>(block)
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::classify_x86::block::<CASE>(block)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        super::classify_wasm::block::<CASE>(block)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        super::classify_portable::block_portable::<CASE>(block)
    }
}

/// Classify `text[start..start + 64 * blocks]` into `maps`, with non-ASCII
/// chars from `table`. Returns whether any byte is non-ASCII.
///
/// `text` must hold the whole of every char that starts in the range, and
/// `start` must be a char boundary.
#[inline(always)]
pub(crate) fn classify<const CASE: bool, const W: usize>(
    table: &ClassTable,
    text: &[u8],
    start: usize,
    blocks: usize,
    maps: &mut Bitmaps<W>,
) -> bool {
    debug_assert!(blocks < W && start + 64 * blocks <= text.len());
    let mut any_hi = 0;
    for k in 0..blocks {
        let at = start + 64 * k;
        let b: &[u8; 64] = text[at..at + 64].try_into().unwrap();
        let m = block::<CASE>(b);
        for (map, &v) in m.iter().enumerate() {
            if CASE || (map != M_LW && map != M_SL) {
                maps.store_word(map, k, v);
            }
        }
        any_hi |= m[M_CONT];
    }
    for map in 0..MAPS {
        if CASE || (map != M_LW && map != M_SL) {
            maps.store_word(map, blocks, 0);
        }
    }
    if any_hi == 0 {
        return false;
    }
    fixup::<CASE, W>(table, text, start, blocks, maps);
    true
}

/// Write the class of every non-ASCII char on all of its bytes, and replace
/// the high-bit masks with continuation-byte masks.
#[inline(never)]
fn fixup<const CASE: bool, const W: usize>(
    table: &ClassTable,
    text: &[u8],
    start: usize,
    blocks: usize,
    maps: &mut Bitmaps<W>,
) {
    // bytes of word k that belong to a char started in word k - 1, as
    // [cont, l, lw, w, hard] bits
    let mut carry = [0u64; 5];
    for k in 0..blocks {
        let hi = maps.load_word(M_CONT, k);
        if hi == 0 {
            // no spill either: spilled bytes are continuation bytes
            continue;
        }
        let mut acc = [0u128; 5];
        for (a, &c) in acc.iter_mut().zip(carry.iter()) {
            *a = c as u128;
        }
        let mut m = hi & !carry[0];
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            let (cp, len) = decode(text, start + 64 * k + i);
            let x = table[category(cp) as usize];
            let bits = ((1u128 << len) - 1) << i;
            acc[0] |= bits & !(1u128 << i);
            if x & X_L != 0 {
                acc[1] |= bits;
            }
            if x & X_LW != 0 {
                acc[2] |= bits;
            }
            if x & X_W != 0 {
                acc[3] |= bits;
            }
            if x & X_HARD != 0 {
                acc[4] |= bits;
            }
            m &= !(bits as u64);
        }
        for (&map, &a) in [M_CONT, M_L, M_LW, M_W, M_HARD].iter().zip(acc.iter()) {
            if !CASE && map == M_LW {
                continue;
            }
            let lo = if map == M_CONT {
                a as u64
            } else {
                maps.load_word(map, k) | a as u64
            };
            maps.store_word(map, k, lo);
        }
        // a char across the end of the range spills into the guard word,
        // which no window reads as valid
        for (c, &a) in carry.iter_mut().zip(acc.iter()) {
            *c = (a >> 64) as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::classify_portable::{ascii_bits, block_portable};
    use super::*;

    fn random_text(seed: u64, len: usize) -> Vec<u8> {
        const CHARS: &[&str] = &[
            "a", "Z", "q", "0", "9", " ", "\t", "\n", "\r", "\x0b", "'", "/", ".", "~", "\x7f",
            "\0", "é", "É", "’", "世", "\u{3000}", "\u{85}", "😀", "\u{301}", "٣", "ʰ",
        ];
        let mut s = seed | 1;
        let mut out = Vec::new();
        while out.len() < len {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out.extend_from_slice(CHARS[(s % CHARS.len() as u64) as usize].as_bytes());
        }
        out
    }

    fn check_block<const CASE: bool>(b: &[u8; 64]) {
        assert_eq!(block::<CASE>(b), block_portable::<CASE>(b), "{b:?}");
    }

    #[test]
    fn simd_blocks_match_portable() {
        for seed in 0..2000 {
            let t = random_text(seed, 64);
            let b: &[u8; 64] = t[..64].try_into().unwrap();
            check_block::<false>(b);
            check_block::<true>(b);
        }
        // every byte value at every lane
        for v in 0..=255u8 {
            for lane in 0..64 {
                let mut b = [b'a'; 64];
                b[lane] = v;
                b[63 - lane] = v.wrapping_add(lane as u8);
                check_block::<false>(&b);
                check_block::<true>(&b);
            }
        }
    }

    #[test]
    fn portable_matches_bytewise_definition() {
        let t = random_text(7, 64);
        let b: &[u8; 64] = t[..64].try_into().unwrap();
        let m = block_portable::<true>(b);
        for (i, &c) in b.iter().enumerate() {
            let bits = if c < 0x80 { ascii_bits(c, true) } else { 0 };
            for (map, &w) in m.iter().enumerate().take(M_SL + 1) {
                assert_eq!(
                    w >> i & 1,
                    (bits >> map & 1) as u64,
                    "byte {c:#x} map {map}"
                );
            }
            assert_eq!(m[M_CONT] >> i & 1, (c >= 0x80) as u64);
        }
    }

    #[test]
    fn non_ascii_classes_cover_every_byte_of_a_char() {
        let text = "a\u{3000}é世’٣x".repeat(20);
        let bytes = text.as_bytes();
        let blocks = bytes.len() / 64;
        let mut maps = Bitmaps::<8>::new();
        assert!(classify::<false, 8>(
            &CLASS_CL100K,
            bytes,
            0,
            blocks,
            &mut maps
        ));
        let mut i = 0;
        for ch in text.chars() {
            let len = ch.len_utf8();
            for j in i..(i + len).min(64 * blocks) {
                let bit = |map| maps.word(map, j / 64) >> (j % 64) & 1;
                assert_eq!(bit(M_CONT), (j > i) as u64, "{ch:?} byte {j}");
                assert_eq!(bit(M_L), ch.is_alphabetic() as u64, "{ch:?}");
                assert_eq!(bit(M_W), ch.is_whitespace() as u64, "{ch:?}");
                assert_eq!(bit(M_HARD), (ch == '٣') as u64, "{ch:?}");
            }
            i += len;
        }
    }
}
