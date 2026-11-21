//! Unicode character classes for the pre-tokenizers.
//!
//! Every code point maps to one category byte through a two-level table
//! (`tables.rs`, generated from the `regex` crate's own Unicode data by the
//! `generate` test module, so the scanners agree with the patterns' `\p{..}`
//! classes code point for code point):
//!
//! - bits 0..4: the general-category group ([`LU`], [`LL`], [`LT`], [`LM`],
//!   [`LO`], [`MARK`], [`NUM`], [`PUNCT`], [`SYM`], [`SEP`], [`NL`] or [`OTHER`])
//! - bit 4 ([`HAN`]): `Script=Han`
//! - bit 5 ([`WS`]): `White_Space`, which is what `\s` matches
//!
//! Each pattern family turns a category byte into the class bits its rules
//! test through one 256-entry map ([`FamilyMap`]).

mod tables;

#[cfg(test)]
mod generate;
#[cfg(test)]
mod tests;

// the scanners never test for these two groups by name
#[cfg(test)]
pub(crate) const OTHER: u8 = 0;
pub(crate) const LU: u8 = 1;
pub(crate) const LL: u8 = 2;
pub(crate) const LT: u8 = 3;
pub(crate) const LM: u8 = 4;
pub(crate) const LO: u8 = 5;
pub(crate) const MARK: u8 = 6;
pub(crate) const NUM: u8 = 7;
pub(crate) const PUNCT: u8 = 8;
pub(crate) const SYM: u8 = 9;
/// `Zs`, `Zl`, `Zp`
#[cfg(test)]
pub(crate) const SEP: u8 = 10;
/// `\r` and `\n` only (both also carry [`WS`])
pub(crate) const NL: u8 = 11;
pub(crate) const GROUP_MASK: u8 = 0x0F;
pub(crate) const HAN: u8 = 0x10;
pub(crate) const WS: u8 = 0x20;

/// Category byte of a code point. `cp` must be a Unicode scalar value.
#[inline(always)]
pub(crate) const fn category(cp: u32) -> u8 {
    let block = tables::STAGE1[(cp >> tables::SHIFT) as usize] as usize;
    tables::STAGE2[(block << tables::SHIFT) | (cp as usize & tables::LOW_MASK)]
}

/// Decode the char starting at byte `i` of valid UTF-8, returning its code
/// point and byte length.
#[inline(always)]
pub(crate) fn decode(b: &[u8], i: usize) -> (u32, usize) {
    let c0 = b[i];
    if c0 < 0x80 {
        (c0 as u32, 1)
    } else if c0 < 0xE0 {
        ((((c0 & 0x1F) as u32) << 6) | (b[i + 1] & 0x3F) as u32, 2)
    } else if c0 < 0xF0 {
        (
            (((c0 & 0x0F) as u32) << 12)
                | (((b[i + 1] & 0x3F) as u32) << 6)
                | (b[i + 2] & 0x3F) as u32,
            3,
        )
    } else {
        (
            (((c0 & 0x07) as u32) << 18)
                | (((b[i + 1] & 0x3F) as u32) << 12)
                | (((b[i + 2] & 0x3F) as u32) << 6)
                | (b[i + 3] & 0x3F) as u32,
            4,
        )
    }
}

/// Code point of the char that ends right before byte `i` (`i > 0`).
#[inline]
pub(crate) fn decode_before(b: &[u8], i: usize) -> u32 {
    let mut s = i - 1;
    while b[s] & 0xC0 == 0x80 {
        s -= 1;
    }
    decode(b, s).0
}

/// Per-family class bits, indexed by category byte.
pub(crate) type FamilyMap = [u8; 256];

/// The family's word-letter class (`\p{L}`, or `[\p{L}\p{M}]` where the
/// pattern says so).
pub(crate) const C_LETTER: u8 = 1;
/// `\p{N}`
pub(crate) const C_NUM: u8 = 2;
/// `\s`
pub(crate) const C_WS: u8 = 4;
/// Case-split families: the upper class `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`.
pub(crate) const C_UPPER: u8 = 8;
/// Case-split families: the lower class `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
pub(crate) const C_LOWER: u8 = 16;
/// The punctuation rule's run class (`[^\s\p{L}\p{N}]` and its variants).
pub(crate) const C_PUNCT: u8 = 32;
/// The optional leading char of the word rules (`[^\r\n\p{L}\p{N}]` and its
/// variants).
pub(crate) const C_PREFIX: u8 = 64;
/// `\p{Han}`
pub(crate) const C_HAN: u8 = 128;

#[derive(Clone, Copy)]
struct MapSpec {
    /// word letters also take `\p{M}`
    letter_mark: bool,
    /// punctuation is `[\p{P}\p{S}]` and the prefix excludes `\p{P}\p{S}`
    /// instead of `\p{N}` (deepseek)
    punct_ps: bool,
    /// the punctuation class also excludes `\p{M}` (qwen3_5)
    punct_no_mark: bool,
    /// Han is excluded from the case-split classes (kimi)
    han_apart: bool,
}

const fn build_map(s: MapSpec) -> FamilyMap {
    let mut m = [0u8; 256];
    let mut c = 0;
    while c < 256 {
        let cat = c as u8;
        let g = cat & GROUP_MASK;
        let han = cat & HAN != 0;
        let ws = cat & WS != 0;
        let is_l = matches!(g, LU | LL | LT | LM | LO);
        let is_m = g == MARK;
        let is_n = g == NUM;
        let is_ps = g == PUNCT || g == SYM;
        let mut f = 0u8;
        if is_l || (s.letter_mark && is_m) {
            f |= C_LETTER;
        }
        if is_n {
            f |= C_NUM;
        }
        if ws {
            f |= C_WS;
        }
        let case_ok = !(s.han_apart && han);
        if case_ok && matches!(g, LU | LT | LM | LO | MARK) {
            f |= C_UPPER;
        }
        if case_ok && matches!(g, LL | LM | LO | MARK) {
            f |= C_LOWER;
        }
        let punct = if s.punct_ps {
            is_ps
        } else {
            !ws && !is_l && !is_n && !(s.punct_no_mark && is_m)
        };
        if punct {
            f |= C_PUNCT;
        }
        let prefix = g != NL && !is_l && if s.punct_ps { !is_ps } else { !is_n };
        if prefix {
            f |= C_PREFIX;
        }
        if han {
            f |= C_HAN;
        }
        m[c] = f;
        c += 1;
    }
    m
}

const BASE: MapSpec = MapSpec {
    letter_mark: false,
    punct_ps: false,
    punct_no_mark: false,
    han_apart: false,
};

/// cl100k_base, llama3, glm4, glm5, qwen2, p50k_base, r50k_base, and the
/// o200k-shaped families other than kimi.
pub(crate) static MAP_BASE: FamilyMap = build_map(BASE);
/// qwen3_5: `[\p{L}\p{M}]+` words, punctuation without marks.
pub(crate) static MAP_QWEN35: FamilyMap = build_map(MapSpec {
    letter_mark: true,
    punct_no_mark: true,
    ..BASE
});
/// kimi_k2, kimi_k3: Han excluded from the case-split classes.
pub(crate) static MAP_KIMI: FamilyMap = build_map(MapSpec {
    han_apart: true,
    ..BASE
});
/// deepseek_v3, deepseek_v4.
pub(crate) static MAP_DEEPSEEK: FamilyMap = build_map(MapSpec {
    letter_mark: true,
    punct_ps: true,
    ..BASE
});

/// Class bits and byte length of the char at byte `i`.
#[inline(always)]
pub(crate) fn class_at(map: &FamilyMap, b: &[u8], i: usize) -> (u8, usize) {
    let c0 = b[i];
    if c0 < 0x80 {
        return (map[tables::ASCII[c0 as usize] as usize], 1);
    }
    let (cp, len) = decode(b, i);
    (map[category(cp) as usize], len)
}

/// One bit per code point below 0x10000, set when its class bits under a
/// family map give `want` under `mask`.
pub(crate) type BmpBits = [u64; 1024];

const fn bmp_bits(map: &FamilyMap, mask: u8, want: u8) -> BmpBits {
    let mut t = [0u64; 1024];
    let mut cp = 0;
    while cp < 0x10000 {
        if map[category(cp as u32) as usize] & mask == want {
            t[cp / 64] |= 1 << (cp % 64);
        }
        cp += 1;
    }
    t
}

/// Letters under [`MAP_BASE`].
pub(crate) static LETTERS: BmpBits = bmp_bits(&MAP_BASE, C_LETTER, C_LETTER);
/// Letters under [`MAP_QWEN35`].
pub(crate) static LETTERS_QWEN35: BmpBits = bmp_bits(&MAP_QWEN35, C_LETTER, C_LETTER);
/// Chars in both case classes that cannot lead a word, under [`MAP_BASE`].
pub(crate) static BOTH_CASES: BmpBits =
    bmp_bits(&MAP_BASE, C_UPPER | C_LOWER | C_PREFIX, C_UPPER | C_LOWER);
