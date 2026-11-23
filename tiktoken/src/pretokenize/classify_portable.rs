//! The portable ASCII classifier: the reference the SIMD classifiers are
//! tested against, and the classifier of targets without one.

use super::classify::{BlockMasks, M_AP, M_CONT, M_L, M_LW, M_N, M_NL, M_SL, M_SP, M_W, MAPS};

/// Class bits of an ASCII byte, in map order.
pub(super) const fn ascii_bits(c: u8, case: bool) -> u8 {
    let mut b = 0u8;
    if c.is_ascii_uppercase() || (!case && c.is_ascii_lowercase()) {
        b |= 1 << M_L;
    }
    if case && c.is_ascii_lowercase() {
        b |= 1 << M_LW;
    }
    if c.is_ascii_digit() {
        b |= 1 << M_N;
    }
    if matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r') {
        b |= 1 << M_W;
    }
    if c == b'\n' || c == b'\r' {
        b |= 1 << M_NL;
    }
    if c == b' ' {
        b |= 1 << M_SP;
    }
    if c == b'\'' {
        b |= 1 << M_AP;
    }
    if case && c == b'/' {
        b |= 1 << M_SL;
    }
    b
}

const fn ascii_table(case: bool) -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 128 {
        t[c] = ascii_bits(c as u8, case);
        c += 1;
    }
    t
}

static ASCII_CL100K: [u8; 256] = ascii_table(false);
static ASCII_CASE: [u8; 256] = ascii_table(true);

/// Bit 0 of each byte of `x` gathered into the low 8 bits, byte `k` to bit `k`.
#[inline(always)]
const fn gather(x: u64) -> u64 {
    ((x & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080)) >> 56
}

/// The reference classifier: one table lookup per byte, eight bytes gathered
/// per map with a multiply.
pub(crate) fn block_portable<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
    let table = if CASE { &ASCII_CASE } else { &ASCII_CL100K };
    let mut out = [0u64; MAPS];
    for (g, eight) in block.as_chunks::<8>().0.iter().enumerate() {
        let mut cls = 0u64;
        for (i, &c) in eight.iter().enumerate() {
            cls |= (table[c as usize] as u64) << (8 * i);
        }
        let raw = u64::from_le_bytes(*eight);
        for (map, o) in out.iter_mut().enumerate().take(M_SL + 1) {
            *o |= gather(cls >> map) << (8 * g);
        }
        out[M_CONT] |= gather(raw >> 7) << (8 * g);
    }
    out
}
