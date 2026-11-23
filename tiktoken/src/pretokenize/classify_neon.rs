//! ASCII classifier for aarch64.
//!
//! `vld4q_u8` loads the block de-interleaved (lane `i` of vector `k` is byte
//! `4i + k`); the bit-select reduction below interleaves the four compare
//! results back, so the narrowed 64-bit mask is in byte order.

use std::arch::aarch64::*;

use super::classify::{BlockMasks, M_AP, M_CONT, M_L, M_LW, M_N, M_NL, M_SL, M_SP, M_W, MAPS};

/// `t` with the bits selected by `mask` taken from `a` (one `bit`).
#[inline(always)]
unsafe fn bit(t: uint8x16_t, a: uint8x16_t, mask: uint8x16_t) -> uint8x16_t {
    let mut t = t;
    // SAFETY: a register-only neon instruction; written as asm because llvm
    // turns `vbslq_u8` on compare results back into and/or pairs
    unsafe {
        std::arch::asm!(
            "bit {t:v}.16b, {a:v}.16b, {m:v}.16b",
            t = inout(vreg) t,
            a = in(vreg) a,
            m = in(vreg) mask,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    t
}

#[inline(always)]
unsafe fn movemask(m: [uint8x16_t; 4]) -> u64 {
    // SAFETY: neon is always available on aarch64
    unsafe {
        // compare results are all ones or all zeros, so three selects put
        // bit k of both nibbles of lane i from m[k]; the narrowing shift then
        // takes the high nibble of lane 2j and the low nibble of lane 2j + 1
        let t = bit(m[0], m[1], vdupq_n_u8(0xEE));
        let t = bit(t, m[2], vdupq_n_u8(0xCC));
        let t = bit(t, m[3], vdupq_n_u8(0x88));
        let n = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t));
        vget_lane_u64::<0>(vreinterpret_u64_u8(n))
    }
}

#[inline(always)]
unsafe fn map4(v: &[uint8x16_t; 4], f: impl Fn(uint8x16_t) -> uint8x16_t) -> u64 {
    // SAFETY: as above
    unsafe { movemask([f(v[0]), f(v[1]), f(v[2]), f(v[3])]) }
}

#[inline(always)]
pub(super) fn block<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
    // SAFETY: neon is always available on aarch64, and the load reads
    // exactly the 64 bytes of `block`
    unsafe {
        let x = vld4q_u8(block.as_ptr());
        let v = [x.0, x.1, x.2, x.3];
        let range =
            |c: uint8x16_t, lo: u8, n: u8| vcltq_u8(vsubq_u8(c, vdupq_n_u8(lo)), vdupq_n_u8(n));
        let eq = |c: uint8x16_t, b: u8| vceqq_u8(c, vdupq_n_u8(b));
        let mut out = [0u64; MAPS];
        if CASE {
            out[M_L] = map4(&v, |c| range(c, b'A', 26));
            out[M_LW] = map4(&v, |c| range(c, b'a', 26));
            out[M_SL] = map4(&v, |c| eq(c, b'/'));
        } else {
            out[M_L] = map4(&v, |c| range(vorrq_u8(c, vdupq_n_u8(0x20)), b'a', 26));
        }
        // a table lookup answers bytes below 16 and gives 0 for the rest
        let ctl = |set: &[u8]| {
            let mut t = [0u8; 16];
            for &b in set {
                t[b as usize] = 0xFF;
            }
            vld1q_u8(t.as_ptr())
        };
        let (ws, nl) = (ctl(&[9, 10, 11, 12, 13]), ctl(b"\n\r"));
        let sp = [
            eq(v[0], b' '),
            eq(v[1], b' '),
            eq(v[2], b' '),
            eq(v[3], b' '),
        ];
        out[M_N] = map4(&v, |c| range(c, b'0', 10));
        out[M_W] = movemask([0, 1, 2, 3].map(|k| vorrq_u8(vqtbl1q_u8(ws, v[k]), sp[k])));
        out[M_NL] = map4(&v, |c| vqtbl1q_u8(nl, c));
        out[M_SP] = movemask(sp);
        out[M_AP] = map4(&v, |c| eq(c, b'\''));
        out[M_CONT] = map4(&v, |c| vcgeq_u8(c, vdupq_n_u8(0x80)));
        out
    }
}

/// Utf-8 continuation bytes (`0x80..0xC0`) of a block.
#[inline(always)]
pub(super) fn cont_mask(block: &[u8; 64]) -> u64 {
    // SAFETY: neon is always available on aarch64, and the load reads
    // exactly the 64 bytes of `block`
    unsafe {
        let x = vld4q_u8(block.as_ptr());
        let c = |v: uint8x16_t| vceqq_u8(vandq_u8(v, vdupq_n_u8(0xC0)), vdupq_n_u8(0x80));
        movemask([c(x.0), c(x.1), c(x.2), c(x.3)])
    }
}
