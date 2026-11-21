//! ASCII classifier for aarch64.
//!
//! `vld4q_u8` loads the block de-interleaved (lane `i` of vector `k` is byte
//! `4i + k`); the shift-insert reduction below interleaves the four compare
//! results back, so the narrowed 64-bit mask is in byte order.

use std::arch::aarch64::*;

use super::classify::{BlockMasks, M_AP, M_CONT, M_L, M_LW, M_N, M_NL, M_SL, M_SP, M_W, MAPS};

#[inline(always)]
unsafe fn movemask(m: [uint8x16_t; 4]) -> u64 {
    // SAFETY: neon is always available on aarch64
    unsafe {
        let t0 = vsriq_n_u8::<1>(m[1], m[0]);
        let t1 = vsriq_n_u8::<1>(m[3], m[2]);
        let t2 = vsriq_n_u8::<2>(t1, t0);
        let t3 = vsriq_n_u8::<4>(t2, t2);
        let t4 = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t3));
        vget_lane_u64::<0>(vreinterpret_u64_u8(t4))
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
        out[M_N] = map4(&v, |c| range(c, b'0', 10));
        out[M_W] = map4(&v, |c| vorrq_u8(range(c, 9, 5), eq(c, b' ')));
        out[M_NL] = map4(&v, |c| vorrq_u8(eq(c, b'\n'), eq(c, b'\r')));
        out[M_SP] = map4(&v, |c| eq(c, b' '));
        out[M_AP] = map4(&v, |c| eq(c, b'\''));
        out[M_CONT] = map4(&v, |c| vcgeq_u8(c, vdupq_n_u8(0x80)));
        out
    }
}
