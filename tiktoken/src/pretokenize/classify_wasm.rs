//! ASCII classifier for wasm32 builds with `simd128` enabled.

use std::arch::wasm32::*;

use super::classify::{BlockMasks, M_AP, M_CONT, M_L, M_LW, M_N, M_NL, M_SL, M_SP, M_W, MAPS};

#[inline(always)]
fn map4(v: &[v128; 4], f: impl Fn(v128) -> v128) -> u64 {
    let m = |x| u8x16_bitmask(f(x)) as u64;
    m(v[0]) | m(v[1]) << 16 | m(v[2]) << 32 | m(v[3]) << 48
}

#[inline(always)]
pub(super) fn block<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
    let p = block.as_ptr().cast::<v128>();
    // SAFETY: the loads read exactly the 64 bytes of `block`
    let v = unsafe {
        [
            v128_load(p),
            v128_load(p.add(1)),
            v128_load(p.add(2)),
            v128_load(p.add(3)),
        ]
    };
    let range = |c: v128, lo: u8, n: u8| u8x16_lt(u8x16_sub(c, u8x16_splat(lo)), u8x16_splat(n));
    let eq = |c: v128, b: u8| u8x16_eq(c, u8x16_splat(b));
    let mut out = [0u64; MAPS];
    if CASE {
        out[M_L] = map4(&v, |c| range(c, b'A', 26));
        out[M_LW] = map4(&v, |c| range(c, b'a', 26));
        out[M_SL] = map4(&v, |c| eq(c, b'/'));
    } else {
        out[M_L] = map4(&v, |c| range(v128_or(c, u8x16_splat(0x20)), b'a', 26));
    }
    out[M_N] = map4(&v, |c| range(c, b'0', 10));
    out[M_W] = map4(&v, |c| v128_or(range(c, 9, 5), eq(c, b' ')));
    out[M_NL] = map4(&v, |c| v128_or(eq(c, b'\n'), eq(c, b'\r')));
    out[M_SP] = map4(&v, |c| eq(c, b' '));
    out[M_AP] = map4(&v, |c| eq(c, b'\''));
    out[M_CONT] = map4(&v, |c| c);
    out
}
