//! ASCII classifier for x86_64 (SSE2, part of the baseline).

use std::arch::x86_64::*;

use super::classify::{BlockMasks, M_AP, M_CONT, M_L, M_LW, M_N, M_NL, M_SL, M_SP, M_W, MAPS};

#[inline(always)]
fn map4(v: &[__m128i; 4], f: impl Fn(__m128i) -> __m128i) -> u64 {
    // SAFETY: sse2 is part of the x86_64 baseline
    unsafe {
        let m = |x| _mm_movemask_epi8(f(x)) as u32 as u64;
        m(v[0]) | m(v[1]) << 16 | m(v[2]) << 32 | m(v[3]) << 48
    }
}

#[inline(always)]
pub(super) fn block<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
    // SAFETY: sse2 is part of the x86_64 baseline; the loads read exactly the
    // 64 bytes of `block`
    unsafe {
        let p = block.as_ptr().cast::<__m128i>();
        let v = [
            _mm_loadu_si128(p),
            _mm_loadu_si128(p.add(1)),
            _mm_loadu_si128(p.add(2)),
            _mm_loadu_si128(p.add(3)),
        ];
        // c in lo..lo + n: shift the range to the bottom of the signed bytes
        let range = |c: __m128i, lo: u8, n: u8| {
            let x = _mm_add_epi8(c, _mm_set1_epi8(0x80u8.wrapping_sub(lo) as i8));
            _mm_cmplt_epi8(x, _mm_set1_epi8((0x80u8 + n) as i8))
        };
        let eq = |c: __m128i, b: u8| _mm_cmpeq_epi8(c, _mm_set1_epi8(b as i8));
        let mut out = [0u64; MAPS];
        if CASE {
            out[M_L] = map4(&v, |c| range(c, b'A', 26));
            out[M_LW] = map4(&v, |c| range(c, b'a', 26));
            out[M_SL] = map4(&v, |c| eq(c, b'/'));
        } else {
            out[M_L] = map4(&v, |c| {
                range(_mm_or_si128(c, _mm_set1_epi8(0x20)), b'a', 26)
            });
        }
        out[M_N] = map4(&v, |c| range(c, b'0', 10));
        out[M_W] = map4(&v, |c| _mm_or_si128(range(c, 9, 5), eq(c, b' ')));
        out[M_NL] = map4(&v, |c| _mm_or_si128(eq(c, b'\n'), eq(c, b'\r')));
        out[M_SP] = map4(&v, |c| eq(c, b' '));
        out[M_AP] = map4(&v, |c| eq(c, b'\''));
        out[M_CONT] = map4(&v, |c| c);
        out
    }
}
