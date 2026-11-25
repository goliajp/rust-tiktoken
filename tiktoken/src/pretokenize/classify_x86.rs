//! ASCII classifiers for x86_64: SSE2, part of the baseline, and AVX2, used
//! when the CPU has it.

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
pub(super) fn block_sse2<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
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

/// [`block_sse2`] 32 bytes at a time.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn block_avx2<const CASE: bool>(block: &[u8; 64]) -> BlockMasks {
    // SAFETY: the loads read exactly the 64 bytes of `block`
    let v = unsafe {
        let p = block.as_ptr().cast::<__m256i>();
        [_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1))]
    };
    let map2 = |f: &dyn Fn(__m256i) -> __m256i| {
        let m = |x| _mm256_movemask_epi8(f(x)) as u32 as u64;
        m(v[0]) | m(v[1]) << 32
    };
    let range = |c: __m256i, lo: u8, n: u8| {
        let x = _mm256_add_epi8(c, _mm256_set1_epi8(0x80u8.wrapping_sub(lo) as i8));
        _mm256_cmpgt_epi8(_mm256_set1_epi8((0x80u8 + n) as i8), x)
    };
    let eq = |c: __m256i, b: u8| _mm256_cmpeq_epi8(c, _mm256_set1_epi8(b as i8));
    let mut out = [0u64; MAPS];
    if CASE {
        out[M_L] = map2(&|c| range(c, b'A', 26));
        out[M_LW] = map2(&|c| range(c, b'a', 26));
        out[M_SL] = map2(&|c| eq(c, b'/'));
    } else {
        out[M_L] = map2(&|c| range(_mm256_or_si256(c, _mm256_set1_epi8(0x20)), b'a', 26));
    }
    out[M_N] = map2(&|c| range(c, b'0', 10));
    out[M_W] = map2(&|c| _mm256_or_si256(range(c, 9, 5), eq(c, b' ')));
    out[M_NL] = map2(&|c| _mm256_or_si256(eq(c, b'\n'), eq(c, b'\r')));
    out[M_SP] = map2(&|c| eq(c, b' '));
    out[M_AP] = map2(&|c| eq(c, b'\''));
    out[M_CONT] = map2(&|c| c);
    out
}
