//! Vector scans for cut candidates: a block of lanes at once, each lane set
//! where [`is_candidate`](super::is_candidate) would be, then [`is_cut`] on
//! the set lanes only. Lane `k` of the block at `i` is byte `i + k`; the block
//! reads `b[i - 1..i + LANES + 1]`.
//!
//! x86_64 uses AVX2 when the CPU has it and SSE2 (always there) otherwise;
//! aarch64 uses NEON. Other targets scan byte by byte in `mod.rs`.

use super::{Rules, is_cut};

/// The block's candidate lanes, `STRIDE` bits per lane.
type Mask = u64;

/// Forward from `i`: `Ok(cut)`, or `Err(next byte to look at)`.
#[inline(always)]
fn next_with<const LANES: usize, const STRIDE: u32>(
    b: &[u8],
    mut i: usize,
    rules: Rules,
    mask: impl Fn(&[u8], usize) -> Mask,
) -> Result<usize, usize> {
    let keep = lane_bits::<STRIDE>();
    while i + LANES < b.len() {
        let mut m = mask(b, i) & keep;
        while m != 0 {
            let k = (m.trailing_zeros() / STRIDE) as usize;
            if is_cut(b, i + k, rules) {
                return Ok(i + k);
            }
            m &= m - 1;
        }
        i += LANES;
    }
    Err(i)
}

/// Backward from `j` inclusive: `Ok(cut)`, or `Err(last byte left to look at)`.
#[inline(always)]
fn prev_with<const LANES: usize, const STRIDE: u32>(
    b: &[u8],
    mut j: usize,
    rules: Rules,
    mask: impl Fn(&[u8], usize) -> Mask,
) -> Result<usize, usize> {
    // the last byte has no byte after it to load
    if j + 1 >= b.len() {
        if is_cut(b, j, rules) {
            return Ok(j);
        }
        j -= 1;
    }
    let keep = lane_bits::<STRIDE>();
    while j > LANES {
        let i = j + 1 - LANES;
        let mut m = mask(b, i) & keep;
        while m != 0 {
            let top = 63 - m.leading_zeros();
            let k = (top / STRIDE) as usize;
            if is_cut(b, i + k, rules) {
                return Ok(i + k);
            }
            m &= !(1 << top);
        }
        j = i - 1;
    }
    Err(j)
}

/// One bit per lane, the lowest of its `STRIDE`.
const fn lane_bits<const STRIDE: u32>() -> Mask {
    let mut m: Mask = 0;
    let mut k = 0;
    while k < 64 {
        m |= 1 << k;
        k += STRIDE;
    }
    m
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use std::arch::x86_64::*;

    use super::{Mask, Rules, next_with, prev_with};

    /// # Safety
    ///
    /// `1 <= i` and `i + 17 <= b.len()`.
    #[inline(always)]
    unsafe fn mask_sse2(b: &[u8], i: usize) -> Mask {
        debug_assert!(i >= 1 && i + 17 <= b.len());
        // SAFETY: sse2 is part of the x86_64 baseline; the caller keeps the
        // three loads inside `b`
        unsafe {
            let p = b.as_ptr().add(i);
            let prv = _mm_loadu_si128(p.sub(1).cast());
            let cur = _mm_loadu_si128(p.cast());
            let nxt = _mm_loadu_si128(p.add(1).cast());
            let eq = |v: __m128i, c: u8| _mm_cmpeq_epi8(v, _mm_set1_epi8(c as i8));
            let blank = _mm_or_si128(
                _mm_or_si128(eq(cur, b' '), eq(cur, b'\t')),
                _mm_or_si128(eq(cur, b'\n'), eq(prv, b'\n')),
            );
            let cjk = _mm_or_si128(
                _mm_and_si128(eq(cur, 0xE3), eq(nxt, 0x80)),
                _mm_and_si128(eq(cur, 0xEF), eq(nxt, 0xBC)),
            );
            _mm_movemask_epi8(_mm_or_si128(blank, cjk)) as u32 as Mask
        }
    }

    /// # Safety
    ///
    /// The CPU must support AVX2; `1 <= i` and `i + 33 <= b.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mask_avx2(b: &[u8], i: usize) -> Mask {
        debug_assert!(i >= 1 && i + 33 <= b.len());
        // SAFETY: the caller checked for AVX2 and keeps the loads inside `b`
        unsafe {
            let p = b.as_ptr().add(i);
            let prv = _mm256_loadu_si256(p.sub(1).cast());
            let cur = _mm256_loadu_si256(p.cast());
            let nxt = _mm256_loadu_si256(p.add(1).cast());
            let eq = |v: __m256i, c: u8| _mm256_cmpeq_epi8(v, _mm256_set1_epi8(c as i8));
            let blank = _mm256_or_si256(
                _mm256_or_si256(eq(cur, b' '), eq(cur, b'\t')),
                _mm256_or_si256(eq(cur, b'\n'), eq(prv, b'\n')),
            );
            let cjk = _mm256_or_si256(
                _mm256_and_si256(eq(cur, 0xE3), eq(nxt, 0x80)),
                _mm256_and_si256(eq(cur, 0xEF), eq(nxt, 0xBC)),
            );
            _mm256_movemask_epi8(_mm256_or_si256(blank, cjk)) as u32 as Mask
        }
    }

    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn next_avx2(b: &[u8], i: usize, rules: Rules) -> Result<usize, usize> {
        // SAFETY: `next_with` keeps `i + 32 < b.len()`; the CPU has AVX2
        next_with::<32, 1>(b, i, rules, |b, i| unsafe { mask_avx2(b, i) })
    }

    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn prev_avx2(b: &[u8], j: usize, rules: Rules) -> Result<usize, usize> {
        // SAFETY: `prev_with` keeps `1 <= i` and `i + 32 < b.len()`
        prev_with::<32, 1>(b, j, rules, |b, i| unsafe { mask_avx2(b, i) })
    }

    /// The SSE2 scans, which a CPU with AVX2 never takes otherwise.
    #[cfg(test)]
    pub(crate) fn next_sse2(b: &[u8], i: usize, rules: Rules) -> Result<usize, usize> {
        // SAFETY: as in `next`
        next_with::<16, 1>(b, i, rules, |b, i| unsafe { mask_sse2(b, i) })
    }

    #[cfg(test)]
    pub(crate) fn prev_sse2(b: &[u8], j: usize, rules: Rules) -> Result<usize, usize> {
        // SAFETY: as in `prev`
        prev_with::<16, 1>(b, j, rules, |b, i| unsafe { mask_sse2(b, i) })
    }

    pub(crate) fn next(b: &[u8], i: usize, rules: Rules) -> Result<usize, usize> {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2
            return unsafe { next_avx2(b, i, rules) };
        }
        // SAFETY: `next_with` keeps `i + 16 < b.len()`, and `i >= 1`
        next_with::<16, 1>(b, i, rules, |b, i| unsafe { mask_sse2(b, i) })
    }

    pub(crate) fn prev(b: &[u8], j: usize, rules: Rules) -> Result<usize, usize> {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2
            return unsafe { prev_avx2(b, j, rules) };
        }
        // SAFETY: `prev_with` keeps `1 <= i` and `i + 16 < b.len()`
        prev_with::<16, 1>(b, j, rules, |b, i| unsafe { mask_sse2(b, i) })
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::aarch64::*;

    use super::{Mask, Rules, next_with, prev_with};

    /// Four bits per lane.
    ///
    /// # Safety
    ///
    /// `1 <= i` and `i + 17 <= b.len()`.
    #[inline(always)]
    unsafe fn mask_neon(b: &[u8], i: usize) -> Mask {
        debug_assert!(i >= 1 && i + 17 <= b.len());
        // SAFETY: neon is always available on aarch64; the caller keeps the
        // three loads inside `b`
        unsafe {
            let p = b.as_ptr().add(i);
            let prv = vld1q_u8(p.sub(1));
            let cur = vld1q_u8(p);
            let nxt = vld1q_u8(p.add(1));
            let eq = |v: uint8x16_t, c: u8| vceqq_u8(v, vdupq_n_u8(c));
            let blank = vorrq_u8(
                vorrq_u8(eq(cur, b' '), eq(cur, b'\t')),
                vorrq_u8(eq(cur, b'\n'), eq(prv, b'\n')),
            );
            let cjk = vorrq_u8(
                vandq_u8(eq(cur, 0xE3), eq(nxt, 0x80)),
                vandq_u8(eq(cur, 0xEF), eq(nxt, 0xBC)),
            );
            let m = vorrq_u8(blank, cjk);
            // a narrowing shift keeps one nibble per lane, in lane order
            let n = vshrn_n_u16::<4>(vreinterpretq_u16_u8(m));
            vget_lane_u64::<0>(vreinterpret_u64_u8(n))
        }
    }

    pub(crate) fn next(b: &[u8], i: usize, rules: Rules) -> Result<usize, usize> {
        // SAFETY: `next_with` keeps `i + 16 < b.len()`, and `i >= 1`
        next_with::<16, 4>(b, i, rules, |b, i| unsafe { mask_neon(b, i) })
    }

    pub(crate) fn prev(b: &[u8], j: usize, rules: Rules) -> Result<usize, usize> {
        // SAFETY: `prev_with` keeps `1 <= i` and `i + 16 < b.len()`
        prev_with::<16, 4>(b, j, rules, |b, i| unsafe { mask_neon(b, i) })
    }
}

pub(crate) use arch::{next, prev};
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use arch::{next_sse2, prev_sse2};
