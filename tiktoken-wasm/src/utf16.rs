//! UTF-16 to UTF-8, for text that is mostly outside ASCII: copying a JS
//! string's code units out by hand (about 0.9 ns a char in Chrome) and
//! converting them here takes less time than `TextEncoder.encodeInto`, which
//! spends about 1.6 ns on each such char. Lone surrogates become U+FFFD, as
//! `encodeInto` makes them.

/// Bytes [`transcode`] may write past the end of its output.
pub(crate) const SLACK: usize = 16;

/// Write the UTF-8 of `src` at `dst`; returns its length.
///
/// # Safety
///
/// `dst` has room for `3 * src.len() + SLACK` bytes.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) unsafe fn transcode(src: &[u16], dst: *mut u8) -> usize {
    // SAFETY: passed on from the caller
    unsafe { simd::transcode(src, dst) }
}

/// Write the UTF-8 of `src` at `dst`; returns its length.
///
/// # Safety
///
/// `dst` has room for `3 * src.len() + SLACK` bytes.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub(crate) unsafe fn transcode(src: &[u16], dst: *mut u8) -> usize {
    // SAFETY: passed on from the caller
    unsafe { scalar(src, 0, dst, 0).1 }
}

/// Convert from unit `i` and byte `o` on until the end of `src`, or until a
/// unit past `stop` when `stop` is smaller; returns the next unit and byte.
///
/// # Safety
///
/// `dst` has room for `o + 3 * (src.len() - i)` bytes.
#[inline(always)]
unsafe fn scalar_until(
    src: &[u16],
    mut i: usize,
    dst: *mut u8,
    mut o: usize,
    stop: usize,
) -> (usize, usize) {
    // SAFETY: each unit writes at most 3 bytes, a pair 4 for its 2 units
    unsafe {
        while i < src.len() && i < stop {
            let c = src[i] as u32;
            i += 1;
            if c < 0x80 {
                *dst.add(o) = c as u8;
                o += 1;
            } else if c < 0x800 {
                *dst.add(o) = 0xC0 | (c >> 6) as u8;
                *dst.add(o + 1) = 0x80 | (c & 0x3F) as u8;
                o += 2;
            } else if c & 0xF800 != 0xD800 {
                put3(dst.add(o), c);
                o += 3;
            } else if c < 0xDC00 && i < src.len() && src[i] & 0xFC00 == 0xDC00 {
                let cp = 0x10000 + ((c - 0xD800) << 10) + (src[i] as u32 - 0xDC00);
                i += 1;
                *dst.add(o) = 0xF0 | (cp >> 18) as u8;
                *dst.add(o + 1) = 0x80 | ((cp >> 12) & 0x3F) as u8;
                *dst.add(o + 2) = 0x80 | ((cp >> 6) & 0x3F) as u8;
                *dst.add(o + 3) = 0x80 | (cp & 0x3F) as u8;
                o += 4;
            } else {
                put3(dst.add(o), 0xFFFD);
                o += 3;
            }
        }
    }
    (i, o)
}

#[inline(always)]
unsafe fn put3(at: *mut u8, c: u32) {
    // SAFETY: passed on from the caller
    unsafe {
        *at = 0xE0 | (c >> 12) as u8;
        *at.add(1) = 0x80 | ((c >> 6) & 0x3F) as u8;
        *at.add(2) = 0x80 | (c & 0x3F) as u8;
    }
}

/// [`scalar_until`] to the end.
///
/// # Safety
///
/// As for [`scalar_until`].
unsafe fn scalar(src: &[u16], i: usize, dst: *mut u8, o: usize) -> (usize, usize) {
    // SAFETY: passed on from the caller
    unsafe { scalar_until(src, i, dst, o, usize::MAX) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use std::arch::wasm32::*;

    /// For four chars of 1 to 3 bytes, indexed by the mask of those of 2 or
    /// more bytes plus 16 times the mask of those of 3: the swizzle that packs
    /// their bytes out of a vector holding the four first bytes, then the four
    /// second bytes, then the four third bytes; and the packed length.
    struct Pack {
        shuffle: [[u8; 16]; 256],
        len: [u8; 256],
    }

    const PACK: Pack = {
        let mut shuffle = [[0x80u8; 16]; 256];
        let mut len = [0u8; 256];
        let mut idx = 0;
        while idx < 256 {
            let mut pos = 0;
            let mut lane = 0;
            while lane < 4 {
                let n = 1 + (idx >> lane & 1) + (idx >> (lane + 4) & 1);
                let mut k = 0;
                while k < n {
                    shuffle[idx][pos] = (4 * k + lane) as u8;
                    pos += 1;
                    k += 1;
                }
                lane += 1;
            }
            len[idx] = pos as u8;
            idx += 1;
        }
        Pack { shuffle, len }
    };

    /// # Safety
    ///
    /// `dst` has room for 16 bytes from `o`.
    #[inline(always)]
    unsafe fn pack(bytes: v128, idx: usize, dst: *mut u8, o: usize) -> usize {
        // SAFETY: passed on from the caller; the table rows are 16 bytes
        unsafe {
            let s = v128_load(PACK.shuffle[idx].as_ptr() as *const v128);
            v128_store(dst.add(o) as *mut v128, i8x16_swizzle(bytes, s));
        }
        o + PACK.len[idx] as usize
    }

    /// # Safety
    ///
    /// As for [`super::transcode`].
    pub(super) unsafe fn transcode(src: &[u16], dst: *mut u8) -> usize {
        let n = src.len();
        let mut i = 0;
        let mut o = 0;
        let low6 = |v: v128| v128_or(v128_and(v, u16x8_splat(0x3F)), u16x8_splat(0x80));
        // SAFETY: loads stay inside `src`; a group of 8 units stores up to 16
        // bytes at most 12 bytes past 3 bytes per unit read before it
        unsafe {
            while i + 8 <= n {
                let v = v128_load(src.as_ptr().add(i) as *const v128);
                let m2 = u16x8_ge(v, u16x8_splat(0x80));
                if !v128_any_true(m2) {
                    v128_store64_lane::<0>(u8x16_narrow_i16x8(v, v), dst.add(o) as *mut u64);
                    i += 8;
                    o += 8;
                    continue;
                }
                let sur = u16x8_eq(v128_and(v, u16x8_splat(0xF800)), u16x8_splat(0xD800));
                if v128_any_true(sur) {
                    // a pair may straddle the group's end
                    (i, o) = super::scalar_until(src, i, dst, o, i + 8);
                    continue;
                }
                let m3 = u16x8_ge(v, u16x8_splat(0x800));
                let last = low6(v);
                let lead3 = v128_or(u16x8_shr(v, 12), u16x8_splat(0xE0));
                if i16x8_all_true(m3) {
                    // eight chars of 3 bytes, the usual group in CJK text
                    let fs = u8x16_narrow_i16x8(lead3, low6(u16x8_shr(v, 6)));
                    let t = u8x16_narrow_i16x8(last, last);
                    let a = i8x16_shuffle::<0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5>(
                        fs, t,
                    );
                    let b = i8x16_shuffle::<13, 21, 6, 14, 22, 7, 15, 23, 0, 0, 0, 0, 0, 0, 0, 0>(
                        fs, t,
                    );
                    v128_store(dst.add(o) as *mut v128, a);
                    v128_store64_lane::<0>(b, dst.add(o + 16) as *mut u64);
                    i += 8;
                    o += 24;
                    continue;
                }
                let lead2 = v128_or(u16x8_shr(v, 6), u16x8_splat(0xC0));
                let first = v128_bitselect(lead3, v128_bitselect(lead2, v, m2), m3);
                let second = v128_bitselect(low6(u16x8_shr(v, 6)), last, m3);
                let fs = u8x16_narrow_i16x8(first, second);
                let t = u8x16_narrow_i16x8(last, last);
                let lo =
                    i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 0, 0, 0, 0>(fs, t);
                let hi =
                    i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 0, 0, 0, 0>(fs, t);
                let b2 = i16x8_bitmask(m2) as usize;
                let b3 = i16x8_bitmask(m3) as usize;
                o = pack(lo, (b2 & 15) | (b3 & 15) << 4, dst, o);
                o = pack(hi, b2 >> 4 | (b3 >> 4) << 4, dst, o);
                i += 8;
            }
            super::scalar(src, i, dst, o).1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(units: &[u16]) {
        let mut out = vec![0u8; 3 * units.len() + SLACK];
        // SAFETY: sized as required
        let n = unsafe { transcode(units, out.as_mut_ptr()) };
        assert_eq!(
            &out[..n],
            String::from_utf16_lossy(units).as_bytes(),
            "{units:x?}"
        );
    }

    #[test]
    fn matches_lossy_utf16_decoding() {
        let mut x: u32 = 1;
        let mut next = || {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            x >> 8
        };
        let picks: [&dyn Fn(u32) -> u16; 6] = [
            &|r| (r % 0x80) as u16,
            &|r| (0x80 + r % 0x780) as u16,
            &|r| (0x800 + r % 0xD000) as u16,
            &|r| (0xD800 + r % 0x400) as u16,
            &|r| (0xDC00 + r % 0x400) as u16,
            &|r| (0xE000 + r % 0x2000) as u16,
        ];
        for len in 0..80 {
            for _ in 0..200 {
                let units: Vec<u16> = (0..len)
                    .map(|_| {
                        let r = next();
                        picks[(r % 6) as usize](r / 8)
                    })
                    .collect();
                check(&units);
            }
        }
        check(&[0xD83D, 0xDE80, 0x41, 0xD83D]);
        check(&[0xDE80, 0xD83D]);
    }
}
