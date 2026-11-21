//! Packed keys for byte strings of up to [`FAST_KEY_MAX`] bytes.
//!
//! A key is two words: `lo` holds bytes 0..8 and `hi` bytes 8..15, both
//! little-endian and zero padded, with the length in the top byte of `hi`.
//! Equal keys mean equal byte strings (the length separates "ab" from
//! "ab\0"), and `hi` is never zero, so an all-zero slot reads as empty.

/// Longest byte string with a packed key.
pub(crate) const FAST_KEY_MAX: usize = 15;

/// Bytes a padded load may read from the start of a key.
pub(crate) const KEY_READ: usize = 16;

const LO_MASK: [u64; FAST_KEY_MAX + 1] = {
    let mut m = [u64::MAX; FAST_KEY_MAX + 1];
    let mut len = 0;
    while len < 8 {
        m[len] = (1u64 << (8 * len)) - 1;
        len += 1;
    }
    m
};

const HI_MASK: [u64; FAST_KEY_MAX + 1] = {
    let mut m = [0u64; FAST_KEY_MAX + 1];
    let mut len = 9;
    while len <= FAST_KEY_MAX {
        m[len] = (1u64 << (8 * (len - 8))) - 1;
        len += 1;
    }
    m
};

#[inline(always)]
fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Key of `bytes[at..at + len]`, reading `bytes[at..at + KEY_READ]`.
///
/// Caller guarantees `1 <= len <= FAST_KEY_MAX` and `at + KEY_READ <= bytes.len()`.
#[inline(always)]
pub(crate) fn key_padded(bytes: &[u8], at: usize, len: usize) -> (u64, u64) {
    debug_assert!((1..=FAST_KEY_MAX).contains(&len));
    debug_assert!(at + KEY_READ <= bytes.len());
    // SAFETY: the caller guarantees the 16 bytes from `at` are in bounds
    let w = unsafe { bytes.get_unchecked(at..at + KEY_READ) };
    let lo = read_u64(w, 0) & LO_MASK[len];
    let hi = (read_u64(w, 8) & HI_MASK[len]) | ((len as u64) << 56);
    (lo, hi)
}

/// Key of a whole slice of `1..=FAST_KEY_MAX` bytes, with no read past its end.
#[inline]
pub(crate) fn key_of(bytes: &[u8]) -> (u64, u64) {
    let mut buf = [0u8; KEY_READ];
    buf[..bytes.len()].copy_from_slice(bytes);
    key_padded(&buf, 0, bytes.len())
}

/// Key of `bytes[start..end]` inside a larger buffer: a direct load when 16
/// bytes are readable from `start`, else a copy of the piece.
#[inline(always)]
pub(crate) fn key_in(bytes: &[u8], start: usize, end: usize) -> (u64, u64) {
    if start + KEY_READ <= bytes.len() {
        key_padded(bytes, start, end - start)
    } else {
        key_of(&bytes[start..end])
    }
}

#[inline(always)]
pub(crate) fn hash(lo: u64, hi: u64) -> u64 {
    (lo ^ hi.rotate_left(27)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Hash for keys longer than [`FAST_KEY_MAX`]: one multiply per 8-byte
/// word, the last word overlapping the previous one.
#[inline]
pub(crate) fn hash_long(bytes: &[u8]) -> u64 {
    let n = bytes.len();
    debug_assert!(n >= 8);
    let mut h = (n as u64).wrapping_mul(0xA24B_AED4_963E_E407);
    let mut i = 0;
    while i + 8 < n {
        h = (h ^ read_u64(bytes, i)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = h.rotate_left(29);
        i += 8;
    }
    h = (h ^ read_u64(bytes, n - 8)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(bytes: &[u8]) -> (u64, u64) {
        let mut lo = 0u64;
        let mut hi = (bytes.len() as u64) << 56;
        for (i, &b) in bytes.iter().enumerate() {
            if i < 8 {
                lo |= (b as u64) << (8 * i);
            } else {
                hi |= (b as u64) << (8 * (i - 8));
            }
        }
        (lo, hi)
    }

    #[test]
    fn padded_load_matches_naive_packing_at_every_length_and_offset() {
        let text: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) | 0x80).collect();
        for len in 1..=FAST_KEY_MAX {
            for at in 0..=text.len() - len {
                let expected = naive(&text[at..at + len]);
                assert_eq!(key_of(&text[at..at + len]), expected, "len {len} at {at}");
                assert_eq!(key_in(&text, at, at + len), expected, "len {len} at {at}");
                if at + KEY_READ <= text.len() {
                    assert_eq!(key_padded(&text, at, len), expected);
                }
            }
        }
    }

    #[test]
    fn tail_loads_never_read_past_the_slice() {
        // every key ends exactly at the end of its buffer
        for len in 1..=FAST_KEY_MAX {
            let piece: Vec<u8> = (1..=len as u8).collect();
            assert_eq!(key_in(&piece, 0, len), naive(&piece));
        }
    }

    #[test]
    fn length_separates_nul_padding() {
        assert_ne!(key_of(b"ab"), key_of(b"ab\0"));
        assert_ne!(key_of(b"abcdefgh"), key_of(b"abcdefgh\0"));
        assert!(key_of(b"\0").1 != 0);
    }
}
