//! Bit-parallel helpers for the window scanners. Bit `i` of a mask stands for
//! byte `i` of the window.

/// Bit `i` holds the value of bit `i - 1` (the byte before).
#[inline(always)]
pub(super) const fn prev(x: u64) -> u64 {
    x << 1
}

/// Bit `i` holds the value of bit `i + 1` (the byte after).
#[inline(always)]
pub(super) const fn next(x: u64) -> u64 {
    x >> 1
}

/// From each marker in `m`, upward through the run of `c` it sits in.
#[inline(always)]
pub(super) const fn fill_up(m: u64, c: u64) -> u64 {
    // the carry of `marker + run` clears the run from the marker up; xor turns
    // that into ones, and `| m` restores markers a lower carry passed over
    ((((m & c).wrapping_add(c)) ^ c) | m) & c
}

/// From each marker in `m`, downward through the run of `c` it sits in.
#[inline(always)]
pub(super) const fn fill_down(m: u64, c: u64) -> u64 {
    fill_up(m.reverse_bits(), c.reverse_bits()).reverse_bits()
}

/// Moves each bit of `m`, which sits on the last byte of a char, down to the
/// char's lead byte. `cont` marks utf-8 continuation bytes.
#[inline(always)]
pub(super) const fn to_lead(m: u64, cont: u64) -> u64 {
    let mut r = m & !cont;
    let mut x = (m & cont) >> 1;
    r |= x & !cont;
    x = (x & cont) >> 1;
    r |= x & !cont;
    x = (x & cont) >> 1;
    r | (x & !cont)
}

/// Moves each bit of `m`, which sits on the lead byte of a char, up to the
/// char's last byte.
#[inline(always)]
pub(super) const fn last_of(m: u64, cont: u64) -> u64 {
    let n = cont >> 1;
    let mut r = m & !n;
    let mut x = (m & n) << 1;
    r |= x & !n;
    x = (x & n) << 1;
    r |= x & !n;
    x = (x & n) << 1;
    r | x
}

/// Mask of the bits below the lowest set bit of `x` (all ones when `x == 0`).
#[inline(always)]
pub(super) const fn below_lowest(x: u64) -> u64 {
    x.isolate_lowest_one().wrapping_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_fill_up(m: u64, c: u64) -> u64 {
        let mut r = 0;
        for i in 0..64 {
            if m >> i & 1 == 1 && c >> i & 1 == 1 {
                let mut j = i;
                while j < 64 && c >> j & 1 == 1 {
                    r |= 1 << j;
                    j += 1;
                }
            }
        }
        r
    }

    fn naive_fill_down(m: u64, c: u64) -> u64 {
        naive_fill_up(m.reverse_bits(), c.reverse_bits()).reverse_bits()
    }

    #[test]
    fn fills_match_naive() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..20_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let m = s & s.rotate_left(11) & s.rotate_left(29);
            let c = s | s.rotate_left(5);
            assert_eq!(fill_up(m, c), naive_fill_up(m, c));
            assert_eq!(fill_down(m, c), naive_fill_down(m, c));
        }
    }

    #[test]
    fn lead_and_last_follow_char_widths() {
        // chars: a(1) é(2) 世(3) 😀(4) b(1) at bytes 0, 1, 3, 6, 10
        let text = "aé世😀b".as_bytes();
        let cont = text
            .iter()
            .enumerate()
            .fold(0u64, |a, (i, &b)| a | (((b & 0xC0 == 0x80) as u64) << i));
        let leads = 1 | 1 << 1 | 1 << 3 | 1 << 6 | 1 << 10;
        let lasts = 1 | 1 << 2 | 1 << 5 | 1 << 9 | 1 << 10;
        assert_eq!(to_lead(lasts, cont), leads);
        assert_eq!(last_of(leads, cont), lasts);
        assert_eq!(to_lead(1 << 5, cont), 1 << 3);
        assert_eq!(last_of(1 << 6, cont), 1 << 9);
    }

    #[test]
    fn below_lowest_bit() {
        assert_eq!(below_lowest(0), u64::MAX);
        assert_eq!(below_lowest(1), 0);
        assert_eq!(below_lowest(0b1011000), 0b111);
    }
}
