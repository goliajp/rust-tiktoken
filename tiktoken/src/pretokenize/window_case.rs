//! Windowed piece-start decision for the case-split family (o200k_base,
//! minimax_m2, llama4, mistral_v3); see `window.rs` for the scheme.
//!
//! The window only knows upper-case (`Lu`, `Lt`) and lower-case (`Ll`)
//! letters. Chars in both case classes (`Lm`, `Lo`, `\p{M}`) are hard bytes,
//! so within a window a letter run is upper-case chars then lower-case chars,
//! and a new piece starts wherever an upper-case letter follows a lower-case
//! one.

use super::bitops::prev;
use super::window::{Contraction, Masks, contraction, core, finish};

/// Starts and cut of a case-split window. `CONTRACTIONS` is the optional
/// contraction suffix of the word rules; the punctuation tail is `[\r\n/]*`.
#[inline(always)]
pub(crate) fn decide_case<const CONTRACTIONS: bool, const MAX_DIGITS: usize, const CONT: bool>(
    m: &Masks,
    eot: bool,
    win: &[u8],
) -> (u64, u32) {
    let (v, no_hard) = m.valid();
    if v == 0 {
        return (1, 0);
    }
    let eot = eot && no_hard;
    let u = m.l & v;
    let lw = m.lw & v;
    let l = u | lw;
    let c = core::<MAX_DIGITS, CONT>(m, v, l, (m.nl | m.sl) & v);
    let mut s = c.starts | (u & prev(lw));
    if CONTRACTIONS {
        // the suffix follows the last letter of a word; an apostrophe right
        // after a suffix begins a new piece instead
        let mut ca = m.ap & v & prev(l);
        let mut after = u32::MAX;
        while ca != 0 {
            let p = ca.trailing_zeros();
            ca &= ca - 1;
            if p == after {
                continue;
            }
            match contraction(win, p, v, eot) {
                Contraction::No => {}
                Contraction::Len(k) => {
                    s &= !(((1u64 << k) - 1) << p);
                    if p + k < 64 {
                        s |= l & (1u64 << (p + k));
                    }
                    after = p + k;
                }
                Contraction::Unknown => {
                    // the word before the apostrophe may not be complete
                    s &= (1u64 << p) - 1;
                    break;
                }
            }
        }
    }
    finish(s, &c, v, eot)
}
