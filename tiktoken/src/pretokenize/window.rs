//! Windowed piece-start decision for the cl100k family (cl100k_base, llama3,
//! glm4, glm5, qwen2, qwen3_5), and the parts shared with the case-split
//! family in `window_case.rs`.
//!
//! A window is up to 64 bytes from a known piece start (bit 0). From the class
//! bitmaps it computes every piece start in the window as if the text ended at
//! the window's end, then drops the starts that bytes past the end could
//! change. The highest remaining start is the cut: the pieces between the
//! starts below it are final, and the next window opens at the cut. Nothing
//! carries over between windows; none of the patterns looks behind a piece
//! start. A cut of 0 means the window decided nothing (a piece as long as the
//! window, or a [`M_HARD`] byte at the start), and the caller scans one piece
//! with the scalar scanner instead.

use super::bitops::{below_lowest, fill_down, fill_up, last_of, next, prev, to_lead};
use super::classify::{
    Bitmaps, CLASS_CASE, CLASS_CL100K, CLASS_QWEN35, ClassTable, M_AP, M_CONT, M_HARD, M_L, M_LW,
    M_N, M_NL, M_SL, M_SP, M_W,
};

/// The class masks of one window. Bits at and above `v`'s length are zero
/// only after masking with `v`.
pub(crate) struct Masks {
    pub(super) l: u64,
    pub(super) lw: u64,
    pub(super) n: u64,
    pub(super) w: u64,
    pub(super) nl: u64,
    pub(super) sp: u64,
    pub(super) ap: u64,
    pub(super) sl: u64,
    pub(super) hard: u64,
    pub(super) cont: u64,
    /// Bytes inside the window.
    pub(super) v: u64,
}

impl Masks {
    /// The window at bit `r` of `maps`, clipped to `limit`. Also returns
    /// whether the window reaches `limit`.
    #[inline(always)]
    pub(crate) fn load<const CASE: bool, const W: usize>(
        maps: &Bitmaps<W>,
        r: usize,
        limit: usize,
    ) -> (Self, bool) {
        debug_assert!(r < limit);
        let wl = (64 - (r & 7)).min(limit - r);
        let v = if wl == 64 { u64::MAX } else { (1u64 << wl) - 1 };
        let get = |map| maps.window(map, r);
        let m = Self {
            l: get(M_L),
            lw: if CASE { get(M_LW) } else { 0 },
            n: get(M_N),
            w: get(M_W),
            nl: get(M_NL),
            sp: get(M_SP),
            ap: get(M_AP),
            sl: if CASE { get(M_SL) } else { 0 },
            hard: get(M_HARD),
            cont: get(M_CONT),
            v,
        };
        (m, r + wl == limit)
    }

    /// Valid bytes: inside the window and below the first hard byte.
    #[inline(always)]
    pub(super) fn valid(&self) -> (u64, bool) {
        let h = self.hard & self.v;
        (self.v & below_lowest(h), h == 0)
    }
}

/// Piece starts that do not depend on the letter rules' details, and what
/// the letter rules need.
pub(super) struct Core {
    /// Starts of whitespace, punctuation and digit pieces, and of letter runs.
    pub(super) starts: u64,
    /// Single-char punctuation runs that begin a piece and prefix a word.
    pub(super) px: u64,
    /// Whitespace not taken by a punctuation tail.
    pub(super) w1: u64,
    /// Starts of `w1` runs.
    pub(super) ws: u64,
}

/// The rules shared by both families. `l` is every letter, `tail` the
/// punctuation tail class (`[\r\n]` or `[\r\n/]`), both masked to `v`.
#[inline(always)]
pub(super) fn core<const MAX_DIGITS: usize, const CONT: bool>(
    m: &Masks,
    v: u64,
    l: u64,
    tail: u64,
) -> Core {
    let cont = if CONT { m.cont & v } else { 0 };
    let n = m.n & v;
    let w = m.w & v;
    let nl = m.nl & v;
    let o = v & !(l | n | w);

    // ` ?P+[\r\n]*`: the tail after a punctuation run
    let t = fill_up(nl & prev(o), tail);
    let o1 = o & !t;
    let w1 = w & !t;
    let nl1 = nl & !t;

    // whitespace: `\s*[\r\n]+` takes a run up to its last newline; the rest
    // of the run gives its last char back when a non-space follows
    let upto_nl = fill_down(nl1, w1);
    let rest = w1 & !upto_nl;
    let ws = w1 & !prev(w1);
    let rs = rest & !prev(rest);
    let split = rest & !next(rest) & next(v & !w);
    let jo = split & m.sp; // only ' ' joins a following punctuation run

    let osr = o1 & !prev(o1);
    let ostart = osr & !prev(jo);
    let px = ostart & to_lead(o1 & next(l), cont);
    let lstart = l & !prev(l) & !prev(split) & !prev(last_of(px, cont));

    let mut nstart = n;
    if MAX_DIGITS == 3 {
        nstart = n & !prev(n);
        let n3 = n & prev(n) & prev(prev(n)) & prev(prev(prev(n)));
        if n3 != 0 {
            let mut g = nstart;
            loop {
                g = (g << 3) & n3;
                if g == 0 {
                    break;
                }
                nstart |= g;
            }
        }
    }

    Core {
        starts: 1 | ws | rs | to_lead(split, cont) | ostart | lstart | nstart,
        px,
        w1,
        ws,
    }
}

/// A contraction `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at an apostrophe.
pub(super) enum Contraction {
    No,
    Len(u32),
    /// The bytes that decide it are past the window.
    Unknown,
}

#[inline]
pub(super) fn contraction(win: &[u8], p: u32, v: u64, eot: bool) -> Contraction {
    let inside = |i: u32| i < 64 && v >> i & 1 == 1;
    let past = if eot {
        Contraction::No
    } else {
        Contraction::Unknown
    };
    if !inside(p + 1) {
        return past;
    }
    let c1 = win[p as usize + 1] | 0x20;
    match c1 {
        b's' | b't' | b'm' | b'd' => Contraction::Len(2),
        b'r' | b'v' | b'l' => {
            if !inside(p + 2) {
                return past;
            }
            let c2 = win[p as usize + 2] | 0x20;
            if c2 == if c1 == b'l' { b'l' } else { b'e' } {
                Contraction::Len(3)
            } else {
                Contraction::No
            }
        }
        _ => Contraction::No,
    }
}

/// Drop the starts that bytes past the window could change, and pick the
/// cut. At the end of the text every start is final and the cut is the end.
#[inline(always)]
pub(super) fn finish(mut s: u64, c: &Core, v: u64, eot: bool) -> (u64, u32) {
    let len = 64 - v.leading_zeros();
    if eot {
        return (s, len);
    }
    // a whitespace run reaching the end may hold more newlines, or be
    // followed by a non-space: only its first start is final
    let top = 1u64 << (len - 1);
    if c.w1 & top != 0 {
        s &= !(fill_down(top, c.w1) & !c.ws);
    }
    (s, 63 - s.leading_zeros())
}

/// Starts and cut of a cl100k-family window. `win` holds the window's bytes.
#[inline(always)]
pub(crate) fn decide_cl100k<const MAX_DIGITS: usize, const CONT: bool>(
    m: &Masks,
    eot: bool,
    win: &[u8],
) -> (u64, u32) {
    let (v, no_hard) = m.valid();
    if v == 0 {
        return (1, 0);
    }
    let eot = eot && no_hard;
    let l = m.l & v;
    let c = core::<MAX_DIGITS, CONT>(m, v, l, m.nl & v);
    let mut s = c.starts;
    // a standalone contraction can only start at a one-char punctuation run
    // that prefixes a word
    let mut ca = m.ap & c.px;
    while ca != 0 {
        let p = ca.trailing_zeros();
        ca &= ca - 1;
        if let Contraction::Len(k) = contraction(win, p, v, eot)
            && p + k < 64
        {
            s |= l & (1u64 << (p + k));
        }
    }
    finish(s, &c, v, eot)
}

/// One family's window rules, for the encode loop.
pub(crate) trait WindowRules {
    /// Whether the case-split maps are classified.
    const CASE: bool;
    fn classes() -> &'static ClassTable;
    fn decide<const CONT: bool>(m: &Masks, eot: bool, win: &[u8]) -> (u64, u32);
}

/// cl100k_base, llama3, glm4, glm5 (`MAX_DIGITS = 3`), qwen2 and qwen3_5
/// (`MAX_DIGITS = 1`, `QWEN35` for qwen3_5's mark rules).
pub(crate) struct Cl100kRules<const MAX_DIGITS: usize, const QWEN35: bool>;

impl<const MAX_DIGITS: usize, const QWEN35: bool> WindowRules for Cl100kRules<MAX_DIGITS, QWEN35> {
    const CASE: bool = false;

    #[inline(always)]
    fn classes() -> &'static ClassTable {
        if QWEN35 { &CLASS_QWEN35 } else { &CLASS_CL100K }
    }

    #[inline(always)]
    fn decide<const CONT: bool>(m: &Masks, eot: bool, win: &[u8]) -> (u64, u32) {
        decide_cl100k::<MAX_DIGITS, CONT>(m, eot, win)
    }
}

/// o200k_base, minimax_m2, llama4 (`<true, 3>`) and mistral_v3 (`<false, 1>`).
pub(crate) struct CaseRules<const CONTRACTIONS: bool, const MAX_DIGITS: usize>;

impl<const CONTRACTIONS: bool, const MAX_DIGITS: usize> WindowRules
    for CaseRules<CONTRACTIONS, MAX_DIGITS>
{
    const CASE: bool = true;

    #[inline(always)]
    fn classes() -> &'static ClassTable {
        &CLASS_CASE
    }

    #[inline(always)]
    fn decide<const CONT: bool>(m: &Masks, eot: bool, win: &[u8]) -> (u64, u32) {
        super::window_case::decide_case::<CONTRACTIONS, MAX_DIGITS, CONT>(m, eot, win)
    }
}
