//! The merge of pieces made mostly of multibyte chars, one char at a time.
//!
//! Atoms start whole and wall chars as their parts (see the `atoms` and
//! `walls` modules of `vocab`). A wall char's parts never merge with
//! anything, so they go straight to the output and cut the piece into runs
//! of parts that merge only among themselves. In text of such chars most
//! runs hold one or two parts, which need at most one lookup; longer runs
//! merge with the parts packed at the front of small arrays.

use super::PartEnds;
use crate::vocab::{ABSENT, Atoms, RankProbe, Vocab, copy_ids};

/// Most parts in one run; a piece with a longer run takes the general merge.
const RUN_MAX: usize = 32;

/// Merge a run of `m >= 3` parts: `id[j]` covers bytes `at[j]..at[j + 1]`.
/// Returns the part count after merging, with the ids at the front of `id`.
#[inline(never)]
fn merge_run(
    id: &mut [u32; RUN_MAX],
    at: &mut [u8; RUN_MAX + 1],
    mut m: usize,
    piece: &[u8],
    probe: RankProbe<'_>,
) -> usize {
    let span = |at: &[u8; RUN_MAX + 1], a: usize, b: usize| {
        probe.rank_of(&piece[at[a] as usize..at[b] as usize])
    };
    // rank[j]: rank of parts j and j + 1 joined
    let mut rank = [u32::MAX; RUN_MAX];
    for (j, r) in rank.iter_mut().enumerate().take(m - 1) {
        *r = span(at, j, j + 2);
    }
    while m > 1 {
        let (mut k, mut r) = (0, rank[0]);
        for (j, &x) in rank.iter().enumerate().take(m - 1).skip(1) {
            if x < r {
                (k, r) = (j, x);
            }
        }
        if r == u32::MAX {
            break;
        }
        id[k] = r;
        m -= 1;
        for j in k + 1..m {
            id[j] = id[j + 1];
            rank[j] = rank[j + 1];
        }
        for j in k + 1..=m {
            at[j] = at[j + 1];
        }
        rank[k] = if k + 1 < m {
            span(at, k, k + 2)
        } else {
            u32::MAX
        };
        if k > 0 {
            rank[k - 1] = span(at, k - 1, k + 1);
        }
    }
    m
}

/// Up to four bytes of `piece` from `i`, little-endian, zero past its end.
#[inline(always)]
fn window(piece: &[u8], i: usize) -> u32 {
    let n = piece.len();
    if let Some(w) = piece.get(i..i + 4) {
        return u32::from_le_bytes(w.try_into().unwrap());
    }
    if let Some(w) = n.checked_sub(4).map(|s| &piece[s..]) {
        // the piece's last four bytes, shifted down to start at `i`
        return u32::from_le_bytes(w.try_into().unwrap()) >> (8 * (i + 4 - n));
    }
    piece[i..]
        .iter()
        .rev()
        .fold(0, |w, &b| w << 8 | u32::from(b))
}

/// Merge `piece` (2 to 255 bytes) into `ids`, which has room for one id per
/// byte; returns the id count, or `None` when a run outgrows the arrays.
/// The end of each id's token in the piece goes to `ends`.
#[inline(always)]
pub(super) fn merge_chars<E: PartEnds>(
    piece: &[u8],
    vocab: &Vocab,
    atoms: &Atoms,
    ids: &mut [u32],
    ends: &mut E,
) -> Option<usize> {
    let n = piece.len();
    debug_assert!((2..=255).contains(&n));
    assert!(ids.len() >= n && (!E::ON || ends.len() >= n));
    let table = atoms.walls();
    let probe = vocab.rank_probe();
    let single = |b: u32| vocab.rank_single(b as u8);
    // the current run: `m` parts, `at[m]` the end of the last
    let mut id = [0u32; RUN_MAX];
    let mut at = [0u8; RUN_MAX + 1];
    let mut m = 0;
    let mut c = 0;
    // write one id at `c`: every id written covers at least one byte, so `c`
    // stays at most the bytes consumed, below `n <= ids.len()`
    macro_rules! put {
        ($x:expr) => {{
            debug_assert!(c < n);
            // SAFETY: c < n <= ids.len(), as above
            unsafe { *ids.get_unchecked_mut(c) = $x };
            c += 1;
        }};
        ($x:expr, $end:expr) => {{
            debug_assert!(c < n);
            // SAFETY: c < n <= ids.len(), as above
            unsafe { *ids.get_unchecked_mut(c) = $x };
            if E::ON {
                // SAFETY: c < n <= ends.len() when it keeps ends
                unsafe { ends.put(c, $end as usize) };
            }
            c += 1;
        }};
    }
    macro_rules! push {
        ($id:expr, $at:expr, $len:expr) => {{
            if m == RUN_MAX {
                return None;
            }
            id[m] = $id;
            at[m] = $at as u8;
            m += 1;
            at[m] = ($at + $len) as u8;
        }};
    }
    // merge the run and write its ids at `c`; the caller empties it
    macro_rules! flush {
        () => {{
            match m {
                0 => {}
                1 => put!(id[0], at[1]),
                2 => {
                    let r = probe.rank_of(&piece[at[0] as usize..at[2] as usize]);
                    if r != ABSENT {
                        put!(r, at[2]);
                    } else {
                        put!(id[0], at[1]);
                        put!(id[1], at[2]);
                    }
                }
                _ => {
                    let k = merge_run(&mut id, &mut at, m, piece, probe);
                    assert!(c + k <= ids.len());
                    // SAFETY: both ranges are in bounds, and the merge
                    // leaves at least one part
                    unsafe { copy_ids(id.as_ptr(), ids.as_mut_ptr().add(c), k) };
                    if E::ON {
                        // the merge left part j covering at[j]..at[j + 1]
                        for j in 0..k {
                            // SAFETY: c + k <= n, as for the ids
                            unsafe { ends.put(c + j, at[j + 1] as usize) };
                        }
                    }
                    c += k;
                }
            }
        }};
    }
    let mut i = 0;
    while i < n {
        let w = window(piece, i);
        // a 2- or 3-byte char: its length and code point
        let (l, cp) = if w & 0xC0E0 == 0x80C0 && w & 0x1E != 0 {
            (2, (w & 0x1F) << 6 | (w >> 8) & 0x3F)
        } else if w & 0xC0_C0F0 == 0x80_80E0 && w & 0x200F != 0 {
            (3, (w & 0x0F) << 12 | (w & 0x3F00) >> 2 | (w >> 16) & 0x3F)
        } else {
            match (w & 0xF8 == 0xF0 && i + 4 <= n).then(|| atoms.at(piece, i, 4)) {
                Some(Some(r)) => {
                    push!(r, i, 4);
                    i += 4;
                }
                _ => {
                    push!(single(w & 0xFF), i, 1);
                    i += 1;
                }
            }
            continue;
        };
        let cw = table.word(cp);
        if cw.is_plain() || !table.clear(piece, i, l, cw) {
            for j in 0..l {
                push!(single(w >> (8 * j) & 0xFF), i + j, 1);
            }
        } else if let Some(r) = cw.atom() {
            // an atom that merges with nothing on one side ends the run
            // there
            let (before, after) = cw.alone();
            if before {
                flush!();
                m = 0;
            }
            push!(r, i, l);
            if after {
                flush!();
                m = 0;
            }
        } else {
            flush!();
            m = 0;
            let b = |j: u32| single(w >> (8 * j) & 0xFF);
            let (p0, p1, p2) = match cw.joined() {
                (Some(r), _) => (r, b(2), None),
                (_, Some(r)) => (b(0), r, None),
                _ => (b(0), b(1), (l == 3).then(|| b(2))),
            };
            put!(p0);
            put!(p1);
            if let Some(p) = p2 {
                put!(p);
            }
            if E::ON {
                // the parts end at i + 1, i + 2 and i + 3 when they are the
                // bytes; a joined pair ends one byte later or where the
                // char does
                let (k, first) = match cw.joined() {
                    (Some(_), _) => (2, 2),
                    (_, Some(_)) => (2, 1),
                    _ => (l, 1),
                };
                // SAFETY: the `k` parts were written just before `c`
                unsafe {
                    ends.put(c - k, i + first);
                    ends.put(c - k + 1, if k == 3 { i + 2 } else { i + l });
                    if k == 3 {
                        ends.put(c - 1, i + 3);
                    }
                }
            }
        }
        i += l;
    }
    flush!();
    Some(c)
}
