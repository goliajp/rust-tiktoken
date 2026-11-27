//! Pieces the ends sink found the shared tables not to hold: their ids, as
//! the encode loops' miss path gives them, with each token's end taken from
//! the merge's part spans rather than from the token lengths.
//!
//! Only pieces that are merged here get their ends that way: those the
//! shared tables take, and long pieces that are a token. Pieces the
//! thread-local cache answers once the tables are full, and pieces too long
//! for any table, go through the plain miss path and the token lengths.

use crate::encode_loop::{Engine, NO_KEY};
use crate::encode_miss::MID_LONG;
use crate::merge::{self, SMALL_MAX};
use crate::piece_cache;
use crate::vocab::{ABSENT, FAST_KEY_MAX, LONG_KEY_MAX, LongPiece, Vocab, copy_ids};

/// Merge `piece` (up to [`LONG_KEY_MAX`] bytes) into `ids`, and the end of
/// each id's token in the piece into `ends`; both have room for an id per
/// byte. Returns the id count.
#[inline]
fn merge_checked_ends(piece: &[u8], vocab: &Vocab, ids: &mut [u32], ends: &mut [u8]) -> usize {
    let n = if piece.len() <= SMALL_MAX {
        merge::merge_small_ends(piece, vocab, ids, ends)
    } else {
        // the heap merge keeps no spans; pieces this long are rare
        let mut out = Vec::with_capacity(piece.len());
        merge::bpe_encode(piece, vocab, &mut out);
        let mut e = 0;
        for (j, &id) in out.iter().enumerate() {
            ids[j] = id;
            e += vocab.token_len(id);
            ends[j] = e as u8;
        }
        out.len()
    };
    assert!(
        vocab.has_every_byte() || !ids[..n].contains(&ABSENT),
        "single byte not in vocab"
    );
    n
}

/// Write `got` at `ip`, and `at` plus each of `rel` at `ep`; returns the
/// count.
///
/// # Safety
///
/// `got.len()` ids are writable at `ip` and as many ends at `ep`.
#[inline(always)]
unsafe fn put(ip: *mut u32, ep: *mut usize, got: &[u32], rel: &[u8], at: usize) -> usize {
    let n = got.len();
    debug_assert!(n >= 1 && rel.len() == n);
    // SAFETY: passed on from the caller
    unsafe {
        copy_ids(got.as_ptr(), ip, n);
        for (j, &e) in rel.iter().enumerate() {
            ep.add(j).write(at + e as usize);
        }
    }
    n
}

impl Engine<'_> {
    /// [`encode_unheld`](Engine::encode_unheld) of a piece of up to
    /// [`LONG_KEY_MAX`] bytes when this merges it: its ids written at `ip`
    /// and the end of each one's token at `ep`, as `at` plus its end in the
    /// piece; returns the id count. `None`, with nothing written, when the
    /// thread-local cache has the piece's ids instead.
    ///
    /// # Safety
    ///
    /// `piece.len()` ids are writable at `ip` and as many ends at `ep`.
    #[inline(never)]
    pub(crate) unsafe fn unheld_ends(
        &self,
        piece: &[u8],
        key: (u64, u64),
        ip: *mut u32,
        ep: *mut usize,
        at: usize,
    ) -> Option<usize> {
        debug_assert!(piece.len() <= LONG_KEY_MAX);
        let vocab = self.vocab;
        if key == NO_KEY {
            let lp = vocab.long_piece(piece);
            // SAFETY (both): passed on from the caller
            return Some(if piece.len() <= MID_LONG {
                unsafe { self.long_ends::<MID_LONG>(&lp, piece, ip, ep, at) }
            } else {
                unsafe { self.long_ends::<LONG_KEY_MAX>(&lp, piece, ip, ep, at) }
            });
        }
        if !vocab.accepts_derived() {
            return None;
        }
        let mut buf = [0u32; FAST_KEY_MAX];
        let mut rel = [0u8; FAST_KEY_MAX];
        let n = merge_checked_ends(piece, vocab, &mut buf, &mut rel);
        let got = &buf[..n];
        if !vocab.insert_derived(key.0, key.1, got) {
            self.keep_local(piece, got);
        }
        // SAFETY: a piece has at least as many bytes as ids; passed on
        Some(unsafe { put(ip, ep, got, &rel[..n], at) })
    }

    /// A piece of 16 to `M` bytes that the shared table of long results
    /// lacks: published there with its ends, which the next encode with
    /// ends to meet it reads instead of the token lengths.
    ///
    /// # Safety
    ///
    /// As [`unheld_ends`](Engine::unheld_ends).
    #[inline(always)]
    unsafe fn long_ends<const M: usize>(
        &self,
        lp: &LongPiece<'_>,
        piece: &[u8],
        ip: *mut u32,
        ep: *mut usize,
        at: usize,
    ) -> usize {
        let vocab = self.vocab;
        let mut buf = [0u32; M];
        let mut rel = [0u8; M];
        let n = match vocab.long_token(lp) {
            Some(id) => {
                buf[0] = id;
                rel[0] = piece.len() as u8;
                1
            }
            None => merge_checked_ends(piece, vocab, &mut buf, &mut rel),
        };
        let (got, rel) = (&buf[..n], &rel[..n]);
        match vocab.insert_long_ids(lp, got) {
            Some(kept) => vocab.keep_long_ends(&kept, rel),
            None => self.keep_local(piece, got),
        }
        // SAFETY: as in `unheld_ends`
        unsafe { put(ip, ep, got, rel, at) }
    }

    /// Keep a piece's ids in the thread-local cache, where the next
    /// occurrence looks once the shared tables refuse it.
    #[cold]
    fn keep_local(&self, piece: &[u8], got: &[u32]) {
        let fill = |b: &mut [u32; piece_cache::KEY_MAX]| {
            b[..got.len()].copy_from_slice(got);
            got.len()
        };
        piece_cache::with_ids(self.cache_id, piece, fill, |_| ());
    }
}
