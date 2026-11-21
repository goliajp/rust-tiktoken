//! Pieces the encode loops' home-line probe did not answer.
//!
//! - up to 15 bytes: one walk of the shared table finds a token further down
//!   its probe chain or a derived entry; otherwise the piece is merged and its
//!   result published as a derived entry, so every later occurrence, on any
//!   thread, is one probe. Once the shared table is full these pieces go
//!   through the thread-local cache instead.
//! - longer: a long-token lookup, then the thread-local cache (up to its key
//!   limit), then a plain merge.

use crate::merge::{self, SMALL_MAX};
use crate::piece_cache;
use crate::vocab::{ABSENT, Entry, Vocab};

/// Packed key of a piece of up to 15 bytes, or `None` for longer ones.
pub(crate) type Key = Option<(u64, u64)>;

#[inline]
fn merge_checked(piece: &[u8], vocab: &Vocab, ids: &mut [u32]) -> usize {
    let n = if piece.len() <= SMALL_MAX {
        merge::merge_small(piece, vocab, ids)
    } else {
        let mut out = Vec::with_capacity(piece.len());
        merge::bpe_encode(piece, vocab, &mut out);
        ids[..out.len()].copy_from_slice(&out);
        out.len()
    };
    assert!(!ids[..n].contains(&ABSENT), "single byte not in vocab");
    n
}

/// Where a piece's ids go: appended to the output, or only counted.
trait Sink {
    type Out;
    fn entry(self, vocab: &Vocab, e: Entry) -> Self::Out;
    fn ids(self, ids: &[u32]) -> Self::Out;
}

impl Sink for &mut Vec<u32> {
    type Out = ();
    #[inline(always)]
    fn entry(self, vocab: &Vocab, e: Entry) {
        vocab.extend_entry(e, self);
    }
    #[inline(always)]
    fn ids(self, ids: &[u32]) {
        self.extend_from_slice(ids);
    }
}

struct Count;

impl Sink for Count {
    type Out = usize;
    #[inline(always)]
    fn entry(self, _: &Vocab, e: Entry) -> usize {
        e.len()
    }
    #[inline(always)]
    fn ids(self, ids: &[u32]) -> usize {
        ids.len()
    }
}

#[inline(always)]
fn resolve<S: Sink>(vocab: &Vocab, cache_id: u64, piece: &[u8], key: Key, sink: S) -> S::Out {
    if let Some((lo, hi)) = key {
        if let Some(e) = vocab.entry(lo, hi) {
            return sink.entry(vocab, e);
        }
        if vocab.accepts_derived() {
            let mut ids = [0u32; SMALL_MAX];
            let n = merge_checked(piece, vocab, &mut ids);
            let ids = &ids[..n];
            if vocab.insert_derived(lo, hi, ids) {
                return sink.ids(ids);
            }
            // full now: keep it where the next occurrence will look
            let fill = |buf: &mut [u32; piece_cache::KEY_MAX]| {
                buf[..n].copy_from_slice(ids);
                n
            };
            return piece_cache::with_ids(cache_id, piece, fill, |ids| sink.ids(ids));
        }
    } else if let Some(id) = vocab.get(piece) {
        return sink.ids(&[id]);
    }
    if piece.len() <= piece_cache::KEY_MAX {
        let fill = |buf: &mut [u32; piece_cache::KEY_MAX]| merge_checked(piece, vocab, buf);
        piece_cache::with_ids(cache_id, piece, fill, |ids| sink.ids(ids))
    } else {
        let mut out = Vec::with_capacity(piece.len());
        merge::bpe_encode(piece, vocab, &mut out);
        sink.ids(&out)
    }
}

pub(crate) fn encode(vocab: &Vocab, cache_id: u64, piece: &[u8], key: Key, out: &mut Vec<u32>) {
    resolve(vocab, cache_id, piece, key, out);
}

pub(crate) fn count(vocab: &Vocab, cache_id: u64, piece: &[u8], key: Key) -> usize {
    resolve(vocab, cache_id, piece, key, Count)
}
