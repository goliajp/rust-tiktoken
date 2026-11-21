//! Pieces the encode loops' home-line probe did not answer.
//!
//! - up to 15 bytes: one walk of the shared table finds a token further down
//!   its probe chain or a derived entry; otherwise the piece is merged and its
//!   result published as a derived entry, so every later occurrence, on any
//!   thread, is one probe. Once the shared table is full these pieces go
//!   through the thread-local cache instead.
//! - longer, up to 255 bytes: the shared table of long merge results; on a
//!   miss, a long-token lookup or a merge, published there for every thread.
//!   Once that table refuses entries they go through the thread-local cache.
//! - longer still: a plain merge.

use crate::CoreBpe;
use crate::merge::{self, SMALL_MAX};
use crate::piece_cache;
use crate::vocab::{ABSENT, Entry, LONG_KEY_MAX, LongIds, Vocab};

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
    fn long(self, ids: LongIds<'_>) -> Self::Out;
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
    #[inline(always)]
    fn long(self, ids: LongIds<'_>) {
        ids.extend_into(self);
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
    #[inline(always)]
    fn long(self, ids: LongIds<'_>) -> usize {
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
    } else {
        if let Some(ids) = vocab.long_ids(piece) {
            return sink.long(ids);
        }
        if piece.len() <= LONG_KEY_MAX {
            let mut buf = [0u32; LONG_KEY_MAX];
            let n = match vocab.get(piece) {
                Some(id) => {
                    buf[0] = id;
                    1
                }
                None => merge_checked(piece, vocab, &mut buf),
            };
            let ids = &buf[..n];
            if vocab.insert_long(piece, ids) {
                return sink.ids(ids);
            }
            let fill = |b: &mut [u32; piece_cache::KEY_MAX]| {
                b[..n].copy_from_slice(ids);
                n
            };
            return piece_cache::with_ids(cache_id, piece, fill, |ids| sink.ids(ids));
        }
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

impl CoreBpe {
    /// Forget the merge results this instance shares between threads, so
    /// the next encode of any text starts as if none had been seen. For
    /// benchmarks of first-pass encoding; not part of the API.
    ///
    /// # Safety
    ///
    /// No other thread may encode with this instance until it returns; a
    /// concurrent encode may return wrong tokens.
    #[doc(hidden)]
    pub unsafe fn __reset_shared_cache(&self) {
        self.vocab.clear_derived();
    }

    #[cfg(test)]
    pub(crate) fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use crate::CoreBpe;
    use crate::hash::FxHashMap;
    use crate::merge;
    use crate::pretokenize::{FastPath, PreTokenizer, WhitespaceRules};

    // every byte, plus tokens over a small alphabet, in a vocabulary small
    // enough that the shared table fills after a few hundred derived entries
    fn small_bpe() -> CoreBpe {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for b in 0..=255u8 {
            encoder.insert(vec![b], b as u32);
        }
        let alpha = b"abcdefgh";
        let mut rank = 256;
        for &x in alpha {
            for &y in alpha {
                if (x + y) % 3 != 0 {
                    encoder.insert(vec![x, y], rank);
                    encoder.insert(vec![b' ', x, y], rank + 1);
                    encoder.insert(vec![x, y, x], rank + 2);
                    rank += 3;
                }
            }
        }
        let special = FxHashMap::default();
        CoreBpe::new(
            encoder,
            special,
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        )
    }

    fn text(seed: u64) -> String {
        let mut x = seed | 1;
        let mut s = String::new();
        for _ in 0..3000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.push(' ');
            for i in 0..1 + x % 30 {
                s.push(b"abcdefgh"[((x >> (2 * i)) % 8) as usize] as char);
            }
        }
        s
    }

    #[test]
    fn full_shared_table_falls_back_to_the_thread_cache() {
        let bpe = small_bpe();
        let t = text(7);
        let mut want = vec![];
        let mut pos = 0;
        while let Some((s, e)) = bpe.pre_tokenizer().next_match(&t, pos) {
            merge::bpe_encode(&t.as_bytes()[s..e], bpe.vocab(), &mut want);
            pos = e;
        }
        for _ in 0..3 {
            assert_eq!(bpe.encode(&t), want);
            assert_eq!(bpe.count(&t), want.len());
        }
        assert!(!bpe.vocab().accepts_derived());
        // other threads see the shared entries and build their own cache
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| assert_eq!(bpe.encode(&t), want));
            }
        });
    }
}
