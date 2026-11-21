//! The per-piece encode and count loops.
//!
//! Each pre-token piece of 1..=15 bytes is packed into a two-word key and
//! checked against its home slot in the vocabulary's fast table; that answers
//! most pieces of real text with one slot read. Everything else (keys that
//! sit further down their probe chain, longer pieces, pieces that need a BPE
//! merge) leaves the loop through an out-of-line call, so the loop body stays
//! small next to the inlined scanner.

use crate::merge;
use crate::piece_cache;
use crate::pretokenize::{PreTokenizer, RegexPreTokenizer};
use crate::vocab::{FAST_KEY_MAX, Vocab, key_in};

pub(crate) struct Engine<'a> {
    pub(crate) vocab: &'a Vocab,
    pub(crate) pre: &'a RegexPreTokenizer,
    pub(crate) cache_id: u64,
}

trait Sink {
    fn token(&mut self, id: u32);
    fn piece(&mut self, engine: &Engine<'_>, piece: &[u8]);
}

struct Encode<'v>(&'v mut Vec<u32>);

impl Sink for Encode<'_> {
    #[inline(always)]
    fn token(&mut self, id: u32) {
        self.0.push(id);
    }

    #[inline(never)]
    fn piece(&mut self, engine: &Engine<'_>, piece: &[u8]) {
        let vocab = engine.vocab;
        if let Some(token) = vocab.get(piece) {
            self.0.push(token);
        } else if (3..=piece_cache::KEY_MAX).contains(&piece.len()) {
            piece_cache::encode_piece(engine.cache_id, piece, self.0, |buf| {
                merge::bpe_encode_buf(piece, vocab, buf)
            });
        } else {
            merge::bpe_encode(piece, vocab, self.0);
        }
    }
}

struct Count(usize);

impl Sink for Count {
    #[inline(always)]
    fn token(&mut self, _id: u32) {
        self.0 += 1;
    }

    #[inline(never)]
    fn piece(&mut self, engine: &Engine<'_>, piece: &[u8]) {
        let vocab = engine.vocab;
        self.0 += if vocab.contains_key(piece) {
            1
        } else if (3..=piece_cache::KEY_MAX).contains(&piece.len()) {
            piece_cache::count_piece(engine.cache_id, piece, |buf| {
                merge::bpe_encode_buf(piece, vocab, buf)
            })
        } else {
            merge::bpe_count(piece, vocab)
        };
    }
}

#[inline(always)]
fn run(engine: &Engine<'_>, text: &str, sink: &mut impl Sink) {
    let bytes = text.as_bytes();
    let mut pos = 0;
    while let Some((start, end)) = engine.pre.next_match(text, pos) {
        pos = end;
        // 1..=FAST_KEY_MAX in one compare; an empty match falls through
        if (end - start).wrapping_sub(1) < FAST_KEY_MAX {
            let (lo, hi) = key_in(bytes, start, end);
            if let Some(id) = engine.vocab.get_home(lo, hi) {
                sink.token(id);
                continue;
            }
        }
        sink.piece(engine, &bytes[start..end]);
    }
}

impl Engine<'_> {
    pub(crate) fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        run(self, text, &mut Encode(out));
    }

    pub(crate) fn count(&self, text: &str) -> usize {
        let mut sink = Count(0);
        run(self, text, &mut sink);
        sink.0
    }
}
