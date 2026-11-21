//! The per-piece encode and count loop.
//!
//! Each pre-token piece of 1..=15 bytes is packed into a two-word key and
//! checked against its home line in the vocabulary's fast table; that answers
//! most pieces of real text with one cache-line read. Everything else (keys
//! further down their probe chain, longer pieces, pieces that need a BPE
//! merge) leaves the loop through an out-of-line call.
//!
//! Encode and count share one non-generic loop: with a single caller, the
//! scanner's `next_match` is inlined into it, which a second copy of the loop
//! prevented.

use crate::merge;
use crate::piece_cache;
use crate::pretokenize::{PreTokenizer, RegexPreTokenizer};
use crate::vocab::{FAST_KEY_MAX, KEY_READ, Vocab, key_in, key_of};

// a real key never has a zero high word
const NO_KEY: (u64, u64) = (0, 0);

pub(crate) struct Engine<'a> {
    pub(crate) vocab: &'a Vocab,
    pub(crate) pre: &'a RegexPreTokenizer,
    pub(crate) cache_id: u64,
}

impl Engine<'_> {
    pub(crate) fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        self.run(text, Some(out));
    }

    pub(crate) fn count(&self, text: &str) -> usize {
        self.run(text, None)
    }

    /// Appends the tokens to `out` when given; otherwise returns their count.
    #[inline(never)]
    fn run(&self, text: &str, mut out: Option<&mut Vec<u32>>) -> usize {
        let bytes = text.as_bytes();
        // the shift-based tail load needs 16 readable bytes
        let wide = bytes.len() >= KEY_READ;
        let mut count = 0;
        let mut pos = 0;
        while let Some((start, end)) = self.pre.next_match(text, pos) {
            pos = end;
            // 1..=FAST_KEY_MAX in one compare; an empty match falls through
            let mut key = NO_KEY;
            if (end - start).wrapping_sub(1) < FAST_KEY_MAX {
                key = if wide {
                    key_in(bytes, start, end)
                } else {
                    key_of(&bytes[start..end])
                };
                if let Some(id) = self.vocab.get_home(key.0, key.1) {
                    match out {
                        Some(ref mut v) => v.push(id),
                        None => count += 1,
                    }
                    continue;
                }
            }
            let piece = &bytes[start..end];
            match out {
                Some(ref mut v) => self.encode_piece(piece, key, v),
                None => count += self.count_piece(piece, key),
            }
        }
        count
    }

    /// Rank of a piece whose home-line probe missed, reusing its packed key.
    #[inline(always)]
    fn rank(&self, piece: &[u8], key: (u64, u64)) -> Option<u32> {
        if key == NO_KEY {
            self.vocab.get(piece)
        } else {
            self.vocab.get_key(key.0, key.1)
        }
    }

    #[inline(never)]
    fn encode_piece(&self, piece: &[u8], key: (u64, u64), out: &mut Vec<u32>) {
        let vocab = self.vocab;
        if let Some(token) = self.rank(piece, key) {
            out.push(token);
        } else if (3..=piece_cache::KEY_MAX).contains(&piece.len()) {
            piece_cache::encode_piece(self.cache_id, piece, out, |buf| {
                merge::bpe_encode_buf(piece, vocab, buf)
            });
        } else {
            merge::bpe_encode(piece, vocab, out);
        }
    }

    #[inline(never)]
    fn count_piece(&self, piece: &[u8], key: (u64, u64)) -> usize {
        let vocab = self.vocab;
        if self.rank(piece, key).is_some() {
            1
        } else if (3..=piece_cache::KEY_MAX).contains(&piece.len()) {
            piece_cache::count_piece(self.cache_id, piece, |buf| {
                merge::bpe_encode_buf(piece, vocab, buf)
            })
        } else {
            merge::bpe_count(piece, vocab)
        }
    }
}
