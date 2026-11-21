//! The per-piece encode and count loops.
//!
//! Each pre-token piece of 1..=15 bytes is packed into a two-word key and
//! checked against its home line in the vocabulary's fast table; that answers
//! most pieces of real text with one cache-line read. Everything else (keys
//! further down their probe chain, longer pieces, pieces that need a BPE
//! merge) leaves the loop through an out-of-line call.
//!
//! The two loops are written out separately on purpose. The scanner's
//! `next_match` is inlined into each of them in this shape; folding them into
//! one generic or `Option`-driven loop changed that and cost up to 20% on
//! some inputs. `count` stays `#[inline]` so it can inline into its caller.

use crate::encode_miss;
use crate::pretokenize::{PreTokenizer, ScalarPreTokenizer};
use crate::vocab::{FAST_KEY_MAX, KEY_READ, Vocab, key_in, key_of};

// a real key never has a zero high word
const NO_KEY: (u64, u64) = (0, 0);

pub(crate) struct Engine<'a> {
    pub(crate) vocab: &'a Vocab,
    pub(crate) pre: &'a ScalarPreTokenizer,
    pub(crate) cache_id: u64,
}

impl Engine<'_> {
    pub(crate) fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let wide = bytes.len() >= KEY_READ;
        let mut pos = 0;
        while let Some((start, end)) = self.pre.next_match(text, pos) {
            pos = end;
            let mut key = NO_KEY;
            if (end - start).wrapping_sub(1) < FAST_KEY_MAX {
                key = if wide {
                    key_in(bytes, start, end)
                } else {
                    key_of(&bytes[start..end])
                };
                if let Some(id) = self.vocab.get_home(key.0, key.1) {
                    out.push(id);
                    continue;
                }
            }
            self.encode_piece(&bytes[start..end], key, out);
        }
    }

    #[inline]
    pub(crate) fn count(&self, text: &str) -> usize {
        let bytes = text.as_bytes();
        let wide = bytes.len() >= KEY_READ;
        let mut count = 0;
        let mut pos = 0;
        while let Some((start, end)) = self.pre.next_match(text, pos) {
            pos = end;
            let mut key = NO_KEY;
            if (end - start).wrapping_sub(1) < FAST_KEY_MAX {
                key = if wide {
                    key_in(bytes, start, end)
                } else {
                    key_of(&bytes[start..end])
                };
                if self.vocab.get_home(key.0, key.1).is_some() {
                    count += 1;
                    continue;
                }
            }
            count += self.count_piece(&bytes[start..end], key);
        }
        count
    }

    #[inline(always)]
    fn miss_key(key: (u64, u64)) -> encode_miss::Key {
        (key != NO_KEY).then_some(key)
    }

    #[inline(never)]
    fn encode_piece(&self, piece: &[u8], key: (u64, u64), out: &mut Vec<u32>) {
        encode_miss::encode(self.vocab, self.cache_id, piece, Self::miss_key(key), out);
    }

    #[inline(never)]
    fn count_piece(&self, piece: &[u8], key: (u64, u64)) -> usize {
        encode_miss::count(self.vocab, self.cache_id, piece, Self::miss_key(key))
    }
}
