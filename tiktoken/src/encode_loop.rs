//! The per-piece encode and count loops.
//!
//! Each pre-token piece of 1..=15 bytes is packed into a two-word key and
//! checked against its home line in the vocabulary's fast table; that answers
//! most pieces of real text with one cache-line read. Everything else (keys
//! further down their probe chain, longer pieces, pieces that need a BPE
//! merge) leaves the loop through an out-of-line call.
//!
//! Patterns with window rules take their pieces from
//! `pretokenize::windowed`, which hands each piece to a sink
//! (`encode_sink`, `count_sink`); the sinks probe the same table, with the
//! key and probe helpers here. Short texts and the other patterns use the scalar
//! loops below.
//!
//! The two scalar loops are written out separately on purpose. The scanner's
//! `next_match` is inlined into each of them in this shape; folding them into
//! one generic or `Option`-driven loop changed that and cost up to 20% on
//! some inputs. `count` stays `#[inline]` so it can inline into its caller.

use crate::count_sink::CountSink;
use crate::encode_miss;
use crate::encode_sink::{EncodeSink, ROOM};
use crate::pretokenize::window::{CaseRules, Cl100kRules};
use crate::pretokenize::windowed::{self, PieceSink};
use crate::pretokenize::{PreTokenizer, ScalarPreTokenizer, WindowKind};
use crate::vocab::{Entry, FAST_KEY_MAX, HomeProbe, KEY_READ, Vocab, key_in, key_of, key_padded};

// a real key never has a zero high word
pub(crate) const NO_KEY: (u64, u64) = (0, 0);

pub(crate) struct Engine<'a> {
    pub(crate) vocab: &'a Vocab,
    pub(crate) pre: &'a ScalarPreTokenizer,
    pub(crate) cache_id: u64,
}

/// Texts shorter than this skip the window rules: classifying costs more than
/// it saves on a handful of pieces.
const WINDOW_MIN: usize = 32;

/// Ids to reserve for encoding `text`: a third of its length, or two thirds
/// when its first 64 bytes are mostly non-ASCII, as dense CJK, Hangul or
/// Arabic text takes 0.35 to 0.65 ids per byte. Texts for the window sink get
/// its [`ROOM`] on top, which it keeps free before each window; without it a
/// text of a few hundred bytes grows the vector near its end.
#[inline(always)]
pub(crate) fn id_capacity(text: &str) -> usize {
    let b = text.as_bytes();
    if b.len() < WINDOW_MIN {
        return b.len() / 3;
    }
    let dense = b.first_chunk::<64>().is_some_and(|head| {
        let hi: u32 = head
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| (u64::from_le_bytes(*w) & 0x8080_8080_8080_8080).count_ones())
            .sum();
        hi >= 16
    });
    let ids = if dense { b.len() / 3 * 2 } else { b.len() / 3 };
    ids + ROOM
}

impl Engine<'_> {
    pub(crate) fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.len() >= WINDOW_MIN
            && let Some(kind) = self.pre.window_kind()
        {
            let mut sink = EncodeSink::new(self, out);
            window_dispatch(self.pre, kind, text, &mut sink);
            sink.finish();
            return;
        }
        self.encode_scalar(text, out);
    }

    #[inline]
    pub(crate) fn count(&self, text: &str) -> usize {
        if text.len() >= WINDOW_MIN
            && let Some(kind) = self.pre.window_kind()
        {
            let mut sink = CountSink {
                engine: self,
                home: self.vocab.home_probe(),
                count: 0,
            };
            window_dispatch(self.pre, kind, text, &mut sink);
            return sink.count;
        }
        self.count_scalar(text)
    }

    fn encode_scalar(&self, text: &str, out: &mut Vec<u32>) {
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
    fn count_scalar(&self, text: &str) -> usize {
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
    pub(crate) fn encode_piece(&self, piece: &[u8], key: (u64, u64), out: &mut Vec<u32>) {
        encode_miss::encode(self.vocab, self.cache_id, piece, Self::miss_key(key), out);
    }

    #[inline(never)]
    pub(crate) fn count_piece(&self, piece: &[u8], key: (u64, u64)) -> usize {
        encode_miss::count(self.vocab, self.cache_id, piece, Self::miss_key(key))
    }
}

#[inline(always)]
fn window_dispatch<S: PieceSink>(
    pre: &ScalarPreTokenizer,
    kind: WindowKind,
    text: &str,
    sink: &mut S,
) {
    match kind {
        WindowKind::Cl100k => windowed::split::<Cl100kRules<3, false>, S>(pre, text, sink),
        WindowKind::Qwen2 => windowed::split::<Cl100kRules<1, false>, S>(pre, text, sink),
        WindowKind::Qwen35 => windowed::split::<Cl100kRules<1, true>, S>(pre, text, sink),
        WindowKind::O200k => windowed::split::<CaseRules<true, 3>, S>(pre, text, sink),
        WindowKind::Tekken => windowed::split::<CaseRules<false, 1>, S>(pre, text, sink),
    }
}

/// Masks of the two key words by key length, one load for both.
#[repr(C, align(16))]
struct KeyMasks([[u64; 2]; FAST_KEY_MAX + 1]);

static KEY_MASKS: KeyMasks = {
    let mut m = [[0u64; 2]; FAST_KEY_MAX + 1];
    let mut len = 1;
    while len <= FAST_KEY_MAX {
        m[len] = if len < 8 {
            [(1u64 << (8 * len)) - 1, 0]
        } else {
            [u64::MAX, (1u64 << (8 * (len - 8))) - 1]
        };
        len += 1;
    }
    KeyMasks(m)
};

/// The key of `buf[start..end]` (equal to [`key_padded`]'s), or `None` past
/// [`FAST_KEY_MAX`] bytes.
#[inline(always)]
pub(crate) fn piece_key(buf: &[u8], start: usize, end: usize) -> Option<(u64, u64)> {
    let len = end - start;
    if len > FAST_KEY_MAX {
        return None;
    }
    debug_assert!(start + KEY_READ <= buf.len());
    // SAFETY: the sink contract leaves KEY_READ bytes readable from `start`
    let w = unsafe { buf.as_ptr().add(start).cast::<[u64; 2]>().read_unaligned() };
    let m = &KEY_MASKS.0[len];
    let lo = u64::from_le(w[0]) & m[0];
    let hi = (u64::from_le(w[1]) & m[1]) | ((len as u64) << 56);
    debug_assert_eq!((lo, hi), key_padded(buf, start, len));
    Some((lo, hi))
}

/// The fast-table probe of a piece: `Ok(id)` on a token hit in the home line,
/// otherwise the key to hand to the miss path (`NO_KEY` past
/// [`FAST_KEY_MAX`] bytes). The key equals [`key_padded`]'s.
#[inline(always)]
pub(crate) fn probe(
    home: &HomeProbe<'_>,
    buf: &[u8],
    start: usize,
    end: usize,
) -> Result<u32, (u64, u64)> {
    let len = end - start;
    if len > FAST_KEY_MAX {
        return Err(NO_KEY);
    }
    debug_assert!(start + KEY_READ <= buf.len());
    // SAFETY: the sink contract leaves KEY_READ bytes readable from `start`
    let w = unsafe { buf.as_ptr().add(start).cast::<[u64; 2]>().read_unaligned() };
    let m = &KEY_MASKS.0[len];
    let lo = u64::from_le(w[0]) & m[0];
    let hi = (u64::from_le(w[1]) & m[1]) | ((len as u64) << 56);
    debug_assert_eq!((lo, hi), key_padded(buf, start, len));
    home.home(lo, hi).ok_or((lo, hi))
}

/// The table entry of a piece, derived entries included, wherever it sits on
/// its probe chain; otherwise the key for the miss path.
#[inline(always)]
pub(crate) fn probe_entry(
    vocab: &Vocab,
    buf: &[u8],
    start: usize,
    end: usize,
) -> Result<Entry, (u64, u64)> {
    let len = end - start;
    if len > FAST_KEY_MAX {
        return Err(NO_KEY);
    }
    let (lo, hi) = key_padded(buf, start, len);
    vocab.entry(lo, hi).ok_or((lo, hi))
}

#[cfg(test)]
mod tests {
    use crate::merge;
    use crate::pretokenize::PreTokenizer;

    // pieces that are not tokens, merging into one to many ids, repeated so
    // that later copies hit the shared entries the first copies published
    #[test]
    fn repeated_pieces_match_the_merge() {
        let words = [
            "zqxj",
            " qzvvx",
            " xkcdqz",
            "Qzxwv'",
            "qqqqqqqqqqqqqqq",
            " zxqvkwjqpzxqv",
            "!!?!?",
            "1234567",
            "\n\n\t  \n",
            " ʰqzʰ",
        ];
        let mut text = String::new();
        for i in 0..900 {
            text.push_str(words[i % words.len()]);
            text.push_str(" the");
        }
        for name in ["cl100k_base", "o200k_base", "qwen3_5", "mistral_v3"] {
            let bpe = crate::get_encoding(name).unwrap();
            let mut want = vec![];
            let mut pos = 0;
            while let Some((s, e)) = bpe.pre_tokenizer().next_match(&text, pos) {
                merge::bpe_encode(&text.as_bytes()[s..e], bpe.vocab(), &mut want);
                pos = e;
            }
            for _ in 0..3 {
                assert_eq!(bpe.encode(&text), want, "{name}");
                assert_eq!(bpe.count(&text), want.len(), "{name}");
            }
        }
    }
}
