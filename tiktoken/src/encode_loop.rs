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

use crate::merge;
use crate::piece_cache;
use crate::pretokenize::window::{CaseRules, Cl100kRules};
use crate::pretokenize::windowed::{self, PieceSink};
use crate::pretokenize::{PreTokenizer, ScalarPreTokenizer, WindowKind};
use crate::vocab::{FAST_KEY_MAX, KEY_READ, Vocab, key_in, key_of, key_padded};

// a real key never has a zero high word
const NO_KEY: (u64, u64) = (0, 0);

pub(crate) struct Engine<'a> {
    pub(crate) vocab: &'a Vocab,
    pub(crate) pre: &'a ScalarPreTokenizer,
    pub(crate) cache_id: u64,
}

/// Texts shorter than this skip the window rules: classifying costs more than
/// it saves on a handful of pieces.
const WINDOW_MIN: usize = 32;

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

/// The fast-table probe of a piece: `Ok(id)` on a hit in the home line,
/// otherwise the key to hand to the miss path (`NO_KEY` past
/// [`FAST_KEY_MAX`] bytes).
#[inline(always)]
fn probe(vocab: &Vocab, buf: &[u8], start: usize, end: usize) -> Result<u32, (u64, u64)> {
    let len = end - start;
    if len > FAST_KEY_MAX {
        return Err(NO_KEY);
    }
    let (lo, hi) = key_padded(buf, start, len);
    vocab.get_home(lo, hi).ok_or((lo, hi))
}

/// Writes ids through a raw pointer; every window first makes room for its
/// at most 64 pieces.
struct EncodeSink<'e, 'v> {
    engine: &'e Engine<'e>,
    out: &'v mut Vec<u32>,
    ptr: *mut u32,
    len: usize,
    cap: usize,
}

impl<'e, 'v> EncodeSink<'e, 'v> {
    fn new(engine: &'e Engine<'e>, out: &'v mut Vec<u32>) -> Self {
        out.reserve(out.len() / 4 + 64);
        Self {
            engine,
            ptr: out.as_mut_ptr(),
            len: out.len(),
            cap: out.capacity(),
            out,
        }
    }

    #[inline(always)]
    fn sync(&mut self) {
        // SAFETY: `len` ids were written below `cap`
        unsafe { self.out.set_len(self.len) }
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        self.sync();
        self.out.reserve(64.max(self.len / 2));
        self.ptr = self.out.as_mut_ptr();
        self.cap = self.out.capacity();
    }

    #[inline(always)]
    fn finish(mut self) {
        self.sync();
    }
}

impl PieceSink for EncodeSink<'_, '_> {
    #[inline(always)]
    fn window(&mut self) {
        if self.cap - self.len < 64 {
            self.grow();
        }
    }

    #[inline(always)]
    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe(self.engine.vocab, buf, start, end) {
            Ok(id) => {
                debug_assert!(self.len < self.cap);
                // SAFETY: `window` left room for every piece of the window,
                // and `miss` restores that room
                unsafe { self.ptr.add(self.len).write(id) };
                self.len += 1;
            }
            Err(key) => self.miss(&buf[start..end], key),
        }
    }
}

impl EncodeSink<'_, '_> {
    #[inline(never)]
    fn miss(&mut self, piece: &[u8], key: (u64, u64)) {
        self.sync();
        self.engine.encode_piece(piece, key, self.out);
        self.len = self.out.len();
        self.out.reserve(64);
        self.ptr = self.out.as_mut_ptr();
        self.cap = self.out.capacity();
    }
}

struct CountSink<'e> {
    engine: &'e Engine<'e>,
    count: usize,
}

impl PieceSink for CountSink<'_> {
    #[inline(always)]
    fn window(&mut self) {}

    #[inline(always)]
    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe(self.engine.vocab, buf, start, end) {
            Ok(_) => self.count += 1,
            Err(key) => self.count += self.engine.count_piece(&buf[start..end], key),
        }
    }
}
