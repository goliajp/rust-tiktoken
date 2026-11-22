//! The per-piece encode and count loops.
//!
//! Each pre-token piece of 1..=15 bytes is packed into a two-word key and
//! checked against its home line in the vocabulary's fast table; that answers
//! most pieces of real text with one cache-line read. Everything else (keys
//! further down their probe chain, longer pieces, pieces that need a BPE
//! merge) leaves the loop through an out-of-line call.
//!
//! Patterns with window rules take their pieces from
//! `pretokenize::windowed`, which hands each piece to a sink here; the sinks
//! probe the same table. Short texts and the other patterns use the scalar
//! loops below.
//!
//! The two scalar loops are written out separately on purpose. The scanner's
//! `next_match` is inlined into each of them in this shape; folding them into
//! one generic or `Option`-driven loop changed that and cost up to 20% on
//! some inputs. `count` stays `#[inline]` so it can inline into its caller.

use crate::count_sink::CountSink;
use crate::encode_miss;
use crate::pretokenize::window::{CaseRules, Cl100kRules};
use crate::pretokenize::windowed::{self, PieceSink};
use crate::pretokenize::{PreTokenizer, ScalarPreTokenizer, WindowKind};
use crate::vocab::{
    Entry, FAST_KEY_MAX, Home, HomeProbe, KEY_READ, LongIds, Vocab, copy_ids, key_in, key_of,
    key_padded,
};

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
/// Arabic text takes 0.35 to 0.65 ids per byte and would otherwise grow the
/// vector mid-scan.
#[inline(always)]
pub(crate) fn id_capacity(text: &str) -> usize {
    let b = text.as_bytes();
    let Some(head) = b.first_chunk::<64>() else {
        return b.len() / 3;
    };
    let hi: u32 = head
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| (u64::from_le_bytes(*w) & 0x8080_8080_8080_8080).count_ones())
        .sum();
    if hi >= 16 {
        b.len() / 3 * 2
    } else {
        b.len() / 3
    }
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
    fn encode_piece(&self, piece: &[u8], key: (u64, u64), out: &mut Vec<u32>) {
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

/// Room an [`EncodeSink`] keeps before each window: a window has at most 64
/// pieces, a home-line hit writes one id, and the first miss may write three
/// ids' worth, after which [`encode_miss`] restores this room. A dense piece
/// right after a window writes at most 15.
const ROOM: usize = 64 + 2;

/// Writes ids through a raw pointer. Batches of pieces go through a leaf loop
/// that writes up to three ids per piece, as many pieces at a time as the
/// spare capacity holds; single pieces come after a
/// [`window`](PieceSink::window) call that makes [`ROOM`].
///
/// The out-of-line paths take the fields by value rather than `&mut self`,
/// so that the sink does not escape and its fields stay in registers in the
/// piece loop.
struct EncodeSink<'e, 'v> {
    engine: &'e Engine<'e>,
    home: HomeProbe<'e>,
    out: &'v mut Vec<u32>,
    ptr: *mut u32,
    len: usize,
    cap: usize,
}

impl<'e, 'v> EncodeSink<'e, 'v> {
    fn new(engine: &'e Engine<'e>, out: &'v mut Vec<u32>) -> Self {
        out.reserve(ROOM);
        Self {
            engine,
            home: engine.vocab.home_probe(),
            ptr: out.as_mut_ptr(),
            len: out.len(),
            cap: out.capacity(),
            out,
        }
    }

    /// Take `ptr` and `cap` from `out` after an out-of-line call.
    #[inline(always)]
    fn reload(&mut self) {
        self.ptr = self.out.as_mut_ptr();
        self.cap = self.out.capacity();
    }

    #[inline(always)]
    fn finish(self) {
        // SAFETY: `len` ids were written below `cap`
        unsafe { self.out.set_len(self.len) }
    }
}

/// Make room for `need` more ids after `len` ids.
#[cold]
#[inline(never)]
fn grow(out: &mut Vec<u32>, len: usize, need: usize) {
    // SAFETY: `len` ids were written below the capacity
    unsafe { out.set_len(len) };
    out.reserve(need.max(len / 2));
}

/// Writes the ids of consecutive pieces `offs[i]..offs[i + 1]` from `i` on
/// while their home lines answer them (a token, or a derived entry of up to
/// three ids), after `len` ids at `ptr`. Stops at the first piece it does
/// not answer, and returns that piece's index (`offs.len() - 1` when all are
/// done), the new length and the piece's key. A leaf, so the loop keeps
/// everything in registers.
#[inline(never)]
fn encode_hits(
    home: HomeProbe<'_>,
    buf: &[u8],
    start: usize,
    offs: &[u16],
    mut i: usize,
    ptr: *mut u32,
    mut len: usize,
) -> (usize, usize, (u64, u64)) {
    while i + 1 < offs.len() {
        // SAFETY: i + 1 < offs.len()
        let (a, b) = unsafe { (*offs.get_unchecked(i), *offs.get_unchecked(i + 1)) };
        let Some((lo, hi)) = piece_key(buf, start + a as usize, start + b as usize) else {
            return (i, len, NO_KEY);
        };
        // a key off its home line is mostly on the next one; walking the
        // chain here keeps the piece in the leaf
        let found = match home.lookup(lo, hi) {
            Home::Miss => home.lookup_chain(lo, hi),
            h => h,
        };
        // SAFETY (writes): the caller made room for three ids per piece
        match found {
            Home::Token(id) => {
                unsafe { ptr.add(len).write(id) };
                len += 1;
            }
            // ids past the entry's count are overwritten later or left past
            // the end
            Home::Derived(e) if e.is_inline() => {
                unsafe {
                    let p = ptr.add(len);
                    p.write(e.first());
                    p.add(1).cast::<u64>().write_unaligned(e.rest().to_le());
                }
                len += e.len();
            }
            _ => return (i, len, (lo, hi)),
        }
        i += 1;
    }
    (i, len, NO_KEY)
}

/// A piece the home-line probe did not answer, written after `len` ids at
/// `ptr`; returns the new length, with room for the rest of the window.
#[inline(never)]
fn encode_miss(
    engine: &Engine<'_>,
    out: &mut Vec<u32>,
    ptr: *mut u32,
    len: usize,
    piece: &[u8],
    key: (u64, u64),
) -> usize {
    let entry = if key != NO_KEY {
        engine.vocab.entry(key.0, key.1)
    } else {
        None
    };
    if let Some(e) = entry
        && e.is_inline()
    {
        debug_assert!(len + 3 <= out.capacity());
        // SAFETY: the window left three ids of room for every piece; ids past
        // the entry's count are overwritten later or left past the end
        unsafe {
            let p = ptr.add(len);
            p.write(e.first());
            p.add(1).cast::<u64>().write_unaligned(e.rest().to_le());
        }
        return len + e.len();
    }
    // SAFETY: `len` ids were written below the capacity
    unsafe { out.set_len(len) };
    match entry {
        Some(e) => engine.vocab.extend_entry(e, out),
        None => engine.encode_piece(piece, key, out),
    }
    if out.capacity() - out.len() < ROOM {
        out.reserve(ROOM);
    }
    out.len()
}

impl PieceSink for EncodeSink<'_, '_> {
    #[inline(always)]
    fn window(&mut self) {
        if self.cap - self.len < ROOM {
            grow(self.out, self.len, ROOM);
            self.reload();
        }
    }

    #[inline(always)]
    fn pieces(&mut self, buf: &[u8], start: usize, offs: &[u16]) {
        let last = offs.len() - 1;
        let mut i = 0;
        while i < last {
            // grow only when fewer than 64 pieces (or the rest) fit: growing
            // for the whole batch up front reallocated short texts that
            // never needed it
            let left = last - i;
            if self.cap - self.len < 3 * left.min(64) {
                grow(self.out, self.len, 3 * left);
                self.reload();
            }
            let end = i + left.min((self.cap - self.len) / 3);
            let batch = &offs[..=end];
            let (at, len, key) = encode_hits(self.home, buf, start, batch, i, self.ptr, self.len);
            self.len = len;
            i = at;
            if i < end {
                let piece = &buf[start + offs[i] as usize..start + offs[i + 1] as usize];
                self.len = encode_miss(self.engine, self.out, self.ptr, self.len, piece, key);
                self.reload();
                i += 1;
            }
        }
    }

    #[inline(always)]
    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe(&self.home, buf, start, end) {
            Ok(id) => {
                debug_assert!(self.len < self.cap);
                // SAFETY: `window` left room for every piece of the window,
                // and `encode_miss` restores that room
                unsafe { self.ptr.add(self.len).write(id) };
                self.len += 1;
            }
            Err(key) => {
                let piece = &buf[start..end];
                self.len = encode_miss(self.engine, self.out, self.ptr, self.len, piece, key);
                self.reload();
            }
        }
    }

    #[inline(always)]
    fn dense_piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe_entry(self.engine.vocab, buf, start, end) {
            Ok(e) => {
                debug_assert!(self.len + 15 <= self.cap);
                // SAFETY: `window` came right before and left room for 64
                // ids, and a piece of up to 15 bytes has at most 15
                unsafe { self.engine.vocab.write_entry(e, self.ptr.add(self.len)) };
                self.len += e.len();
            }
            Err(NO_KEY) => {
                let piece = &buf[start..end];
                match self.engine.vocab.long_ids(piece) {
                    Some(ids) => self.long(ids),
                    None => self.miss(piece, NO_KEY),
                }
            }
            Err(key) => self.miss(&buf[start..end], key),
        }
    }
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

impl EncodeSink<'_, '_> {
    #[inline(always)]
    fn long(&mut self, ids: LongIds<'_>) {
        let n = ids.len();
        if self.cap - self.len < n + ROOM {
            grow(self.out, self.len, n + ROOM);
            self.reload();
        }
        // SAFETY: room for `n + ROOM` ids was made above, and the ids are
        // not written concurrently
        unsafe { copy_ids(ids.as_ptr(), self.ptr.add(self.len), n) };
        self.len += n;
    }

    #[inline(always)]
    fn miss(&mut self, piece: &[u8], key: (u64, u64)) {
        self.len = encode_miss(self.engine, self.out, self.ptr, self.len, piece, key);
        self.reload();
    }
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
