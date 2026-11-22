//! The encoding sink of the windowed scan: writes ids through a raw pointer
//! into the output vector, keeping enough spare capacity that the piece loops
//! never check it per id.

use crate::encode_loop::{Engine, NO_KEY, piece_key, probe, probe_entry};
use crate::pretokenize::windowed::PieceSink;
use crate::vocab::{Entry, Home, HomeProbe, LongIds, copy_ids};

/// Room an [`EncodeSink`] keeps before each window: a window has at most 64
/// pieces, a home-line hit writes one id, and the first miss may write three
/// ids' worth, after which [`encode_miss`] restores this room. A dense piece
/// right after a window writes at most 15.
pub(crate) const ROOM: usize = 64 + 2;

/// Writes ids through a raw pointer. Batches of pieces go through a leaf loop
/// that writes up to three ids per piece, as many pieces at a time as the
/// spare capacity holds; single pieces come after a
/// [`window`](PieceSink::window) call that makes [`ROOM`].
///
/// The out-of-line paths take the fields by value rather than `&mut self`,
/// so that the sink does not escape and its fields stay in registers in the
/// piece loop.
pub(crate) struct EncodeSink<'e, 'v> {
    engine: &'e Engine<'e>,
    home: HomeProbe<'e>,
    out: &'v mut Vec<u32>,
    ptr: *mut u32,
    len: usize,
    cap: usize,
}

impl<'e, 'v> EncodeSink<'e, 'v> {
    pub(crate) fn new(engine: &'e Engine<'e>, out: &'v mut Vec<u32>) -> Self {
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
    pub(crate) fn finish(self) {
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
/// while the table answers them (a token, or a derived entry of up to three
/// ids), after `len` ids at `ptr`. Stops at the first piece it does not
/// answer, and returns that piece's index (`offs.len() - 1` when all are
/// done), the new length, the piece's key and its entry if it has one. A
/// leaf, so the loop keeps everything in registers.
#[inline(never)]
fn encode_hits(
    home: HomeProbe<'_>,
    buf: &[u8],
    start: usize,
    offs: &[u16],
    mut i: usize,
    ptr: *mut u32,
    mut len: usize,
) -> (usize, usize, (u64, u64), Option<Entry>) {
    while i + 1 < offs.len() {
        // SAFETY: i + 1 < offs.len()
        let (a, b) = unsafe { (*offs.get_unchecked(i), *offs.get_unchecked(i + 1)) };
        let Some((lo, hi)) = piece_key(buf, start + a as usize, start + b as usize) else {
            return (i, len, NO_KEY, None);
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
            Home::Derived(e) => return (i, len, (lo, hi), Some(e)),
            Home::Miss => return (i, len, (lo, hi), None),
        }
        i += 1;
    }
    (i, len, NO_KEY, None)
}

/// [`encode_miss`] for a piece whose probe chain has not been walked yet.
#[inline(never)]
fn encode_unwalked(
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
    encode_miss(engine, out, ptr, len, piece, key, entry)
}

/// A piece the leaf loops did not write, with its table entry if it has one
/// (none for a piece the table does not hold), written after `len` ids at
/// `ptr`; returns the new length, with room for the rest of the window.
#[inline(never)]
fn encode_miss(
    engine: &Engine<'_>,
    out: &mut Vec<u32>,
    ptr: *mut u32,
    len: usize,
    piece: &[u8],
    key: (u64, u64),
    entry: Option<Entry>,
) -> usize {
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

    // out of line: it runs once per batch, and inlined under the window
    // loop it measured slower
    #[inline(never)]
    fn pieces(&mut self, buf: &[u8], start: usize, offs: &[u16]) {
        let last = offs.len() - 1;
        let mut i = 0;
        while i < last {
            // grow only when fewer than 16 pieces (or the rest) fit: growing
            // for the whole batch up front reallocated short texts that
            // never needed it
            let left = last - i;
            if self.cap - self.len < 3 * left.min(16) {
                grow(self.out, self.len, 3 * left);
                self.reload();
            }
            let end = i + left.min((self.cap - self.len) / 3);
            let batch = &offs[..=end];
            let (at, len, key, entry) =
                encode_hits(self.home, buf, start, batch, i, self.ptr, self.len);
            self.len = len;
            i = at;
            if i < end {
                let piece = &buf[start + offs[i] as usize..start + offs[i + 1] as usize];
                let (out, ptr) = (&mut *self.out, self.ptr);
                self.len = encode_miss(self.engine, out, ptr, self.len, piece, key, entry);
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
                self.len = encode_unwalked(self.engine, self.out, self.ptr, self.len, piece, key);
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
            // `probe_entry` walked the chain: the table does not hold it
            Err(key) => self.miss(&buf[start..end], key),
        }
    }
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

    /// A piece the table does not hold.
    #[inline(always)]
    fn miss(&mut self, piece: &[u8], key: (u64, u64)) {
        self.len = encode_miss(self.engine, self.out, self.ptr, self.len, piece, key, None);
        self.reload();
    }
}

#[cfg(test)]
mod tests {
    use crate::merge;
    use crate::pretokenize::PreTokenizer;

    // every length around the sink's room and batch thresholds, for ascii
    // text, dense text whose entries hold many ids, and pieces that miss
    #[test]
    fn prefixes_match_the_merge() {
        let units = [
            "The quick brown fox jumps over the lazy dog. ",
            "你好世界！こんにちは世界！안녕하세요 세계！مرحبا بالعالم ",
            "zqxj qzvvx' 1234567 \n\n ʰqzʰ ",
        ];
        for name in ["cl100k_base", "o200k_base"] {
            let bpe = crate::get_encoding(name).unwrap();
            for unit in units {
                let text = unit.repeat(30);
                for len in 0..text.len().min(1200) {
                    if !text.is_char_boundary(len) {
                        continue;
                    }
                    let t = &text[..len];
                    let mut want = vec![];
                    let mut pos = 0;
                    while let Some((s, e)) = bpe.pre_tokenizer().next_match(t, pos) {
                        merge::bpe_encode(&t.as_bytes()[s..e], bpe.vocab(), &mut want);
                        pos = e;
                    }
                    assert_eq!(bpe.encode(t), want, "{name} {len}");
                    assert_eq!(bpe.count(t), want.len(), "{name} {len}");
                }
            }
        }
    }
}
