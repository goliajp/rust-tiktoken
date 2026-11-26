//! The windowed scan's sink for encoding with token ends: the encode sink's
//! raw-pointer writes, with each id's end offset written next to it.
//!
//! Pieces are contiguous, so the sink carries the text offset of the next
//! piece itself; the splitter's offsets may be into a padded copy of the
//! text's last bytes. A piece answered by one token ends where the piece
//! ends; the ids of any other piece get their ends from the token lengths.

use crate::encode_loop::{Engine, NO_KEY, piece_key, probe, probe_entry};
use crate::encode_sink::ROOM;
use crate::pretokenize::windowed::PieceSink;
use crate::vocab::{Entry, Home, HomeProbe, LongIds, Vocab, copy_ids};

pub(crate) struct EndsSink<'e, 'v> {
    engine: &'e Engine<'e>,
    home: HomeProbe<'e>,
    ids: &'v mut Vec<u32>,
    ends: &'v mut Vec<usize>,
    // where the next id and end go: past the vectors' lengths by `n`
    ip: *mut u32,
    ep: *mut usize,
    n: usize,
    // spare room at `ip` and `ep` when they were taken
    room: usize,
    // text offset of the next piece
    at: usize,
}

impl<'e, 'v> EndsSink<'e, 'v> {
    /// Ends are written as `base` plus the offset in the text.
    pub(crate) fn new(
        engine: &'e Engine<'e>,
        ids: &'v mut Vec<u32>,
        ends: &'v mut Vec<usize>,
        base: usize,
    ) -> Self {
        let mut sink = Self {
            engine,
            home: engine.vocab.home_probe(),
            ids,
            ends,
            ip: std::ptr::null_mut(),
            ep: std::ptr::null_mut(),
            n: 0,
            room: 0,
            at: base,
        };
        sink.reload(ROOM);
        sink
    }

    /// Take the `n` written ids and ends into the vectors' lengths.
    #[inline(always)]
    fn sync(&mut self) {
        // SAFETY: `n` ids and ends were written past the lengths, below the
        // capacities
        unsafe {
            self.ids.set_len(self.ids.len() + self.n);
            self.ends.set_len(self.ends.len() + self.n);
        }
        self.n = 0;
    }

    /// Make room for `need` more ids and ends past the lengths, and take the
    /// pointers.
    #[inline(always)]
    fn reload(&mut self, need: usize) {
        debug_assert_eq!(self.n, 0);
        let (il, el) = (self.ids.len(), self.ends.len());
        if self.ids.capacity() - il < need {
            self.ids.reserve(need.max(il / 2));
        }
        if self.ends.capacity() - el < need {
            self.ends.reserve(need.max(el / 2));
        }
        // SAFETY: both offsets are the lengths, within the allocations
        unsafe {
            self.ip = self.ids.as_mut_ptr().add(il);
            self.ep = self.ends.as_mut_ptr().add(el);
        }
        self.room = (self.ids.capacity() - il).min(self.ends.capacity() - el);
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, need: usize) {
        self.sync();
        self.reload(need);
    }

    pub(crate) fn finish(mut self) {
        self.sync();
    }

    /// A piece the home line did not answer: a long piece's shared merge
    /// result when there is one, otherwise the general miss path.
    #[inline(always)]
    fn other(&mut self, piece: &[u8], key: (u64, u64)) {
        let vocab = self.engine.vocab;
        if key == NO_KEY {
            if let Some(ids) = vocab.long_ids(piece) {
                return self.long(ids, piece.len());
            }
        } else if let Some(e) = vocab.entry(key.0, key.1) {
            return self.entry(e, piece.len());
        }
        self.slow(piece, key);
    }

    /// A piece's table entry, of up to 15 ids.
    #[inline(always)]
    fn entry(&mut self, e: Entry, len: usize) {
        if self.room - self.n < ROOM {
            self.grow(ROOM);
        }
        let at = self.at;
        self.at += len;
        // SAFETY: an entry has at most 15 ids, and `write_entry` writes at
        // most two more; ROOM is more than that
        unsafe {
            let ip = self.ip.add(self.n);
            self.engine.vocab.write_entry(e, ip);
            entry_ends(self.engine.vocab, e, ip, self.ep.add(self.n), at, self.at);
        }
        self.n += e.len();
    }

    #[inline(always)]
    fn long(&mut self, ids: LongIds<'_>, len: usize) {
        let k = ids.len();
        if self.room - self.n < k + ROOM {
            self.grow(k + ROOM);
        }
        let at = self.at;
        self.at += len;
        // SAFETY: room for `k` ids and ends was made above
        unsafe {
            let ip = self.ip.add(self.n);
            copy_ids(ids.as_ptr(), ip, k);
            piece_ends(self.engine.vocab, ip, self.ep.add(self.n), k, at, self.at);
        }
        self.n += k;
    }

    /// The ids of a piece that the loops did not answer, by the general miss
    /// path, and their ends.
    #[inline(never)]
    fn slow(&mut self, piece: &[u8], key: (u64, u64)) {
        self.sync();
        let k = self.ids.len();
        self.engine.encode_piece(piece, key, self.ids);
        let at = self.at;
        self.at += piece.len();
        let new = &self.ids[k..];
        self.ends.reserve(new.len());
        let mut e = at;
        for &id in &new[..new.len() - 1] {
            e += self.engine.vocab.token_len(id);
            self.ends.push(e);
        }
        self.ends.push(self.at);
        debug_assert_eq!(e + self.engine.vocab.token_len(new[new.len() - 1]), self.at);
        self.reload(ROOM);
    }
}

/// Write the ends of `k` ids at `ip`, the ids of the piece `at..end`, at `ep`.
///
/// # Safety
///
/// `k >= 1`, `k` ids are readable at `ip`, and `k` ends writable at `ep`.
#[inline(always)]
unsafe fn piece_ends(
    vocab: &Vocab,
    ip: *const u32,
    ep: *mut usize,
    k: usize,
    at: usize,
    end: usize,
) {
    let mut e = at;
    // SAFETY: passed on from the caller
    unsafe {
        for j in 0..k - 1 {
            e += vocab.token_len(*ip.add(j));
            ep.add(j).write(e);
        }
        ep.add(k - 1).write(end);
    }
}

/// [`piece_ends`] of an entry's ids just written at `ip`, without a loop
/// for an entry of up to three ids.
///
/// # Safety
///
/// The entry's ids are readable at `ip`, and `max(3, e.len())` ends
/// writable at `ep`.
#[inline(always)]
unsafe fn entry_ends(
    vocab: &Vocab,
    e: Entry,
    ip: *const u32,
    ep: *mut usize,
    at: usize,
    end: usize,
) {
    let k = e.len();
    // SAFETY: passed on from the caller
    unsafe {
        if !e.is_inline() {
            return piece_ends(vocab, ip, ep, k, at, end);
        }
        // the words past the entry's ids are arbitrary: the second id is
        // read only when there are three
        let i0 = *ip;
        let i1 = if k == 3 { *ip.add(1) } else { i0 };
        let e0 = at + vocab.token_len(i0);
        let e1 = e0 + vocab.token_len(i1);
        ep.write(if k == 1 { end } else { e0 });
        ep.add(1).write(if k == 2 { end } else { e1 });
        ep.add(2).write(end);
    }
}

impl PieceSink for EndsSink<'_, '_> {
    #[inline(always)]
    fn window(&mut self) {
        if self.room - self.n < ROOM {
            self.grow(ROOM);
        }
    }

    #[inline(never)]
    fn pieces(&mut self, buf: &[u8], start: usize, offs: &[u16]) {
        let last = offs.len() - 1;
        // the text offset of a buffer offset `x` is `x + delta`
        let delta = self.at.wrapping_sub(start + offs[0] as usize);
        let vocab = self.engine.vocab;
        let home = self.home;
        let mut i = 0;
        while i < last {
            let left = last - i;
            if self.room - self.n < 3 * left.min(16) {
                self.grow(3 * left);
            }
            let stop = i + left.min((self.room - self.n) / 3);
            let (ip, ep) = (self.ip, self.ep);
            let mut n = self.n;
            let mut key = NO_KEY;
            let mut entry = None;
            while i < stop {
                let a = start + offs[i] as usize;
                let b = start + offs[i + 1] as usize;
                let Some((lo, hi)) = piece_key(buf, a, b) else {
                    break;
                };
                let found = match home.lookup(lo, hi) {
                    Home::Miss => home.lookup_chain(lo, hi),
                    h => h,
                };
                // SAFETY (writes): room for three ids and ends per piece up
                // to `stop` was made above
                match found {
                    Home::Token(id) => unsafe {
                        ip.add(n).write(id);
                        ep.add(n).write(b.wrapping_add(delta));
                        n += 1;
                    },
                    Home::Derived(e) if e.is_inline() => unsafe {
                        vocab.write_entry(e, ip.add(n));
                        let (at, end) = (a.wrapping_add(delta), b.wrapping_add(delta));
                        entry_ends(vocab, e, ip.add(n), ep.add(n), at, end);
                        n += e.len();
                    },
                    Home::Derived(e) => {
                        entry = Some(e);
                        break;
                    }
                    Home::Miss => {
                        key = (lo, hi);
                        break;
                    }
                }
                i += 1;
            }
            self.n = n;
            self.at = (start + offs[i] as usize).wrapping_add(delta);
            if i < stop {
                let piece = &buf[start + offs[i] as usize..start + offs[i + 1] as usize];
                match entry {
                    Some(e) => self.entry(e, piece.len()),
                    // a short piece's chain was walked: it is not in the table
                    None if key != NO_KEY => self.slow(piece, key),
                    None => self.other(piece, key),
                }
                i += 1;
            }
        }
    }

    #[inline(always)]
    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe(&self.home, buf, start, end) {
            Ok(id) => {
                debug_assert!(self.n < self.room);
                self.at += end - start;
                // SAFETY: `window` left room for every piece of the window,
                // and `slow` restores that room
                unsafe {
                    self.ip.add(self.n).write(id);
                    self.ep.add(self.n).write(self.at);
                }
                self.n += 1;
            }
            Err(key) => self.other(&buf[start..end], key),
        }
    }

    #[inline(always)]
    fn dense_piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe_entry(self.engine.vocab, buf, start, end) {
            Ok(e) => {
                debug_assert!(self.n + 15 <= self.room);
                let k = e.len();
                let at = self.at;
                self.at += end - start;
                // SAFETY: `window` came right before and left room for 64
                // ids, and a piece of up to 15 bytes has at most 15
                unsafe {
                    let vocab = self.engine.vocab;
                    vocab.write_entry(e, self.ip.add(self.n));
                    let ip = self.ip.add(self.n);
                    entry_ends(vocab, e, ip, self.ep.add(self.n), at, self.at);
                }
                self.n += k;
            }
            Err(key) => self.other(&buf[start..end], key),
        }
    }
}
