//! The counting sink of the windowed scan: the same probes as the encoding
//! sink in `encode_sink`, adding up id counts instead of writing ids.

use crate::encode_loop::{Engine, NO_KEY, piece_key, probe, probe_entry};
use crate::pretokenize::windowed::PieceSink;
use crate::vocab::{Home, HomeProbe};

/// The counting twin of `encode_hits` in `encode_sink`.
#[inline(never)]
fn count_hits(
    home: HomeProbe<'_>,
    buf: &[u8],
    start: usize,
    offs: &[u16],
    mut i: usize,
    mut count: usize,
) -> (usize, usize, (u64, u64)) {
    while i + 1 < offs.len() {
        // SAFETY: i + 1 < offs.len()
        let (a, b) = unsafe { (*offs.get_unchecked(i), *offs.get_unchecked(i + 1)) };
        let Some((lo, hi)) = piece_key(buf, start + a as usize, start + b as usize) else {
            return (i, count, NO_KEY);
        };
        let found = match home.lookup(lo, hi) {
            Home::Miss => home.lookup_chain(lo, hi),
            h => h,
        };
        match found {
            Home::Token(_) => count += 1,
            Home::Derived(e) => count += e.len(),
            Home::Miss => return (i, count, (lo, hi)),
        }
        i += 1;
    }
    (i, count, NO_KEY)
}

pub(crate) struct CountSink<'e> {
    pub(crate) engine: &'e Engine<'e>,
    pub(crate) home: HomeProbe<'e>,
    pub(crate) count: usize,
}

/// Ids of a piece the home-line probe did not answer.
#[inline(never)]
fn count_miss(engine: &Engine<'_>, piece: &[u8], key: (u64, u64)) -> usize {
    if key != NO_KEY
        && let Some(e) = engine.vocab.entry(key.0, key.1)
    {
        return e.len();
    }
    engine.count_piece(piece, key)
}

impl PieceSink for CountSink<'_> {
    #[inline(always)]
    fn window(&mut self) {}

    // out of line: it runs once per batch, and inlined under the window
    // loop it measured slower
    #[inline(never)]
    fn pieces(&mut self, buf: &[u8], start: usize, offs: &[u16]) {
        let mut i = 0;
        while i + 1 < offs.len() {
            let (at, count, key) = count_hits(self.home, buf, start, offs, i, self.count);
            self.count = count;
            i = at;
            if i + 1 < offs.len() {
                let piece = &buf[start + offs[i] as usize..start + offs[i + 1] as usize];
                self.count += count_miss(self.engine, piece, key);
                i += 1;
            }
        }
    }

    #[inline(always)]
    fn piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe(&self.home, buf, start, end) {
            Ok(_) => self.count += 1,
            Err(key) => self.count += count_miss(self.engine, &buf[start..end], key),
        }
    }

    #[inline(always)]
    fn dense_piece(&mut self, buf: &[u8], start: usize, end: usize) {
        match probe_entry(self.engine.vocab, buf, start, end) {
            Ok(e) => self.count += e.len(),
            Err(NO_KEY) => {
                let piece = &buf[start..end];
                self.count += match self.engine.vocab.long_ids(piece) {
                    Some(ids) => ids.len(),
                    None => self.engine.count_piece(piece, NO_KEY),
                }
            }
            Err(key) => self.count += self.engine.count_piece(&buf[start..end], key),
        }
    }
}
