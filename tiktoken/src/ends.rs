//! Encoding with the byte offset at which each token ends.

use crate::CoreBpe;
use crate::encode_loop::{Engine, NO_KEY, WINDOW_MIN, id_capacity, window_dispatch};
use crate::ends_sink::EndsSink;
use crate::pretokenize::PreTokenizer;
use crate::vocab::{FAST_KEY_MAX, KEY_READ, key_in, key_of};

impl CoreBpe {
    /// [`encode`](Self::encode), with the byte offset in `text` at which
    /// each token ends.
    ///
    /// `ends[i]` is the sum of the lengths of
    /// [`token_bytes`](Self::token_bytes) over `ids[..=i]`: token `i` covers
    /// `ends[i - 1]..ends[i]` of `text` (from `0` for the first), and the
    /// last end is `text.len()`. The ends are found while encoding, which
    /// costs less than summing the lengths afterwards.
    ///
    /// A token can end inside a multi-byte character: a character that no
    /// token holds whole is split over several byte tokens, and the ends
    /// between them are not char boundaries of `text`. Check
    /// `str::is_char_boundary` before slicing `text` at an end.
    ///
    /// Special tokens are treated as ordinary text, as in `encode`. Like
    /// `encode`, a text of 32 KiB or more is encoded on several threads.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("o200k_base").unwrap();
    /// let text = "hello world";
    /// let (ids, ends) = enc.encode_with_ends(text);
    /// assert_eq!(ids, enc.encode(text));
    /// assert_eq!(ends, [5, 11]);
    /// assert_eq!(&text[ends[0]..ends[1]], " world");
    /// ```
    #[must_use]
    pub fn encode_with_ends(&self, text: &str) -> (Vec<u32>, Vec<usize>) {
        // a short text has room for an id per byte, which its loop makes
        let cap = if text.len() < WINDOW_MIN {
            text.len()
        } else {
            id_capacity(text)
        };
        let mut ids = Vec::with_capacity(cap);
        let mut ends = Vec::with_capacity(cap);
        self.encode_with_ends_into(text, &mut ids, &mut ends);
        (ids, ends)
    }

    /// Append the token ids of `text` to `ids` and the offset in `text` at
    /// which each token ends to `ends`, as
    /// [`encode_with_ends`](Self::encode_with_ends) returns them.
    ///
    /// Both vectors grow by the same number of entries. The ends are offsets
    /// in this `text`, whatever the vectors held before; reusing them across
    /// calls skips the allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// let (mut ids, mut ends) = (Vec::new(), Vec::new());
    /// enc.encode_with_ends_into("你好", &mut ids, &mut ends);
    /// // one token per character here
    /// assert_eq!(ends, [3, 6]);
    /// ids.clear();
    /// ends.clear();
    /// enc.encode_with_ends_into("hi there", &mut ids, &mut ends);
    /// assert_eq!(ends, [2, 8]);
    /// ```
    pub fn encode_with_ends_into(&self, text: &str, ids: &mut Vec<u32>, ends: &mut Vec<usize>) {
        let engine = self.engine();
        #[cfg(not(target_arch = "wasm32"))]
        if text.len() >= crate::bpe::PARALLEL_MIN && parallel(&engine, text, ids, ends) {
            return;
        }
        engine.encode_ends(text, ids, ends, 0);
    }
}

/// Encode `text` as chunks on the pool; false when it stays on this thread.
#[cfg(not(target_arch = "wasm32"))]
#[inline(never)]
fn parallel(engine: &Engine<'_>, text: &str, ids: &mut Vec<u32>, ends: &mut Vec<usize>) -> bool {
    let Some((threads, cuts)) = crate::split::plan(engine, text, crate::split::CHUNK, None) else {
        return false;
    };
    let parts = crate::pool::run(cuts.len() - 1, threads, |i| {
        let (mut ids, mut ends) = (Vec::new(), Vec::new());
        engine.encode_ends(&text[cuts[i]..cuts[i + 1]], &mut ids, &mut ends, cuts[i]);
        (ids, ends)
    });
    let Some(parts) = parts else {
        return false;
    };
    let total: usize = parts.iter().map(|(i, _)| i.len()).sum();
    ids.reserve(total);
    ends.reserve(total);
    for (i, e) in &parts {
        ids.extend_from_slice(i);
        ends.extend_from_slice(e);
    }
    true
}

impl Engine<'_> {
    /// Append the ids of `text` and their ends, plus `base`.
    pub(crate) fn encode_ends(
        &self,
        text: &str,
        ids: &mut Vec<u32>,
        ends: &mut Vec<usize>,
        base: usize,
    ) {
        if text.len() < WINDOW_MIN {
            return self.short_ends(text, ids, ends, base);
        }
        let mut sink = EndsSink::new(self, ids, ends, base, id_capacity(text));
        match self.pre.window_kind() {
            Some(kind) => window_dispatch(self.pre, kind, text, &mut sink),
            None => self.scalar_ends(text, &mut sink),
        }
        sink.finish();
    }

    /// A text shorter than [`WINDOW_MIN`]: it has at most one id per byte,
    /// so room for that many is made once and nothing is checked after.
    fn short_ends(&self, text: &str, ids: &mut Vec<u32>, ends: &mut Vec<usize>, base: usize) {
        let bytes = text.as_bytes();
        ids.reserve(bytes.len());
        ends.reserve(bytes.len());
        let (mut il, mut el) = (ids.len(), ends.len());
        let (mut ip, mut ep) = (ids.as_mut_ptr(), ends.as_mut_ptr());
        let mut pos = 0;
        while let Some((start, end)) = self.pre.next_match(text, pos) {
            pos = end;
            let mut key = NO_KEY;
            if (end - start).wrapping_sub(1) < FAST_KEY_MAX {
                key = key_of(&bytes[start..end]);
                if let Some(id) = self.vocab.get_home(key.0, key.1) {
                    // SAFETY: room for an id and an end per byte was made
                    // above, and every id takes at least one
                    unsafe {
                        ip.add(il).write(id);
                        ep.add(el).write(base + end);
                    }
                    (il, el) = (il + 1, el + 1);
                    continue;
                }
            }
            // SAFETY: `il` ids and `el` ends were written
            unsafe {
                ids.set_len(il);
                ends.set_len(el);
            }
            self.short_miss(&bytes[start..end], key, ids, ends, base + start);
            (il, el) = (ids.len(), ends.len());
            (ip, ep) = (ids.as_mut_ptr(), ends.as_mut_ptr());
        }
        // SAFETY: as above
        unsafe {
            ids.set_len(il);
            ends.set_len(el);
        }
    }

    #[inline(never)]
    fn short_miss(
        &self,
        piece: &[u8],
        key: (u64, u64),
        ids: &mut Vec<u32>,
        ends: &mut Vec<usize>,
        at: usize,
    ) {
        let k = ids.len();
        self.encode_piece(piece, key, ids);
        let mut e = at;
        for &id in &ids[k..] {
            e += self.vocab.token_len(id);
            ends.push(e);
        }
    }

    fn scalar_ends(&self, text: &str, sink: &mut EndsSink<'_, '_>) {
        let bytes = text.as_bytes();
        let wide = bytes.len() >= KEY_READ;
        let mut pos = 0;
        while let Some((start, end)) = self.pre.next_match(text, pos) {
            debug_assert_eq!(start, pos);
            pos = end;
            let mut key = NO_KEY;
            if (end - start).wrapping_sub(1) < FAST_KEY_MAX {
                key = if wide {
                    key_in(bytes, start, end)
                } else {
                    key_of(&bytes[start..end])
                };
                if let Some(id) = self.vocab.get_home(key.0, key.1) {
                    sink.token(id, end - start);
                    continue;
                }
            }
            sink.miss(&bytes[start..end], key);
        }
    }
}

#[cfg(test)]
#[path = "ends_tests.rs"]
mod tests;
