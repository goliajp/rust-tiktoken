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
        let mut ids = Vec::new();
        let mut ends = Vec::new();
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
        if text.len() >= WINDOW_MIN
            && let Some(kind) = self.pre.window_kind()
        {
            let mut sink = EndsSink::new(self, ids, ends, base);
            window_dispatch(self.pre, kind, text, &mut sink);
            sink.finish();
            return;
        }
        self.encode_scalar_ends(text, ids, ends, base);
    }

    fn encode_scalar_ends(
        &self,
        text: &str,
        ids: &mut Vec<u32>,
        ends: &mut Vec<usize>,
        base: usize,
    ) {
        let cap = id_capacity(text);
        ids.reserve(cap);
        ends.reserve(cap);
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
                    ids.push(id);
                    ends.push(base + end);
                    continue;
                }
            }
            let k = ids.len();
            self.encode_piece(&bytes[start..end], key, ids);
            let mut e = base + start;
            for &id in &ids[k..ids.len() - 1] {
                e += self.vocab.token_len(id);
                ends.push(e);
            }
            ends.push(base + end);
        }
    }
}

#[cfg(test)]
#[path = "ends_tests.rs"]
mod tests;
