use std::collections::VecDeque;

use crate::CoreBpe;
use crate::hash::FxHashMap;

// charged per entry on top of its text and ids, for the bookkeeping
const ENTRY_OVERHEAD: usize = 128;
const NONE: usize = usize::MAX;

struct Entry {
    fingerprint: u64,
    text: Box<str>,
    ids: Box<[u32]>,
    // next entry with the same fingerprint
    next: usize,
    used: bool,
}

impl Entry {
    fn cost(&self) -> usize {
        cost(self.text.len(), self.ids.len())
    }
}

fn cost(text: usize, ids: usize) -> usize {
    text + ids * size_of::<u32>() + ENTRY_OVERHEAD
}

/// Remembers whole texts and their token ids, for callers that encode the
/// same text again and again (a system prompt, a document in a chat history).
///
/// A text seen before is compared byte for byte and its stored ids are
/// returned; any other text is encoded and kept. The stored texts and ids stay
/// within the byte budget given to [`CoreBpe::memo`]; texts that were not
/// asked for again are dropped first.
///
/// # Examples
///
/// ```
/// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
/// let mut memo = enc.memo(16 << 20);
/// assert_eq!(memo.encode("hello world"), [15339, 1917]);
/// assert_eq!(memo.count("hello world"), 2);
/// assert_eq!(memo.len(), 1);
/// ```
pub struct TextMemo<'a> {
    bpe: &'a CoreBpe,
    max_bytes: usize,
    bytes: usize,
    heads: FxHashMap<u64, usize>,
    entries: Vec<Option<Entry>>,
    free: Vec<usize>,
    queue: VecDeque<usize>,
    // ids of the last text that did not fit
    oversized: Vec<u32>,
}

impl<'a> TextMemo<'a> {
    pub(crate) fn new(bpe: &'a CoreBpe, max_bytes: usize) -> Self {
        Self {
            bpe,
            max_bytes,
            bytes: 0,
            heads: FxHashMap::default(),
            entries: Vec::new(),
            free: Vec::new(),
            queue: VecDeque::new(),
            oversized: Vec::new(),
        }
    }

    /// The token ids of `text`, the same as [`CoreBpe::encode`] returns.
    pub fn encode(&mut self, text: &str) -> &[u32] {
        let fp = fingerprint(text.as_bytes());
        if let Some(at) = self.find(fp, text) {
            return self.hit(at);
        }
        let ids = self.bpe.encode(text);
        match self.insert(fp, text, ids) {
            Some(at) => &self.entry(at).ids,
            None => &self.oversized,
        }
    }

    /// The number of token ids of `text`. An unseen text is encoded and kept.
    pub fn count(&mut self, text: &str) -> usize {
        self.encode(text).len()
    }

    /// Number of texts held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether no text is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Forget every text.
    pub fn clear(&mut self) {
        *self = Self::new(self.bpe, self.max_bytes);
    }

    fn entry(&self, at: usize) -> &Entry {
        self.entries[at].as_ref().unwrap()
    }

    fn find(&self, fp: u64, text: &str) -> Option<usize> {
        let mut at = *self.heads.get(&fp)?;
        while at != NONE {
            let e = self.entry(at);
            if *e.text == *text {
                return Some(at);
            }
            at = e.next;
        }
        None
    }

    fn hit(&mut self, at: usize) -> &[u32] {
        let e = self.entries[at].as_mut().unwrap();
        e.used = true;
        &e.ids
    }

    fn insert(&mut self, fp: u64, text: &str, ids: Vec<u32>) -> Option<usize> {
        let need = cost(text.len(), ids.len());
        if need > self.max_bytes {
            self.oversized = ids;
            return None;
        }
        while self.bytes + need > self.max_bytes {
            self.evict_one();
        }
        let entry = Entry {
            fingerprint: fp,
            text: text.into(),
            ids: ids.into_boxed_slice(),
            next: self.heads.get(&fp).copied().unwrap_or(NONE),
            used: false,
        };
        let at = self.free.pop().unwrap_or_else(|| {
            self.entries.push(None);
            self.entries.len() - 1
        });
        self.entries[at] = Some(entry);
        self.heads.insert(fp, at);
        self.queue.push_back(at);
        self.bytes += need;
        Some(at)
    }

    /// Drop the oldest text not used since it was last considered.
    fn evict_one(&mut self) {
        while let Some(at) = self.queue.pop_front() {
            let e = self.entries[at].as_mut().unwrap();
            if !e.used {
                self.remove(at);
                return;
            }
            e.used = false;
            self.queue.push_back(at);
        }
    }

    fn remove(&mut self, at: usize) {
        let e = self.entries[at].take().unwrap();
        self.bytes -= e.cost();
        self.free.push(at);
        let head = self.heads[&e.fingerprint];
        if head != at {
            let mut prev = head;
            while self.entry(prev).next != at {
                prev = self.entry(prev).next;
            }
            self.entries[prev].as_mut().unwrap().next = e.next;
        } else if e.next == NONE {
            self.heads.remove(&e.fingerprint);
        } else {
            self.heads.insert(e.fingerprint, e.next);
        }
    }
}

/// Hash of the length and of eight 16-byte samples; texts that differ only
/// between the samples share it.
#[inline]
fn fingerprint(b: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let fold = |a: u64, b: u64| {
        let p = a as u128 * b as u128;
        (p as u64) ^ ((p >> 64) as u64)
    };
    let n = b.len();
    let mut h = n as u64 ^ K;
    let mut mix = |w: &[u8]| {
        let lo = u64::from_le_bytes(w[..8].try_into().unwrap());
        let hi = u64::from_le_bytes(w[8..16].try_into().unwrap());
        h = fold(h ^ lo, hi ^ K);
    };
    if n >= 16 {
        for i in 0..8 {
            let at = (n - 16) / 7 * i;
            let at = if i == 7 { n - 16 } else { at };
            mix(&b[at..at + 16]);
        }
    } else {
        let mut w = [0u8; 16];
        w[..n].copy_from_slice(b);
        mix(&w);
    }
    fold(h, K)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc() -> &'static CoreBpe {
        crate::get_encoding("cl100k_base").unwrap()
    }

    #[test]
    fn same_ids_as_encode_first_and_again() {
        let mut memo = enc().memo(1 << 24);
        let long = "The quick brown 狐 jumps. ".repeat(4000);
        for text in ["", "a", "hello world", "exactly 16 bytes", &long] {
            for _ in 0..3 {
                assert_eq!(memo.encode(text), enc().encode(text));
                assert_eq!(memo.count(text), enc().count(text));
            }
        }
        assert_eq!(memo.len(), 5);
    }

    #[test]
    fn same_fingerprint_different_text() {
        let texts: Vec<String> = (0..4)
            .map(|i| {
                let mut s = "x ".repeat(1000).into_bytes();
                s[40 + i] = b'7';
                String::from_utf8(s).unwrap()
            })
            .collect();
        let fp = fingerprint(texts[0].as_bytes());
        assert!(texts.iter().all(|t| fingerprint(t.as_bytes()) == fp));
        let mut memo = enc().memo(1 << 20);
        for _ in 0..2 {
            for t in &texts {
                assert_eq!(memo.encode(t), enc().encode(t));
            }
        }
        assert_eq!(memo.len(), 4);
        // unlink from the middle, the tail and the head of the chain
        for victim in [1, 0, 3, 2] {
            let at = memo.find(fp, &texts[victim]).unwrap();
            memo.queue.retain(|&q| q != at);
            memo.remove(at);
            assert!(memo.find(fp, &texts[victim]).is_none());
        }
        assert!(memo.heads.is_empty());
        assert_eq!(memo.bytes, 0);
    }

    #[test]
    fn stays_in_budget_and_keeps_what_is_used() {
        let budget = 4000;
        let mut memo = enc().memo(budget);
        memo.encode("kept prompt");
        for i in 0..500 {
            let t = format!("one-off text number {i}");
            assert_eq!(memo.encode(&t), enc().encode(&t));
            memo.encode("kept prompt");
            assert!(memo.bytes <= budget);
            let sum: usize = memo.entries.iter().flatten().map(Entry::cost).sum();
            assert_eq!(sum, memo.bytes);
        }
        let fp = fingerprint(b"kept prompt");
        assert!(memo.find(fp, "kept prompt").is_some());
        assert!(memo.len() < 40);
    }

    #[test]
    fn oversized_text_is_encoded_but_not_kept() {
        let mut memo = enc().memo(200);
        memo.encode("hi");
        let big = "word ".repeat(200);
        assert_eq!(memo.encode(&big), enc().encode(&big));
        assert_eq!(memo.len(), 1);
        memo.clear();
        assert!(memo.is_empty());
        assert_eq!(memo.encode("hi"), enc().encode("hi"));
    }
}
