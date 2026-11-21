//! Blocked Bloom filter over the packed-key hashes of every vocabulary token
//! of up to 15 bytes. Most substring probes in a merge miss; the filter turns
//! away most of those from a cache line it keeps resident, instead of a walk
//! over a large table's random lines. Each key sets two bits in one word, so
//! a query costs one load. No false negatives: every inserted hash passes.

pub(crate) struct KeyFilter {
    words: Box<[u64]>,
    shift: u32,
}

#[inline(always)]
fn bits(h: u64) -> u64 {
    // the low bits of a multiplicative hash are weak; take them from the middle
    (1u64 << ((h >> 20) & 63)) | (1u64 << ((h >> 26) & 63))
}

impl KeyFilter {
    /// Build from the packed-key hashes of all short tokens; about eight bits
    /// per key.
    pub(crate) fn build(hashes: impl ExactSizeIterator<Item = u64>) -> Self {
        let n_words = (hashes.len() / 8).max(64).next_power_of_two();
        let mut words = vec![0u64; n_words].into_boxed_slice();
        let shift = 64 - n_words.trailing_zeros();
        for h in hashes {
            words[(h >> shift) as usize] |= bits(h);
        }
        Self { words, shift }
    }

    #[inline(always)]
    pub(crate) fn may_contain(&self, h: u64) -> bool {
        let b = bits(h);
        self.words[(h >> self.shift) as usize] & b == b
    }
}
