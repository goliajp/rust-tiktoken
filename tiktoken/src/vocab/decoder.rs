//! Token bytes by rank.

use super::Vocab;

impl Vocab {
    /// Try to get the token bytes for a given rank.
    /// Returns `None` if the rank is out of range or not present in the vocabulary.
    #[inline]
    pub(crate) fn try_decode(&self, rank: u32) -> Option<&[u8]> {
        let idx = rank as usize;
        if idx >= self.decoder.len() {
            return None;
        }
        let (offset, len) = self.decoder[idx];
        // u32::MAX offset = sentinel for "rank not present"
        if offset == u32::MAX {
            return None;
        }
        Some(&self.arena[offset as usize..(offset as usize + len as usize)])
    }

    /// The byte length of a token the encoder produced.
    #[inline(always)]
    pub(crate) fn token_len(&self, rank: u32) -> usize {
        debug_assert!(self.try_decode(rank).is_some());
        self.decoder[rank as usize].1 as usize
    }
}
