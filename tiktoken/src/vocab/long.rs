//! Lookup for tokens longer than the packed key: open addressing over arena
//! references, with an 8-bit hash tag so most mismatched slots are rejected
//! without loading the arena bytes.

use super::key::hash_long;

#[derive(Clone, Copy)]
#[repr(C)]
struct LongSlot {
    // arena offset in the low 32 bits, hash tag in bits 32..40
    key: u64,
    rank: u32,
    // 0 = empty
    len: u32,
}

const EMPTY: LongSlot = LongSlot {
    key: 0,
    rank: 0,
    len: 0,
};

pub(crate) struct LongTable {
    slots: Box<[LongSlot]>,
    mask: usize,
}

#[inline]
fn tag(hash: u64) -> u64 {
    (hash >> 56) & 0xFF
}

impl LongTable {
    /// Build from `(rank, arena offset, len)` of every token longer than the
    /// packed key.
    pub(crate) fn build(arena: &[u8], items: &[(u32, u32, u32)]) -> Self {
        let size = (items.len() * 2).max(2).next_power_of_two();
        let mask = size - 1;
        let mut slots = vec![EMPTY; size].into_boxed_slice();
        for &(rank, offset, len) in items {
            let hash = hash_long(&arena[offset as usize..(offset + len) as usize]);
            let mut idx = hash as usize & mask;
            while slots[idx].len != 0 {
                idx = (idx + 1) & mask;
            }
            slots[idx] = LongSlot {
                key: offset as u64 | (tag(hash) << 32),
                rank,
                len,
            };
        }
        Self { slots, mask }
    }

    pub(crate) fn get(&self, arena: &[u8], token: &[u8]) -> Option<u32> {
        self.get_hashed(arena, token, hash_long(token))
    }

    /// [`get`](Self::get) with the token's [`hash_long`] already computed.
    pub(crate) fn get_hashed(&self, arena: &[u8], token: &[u8], hash: u64) -> Option<u32> {
        let tag = tag(hash);
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = self.slots[idx];
            if slot.len == 0 {
                return None;
            }
            if slot.len as usize == token.len() && (slot.key >> 32) & 0xFF == tag {
                let offset = (slot.key & 0xFFFF_FFFF) as usize;
                if &arena[offset..offset + token.len()] == token {
                    return Some(slot.rank);
                }
            }
            idx = (idx + 1) & self.mask;
        }
    }
}
