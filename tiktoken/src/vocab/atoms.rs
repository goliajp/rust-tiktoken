//! Multibyte chars that the merge may start from as whole parts.
//!
//! A char with bytes `B` (2 to 4 of them) is an atom when
//! 1. merging `B` alone yields the single token `B`, through merges of rank
//!    at most `R`, and
//! 2. every token that has `B` as a proper prefix or suffix ranks above `R`,
//!
//! and starting from it whole is exact at an occurrence when also
//! 3. no token occurs there that covers part of the char and bytes outside
//!    it. Such tokens begin with continuation bytes or end in an incomplete
//!    char ("fragments"); there are a few hundred per vocabulary, and each
//!    atom keeps those that could cover part of it.
//!
//! Why the result is the same: by 3, no pair across the char's edges is a
//! token while it is incomplete, so its inner merges and the merges outside
//! it run as independent sequences, and the inner ones are those of merging
//! `B` alone. While the char is incomplete an inner pair of rank at most `R`
//! is available, so no merge above `R` happens anywhere; by 2 every merge
//! with the whole char as an operand ranks above `R`. Starting with the char
//! whole therefore changes neither the outside sequence nor its order (ties
//! between inner and outside pairs only reorder independent merges). The
//! argument applies to one char at a time, so any set of atoms may start
//! whole together.

use std::collections::HashMap;

#[derive(Clone, Copy, Default)]
struct Slot {
    // the char's bytes, little-endian; 0 = empty
    key: u32,
    rank: u32,
    // index into `rules`; 0 = no fragment can cover part of it
    rule: u32,
}

/// Fragments that could cover part of one atom, by the bytes they need
/// outside it: `before` are what a fragment ending inside the atom needs
/// right before it, `after` what one starting inside it needs right after.
/// The byte sets hold their last and first bytes, so most occurrences are
/// cleared by one bit test.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
struct Rule {
    before: Vec<Box<[u8]>>,
    after: Vec<Box<[u8]>>,
    before_last: [u64; 4],
    after_first: [u64; 4],
}

#[inline(always)]
fn has(set: &[u64; 4], b: u8) -> bool {
    set[b as usize / 64] >> (b % 64) & 1 != 0
}

pub(crate) struct Atoms {
    // one bit per hashed key, set for every atom: most chars that are not
    // atoms stop at this word, which stays in cache
    filter: Box<[u64; FILTER_WORDS]>,
    slots: Box<[Slot]>,
    rules: Vec<Rule>,
}

const FILTER_WORDS: usize = 1024;

#[inline(always)]
fn filter_bit(key: u32) -> (usize, u64) {
    let h = (key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
    ((h as usize / 64) % FILTER_WORDS, 1 << (h % 64))
}

#[inline(always)]
fn slot_of(key: u32, mask: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B1) >> 7) as usize & mask
}

/// Byte length of the char led by `b`, 1 for anything that does not lead
/// a multibyte char.
#[inline(always)]
pub(crate) fn char_len(b: u8) -> usize {
    match b {
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

fn single_char(t: &[u8]) -> bool {
    t.len() >= 2 && char_len(t[0]) == t.len() && std::str::from_utf8(t).is_ok()
}

fn key_of(b: &[u8]) -> u32 {
    let mut w = [0u8; 4];
    w[..b.len()].copy_from_slice(b);
    u32::from_le_bytes(w)
}

/// The parts of merging `b` alone and the highest rank merged.
fn merge_alone(b: &[u8], get: &impl Fn(&[u8]) -> Option<u32>) -> (usize, u32) {
    let mut parts: Vec<Vec<u8>> = b.iter().map(|&x| vec![x]).collect();
    let mut high = 0;
    loop {
        let best = (0..parts.len().saturating_sub(1))
            .filter_map(|i| get(&[&parts[i][..], &parts[i + 1][..]].concat()).map(|r| (r, i)))
            .min();
        let Some((r, i)) = best else { break };
        high = high.max(r);
        let next = parts.remove(i + 1);
        parts[i].extend(next);
    }
    (parts.len(), high)
}

impl Atoms {
    /// Build from every token as `(rank, bytes)`; `get` looks up a rank.
    pub(crate) fn build<'a>(
        tokens: impl Iterator<Item = (u32, &'a [u8])> + Clone,
        get: impl Fn(&[u8]) -> Option<u32>,
    ) -> Self {
        // lowest rank of a token with the char as a proper prefix or suffix
        let mut edge: HashMap<u32, u32> = HashMap::new();
        let mut heads: HashMap<Vec<u8>, Vec<Box<[u8]>>> = HashMap::new();
        let mut tails: HashMap<Vec<u8>, Vec<Box<[u8]>>> = HashMap::new();
        let mut singles = Vec::new();
        for (rank, t) in tokens {
            if single_char(t) {
                singles.push((rank, t));
                continue;
            }
            for l in 2..=4.min(t.len() - 1) {
                for c in [&t[..l], &t[t.len() - l..]] {
                    if single_char(c) {
                        let e = edge.entry(key_of(c)).or_insert(u32::MAX);
                        *e = (*e).min(rank);
                    }
                }
            }
            let lead = t.iter().take_while(|&&b| b & 0xC0 == 0x80).count();
            if (1..=3).contains(&lead) && lead < t.len() {
                tails.entry(t[..lead].to_vec()).or_default().push(t.into());
            }
            if let Some(p) = t.iter().rposition(|&b| b & 0xC0 != 0x80)
                && p > 0
                && t.len() - p < char_len(t[p])
            {
                heads.entry(t[p..].to_vec()).or_default().push(t.into());
            }
        }
        let mut rules = vec![Rule::default()];
        let mut rule_ix: HashMap<Rule, u32> = HashMap::new();
        let mut atoms = Vec::new();
        for (rank, b) in singles {
            let (parts, high) = merge_alone(b, &get);
            let key = key_of(b);
            if parts != 1 || edge.get(&key).is_some_and(|&e| e <= high) {
                continue;
            }
            let mut rule = Rule::default();
            for j in 1..b.len() {
                for h in heads.get(&b[..j]).into_iter().flatten() {
                    let x: Box<[u8]> = h[..h.len() - j].into();
                    let last = x[x.len() - 1];
                    rule.before_last[last as usize / 64] |= 1 << (last % 64);
                    rule.before.push(x);
                }
                for t in tails.get(&b[j..]).into_iter().flatten() {
                    let y: Box<[u8]> = t[b.len() - j..].into();
                    rule.after_first[y[0] as usize / 64] |= 1 << (y[0] % 64);
                    rule.after.push(y);
                }
            }
            for v in [&mut rule.before, &mut rule.after] {
                v.sort();
                v.dedup();
            }
            let ix = if rule == Rule::default() {
                0
            } else {
                *rule_ix.entry(rule.clone()).or_insert_with(|| {
                    rules.push(rule);
                    (rules.len() - 1) as u32
                })
            };
            atoms.push(Slot {
                key,
                rank,
                rule: ix,
            });
        }
        let size = (atoms.len() * 2).max(16).next_power_of_two();
        let mut slots = vec![Slot::default(); size].into_boxed_slice();
        let mut filter = Box::new([0u64; FILTER_WORDS]);
        for a in atoms {
            let (w, bit) = filter_bit(a.key);
            filter[w] |= bit;
            let mut i = slot_of(a.key, size - 1);
            while slots[i].key != 0 {
                i = (i + 1) & (size - 1);
            }
            slots[i] = a;
        }
        Self {
            filter,
            slots,
            rules,
        }
    }

    /// The rank of the char at `piece[a..a + len]` (`len >= 2`) when the
    /// merge of `piece` may start from it whole.
    #[inline]
    pub(crate) fn at(&self, piece: &[u8], a: usize, len: usize) -> Option<u32> {
        let key = match piece.get(a..a + 4) {
            Some(w) => u32::from_le_bytes(w.try_into().unwrap()) & (u32::MAX >> (32 - 8 * len)),
            None => key_of(&piece[a..a + len]),
        };
        let (w, bit) = filter_bit(key);
        if self.filter[w] & bit == 0 {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut i = slot_of(key, mask);
        let s = loop {
            let s = self.slots[i];
            if s.key == key {
                break s;
            }
            if s.key == 0 {
                return None;
            }
            i = (i + 1) & mask;
        };
        if s.rule != 0 {
            let r = &self.rules[s.rule as usize];
            if a > 0
                && has(&r.before_last, piece[a - 1])
                && r.before.iter().any(|x| piece[..a].ends_with(x))
            {
                return None;
            }
            let e = a + len;
            if e < piece.len()
                && has(&r.after_first, piece[e])
                && r.after.iter().any(|y| piece[e..].starts_with(y))
            {
                return None;
            }
        }
        Some(s.rank)
    }
}
