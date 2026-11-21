//! Search for special-token literals in text.
//!
//! Candidate positions come from the set of first bytes of all literals
//! (eight bytes at a time when the set is small, which is the usual case:
//! `<` and `[`), then the literals starting with that byte are compared
//! longest first. No literal of any shipped encoding is a prefix of another
//! (checked by a test), so "longest first" and "first in any order" agree.

use std::collections::HashMap;
use std::hash::BuildHasher;

/// At most this many distinct first bytes use the word-at-a-time search.
const SWAR_MAX: usize = 4;
const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

pub(crate) struct SpecialMatcher {
    /// literals with their ids, grouped by first byte, longest first within
    /// a group
    tokens: Vec<(Vec<u8>, u32)>,
    /// `tokens[group[b].0..group[b].1]` start with byte `b`
    group: [(u32, u32); 256],
    /// distinct first bytes, when there are at most [`SWAR_MAX`]
    firsts: Vec<u8>,
}

impl SpecialMatcher {
    /// `None` when there are no literals. Literals must be non-empty.
    pub(crate) fn new<S: BuildHasher>(literals: &HashMap<Vec<u8>, u32, S>) -> Option<Self> {
        if literals.is_empty() {
            return None;
        }
        let mut tokens: Vec<(Vec<u8>, u32)> =
            literals.iter().map(|(k, &v)| (k.clone(), v)).collect();
        assert!(tokens.iter().all(|(k, _)| !k.is_empty()));
        tokens.sort_by(|a, b| a.0[0].cmp(&b.0[0]).then(b.0.len().cmp(&a.0.len())));
        let mut group = [(0u32, 0u32); 256];
        for (i, (k, _)) in tokens.iter().enumerate() {
            let g = &mut group[k[0] as usize];
            if g.0 == g.1 {
                g.0 = i as u32;
            }
            g.1 = i as u32 + 1;
        }
        let mut firsts: Vec<u8> = (0..=255u8).filter(|&b| group[b as usize].1 > 0).collect();
        if firsts.len() > SWAR_MAX {
            firsts.clear();
        }
        Some(Self {
            tokens,
            group,
            firsts,
        })
    }

    /// The first special token at or after `from`, as `(start, end, id)`.
    pub(crate) fn find(&self, text: &[u8], mut from: usize) -> Option<(usize, usize, u32)> {
        while let Some(i) = self.next_candidate(text, from) {
            let (a, b) = self.group[text[i] as usize];
            for (lit, id) in &self.tokens[a as usize..b as usize] {
                if text[i..].starts_with(lit) {
                    return Some((i, i + lit.len(), *id));
                }
            }
            from = i + 1;
        }
        None
    }

    /// The first position at or after `from` holding a first byte.
    #[inline]
    fn next_candidate(&self, text: &[u8], mut i: usize) -> Option<usize> {
        if !self.firsts.is_empty() {
            while i + 8 <= text.len() {
                let w = u64::from_le_bytes(text[i..i + 8].try_into().unwrap());
                let mut hit = 0;
                for &f in &self.firsts {
                    let x = w ^ (LO * f as u64);
                    hit |= x.wrapping_sub(LO) & !x & HI;
                }
                if hit != 0 {
                    // the lowest flagged byte is always a true match
                    return Some(i + (hit.trailing_zeros() / 8) as usize);
                }
                i += 8;
            }
        }
        (i..text.len()).find(|&j| self.group[text[j] as usize].1 > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(lits: &[&str]) -> SpecialMatcher {
        let map: HashMap<Vec<u8>, u32> = lits
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().to_vec(), i as u32))
            .collect();
        SpecialMatcher::new(&map).unwrap()
    }

    fn all(m: &SpecialMatcher, text: &str) -> Vec<(usize, usize, u32)> {
        let mut out = vec![];
        let mut at = 0;
        while let Some(hit) = m.find(text.as_bytes(), at) {
            at = hit.1;
            out.push(hit);
        }
        out
    }

    #[test]
    fn finds_every_literal_in_order() {
        let m = matcher(&["<|end|>", "[INST]", "<s>"]);
        assert_eq!(
            all(&m, "a<|end|>bb[INST]<s><s<|end|"),
            vec![(1, 8, 0), (10, 16, 1), (16, 19, 2)]
        );
        assert_eq!(all(&m, ""), vec![]);
        assert_eq!(all(&m, "<<<<<<<<<<<<<<<<<s>"), vec![(16, 19, 2)]);
    }

    #[test]
    fn many_first_bytes_use_the_table_scan() {
        let m = matcher(&["a1", "b2", "c3", "d4", "e5"]);
        assert!(m.firsts.is_empty());
        assert_eq!(all(&m, "xxe5xxxxxxxxa1"), vec![(2, 4, 4), (12, 14, 0)]);
    }

    #[test]
    fn matches_at_every_offset_of_a_word() {
        let m = matcher(&["<s>"]);
        for pad in 0..20 {
            let text = format!("{}<s>{}", "x".repeat(pad), "y".repeat(pad));
            assert_eq!(all(&m, &text), vec![(pad, pad + 3, 0)], "{text}");
        }
    }

    /// Against the regex alternation the matcher replaced, on text that
    /// glues every literal of every encoding to fragments of the others.
    #[test]
    fn matches_the_regex_alternation() {
        for &name in crate::list_encodings() {
            let specials = crate::get_encoding(name).unwrap().special_tokens();
            let m = SpecialMatcher::new(specials).unwrap();
            let pat: Vec<String> = specials
                .keys()
                .map(|k| regex::escape(std::str::from_utf8(k).unwrap()))
                .collect();
            let re = regex::Regex::new(&pat.join("|")).unwrap();
            let mut text = String::new();
            for (i, k) in specials.keys().enumerate().take(400) {
                let k = std::str::from_utf8(k).unwrap();
                let cut = k.char_indices().nth(k.chars().count() / 2).unwrap().0;
                text.push_str(&k[..cut]);
                text.push_str(["x", "<", "[", "你", " "][i % 5]);
                text.push_str(k);
                text.push_str(&k[cut..]);
            }
            let want: Vec<(usize, usize, u32)> = re
                .find_iter(&text)
                .map(|x| (x.start(), x.end(), specials[x.as_str().as_bytes()]))
                .collect();
            assert!(!want.is_empty());
            assert_eq!(all(&m, &text), want, "{name}");
        }
    }

    /// The longest-first rule equals the regex alternation it replaced only
    /// while no literal is a prefix of another.
    #[test]
    fn no_special_token_is_a_prefix_of_another() {
        for &name in crate::list_encodings() {
            let enc = crate::get_encoding(name).unwrap();
            let keys: Vec<&Vec<u8>> = enc.special_tokens().keys().collect();
            for a in &keys {
                for b in &keys {
                    assert!(
                        a == b || !b.starts_with(a),
                        "{name}: {:?} is a prefix of {:?}",
                        String::from_utf8_lossy(a),
                        String::from_utf8_lossy(b)
                    );
                }
            }
        }
    }
}
