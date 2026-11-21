//! Merges that start from atoms against merges from bytes, for every
//! encoding.

use super::{SMALL_MAX, merge_small, merge_small_bytes};
use crate::pretokenize::PreTokenizer;
use crate::vocab::{Vocab, char_len};

fn check(v: &Vocab, piece: &[u8], name: &str) {
    if piece.is_empty() || piece.len() > SMALL_MAX {
        return;
    }
    let (mut a, mut b) = ([0u32; SMALL_MAX], [0u32; SMALL_MAX]);
    let na = merge_small(piece, v, &mut a);
    let nb = merge_small_bytes(piece, v, &mut b);
    assert_eq!(
        a[..na],
        b[..nb],
        "{name}: {:?} ({piece:x?})",
        String::from_utf8_lossy(piece)
    );
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn tokens(v: &Vocab) -> impl Iterator<Item = &[u8]> {
    (0..v.len() as u32 + 4096).filter_map(|r| v.try_decode(r))
}

/// Chars of one vocabulary's single-char tokens, plus chars from the
/// scripts the atoms come from.
fn alphabet(v: &Vocab, rng: &mut Rng) -> Vec<String> {
    let mut out: Vec<String> = tokens(v)
        .filter_map(|t| std::str::from_utf8(t).ok())
        .filter(|s| s.chars().count() == 1 && s.len() > 1)
        .map(str::to_string)
        .collect();
    let blocks = [
        (0x4E00, 0x9FA5),
        (0x3040, 0x30FF),
        (0xAC00, 0xD7A3),
        (0x0600, 0x06FF),
        (0x0400, 0x04FF),
        (0xFF00, 0xFF65),
        (0x1F300, 0x1F64F),
        (0x80, 0x24F),
    ];
    for _ in 0..4000 {
        let (lo, hi) = blocks[rng.below(blocks.len())];
        if let Some(c) = char::from_u32((lo + rng.below(hi - lo + 1)) as u32) {
            out.push(c.to_string());
        }
    }
    for s in [" ", "a", "e", "\n", ",", "1", "'"] {
        out.push(s.to_string());
    }
    out
}

/// Valid utf-8 around every fragment token: the bytes it shares with a
/// char completed with random bytes, between random chars.
fn fragment_texts(v: &Vocab, rng: &mut Rng, alpha: &[String]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for t in tokens(v) {
        let lead = t.iter().take_while(|&&b| b & 0xC0 == 0x80).count();
        let tail_cut = t
            .iter()
            .rposition(|&b| b & 0xC0 != 0x80)
            .filter(|&p| t.len() - p < char_len(t[p]));
        if lead == 0 && tail_cut.is_none() {
            continue;
        }
        for _ in 0..4 {
            let mut s = Vec::new();
            for _ in 0..rng.below(3) {
                s.extend_from_slice(alpha[rng.below(alpha.len())].as_bytes());
            }
            if lead > 0 {
                // a char whose last `lead` bytes are the token's first
                let l = lead + 1 + rng.below(4 - lead.min(3));
                if l > 4 || lead >= l {
                    continue;
                }
                // lead bytes that take any continuation byte
                let (first, count) = [(0, 0), (0, 0), (0xC2, 30), (0xE1, 12), (0xF1, 3)][l];
                s.push(first + rng.below(count) as u8);
                for _ in 0..l - 1 - lead {
                    s.push(0x80 | rng.below(64) as u8);
                }
            }
            s.extend_from_slice(t);
            if let Some(p) = tail_cut {
                for _ in 0..char_len(t[p]) - (t.len() - p) {
                    s.push(0x80 | rng.below(64) as u8);
                }
            }
            for _ in 0..rng.below(3) {
                s.extend_from_slice(alpha[rng.below(alpha.len())].as_bytes());
            }
            if std::str::from_utf8(&s).is_ok() {
                out.push(s);
            }
        }
    }
    out
}

#[test]
fn atoms_match_byte_merges_on_random_and_fragment_texts() {
    for &name in crate::list_encodings() {
        let v = crate::get_encoding(name).unwrap().vocab();
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let alpha = alphabet(v, &mut rng);
        let frags = fragment_texts(v, &mut rng, &alpha);
        assert!(frags.len() > 50, "{name}: {} fragment texts", frags.len());
        for s in &frags {
            check(v, s, name);
        }
        for _ in 0..20_000 {
            let mut s = Vec::new();
            let len = 2 + rng.below(40);
            while s.len() < len {
                // runs of chars from one stretch of the alphabet, like words
                s.extend_from_slice(alpha[rng.below(alpha.len())].as_bytes());
            }
            check(v, &s, name);
        }
    }
}

#[test]
fn atoms_match_byte_merges_on_fixture_pieces() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    for &name in crate::list_encodings() {
        let bpe = crate::get_encoding(name).unwrap();
        let v = bpe.vocab();
        let mut pieces = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let cases: Vec<serde_json::Value> =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            for case in cases {
                let text = case["text"].as_str().unwrap();
                if text.is_ascii() {
                    continue;
                }
                let mut pos = 0;
                while let Some((s, e)) = bpe.pre_tokenizer().next_match(text, pos) {
                    check(v, &text.as_bytes()[s..e], name);
                    pieces += 1;
                    pos = e;
                }
            }
        }
        assert!(pieces > 10_000, "{name}: {pieces} pieces");
    }
}
