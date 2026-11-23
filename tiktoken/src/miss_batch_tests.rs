use crate::merge::{self, lanes};
use crate::pretokenize::PreTokenizer;
use crate::vocab::Vocab;

// every shipped vocabulary meets the preconditions, and the table stays
// within two slots per token
#[test]
fn every_vocabulary_merges_by_id() {
    for &name in crate::list_encodings() {
        let vocab = crate::get_encoding(name).unwrap().vocab();
        assert!(vocab.merges_by_id(), "{name}");
        let splits = vocab.splits().expect(name);
        assert!(
            splits.bytes() <= 4 * vocab.decoder_len() * size_of::<u64>(),
            "{name}"
        );
    }
}

fn lanes_of(pieces: &[&[u8]], vocab: &Vocab) -> Vec<Vec<u32>> {
    let splits = vocab.splits().expect("every vocabulary merges by id");
    let mut out = vec![[0u32; 16]; pieces.len()];
    let mut counts = vec![0u8; pieces.len()];
    lanes::merge_by_id(pieces, vocab, splits, &mut out, &mut counts);
    out.iter()
        .zip(&counts)
        .map(|(o, &c)| o[..c as usize].to_vec())
        .collect()
}

// tiktoken's rule spelled out: join the adjacent pair whose concatenation
// is the lowest-ranked token, leftmost on ties, until none is a token
fn by_substrings(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    let mut parts: Vec<&[u8]> = piece.chunks(1).collect();
    loop {
        let best = (0..parts.len().saturating_sub(1))
            .filter_map(|i| {
                vocab
                    .get(&[parts[i], parts[i + 1]].concat())
                    .map(|r| (r, i))
            })
            .min();
        let Some((_, i)) = best else { break };
        let len = parts[i].len() + parts[i + 1].len();
        let start = parts[i].as_ptr() as usize - piece.as_ptr() as usize;
        parts[i] = &piece[start..start + len];
        parts.remove(i + 1);
    }
    parts.iter().map(|p| vocab.get(p).unwrap()).collect()
}

fn check(pieces: &[&[u8]], vocab: &Vocab, name: &str) {
    let got = lanes_of(pieces, vocab);
    for (p, g) in pieces.iter().zip(got) {
        assert_eq!(
            g,
            by_substrings(p, vocab),
            "{name} {:?}",
            String::from_utf8_lossy(p)
        );
    }
}

// every token of 2..=15 bytes merged from its bytes, alone and next to its
// neighbours in rank order: tokens that split into two tokens in several
// ways, and tokens that do not merge from their own bytes at all
#[test]
fn lanes_match_the_merge_on_every_token() {
    for &name in crate::list_encodings() {
        let vocab = crate::get_encoding(name).unwrap().vocab();
        let tokens: Vec<&[u8]> = (0..vocab.decoder_len() as u32)
            .filter_map(|r| vocab.try_decode(r))
            .filter(|t| (2..=lanes::LANE_MAX).contains(&t.len()))
            .collect();
        check(&tokens, vocab, name);
        let pairs: Vec<Vec<u8>> = tokens
            .windows(2)
            .map(|w| [w[0], w[1]].concat())
            .filter(|p| p.len() <= lanes::LANE_MAX)
            .collect();
        let pairs: Vec<&[u8]> = pairs.iter().map(Vec::as_slice).collect();
        check(&pairs, vocab, name);
    }
}

#[test]
fn lanes_match_the_merge_on_random_pieces() {
    let alphabets: [&[u8]; 4] = [
        b"etaoinshrdlu ",
        b"aaaaab",
        b"{}();:=<>!&|*-_ \n\t0123456789",
        "\u{e9}\u{fc}\u{df}\u{2019}x".as_bytes(),
    ];
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for &name in crate::list_encodings() {
        let vocab = crate::get_encoding(name).unwrap().vocab();
        let mut owned = vec![];
        for case in 0..3000 {
            let alpha = alphabets[case % alphabets.len()];
            let len = 2 + (next() % (lanes::LANE_MAX as u64 - 1)) as usize;
            // any bytes: the lanes do not care about char boundaries
            let p: Vec<u8> = (0..len)
                .map(|_| alpha[(next() % alpha.len() as u64) as usize])
                .collect();
            owned.push(p);
        }
        let pieces: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        // batches of every size, so that lanes refill and idle
        let mut at = 0;
        let mut size = 1;
        while at < pieces.len() {
            let end = (at + size).min(pieces.len());
            check(&pieces[at..end], vocab, name);
            at = end;
            size = size % 40 + 1;
        }
    }
}

fn reference(bpe: &crate::CoreBpe, text: &str) -> Vec<u32> {
    let mut want = vec![];
    let mut pos = 0;
    while let Some((s, e)) = bpe.pre_tokenizer().next_match(text, pos) {
        merge::bpe_encode(&text.as_bytes()[s..e], bpe.vocab(), &mut want);
        pos = e;
    }
    want
}

// fresh words between common ones and long pieces, repeated within a batch
// and across batches, for every encoding: encode and count both match the
// one-piece-at-a-time merge, first pass and second (the shared cache is
// not reset: other tests use the same instances)
#[test]
fn texts_with_many_misses_match_on_every_encoding() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for &name in crate::list_encodings() {
        let bpe = crate::get_encoding(name).unwrap();
        let mut text = String::new();
        let mut words: Vec<String> = vec![];
        for i in 0..2500 {
            let r = next();
            match r % 9 {
                0..=3 => {
                    let w: String = (0..2 + (r >> 8) % 13)
                        .map(|k| b"qzxjvkwyeat"[((r >> (12 + 3 * k)) % 11) as usize] as char)
                        .collect();
                    text.push(' ');
                    text.push_str(&w);
                    words.push(w);
                }
                4 if !words.is_empty() => {
                    let w = &words[words.len() - 1 - (r >> 8) as usize % words.len().min(40)];
                    text.push(' ');
                    text.push_str(w);
                }
                5 => text.push_str(" the"),
                6 => text.push_str(&format!(" {}", r % 100_000)),
                7 if i % 13 == 0 => text.push_str(" qzxjvkwyqzxjvkwyqzxjvkwy"),
                7 => text.push_str(" na\u{ef}ve caf\u{e9}"),
                _ => text.push_str(", \u{bf}\n"),
            }
        }
        let want = reference(bpe, &text);
        assert_eq!(bpe.count(&text), want.len(), "{name}");
        assert_eq!(bpe.encode(&text), want, "{name}");
        assert_eq!(bpe.encode(&text), want, "{name}");
        assert_eq!(bpe.count(&text), want.len(), "{name}");
    }
}
