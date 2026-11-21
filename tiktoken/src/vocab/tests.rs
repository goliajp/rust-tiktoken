use super::*;

fn sample_entries() -> Vec<(Vec<u8>, u32)> {
    vec![
        (b"a".to_vec(), 0),
        (b"b".to_vec(), 1),
        (b"ab".to_vec(), 2),
        (b"abc".to_vec(), 3),
    ]
}

#[test]
fn test_build_and_lookup() {
    let vocab = Vocab::from_entries(sample_entries());
    assert_eq!(vocab.get(b"a"), Some(0));
    assert_eq!(vocab.get(b"b"), Some(1));
    assert_eq!(vocab.get(b"ab"), Some(2));
    assert_eq!(vocab.get(b"abc"), Some(3));
}

#[test]
fn test_missing_key() {
    let vocab = Vocab::from_entries(sample_entries());
    assert_eq!(vocab.get(b"xyz"), None);
    assert_eq!(vocab.get(b"abcd"), None);
    assert_eq!(vocab.get(b""), None);
}

#[test]
fn test_contains_key() {
    let vocab = Vocab::from_entries(sample_entries());
    assert!(vocab.contains_key(b"a"));
    assert!(vocab.contains_key(b"abc"));
    assert!(!vocab.contains_key(b"xyz"));
}

#[test]
fn test_get_pair() {
    let vocab = Vocab::from_entries(sample_entries());
    assert_eq!(vocab.get_pair(b'a', b'b'), Some(2));
    assert_eq!(vocab.get_pair(b'b', b'a'), None);
    assert_eq!(vocab.get_pair(0, 0), None);
}

#[test]
fn test_decode_roundtrip() {
    let entries = sample_entries();
    let vocab = Vocab::from_entries(entries.clone());
    for (token, rank) in &entries {
        assert_eq!(vocab.decode(*rank), token.as_slice());
    }
}

#[test]
fn test_all_single_bytes() {
    let entries: Vec<_> = (0u8..=255).map(|b| (vec![b], b as u32)).collect();
    let vocab = Vocab::from_entries(entries);
    for b in 0u8..=255 {
        assert_eq!(vocab.get(&[b]), Some(b as u32));
        assert_eq!(vocab.decode(b as u32), &[b]);
    }
}

#[test]
fn test_empty_vocab() {
    let vocab = Vocab::from_entries(vec![]);
    assert_eq!(vocab.get(b"anything"), None);
    assert!(!vocab.contains_key(b"x"));
    assert_eq!(vocab.get_pair(b'a', b'b'), None);
}

#[test]
fn test_long_token() {
    let long = vec![0x42u8; 1000];
    let vocab = Vocab::from_entries(vec![(long.clone(), 99)]);
    assert_eq!(vocab.get(&long), Some(99));
    assert_eq!(vocab.decode(99), long.as_slice());
}

#[test]
fn test_inline_boundary_lengths() {
    // exercise every routing class: 1, 2, 3, 4, 7, 8 (inline), 9 (spill)
    let entries: Vec<(Vec<u8>, u32)> = [1usize, 2, 3, 4, 7, 8, 9]
        .iter()
        .enumerate()
        .map(|(i, &l)| (vec![b'x'; l], i as u32))
        .collect();
    let vocab = Vocab::from_entries(entries.clone());
    for (token, rank) in &entries {
        assert_eq!(vocab.get(token), Some(*rank), "len={}", token.len());
    }
    assert_eq!(vocab.get(&[b'x'; 5]), None);
    assert_eq!(vocab.get(&[b'x'; 10]), None);
}

#[test]
fn test_nul_padding_not_confused() {
    // "ab" (len 2) and "ab\0" (len 3) and "ab\0\0" (len 4) share padded
    // key bits; length must separate them in every class
    let entries = vec![
        (b"ab".to_vec(), 1),
        (b"ab\0".to_vec(), 2),
        (b"ab\0\0".to_vec(), 3),
    ];
    let vocab = Vocab::from_entries(entries);
    assert_eq!(vocab.get(b"ab"), Some(1));
    assert_eq!(vocab.get(b"ab\0"), Some(2));
    assert_eq!(vocab.get(b"ab\0\0"), Some(3));
    assert_eq!(vocab.get(b"ab\0\0\0"), None);
}

#[test]
fn test_try_decode_out_of_range() {
    let vocab = Vocab::from_entries(sample_entries());
    assert!(vocab.try_decode(0).is_some());
    assert!(vocab.try_decode(3).is_some());
    assert!(vocab.try_decode(99999).is_none());
}

#[test]
fn test_try_decode_sparse_rank_returns_none() {
    // rank 50 is between rank 0 ("hello") and rank 100 ("world"), but not present
    let vocab = Vocab::from_entries(vec![(b"hello".to_vec(), 0), (b"world".to_vec(), 100)]);
    assert_eq!(vocab.try_decode(0), Some(b"hello".as_slice()));
    assert_eq!(vocab.try_decode(100), Some(b"world".as_slice()));
    // rank 50 does not exist — must return None, not Some(b"")
    assert_eq!(vocab.try_decode(50), None);
    assert_eq!(vocab.try_decode(1), None);
    assert_eq!(vocab.try_decode(99), None);
}

#[cfg(feature = "vocab-cl100k_base")]
#[test]
fn test_matches_hashmap_cl100k() {
    let hashmap = crate::encoding::parse_tiktoken_data_for_test();
    let entries: Vec<_> = hashmap.iter().map(|(k, &v)| (k.clone(), v)).collect();
    let vocab = Vocab::from_entries(entries);

    for (key, &expected_rank) in &hashmap {
        assert_eq!(
            vocab.get(key),
            Some(expected_rank),
            "mismatch for key len={}",
            key.len()
        );
    }
}

#[test]
fn test_sparse_ranks() {
    let entries = vec![(b"hello".to_vec(), 100), (b"world".to_vec(), 50000)];
    let vocab = Vocab::from_entries(entries);
    assert_eq!(vocab.get(b"hello"), Some(100));
    assert_eq!(vocab.get(b"world"), Some(50000));
    assert_eq!(vocab.decode(100), b"hello");
    assert_eq!(vocab.decode(50000), b"world");
}

#[test]
fn test_len() {
    let vocab = Vocab::from_entries(sample_entries());
    assert_eq!(vocab.len(), 4);
}

#[test]
fn test_len_empty() {
    let vocab = Vocab::from_entries(vec![]);
    assert_eq!(vocab.len(), 0);
}

#[test]
fn test_len_sparse() {
    let vocab = Vocab::from_entries(vec![(b"hello".to_vec(), 0), (b"world".to_vec(), 100)]);
    assert_eq!(vocab.len(), 2);
}

#[test]
fn test_many_entries_no_false_positives() {
    let mut entries: Vec<(Vec<u8>, u32)> = Vec::new();
    for i in 0u32..10000 {
        entries.push((i.to_le_bytes().to_vec(), i));
    }
    let vocab = Vocab::from_entries(entries.clone());

    for (token, rank) in &entries {
        assert_eq!(vocab.get(token), Some(*rank));
    }

    for i in 10000u32..10100 {
        let token = i.to_le_bytes().to_vec();
        assert_eq!(vocab.get(&token), None);
    }
}

#[test]
fn test_every_builtin_vocabulary_entry_is_found() {
    for (name, chain) in crate::encoding::vocab_oracle::chains() {
        let ranks = crate::encoding::parse_tkv(chain);
        let entries: Vec<_> = ranks.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let vocab = Vocab::from_entries(entries);
        let mut text = Vec::new();
        for (key, &rank) in &ranks {
            assert_eq!(vocab.get(key), Some(rank), "{name}: len {}", key.len());
            if key.len() <= FAST_KEY_MAX {
                // the key inside a longer buffer, followed by unrelated bytes
                text.clear();
                text.extend_from_slice(b"\xff\xfe");
                text.extend_from_slice(key);
                text.extend_from_slice(&[0xAB; 20]);
                let end = 2 + key.len();
                let (lo, hi) = key_in(&text, 2, end);
                assert_eq!(vocab.fast.get(lo, hi), Some(rank), "{name}");
                if let Some(home) = vocab.get_home(lo, hi) {
                    assert_eq!(home, rank, "{name}");
                }
                // and ending at the very end of the buffer
                assert_eq!(key_in(&text[..end], 2, end), (lo, hi), "{name}");
                assert_eq!(vocab.rank_padded(&text, 2, key.len()), rank, "{name}");
            }
            // neighbours of a real key are found exactly when they are tokens
            let mut longer = key.clone();
            longer.push(b'\x01');
            assert_eq!(vocab.get(&longer), ranks.get(&longer).copied(), "{name}");
            let shorter = &key[..key.len() - 1];
            assert_eq!(vocab.get(shorter), ranks.get(shorter).copied(), "{name}");
            let mut flipped = key.clone();
            *flipped.last_mut().unwrap() ^= 0x40;
            assert_eq!(vocab.get(&flipped), ranks.get(&flipped).copied(), "{name}");
        }
    }
}

#[test]
fn test_fast_and_long_lengths() {
    // every length on both sides of the packed-key limit, with shared prefixes
    let mut entries = Vec::new();
    for len in 1..=40usize {
        entries.push((vec![b'q'; len], len as u32));
        let mut v: Vec<u8> = (0..len as u8).map(|i| b'a' + i % 26).collect();
        entries.push((v.clone(), 100 + len as u32));
        v[len - 1] = 0;
        entries.push((v, 200 + len as u32));
    }
    let vocab = Vocab::from_entries(entries.clone());
    for (token, rank) in &entries {
        assert_eq!(vocab.get(token), Some(*rank), "len {}", token.len());
        let mut padded = token.clone();
        padded.resize(token.len() + KEY_READ, 0x55);
        assert_eq!(vocab.rank_padded(&padded, 0, token.len()), *rank);
    }
    assert_eq!(vocab.get(&[b'q'; 41]), None);
    assert_eq!(vocab.get(b"qqqqqqqqqqqqqqqqqqqqx"), None);
    let mut padded = vec![b'q'; 16 + KEY_READ];
    padded[14] = b'x';
    assert_eq!(vocab.rank_padded(&padded, 0, 16), ABSENT);
    assert_eq!(vocab.rank_padded(&padded, 0, 15), ABSENT);
}

#[test]
fn test_table_keeps_an_empty_slot() {
    let entries: Vec<(Vec<u8>, u32)> = (0u32..1000)
        .map(|i| (i.to_le_bytes().to_vec(), i))
        .collect();
    let vocab = Vocab::from_entries(entries);
    if !cfg!(target_arch = "wasm32") {
        assert!(vocab.fast.len() >= 2000);
    }
    assert!(vocab.fast.len() > 1000);
}

#[test]
fn test_lowest_rank_owns_its_home_slot() {
    // inserted first, so nothing can displace it
    let entries: Vec<(Vec<u8>, u32)> = (0u32..5000)
        .map(|i| (format!("k{i}").into_bytes(), i))
        .collect();
    let vocab = Vocab::from_entries(entries.clone());
    let (lo, hi) = key_of(&entries[0].0);
    assert_eq!(vocab.fast.is_home(lo, hi), Some(true));
    let (lo, hi) = key_of(b"absent");
    assert_eq!(vocab.fast.is_home(lo, hi), None);
}
