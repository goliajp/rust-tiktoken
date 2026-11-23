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
                text.extend_from_slice(&[0xFF; KEY_READ]);
                text.extend_from_slice(key);
                text.extend_from_slice(&[0xAB; 20]);
                let end = KEY_READ + key.len();
                let (lo, hi) = key_in(&text, KEY_READ, end);
                assert_eq!(vocab.fast.get(lo, hi), Some(rank), "{name}");
                // the merge filter has no false negatives
                assert!(vocab.filter.may_contain(key::hash(lo, hi)), "{name}");
                if let Some(home) = vocab.get_home(lo, hi) {
                    assert_eq!(home, rank, "{name}");
                }
                // and ending at the very end of the buffer
                assert_eq!(key_in(&text[..end], KEY_READ, end), (lo, hi), "{name}");
                assert_eq!(
                    vocab.rank_padded(&text, KEY_READ, key.len()),
                    rank,
                    "{name}"
                );
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
fn test_compact_table_keeps_room_for_derived_entries() {
    use super::fast::table_slots;
    for n in [
        1usize, 3, 1000, 50_257, 100_256, 128_000, 151_643, 199_998, 248_044,
    ] {
        let full = table_slots(n, false);
        assert_eq!(full, (n * 2).max(4).next_power_of_two());
        let slots = table_slots(n, true);
        assert!(
            slots.is_multiple_of(2) && slots > n && slots <= n * 8 / 5 + 4,
            "n {n}: {slots}"
        );
        // derived capacity as FastTable::build computes it
        let cap = (slots / 4).min((slots * 3 / 4).saturating_sub(n));
        assert!(cap >= n / 5, "n {n}: {slots} slots leave {cap} derived");
    }
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

fn ids_of(vocab: &Vocab, key: (u64, u64)) -> Option<Vec<u32>> {
    let e = vocab.entry(key.0, key.1)?;
    let mut out = vec![];
    vocab.extend_entry(e, &mut out);
    assert_eq!(out.len(), e.len());
    Some(out)
}

#[test]
fn derived_entries_are_found_only_as_entries() {
    let entries: Vec<(Vec<u8>, u32)> = (0u32..1000)
        .map(|i| (format!("t{i}").into_bytes(), i))
        .collect();
    let vocab = Vocab::from_entries(entries);
    let cap = vocab.fast.derived_cap() as usize;
    assert!(cap > 100);
    let key = |i: usize| key_of(format!("d{i:0>width$}", width = 1 + i % 14).as_bytes());
    let ids = |i: usize| -> Vec<u32> { (0..2 + i % 14).map(|j| (i * 31 + j) as u32).collect() };
    // entries of more than three ids may also be refused when their arena
    // runs out; the rest must all be taken until the table is full
    let mut stored = vec![];
    let mut i = 0;
    while vocab.accepts_derived() {
        let (lo, hi) = key(i);
        if vocab.insert_derived(lo, hi, &ids(i)) {
            // publishing the same key again changes nothing
            vocab.insert_derived(lo, hi, &[7, 7]);
            stored.push(i);
        } else {
            assert!(ids(i).len() > 3, "{i}");
            assert_eq!(ids_of(&vocab, (lo, hi)), None);
        }
        i += 1;
        assert!(i < 100 * cap);
    }
    assert_eq!(stored.len(), cap);
    assert!(!vocab.insert_derived(1, 1 << 56, &[1, 2]));
    for &i in &stored {
        let (lo, hi) = key(i);
        assert_eq!(ids_of(&vocab, (lo, hi)), Some(ids(i)), "{i}");
        // vocabulary lookups never see a derived entry
        assert_eq!(vocab.fast.get(lo, hi), None);
        assert_eq!(vocab.get_home(lo, hi), None);
    }
    for i in 0..1000u32 {
        let k = format!("t{i}");
        assert_eq!(vocab.get(k.as_bytes()), Some(i));
        assert_eq!(ids_of(&vocab, key_of(k.as_bytes())), Some(vec![i]));
    }
    vocab.clear_derived();
    assert!(vocab.accepts_derived());
    for &i in &stored {
        assert_eq!(ids_of(&vocab, key(i)), None);
    }
    for i in 0..1000u32 {
        assert_eq!(vocab.get(format!("t{i}").as_bytes()), Some(i));
    }
    let (lo, hi) = key(3);
    assert!(vocab.insert_derived(lo, hi, &ids(3)));
    assert_eq!(ids_of(&vocab, (lo, hi)), Some(ids(3)));
}

#[test]
fn derived_entries_published_concurrently_read_back_whole() {
    let entries: Vec<(Vec<u8>, u32)> = (0u32..200)
        .map(|i| (format!("t{i}").into_bytes(), i))
        .collect();
    let vocab = Vocab::from_entries(entries);
    let keys: Vec<Vec<u8>> = (0..48)
        .map(|i| format!("p{i:0>w$}", w = 1 + i % 13).into_bytes())
        .collect();
    // the ids are a function of the key, so whichever thread wins a race
    // publishes the same value
    let ids = |k: &[u8]| -> Vec<u32> {
        let seed = k.iter().fold(0u32, |a, &b| a.wrapping_mul(31) ^ b as u32) & 0xFFFF;
        (0..2 + k.len() % 7).map(|j| seed + j as u32).collect()
    };
    std::thread::scope(|s| {
        for t in 0..4 {
            let (vocab, keys) = (&vocab, &keys);
            s.spawn(move || {
                for round in 0..2 {
                    for (i, k) in keys.iter().enumerate() {
                        let key = key_of(k);
                        if (i + t + round) % 2 == 0 && ids_of(vocab, key).is_none() {
                            vocab.insert_derived(key.0, key.1, &ids(k));
                        } else if let Some(got) = ids_of(vocab, key) {
                            assert_eq!(got, ids(k));
                        }
                    }
                }
            });
        }
    });
    for k in &keys {
        assert_eq!(ids_of(&vocab, key_of(k)), Some(ids(k)));
    }
}

#[test]
fn sharded_count_stops_inserts_shortly_after_cap() {
    let entries: Vec<(Vec<u8>, u32)> = (0u32..5000)
        .map(|i| (format!("t{i}").into_bytes(), i))
        .collect();
    let vocab = Vocab::from_entries(entries);
    let cap = vocab.fast.derived_cap() as usize;
    // large enough that inserts are counted per shard
    assert!(cap >= 2048, "{cap}");
    let key = |t: usize, i: usize| key_of(format!("{t}:{i}").as_bytes());
    let ids = |t: usize, i: usize| vec![t as u32, i as u32];
    let stored: Vec<Vec<usize>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let vocab = &vocab;
                s.spawn(move || {
                    let mut mine = vec![];
                    let mut i = 0;
                    while vocab.accepts_derived() {
                        let (lo, hi) = key(t, i);
                        if vocab.insert_derived(lo, hi, &ids(t, i)) {
                            mine.push(i);
                        }
                        i += 1;
                    }
                    mine
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let total: usize = stored.iter().map(Vec::len).sum();
    assert!((cap..=cap + 16 * 32).contains(&total), "{total} of {cap}");
    for (t, mine) in stored.iter().enumerate() {
        for &i in mine {
            assert_eq!(ids_of(&vocab, key(t, i)), Some(ids(t, i)), "{t}:{i}");
        }
    }
    for i in 0..5000u32 {
        assert_eq!(vocab.get(format!("t{i}").as_bytes()), Some(i));
    }
}

#[test]
fn clearing_derived_entries_forgets_all_of_them() {
    let entries: Vec<(Vec<u8>, u32)> = (0u32..5000)
        .map(|i| (format!("t{i}").into_bytes(), i))
        .collect();
    let vocab = Vocab::from_entries(entries);
    let key = |t: usize, i: usize| key_of(format!("{t}:{i}").as_bytes());
    let long = |t: usize, i: usize| format!("{t}:{i}:{}", "x".repeat(16 + i % 40)).into_bytes();
    for (threads, per_thread) in [(3, 10), (3, 700), (1, 2500)] {
        std::thread::scope(|s| {
            for t in 0..threads {
                let vocab = &vocab;
                s.spawn(move || {
                    for i in 0..per_thread {
                        let (lo, hi) = key(t, i);
                        assert!(vocab.insert_derived(lo, hi, &[t as u32, i as u32]));
                        let p = long(t, i);
                        assert!(vocab.insert_long(&vocab.long_piece(&p), &[1, 2, i as u32]));
                    }
                });
            }
        });
        for t in 0..threads {
            for i in 0..per_thread {
                assert_eq!(ids_of(&vocab, key(t, i)), Some(vec![t as u32, i as u32]));
                assert!(vocab.long_ids(&long(t, i)).is_some());
            }
        }
        vocab.clear_derived();
        for t in 0..threads {
            for i in 0..per_thread {
                assert_eq!(ids_of(&vocab, key(t, i)), None, "{per_thread} {t}:{i}");
                assert!(
                    vocab.long_ids(&long(t, i)).is_none(),
                    "{per_thread} {t}:{i}"
                );
            }
        }
        for i in 0..5000u32 {
            assert_eq!(vocab.get(format!("t{i}").as_bytes()), Some(i));
        }
    }
}
