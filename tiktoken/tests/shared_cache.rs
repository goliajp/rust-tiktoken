//! Merge results are shared between threads through the vocabulary table.
//! Threads that encode at the same time, inserting entries for the same and
//! for different pieces, must get exactly what a single thread on a fresh
//! instance gets.

use std::sync::Barrier;
use std::thread;

use tiktoken::CoreBpe;

const THREADS: usize = 8;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

const SCRIPTS: [(u32, u32); 6] = [
    (0x61, 0x7b),
    (0x41, 0x5b),
    (0x4e00, 0x9fa0),
    (0x3040, 0x30a0),
    (0xac00, 0xd7a0),
    (0x620, 0x650),
];

// 1..=9 characters of one script: pieces of every length class, many of
// them not tokens
fn word(next: &mut impl FnMut() -> u64) -> String {
    let r = next();
    let (lo, hi) = SCRIPTS[(r >> 4) as usize % SCRIPTS.len()];
    (0..1 + (r >> 12) % 9)
        .map(|_| char::from_u32(lo + (next() % (hi - lo) as u64) as u32).unwrap())
        .collect()
}

// a third of the words come from `shared`, which every thread's text uses
fn text(seed: u64, shared: &[String]) -> String {
    let mut next = rng(seed);
    let mut s = String::new();
    for _ in 0..1500 {
        let r = next();
        if r.is_multiple_of(3) {
            s.push_str(&shared[(r >> 8) as usize % shared.len()]);
        } else {
            s.push_str(&word(&mut next));
        }
        s.push([' ', ' ', ',', '\n', '.'][(r >> 20) as usize % 5]);
    }
    s
}

fn check(make: fn() -> CoreBpe) {
    let mut next = rng(99);
    let shared: Vec<String> = (0..64).map(|_| word(&mut next)).collect();
    let texts: Vec<String> = (0..THREADS as u64).map(|i| text(i + 1, &shared)).collect();
    let reference = make();
    let want: Vec<Vec<u32>> = texts.iter().map(|t| reference.encode(t)).collect();

    let enc = make();
    for round in 0..3 {
        let barrier = Barrier::new(THREADS);
        thread::scope(|s| {
            for (t, w) in texts.iter().zip(&want) {
                let (enc, barrier) = (&enc, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    assert_eq!(&enc.encode(t), w, "round {round}");
                    assert_eq!(enc.count(t), w.len(), "round {round}");
                });
            }
        });
        if round == 1 {
            // SAFETY: no thread is encoding with `enc` here
            unsafe { enc.__reset_shared_cache() };
        }
    }
}

#[cfg(feature = "vocab-cl100k_base")]
#[test]
fn concurrent_first_encodes_match_a_fresh_instance_cl100k() {
    check(tiktoken::encoding::cl100k_base);
}

#[cfg(feature = "vocab-o200k_base")]
#[test]
fn concurrent_first_encodes_match_a_fresh_instance_o200k() {
    check(tiktoken::encoding::o200k_base);
}
