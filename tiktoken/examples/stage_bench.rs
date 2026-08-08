//! Same-machine comparison harness for the Unicode-path work.
//!
//! Mirrors `web/bench/bench.ts` exactly — same corpora (the `varied`
//! generator is reproduced byte-for-byte), same shape of timing (warmup, then
//! median of 9 rounds) — so a native number and a browser number from the same
//! machine are directly comparable. Absolute README numbers still come from
//! the mini; this exists to attribute changes, not to publish.
//!
//!   cargo run --release --example stage_bench

use std::time::Instant;

fn varied(n: u32) -> String {
    let mut s = String::new();
    let mut x: u32 = 12345;
    let blocks = [
        (0x4e00u32, 0x9fa0u32),
        (0x3040, 0x30a0),
        (0xac00, 0xd780),
        (0x620, 0x650),
    ];
    for i in 0..n {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let (lo, hi) = blocks[((x >> 7) % 4) as usize];
        s.push(char::from_u32(lo + ((x >> 9) % (hi - lo))).unwrap());
        if i % 7 == 6 {
            s.push('\u{ff0c}');
        }
        if i % 23 == 22 {
            s.push(' ');
        }
    }
    s
}

fn main() {
    let cases: Vec<(&str, String)> = vec![
        ("short_13b", "Hello, world!".into()),
        (
            "medium_900b",
            "The quick brown fox jumps over the lazy dog. ".repeat(20),
        ),
        (
            "long_45kb",
            "The quick brown fox jumps over the lazy dog. ".repeat(1000),
        ),
        (
            "unicode_4kb",
            "你好世界！こんにちは世界！안녕하세요 세계！مرحبا بالعالم ".repeat(50),
        ),
        ("unicode_varied_4kb", varied(1200)),
        (
            "code_3kb",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# compute first 100 fibonacci numbers\nresults = [fibonacci(i) for i in range(100)]\nprint(results)\n".repeat(20),
        ),
    ];
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    println!("{:<20}{:>12}  tokens", "corpus", "count()");
    for (name, text) in &cases {
        for _ in 0..50 {
            std::hint::black_box(enc.count(text));
        }
        let n = if text.len() < 100 {
            20000
        } else if text.len() < 2000 {
            2000
        } else {
            200
        };
        let mut runs = vec![];
        for _ in 0..9 {
            let s = Instant::now();
            for _ in 0..n {
                std::hint::black_box(enc.count(text));
            }
            runs.push(s.elapsed().as_nanos() as f64 / n as f64);
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name:<20}{:>10.0} ns  {}", runs[4], enc.count(text));
    }
}
