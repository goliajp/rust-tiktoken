//! Ours vs tiktoken-rs on identical corpora. Corpora are byte-identical to
//! `tiktoken/examples/stage_bench.rs`, `tiktoken/benches/tokenize.rs`,
//! `tiktoken/benches/bench_python.py` and `web/bench/bench.ts`.

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

pub fn cases() -> Vec<(&'static str, String)> {
    vec![
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
            "zh_prose_4kb",
            "分词器把文本切成 token，模型按 token 计费。同一段话在不同词表下的 token 数可能相差一倍以上，因此计费、上下文上限和截断位置都取决于分词是否准确。本实现覆盖多家厂商的编码，每一套都与参考实现逐字节比对，至今没有发现分歧。速度来自手写的扫描器：常见片段不进正则引擎，词级片段在栈上合并，零分配。"
                .repeat(10),
        ),
        (
            "ja_prose_4kb",
            "トークナイザーはテキストをトークンへ分割し、モデルはトークン単位で課金します。同じ文章でも語彙が違えばトークン数は大きく変わるため、分割の正確さは請求額と文脈上限に直結します。本実装は各ベンダーのエンコーディングを収録し、いずれも参照実装とバイト単位で照合済みです。速度は手書きスキャナによるもので、一般的な断片は正規表現エンジンを通しません。"
                .repeat(9),
        ),
        (
            "code_3kb",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# compute first 100 fibonacci numbers\nresults = [fibonacci(i) for i in range(100)]\nprint(results)\n".repeat(20),
        ),
    ]
}

fn median_ns(text: &str, mut f: impl FnMut(&str) -> usize) -> f64 {
    for _ in 0..50 {
        std::hint::black_box(f(text));
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
            std::hint::black_box(f(text));
        }
        runs.push(s.elapsed().as_nanos() as f64 / n as f64);
    }
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    runs[4]
}

fn run(enc_name: &str, ours: &tiktoken::CoreBpe, rs: &tiktoken_rs::CoreBPE) {
    println!("\n{enc_name}");
    println!(
        "{:<20}{:>14}{:>14}{:>14}  agree",
        "corpus", "ours encode", "rs encode", "ours count"
    );
    for (name, text) in cases() {
        let a = ours.encode(&text);
        let b = rs.encode_ordinary(&text);
        let agree = a.len() == b.len()
            && a.iter()
                .map(|&x| x as usize)
                .eq(b.iter().map(|&x| x as usize));
        let oe = median_ns(&text, |t| ours.encode(t).len());
        let re = median_ns(&text, |t| rs.encode_ordinary(t).len());
        let oc = median_ns(&text, |t| ours.count(t));
        // tiktoken-rs has no count(); its users count via encode().len()
        println!(
            "{name:<20}{oe:>12.0}ns{re:>12.0}ns{oc:>12.0}ns  {}",
            if agree {
                "yes"
            } else {
                "NO — invalid comparison"
            }
        );
    }
}

fn main() {
    run(
        "cl100k_base",
        tiktoken::get_encoding("cl100k_base").unwrap(),
        &tiktoken_rs::cl100k_base().unwrap(),
    );
    run(
        "o200k_base",
        tiktoken::get_encoding("o200k_base").unwrap(),
        &tiktoken_rs::o200k_base().unwrap(),
    );
}
