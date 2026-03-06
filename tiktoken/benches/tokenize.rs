use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn make_cases() -> Vec<(&'static str, String)> {
    vec![
        ("short_13b", "Hello, world!".to_string()),
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
        (
            "code_3kb",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# compute first 100 fibonacci numbers\nresults = [fibonacci(i) for i in range(100)]\nprint(results)\n".repeat(20),
        ),
    ]
}

fn bench_encoding(c: &mut Criterion, enc_name: &str) {
    let enc = tiktoken::get_encoding(enc_name).unwrap();
    let cases = make_cases();

    let mut group = c.benchmark_group(format!("{enc_name}/encode"));
    for (name, text) in &cases {
        group.bench_with_input(BenchmarkId::new(*name, text.len()), text, |b, t| {
            b.iter(|| enc.encode(t));
        });
    }
    group.finish();

    let mut group = c.benchmark_group(format!("{enc_name}/count"));
    for (name, text) in &cases {
        group.bench_with_input(BenchmarkId::new(*name, text.len()), text, |b, t| {
            b.iter(|| enc.count(t));
        });
    }
    group.finish();
}

fn bench_cl100k(c: &mut Criterion) {
    bench_encoding(c, "cl100k_base");
}

fn bench_o200k(c: &mut Criterion) {
    bench_encoding(c, "o200k_base");
}

criterion_group!(benches, bench_cl100k, bench_o200k);
criterion_main!(benches);
