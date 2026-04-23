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

    // encode
    let mut group = c.benchmark_group(format!("{enc_name}/encode"));
    for (name, text) in &cases {
        group.bench_with_input(BenchmarkId::new(*name, text.len()), text, |b, t| {
            b.iter(|| enc.encode(t));
        });
    }
    group.finish();

    // count
    let mut group = c.benchmark_group(format!("{enc_name}/count"));
    for (name, text) in &cases {
        group.bench_with_input(BenchmarkId::new(*name, text.len()), text, |b, t| {
            b.iter(|| enc.count(t));
        });
    }
    group.finish();

    // decode (encode once, then benchmark decoding)
    let pre_encoded: Vec<(&str, Vec<u32>)> = cases
        .iter()
        .map(|(name, text)| (*name, enc.encode(text)))
        .collect();
    let mut group = c.benchmark_group(format!("{enc_name}/decode"));
    for (name, tokens) in &pre_encoded {
        group.bench_with_input(BenchmarkId::new(*name, tokens.len()), tokens, |b, toks| {
            b.iter(|| enc.decode(toks));
        });
    }
    group.finish();
}

fn bench_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("init");
    group.sample_size(10); // init is slow, fewer samples needed

    macro_rules! bench_init_encoding {
        ($($name:literal => $fn:path),+ $(,)?) => {
            $(
                group.bench_function($name, |b| {
                    b.iter(|| $fn());
                });
            )+
        };
    }

    bench_init_encoding! {
        "cl100k_base" => tiktoken::encoding::cl100k_base,
        "o200k_base"  => tiktoken::encoding::o200k_base,
        "p50k_base"   => tiktoken::encoding::p50k_base,
        "p50k_edit"   => tiktoken::encoding::p50k_edit,
        "r50k_base"   => tiktoken::encoding::r50k_base,
        "llama3"      => tiktoken::encoding::llama3,
        "deepseek_v3" => tiktoken::encoding::deepseek_v3,
        "qwen2"       => tiktoken::encoding::qwen2,
        "mistral_v3"  => tiktoken::encoding::mistral_v3,
    }

    group.finish();
}

// generate one bench function per encoding to keep criterion groups separate
macro_rules! bench_fns {
    ($($fn_name:ident => $enc_name:literal),+ $(,)?) => {
        $(
            fn $fn_name(c: &mut Criterion) {
                bench_encoding(c, $enc_name);
            }
        )+
    };
}

bench_fns! {
    bench_cl100k_base => "cl100k_base",
    bench_o200k_base  => "o200k_base",
    bench_p50k_base   => "p50k_base",
    bench_p50k_edit   => "p50k_edit",
    bench_r50k_base   => "r50k_base",
    bench_llama3      => "llama3",
    bench_deepseek_v3 => "deepseek_v3",
    bench_qwen2       => "qwen2",
    bench_mistral_v3  => "mistral_v3",
}

criterion_group!(
    benches,
    bench_init,
    bench_cl100k_base,
    bench_o200k_base,
    bench_p50k_base,
    bench_p50k_edit,
    bench_r50k_base,
    bench_llama3,
    bench_deepseek_v3,
    bench_qwen2,
    bench_mistral_v3,
);
criterion_main!(benches);
