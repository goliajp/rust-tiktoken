"""Benchmark Python tiktoken with identical test data as Rust Criterion benchmarks."""

import time
import tiktoken

CASES = [
    ("short_13b", "Hello, world!"),
    ("medium_900b", "The quick brown fox jumps over the lazy dog. " * 20),
    ("long_45kb", "The quick brown fox jumps over the lazy dog. " * 1000),
    ("unicode_2kb", "你好世界！こんにちは世界！안녕하세요 세계！مرحبا بالعالم " * 50),
    ("code_3kb", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# compute first 100 fibonacci numbers\nresults = [fibonacci(i) for i in range(100)]\nprint(results)\n" * 20),
]

ENCODINGS = ["cl100k_base", "o200k_base"]
WARMUP = 50
ITERATIONS = 500


def bench_encode(enc, text, iterations):
    # warmup
    for _ in range(WARMUP):
        enc.encode(text)

    start = time.perf_counter_ns()
    for _ in range(iterations):
        enc.encode(text)
    elapsed = time.perf_counter_ns() - start
    return elapsed / iterations


def format_time(ns):
    if ns < 1_000:
        return f"{ns:.0f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f} µs"
    else:
        return f"{ns / 1_000_000:.1f} ms"


def main():
    print(f"Python tiktoken {tiktoken.__version__}")
    print(f"{'':=<80}")

    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        print(f"\n{enc_name} encode:")
        print(f"  {'Case':<20} {'Bytes':>8} {'Time':>12} {'Tokens':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*8}")

        for case_name, text in CASES:
            ns_per_iter = bench_encode(enc, text, ITERATIONS)
            tokens = len(enc.encode(text))
            print(f"  {case_name:<20} {len(text.encode('utf-8')):>8} {format_time(ns_per_iter):>12} {tokens:>8}")


if __name__ == "__main__":
    main()
