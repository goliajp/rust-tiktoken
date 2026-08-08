"""Benchmark Python tiktoken with identical test data as Rust Criterion benchmarks."""

import time
import tiktoken

def _varied(n):
    """Non-repeating CJK stream; byte-identical to bench-compare/src/main.rs."""
    out, x = [], 12345
    blocks = [(0x4E00, 0x9FA0), (0x3040, 0x30A0), (0xAC00, 0xD780), (0x0620, 0x0650)]
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        lo, hi = blocks[(x >> 7) % 4]
        out.append(chr(lo + (x >> 9) % (hi - lo)))
        if i % 7 == 6:
            out.append("，")
        if i % 23 == 22:
            out.append(" ")
    return "".join(out)


CASES = [
    ("short_13b", "Hello, world!"),
    ("medium_900b", "The quick brown fox jumps over the lazy dog. " * 20),
    ("long_45kb", "The quick brown fox jumps over the lazy dog. " * 1000),
    ("unicode_4kb", "你好世界！こんにちは世界！안녕하세요 세계！مرحبا بالعالم " * 50),
    ("unicode_varied_4kb", _varied(1200)),
    ("zh_prose_4kb", "分词器把文本切成 token，模型按 token 计费。同一段话在不同词表下的 token 数可能相差一倍以上，因此计费、上下文上限和截断位置都取决于分词是否准确。本实现覆盖多家厂商的编码，每一套都与参考实现逐字节比对，至今没有发现分歧。速度来自手写的扫描器：常见片段不进正则引擎，词级片段在栈上合并，零分配。" * 10),
    ("ja_prose_4kb", "トークナイザーはテキストをトークンへ分割し、モデルはトークン単位で課金します。同じ文章でも語彙が違えばトークン数は大きく変わるため、分割の正確さは請求額と文脈上限に直結します。本実装は各ベンダーのエンコーディングを収録し、いずれも参照実装とバイト単位で照合済みです。速度は手書きスキャナによるもので、一般的な断片は正規表現エンジンを通しません。" * 9),
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
