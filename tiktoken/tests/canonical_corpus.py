"""Generate a canonical-tiktoken corpus for the `canonical_parity` test.

Emits {encoding: [[text, [token, ...]], ...]} using the reference Python
implementation, over a corpus that deliberately stresses whitespace and newline
runs — the axis where issue #5 diverged.

    pip install tiktoken
    python tiktoken/tests/canonical_corpus.py /tmp/canonical.json
    CANONICAL_JSON=/tmp/canonical.json cargo test -p tiktoken \
        --test canonical_parity -- --ignored --nocapture
"""

import json
import random
import sys

import tiktoken

ENCODINGS = ["cl100k_base", "o200k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2"]

WS = ["", " ", "  ", "   ", "\t", "\t\t", "\n", "\n\n", "\n\n\n", "\r", "\r\n",
      "\r\n\r\n", " \n", "\n ", " \n ", "  \n  ", "\t\n", "\n\t", " \r\n ",
      "\x0b", "\x0c", " ", " ", " ", "　", " ",
      "   ", "\n \n", "　\n", "\n　"]

NEXT = ["", "a", "A", "1", "!", "@", ".", "'s", " ", "\n", "你", "🎉", "@rem",
        "word", "-", "_", " ", "　", "'", "\"", ")", "#", "/", "//x"]

PREV = ["", "word", "a", "1", "!", ".", "你好", "🎉", "end.", "x"]


def corpus():
    texts = []

    # ---- baseline corpus (mirrors tests/generate_oracle.rs) ----
    texts += [
        "", " ", "  ", "   ", "\n", "\t", "\r\n", "  \n  \n  ",
        "a", "Z", "0", "!", "@", "#",
        "hello", "hello world", "Hello World", "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "  hello", "   hello", "hello   ", "hello\t  world",
        "hello\nworld", "hello  \n  world",
        "I'm", "don't", "they're", "we've", "she'll", "it'd",
        "123", "1234567890", "3.14159", "1,000,000",
        "你好", "你好世界", "こんにちは", "こんにちは世界", "안녕하세요", "世界你好世界",
        "Hello 你好 World", "Hello 你好 🌍", "café résumé naïve über",
        "日本語テスト 🎉", "café résumé naïve über 日本語 한국어 العربية",
        "🎉", "🚀💡🎯", "👨‍👩‍👧‍👦", "🇯🇵",
        'fn main() { println!("Hello"); }',
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        'const x = { key: "value", arr: [1, 2, 3] };',
        "<|endoftext|>", "test<|endoftext|>test", "<|fim_prefix|>hello<|fim_suffix|>",
        "Hello!!! What?? Yes... No---maybe",
        "a@b.com http://example.com/path?q=1&r=2",
        "a" * 100, "hello " * 100, "word " * 1000,
        "The quick brown fox. " * 200, "你好世界！" * 200,
        "مرحبا بالعالم", "인공지능 기술이 빠르게 발전하고 있습니다",
        "\t\t\thello\t\t\t", "line1\nline2\nline3", "line1\r\nline2\r\nline3",
        "The year is 2024 and pi is 3.14159.", "100% of $1,000.00 = $1,000.00",
        "x", "xx", "xxx", "x" * 10, "x" * 50, "x" * 255, "x" * 256, "x" * 257,
    ]

    # ---- adversarial: prev x whitespace x next matrix ----
    for p in PREV:
        for w in WS:
            for n in NEXT:
                texts.append(p + w + n)

    # ---- realistic newline-heavy documents ----
    texts += [
        "word\n\nnext",
        "\r\n@rem",
        "@echo off\r\n@rem comment\r\n\r\nexit /b 0",
        "# Title\n\nParagraph one.\n\nParagraph two.\n\n- item\n- item\n",
        "def f():\n\n    return 1\n\n\ndef g():\n\n    return 2\n",
        "a\n\nb\n\n\nc\n\n\n\nd",
        "line\r\n\r\nline\r\n\r\n\r\nline",
        "trailing\n", "trailing\n\n", "trailing\r\n", "trailing \n",
        "\n\nleading", "\r\n\r\nleading", " \n\nleading",
        "mixed \t\n\n\t end",
        "json:\n{\n  \"a\": 1,\n\n  \"b\": 2\n}\n",
        "CSV\r\n1,2,3\r\n4,5,6\r\n",
    ]

    # ---- unicode whitespace runs ----
    for w in [" ", " ", " ", "　", " ", " ", " "]:
        for n in ["", "a", "\n", " ", "你"]:
            texts.append("x" + w + n)
            texts.append("x" + w + w + n)
            texts.append("x" + w + "\n" + n)
            texts.append("x\n" + w + n)

    # ---- random fuzz over an interesting alphabet ----
    rng = random.Random(20260808)
    alphabet = list(" \t\n\r\x0b\x0cabcXYZ019!@#'\".,-_()[]{}你好🎉 　é")
    for _ in range(4000):
        length = rng.randint(1, 40)
        texts.append("".join(rng.choice(alphabet) for _ in range(length)))

    # dedupe, keep order
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    texts = corpus()
    data = {}
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        # encode ordinary text only; special tokens stay literal
        data[name] = [[t, enc.encode(t, disallowed_special=())] for t in texts]
        print(f"{name}: {len(texts)} cases", file=sys.stderr)
    out = sys.argv[1] if len(sys.argv) > 1 else "canonical.json"
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"wrote {out} ({len(texts)} texts x {len(ENCODINGS)} encodings)", file=sys.stderr)


if __name__ == "__main__":
    main()
