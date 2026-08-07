#!/usr/bin/env python3
"""Regenerate the OpenAI oracle fixtures from canonical `openai/tiktoken`.

These fixtures are ground truth for the encodings this crate reimplements, so
they MUST come from the reference implementation — never from this crate's own
output. A self-generated snapshot only proves we still agree with ourselves,
which is how the issue #5 newline-splitting bug survived a green test suite.

Usage:
    pip install tiktoken
    python tiktoken/tests/fixtures/generate_openai_fixtures.py

Fixtures for HuggingFace-sourced encodings (llama3, qwen2, deepseek_v3,
mistral_v3) are generated from their respective HF reference tokenizers and are
not touched by this script.
"""

import json
import pathlib

import tiktoken

ENCODINGS = ["cl100k_base", "o200k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2"]

OUT_DIR = pathlib.Path(__file__).resolve().parent

# Whitespace runs, the axis that distinguishes the `\s*[\r\n]+` branch from the
# generic `\s+(?!\S)|\s+` branch. Regressions here silently change token ids.
WHITESPACE = [
    " ", "  ", "   ", "\t", "\t\t",
    "\n", "\n\n", "\n\n\n", "\r", "\r\n", "\r\n\r\n",
    " \n", "\n ", " \n ", "  \n  ", "\t\n", "\n\t", " \r\n ", "\n \n",
    "\x0b", "\x0c",
    " ", " ", " ", "　", " ",  # unicode spaces
    "　\n", "\n　",
]

# What follows the run decides whether the lookahead would have trimmed it.
FOLLOWERS = ["", "a", "A", "1", "!", "@", ".", "'s", " ", "\n", "你", "\U0001f389", "@rem", "word"]

# What precedes it decides which branch claims the leading character.
PRECEDERS = ["", "word", "1", "!", "你好", "\U0001f389"]


def corpus():
    texts = [
        # empty / trivial
        "", " ", "  ", "   ", "\n", "\t", "\r\n", "  \n  \n  ",
        # single characters
        "a", "Z", "0", "!", "@", "#",
        # short english
        "hello", "hello world", "Hello World", "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        # whitespace variations
        "  hello", "   hello", "hello   ", "hello\t  world",
        "hello\nworld", "hello  \n  world",
        # contractions
        "I'm", "don't", "they're", "we've", "she'll", "it'd",
        # numbers
        "123", "1234567890", "3.14159", "1,000,000",
        # CJK
        "你好", "你好世界", "こんにちは",
        "こんにちは世界", "안녕하세요",
        "世界你好世界",
        # mixed script
        "Hello 你好 World", "Hello 你好 \U0001f30d",
        "café résumé naïve über",
        "日本語テスト \U0001f389",
        "café résumé naïve über 日本語 한국어 العربية",
        # emoji
        "\U0001f389", "\U0001f680\U0001f4a1\U0001f3af",
        "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466", "\U0001f1ef\U0001f1f5",
        # code
        'fn main() { println!("Hello"); }',
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        'const x = { key: "value", arr: [1, 2, 3] };',
        # special token text (as plain text, not special tokens)
        "<|endoftext|>", "test<|endoftext|>test", "<|fim_prefix|>hello<|fim_suffix|>",
        # punctuation heavy
        "Hello!!! What?? Yes... No---maybe",
        "a@b.com http://example.com/path?q=1&r=2",
        # long repetitive
        "a" * 100, "hello " * 100, "word " * 1000,
        "The quick brown fox. " * 200, "你好世界！" * 200,
        # arabic / korean
        "مرحبا بالعالم",
        "인공지능 기술이 빠르게 발전하고 있습니다",
        # tabs and mixed whitespace
        "\t\t\thello\t\t\t", "line1\nline2\nline3", "line1\r\nline2\r\nline3",
        # numbers in context
        "The year is 2024 and pi is 3.14159.", "100% of $1,000.00 = $1,000.00",
        # boundary lengths
        "x", "xx", "xxx", "x" * 10, "x" * 50, "x" * 255, "x" * 256, "x" * 257,
    ]

    # --- issue #5 regression surface: newline tokens must stay whole ---
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
        'json:\n{\n  "a": 1,\n\n  "b": 2\n}\n',
        "CSV\r\n1,2,3\r\n4,5,6\r\n",
    ]

    # --- systematic preceder x whitespace-run x follower matrix ---
    for p in PRECEDERS:
        for w in WHITESPACE:
            for f in FOLLOWERS:
                texts.append(p + w + f)

    # dedupe, preserving order
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    texts = corpus()
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        cases = []
        for text in texts:
            # encode as ordinary text: special tokens stay literal, matching
            # CoreBpe::encode (not encode_with_special_tokens)
            tokens = enc.encode(text, disallowed_special=())
            cases.append({"text": text, "tokens": tokens, "count": len(tokens)})
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
