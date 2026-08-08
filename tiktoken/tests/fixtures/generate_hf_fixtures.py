#!/usr/bin/env python3
"""Regenerate the HuggingFace-sourced oracle fixtures from reference tokenizers.

Counterpart to `generate_openai_fixtures.py`. Same rule applies: these fixtures
are ground truth for encodings this crate reimplements, so they must come from
the reference tokenizer, never from this crate's own output.

Usage:
    pip install tokenizers
    python tiktoken/tests/fixtures/generate_hf_fixtures.py

Reference repos are ungated mirrors carrying the same tokenizer as the gated
originals; each is pinned with its expected vocab size, so a mirror that drifts
fails loudly instead of quietly becoming a bad oracle.

Texts containing an added token are skipped: HF always splits on added tokens
inside plain text, while this crate's `encode` treats them as ordinary text
(`encode_with_special_tokens` is the equivalent). That difference is by design,
not a divergence to pin.
"""

import json
import pathlib
import sys

from tokenizers import Tokenizer

OUT_DIR = pathlib.Path(__file__).resolve().parent

# encoding name -> (reference repo, expected base vocab size)
REFERENCES = {
    "llama3": ("unsloth/llama-3-8b", 128_000),
    "qwen2": ("Qwen/Qwen2.5-7B", 151_643),
    "deepseek_v3": ("deepseek-ai/DeepSeek-V3", 128_000),
    "deepseek_v4": ("deepseek-ai/DeepSeek-V4-Flash", 128_000),
    "mistral_v3": ("mistralai/Mistral-Nemo-Base-2407", 131_072),
    "glm4": ("zai-org/GLM-4.5", 151_329),
    "glm5": ("zai-org/GLM-5.2", 154_820),
    "minimax_m2": ("MiniMaxAI/MiniMax-M2", 200_000),
}

WHITESPACE = [
    " ", "  ", "   ", "\t", "\t\t",
    "\n", "\n\n", "\n\n\n", "\r", "\r\n", "\r\n\r\n",
    " \n", "\n ", " \n ", "  \n  ", "\t\n", "\n\t", " \r\n ", "\n \n",
    "\x0b", "\x0c", " ", " ", "　",
]

# Followers chosen to hit the axes where these patterns disagree with each other:
# digits and CJK are deepseek_v3 split-pipeline boundaries; `/` is the Tekken
# punctuation rule's `[\r\n/]*` tail; letters exercise the generic lookahead.
FOLLOWERS = ["", "a", "A", "1", "12", "!", "@", "/", "//", "/a", ".", "'s", " ", "\n", "你", "\U0001f389"]

PRECEDERS = ["", "word", "1", "!", ".", ")", "你好", "\U0001f389"]


def corpus():
    texts = [
        "", " ", "  ", "\n", "\t", "\r\n",
        "hello", "hello world", "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I'm", "don't", "they're", "we've", "she'll", "it'd",
        "O'Brien", "Y'all", "ALLCAPS'S",
        # case-splitting axis (o200k / Tekken split on case, cl100k does not)
        "CamelCase", "HTTPServer", "XMLHttpRequest", "ABCdef", "aBC", "ID", "IDs",
        "mixedScript123", "snake_case_name", "kebab-case-name",
        # digits (Tekken and qwen2 take one digit at a time)
        "123", "1234567890", "3.14159", "1,000,000", "2024", "a1", "1a", "/1", "1/",
        # CJK / unicode
        "你好", "你好世界", "こんにちは", "안녕하세요", "世界你好世界",
        "Hello 你好 World", "café résumé naïve über",
        "日本語テスト \U0001f389", "مرحبا بالعالم",
        "\U0001f389", "\U0001f680\U0001f4a1\U0001f3af", "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466",
        # code / paths — the `/` axis for Tekken
        'fn main() { println!("Hello"); }',
        "def f(n):\n    if n <= 1:\n        return n\n    return f(n-1)",
        "import os\n\nfrom a.b import c\n",
        "https://example.com/path?q=1&r=2",
        "a/b/c", "./rel/path", "../up/path", "/abs/path/file.rs",
        "// comment\n/* block */",
        "Hello!!! What?? Yes... No---maybe",
        # newline-run regression surface (issue #5)
        "word\n\nnext", "\r\n@rem",
        "@echo off\r\n@rem comment\r\n\r\nexit /b 0",
        "# Title\n\nPara one.\n\nPara two.\n\n- item\n- item\n",
        "a\n\nb\n\n\nc\n\n\n\nd",
        "line\r\n\r\nline\r\n\r\n\r\nline",
        "trailing\n", "trailing\n\n", "trailing\r\n", "trailing \n",
        "\n\nleading", "\r\n\r\nleading", " \n\nleading",
        'json:\n{\n  "a": 1,\n\n  "b": 2\n}\n',
        "CSV\r\n1,2,3\r\n4,5,6\r\n",
        # repetition / length boundaries
        "a" * 100, "hello " * 100, "The quick brown fox. " * 50, "你好世界！" * 50,
        "x", "x" * 10, "x" * 255, "x" * 256, "x" * 257,
    ]

    for p in PRECEDERS:
        for w in WHITESPACE:
            for f in FOLLOWERS:
                texts.append(p + w + f)

    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    texts = corpus()
    for name, (repo, want_vocab) in REFERENCES.items():
        tok = Tokenizer.from_pretrained(repo)
        got_vocab = tok.get_vocab_size(with_added_tokens=False)
        if got_vocab != want_vocab:
            raise SystemExit(
                f"{name}: {repo} has vocab {got_vocab}, expected {want_vocab} — "
                "the mirror drifted and is no longer a valid oracle"
            )

        added = {a.content for a in tok.get_added_tokens_decoder().values()}
        cases = []
        skipped = 0
        for text in texts:
            if any(a in text for a in added):
                skipped += 1
                continue
            tokens = tok.encode(text, add_special_tokens=False).ids
            cases.append({"text": text, "tokens": tokens, "count": len(tokens)})

        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(cases)} cases to {path} ({skipped} skipped)")


if __name__ == "__main__":
    main()
