#!/usr/bin/env python3
"""Generate a HuggingFace-reference corpus for the `canonical_parity` test.

Counterpart to `canonical_corpus.py`: that one covers the OpenAI encodings via
`openai/tiktoken`, this one covers the encodings sourced from HuggingFace
tokenizers. Emits the same shape — {encoding: [[text, [token, ...]], ...]}.

    pip install tokenizers
    python tiktoken/tests/hf_corpus.py /tmp/hf.json
    CANONICAL_JSON=/tmp/hf.json cargo test -p tiktoken \
        --test canonical_parity -- --ignored --nocapture

Reference repos are ungated mirrors with the same tokenizer as the gated
originals; each is pinned below with the vocab size to check against, since a
mirror that drifts would silently become a bad oracle.

Two known, intentional divergences are excluded from the emitted corpus:

  - HF always matches *added tokens* inside plain text, while this crate's
    `encode` treats them as ordinary text (`encode_with_special_tokens` is the
    equivalent). Texts containing an added token are therefore skipped.
"""

import json
import pathlib
import sys

from tokenizers import Tokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from canonical_corpus import corpus  # noqa: E402

# encoding name -> (reference repo, expected base vocab size)
REFERENCES = {
    "llama3": ("unsloth/llama-3-8b", 128_000),
    "qwen2": ("Qwen/Qwen2.5-7B", 151_643),
    "deepseek_v3": ("deepseek-ai/DeepSeek-V3", 128_000),
    "mistral_v3": ("mistralai/Mistral-Nemo-Base-2407", 131_072),
}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "hf.json"
    texts = corpus()
    data = {}

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
        for text in texts:
            # skip texts containing an added token: HF always splits on those,
            # this crate's `encode` does not (that is `encode_with_special_tokens`)
            if any(a in text for a in added):
                continue
            cases.append([text, tok.encode(text, add_special_tokens=False).ids])

        data[name] = cases
        print(f"{name}: {len(cases)} cases ({len(texts) - len(cases)} skipped)", file=sys.stderr)

    with open(out, "w") as f:
        json.dump(data, f)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
