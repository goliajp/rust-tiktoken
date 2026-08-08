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

import base64
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
    "deepseek_v4": ("deepseek-ai/DeepSeek-V4-Flash", 128_000),
    "mistral_v3": ("mistralai/Mistral-Nemo-Base-2407", 131_072),
    "glm4": ("zai-org/GLM-4.5", 151_329),
    "glm5": ("zai-org/GLM-5.2", 154_820),
    "minimax_m2": ("MiniMaxAI/MiniMax-M2", 200_000),
}


def kimi_cases(texts):
    """Kimi reference: Moonshot ships a native tiktoken vocab, not an HF
    tokenizer.json — build the reference Encoding exactly as their
    tokenization_kimi.py does. Serves kimi_k2 and kimi_k3 (shared vocab)."""
    import tiktoken as pytiktoken
    from huggingface_hub import hf_hub_download

    pat = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )
    path = hf_hub_download("moonshotai/Kimi-K3", "tiktoken.model")
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    if len(ranks) != 163_584:
        raise SystemExit(f"kimi vocab {len(ranks)} != 163584 — reference drifted")
    enc = pytiktoken.Encoding(name="kimi", pat_str=pat, mergeable_ranks=ranks, special_tokens={})
    return [[t, enc.encode(t, disallowed_special=())] for t in texts]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "hf.json"
    texts = corpus()
    data = {}

    cases = kimi_cases(texts)
    data["kimi_k2"] = cases
    data["kimi_k3"] = cases
    print(f"kimi_k2 / kimi_k3: {len(cases)} cases", file=sys.stderr)

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
