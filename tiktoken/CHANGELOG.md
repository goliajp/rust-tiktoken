# Changelog

## [3.8.2] - 2026-08-09

### Docs
- npm-page fix, in lockstep: the wasm package's README carried a hand-copied
  "Supported Models" listing frozen at the 3.5-era catalogue (no gpt-5
  family, no Moonshot/Zhipu/MiniMax, no DeepSeek V4) — its second silent
  drift. Replaced with a per-provider summary (107 models / 10 providers)
  that names `allModels()` and docs.rs as the authority. Top-level
  quickstarts now pin `tiktoken = "3.8"`. No code changes.

## [3.8.1] - 2026-08-09

### Docs
- The crate READMEs (en/zh/ja) shipped in 3.8.0 still opened with the
  pre-3.8.0 headline (15–40x on ASCII, ≈2x on CJK) above the new tables —
  contradicting them on the crates.io page. Headlines, feature bullets and
  the zh/ja performance sections now carry the 3.8.0 numbers: 5–49x vs
  tiktoken-rs native, 2–4x vs gpt-tokenizer in-browser, CJK prose 15–17x.
  No code changes.

## [3.8.0] - 2026-08-09

### Fixed
- **o200k punctuation rule was missing the `/` in its `[\r\n/]*` tail.**
  Upstream o200k_base admits slashes after the newline tail of the punctuation
  rule, and the vocabulary leans on it — `".\n/"` is a single token (118550).
  This crate had cl100k's plain `[\r\n]*` tail, so such inputs split into two
  pieces and produced different ids. The fixture corpus never contained a
  slash after a newline, which is how a green suite proved nothing about this
  rule; the generators now include slash-after-newline shapes (2,374 → 2,728
  cases per OpenAI encoding, all fixtures regenerated from Python tiktoken,
  red-then-green verified). Kimi — which rode on the o200k fast path but whose
  upstream pattern has no slash — got its own scanner variant, now pinned by
  its own property tests.

### Changed — performance
A three-round, decomposition-driven attack on the Unicode path. Native
`encode` is now **5–49x faster than tiktoken-rs 0.9** (was 15–40x on ASCII
but ~2x on CJK); in the browser the wasm build is **2–4x faster than
gpt-tokenizer 3.4 on every corpus, CJK prose included** (it was 2–3x slower
there before). Apple M4 Mac mini, token outputs asserted identical before
timing; corpora and harnesses ship in-repo (`bench-compare/`, `web/bench/`).

- **Vocabulary lookup, layered by key size** — the BPE merge is lookup-bound
  (~2.4 probes per emitted token on CJK, 77% misses, 96.7% of keys ≤ 8 bytes).
  2-byte keys (the adjacent-pair scan, 59% of probes) now hit a direct-indexed
  65,536-entry table; 3–8-byte keys live in 16-byte slots with the key bytes
  inlined, so a probe is one memory access and one `u64` compare; longer keys
  keep the arena but gain an 8-bit tag that rejects mismatched slots without
  loading bytes.
- **CJK fast-path scanners** — Han, kana, hangul, fullwidth forms and CJK
  punctuation now resolve without the regex engine, including o200k's
  case-split rules with exact backtracking emulation for caseless letters and
  kimi's dedicated `[\p{Han}]+` branch. A certainty classifier drives the
  scanners; every claim it makes is pinned char-by-char against the regex
  crate's own Unicode tables, and anything unknown defers to the regex. All
  CJK handling lives out-of-line behind the existing non-ASCII branches, so
  the ASCII hot paths keep their shape.
- **Whole-piece memoisation** — a thread-local, direct-mapped, byte-keyed
  cache (4,096 slots, pieces ≤ 96 bytes) turns repeated pieces — chat
  templates, function words, CJK particles — into one hash and one compare.
  Keyed by a per-instance nonce, so encodings never share entries.
- **`bpe_count` no longer allocates** — the small-merge scratch is a
  caller-provided stack array.

### Verification
- 44,518 differential fixture cases across 16 encodings (encode + count per
  case), regenerated from the reference implementations: 0 divergences.
- Canonical full-corpus parity vs Python tiktoken 0.12: 66,546 cases across
  the 6 OpenAI encodings (now including slash-after-newline shapes): 0
  mismatches.
- 23 property tests × 20,000 random inputs per run hold every fast-path
  scanner to the regex's exact segmentation, including six new CJK-dense
  generators and slash-dense generators for o200k and kimi.


## [3.7.1] - 2026-08-08

### Changed
- Dependencies: `base64` `^0.22` → `^0.23`, `ruzstd` `^0.8` → `^0.9`
  (dependabot #10 / #8). Vocabulary decompression and parsing verified on the
  new versions: full oracle suites plus the reference differential corpora —
  167,849 comparisons across all 16 fixture-covered encodings, 0 divergences.
- CI: `actions/checkout` v6 → v7, `actions/setup-node` v6 → v7,
  `taiki-e/install-action` pinned to v2.85.4 (dependabot #4 / #9 / #7).

### Credit
- The 3.6.0 newline-splitting fix was first reported **and first patched** by
  @morluto (issue #5, PR #6). The shipped implementation arrived independently
  with the same core approach; the report and patch are what got it found —
  recording the credit here as it was missing from the 3.6.0 entry.


## [3.7.0] - 2026-08-08

### Added
- **Six new encodings for the 2026-08 model landscape**, with emphasis on the
  Chinese open-weights ecosystem — every one verified byte-exact against its
  reference tokenizer over a 10,489–10,491-case differential corpus
  (104,903 comparisons total, 0 divergences):
  - `kimi_k2` / `kimi_k3` (Moonshot Kimi K2 / K2.5 / K2.6 and K3): 163,584-token
    native-tiktoken vocabulary, byte-identical across both generations; the
    generations differ only in special-token tables (K2's `<|im_*|>` chat
    markers vs K3's `<|end_of_msg|>` / media tokens). The split pattern is an
    o200k variant with a dedicated `[\p{Han}]+` branch and Han-excluding
    character-class intersections.
  - `glm4` (Zhipu GLM-4.5 / 4.6 / 4.7, 151,329 tokens) and `glm5` (GLM-5 /
    5.2, 154,820 tokens): independently trained vocabularies sharing the
    cl100k split pattern and a common 36-entry special-token table.
  - `minimax_m2` (MiniMax M2 / M2.1 / M2.5 / M2.7): 200,000-token vocabulary,
    byte-identical across the family; o200k letter rules with a `[\r\n/]*`
    punctuation tail (caught by the differential — the tail admits `/`).
  - `deepseek_v4` (DeepSeek V4 Pro / Flash): shares V3's vocabulary, merges,
    and pattern; extends the added-token table from 818 to 1,283 entries
    (`<think>`, DSML markup, vision/grounding tags, 415 multimodal span
    placeholders).
- Model mapping for the new families: `kimi-k2*` / `kimi-k3*` / `kimi-latest`,
  `glm-4*` / `glm-5*`, `minimax*`, `deepseek-v4*` — plus the DeepSeek API
  aliases `deepseek-chat` and `deepseek-reasoner`, which point at V4 since
  2026-07-24.
- Pricing: 13 models across three new providers — Moonshot (`kimi-k3`,
  `kimi-k2.7-code`, `kimi-k2.6`, `kimi-k2.5`), Zhipu (`glm-5.2`, `glm-5`,
  `glm-4.7`, `glm-4.5`, `glm-4.5-air`), MiniMax (`minimax-m2.7` / `-m2.5` /
  `-m2.1` / `-m2`) — all first-party rates read from the providers' official
  price cards on 2026-08-08. The table now covers **107 models across 10
  providers**. Max-output limits are not published by these three providers;
  entries carry a conservative 32K placeholder.
- `tests/fixtures/generate_kimi_fixtures.py` (reference fixtures from
  Moonshot's native `tiktoken.model` + `tokenization_kimi.py` pat_str) and
  `src/encodings/convert_hf_vocab.py` (the HF ByteLevel-BPE → `.tiktoken.zst`
  converter with a built-in differential self-check — previously this
  conversion was done by unversioned scripts).

### Changed
- **Breaking (match exhaustiveness):** `pricing::Provider` gains `Moonshot`,
  `Zhipu`, and `MiniMax` variants.
- Embedded vocabulary data grows from ~5.9 MB to ~10.3 MB compressed
  (kimi 1.05 MB, glm4 0.93 MB, glm5 0.94 MB, minimax_m2 1.29 MB), which
  increases compiled artifact sizes accordingly (the wasm binary most
  noticeably).
- The published package now excludes `tests/` (several MB of oracle fixtures
  that only validate this crate against upstream tokenizers — they remain in
  the repository and CI) and the `examples/react-app/` demo. This keeps the
  package under crates.io's 10 MB cap, at 99.4% of it; adding another
  vocabulary will require splitting the data into a separate crate or a cap
  increase from crates.io — noted here so the next release plans for it.


## [3.6.0] - 2026-08-08

### Fixed
- **`cl100k_base` and `o200k_base` split canonical newline tokens**
  ([#5](https://github.com/goliajp/rust-tiktoken/issues/5)). The DFA-compatible
  pre-tokenizer applied its `\s+(?!\S)` lookahead emulation to every match,
  including matches produced by the patterns' dedicated `\s*[\r\n]+` branch.
  That branch carries no lookahead, so trimming it split `"\n\n"` and `"\r\n"`
  into separate single-newline pieces whenever non-whitespace followed —
  changing token ids and overcounting affected text. `encode("word\n\nnext")`
  returned `[1178, 198, 198, 3684]` instead of the canonical `[1178, 271, 3684]`.

  The emulation is now pattern-aware: encodings whose pattern has a newline
  branch ahead of the generic whitespace rules (`cl100k_base`, `o200k_base`,
  `o200k_harmony`, `llama3`, `mistral_v3`, `qwen2`, `deepseek_v3`) never trim a
  match ending in `\r`/`\n`, while `p50k_base` / `p50k_edit` / `r50k_base` keep
  the unconditional trim their patterns require.

  Verified against canonical `openai/tiktoken` 0.13.0 over a 10,491-case
  adversarial corpus: **2,504 divergences before, 0 after**, across all six
  OpenAI encodings (62,946 comparisons).

  **This changes token ids for affected text.** Any cached token counts or
  stored token id sequences for newline-containing text should be recomputed.

- **`mistral_v3` used the cl100k pattern instead of Tekken's.** The encoding was
  defined as `MISTRAL_V3_PATTERN = CL100K_PATTERN`, but Tekken's pre-tokenizer
  differs in three ways: it splits on case like o200k, it has no contraction
  rule at all, its number rule is `\p{N}` rather than `\p{N}{1,3}`, and its
  punctuation rule's trailing class is `[\r\n/]*` — admitting `/`. The first two
  differences happen to be masked by the Tekken vocabulary, but the `/` tail is
  not: any punctuation run followed by a newline and then a slash tokenized
  differently from the reference. `mistral_v3` now has its own pattern and a
  dedicated ASCII fast path.

- **`deepseek_v3` split whitespace runs before digits and CJK.** Upstream runs
  three sequential `Split` stages; this crate folds them into one alternation,
  which let the `\s+(?!\S)` lookahead peek past a stage boundary. A whitespace
  run followed by a digit or CJK character was trimmed when upstream would keep
  it whole (`"  1"` → two single-space tokens instead of one `"  "` token).

- **`model_to_encoding` resolved several legacy model ids incorrectly.** A flat
  prefix scan let a short prefix capture a more specific id: `davinci-codex`
  returned `r50k_base` (canonical: `p50k_base`) and `code-davinci-edit-001`
  returned `p50k_base` (canonical: `p50k_edit`). Ten first-generation search /
  similarity embedding models (`text-search-*`, `text-similarity-*`,
  `code-search-*`) returned `None`. Lookup is now exact-match-first, then
  prefix, mirroring upstream's registry structure — all 62 canonical entries now
  agree.

### Added
- **Missing added tokens.** `qwen2` gained the 8 ids upstream defines above
  `<|video_pad|>` (`<tool_call>`, `</tool_call>`, the four `<|fim_*|>` markers,
  `<|repo_name|>`, `<|file_sep|>` — ids 151657..=151664). `deepseek_v3` gained
  the 14 named tokens at 128800..=128814 — including `<｜User｜>` and
  `<｜Assistant｜>`, the primary chat-template markers — plus the 800
  `<｜place▁holder▁no▁N｜>` entries at 128000..=128799. Its doc comment claimed
  804 special tokens while the code registered 4;
  `encode_with_special_tokens` could not produce any of the missing ids.
- `gpt2` oracle fixture and test — the encoding was in the registry but had no
  fixture coverage.
- Pricing: 26 models across the 2026-08 landscape — OpenAI GPT-5.6
  (`sol` / `terra` / `luna`), the `-pro` SKUs and 5.1 / 5.2 point releases;
  Anthropic's Claude 5 generation (`fable-5`, `mythos-5`, `opus-5`, `sonnet-5`);
  `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-2.5-flash-lite`;
  `deepseek-v4-pro` / `-flash`; `qwen3.8-max` / `qwen3.5-plus`; Mistral's
  Devstral 2 and Ministral 3 SKUs. The table now covers **94 models**.

### Changed
- Pricing corrections: Claude 4.6+ context windows are 1M (those models carry
  the full window at standard rates); `mistral-small` updated to the Small 4
  rates ($0.15/$0.60). `deepseek-v3` / `deepseek-r1` and `gemini-2.0-flash` are
  marked DEPRECATED — they are off their vendors' current price cards.
- Oracle fixtures are now generated from the reference implementations, not from
  this crate's own output — `tests/fixtures/generate_openai_fixtures.py` for the
  OpenAI encodings (canonical `openai/tiktoken`) and
  `tests/fixtures/generate_hf_fixtures.py` for the HuggingFace-sourced ones. The
  previous snapshots were self-generated, so they could only prove the crate
  still agreed with itself, which is how issue #5 survived a green test suite.
  Corpora grew from ~90 to 2,374 cases (OpenAI) and from 41 to 2,815 cases (HF),
  with systematic whitespace-run × follower matrices covering the newline,
  digit, CJK and slash axes where these patterns disagree.

### Performance
- ASCII fast paths are **5–26% faster than 3.5.1** (same machine, same
  toolchain, criterion n=100): cl100k encode 13 B 45.7→37.9 ns, 45 KB
  77.8→67.3 µs; o200k 900 B 2.06→1.52 µs, 45 KB 92.9→79.2 µs. The scanners'
  per-pattern variations (digit cap, contraction rule, punctuation tail) are
  const-generic parameters with `#[inline(always)]`, so each `FastPath` gets
  fully specialized, inlined codegen. Unicode/CJK inputs are unchanged (regex
  path).

### Internal
- Removed `tests/generate_oracle.rs` (the self-generating oracle it replaced).
- Added `tests/canonical_parity.rs`, an `--ignored` full-corpus differential
  against the reference implementations, plus `tests/canonical_corpus.py` and
  `tests/hf_corpus.py` to generate its input. The 3.6.0 fixes were verified with
  it over 104,907 comparisons across all 10 encodings, 0 divergences.
- Added fast-path/regex equivalence proptests for the new Tekken scanner
  (including a slash-dense generator), newline-dense generators for the `cl100k`
  / `o200k` fast paths, and a `p50k` equivalence proptest.

## [3.5.1] - 2026-06-07

### Changed
- Dual-licensed under **MIT OR Apache-2.0** (previously MIT only). `LICENSE` is
  split into `LICENSE-MIT` + `LICENSE-APACHE`, and the crate `license` field is
  now `"MIT OR Apache-2.0"` — the conventional permissive dual-license for Rust
  crates.

### Internal
- rustfmt + clippy (`-D warnings`) cleanup of the 3.5.0 fast-path code — no
  behavior change. Workspace README rewritten; `LICENSE-MIT` copyright year
  unified to 2023–2026.

## [3.5.0] - 2026-06-07

> Version note: `tiktoken` and `tiktoken-wasm` are now kept in lockstep at the
> same version number (both jump to 3.5.0 here), so a given release tag means the
> same code across both packages.

### Added
- ASCII fast-path pre-tokenizers for the `cl100k`, `o200k`, `qwen2`, and
  `deepseek_v3` patterns (also covering `llama3` / `mistral_v3`, which share the
  cl100k pattern). These hand-written scanners resolve the common ASCII pieces
  (letters, digits, punctuation, contractions) without invoking the regex
  engine, deferring to it only on non-ASCII bytes or whitespace runs. The
  scanner is selected per-encoding via an internal `FastPath` injected from
  `encoding.rs`, so the pre-tokenizer stays unaware of any specific pattern
  string. Verified byte-for-byte identical to the regex by per-pattern
  equivalence proptests (20k cases each).
- Pricing entries: `claude-opus-4.8`, `claude-opus-4.7`,
  `gemini-3.1-pro-preview`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`
  (verified against official docs; Claude/Gemini remain pricing-only — neither
  vendor publishes a tokenizer, so they have no encoding).

### Changed
- BPE merge is now hybrid: pieces up to 32 bytes use an allocation-free,
  stack-based linear scan; longer pieces keep the heap-accelerated O(n log n)
  algorithm. After pre-tokenization most pieces are word-sized, where the heap's
  allocation + bookkeeping overhead dominated.
- `encode_parallel` rewritten as a chunked two-pass (one buffer per worker
  instead of one `Vec` per piece) and its serial-fallback threshold raised from
  4 KB to 512 KB: with the fast-path making serial encoding cheap and
  pre-tokenization staying serial (Amdahl), the parallel path only wins above
  ~460 KB.

### Performance
- ASCII-heavy `encode` / `count` is **2.3–5.5x faster** across cl100k / o200k /
  qwen2 / deepseek (e.g. cl100k encode of 45 KB English: ~350 µs → ~70 µs on an
  Apple M4 Mac mini). Unicode/CJK text is unchanged — it defers to the regex,
  whose Unicode-property matching was never the bottleneck the fast-path targets.

## [3.3.0] - 2026-06-07

### Added
- `gpt2` registered as the 11th encoding name (alias for `r50k_base`).
  `model_to_encoding` resolves `gpt2` / `gpt-2` model prefixes to
  `"r50k_base"`; `get_encoding("gpt2")` shares r50k_base's cache slot.
- Six OpenAI GPT-5.x model entries with full available tier data:
  `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (Standard only), `gpt-5.4`
  (Standard + Batch + Flex + long-context >272K),
  `gpt-5.4-mini` (Standard + Batch + Flex),
  `gpt-5.5` (Standard + Batch + Flex + Priority + long-context).
- `pricing::TierRates { input_per_1m, cached_input_per_1m: Option<f64>,
  output_per_1m }` — unified rate type shared by all OpenAI service tiers.
- `ExtendedPricing.flex` and `.priority`, both `Option<TierRates>`.
- `Model::with_batch_cached`, `with_flex`, `with_priority` const builders.
- `Model::estimate_flex_cost` and `estimate_priority_cost`.

### Changed
- **Breaking** — `pricing::VisionPricing` is now a provider-specific enum
  (`OpenAITileBased` / `OpenAIPatchBased` / `AnthropicDivisor` /
  `GeminiTileBased`) instead of the 3.2 placeholder
  `VisionPricing { per_image: f64 }`. Image inputs are billed at the
  model's standard `input_per_1m` rate, not a flat per-image fee, so the
  enum captures each provider's published image→tokens formula
  (`VisionPricing::image_tokens(width, height, detail)`) and
  `Model::estimate_image_cost(width, height, detail)` returns the
  end-to-end USD figure (auto-applies high-tier rates above the
  gemini-2.5-pro 200k threshold). The unused 3.2 builder
  `with_vision_per_image` is removed; use `with_vision(VisionPricing)`.
  Vision data populated for gpt-4o / gpt-4o-mini / gpt-4.1 / gpt-4.1-mini /
  gpt-4.1-nano / o1 / o3 / o4-mini, claude-haiku-4.5, claude-sonnet-4.5 /
  4.6, claude-opus-4.5 / 4.6, gemini-2.5-pro / 2.5-flash.
- `pricing::BatchPricing` is now a type alias for `TierRates` (was a
  distinct struct in 3.2.x). Existing `with_batch(input, output)` calls
  still compile; the `cached_input_per_1m` field defaults to `None`.
- README files across all 3 languages × 3 levels bump encoding count
  `9 → 11` and model count `57 → 63`; added `o200k_harmony` and `gpt2`
  rows to the Supported Encodings tables.

## [3.2.0] - 2026-06-06

### Added
- `o200k_harmony` encoding (10th encoding) for OpenAI gpt-oss models /
  harmony chat format. Shares o200k_base merge ranks and regex pattern;
  only the special-token table differs (15 named + 1075 reserved
  placeholders). `model_to_encoding` gains a `gpt-oss` prefix that
  routes to it.
- Pricing schema extensions (`pricing::BatchPricing`,
  `pricing::HighTierPricing`, `pricing::AudioPricing`,
  `pricing::VisionPricing`, grouped under `pricing::ExtendedPricing` on
  `Model`). Optional dimensions for batch-API discounts, Google's
  input-token-count-based dual-tier rates, and per-modality pricing.
- `Model::pricing_for_input(input_tokens)` returns the tier-appropriate
  `Pricing` (auto-switches when `extended.high_tier` is set and input
  exceeds its threshold).
- `Model::estimate_batch_cost(input, output)` returns `Some(cost)` when
  the model has batch pricing, else `None`.
- `const fn` builder methods on `Model`: `with_batch`, `with_high_tier`,
  `with_audio_input`, `with_vision_per_image`.
- Data filled (only where verified by 2026-06 research): batch pricing
  for 13 OpenAI + 5 active Anthropic models; `gemini-2.5-pro` high-tier
  (>200k input) rates; `gemini-2.5-flash` audio input rate $1.00/M.

### Changed
- `pricing::Model` is now `#[non_exhaustive]`. Reading its fields is
  unchanged, but constructing it via a struct literal outside the crate
  is no longer allowed (use the internal `model()` + builder pattern
  instead).
- `estimate_cost` and `estimate_cost_with_cache` now route through
  `pricing_for_input`, so they auto-pick high-tier rates for
  `gemini-2.5-pro` above the 200k input threshold.
- Meta llama prices re-sourced per-model: pinned to DeepInfra
  (`llama-3.1-8b`, `llama-3.3-70b`, `llama-4-maverick`) and Groq
  (`llama-4-scout`) with source URLs in each entry's doc comment.
  `llama-3.1-405b` and `llama-3.1-70b` marked `DEPRECATED` —
  no major hoster offers them as serverless inference any longer.

## [3.1.5] - 2026-06-06

### Added
- `model_to_encoding` now supports `gpt-5` and `gpt-4.5` prefixes → `o200k_base`.
- `model_to_encoding` strips a leading `ft:` prefix so fine-tuned model IDs (e.g.
  `ft:gpt-4o:my-org::abc123`) resolve to the base model's encoding.
- Azure-style `gpt-35-turbo` alias → `cl100k_base`.

### Fixed
- `davinci-002` and `babbage-002` now correctly route to `cl100k_base` (they were
  greedily caught by the `davinci`/`babbage` prefix in the `r50k_base` block).

### Changed
- Pricing refreshed against official 2026-06 docs across all 7 providers:
  - 13 price corrections (notably OpenAI `o1-mini` to match `o3-mini` at $1.10/$4.40,
    Anthropic `claude-opus-4`/`claude-sonnet-4` cache values to cache-read convention,
    Alibaba Qwen 2.5 family now uses official Model Studio split rates, Mistral
    `mistral-large` cut to $0.5/$1.5, `mistral-medium` repositioned to $1.5/$7.5).
  - 21 `DEPRECATED` doc-comments (OpenAI shutdown 2026-10-23, Anthropic Claude 3.x
    retired, claude-opus-4/sonnet-4 retire 2026-06-15, Gemini 2.0 Flash shutdown,
    Gemini 1.5 family removed from official pricing, DeepSeek `deepseek-chat`/
    `deepseek-reasoner` deprecate 2026-07-24, Mistral `pixtral-large`).
  - Module-level caveats document the Anthropic cache-read convention, the Meta
    llama legacy-Together source, and the `gemini-2.5-pro` single-tier limitation.
- Dropped `rust-version` MSRV pin from workspace; tracks current stable.
- CI release workflow: Node 20 → 24, `wasm-pack` install switched to
  `taiki-e/install-action` (prebuilt binary, ~5s vs ~5–10 min compile).

## [3.1.4] - 2026-04-24

### Changed
- Smoke-test release via the new repo's GitHub Actions publish workflow.
  No code changes.

## [3.1.3] - 2026-04-24

### Changed
- Migrated from `goliajp/airs` mono-repo to standalone `goliajp/rust-tiktoken` (shares repo with `tiktoken-wasm`).
  No code changes; `repository` URL updated.

## [3.0.1] - 2026-03-07

### Changed

- Updated benchmark tables in README to v3.0 numbers (all three languages)

## [3.0.0] - 2026-03-07

### Added

- **Multi-provider tokenizer support**: Llama 3 (`llama3`), DeepSeek V3 (`deepseek_v3`), Qwen 2 (`qwen2`), Mistral V3 (`mistral_v3`) encodings
- **Parallel encoding**: `encode_parallel()` method behind `parallel` feature flag, uses rayon for texts >= 4KB
- **count_with_special_tokens()**: counting method that recognizes special tokens, matching `encode_with_special_tokens()` behavior
- **Multi-provider pricing**: Meta Llama, DeepSeek, Alibaba Qwen, Mistral models added to pricing module (39 total models across 7 providers)
- HuggingFace oracle tests: cross-validation against Python HF tokenizer output for all new encodings
- Property-based testing: 125,000 random input roundtrip tests via proptest
- Third-party license documentation (`LICENSE-3RD-PARTY`)
- Vocab conversion scripts (`scripts/convert_vocab.py`, `scripts/generate_hf_oracle.py`)

### Changed

- **Breaking**: internal architecture rewritten for performance
  - `FxHashMap<Vec<u8>, u32>` replaced with arena-based `Vocab` (single allocation, cache-friendly)
  - BPE merge algorithm: O(n*m) linear scan replaced with O(n log n) heap-accelerated merge (BinaryHeap + doubly-linked list)
  - Pre-tokenization abstracted behind `PreTokenizer` trait (`RegexPreTokenizer` implementation)
- Vocabulary data now zstd-compressed at rest (~63% compression ratio, 7MB -> 2.6MB for OpenAI vocabs)
- Internal modules (`vocab`, `merge`, `pretokenize`) use `pub(crate)` visibility
- `model_to_encoding()` now supports Llama, DeepSeek, Qwen, and Mistral model name prefixes

### Fixed

- NBSP overflow bug in whitespace lookahead emulation: single multi-byte whitespace characters (e.g. U+00A0) no longer cause empty piece underflow
- DeepSeek ZWJ character handling: format characters (Unicode Cf category) no longer skipped by regex

### Performance

- Arena-based vocabulary: single contiguous allocation replaces 200k individual `Vec<u8>` heap allocations
- Heap-accelerated BPE merge: O(n log n) vs O(n*m) for the ~5% of pieces that enter the merge path
- Fast paths for 1-byte and 2-byte pieces in BPE merge

## [2.1.1] - 2026-03-07

### Changed

- Upgraded Rust edition 2021 -> 2024, MSRV 1.85 -> 1.94
- Upgraded `criterion` dev-dependency 0.5 -> 0.8
- Migrated Cargo.toml metadata to workspace inheritance (edition, rust-version, license, repository, homepage, authors)
- Moved to `airs` monorepo workspace

## [2.1.0] - 2026-03-06

### Performance

- Rank-cached BPE merge: only recomputes 2 neighbor ranks per merge step
- 2-byte piece fast path in `byte_pair_merge`
- Unicode encode/count improved ~52%, ASCII text improved 2-10%

### Added

- `decode_to_string()` method for ergonomic UTF-8 decoding
- `model_to_encoding()` as public API
- `o200k_base` encoding (GPT-4o, o1, o3, o4-mini)
- Multi-provider pricing module (`tiktoken::pricing`) -- OpenAI, Anthropic Claude, Google Gemini (26 models)
- WebAssembly bindings (`tiktoken-wasm`) with encode/decode/count/pricing
- React demo app (`examples/react-app`)
- Criterion benchmarks with Python tiktoken 0.12.0 comparison
- Trilingual documentation (English, simplified Chinese, Japanese)
- 93 tests (82 unit + 11 doc tests), 97% line coverage

### Changed

- **Breaking**: rewritten from scratch -- new engine, new dependencies, new API surface
- Replaced `pcre2` (backtracking) with `regex` crate (DFA engine)
- Replaced `HashMap` with `FxHashMap` for faster small-key hashing
- Replaced `lazy_static` with `OnceLock` for encoding instance caching
- Upgraded `base64` 0.21 -> 0.22
- Upgraded edition 2021 -> 2024, MSRV set to 1.85
- Removed dependencies: `anyhow`, `maplit`, `rust_decimal`, `rust_decimal_macros`

### Removed

- `src/models.rs` and `src/price.rs` (replaced by `src/pricing.rs`)

## [1.0.1] - 2023-08-18

- Initial release on crates.io
