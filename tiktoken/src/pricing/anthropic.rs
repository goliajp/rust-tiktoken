//! Anthropic Claude price table.

use super::*;

// Source: platform.claude.com/docs/en/about-claude/pricing and models/overview,
// 2026-11. Claude 4.6 and later (except Haiku 5.5) carry the full 1M-token
// context window at standard rates, so there is no long-context tier to encode.
//
// `cached_input` is the cache-READ price. It is 0.1× base input except where an
// entry says otherwise (0.025× on Fable / Mythos 5.1 and 5.2, 0.05× on Opus and
// Sonnet 5.5). Cache writes are listed per lifetime (5 minutes, 1 hour); Haiku
// 5.5's long-prompt write rates ($0.625 / $1) are not modelled. Fast mode is
// stored as `priority` (Opus 4.8's is withdrawn on 2026-11-30); cached input
// in Fast mode is unpublished.
//
// Tokenizer note: Claude 4.7 and later use a newer tokenizer that produces
// roughly 30% more tokens for the same text than Claude 4.6 and earlier. This
// crate ships no Anthropic tokenizer, so token counts for cost estimation must
// come from Anthropic's count-tokens endpoint, not from a tiktoken encoding.

/// Claude 4.7 and later: images up to a 2576 px long edge and 4784 tokens.
const VISION_HIGH_RES: VisionPricing = VisionPricing::AnthropicPatch {
    max_edge_px: 2576,
    max_tokens: 4784,
};

/// Earlier models: images up to a 1568 px long edge and 1568 tokens.
const VISION_STANDARD: VisionPricing = VisionPricing::AnthropicPatch {
    max_edge_px: 1568,
    max_tokens: 1568,
};

/// claude-fable-5.2 — frontier tier (2026-11-09). Cache reads are 0.025× input.
const CLAUDE_FABLE_52: Model = model(
    "claude-fable-5.2",
    Provider::Anthropic,
    8.00,
    40.00,
    Some(0.20),
    1_000_000,
    128_000,
)
.with_batch(4.00, 20.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 10.00), CacheWrite::new(60, 16.00)])
.with_priority(16.00, None, 80.00);

/// claude-mythos-5.2 — frontier tier, limited availability. Same rates as Fable 5.2.
const CLAUDE_MYTHOS_52: Model = model(
    "claude-mythos-5.2",
    Provider::Anthropic,
    8.00,
    40.00,
    Some(0.20),
    1_000_000,
    128_000,
)
.with_batch(4.00, 20.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 10.00), CacheWrite::new(60, 16.00)])
.with_priority(16.00, None, 80.00);

/// claude-fable-5.1 — frontier tier (2026-09-01). Cache reads are 0.025× input.
const CLAUDE_FABLE_51: Model = model(
    "claude-fable-5.1",
    Provider::Anthropic,
    10.00,
    50.00,
    Some(0.25),
    1_000_000,
    128_000,
)
.with_batch(5.00, 25.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 12.50), CacheWrite::new(60, 20.00)]);

/// claude-mythos-5.1 — frontier tier, limited availability. Same rates as Fable 5.1.
const CLAUDE_MYTHOS_51: Model = model(
    "claude-mythos-5.1",
    Provider::Anthropic,
    10.00,
    50.00,
    Some(0.25),
    1_000_000,
    128_000,
)
.with_batch(5.00, 25.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 12.50), CacheWrite::new(60, 20.00)]);

/// claude-fable-5 — frontier tier.
const CLAUDE_FABLE_5: Model = model(
    "claude-fable-5",
    Provider::Anthropic,
    10.00,
    50.00,
    Some(1.00),
    1_000_000,
    128_000,
)
.with_batch(5.00, 25.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 12.50), CacheWrite::new(60, 20.00)]);

/// claude-mythos-5 — frontier tier, limited availability. Same rates as Fable 5.
const CLAUDE_MYTHOS_5: Model = model(
    "claude-mythos-5",
    Provider::Anthropic,
    10.00,
    50.00,
    Some(1.00),
    1_000_000,
    128_000,
)
.with_batch(5.00, 25.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 12.50), CacheWrite::new(60, 20.00)]);

/// claude-opus-5.5 — flagship (2026-09-22). Cache reads are 0.05× input.
const CLAUDE_OPUS_55: Model = model(
    "claude-opus-5.5",
    Provider::Anthropic,
    4.00,
    20.00,
    Some(0.20),
    1_000_000,
    128_000,
)
.with_batch(2.00, 10.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 5.00), CacheWrite::new(60, 8.00)])
.with_priority(8.00, None, 40.00);

/// claude-opus-5 — flagship of the first Claude 5 release.
const CLAUDE_OPUS_5: Model = model(
    "claude-opus-5",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 6.25), CacheWrite::new(60, 10.00)])
.with_priority(10.00, None, 50.00);

/// claude-sonnet-5.5 — balanced tier (2026-09-28). Cache reads are 0.05× input
/// (lowered from 0.1× on 2026-10-07).
const CLAUDE_SONNET_55: Model = model(
    "claude-sonnet-5.5",
    Provider::Anthropic,
    2.00,
    10.00,
    Some(0.10),
    1_000_000,
    128_000,
)
.with_batch(1.00, 5.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 2.50), CacheWrite::new(60, 4.00)]);

/// claude-sonnet-5 — balanced tier. Launched with $2/$10 as introductory pricing;
/// Anthropic made it the standard price on 2026-08-10, cancelling the
/// scheduled rise to $3/$15.
const CLAUDE_SONNET_5: Model = model(
    "claude-sonnet-5",
    Provider::Anthropic,
    2.00,
    10.00,
    Some(0.20),
    1_000_000,
    128_000,
)
.with_batch(1.00, 5.00)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 2.50), CacheWrite::new(60, 4.00)]);

/// claude-haiku-5.5 — fast tier (2026-10-07). Priced by prompt length: a
/// request whose input (cache reads and writes included) is over 100,000
/// tokens pays $0.50 / $2.50, cache reads $0.05, for the whole request. The
/// long-prompt batch rates ($0.25 / $1.25) are not modelled.
const CLAUDE_HAIKU_55: Model = model(
    "claude-haiku-5.5",
    Provider::Anthropic,
    0.10,
    0.50,
    Some(0.01),
    1_000_000,
    128_000,
)
.with_batch(0.05, 0.25)
.with_high_tier(0.50, 2.50, Some(0.05), 100_000)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 0.125), CacheWrite::new(60, 0.20)]);

const CLAUDE_OPUS_48: Model = model(
    "claude-opus-4.8",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 6.25), CacheWrite::new(60, 10.00)])
.with_priority(10.00, None, 50.00);

const CLAUDE_OPUS_47: Model = model(
    "claude-opus-4.7",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VISION_HIGH_RES)
.with_cache_writes(&[CacheWrite::new(5, 6.25), CacheWrite::new(60, 10.00)]);

const CLAUDE_OPUS_46: Model = model(
    "claude-opus-4.6",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VISION_STANDARD)
.with_cache_writes(&[CacheWrite::new(5, 6.25), CacheWrite::new(60, 10.00)]);

const CLAUDE_SONNET_46: Model = model(
    "claude-sonnet-4.6",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    1_000_000,
    64_000,
)
.with_batch(1.50, 7.50)
.with_vision(VISION_STANDARD)
.with_cache_writes(&[CacheWrite::new(5, 3.75), CacheWrite::new(60, 6.00)]);

const CLAUDE_HAIKU_45: Model = model(
    "claude-haiku-4.5",
    Provider::Anthropic,
    1.00,
    5.00,
    Some(0.10),
    200_000,
    64_000,
)
.with_batch(0.50, 2.50)
.with_vision(VISION_STANDARD)
.with_cache_writes(&[CacheWrite::new(5, 1.25), CacheWrite::new(60, 2.00)]);

const CLAUDE_OPUS_45: Model = model(
    "claude-opus-4.5",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    200_000,
    64_000,
)
.with_batch(2.50, 12.50)
.with_vision(VISION_STANDARD)
.with_cache_writes(&[CacheWrite::new(5, 6.25), CacheWrite::new(60, 10.00)]);

/// DEPRECATED — deprecated 2026-09-30, retires 2026-11-30. Replaced by
/// claude-sonnet-5.5.
const CLAUDE_SONNET_45: Model = model(
    "claude-sonnet-4.5",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    200_000,
    64_000,
)
.with_batch(1.50, 7.50)
.with_vision(VISION_STANDARD)
.with_cache_writes(&[CacheWrite::new(5, 3.75), CacheWrite::new(60, 6.00)]);

/// DEPRECATED — retired 2026-06-15 on the Claude API. Replaced by
/// claude-opus-4.5 and later. Cache price corrected 2026-06:
/// was $7.50 (cache-write/2 convention), now $1.50 (cache-read = input × 10%).
const CLAUDE_OPUS_4: Model = model(
    "claude-opus-4",
    Provider::Anthropic,
    15.00,
    75.00,
    Some(1.50),
    200_000,
    32_000,
)
.with_cache_writes(&[CacheWrite::new(5, 18.75), CacheWrite::new(60, 30.00)]);

/// DEPRECATED — retired 2026-06-15 on the Claude API. Replaced by
/// claude-sonnet-4.6 and later. Cache price corrected 2026-06:
/// was $1.50 (cache-write/2 convention), now $0.30 (cache-read = input × 10%).
const CLAUDE_SONNET_4: Model = model(
    "claude-sonnet-4",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    200_000,
    64_000,
)
.with_cache_writes(&[CacheWrite::new(5, 3.75), CacheWrite::new(60, 6.00)]);

/// DEPRECATED — retired 2026-02-19 on Anthropic-operated API
/// (still available on AWS Bedrock and Google Vertex AI).
const CLAUDE_HAIKU_35: Model = model(
    "claude-3.5-haiku",
    Provider::Anthropic,
    0.80,
    4.00,
    Some(0.40),
    200_000,
    8_192,
);

/// DEPRECATED — retired 2025-10-28.
const CLAUDE_SONNET_35: Model = model(
    "claude-3.5-sonnet",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(1.50),
    200_000,
    8_192,
);

/// DEPRECATED — retired 2026-01-05.
const CLAUDE_OPUS_3: Model = model(
    "claude-3-opus",
    Provider::Anthropic,
    15.00,
    75.00,
    Some(7.50),
    200_000,
    4_096,
);

/// DEPRECATED — retired 2026-04-20.
const CLAUDE_HAIKU_3: Model = model(
    "claude-3-haiku",
    Provider::Anthropic,
    0.25,
    1.25,
    Some(0.03),
    200_000,
    4_096,
);

pub(super) const MODELS: &[Model] = &[
    CLAUDE_FABLE_52,
    CLAUDE_MYTHOS_52,
    CLAUDE_FABLE_51,
    CLAUDE_MYTHOS_51,
    CLAUDE_FABLE_5,
    CLAUDE_MYTHOS_5,
    CLAUDE_OPUS_55,
    CLAUDE_OPUS_5,
    CLAUDE_SONNET_55,
    CLAUDE_SONNET_5,
    CLAUDE_HAIKU_55,
    CLAUDE_OPUS_48,
    CLAUDE_OPUS_47,
    CLAUDE_OPUS_46,
    CLAUDE_SONNET_46,
    CLAUDE_HAIKU_45,
    CLAUDE_OPUS_45,
    CLAUDE_SONNET_45,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    CLAUDE_HAIKU_35,
    CLAUDE_SONNET_35,
    CLAUDE_OPUS_3,
    CLAUDE_HAIKU_3,
];

#[cfg(test)]
mod tests {
    use crate::pricing::*;

    #[test]
    fn claude_2026_11_lineup_prices() {
        for (id, input, output, cached) in [
            ("claude-fable-5.2", 8.0, 40.0, 0.20),
            ("claude-fable-5.1", 10.0, 50.0, 0.25),
            ("claude-opus-5.5", 4.0, 20.0, 0.20),
            ("claude-sonnet-5.5", 2.0, 10.0, 0.10),
            ("claude-haiku-5.5", 0.10, 0.50, 0.01),
            ("claude-sonnet-5", 2.0, 10.0, 0.20),
        ] {
            let m = get_model(id).unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(m.pricing.input_per_1m, input, "{id}");
            assert_eq!(m.pricing.output_per_1m, output, "{id}");
            assert_eq!(m.pricing.cached_input_per_1m, Some(cached), "{id}");
        }
    }

    #[test]
    fn claude_api_ids_resolve_to_dotted_entries() {
        assert_eq!(
            resolve_model("claude-opus-5-5").unwrap().model.id,
            "claude-opus-5.5"
        );
        assert_eq!(
            resolve_model("claude-fable-5-2").unwrap().model.id,
            "claude-fable-5.2"
        );
        assert_eq!(
            resolve_model("claude-haiku-5-5").unwrap().model.id,
            "claude-haiku-5.5"
        );
    }

    #[test]
    fn haiku_5_5_switches_rates_above_100k_prompt_tokens() {
        let m = get_model("claude-haiku-5.5").unwrap();
        // exactly 100,000 is still the short-prompt rate
        let short = m.estimate_cost(100_000, 1_000_000);
        assert!((short - (0.01 + 0.50)).abs() < 1e-9, "{short}");
        let long = m.estimate_cost(100_001, 1_000_000);
        assert!(
            (long - (100_001.0 * 0.50 / 1e6 + 2.50)).abs() < 1e-9,
            "{long}"
        );
    }
}
