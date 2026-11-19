//! OpenAI price table.

use super::openai_legacy::*;
use super::*;

// GPT-5.x family (2026 generation). Standard rates from per-model docs pages;
// Batch / Flex / Priority rates from the aggregate /api/docs/pricing toggle.
// Long-context (>272K input) policy: Standard/Batch/Flex inputs multiply by 2×
// and outputs by 1.5× for gpt-5.4 and gpt-5.5 — encoded via `with_high_tier`.

// GPT-6 family (2026-09 to 2026-11). Source: developers.openai.com/api/docs/
// pricing and per-model pages, 2026-11. Every SKU has a 1.05M context window and
// 128K max output; above 272K input the whole request bills at the long-context
// rates (input and cache 2×, output 1.5×). Astra is a new tier above Sol.
// OpenAI renamed the Priority tier "Fast" on 2026-07-30; it is stored as
// `priority`. The 6.2 SKUs have no Flex tier.

/// gpt-6.2-astra — top tier (2026-11-17).
const OPENAI_GPT62_ASTRA: Model = model(
    "gpt-6.2-astra",
    Provider::OpenAI,
    8.00,
    40.00,
    Some(0.40),
    1_050_000,
    128_000,
)
.with_batch_cached(4.00, 0.20, 20.00)
.with_priority(16.00, Some(0.80), 80.00)
.with_high_tier(16.00, 60.00, Some(0.80), 272_000);

/// gpt-6.2-sol — frontier tier (2026-11-17).
const OPENAI_GPT62_SOL: Model = model(
    "gpt-6.2-sol",
    Provider::OpenAI,
    2.00,
    10.00,
    Some(0.10),
    1_050_000,
    128_000,
)
.with_batch_cached(1.00, 0.05, 5.00)
.with_priority(4.00, Some(0.20), 20.00)
.with_high_tier(4.00, 15.00, Some(0.20), 272_000);

/// gpt-6.1-terra — balanced tier (2026-10-26).
const OPENAI_GPT61_TERRA: Model = model(
    "gpt-6.1-terra",
    Provider::OpenAI,
    1.00,
    5.00,
    Some(0.05),
    1_050_000,
    128_000,
)
.with_batch_cached(0.50, 0.025, 2.50)
.with_flex(0.50, Some(0.025), 2.50)
.with_priority(2.00, Some(0.10), 10.00)
.with_high_tier(2.00, 7.50, Some(0.10), 272_000);

/// gpt-6.1-sol — frontier tier (2026-09-29).
const OPENAI_GPT61_SOL: Model = model(
    "gpt-6.1-sol",
    Provider::OpenAI,
    2.00,
    10.00,
    Some(0.10),
    1_050_000,
    128_000,
)
.with_batch_cached(1.00, 0.05, 5.00)
.with_flex(1.00, Some(0.05), 5.00)
.with_priority(4.00, Some(0.20), 20.00)
.with_high_tier(4.00, 15.00, Some(0.20), 272_000);

/// gpt-6-astra — top tier (2026-09-03).
const OPENAI_GPT6_ASTRA: Model = model(
    "gpt-6-astra",
    Provider::OpenAI,
    10.00,
    50.00,
    Some(1.00),
    1_050_000,
    128_000,
)
.with_batch_cached(5.00, 0.50, 25.00)
.with_flex(5.00, Some(0.50), 25.00)
.with_priority(20.00, Some(2.00), 100.00)
.with_high_tier(20.00, 75.00, Some(2.00), 272_000);

/// gpt-6-sol — frontier tier (2026-09-22).
const OPENAI_GPT6_SOL: Model = model(
    "gpt-6-sol",
    Provider::OpenAI,
    2.00,
    10.00,
    Some(0.20),
    1_050_000,
    128_000,
)
.with_batch_cached(1.00, 0.10, 5.00)
.with_flex(1.00, Some(0.10), 5.00)
.with_priority(4.00, Some(0.40), 20.00)
.with_high_tier(4.00, 15.00, Some(0.40), 272_000);

/// gpt-6-luna — high-volume tier (2026-09-22).
const OPENAI_GPT6_LUNA: Model = model(
    "gpt-6-luna",
    Provider::OpenAI,
    0.10,
    0.50,
    Some(0.01),
    1_050_000,
    128_000,
)
.with_batch_cached(0.05, 0.005, 0.25)
.with_flex(0.05, Some(0.005), 0.25)
.with_priority(0.20, Some(0.02), 1.00)
.with_high_tier(0.20, 0.75, Some(0.02), 272_000);

// GPT-5.6 family (2026-07-09). The generation number is the family; Sol / Terra
// / Luna are durable capability tiers replacing the base / mini / nano names.
// Terra and Luna were repriced downward on 2026-07-30 (-20% / -80%). Flex rates
// equal Batch. Long-context (>272K input) rates are per the docs pricing page.

/// gpt-5.6-sol — frontier tier: complex reasoning, long-horizon agentic work.
///
/// Holds the regular rates. A promotion bills $4 / $20 (cached $0.40, batch
/// $2 / $10, long context $8 / $30) until 2026-11-30 per the pricing page;
/// OpenAI has not said what applies after it, so the launch rates stay here.
const OPENAI_GPT56_SOL: Model = model(
    "gpt-5.6-sol",
    Provider::OpenAI,
    5.00,
    30.00,
    Some(0.50),
    1_050_000,
    128_000,
)
.with_batch_cached(2.50, 0.25, 15.00)
.with_high_tier(10.00, 45.00, None, 272_000);

/// gpt-5.6-terra — balanced tier: everyday coding, reasoning, agentic tasks.
const OPENAI_GPT56_TERRA: Model = model(
    "gpt-5.6-terra",
    Provider::OpenAI,
    2.00,
    12.00,
    Some(0.20),
    1_050_000,
    128_000,
)
.with_batch_cached(1.00, 0.10, 6.00)
.with_flex(1.00, Some(0.10), 6.00)
.with_priority(4.00, Some(0.40), 24.00)
.with_high_tier(4.00, 18.00, None, 272_000);

/// gpt-5.6-luna — high-volume tier: extraction, classification, reformatting.
const OPENAI_GPT56_LUNA: Model = model(
    "gpt-5.6-luna",
    Provider::OpenAI,
    0.20,
    1.20,
    Some(0.02),
    1_050_000,
    128_000,
)
.with_batch_cached(0.10, 0.01, 0.60)
.with_flex(0.10, Some(0.01), 0.60)
.with_priority(0.40, Some(0.04), 2.40)
.with_high_tier(0.40, 1.80, None, 272_000);

/// gpt-5.5 — frontier 2026 model. Has Priority tier (rare among GPT-5.x SKUs).
const OPENAI_GPT55: Model = model(
    "gpt-5.5",
    Provider::OpenAI,
    5.00,
    30.00,
    Some(0.50),
    1_050_000,
    128_000,
)
.with_batch_cached(2.50, 0.25, 15.00)
.with_flex(2.50, Some(0.25), 15.00)
.with_priority(12.50, Some(1.25), 75.00)
.with_high_tier(10.00, 45.00, None, 272_000);

/// gpt-5.4 — 2026 mid-frontier. Long-context multiplier (>272K) per docs.
const OPENAI_GPT54: Model = model(
    "gpt-5.4",
    Provider::OpenAI,
    2.50,
    15.00,
    Some(0.25),
    1_050_000,
    128_000,
)
.with_batch_cached(1.25, 0.125, 7.50)
.with_flex(1.25, None, 7.50)
.with_high_tier(5.00, 22.50, Some(0.50), 272_000);

/// gpt-5.4-mini — 2026 mid-tier. Has Flex; Priority partial data (skipped).
const OPENAI_GPT54_MINI: Model = model(
    "gpt-5.4-mini",
    Provider::OpenAI,
    0.75,
    4.50,
    Some(0.075),
    400_000,
    128_000,
)
.with_batch_cached(0.375, 0.0375, 2.25)
.with_flex(0.375, Some(0.0375), 2.25);

/// DEPRECATED — API shutdown 2026-12-11, replaced by gpt-6.1-sol.
const OPENAI_GPT5: Model = model(
    "gpt-5",
    Provider::OpenAI,
    1.25,
    10.00,
    Some(0.125),
    400_000,
    128_000,
);

/// DEPRECATED — API shutdown 2026-12-11, replaced by gpt-6.1-terra.
const OPENAI_GPT5_MINI: Model = model(
    "gpt-5-mini",
    Provider::OpenAI,
    0.25,
    2.00,
    Some(0.025),
    400_000,
    128_000,
);

/// DEPRECATED — API shutdown 2026-12-11, replaced by gpt-6-luna.
const OPENAI_GPT5_NANO: Model = model(
    "gpt-5-nano",
    Provider::OpenAI,
    0.05,
    0.40,
    Some(0.005),
    400_000,
    128_000,
);

// Reasoning-heavy "pro" SKUs and the intermediate 5.1 / 5.2 point releases.
// The docs pricing page lists rates but not context windows for these; each
// mirrors the window published for its family's base SKU (gpt-5 = 400K,
// gpt-5.4 / 5.5 = 1.05M). Verify before using the window for capacity planning.

/// gpt-5.5-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT55_PRO: Model = model(
    "gpt-5.5-pro",
    Provider::OpenAI,
    30.00,
    180.00,
    None,
    1_050_000,
    128_000,
)
.with_batch(15.00, 90.00);

/// gpt-5.4-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT54_PRO: Model = model(
    "gpt-5.4-pro",
    Provider::OpenAI,
    30.00,
    180.00,
    None,
    1_050_000,
    128_000,
)
.with_batch(15.00, 90.00);

/// DEPRECATED — API shutdown 2027-04-01, replaced by gpt-6-luna.
const OPENAI_GPT54_NANO: Model = model(
    "gpt-5.4-nano",
    Provider::OpenAI,
    0.20,
    1.25,
    Some(0.02),
    400_000,
    128_000,
)
.with_batch_cached(0.10, 0.01, 0.625);

/// gpt-5.2 — 2025-12 point release.
const OPENAI_GPT52: Model = model(
    "gpt-5.2",
    Provider::OpenAI,
    1.75,
    14.00,
    Some(0.175),
    400_000,
    128_000,
)
.with_batch_cached(0.875, 0.0875, 7.00);

/// gpt-5.2-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT52_PRO: Model = model(
    "gpt-5.2-pro",
    Provider::OpenAI,
    21.00,
    168.00,
    None,
    400_000,
    128_000,
)
.with_batch(10.50, 84.00);

/// DEPRECATED — API shutdown 2027-04-01, replaced by gpt-6-sol. Same rates as
/// gpt-5.
const OPENAI_GPT51: Model = model(
    "gpt-5.1",
    Provider::OpenAI,
    1.25,
    10.00,
    Some(0.125),
    400_000,
    128_000,
)
.with_batch_cached(0.625, 0.0625, 5.00);

/// DEPRECATED — API shutdown 2026-12-11, replaced by gpt-6.1-sol with
/// `reasoning.mode: pro`. No cached-input rate published.
const OPENAI_GPT5_PRO: Model = model(
    "gpt-5-pro",
    Provider::OpenAI,
    15.00,
    120.00,
    None,
    400_000,
    128_000,
)
.with_batch(7.50, 60.00);

const OPENAI_EMBED_3_SMALL: Model = model(
    "text-embedding-3-small",
    Provider::OpenAI,
    0.02,
    0.0,
    None,
    8_191,
    0,
);

const OPENAI_EMBED_3_LARGE: Model = model(
    "text-embedding-3-large",
    Provider::OpenAI,
    0.13,
    0.0,
    None,
    8_191,
    0,
);

const OPENAI_EMBED_ADA_002: Model = model(
    "text-embedding-ada-002",
    Provider::OpenAI,
    0.10,
    0.0,
    None,
    8_191,
    0,
);

pub(super) const MODELS: &[Model] = &[
    OPENAI_GPT62_ASTRA,
    OPENAI_GPT62_SOL,
    OPENAI_GPT61_TERRA,
    OPENAI_GPT61_SOL,
    OPENAI_GPT6_ASTRA,
    OPENAI_GPT6_SOL,
    OPENAI_GPT6_LUNA,
    OPENAI_GPT56_SOL,
    OPENAI_GPT56_TERRA,
    OPENAI_GPT56_LUNA,
    OPENAI_GPT55,
    OPENAI_GPT55_PRO,
    OPENAI_GPT54,
    OPENAI_GPT54_MINI,
    OPENAI_GPT54_NANO,
    OPENAI_GPT54_PRO,
    OPENAI_GPT52,
    OPENAI_GPT52_PRO,
    OPENAI_GPT51,
    OPENAI_GPT5,
    OPENAI_GPT5_MINI,
    OPENAI_GPT5_NANO,
    OPENAI_GPT5_PRO,
    OPENAI_GPT41,
    OPENAI_GPT41_MINI,
    OPENAI_GPT41_NANO,
    OPENAI_GPT4O,
    OPENAI_GPT4O_MINI,
    OPENAI_O1,
    OPENAI_O1_MINI,
    OPENAI_O1_PRO,
    OPENAI_O3,
    OPENAI_O3_PRO,
    OPENAI_O3_MINI,
    OPENAI_O4_MINI,
    OPENAI_GPT4_TURBO,
    OPENAI_GPT4,
    OPENAI_GPT35_TURBO,
    OPENAI_EMBED_3_SMALL,
    OPENAI_EMBED_3_LARGE,
    OPENAI_EMBED_ADA_002,
];

#[cfg(test)]
mod tests {
    use crate::pricing::*;

    #[test]
    fn gpt6_family_registered_with_long_context_tier() {
        for id in [
            "gpt-6.2-astra",
            "gpt-6.2-sol",
            "gpt-6.1-terra",
            "gpt-6.1-sol",
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
        ] {
            let m = get_model(id).unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(m.context_window, 1_050_000, "{id}");
            assert_eq!(m.max_output, 128_000, "{id}");
            let ht = m.extended.high_tier.unwrap();
            assert_eq!(ht.threshold_tokens, 272_000, "{id}");
            assert_eq!(ht.input_per_1m, 2.0 * m.pricing.input_per_1m, "{id}");
            assert_eq!(ht.output_per_1m, 1.5 * m.pricing.output_per_1m, "{id}");
            assert!(
                m.extended.batch.is_some() && m.extended.priority.is_some(),
                "{id}"
            );
        }
    }

    #[test]
    fn gpt6_2_has_no_flex_tier() {
        for id in ["gpt-6.2-astra", "gpt-6.2-sol"] {
            assert!(get_model(id).unwrap().extended.flex.is_none(), "{id}");
        }
        assert!(get_model("gpt-6.1-sol").unwrap().extended.flex.is_some());
    }

    #[test]
    fn gpt56_sol_keeps_launch_rate_through_the_promotion() {
        let m = get_model("gpt-5.6-sol").unwrap();
        assert_eq!(
            (m.pricing.input_per_1m, m.pricing.output_per_1m),
            (5.0, 30.0)
        );
    }
}
