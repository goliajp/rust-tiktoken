//! OpenAI price table, older generations: GPT-4.x, GPT-4o, the o-series and
//! GPT-3.5. Most are shut down or scheduled to be; the rates stay so existing
//! cost lookups keep working.

use super::*;

/// DEPRECATED — API shutdown 2027-01-14 (snapshot gpt-4.1-2025-04-14),
/// replaced by gpt-6-luna.
pub(super) const OPENAI_GPT41: Model = model(
    "gpt-4.1",
    Provider::OpenAI,
    2.00,
    8.00,
    Some(0.50),
    1_000_000,
    32_768,
)
.with_batch(1.00, 4.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 85,
    per_tile_tokens: 170,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2027-01-14 (snapshot gpt-4.1-mini-2025-04-14),
/// replaced by gpt-6-luna.
pub(super) const OPENAI_GPT41_MINI: Model = model(
    "gpt-4.1-mini",
    Provider::OpenAI,
    0.40,
    1.60,
    Some(0.10),
    1_000_000,
    32_768,
)
.with_batch(0.20, 0.80)
.with_vision(VisionPricing::OpenAIPatchBased {
    patch_px: 32,
    multiplier: 1.62,
});

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.6-luna.
pub(super) const OPENAI_GPT41_NANO: Model = model(
    "gpt-4.1-nano",
    Provider::OpenAI,
    0.10,
    0.40,
    Some(0.025),
    1_000_000,
    32_768,
)
.with_batch(0.05, 0.20)
.with_vision(VisionPricing::OpenAIPatchBased {
    patch_px: 32,
    multiplier: 2.46,
});

/// DEPRECATED — API shutdown 2027-01-14 (snapshots 2024-08-06 and 2024-11-20;
/// the 2024-05-13 snapshot shut down 2026-10-23), replaced by gpt-6-luna.
pub(super) const OPENAI_GPT4O: Model = model(
    "gpt-4o",
    Provider::OpenAI,
    2.50,
    10.00,
    Some(1.25),
    128_000,
    16_384,
)
.with_batch(1.25, 5.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 85,
    per_tile_tokens: 170,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2027-01-14 (snapshot gpt-4o-mini-2024-07-18),
/// replaced by gpt-6-luna.
pub(super) const OPENAI_GPT4O_MINI: Model = model(
    "gpt-4o-mini",
    Provider::OpenAI,
    0.15,
    0.60,
    Some(0.075),
    128_000,
    16_384,
)
.with_batch(0.075, 0.30)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 2833,
    per_tile_tokens: 5667,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
pub(super) const OPENAI_O1: Model = model(
    "o1",
    Provider::OpenAI,
    15.00,
    60.00,
    Some(7.50),
    200_000,
    100_000,
)
.with_batch(7.50, 30.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 75,
    per_tile_tokens: 150,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2025-10-27, replaced by o4-mini. Price updated
/// 2026-06: input/output dropped from $3.00/$12.00 to match the o3-mini rate of
/// $1.10/$4.40.
pub(super) const OPENAI_O1_MINI: Model = model(
    "o1-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.55),
    128_000,
    65_536,
)
.with_batch(0.55, 2.20);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5-pro.
pub(super) const OPENAI_O1_PRO: Model = model(
    "o1-pro",
    Provider::OpenAI,
    150.00,
    600.00,
    None,
    200_000,
    100_000,
);

/// DEPRECATED — API shutdown 2026-12-11, replaced by gpt-6.1-sol.
pub(super) const OPENAI_O3: Model = model(
    "o3",
    Provider::OpenAI,
    2.00,
    8.00,
    Some(0.50),
    200_000,
    100_000,
)
.with_batch(1.00, 4.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 75,
    per_tile_tokens: 150,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2026-12-11, replaced by gpt-6.1-sol with
/// `reasoning.mode: pro`.
pub(super) const OPENAI_O3_PRO: Model = model(
    "o3-pro",
    Provider::OpenAI,
    20.00,
    80.00,
    None,
    200_000,
    100_000,
)
.with_batch(10.00, 40.00);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
pub(super) const OPENAI_O3_MINI: Model = model(
    "o3-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.55),
    200_000,
    100_000,
)
.with_batch(0.55, 2.20);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.4-mini.
pub(super) const OPENAI_O4_MINI: Model = model(
    "o4-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.275),
    200_000,
    100_000,
)
.with_batch(0.55, 2.20)
.with_vision(VisionPricing::OpenAIPatchBased {
    patch_px: 32,
    multiplier: 1.72,
});

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
pub(super) const OPENAI_GPT4_TURBO: Model = model(
    "gpt-4-turbo",
    Provider::OpenAI,
    10.00,
    30.00,
    None,
    128_000,
    4_096,
)
.with_batch(5.00, 15.00);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
pub(super) const OPENAI_GPT4: Model =
    model("gpt-4", Provider::OpenAI, 30.00, 60.00, None, 8_192, 8_192).with_batch(15.00, 30.00);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.4-mini.
pub(super) const OPENAI_GPT35_TURBO: Model = model(
    "gpt-3.5-turbo",
    Provider::OpenAI,
    0.50,
    1.50,
    None,
    16_385,
    4_096,
)
.with_batch(0.25, 0.75);
