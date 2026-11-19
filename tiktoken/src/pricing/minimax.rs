//! MiniMax price table.

use super::*;

// USD rates read directly from the official pay-as-you-go card
// (platform.minimax.io/docs/guides/pricing-paygo), 2026-11-19. M3 and M3.1
// share the M2 vocabulary and split pattern. Context windows: 1M for M3 and
// M3.1 (text-generation guide), 204,800 for the M2 family. Max output is not
// published — 32K placeholder. Cache writes are not priced on the card.

/// minimax-m3 — mainline (2026-06-01). The card lists these as a permanent 50%
/// discount. Prompts over 512K tokens bill at $0.60 / $2.40, cache $0.12; the
/// priority tier is 1.5× standard.
const MINIMAX_M3: Model = model(
    "minimax-m3",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.06),
    1_000_000,
    32_768,
)
.with_priority(0.45, Some(0.09), 1.80)
.with_high_tier(0.60, 2.40, Some(0.12), 512_000);

/// minimax-m3.1-flash — low-cost tier (2026-10-19).
const MINIMAX_M31_FLASH: Model = model(
    "minimax-m3.1-flash",
    Provider::MiniMax,
    0.15,
    0.60,
    Some(0.015),
    1_000_000,
    32_768,
);

// The M2 family is no longer on the price card as of 2026-11, and MiniMax has
// published no shutdown date for it. The rates below are the 2026-08 card.

/// DEPRECATED — off the price card as of 2026-11. Previous mainline.
const MINIMAX_M27: Model = model(
    "minimax-m2.7",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.06),
    204_800,
    32_768,
);

/// DEPRECATED — off the price card as of 2026-11.
const MINIMAX_M25: Model = model(
    "minimax-m2.5",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.03),
    204_800,
    32_768,
);

/// DEPRECATED — off the price card as of 2026-11.
const MINIMAX_M21: Model = model(
    "minimax-m2.1",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.03),
    204_800,
    32_768,
);

/// DEPRECATED — off the price card as of 2026-11. The original M2 (2025-10).
const MINIMAX_M2_MODEL: Model = model(
    "minimax-m2",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.03),
    204_800,
    32_768,
);

pub(super) const MODELS: &[Model] = &[
    MINIMAX_M3,
    MINIMAX_M31_FLASH,
    MINIMAX_M27,
    MINIMAX_M25,
    MINIMAX_M21,
    MINIMAX_M2_MODEL,
];
