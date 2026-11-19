//! Moonshot (Kimi) price table.

use super::*;

// USD rates from the official Moonshot API price card (platform.kimi.ai),
// read 2026-11-19. Max-output limits are not published; entries carry a
// conservative 32K placeholder — verify before capacity planning. Batch bills
// at 60% of the standard rate; the 2026-11 card lists batch for the K3 series
// only, and the K2 batch rates below date from 2026-08. The K3 series also
// charges for cache writes (1× input for a 5-minute TTL, 2× for 1 hour), which
// is not modelled.

/// kimi-k3.1-pro — flagship (2026-10-23). Flat rate across the full 1M context.
const KIMI_K31_PRO: Model = model(
    "kimi-k3.1-pro",
    Provider::Moonshot,
    2.00,
    10.00,
    Some(0.20),
    1_048_576,
    32_768,
)
.with_batch(1.20, 6.00);

/// kimi-k3.1-pro-highspeed — the same model served faster, at twice the rate.
const KIMI_K31_PRO_HIGHSPEED: Model = model(
    "kimi-k3.1-pro-highspeed",
    Provider::Moonshot,
    4.00,
    20.00,
    Some(0.20),
    1_048_576,
    32_768,
);

/// kimi-k3 — previous flagship (2026-07-16). Flat rate across the full 1M context.
const KIMI_K3_MODEL: Model = model(
    "kimi-k3",
    Provider::Moonshot,
    3.00,
    15.00,
    Some(0.30),
    1_048_576,
    32_768,
)
.with_batch(1.80, 9.00);

/// kimi-k2.7-code — dedicated coding SKU. The K2 series closed to new users on
/// 2026-10-31 and is scheduled to retire at the end of 2026-12.
const KIMI_K27_CODE: Model = model(
    "kimi-k2.7-code",
    Provider::Moonshot,
    0.95,
    4.00,
    Some(0.19),
    262_144,
    32_768,
)
.with_batch(0.57, 2.40);

/// kimi-k2.6 — vision + text. Closed to new users 2026-10-31, like all K2.
const KIMI_K26: Model = model(
    "kimi-k2.6",
    Provider::Moonshot,
    0.95,
    4.00,
    Some(0.16),
    262_144,
    32_768,
)
.with_batch(0.57, 2.40);

/// DEPRECATED — retired 2026-08; the API returns 404 for it.
const KIMI_K25: Model = model(
    "kimi-k2.5",
    Provider::Moonshot,
    0.60,
    3.00,
    None,
    262_144,
    32_768,
)
.with_batch(0.36, 1.80);

pub(super) const MODELS: &[Model] = &[
    KIMI_K31_PRO,
    KIMI_K31_PRO_HIGHSPEED,
    KIMI_K3_MODEL,
    KIMI_K27_CODE,
    KIMI_K26,
    KIMI_K25,
];
