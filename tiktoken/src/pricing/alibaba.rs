//! Alibaba (Qwen) price table.

use super::*;

// USD prices from Alibaba Cloud Model Studio, International (Singapore) list
// prices, 2026-11. The page publishes a cache-hit rate only for qwen4-max and a
// max output only on qwen4-max's model page; entries without one carry a
// conservative 32K placeholder.

/// qwen4-max — current flagship (2026-11-16). One rate across the 1M window.
/// Max output is 131,072 tokens without thinking (262,144 with). Cache writes
/// ($1.875 for 5 minutes, $3 for 1 hour) are not modelled.
const QWEN_4_MAX: Model = model(
    "qwen4-max",
    Provider::Alibaba,
    1.50,
    4.50,
    Some(0.075),
    1_000_000,
    131_072,
);

/// qwen3.8-flash — high-volume tier (2026-08-26). One rate across the 1M window.
const QWEN_3_8_FLASH: Model = model(
    "qwen3.8-flash",
    Provider::Alibaba,
    0.15,
    0.47,
    None,
    1_000_000,
    32_768,
);

/// qwen3.8-2.4t-a95b — open-weight 3.8 flagship, hosted (2026-08-12).
const QWEN_3_8_2_4T: Model = model(
    "qwen3.8-2.4t-a95b",
    Provider::Alibaba,
    2.00,
    6.00,
    None,
    1_000_000,
    32_768,
);

/// qwen3.8-27b — open-weight 3.8 dense model, hosted (2026-08-17).
const QWEN_3_8_27B: Model = model(
    "qwen3.8-27b",
    Provider::Alibaba,
    0.50,
    3.00,
    None,
    1_000_000,
    32_768,
);

/// qwen3.8-max — 3.8 flagship (2026-08). Flat rate across the full 1M
/// context, with no long-prompt surcharge; `cached` is the implicit cache read.
const QWEN_3_8_MAX: Model = model(
    "qwen3.8-max",
    Provider::Alibaba,
    2.00,
    6.00,
    Some(0.25),
    1_000_000,
    32_768,
);

/// qwen3.5-plus — mid-tier. Prompts over 256K tokens bill at $0.50 / $3.
const QWEN_3_5_PLUS: Model = model(
    "qwen3.5-plus",
    Provider::Alibaba,
    0.40,
    2.40,
    None,
    1_000_000,
    32_768,
)
.with_high_tier(0.50, 3.00, None, 256_000);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
const QWEN_2_5_72B: Model = model(
    "qwen2.5-72b",
    Provider::Alibaba,
    1.4,
    5.6,
    None,
    128_000,
    8_192,
);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
const QWEN_2_5_32B: Model = model(
    "qwen2.5-32b",
    Provider::Alibaba,
    0.7,
    2.8,
    None,
    128_000,
    8_192,
);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
const QWEN_2_5_7B: Model = model(
    "qwen2.5-7b",
    Provider::Alibaba,
    0.175,
    0.7,
    None,
    128_000,
    8_192,
);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
const QWEN_3_MAX: Model = model(
    "qwen3-max",
    Provider::Alibaba,
    1.20,
    6.00,
    None,
    262_144,
    8_192,
);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
const QWEN_3_PLUS: Model = model(
    "qwen3-plus",
    Provider::Alibaba,
    0.40,
    2.4,
    None,
    128_000,
    8_192,
);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
/// Pricing taken from the `qwen3-coder-plus` SKU on Model Studio.
const QWEN_3_CODER: Model = model(
    "qwen3-coder",
    Provider::Alibaba,
    1.0,
    5.0,
    None,
    262_144,
    8_192,
);

/// DEPRECATED — no longer on the International price page as of 2026-11; the
/// shutdown date was not published in a form that could be checked.
const QWEN_3_8B: Model = model(
    "qwen3-8b",
    Provider::Alibaba,
    0.18,
    0.7,
    None,
    128_000,
    8_192,
);

pub(super) const MODELS: &[Model] = &[
    QWEN_4_MAX,
    QWEN_3_8_MAX,
    QWEN_3_8_FLASH,
    QWEN_3_8_2_4T,
    QWEN_3_8_27B,
    QWEN_3_5_PLUS,
    QWEN_3_MAX,
    QWEN_3_PLUS,
    QWEN_3_CODER,
    QWEN_3_8B,
    QWEN_2_5_72B,
    QWEN_2_5_32B,
    QWEN_2_5_7B,
];
