//! Zhipu (GLM / Z.ai) price table.

use super::*;

// USD rates from the official Z.ai pricing page (docs.z.ai), read 2026-11-19.
// Context windows from the model release notes (5.2 and 5.3 = 1M; 4.x =
// 128K/200K). The 5.3 model pages publish a 128K max output; the others do not,
// and carry a 32K placeholder. Cache storage is free for a limited time and not
// modelled.

/// glm-5.3 — flagship (2026-08-18).
const GLM_53: Model = model(
    "glm-5.3",
    Provider::Zhipu,
    1.40,
    4.40,
    Some(0.26),
    1_048_576,
    131_072,
);

/// glm-5.3-flashx — mid tier of 5.3 (2026-08-26). `glm-5.3-flashx-legal` bills
/// the same.
const GLM_53_FLASHX: Model = model(
    "glm-5.3-flashx",
    Provider::Zhipu,
    0.37,
    1.25,
    Some(0.075),
    1_048_576,
    131_072,
);

/// glm-5.3-flash — low-cost tier of 5.3 (2026-08-26).
const GLM_53_FLASH: Model = model(
    "glm-5.3-flash",
    Provider::Zhipu,
    0.15,
    0.50,
    Some(0.03),
    1_048_576,
    131_072,
);

/// glm-5.2 — previous flagship (2026-06-16, MIT open weights).
const GLM_52: Model = model(
    "glm-5.2",
    Provider::Zhipu,
    1.40,
    4.40,
    Some(0.26),
    1_048_576,
    32_768,
);

/// glm-5 — first GLM-5 release (200K context per release notes).
const GLM_5_MODEL: Model = model(
    "glm-5",
    Provider::Zhipu,
    1.00,
    3.20,
    Some(0.20),
    200_000,
    32_768,
);

/// glm-4.7 — current 4.x mainline.
const GLM_47: Model = model(
    "glm-4.7",
    Provider::Zhipu,
    0.60,
    2.20,
    Some(0.11),
    200_000,
    32_768,
);

/// glm-4.5 — 2025 open-weights mainline.
const GLM_45: Model = model(
    "glm-4.5",
    Provider::Zhipu,
    0.60,
    2.20,
    Some(0.11),
    131_072,
    32_768,
);

/// glm-4.5-air — lightweight 4.5 SKU.
const GLM_45_AIR: Model = model(
    "glm-4.5-air",
    Provider::Zhipu,
    0.20,
    1.10,
    Some(0.03),
    131_072,
    32_768,
);

pub(super) const MODELS: &[Model] = &[
    GLM_53,
    GLM_53_FLASHX,
    GLM_53_FLASH,
    GLM_52,
    GLM_5_MODEL,
    GLM_47,
    GLM_45,
    GLM_45_AIR,
];
