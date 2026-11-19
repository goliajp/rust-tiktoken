//! Mistral price table.

use super::*;

// Source: docs.mistral.ai model pages and the model retirement table, 2026-11.
// Max output is not published per model; entries carry a 4K placeholder.

/// mistral-large-4 — Mistral Large 4 (v26.10, released 2026-11-05), also served
/// as `mistral-large-latest`. Uses the 204.8k Tekken v15 vocabulary.
const MISTRAL_LARGE_4: Model = model(
    "mistral-large-4",
    Provider::Mistral,
    0.70,
    2.10,
    Some(0.07),
    1_000_000,
    4_096,
);

/// DEPRECATED — Mistral Large 3 (`mistral-large-2512`): deprecated 2026-10-31,
/// retires 2027-04-30. `mistral-large-latest` now points at mistral-large-4.
const MISTRAL_LARGE: Model = model(
    "mistral-large",
    Provider::Mistral,
    0.5,
    1.5,
    None,
    128_000,
    4_096,
);

/// mistral-small — `mistral-small-latest` (Small 4, `mistral-small-2603`).
/// Context corrected 2026-11 from 128K to the published 256K.
const MISTRAL_SMALL: Model = model(
    "mistral-small",
    Provider::Mistral,
    0.15,
    0.60,
    None,
    256_000,
    4_096,
);

/// DEPRECATED — devstral-2512 (Devstral 2) retired 2026-07-31; no Devstral
/// model is on sale.
/// devstral-medium-latest (Devstral 2) — coding-agent SKU.
const DEVSTRAL_MEDIUM: Model = model(
    "devstral-medium",
    Provider::Mistral,
    0.40,
    2.00,
    None,
    256_000,
    4_096,
);

/// DEPRECATED — labs-devstral-small-2512 retired 2026-03-31.
/// devstral-small-latest (Devstral Small 2) — budget coding-agent SKU.
const DEVSTRAL_SMALL: Model = model(
    "devstral-small",
    Provider::Mistral,
    0.10,
    0.30,
    None,
    256_000,
    4_096,
);

/// ministral-3b-latest (Ministral 3 · 3B) — cheapest hosted Mistral SKU.
const MINISTRAL_3B: Model = model(
    "ministral-3b",
    Provider::Mistral,
    0.10,
    0.10,
    None,
    128_000,
    4_096,
);

/// ministral-8b-latest (Ministral 3 · 8B).
const MINISTRAL_8B: Model = model(
    "ministral-8b",
    Provider::Mistral,
    0.15,
    0.15,
    None,
    128_000,
    4_096,
);

/// ministral-14b-latest (Ministral 3 · 14B). Context corrected 2026-11 from
/// 128K to the published 256K.
const MINISTRAL_14B: Model = model(
    "ministral-14b",
    Provider::Mistral,
    0.20,
    0.20,
    None,
    256_000,
    4_096,
);

const MISTRAL_NEMO: Model = model(
    "mistral-nemo",
    Provider::Mistral,
    0.15,
    0.15,
    None,
    128_000,
    4_096,
);

/// DEPRECATED — Mistral Medium 3.5 deprecated 2026-10-31, retires 2027-04-30.
const MISTRAL_MEDIUM: Model = model(
    "mistral-medium",
    Provider::Mistral,
    1.5,
    7.5,
    None,
    128_000,
    4_096,
);

/// codestral — Codestral 25.08. Context corrected 2026-11 from 256K to the
/// published 128K.
const CODESTRAL: Model = model(
    "codestral",
    Provider::Mistral,
    0.30,
    0.90,
    None,
    128_000,
    4_096,
);

/// DEPRECATED — no longer listed on the official Mistral La Plateforme pricing page.
const PIXTRAL_LARGE: Model = model(
    "pixtral-large",
    Provider::Mistral,
    2.00,
    6.00,
    None,
    131_072,
    4_096,
);

/// mistral-embed — `mistral-embed-2312`, $0.10 / 1M input tokens, 8K input.
const MISTRAL_EMBED: Model = model(
    "mistral-embed",
    Provider::Mistral,
    0.10,
    0.0,
    None,
    8_192,
    0,
);

/// codestral-embed — `codestral-embed-2505`, $0.15 / 1M input tokens, 8K input.
const CODESTRAL_EMBED: Model = model(
    "codestral-embed",
    Provider::Mistral,
    0.15,
    0.0,
    None,
    8_192,
    0,
);

const MIXTRAL_8X7B: Model = model(
    "mixtral-8x7b",
    Provider::Mistral,
    0.7,
    0.7,
    None,
    32_768,
    4_096,
);

pub(super) const MODELS: &[Model] = &[
    MISTRAL_LARGE_4,
    MISTRAL_LARGE,
    MISTRAL_MEDIUM,
    MISTRAL_SMALL,
    MISTRAL_NEMO,
    CODESTRAL,
    DEVSTRAL_MEDIUM,
    DEVSTRAL_SMALL,
    MINISTRAL_3B,
    MINISTRAL_8B,
    MINISTRAL_14B,
    PIXTRAL_LARGE,
    MIXTRAL_8X7B,
    MISTRAL_EMBED,
    CODESTRAL_EMBED,
];
