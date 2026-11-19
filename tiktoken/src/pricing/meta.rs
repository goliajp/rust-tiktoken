//! Meta (Llama via hosted APIs) price table.

use super::*;

// Meta does not sell API access directly. Each model is pinned to a specific
// hoster's serverless inference price, checked 2026-11-19; the source is
// recorded per entry. If your billing uses a different hoster, adjust at the
// call site. Since 2026-08 Meta's open-weight releases carry the Muse name; they
// share the Llama 4 vocabulary.

/// muse-spark-open — Muse Spark Open 816B-A42B (2026-10-15). Source: Together AI
/// serverless inference (https://www.together.ai/pricing). Together publishes
/// no context length; 1M is from Meta's release post. Max output unpublished.
const META_MUSE_SPARK_OPEN: Model = model(
    "muse-spark-open",
    Provider::Meta,
    1.25,
    4.25,
    Some(0.15),
    1_000_000,
    8_192,
);

/// muse-glimmer-30b — Muse Glimmer 30B (2026-08). Source: Together AI
/// serverless inference. No context length is published by Together or Meta;
/// 32K is a placeholder.
const META_MUSE_GLIMMER_30B: Model = model(
    "muse-glimmer-30b",
    Provider::Meta,
    0.35,
    1.50,
    Some(0.04),
    32_768,
    8_192,
);

/// DEPRECATED — no major hoster offers serverless llama-3.1-405b as of 2026-06
/// (Together AI dropped it; DeepInfra / Groq never carried it). Price retained
/// for historical cost lookups; corresponds to Together AI's 2024 flat rate.
const META_LLAMA_3_1_405B: Model = model(
    "llama-3.1-405b",
    Provider::Meta,
    3.00,
    3.00,
    None,
    128_000,
    4_096,
);

/// DEPRECATED — superseded on most hosters by `llama-3.3-70b`. Together AI
/// migrated their 70B serverless SKU to 3.3 Turbo; DeepInfra / Groq host
/// the 3.3 variants only. Price retained for historical cost lookups.
const META_LLAMA_3_1_70B: Model = model(
    "llama-3.1-70b",
    Provider::Meta,
    0.88,
    0.88,
    None,
    128_000,
    4_096,
);

/// Source: DeepInfra serverless inference, Meta-Llama-3.1-8B-Instruct-Turbo
/// (https://deepinfra.com/pricing), 2026-11. Output was $0.05 in 2026-06.
const META_LLAMA_3_1_8B: Model = model(
    "llama-3.1-8b",
    Provider::Meta,
    0.02,
    0.04,
    None,
    128_000,
    4_096,
);

/// Source: DeepInfra serverless inference (Llama-3.3-70B-Instruct-Turbo)
/// https://deepinfra.com/pricing — unchanged 2026-11.
const META_LLAMA_3_3_70B: Model = model(
    "llama-3.3-70b",
    Provider::Meta,
    0.10,
    0.32,
    None,
    128_000,
    4_096,
);

/// Source: DeepInfra serverless inference, Llama-4-Scout-17B-16E
/// (https://deepinfra.com/pricing), 2026-11. Moved from Groq, which no longer
/// lists Scout (was $0.11 / $0.34, 10M context). DeepInfra serves 320K.
const META_LLAMA_4_SCOUT: Model = model(
    "llama-4-scout",
    Provider::Meta,
    0.10,
    0.30,
    None,
    320_000,
    8_192,
);

/// DEPRECATED — as of 2026-11 no major hoster (DeepInfra, Groq, Together)
/// offers it as serverless inference. Last source: DeepInfra
/// (Llama-4-Maverick-17B-128E-Instruct-FP8), 2026-06.
const META_LLAMA_4_MAVERICK: Model = model(
    "llama-4-maverick",
    Provider::Meta,
    0.15,
    0.60,
    None,
    1_000_000,
    8_192,
);

pub(super) const MODELS: &[Model] = &[
    META_MUSE_SPARK_OPEN,
    META_MUSE_GLIMMER_30B,
    META_LLAMA_4_SCOUT,
    META_LLAMA_4_MAVERICK,
    META_LLAMA_3_1_405B,
    META_LLAMA_3_1_70B,
    META_LLAMA_3_1_8B,
    META_LLAMA_3_3_70B,
];
