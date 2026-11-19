//! Voyage AI (embeddings) price table.

use super::*;

// Prices and context lengths from docs.voyageai.com/docs/pricing and
// .../docs/embeddings, 2026-08. All eight share a 32,000-token context.
// The voyage-4 generation and voyage-code-4 include the first 200M tokens
// free; the 3-series does not. Embeddings have no output tokens.

/// voyage-4-large — highest-quality tier of the current generation.
const VOYAGE_4_LARGE: Model = model(
    "voyage-4-large",
    Provider::Voyage,
    0.12,
    0.0,
    None,
    32_000,
    0,
);

/// voyage-4 — balanced tier of the current generation.
const VOYAGE_4: Model = model("voyage-4", Provider::Voyage, 0.06, 0.0, None, 32_000, 0);

/// voyage-4-lite — budget tier of the current generation.
const VOYAGE_4_LITE: Model = model(
    "voyage-4-lite",
    Provider::Voyage,
    0.02,
    0.0,
    None,
    32_000,
    0,
);

/// voyage-code-4 — code-specialised, current generation.
const VOYAGE_CODE_4: Model = model(
    "voyage-code-4",
    Provider::Voyage,
    0.12,
    0.0,
    None,
    32_000,
    0,
);

/// voyage-3-large — previous generation, priced above voyage-4-large.
const VOYAGE_3_LARGE: Model = model(
    "voyage-3-large",
    Provider::Voyage,
    0.18,
    0.0,
    None,
    32_000,
    0,
);

/// voyage-3.5 — previous generation, balanced.
const VOYAGE_3_5: Model = model("voyage-3.5", Provider::Voyage, 0.06, 0.0, None, 32_000, 0);

/// voyage-3.5-lite — previous generation, budget.
const VOYAGE_3_5_LITE: Model = model(
    "voyage-3.5-lite",
    Provider::Voyage,
    0.02,
    0.0,
    None,
    32_000,
    0,
);

/// voyage-code-3 — previous generation, code-specialised.
const VOYAGE_CODE_3: Model = model(
    "voyage-code-3",
    Provider::Voyage,
    0.18,
    0.0,
    None,
    32_000,
    0,
);

pub(super) const MODELS: &[Model] = &[
    VOYAGE_4_LARGE,
    VOYAGE_4,
    VOYAGE_4_LITE,
    VOYAGE_CODE_4,
    VOYAGE_3_LARGE,
    VOYAGE_3_5,
    VOYAGE_3_5_LITE,
    VOYAGE_CODE_3,
];

#[cfg(test)]
mod tests {
    use crate::pricing::*;

    #[test]
    fn voyage_embeddings_price() {
        for (id, per_1m) in [
            ("voyage-4-large", 0.12),
            ("voyage-4", 0.06),
            ("voyage-4-lite", 0.02),
            ("voyage-code-4", 0.12),
            ("voyage-3-large", 0.18),
            ("voyage-3.5", 0.06),
            ("voyage-3.5-lite", 0.02),
            ("voyage-code-3", 0.18),
        ] {
            let m = get_model(id).unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(m.provider, Provider::Voyage, "{id}");
            assert_eq!(m.context_window, 32_000, "{id}");
            let cost = estimate_cost(id, 1_000_000, 0).unwrap();
            assert!((cost - per_1m).abs() < 1e-9, "{id}: {cost} != {per_1m}");
        }
    }

    /// The dashed spelling of a dotted Voyage id must not miss.
    #[test]
    fn voyage_dashed_spelling_resolves() {
        assert_eq!(resolve_model("voyage-3-5").unwrap().model.id, "voyage-3.5");
        assert_eq!(
            resolve_model("voyage-3-5-lite").unwrap().model.id,
            "voyage-3.5-lite"
        );
    }
}
