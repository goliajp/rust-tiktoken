//! Google Gemini price table.

use super::*;

// Source: aistudio.google.com/docs/pricing (formerly ai.google.dev), models and
// deprecations pages, 2026-11. Where a rate is introductory, the entry holds the
// rate the page schedules after it, and the doc says until when the lower one
// applies. Gemini 4 and 3.8 Flash have no image-token formula published in a
// form this table can encode, so they carry no `vision`.

/// gemini-4-neon — Gemini 4 flagship (2026-11-18). One rate across the 1M
/// window; output up to 65,536 tokens by default.
const GEMINI_4_NEON: Model = model(
    "gemini-4-neon",
    Provider::Google,
    3.00,
    15.00,
    Some(0.15),
    1_048_576,
    65_536,
)
.with_batch_cached(1.50, 0.075, 7.50);

/// gemini-4-hydrogen — Gemini 4 (2026-11-12).
const GEMINI_4_HYDROGEN: Model = model(
    "gemini-4-hydrogen",
    Provider::Google,
    1.50,
    7.50,
    Some(0.075),
    1_048_576,
    65_536,
)
.with_batch_cached(0.75, 0.0375, 3.75);

/// gemini-4-carbon — Gemini 4 (2026-11-10). Rates from 2026-12-01; until
/// 2026-11-30 they are half ($2 / $10, cached $0.10, batch $1 / $5).
const GEMINI_4_CARBON: Model = model(
    "gemini-4-carbon",
    Provider::Google,
    4.00,
    20.00,
    Some(0.20),
    1_048_576,
    65_536,
)
.with_batch_cached(2.00, 0.10, 10.00);

/// gemini-4-argon — Gemini 4 (2026-10-13). Rates from 2026-12-01; until
/// 2026-11-30 they are half ($2 / $10, cached $0.10, batch $1 / $5).
const GEMINI_4_ARGON: Model = model(
    "gemini-4-argon",
    Provider::Google,
    4.00,
    20.00,
    Some(0.20),
    1_048_576,
    65_536,
)
.with_batch_cached(2.00, 0.10, 10.00);

/// gemini-3.8-flash — current Flash (2026-09-02). Rates from 2027-01-01; until
/// 2026-12-31 they are half ($0.75 / $3.75, cached $0.075, batch $0.375 /
/// $1.875).
const GEMINI_38_FLASH: Model = model(
    "gemini-3.8-flash",
    Provider::Google,
    1.50,
    7.50,
    Some(0.15),
    1_048_576,
    65_536,
)
.with_batch_cached(0.75, 0.075, 3.75);

/// gemini-3.1-pro-preview — context-tiered like 2.5 Pro: the standard rates
/// cover prompts up to 200k tokens, `with_high_tier` the rest.
const GEMINI_31_PRO: Model = model(
    "gemini-3.1-pro-preview",
    Provider::Google,
    2.00,
    12.00,
    Some(0.20),
    1_048_576,
    65_536,
)
.with_batch(1.00, 6.00)
.with_high_tier(4.00, 18.00, Some(0.40), 200_000)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// gemini-3.6-flash — Flash generation before 3.8. Rates from 2027-01-01;
/// until 2026-12-31 they are half ($0.75 / $3.75, cached $0.075, batch $0.375 /
/// $1.875).
const GEMINI_36_FLASH: Model = model(
    "gemini-3.6-flash",
    Provider::Google,
    1.50,
    7.50,
    Some(0.15),
    1_048_576,
    65_536,
)
.with_batch(0.75, 3.75)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// gemini-3.5-flash-lite — budget tier of the 3.5 generation.
const GEMINI_35_FLASH_LITE: Model = model(
    "gemini-3.5-flash-lite",
    Provider::Google,
    0.30,
    2.50,
    Some(0.03),
    1_048_576,
    65_536,
)
.with_batch(0.15, 1.25)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// gemini-2.5-flash-lite — cheapest generally available Gemini SKU.
/// Text/image/video input is $0.10/M; audio input is $0.30/M.
const GEMINI_25_FLASH_LITE: Model = model(
    "gemini-2.5-flash-lite",
    Provider::Google,
    0.10,
    0.40,
    Some(0.01),
    1_048_576,
    65_536,
)
.with_batch(0.05, 0.20)
.with_audio_input(0.30)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// DEPRECATED — deprecated 2026-10-08; requests now go to gemini-3.6-flash.
const GEMINI_35_FLASH: Model = model(
    "gemini-3.5-flash",
    Provider::Google,
    1.50,
    9.00,
    Some(0.15),
    1_048_576,
    65_536,
)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// Text input is $0.25/M (standard); audio input is $0.50/M via `with_audio_input`.
/// Shuts down no earlier than 2027-05-07; the replacement is gemini-3.5-flash-lite.
const GEMINI_31_FLASH_LITE: Model = model(
    "gemini-3.1-flash-lite",
    Provider::Google,
    0.25,
    1.50,
    Some(0.025),
    1_048_576,
    65_536,
)
.with_audio_input(0.50)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// DEPRECATED — deprecated 2026-11-02. From 2026-12-01 requests go to
/// gemini-3.8-flash and bill at its rates.
const GEMINI_25_PRO: Model = model(
    "gemini-2.5-pro",
    Provider::Google,
    1.25,
    10.00,
    Some(0.125),
    1_048_576,
    65_536,
)
.with_batch(0.625, 5.00)
.with_high_tier(2.50, 15.00, Some(0.25), 200_000)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// DEPRECATED — deprecated 2026-11-02. From 2026-12-01 requests go to
/// gemini-3.5-flash-lite and bill at its rates.
///
/// Text input is $0.30/M (standard); audio input is $1.00/M via
/// `with_audio_input`. Cached input corrected 2026-11 from $0.075 to the
/// published $0.03.
const GEMINI_25_FLASH: Model = model(
    "gemini-2.5-flash",
    Provider::Google,
    0.30,
    2.50,
    Some(0.03),
    1_048_576,
    65_536,
)
.with_batch(0.15, 1.25)
.with_audio_input(1.00)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// DEPRECATED — shut down 2026-06-01. Use gemini-2.5-flash.
const GEMINI_20_FLASH: Model = model(
    "gemini-2.0-flash",
    Provider::Google,
    0.10,
    0.40,
    Some(0.025),
    1_048_576,
    8_192,
);

/// DEPRECATED — no longer on the official Gemini pricing page. Use gemini-2.5-pro.
const GEMINI_15_PRO: Model = model(
    "gemini-1.5-pro",
    Provider::Google,
    1.25,
    5.00,
    Some(0.3125),
    2_097_152,
    8_192,
);

/// DEPRECATED — no longer on the official Gemini pricing page. Use gemini-2.5-flash.
const GEMINI_15_FLASH: Model = model(
    "gemini-1.5-flash",
    Provider::Google,
    0.075,
    0.30,
    Some(0.01875),
    1_048_576,
    8_192,
);

/// DEPRECATED — no longer on the official Gemini pricing page.
const GEMINI_EMBED: Model = model(
    "text-embedding-004",
    Provider::Google,
    0.00,
    0.0,
    None,
    2_048,
    0,
);

/// gemini-embedding-2 — current Gemini embedding model (2026-04-22). $0.20 / 1M
/// text tokens (batch $0.10); image, audio and video inputs are priced
/// separately and not modelled. 8,192-token input limit.
const GEMINI_EMBEDDING_2: Model = model(
    "gemini-embedding-2",
    Provider::Google,
    0.20,
    0.0,
    None,
    8_192,
    0,
)
.with_batch(0.10, 0.0);

/// gemini-embedding-001 — successor to the deprecated `text-embedding-004`
/// above; shuts down no earlier than 2028-05-14, replaced by
/// gemini-embedding-2.
///
/// $0.15 / 1M input tokens (paid tier, batch $0.075), 2,048-token input limit.
const GEMINI_EMBEDDING_001: Model = model(
    "gemini-embedding-001",
    Provider::Google,
    0.15,
    0.0,
    None,
    2_048,
    0,
)
.with_batch(0.075, 0.0);

pub(super) const MODELS: &[Model] = &[
    GEMINI_4_NEON,
    GEMINI_4_HYDROGEN,
    GEMINI_4_CARBON,
    GEMINI_4_ARGON,
    GEMINI_38_FLASH,
    GEMINI_31_PRO,
    GEMINI_36_FLASH,
    GEMINI_35_FLASH,
    GEMINI_35_FLASH_LITE,
    GEMINI_31_FLASH_LITE,
    GEMINI_25_PRO,
    GEMINI_25_FLASH,
    GEMINI_25_FLASH_LITE,
    GEMINI_20_FLASH,
    GEMINI_15_PRO,
    GEMINI_15_FLASH,
    GEMINI_EMBED,
    GEMINI_EMBEDDING_2,
    GEMINI_EMBEDDING_001,
];

#[cfg(test)]
mod tests {
    use crate::pricing::*;

    #[test]
    fn introductory_gemini_entries_hold_the_scheduled_rate() {
        for (id, input, output) in [
            ("gemini-4-argon", 4.0, 20.0),
            ("gemini-4-carbon", 4.0, 20.0),
            ("gemini-3.8-flash", 1.5, 7.5),
            ("gemini-3.6-flash", 1.5, 7.5),
        ] {
            let m = get_model(id).unwrap();
            assert_eq!(
                (m.pricing.input_per_1m, m.pricing.output_per_1m),
                (input, output),
                "{id}"
            );
        }
    }
}
