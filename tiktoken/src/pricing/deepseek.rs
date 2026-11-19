//! DeepSeek price table.

use super::*;

// Source: api-docs.deepseek.com/quick_start/pricing and news, 2026-11. Since
// 2026-08-16 DeepSeek bills peak and off-peak hours separately: peak is 01:00 to
// 04:00 and 06:00 to 10:00 UTC on weekdays (Chinese public holidays excluded),
// and every other hour bills at half the peak rate. The table holds one rate per
// model, so these entries carry the peak rate, an upper bound. Max output is
// 384K for both current models.

/// deepseek-flash — DeepSeek-V4.1-Flash (2026-09-10; snapshot 1017 since
/// 2026-10-17). Peak rates; off-peak is $0.15 / $0.60, cache hit $0.003.
const DEEPSEEK_FLASH: Model = model(
    "deepseek-flash",
    Provider::DeepSeek,
    0.30,
    1.20,
    Some(0.006),
    1_000_000,
    384_000,
);

/// deepseek-v4-pro — DeepSeek-V4-Pro-0813. Peak rates; off-peak is $0.66 /
/// $1.98, cache hit $0.022. Repriced 2026-08-16 from $0.435 / $0.87.
const DEEPSEEK_V4_PRO: Model = model(
    "deepseek-v4-pro",
    Provider::DeepSeek,
    1.32,
    3.96,
    Some(0.044),
    1_000_000,
    384_000,
);

/// DEPRECATED — V4-Flash retired 2026-09-10. The name is still accepted, served
/// by V4.1-Flash and billed as deepseek-flash, so the rates here follow it.
const DEEPSEEK_V4_FLASH: Model = model(
    "deepseek-v4-flash",
    Provider::DeepSeek,
    0.30,
    1.20,
    Some(0.006),
    1_000_000,
    384_000,
);

/// DEPRECATED — alias `deepseek-chat` deprecated 2026-07-24; superseded by
/// `deepseek-v4-flash` and no longer on the official price card.
const DEEPSEEK_V3: Model = model(
    "deepseek-v3",
    Provider::DeepSeek,
    0.27,
    1.10,
    Some(0.07),
    128_000,
    8_192,
);

/// DEPRECATED — alias `deepseek-reasoner` deprecates 2026-07-24; replaced by `deepseek-v4-pro`.
const DEEPSEEK_R1: Model = model(
    "deepseek-r1",
    Provider::DeepSeek,
    0.55,
    2.19,
    Some(0.14),
    128_000,
    8_192,
);

pub(super) const MODELS: &[Model] = &[
    DEEPSEEK_FLASH,
    DEEPSEEK_V4_PRO,
    DEEPSEEK_V4_FLASH,
    DEEPSEEK_V3,
    DEEPSEEK_R1,
];

#[cfg(test)]
mod tests {
    use crate::pricing::*;

    #[test]
    fn deepseek_entries_hold_peak_rates() {
        let flash = get_model("deepseek-flash").unwrap();
        assert_eq!(
            (flash.pricing.input_per_1m, flash.pricing.output_per_1m),
            (0.30, 1.20)
        );
        let pro = get_model("deepseek-v4-pro").unwrap();
        assert_eq!(
            (pro.pricing.input_per_1m, pro.pricing.output_per_1m),
            (1.32, 3.96)
        );
        // the legacy name bills as deepseek-flash
        let legacy = get_model("deepseek-v4-flash").unwrap();
        assert_eq!(legacy.pricing.input_per_1m, flash.pricing.input_per_1m);
    }
}
