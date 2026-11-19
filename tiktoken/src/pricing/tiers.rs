//! Pricing beyond the standard rate: service tiers, long-context tiers, audio,
//! cache writes and background windows, and the builders that set them.

use super::*;

/// Per-1M-token rates for one service tier. Reused by
/// [`ExtendedPricing::batch`], `.flex`, `.priority` and `.ultrafast`.
///
/// Standard tier rates live directly on [`Pricing`] (the model's default).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TierRates {
    /// cost per 1M input tokens in USD for this tier
    pub input_per_1m: f64,
    /// cost per 1M cached input tokens in USD for this tier (if the tier supports
    /// prompt caching). Batch tier typically lacks this; Flex and Priority have it.
    pub cached_input_per_1m: Option<f64>,
    /// cost per 1M output tokens in USD for this tier
    pub output_per_1m: f64,
}

/// Alias kept for backwards compatibility with the 3.2.x API. The `cached_input_per_1m`
/// field is unused by existing Batch data; new entries can fill it.
pub type BatchPricing = TierRates;

/// High-tier pricing for input-token-count-based tiers (e.g. Gemini 2.5 Pro).
/// Applies when input tokens exceed `threshold_tokens`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct HighTierPricing {
    pub input_per_1m: f64,
    pub output_per_1m: f64,
    pub cached_input_per_1m: Option<f64>,
    /// switch to this tier when an estimate's input token count is *strictly greater than* this
    pub threshold_tokens: u32,
}

/// Audio modality pricing.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AudioPricing {
    /// cost per 1M audio input tokens (Gemini 2.5 Flash uses this)
    pub input_per_1m: Option<f64>,
    /// cost per minute of audio (some providers bill this way)
    pub per_minute: Option<f64>,
}

/// The price of writing input into the prompt cache for one cache lifetime.
///
/// Providers that charge for cache writes bill them per lifetime: Anthropic,
/// Moonshot and Alibaba offer 5 minutes and 1 hour, OpenAI 30 minutes and 12
/// hours. Reads are [`Pricing::cached_input_per_1m`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CacheWrite {
    /// how long the written entry stays cached
    pub ttl_minutes: u32,
    /// cost per 1M tokens written, in USD
    pub per_1m: f64,
}

impl CacheWrite {
    pub const fn new(ttl_minutes: u32, per_1m: f64) -> Self {
        Self {
            ttl_minutes,
            per_1m,
        }
    }
}

/// A delayed-completion window billed at a published discount off the
/// standard token rates (OpenAI's background windows). The discount is stored
/// as the provider states it, not as derived rates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BackgroundWindow {
    /// how long the provider may take to complete the request
    pub hours: u32,
    /// fraction off the standard rate: 0.30 is 30% below standard
    pub discount: f64,
}

impl BackgroundWindow {
    pub const fn new(hours: u32, discount: f64) -> Self {
        Self { hours, discount }
    }
}

/// Compute USD cost for `(input_tokens, output_tokens)` at a given tier's
/// rates. Helper used by `estimate_batch_cost`, `estimate_flex_cost`, and
/// `estimate_priority_cost`. The cached input rate is not used here; callers
/// that need cache-aware estimates can split tokens upstream.
fn tier_cost(input_tokens: u64, output_tokens: u64, t: TierRates) -> f64 {
    input_tokens as f64 * t.input_per_1m / 1_000_000.0
        + output_tokens as f64 * t.output_per_1m / 1_000_000.0
}

/// All extended (non-standard) pricing dimensions. `Default` leaves every one
/// unset.
///
/// `#[non_exhaustive]`: providers keep adding ways to bill, and each becomes a
/// field here.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ExtendedPricing {
    /// Batch API tier (asynchronous fulfillment, typically ~50% off Standard).
    pub batch: Option<TierRates>,
    /// Flex tier (OpenAI 2026+; token rates equal Batch, but with explicit
    /// cached input row and narrower model availability).
    pub flex: Option<TierRates>,
    /// The provider's premium-speed tier: OpenAI Priority (renamed Fast on
    /// 2026-07-30), Anthropic Fast mode, MiniMax priority. The multiplier over
    /// standard is not consistent across models — varies per SKU.
    pub priority: Option<TierRates>,
    /// OpenAI's Ultrafast tier, above Fast.
    pub ultrafast: Option<TierRates>,
    pub high_tier: Option<HighTierPricing>,
    pub audio: Option<AudioPricing>,
    pub vision: Option<VisionPricing>,
    /// Cache-write prices, one per cache lifetime the provider offers. Empty
    /// when writes are free or the price is unpublished.
    pub cache_writes: &'static [CacheWrite],
    /// Delayed-completion windows and their discounts.
    pub background: &'static [BackgroundWindow],
}

impl Model {
    /// Estimate cost using batch-API rates. Returns `None` if the model has no
    /// batch pricing set in `extended.batch`.
    pub fn estimate_batch_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let t = self.extended.batch?;
        Some(tier_cost(input_tokens, output_tokens, t))
    }

    /// Estimate cost using Flex tier rates. Returns `None` if the model has no
    /// flex pricing set in `extended.flex`.
    pub fn estimate_flex_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let t = self.extended.flex?;
        Some(tier_cost(input_tokens, output_tokens, t))
    }

    /// Estimate cost using Ultrafast tier rates. Returns `None` if the model has
    /// no `extended.ultrafast`.
    pub fn estimate_ultrafast_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let t = self.extended.ultrafast?;
        Some(tier_cost(input_tokens, output_tokens, t))
    }

    /// Price per 1M tokens of a cache write with the given lifetime, or `None`
    /// when the model lists no write price for that lifetime.
    pub fn cache_write_per_1m(&self, ttl_minutes: u32) -> Option<f64> {
        self.extended
            .cache_writes
            .iter()
            .find(|w| w.ttl_minutes == ttl_minutes)
            .map(|w| w.per_1m)
    }

    /// Estimate cost when the request runs in a background window of `hours`:
    /// the standard estimate (long-context tier included) less the window's
    /// discount. `None` if the model offers no such window.
    pub fn estimate_background_cost(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        hours: u32,
    ) -> Option<f64> {
        let w = self.extended.background.iter().find(|w| w.hours == hours)?;
        Some(self.estimate_cost(input_tokens, output_tokens) * (1.0 - w.discount))
    }

    /// Estimate cost using Priority tier rates. Returns `None` if the model has no
    /// priority pricing set in `extended.priority`.
    pub fn estimate_priority_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let t = self.extended.priority?;
        Some(tier_cost(input_tokens, output_tokens, t))
    }

    /// Set the Ultrafast tier rates.
    pub const fn with_ultrafast(
        mut self,
        input_per_1m: f64,
        cached_input_per_1m: Option<f64>,
        output_per_1m: f64,
    ) -> Self {
        self.extended.ultrafast = Some(TierRates {
            input_per_1m,
            cached_input_per_1m,
            output_per_1m,
        });
        self
    }

    /// Set the cache-write prices, one per lifetime.
    pub const fn with_cache_writes(mut self, writes: &'static [CacheWrite]) -> Self {
        self.extended.cache_writes = writes;
        self
    }

    /// Set the background windows.
    pub const fn with_background(mut self, windows: &'static [BackgroundWindow]) -> Self {
        self.extended.background = windows;
        self
    }

    /// Set the batch-API pricing. Returns a new `Model`. The cached input rate
    /// defaults to `None`; use [`Self::with_batch_cached`] when the tier ships
    /// an explicit cached rate.
    pub const fn with_batch(mut self, input_per_1m: f64, output_per_1m: f64) -> Self {
        self.extended.batch = Some(TierRates {
            input_per_1m,
            cached_input_per_1m: None,
            output_per_1m,
        });
        self
    }

    /// Set the batch-API pricing with an explicit cached-input rate.
    pub const fn with_batch_cached(
        mut self,
        input_per_1m: f64,
        cached_input_per_1m: f64,
        output_per_1m: f64,
    ) -> Self {
        self.extended.batch = Some(TierRates {
            input_per_1m,
            cached_input_per_1m: Some(cached_input_per_1m),
            output_per_1m,
        });
        self
    }

    /// Set the Flex tier rates. Pass `cached_input_per_1m: None` if the docs
    /// page shows `—` for the cached column.
    pub const fn with_flex(
        mut self,
        input_per_1m: f64,
        cached_input_per_1m: Option<f64>,
        output_per_1m: f64,
    ) -> Self {
        self.extended.flex = Some(TierRates {
            input_per_1m,
            cached_input_per_1m,
            output_per_1m,
        });
        self
    }

    /// Set the Priority tier rates. Pass `cached_input_per_1m: None` if the docs
    /// page shows `—`. OpenAI Priority is opt-in and offered on a narrow subset
    /// of models — leave unset for models without a Priority row.
    pub const fn with_priority(
        mut self,
        input_per_1m: f64,
        cached_input_per_1m: Option<f64>,
        output_per_1m: f64,
    ) -> Self {
        self.extended.priority = Some(TierRates {
            input_per_1m,
            cached_input_per_1m,
            output_per_1m,
        });
        self
    }

    /// Set the high-tier (input-token-count-based) pricing. Returns a new `Model`.
    pub const fn with_high_tier(
        mut self,
        input_per_1m: f64,
        output_per_1m: f64,
        cached_input_per_1m: Option<f64>,
        threshold_tokens: u32,
    ) -> Self {
        self.extended.high_tier = Some(HighTierPricing {
            input_per_1m,
            output_per_1m,
            cached_input_per_1m,
            threshold_tokens,
        });
        self
    }

    /// Set the audio-input per-1M-tokens rate. Returns a new `Model`.
    pub const fn with_audio_input(mut self, per_1m: f64) -> Self {
        let prev = match self.extended.audio {
            Some(a) => a,
            None => AudioPricing {
                input_per_1m: None,
                per_minute: None,
            },
        };
        self.extended.audio = Some(AudioPricing {
            input_per_1m: Some(per_1m),
            per_minute: prev.per_minute,
        });
        self
    }

    /// Set the vision pricing variant. Returns a new `Model`.
    pub const fn with_vision(mut self, v: VisionPricing) -> Self {
        self.extended.vision = Some(v);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::pricing::*;

    #[test]
    fn cache_write_rates_by_lifetime() {
        let opus = get_model("claude-opus-5.5").unwrap();
        assert_eq!(opus.cache_write_per_1m(5), Some(5.0));
        assert_eq!(opus.cache_write_per_1m(60), Some(8.0));
        assert_eq!(opus.cache_write_per_1m(30), None);
        let sol = get_model("gpt-6-sol").unwrap();
        assert_eq!(sol.cache_write_per_1m(30), Some(2.5));
        assert_eq!(sol.cache_write_per_1m(720), Some(5.0));
        let kimi = get_model("kimi-k3.1-pro").unwrap();
        assert_eq!(kimi.cache_write_per_1m(60), Some(4.0));
        // providers whose writes are free or unpublished list none
        assert!(
            get_model("gemini-4-neon")
                .unwrap()
                .extended
                .cache_writes
                .is_empty()
        );
    }

    #[test]
    fn cache_writes_cost_more_than_reads_and_grow_with_lifetime() {
        for m in all_models() {
            let writes = m.extended.cache_writes;
            for pair in writes.windows(2) {
                assert!(pair[0].ttl_minutes < pair[1].ttl_minutes, "{}", m.id);
                assert!(pair[0].per_1m <= pair[1].per_1m, "{}", m.id);
            }
            if let (Some(w), Some(read)) = (writes.first(), m.pricing.cached_input_per_1m) {
                assert!(
                    w.per_1m > read,
                    "{}: write {} <= read {read}",
                    m.id,
                    w.per_1m
                );
            }
        }
    }

    #[test]
    fn ultrafast_and_fast_mode_costs() {
        let astra = get_model("gpt-6-astra").unwrap();
        let cost = astra.estimate_ultrafast_cost(1_000_000, 1_000_000).unwrap();
        assert!((cost - 360.0).abs() < 1e-9, "{cost}");
        assert!(
            get_model("gpt-6-luna")
                .unwrap()
                .estimate_ultrafast_cost(1, 1)
                .is_none()
        );
        // Anthropic Fast mode lives in `priority`
        let fable = get_model("claude-fable-5.2").unwrap();
        let fast = fable.estimate_priority_cost(1_000_000, 1_000_000).unwrap();
        assert!((fast - 96.0).abs() < 1e-9, "{fast}");
    }

    #[test]
    fn background_windows_discount_the_standard_estimate() {
        let luna = get_model("gpt-6-luna").unwrap();
        let standard = luna.estimate_cost(1_000_000, 1_000_000);
        let two_h = luna
            .estimate_background_cost(1_000_000, 1_000_000, 2)
            .unwrap();
        let eight_h = luna
            .estimate_background_cost(1_000_000, 1_000_000, 8)
            .unwrap();
        assert!((two_h - standard * 0.70).abs() < 1e-12);
        assert!((eight_h - standard * 0.60).abs() < 1e-12);
        assert!(luna.estimate_background_cost(1, 1, 4).is_none());
        // only the four SKUs OpenAI names get the windows
        assert!(
            get_model("gpt-6-sol")
                .unwrap()
                .extended
                .background
                .is_empty()
        );
    }
}
