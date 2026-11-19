//! Per-model pricing data and cost estimation for OpenAI, Anthropic, Google,
//! Meta, DeepSeek, Alibaba, Mistral, Moonshot, Zhipu, MiniMax, and Voyage.
//!
//! Prices are in USD per 1M tokens. Updated as of 2026-11-19.
//! Pricing changes frequently — verify against official docs before production billing.
//! Each provider's table lives in its own file under `pricing/`, with its sources.
//!
//! # What changed in the 2026-11 refresh
//!
//! - **New billing dimensions** on [`ExtendedPricing`]: cache-write prices per
//!   lifetime (`cache_writes`), OpenAI's Ultrafast tier (`ultrafast`) and
//!   background windows (`background`). Anthropic Fast mode is stored as
//!   `priority`, next to OpenAI's Fast tier.
//! - **OpenAI**: the GPT-6 family (`gpt-6-astra` / `-sol` / `-luna`,
//!   `gpt-6.1-sol` / `-terra`, `gpt-6.2-astra` / `-sol`), with Batch, Flex where
//!   offered, Fast (stored as `priority`) and long-context rates. Flex and Fast
//!   for `gpt-5.6-terra` / `-luna`. Shutdown dates for GPT-5, GPT-5.1,
//!   GPT-5.4-nano, the o3 SKUs and the GPT-4.1 / GPT-4o snapshots.
//! - **Anthropic**: Claude Fable / Mythos 5.1 and 5.2, Opus 5.5, Sonnet 5.5,
//!   Haiku 5.5 (whose prompts over 100K tokens use `with_high_tier`). Sonnet 5's
//!   $2 / $10 became its standard price. Image tokens use Anthropic's current
//!   28-px patch formula, [`VisionPricing::AnthropicPatch`].
//! - **Google**: Gemini 4 (`gemini-4-neon` / `-hydrogen` / `-carbon` /
//!   `-argon`), `gemini-3.8-flash`, `gemini-embedding-2`. Gemini 2.5 Pro /
//!   Flash and 3.5 Flash are deprecated; `gemini-2.5-flash` cached input
//!   corrected to $0.03.
//! - **DeepSeek**: `deepseek-flash` (V4.1); `deepseek-v4-pro` repriced on
//!   2026-08-16; peak / off-peak billing (see the DeepSeek file).
//! - **Alibaba**: `qwen4-max`, `qwen3.8-flash`, `qwen3.8-27b`,
//!   `qwen3.8-2.4t-a95b`; `qwen3.5-plus` gains its >256K tier.
//! - **Mistral**: `mistral-large-4`, `mistral-embed`, `codestral-embed`; Large 3,
//!   Medium 3.5 and Devstral deprecated; context windows corrected for
//!   `codestral`, `mistral-small` and `ministral-14b`.
//! - **Meta**: Muse Spark Open and Muse Glimmer 30B (Together AI);
//!   `llama-4-scout` moved to DeepInfra; `llama-4-maverick` deprecated.
//! - **Moonshot / Zhipu / MiniMax**: `kimi-k3.1-pro` (+ `-highspeed`),
//!   `glm-5.3` / `-flashx` / `-flash`, `minimax-m3` / `-m3.1-flash`;
//!   `kimi-k2.5` and the MiniMax M2 family deprecated.
//!
//! # Known gaps
//!
//! - **Time-limited prices**: entries hold the price that applies once a
//!   promotion or introductory period ends, where the provider publishes it.
//!   Lower rates in force on 2026-11-19: `gpt-5.6-sol` $4 / $20 until
//!   2026-11-30 (OpenAI has not said what follows, so the entry keeps the
//!   launch rate $5 / $30); `gemini-4-argon` / `-carbon` $2 / $10 until
//!   2026-11-30; `gemini-3.8-flash` / `gemini-3.6-flash` $0.75 / $3.75 until
//!   2026-12-31.
//! - **DeepSeek** bills off-peak hours at half the peak rate; entries hold the
//!   peak rate.
//! - OpenAI's pricing page does not publish context windows for the `-pro`,
//!   `gpt-5.1`, and `gpt-5.2` SKUs; those entries mirror their family's window.
//! - Specialized OpenAI SKUs (`gpt-5.3-codex`, `*-cyber`, `gpt-6-rosalind`,
//!   `*-law`, `*-chat-latest`, realtime / audio / image / video models) are not
//!   in the table.
//! - Cache writes are modelled per lifetime (`extended.cache_writes`) but only
//!   at the standard rate: OpenAI's long-context writes (2×) and Claude Haiku
//!   5.5's long-prompt writes are not, and `gpt-5.6-sol` publishes only
//!   promotional write rates, so it has none.
//! - Anthropic's 1.1× `inference_geo: "us"` data-residency multiplier is not
//!   modelled.
//! - Gemini 4 and 3.8 Flash carry no image-token formula.

//! # Caveats
//!
//! - **Anthropic `cached_input`** uses the cache-READ price (≈ input × 10%). Older Claude 3.x
//!   entries store `input × 50%` (which corresponds to cache-write/2, not cache-read);
//!   they are kept as-is since those models are retired and no longer billable.
//! - **Meta prices** are pinned per-model to a specific hoster (DeepInfra or Together)
//!   with the source URL recorded next to each entry. Adjust at the call site if your
//!   billing uses a different hoster. `llama-3.1-405b` / `llama-3.1-70b` are marked
//!   DEPRECATED because no major hoster still offers them as serverless inference.
//! - **`gemini-2.5-pro`** uses `with_high_tier(...)` for the >200k token tier; `estimate_cost`
//!   auto-switches when input tokens exceed the 200k threshold. The standard `pricing` field
//!   holds the ≤200k rates.
//! - Models marked `DEPRECATED` retain historical prices so existing cost lookups keep
//!   working. The note records when the API shuts down or removes the model.
//! - **Vision pricing** is modelled as `VisionPricing` enum variants per provider; image
//!   inputs are billed via the model's standard `input_per_1m` rate. Use
//!   `Model::estimate_image_cost` for the end-to-end USD figure. `pixtral-large` /
//!   `o1-pro` / `o3-pro` / `gpt-4-turbo` are left UNSET because their formulas are not
//!   listed verbatim in the providers' current docs (deferred to a separate pass).
//! - **OpenAI service tiers** (Standard / Batch / Flex / Priority) share a unified
//!   `TierRates` shape. Flex token rates equal Batch by OpenAI policy but each tier
//!   keeps its own `cached_input_per_1m` and availability; do not assume Flex == Batch
//!   for new models. Priority multipliers vary per model (gpt-5.5 is 2.5× standard,
//!   not the often-cited 5×); store raw values, never compute. Some GPT-5.x SKUs lack
//!   Priority — `extended.priority` is `Option<TierRates>`. Long-context (>272K input)
//!   for gpt-5.4 / gpt-5.5 is encoded via `with_high_tier` and auto-applied by
//!   `estimate_cost` / `pricing_for_input`.

use std::sync::LazyLock;

mod alibaba;
mod anthropic;
mod deepseek;
mod google;
mod meta;
mod minimax;
mod mistral;
mod moonshot;
mod openai;
mod openai_legacy;
mod resolve;
mod tiers;
mod vision;
mod voyage;
mod zhipu;

pub use resolve::{Match, Resolved, resolve_model};
pub use tiers::{
    AudioPricing, BackgroundWindow, BatchPricing, CacheWrite, ExtendedPricing, HighTierPricing,
    TierRates,
};
pub use vision::{ImageDetail, VisionPricing};

/// Provider identity
///
/// `#[non_exhaustive]`: the set of vendors in the price table grows, and a
/// `match` over it that must be edited for each new one turns every table
/// refresh into a breaking release. Match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    Meta,
    DeepSeek,
    Alibaba,
    Mistral,
    Moonshot,
    Zhipu,
    MiniMax,
    Voyage,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI => write!(f, "OpenAI"),
            Self::Anthropic => write!(f, "Anthropic"),
            Self::Google => write!(f, "Google"),
            Self::Meta => write!(f, "Meta"),
            Self::DeepSeek => write!(f, "DeepSeek"),
            Self::Alibaba => write!(f, "Alibaba"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Moonshot => write!(f, "Moonshot"),
            Self::Zhipu => write!(f, "Zhipu"),
            Self::MiniMax => write!(f, "MiniMax"),
            Self::Voyage => write!(f, "Voyage"),
        }
    }
}

/// Pricing tier per 1M tokens (USD).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Pricing {
    /// cost per 1M input tokens in USD
    pub input_per_1m: f64,
    /// cost per 1M output tokens in USD
    pub output_per_1m: f64,
    /// cost per 1M cached input tokens in USD (if supported)
    pub cached_input_per_1m: Option<f64>,
}

/// Model metadata including pricing, context window, and output limits.
///
/// Marked `#[non_exhaustive]` to allow future field additions without breaking SemVer;
/// construct via the internal `model()` helper plus `with_*` builders.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Model {
    /// model identifier (e.g. `"gpt-4o"`, `"claude-opus-4"`)
    pub id: &'static str,
    /// which provider this model belongs to
    pub provider: Provider,
    /// standard (low-tier) per-token pricing
    pub pricing: Pricing,
    /// optional extended dimensions: batch, high-tier, audio, vision
    pub extended: ExtendedPricing,
    /// maximum input context window in tokens
    pub context_window: u32,
    /// maximum output tokens per request
    pub max_output: u32,
}

impl Model {
    /// Pricing to apply for a given input token count.
    ///
    /// Returns high-tier pricing when an `extended.high_tier` is set and the
    /// total input token count strictly exceeds its `threshold_tokens`; otherwise
    /// returns the standard `pricing`.
    pub fn pricing_for_input(&self, input_tokens: u64) -> Pricing {
        if let Some(ht) = self.extended.high_tier
            && input_tokens > ht.threshold_tokens as u64
        {
            Pricing {
                input_per_1m: ht.input_per_1m,
                output_per_1m: ht.output_per_1m,
                cached_input_per_1m: ht.cached_input_per_1m,
            }
        } else {
            self.pricing
        }
    }

    /// Estimate cost in USD for a given number of input and output tokens.
    ///
    /// Auto-switches to high-tier pricing when applicable (see
    /// [`Self::pricing_for_input`]).
    ///
    /// # Examples
    ///
    /// ```
    /// let model = tiktoken::pricing::get_model("gpt-4o").unwrap();
    /// let cost = model.estimate_cost(1_000_000, 500_000);
    /// assert!((cost - 7.50).abs() < 0.001); // $2.50 input + $5.00 output
    /// ```
    pub fn estimate_cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let p = self.pricing_for_input(input_tokens);
        let input_cost = input_tokens as f64 * p.input_per_1m / 1_000_000.0;
        let output_cost = output_tokens as f64 * p.output_per_1m / 1_000_000.0;
        input_cost + output_cost
    }

    /// Estimate cost with cached input tokens.
    ///
    /// `input_tokens` are charged at the normal rate, `cached_tokens` at the
    /// discounted cached rate (falls back to normal rate if caching is not available).
    /// Auto-switches to high-tier pricing when `input_tokens + cached_tokens`
    /// exceeds the high-tier threshold.
    pub fn estimate_cost_with_cache(
        &self,
        input_tokens: u64,
        cached_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let p = self.pricing_for_input(input_tokens + cached_tokens);
        let cached_rate = p.cached_input_per_1m.unwrap_or(p.input_per_1m);
        let input_cost = input_tokens as f64 * p.input_per_1m / 1_000_000.0;
        let cached_cost = cached_tokens as f64 * cached_rate / 1_000_000.0;
        let output_cost = output_tokens as f64 * p.output_per_1m / 1_000_000.0;
        input_cost + cached_cost + output_cost
    }

    /// Estimate the USD cost for one image input, end-to-end
    /// (image → tokens → input rate).
    ///
    /// Returns `None` when the model has no `extended.vision` set.
    /// Image tokens count toward the `pricing_for_input` tier-selection threshold,
    /// so the rate auto-switches to high-tier for `gemini-2.5-pro` above 200k.
    pub fn estimate_image_cost(&self, width: u32, height: u32, detail: ImageDetail) -> Option<f64> {
        let v = self.extended.vision?;
        let tokens = v.image_tokens(width, height, detail);
        let p = self.pricing_for_input(tokens);
        Some(tokens as f64 * p.input_per_1m / 1_000_000.0)
    }
}

/// Look up a model by id. Case-insensitive.
///
/// # Examples
///
/// ```
/// let model = tiktoken::pricing::get_model("gpt-4o").unwrap();
/// assert_eq!(model.provider, tiktoken::pricing::Provider::OpenAI);
/// assert_eq!(model.context_window, 128_000);
/// ```
pub fn get_model(id: &str) -> Option<&'static Model> {
    ALL_MODELS.iter().find(|m| m.id.eq_ignore_ascii_case(id))
}

/// Estimate cost for a model by name. Returns `None` for unknown models.
///
/// Ids are resolved with [`resolve_model`], so a model addressed the way its
/// API spells it — `claude-haiku-4-5`, `gpt-4o-2024-08-06`,
/// `anthropic.claude-opus-5` — prices as that model rather than returning
/// `None`. Use [`get_model`] when you need an exact match.
///
/// # Examples
///
/// ```
/// let cost = tiktoken::pricing::estimate_cost("gpt-4o", 1_000, 1_000).unwrap();
/// assert!(cost > 0.0);
///
/// // the same model as its API spells it
/// assert!(tiktoken::pricing::estimate_cost("claude-haiku-4-5", 1_000, 1_000).is_some());
/// ```
pub fn estimate_cost(model_id: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    resolve_model(model_id).map(|r| r.model.estimate_cost(input_tokens, output_tokens))
}

/// List all available models.
///
/// # Examples
///
/// ```
/// let models = tiktoken::pricing::all_models();
/// assert!(models.len() >= 100);
/// ```
pub fn all_models() -> &'static [Model] {
    &ALL_MODELS
}

/// List all models for a given provider.
///
/// # Examples
///
/// ```
/// use tiktoken::pricing::{models_by_provider, Provider};
/// let openai_models = models_by_provider(Provider::OpenAI);
/// assert!(openai_models.len() >= 10);
/// ```
pub fn models_by_provider(provider: Provider) -> Vec<&'static Model> {
    ALL_MODELS
        .iter()
        .filter(|m| m.provider == provider)
        .collect()
}

// helper
const fn model(
    id: &'static str,
    provider: Provider,
    input: f64,
    output: f64,
    cached: Option<f64>,
    ctx: u32,
    max_out: u32,
) -> Model {
    Model {
        id,
        provider,
        pricing: Pricing {
            input_per_1m: input,
            output_per_1m: output,
            cached_input_per_1m: cached,
        },
        extended: ExtendedPricing {
            batch: None,
            flex: None,
            priority: None,
            ultrafast: None,
            high_tier: None,
            audio: None,
            vision: None,
            cache_writes: &[],
            background: &[],
        },
        context_window: ctx,
        max_output: max_out,
    }
}

static ALL_MODELS: LazyLock<Vec<Model>> = LazyLock::new(|| {
    [
        openai::MODELS,
        anthropic::MODELS,
        google::MODELS,
        voyage::MODELS,
        meta::MODELS,
        deepseek::MODELS,
        alibaba::MODELS,
        mistral::MODELS,
        moonshot::MODELS,
        zhipu::MODELS,
        minimax::MODELS,
    ]
    .concat()
});

#[cfg(test)]
mod tests;
