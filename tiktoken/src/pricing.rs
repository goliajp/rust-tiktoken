//! Per-model pricing data and cost estimation for OpenAI, Anthropic, Google,
//! Meta, DeepSeek, Alibaba, Mistral, Moonshot, Zhipu, and MiniMax.
//!
//! Prices are in USD per 1M tokens. Updated as of 2026-08.
//! Pricing changes frequently — verify against official docs before production billing.
//!
//! # What changed in the 2026-08 refresh
//!
//! - **OpenAI**: the GPT-5.6 family (`gpt-5.6-sol` / `-terra` / `-luna`,
//!   launched 2026-07-09, Terra and Luna repriced down on 2026-07-30), the
//!   `-pro` SKUs, and the 5.1 / 5.2 point releases.
//! - **Anthropic**: added the Claude 5 generation (`claude-fable-5`,
//!   `claude-mythos-5`, `claude-opus-5`, `claude-sonnet-5`). Corrected the
//!   context window on Claude 4.6+ entries to 1M — those models carry the full
//!   window at standard rates, so there is no long-context tier to encode.
//! - **Google**: added `gemini-3.6-flash`, `gemini-3.5-flash-lite`, and
//!   `gemini-2.5-flash-lite`.
//! - **DeepSeek**: added `deepseek-v4-pro` / `deepseek-v4-flash`; `deepseek-v3`
//!   and `deepseek-r1` are now off the price card and marked DEPRECATED.
//! - **Alibaba**: added `qwen3.8-max` and `qwen3.5-plus`.
//! - **Mistral**: added the Devstral 2 and Ministral 3 SKUs; corrected
//!   `mistral-small` to the Small 4 rates ($0.15/$0.60).
//! - **Meta**: unchanged. Meta has wound down its first-party Llama API, so
//!   these entries remain third-party hoster rates (see the note below).
//! - **Moonshot / Zhipu / MiniMax** (added late 2026-08): first-party rates
//!   from platform.kimi.ai, docs.z.ai, and platform.minimax.io respectively.
//!   Max-output limits are not published by any of the three — those entries
//!   carry a conservative 32K placeholder.
//!
//! # Known gaps
//!
//! - `claude-sonnet-5` holds **introductory** pricing ($2/$10), which expires
//!   2026-08-31; standard pricing ($3/$15) takes effect 2026-09-01.
//! - DeepSeek announced on 2026-08-06 that it will raise prices "significantly"
//!   but has not published rates or an effective date.
//! - OpenAI's pricing page does not publish context windows for the `-pro`,
//!   `gpt-5.1`, and `gpt-5.2` SKUs; those entries mirror their family's window.
//! - Specialized OpenAI SKUs (`gpt-5.3-codex`, `gpt-5.5-cyber`, `*-chat-latest`,
//!   realtime / audio / image / video models) are not modelled here.
//! - Anthropic **Fast mode** ($10/$50 for Opus 5 / Opus 4.8) and the 1.1×
//!   `inference_geo: "us"` data-residency multiplier are not modelled.
//!
//! # Caveats
//!
//! - **Anthropic `cached_input`** uses the cache-READ price (≈ input × 10%). Older Claude 3.x
//!   entries store `input × 50%` (which corresponds to cache-write/2, not cache-read);
//!   they are kept as-is since those models are retired and no longer billable.
//! - **Meta llama prices** are pinned per-model to a specific hoster (DeepInfra or Groq)
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

/// Provider identity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        }
    }
}

/// Pricing tier per 1M tokens (USD).
#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    /// cost per 1M input tokens in USD
    pub input_per_1m: f64,
    /// cost per 1M output tokens in USD
    pub output_per_1m: f64,
    /// cost per 1M cached input tokens in USD (if supported)
    pub cached_input_per_1m: Option<f64>,
}

/// Per-1M-token rates for a specific OpenAI service tier (Standard, Batch,
/// Flex, Priority). Reused by [`ExtendedPricing::batch`], `.flex`, and
/// `.priority`.
///
/// Standard tier rates live directly on [`Pricing`] (the model's default).
#[derive(Debug, Clone, Copy)]
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
pub struct HighTierPricing {
    pub input_per_1m: f64,
    pub output_per_1m: f64,
    pub cached_input_per_1m: Option<f64>,
    /// switch to this tier when an estimate's input token count is *strictly greater than* this
    pub threshold_tokens: u32,
}

/// Audio modality pricing.
#[derive(Debug, Clone, Copy)]
pub struct AudioPricing {
    /// cost per 1M audio input tokens (Gemini 2.5 Flash uses this)
    pub input_per_1m: Option<f64>,
    /// cost per minute of audio (some providers bill this way)
    pub per_minute: Option<f64>,
}

/// OpenAI image-detail parameter. Other providers ignore it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageDetail {
    /// detail=low — caps tokens at the base/flat rate (OpenAI only).
    Low,
    /// detail=high (default) — full token count.
    #[default]
    High,
}

/// How a model converts an input image to billable input tokens.
///
/// All current providers charge image inputs at the model's standard
/// `input_per_1m` rate; this enum captures *how many tokens an image of
/// `(width, height)` resolves to* per provider's published formula. Use
/// [`Self::image_tokens`] to compute, or [`Model::estimate_image_cost`] for the
/// end-to-end USD figure.
#[derive(Debug, Clone, Copy)]
pub enum VisionPricing {
    /// OpenAI tile-based family (gpt-4o, gpt-4.1, o1, o3).
    ///
    /// Image is rescaled to fit a 2048×2048 box, then split into
    /// `tile_px`-edged tiles. Tokens =
    /// `base_tokens + ceil(w/tile) * ceil(h/tile) * per_tile_tokens`.
    /// `ImageDetail::Low` returns just `base_tokens`.
    OpenAITileBased {
        base_tokens: u32,
        per_tile_tokens: u32,
        tile_px: u32,
    },
    /// OpenAI patch-based family (gpt-4.1-mini, gpt-4.1-nano, o4-mini).
    ///
    /// Image is sampled into `patch_px`-edged patches; tokens ≈
    /// `round(patches × multiplier)`. `ImageDetail` has no effect on this family.
    OpenAIPatchBased { patch_px: u32, multiplier: f64 },
    /// Anthropic Claude vision.
    ///
    /// Tokens ≈ `(width * height) / divisor`, capped at `cap_tokens`.
    AnthropicDivisor { divisor: u32, cap_tokens: u32 },
    /// Google Gemini vision (2.5 Pro/Flash).
    ///
    /// Images with both edges ≤ `flat_threshold_px` charge a flat `flat_tokens`.
    /// Larger images: crop unit = `floor(min(w,h)/1.5)`, tile count =
    /// `ceil(w/crop) × ceil(h/crop)`, each tile = `tile_tokens`.
    GeminiTileBased {
        flat_threshold_px: u32,
        flat_tokens: u32,
        tile_tokens: u32,
    },
}

impl VisionPricing {
    /// Compute approximate input token count for an image of `width × height` px.
    pub fn image_tokens(&self, width: u32, height: u32, detail: ImageDetail) -> u64 {
        match *self {
            VisionPricing::OpenAITileBased {
                base_tokens,
                per_tile_tokens,
                tile_px,
            } => {
                if matches!(detail, ImageDetail::Low) {
                    return base_tokens as u64;
                }
                let (w, h) = rescale_to_fit(width, height, 2048);
                let tiles_w = w.div_ceil(tile_px) as u64;
                let tiles_h = h.div_ceil(tile_px) as u64;
                base_tokens as u64 + tiles_w * tiles_h * per_tile_tokens as u64
            }
            VisionPricing::OpenAIPatchBased {
                patch_px,
                multiplier,
            } => {
                let pw = width.div_ceil(patch_px) as f64;
                let ph = height.div_ceil(patch_px) as f64;
                (pw * ph * multiplier).round() as u64
            }
            VisionPricing::AnthropicDivisor {
                divisor,
                cap_tokens,
            } => {
                let raw = (width as u64 * height as u64) / divisor as u64;
                raw.min(cap_tokens as u64)
            }
            VisionPricing::GeminiTileBased {
                flat_threshold_px,
                flat_tokens,
                tile_tokens,
            } => {
                if width <= flat_threshold_px && height <= flat_threshold_px {
                    return flat_tokens as u64;
                }
                let crop_unit = (width.min(height) as f64 / 1.5).floor().max(1.0) as u32;
                let tiles_w = width.div_ceil(crop_unit) as u64;
                let tiles_h = height.div_ceil(crop_unit) as u64;
                tiles_w * tiles_h * tile_tokens as u64
            }
        }
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

/// Scale `(w, h)` down preserving aspect ratio so neither edge exceeds `max_edge`.
fn rescale_to_fit(w: u32, h: u32, max_edge: u32) -> (u32, u32) {
    if w <= max_edge && h <= max_edge {
        return (w, h);
    }
    let ratio = (w as f64 / max_edge as f64).max(h as f64 / max_edge as f64);
    (
        (w as f64 / ratio).round() as u32,
        (h as f64 / ratio).round() as u32,
    )
}

/// All extended (non-standard) pricing dimensions. Every field is `Option`;
/// `Default` produces an all-`None` value.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtendedPricing {
    /// Batch API tier (asynchronous fulfillment, typically ~50% off Standard).
    pub batch: Option<TierRates>,
    /// Flex tier (OpenAI 2026+; token rates equal Batch, but with explicit
    /// cached input row and narrower model availability).
    pub flex: Option<TierRates>,
    /// Priority tier (OpenAI 2026+; opt-in premium for faster fulfillment).
    /// Multiplier is NOT consistent across models — varies per SKU.
    pub priority: Option<TierRates>,
    pub high_tier: Option<HighTierPricing>,
    pub audio: Option<AudioPricing>,
    pub vision: Option<VisionPricing>,
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

    /// Estimate cost using Priority tier rates. Returns `None` if the model has no
    /// priority pricing set in `extended.priority`.
    pub fn estimate_priority_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let t = self.extended.priority?;
        Some(tier_cost(input_tokens, output_tokens, t))
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
/// # Examples
///
/// ```
/// let cost = tiktoken::pricing::estimate_cost("gpt-4o", 1_000, 1_000).unwrap();
/// assert!(cost > 0.0);
/// ```
pub fn estimate_cost(model_id: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    get_model(model_id).map(|m| m.estimate_cost(input_tokens, output_tokens))
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
    ALL_MODELS
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
            high_tier: None,
            audio: None,
            vision: None,
        },
        context_window: ctx,
        max_output: max_out,
    }
}

// ── OpenAI ──────────────────────────────────────────────

// GPT-5.x family (2026 generation). Standard rates from per-model docs pages;
// Batch / Flex / Priority rates from the aggregate /api/docs/pricing toggle.
// Long-context (>272K input) policy: Standard/Batch/Flex inputs multiply by 2×
// and outputs by 1.5× for gpt-5.4 and gpt-5.5 — encoded via `with_high_tier`.

// GPT-5.6 family (2026-07-09). The generation number is the family; Sol / Terra
// / Luna are durable capability tiers replacing the base / mini / nano names.
// Terra and Luna were repriced downward on 2026-07-30 (-20% / -80%); Sol held at
// its launch rate. Long-context (>272K input) rates are per the docs pricing page.

/// gpt-5.6-sol — frontier tier: complex reasoning, long-horizon agentic work.
const OPENAI_GPT56_SOL: Model = model(
    "gpt-5.6-sol",
    Provider::OpenAI,
    5.00,
    30.00,
    Some(0.50),
    1_050_000,
    128_000,
)
.with_batch_cached(2.50, 0.25, 15.00)
.with_high_tier(10.00, 45.00, None, 272_000);

/// gpt-5.6-terra — balanced tier: everyday coding, reasoning, agentic tasks.
const OPENAI_GPT56_TERRA: Model = model(
    "gpt-5.6-terra",
    Provider::OpenAI,
    2.00,
    12.00,
    Some(0.20),
    1_050_000,
    128_000,
)
.with_batch_cached(1.00, 0.10, 6.00)
.with_high_tier(4.00, 18.00, None, 272_000);

/// gpt-5.6-luna — high-volume tier: extraction, classification, reformatting.
const OPENAI_GPT56_LUNA: Model = model(
    "gpt-5.6-luna",
    Provider::OpenAI,
    0.20,
    1.20,
    Some(0.02),
    1_050_000,
    128_000,
)
.with_batch_cached(0.10, 0.01, 0.60)
.with_high_tier(0.40, 1.80, None, 272_000);

/// gpt-5.5 — frontier 2026 model. Has Priority tier (rare among GPT-5.x SKUs).
const OPENAI_GPT55: Model = model(
    "gpt-5.5",
    Provider::OpenAI,
    5.00,
    30.00,
    Some(0.50),
    1_050_000,
    128_000,
)
.with_batch_cached(2.50, 0.25, 15.00)
.with_flex(2.50, Some(0.25), 15.00)
.with_priority(12.50, Some(1.25), 75.00)
.with_high_tier(10.00, 45.00, None, 272_000);

/// gpt-5.4 — 2026 mid-frontier. Long-context multiplier (>272K) per docs.
const OPENAI_GPT54: Model = model(
    "gpt-5.4",
    Provider::OpenAI,
    2.50,
    15.00,
    Some(0.25),
    1_050_000,
    128_000,
)
.with_batch_cached(1.25, 0.125, 7.50)
.with_flex(1.25, None, 7.50)
.with_high_tier(5.00, 22.50, Some(0.50), 272_000);

/// gpt-5.4-mini — 2026 mid-tier. Has Flex; Priority partial data (skipped).
const OPENAI_GPT54_MINI: Model = model(
    "gpt-5.4-mini",
    Provider::OpenAI,
    0.75,
    4.50,
    Some(0.075),
    400_000,
    128_000,
)
.with_batch_cached(0.375, 0.0375, 2.25)
.with_flex(0.375, Some(0.0375), 2.25);

/// gpt-5 — 2025-08 launch SKU. Model docs page only lists Standard tier.
const OPENAI_GPT5: Model = model(
    "gpt-5",
    Provider::OpenAI,
    1.25,
    10.00,
    Some(0.125),
    400_000,
    128_000,
);

/// gpt-5-mini — 2025-08 launch. Standard only on docs page.
const OPENAI_GPT5_MINI: Model = model(
    "gpt-5-mini",
    Provider::OpenAI,
    0.25,
    2.00,
    Some(0.025),
    400_000,
    128_000,
);

/// gpt-5-nano — smallest GPT-5 SKU. Standard only; no Flex / Priority anywhere.
const OPENAI_GPT5_NANO: Model = model(
    "gpt-5-nano",
    Provider::OpenAI,
    0.05,
    0.40,
    Some(0.005),
    400_000,
    128_000,
);

// Reasoning-heavy "pro" SKUs and the intermediate 5.1 / 5.2 point releases.
// The docs pricing page lists rates but not context windows for these; each
// mirrors the window published for its family's base SKU (gpt-5 = 400K,
// gpt-5.4 / 5.5 = 1.05M). Verify before using the window for capacity planning.

/// gpt-5.5-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT55_PRO: Model = model(
    "gpt-5.5-pro",
    Provider::OpenAI,
    30.00,
    180.00,
    None,
    1_050_000,
    128_000,
)
.with_batch(15.00, 90.00);

/// gpt-5.4-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT54_PRO: Model = model(
    "gpt-5.4-pro",
    Provider::OpenAI,
    30.00,
    180.00,
    None,
    1_050_000,
    128_000,
)
.with_batch(15.00, 90.00);

/// gpt-5.4-nano — smallest 5.4 SKU.
const OPENAI_GPT54_NANO: Model = model(
    "gpt-5.4-nano",
    Provider::OpenAI,
    0.20,
    1.25,
    Some(0.02),
    400_000,
    128_000,
)
.with_batch_cached(0.10, 0.01, 0.625);

/// gpt-5.2 — 2025-12 point release.
const OPENAI_GPT52: Model = model(
    "gpt-5.2",
    Provider::OpenAI,
    1.75,
    14.00,
    Some(0.175),
    400_000,
    128_000,
)
.with_batch_cached(0.875, 0.0875, 7.00);

/// gpt-5.2-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT52_PRO: Model = model(
    "gpt-5.2-pro",
    Provider::OpenAI,
    21.00,
    168.00,
    None,
    400_000,
    128_000,
)
.with_batch(10.50, 84.00);

/// gpt-5.1 — 2025-11 point release; same rates as gpt-5.
const OPENAI_GPT51: Model = model(
    "gpt-5.1",
    Provider::OpenAI,
    1.25,
    10.00,
    Some(0.125),
    400_000,
    128_000,
)
.with_batch_cached(0.625, 0.0625, 5.00);

/// gpt-5-pro — extended-reasoning SKU. No cached-input rate published.
const OPENAI_GPT5_PRO: Model = model(
    "gpt-5-pro",
    Provider::OpenAI,
    15.00,
    120.00,
    None,
    400_000,
    128_000,
)
.with_batch(7.50, 60.00);

const OPENAI_GPT41: Model = model(
    "gpt-4.1",
    Provider::OpenAI,
    2.00,
    8.00,
    Some(0.50),
    1_000_000,
    32_768,
)
.with_batch(1.00, 4.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 85,
    per_tile_tokens: 170,
    tile_px: 512,
});

const OPENAI_GPT41_MINI: Model = model(
    "gpt-4.1-mini",
    Provider::OpenAI,
    0.40,
    1.60,
    Some(0.10),
    1_000_000,
    32_768,
)
.with_batch(0.20, 0.80)
.with_vision(VisionPricing::OpenAIPatchBased {
    patch_px: 32,
    multiplier: 1.62,
});

const OPENAI_GPT41_NANO: Model = model(
    "gpt-4.1-nano",
    Provider::OpenAI,
    0.10,
    0.40,
    Some(0.025),
    1_000_000,
    32_768,
)
.with_batch(0.05, 0.20)
.with_vision(VisionPricing::OpenAIPatchBased {
    patch_px: 32,
    multiplier: 2.46,
});

const OPENAI_GPT4O: Model = model(
    "gpt-4o",
    Provider::OpenAI,
    2.50,
    10.00,
    Some(1.25),
    128_000,
    16_384,
)
.with_batch(1.25, 5.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 85,
    per_tile_tokens: 170,
    tile_px: 512,
});

const OPENAI_GPT4O_MINI: Model = model(
    "gpt-4o-mini",
    Provider::OpenAI,
    0.15,
    0.60,
    Some(0.075),
    128_000,
    16_384,
)
.with_batch(0.075, 0.30)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 2833,
    per_tile_tokens: 5667,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
const OPENAI_O1: Model = model(
    "o1",
    Provider::OpenAI,
    15.00,
    60.00,
    Some(7.50),
    200_000,
    100_000,
)
.with_batch(7.50, 30.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 75,
    per_tile_tokens: 150,
    tile_px: 512,
});

/// DEPRECATED — API shutdown 2026-10-23. Price updated 2026-06: input/output
/// dropped from $3.00/$12.00 to match the o3-mini rate of $1.10/$4.40.
const OPENAI_O1_MINI: Model = model(
    "o1-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.55),
    128_000,
    65_536,
)
.with_batch(0.55, 2.20);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5-pro.
const OPENAI_O1_PRO: Model = model(
    "o1-pro",
    Provider::OpenAI,
    150.00,
    600.00,
    None,
    200_000,
    100_000,
);

const OPENAI_O3: Model = model(
    "o3",
    Provider::OpenAI,
    2.00,
    8.00,
    Some(0.50),
    200_000,
    100_000,
)
.with_batch(1.00, 4.00)
.with_vision(VisionPricing::OpenAITileBased {
    base_tokens: 75,
    per_tile_tokens: 150,
    tile_px: 512,
});

const OPENAI_O3_PRO: Model = model(
    "o3-pro",
    Provider::OpenAI,
    20.00,
    80.00,
    None,
    200_000,
    100_000,
)
.with_batch(10.00, 40.00);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
const OPENAI_O3_MINI: Model = model(
    "o3-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.55),
    200_000,
    100_000,
)
.with_batch(0.55, 2.20);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.4-mini.
const OPENAI_O4_MINI: Model = model(
    "o4-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.275),
    200_000,
    100_000,
)
.with_batch(0.55, 2.20)
.with_vision(VisionPricing::OpenAIPatchBased {
    patch_px: 32,
    multiplier: 1.72,
});

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
const OPENAI_GPT4_TURBO: Model = model(
    "gpt-4-turbo",
    Provider::OpenAI,
    10.00,
    30.00,
    None,
    128_000,
    4_096,
)
.with_batch(5.00, 15.00);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.5.
const OPENAI_GPT4: Model =
    model("gpt-4", Provider::OpenAI, 30.00, 60.00, None, 8_192, 8_192).with_batch(15.00, 30.00);

/// DEPRECATED — API shutdown 2026-10-23, replaced by gpt-5.4-mini.
const OPENAI_GPT35_TURBO: Model = model(
    "gpt-3.5-turbo",
    Provider::OpenAI,
    0.50,
    1.50,
    None,
    16_385,
    4_096,
)
.with_batch(0.25, 0.75);

const OPENAI_EMBED_3_SMALL: Model = model(
    "text-embedding-3-small",
    Provider::OpenAI,
    0.02,
    0.0,
    None,
    8_191,
    0,
);

const OPENAI_EMBED_3_LARGE: Model = model(
    "text-embedding-3-large",
    Provider::OpenAI,
    0.13,
    0.0,
    None,
    8_191,
    0,
);

const OPENAI_EMBED_ADA_002: Model = model(
    "text-embedding-ada-002",
    Provider::OpenAI,
    0.10,
    0.0,
    None,
    8_191,
    0,
);

// ── Anthropic Claude ────────────────────────────────────

// Opus 4.8 / 4.7 share Opus 4.x list pricing ($5/$25, cache-read $0.50, batch
// 2.50/12.50). Note: Opus 4.7+ use a new tokenizer (may emit up to ~35% more
// tokens for the same text) — this affects token counts, not the per-token rate.
// Claude 5 generation (2026). Claude 4.6 and later carry the full 1M-token
// context window at standard rates — there is no long-context tier to encode.
//
// `cached_input` is the cache-READ price (0.1× base input). Cache WRITES are
// 1.25× base (5-minute TTL) or 2× base (1-hour TTL) and are not modelled here.
//
// Tokenizer note: Claude 4.7 and later use a newer tokenizer that produces
// roughly 30% more tokens for the same text than Claude 4.6 and earlier. This
// crate ships no Anthropic tokenizer, so token counts for cost estimation must
// come from Anthropic's count-tokens endpoint, not from a tiktoken encoding.

/// claude-fable-5 — frontier tier.
const CLAUDE_FABLE_5: Model = model(
    "claude-fable-5",
    Provider::Anthropic,
    10.00,
    50.00,
    Some(1.00),
    1_000_000,
    128_000,
)
.with_batch(5.00, 25.00)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

/// claude-mythos-5 — frontier tier, limited availability. Same rates as Fable 5.
const CLAUDE_MYTHOS_5: Model = model(
    "claude-mythos-5",
    Provider::Anthropic,
    10.00,
    50.00,
    Some(1.00),
    1_000_000,
    128_000,
)
.with_batch(5.00, 25.00)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

/// claude-opus-5 — flagship. Fast mode (research preview) bills $10/$50 across
/// the full context window; that premium tier is not modelled here.
const CLAUDE_OPUS_5: Model = model(
    "claude-opus-5",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

/// claude-sonnet-5 — balanced tier.
///
/// Carries introductory pricing of $2/$10 through 2026-08-31; standard pricing
/// of $3/$15 takes effect 2026-09-01 (cache read $0.20 → $0.30, batch $1/$5 →
/// $1.50/$7.50). The entry below holds the introductory rates — update it when
/// the standard rates take effect.
const CLAUDE_SONNET_5: Model = model(
    "claude-sonnet-5",
    Provider::Anthropic,
    2.00,
    10.00,
    Some(0.20),
    1_000_000,
    128_000,
)
.with_batch(1.00, 5.00)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_OPUS_48: Model = model(
    "claude-opus-4.8",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_OPUS_47: Model = model(
    "claude-opus-4.7",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_OPUS_46: Model = model(
    "claude-opus-4.6",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    1_000_000,
    128_000,
)
.with_batch(2.50, 12.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_SONNET_46: Model = model(
    "claude-sonnet-4.6",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    1_000_000,
    64_000,
)
.with_batch(1.50, 7.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_HAIKU_45: Model = model(
    "claude-haiku-4.5",
    Provider::Anthropic,
    1.00,
    5.00,
    Some(0.10),
    200_000,
    64_000,
)
.with_batch(0.50, 2.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_OPUS_45: Model = model(
    "claude-opus-4.5",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    200_000,
    64_000,
)
.with_batch(2.50, 12.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

const CLAUDE_SONNET_45: Model = model(
    "claude-sonnet-4.5",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    200_000,
    64_000,
)
.with_batch(1.50, 7.50)
.with_vision(VisionPricing::AnthropicDivisor {
    divisor: 750,
    cap_tokens: 1568,
});

/// DEPRECATED — deprecated 2026-04-14, retires 2026-06-15. Replaced by
/// claude-opus-4.5 / 4.6 / 4.7 / 4.8. Cache price corrected 2026-06:
/// was $7.50 (cache-write/2 convention), now $1.50 (cache-read = input × 10%).
const CLAUDE_OPUS_4: Model = model(
    "claude-opus-4",
    Provider::Anthropic,
    15.00,
    75.00,
    Some(1.50),
    200_000,
    32_000,
);

/// DEPRECATED — deprecated 2026-04-14, retires 2026-06-15. Replaced by
/// claude-sonnet-4.5 / 4.6. Cache price corrected 2026-06:
/// was $1.50 (cache-write/2 convention), now $0.30 (cache-read = input × 10%).
const CLAUDE_SONNET_4: Model = model(
    "claude-sonnet-4",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    200_000,
    64_000,
);

/// DEPRECATED — retired 2026-02-19 on Anthropic-operated API
/// (still available on AWS Bedrock and Google Vertex AI).
const CLAUDE_HAIKU_35: Model = model(
    "claude-3.5-haiku",
    Provider::Anthropic,
    0.80,
    4.00,
    Some(0.40),
    200_000,
    8_192,
);

/// DEPRECATED — retired 2025-10-28.
const CLAUDE_SONNET_35: Model = model(
    "claude-3.5-sonnet",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(1.50),
    200_000,
    8_192,
);

/// DEPRECATED — retired 2026-01-05.
const CLAUDE_OPUS_3: Model = model(
    "claude-3-opus",
    Provider::Anthropic,
    15.00,
    75.00,
    Some(7.50),
    200_000,
    4_096,
);

/// DEPRECATED — retired 2026-04-20.
const CLAUDE_HAIKU_3: Model = model(
    "claude-3-haiku",
    Provider::Anthropic,
    0.25,
    1.25,
    Some(0.03),
    200_000,
    4_096,
);

// ── Google Gemini ───────────────────────────────────────

/// Standard tier is ≤200k input; high-tier auto-applies above that via `with_high_tier`.
// Gemini 3 series (current lineup). Pro uses context-tiered pricing (>200k) via
// with_high_tier, like 2.5 Pro. IDs follow the official pricing-page names.
const GEMINI_31_PRO: Model = model(
    "gemini-3.1-pro-preview",
    Provider::Google,
    2.00,
    12.00,
    Some(0.20),
    1_048_576,
    65_536,
)
.with_high_tier(4.00, 18.00, Some(0.40), 200_000)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// gemini-3.6-flash — current Flash generation (2026).
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

const GEMINI_25_PRO: Model = model(
    "gemini-2.5-pro",
    Provider::Google,
    1.25,
    10.00,
    Some(0.125),
    1_048_576,
    65_536,
)
.with_high_tier(2.50, 15.00, Some(0.25), 200_000)
.with_vision(VisionPricing::GeminiTileBased {
    flat_threshold_px: 384,
    flat_tokens: 258,
    tile_tokens: 258,
});

/// Text input is $0.30/M (standard); audio input is $1.00/M via `with_audio_input`.
const GEMINI_25_FLASH: Model = model(
    "gemini-2.5-flash",
    Provider::Google,
    0.30,
    2.50,
    Some(0.075),
    1_048_576,
    65_536,
)
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

// ── Meta (Llama via hosted APIs) ──────────────────────────
// Meta does not sell API access directly. Each model is pinned to a specific
// hoster's current serverless inference price as of 2026-06; the source URL
// is recorded per entry. If your billing uses a different hoster, adjust
// at the call site.

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

/// Source: DeepInfra serverless inference (https://deepinfra.com/pricing) 2026-06.
const META_LLAMA_3_1_8B: Model = model(
    "llama-3.1-8b",
    Provider::Meta,
    0.02,
    0.05,
    None,
    128_000,
    4_096,
);

/// Source: DeepInfra serverless inference (Llama-3.3-70B-Instruct-Turbo)
/// https://deepinfra.com/pricing — 2026-06.
const META_LLAMA_3_3_70B: Model = model(
    "llama-3.3-70b",
    Provider::Meta,
    0.10,
    0.32,
    None,
    128_000,
    4_096,
);

/// Source: Groq (https://groq.com/pricing) 2026-06.
const META_LLAMA_4_SCOUT: Model = model(
    "llama-4-scout",
    Provider::Meta,
    0.11,
    0.34,
    None,
    10_000_000,
    8_192,
);

/// Source: DeepInfra (Llama-4-Maverick-17B-128E-Instruct-FP8)
/// https://deepinfra.com/pricing — 2026-06.
const META_LLAMA_4_MAVERICK: Model = model(
    "llama-4-maverick",
    Provider::Meta,
    0.15,
    0.60,
    None,
    1_000_000,
    8_192,
);

// ── DeepSeek ─────────────────────────────────────────────

// DeepSeek announced on 2026-08-06 that it plans to raise API pricing
// "significantly" in the near future; no new rates or effective date have been
// published, so the rates below remain the active card.

/// deepseek-v4-pro — current flagship.
const DEEPSEEK_V4_PRO: Model = model(
    "deepseek-v4-pro",
    Provider::DeepSeek,
    0.435,
    0.87,
    None,
    1_000_000,
    8_192,
);

/// deepseek-v4-flash — current high-volume SKU.
const DEEPSEEK_V4_FLASH: Model = model(
    "deepseek-v4-flash",
    Provider::DeepSeek,
    0.14,
    0.28,
    None,
    1_000_000,
    8_192,
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

// ── Alibaba (Qwen) ──────────────────────────────────────
// USD prices from Alibaba Cloud Model Studio (International).

/// qwen3.8-max — current flagship (2026-08). Flat rate across the full 1M
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

/// qwen3.5-plus — current mid-tier.
const QWEN_3_5_PLUS: Model = model(
    "qwen3.5-plus",
    Provider::Alibaba,
    0.40,
    2.40,
    None,
    1_000_000,
    32_768,
);

const QWEN_2_5_72B: Model = model(
    "qwen2.5-72b",
    Provider::Alibaba,
    1.4,
    5.6,
    None,
    128_000,
    8_192,
);

const QWEN_2_5_32B: Model = model(
    "qwen2.5-32b",
    Provider::Alibaba,
    0.7,
    2.8,
    None,
    128_000,
    8_192,
);

const QWEN_2_5_7B: Model = model(
    "qwen2.5-7b",
    Provider::Alibaba,
    0.175,
    0.7,
    None,
    128_000,
    8_192,
);

const QWEN_3_MAX: Model = model(
    "qwen3-max",
    Provider::Alibaba,
    1.20,
    6.00,
    None,
    262_144,
    8_192,
);

const QWEN_3_PLUS: Model = model(
    "qwen3-plus",
    Provider::Alibaba,
    0.40,
    2.4,
    None,
    128_000,
    8_192,
);

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

const QWEN_3_8B: Model = model(
    "qwen3-8b",
    Provider::Alibaba,
    0.18,
    0.7,
    None,
    128_000,
    8_192,
);

// ── Mistral ─────────────────────────────────────────────

const MISTRAL_LARGE: Model = model(
    "mistral-large",
    Provider::Mistral,
    0.5,
    1.5,
    None,
    128_000,
    4_096,
);

/// mistral-small — `mistral-small-latest` (Small 4) is $0.15/$0.60 as of 2026-08.
const MISTRAL_SMALL: Model = model(
    "mistral-small",
    Provider::Mistral,
    0.15,
    0.60,
    None,
    128_000,
    4_096,
);

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

/// ministral-14b-latest (Ministral 3 · 14B).
const MINISTRAL_14B: Model = model(
    "ministral-14b",
    Provider::Mistral,
    0.20,
    0.20,
    None,
    128_000,
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

const MISTRAL_MEDIUM: Model = model(
    "mistral-medium",
    Provider::Mistral,
    1.5,
    7.5,
    None,
    128_000,
    4_096,
);

const CODESTRAL: Model = model(
    "codestral",
    Provider::Mistral,
    0.30,
    0.90,
    None,
    256_000,
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

const MIXTRAL_8X7B: Model = model(
    "mixtral-8x7b",
    Provider::Mistral,
    0.7,
    0.7,
    None,
    32_768,
    4_096,
);

// ── Moonshot (Kimi) ─────────────────────────────────────
// USD rates from the official Moonshot API price card (platform.kimi.ai),
// read 2026-08-08. Max-output limits are not published on the price card;
// entries carry a conservative 32K placeholder — verify before capacity
// planning.

/// kimi-k3 — flagship (2026-07-16). Flat rate across the full 1M context.
const KIMI_K3_MODEL: Model = model(
    "kimi-k3",
    Provider::Moonshot,
    3.00,
    15.00,
    Some(0.30),
    1_048_576,
    32_768,
);

/// kimi-k2.7-code — dedicated coding SKU.
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

/// kimi-k2.6 — vision + text.
const KIMI_K26: Model = model(
    "kimi-k2.6",
    Provider::Moonshot,
    0.95,
    4.00,
    None,
    262_144,
    32_768,
)
.with_batch(0.57, 2.40);

/// kimi-k2.5 — previous mid-tier.
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

// ── Zhipu (GLM / Z.ai) ──────────────────────────────────
// USD rates from the official Z.ai pricing page (docs.z.ai), read 2026-08-08.
// Context windows from the model release notes (5.2 = 1M; 4.x = 128K/200K);
// max-output not published — 32K placeholder.

/// glm-5.2 — flagship (2026-06-16, MIT open weights).
const GLM_52: Model = model(
    "glm-5.2",
    Provider::Zhipu,
    1.40,
    4.40,
    Some(0.26),
    1_048_576,
    32_768,
);

/// glm-5 — previous flagship (200K context per release notes).
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

// ── MiniMax ─────────────────────────────────────────────
// USD rates read directly from the official pay-as-you-go card
// (platform.minimax.io/docs/guides/pricing-paygo), 2026-08-08. The M2 family
// shares one tokenizer; `-highspeed` variants double the rate for faster
// serving and are not modelled. Context window per the M2 release (204,800);
// max-output not published — 32K placeholder.

/// minimax-m2.7 — current mainline.
const MINIMAX_M27: Model = model(
    "minimax-m2.7",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.06),
    204_800,
    32_768,
);

/// minimax-m2.5
const MINIMAX_M25: Model = model(
    "minimax-m2.5",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.03),
    204_800,
    32_768,
);

/// minimax-m2.1
const MINIMAX_M21: Model = model(
    "minimax-m2.1",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.03),
    204_800,
    32_768,
);

/// minimax-m2 — original M2 (2025-10).
const MINIMAX_M2_MODEL: Model = model(
    "minimax-m2",
    Provider::MiniMax,
    0.30,
    1.20,
    Some(0.03),
    204_800,
    32_768,
);

// ── Master list ─────────────────────────────────────────

static ALL_MODELS: &[Model] = &[
    // OpenAI
    OPENAI_GPT56_SOL,
    OPENAI_GPT56_TERRA,
    OPENAI_GPT56_LUNA,
    OPENAI_GPT55,
    OPENAI_GPT55_PRO,
    OPENAI_GPT54,
    OPENAI_GPT54_MINI,
    OPENAI_GPT54_NANO,
    OPENAI_GPT54_PRO,
    OPENAI_GPT52,
    OPENAI_GPT52_PRO,
    OPENAI_GPT51,
    OPENAI_GPT5,
    OPENAI_GPT5_MINI,
    OPENAI_GPT5_NANO,
    OPENAI_GPT5_PRO,
    OPENAI_GPT41,
    OPENAI_GPT41_MINI,
    OPENAI_GPT41_NANO,
    OPENAI_GPT4O,
    OPENAI_GPT4O_MINI,
    OPENAI_O1,
    OPENAI_O1_MINI,
    OPENAI_O1_PRO,
    OPENAI_O3,
    OPENAI_O3_PRO,
    OPENAI_O3_MINI,
    OPENAI_O4_MINI,
    OPENAI_GPT4_TURBO,
    OPENAI_GPT4,
    OPENAI_GPT35_TURBO,
    OPENAI_EMBED_3_SMALL,
    OPENAI_EMBED_3_LARGE,
    OPENAI_EMBED_ADA_002,
    // Anthropic
    CLAUDE_FABLE_5,
    CLAUDE_MYTHOS_5,
    CLAUDE_OPUS_5,
    CLAUDE_SONNET_5,
    CLAUDE_OPUS_48,
    CLAUDE_OPUS_47,
    CLAUDE_OPUS_46,
    CLAUDE_SONNET_46,
    CLAUDE_HAIKU_45,
    CLAUDE_OPUS_45,
    CLAUDE_SONNET_45,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    CLAUDE_HAIKU_35,
    CLAUDE_SONNET_35,
    CLAUDE_OPUS_3,
    CLAUDE_HAIKU_3,
    // Google
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
    // Meta
    META_LLAMA_4_SCOUT,
    META_LLAMA_4_MAVERICK,
    META_LLAMA_3_1_405B,
    META_LLAMA_3_1_70B,
    META_LLAMA_3_1_8B,
    META_LLAMA_3_3_70B,
    // DeepSeek
    DEEPSEEK_V4_PRO,
    DEEPSEEK_V4_FLASH,
    DEEPSEEK_V3,
    DEEPSEEK_R1,
    // Alibaba
    QWEN_3_8_MAX,
    QWEN_3_5_PLUS,
    QWEN_3_MAX,
    QWEN_3_PLUS,
    QWEN_3_CODER,
    QWEN_3_8B,
    QWEN_2_5_72B,
    QWEN_2_5_32B,
    QWEN_2_5_7B,
    // Mistral
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
    // Moonshot
    KIMI_K3_MODEL,
    KIMI_K27_CODE,
    KIMI_K26,
    KIMI_K25,
    // Zhipu
    GLM_52,
    GLM_5_MODEL,
    GLM_47,
    GLM_45,
    GLM_45_AIR,
    // MiniMax
    MINIMAX_M27,
    MINIMAX_M25,
    MINIMAX_M21,
    MINIMAX_M2_MODEL,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_cost_openai_gpt41() {
        let m = get_model("gpt-4.1").unwrap();
        let standard = m.estimate_cost(1_000_000, 500_000);
        let batch = m.estimate_batch_cost(1_000_000, 500_000).unwrap();
        assert!(
            (batch - standard / 2.0).abs() < 0.001,
            "batch should be ~50% of standard: standard={standard}, batch={batch}",
        );
    }

    #[test]
    fn test_batch_cost_not_supported_for_o1_pro() {
        let m = get_model("o1-pro").unwrap();
        assert!(m.estimate_batch_cost(1, 1).is_none());
    }

    #[test]
    fn test_gemini_25_pro_high_tier_threshold() {
        let m = get_model("gemini-2.5-pro").unwrap();
        let low = m.pricing_for_input(100_000);
        let high = m.pricing_for_input(300_000);
        assert_eq!(low.input_per_1m, 1.25);
        assert_eq!(low.output_per_1m, 10.00);
        assert_eq!(high.input_per_1m, 2.50);
        assert_eq!(high.output_per_1m, 15.00);
    }

    #[test]
    fn test_gemini_25_pro_estimate_cost_auto_switches_tier() {
        let m = get_model("gemini-2.5-pro").unwrap();
        // 300k input tokens → high-tier $2.50/M, not standard $1.25/M.
        let cost = m.estimate_cost(300_000, 0);
        let expected = 300_000.0 * 2.50 / 1_000_000.0;
        assert!(
            (cost - expected).abs() < 0.001,
            "expected {expected}, got {cost}",
        );
    }

    #[test]
    fn test_gemini_25_flash_audio_input() {
        let m = get_model("gemini-2.5-flash").unwrap();
        let audio = m.extended.audio.expect("audio pricing should be set");
        assert_eq!(audio.input_per_1m, Some(1.00));
        // standard text input rate unchanged
        assert_eq!(m.pricing.input_per_1m, 0.30);
    }

    #[test]
    fn test_existing_model_no_extended_pricing_unchanged() {
        let m = get_model("mistral-large").unwrap();
        assert!(m.extended.batch.is_none());
        assert!(m.extended.high_tier.is_none());
        assert!(m.extended.audio.is_none());
        assert!(m.extended.vision.is_none());
    }

    // ── Vision ──────────────────────────────────────────────

    #[test]
    fn test_vision_openai_tile_low_detail_returns_base() {
        let m = get_model("gpt-4o").unwrap();
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(2048, 2048, ImageDetail::Low);
        assert_eq!(toks, 85);
    }

    #[test]
    fn test_vision_openai_tile_high_detail_1024x1024() {
        let m = get_model("gpt-4o").unwrap();
        // 1024 fits the 2048 box. ceil(1024/512)=2 tiles per side → 4 tiles total.
        // 85 + 4*170 = 765.
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(1024, 1024, ImageDetail::High);
        assert_eq!(toks, 765);
    }

    #[test]
    fn test_vision_openai_patch_mini_64x64() {
        let m = get_model("gpt-4.1-mini").unwrap();
        // ceil(64/32)^2 = 4 patches * 1.62 = 6.48 → 6 tokens.
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(64, 64, ImageDetail::High);
        assert_eq!(toks, 6);
    }

    #[test]
    fn test_vision_anthropic_divisor_under_cap() {
        let m = get_model("claude-haiku-4.5").unwrap();
        // 1000 × 750 = 750_000; / 750 = 1000 tokens (well under the 1568 cap).
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(1000, 750, ImageDetail::High);
        assert_eq!(toks, 1000);
    }

    #[test]
    fn test_vision_anthropic_cap_applies_for_large_image() {
        let m = get_model("claude-sonnet-4.6").unwrap();
        // 4000 × 4000 = 16_000_000; / 750 = 21_333 → capped at 1568.
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(4000, 4000, ImageDetail::High);
        assert_eq!(toks, 1568);
    }

    #[test]
    fn test_vision_gemini_flat_under_threshold() {
        let m = get_model("gemini-2.5-pro").unwrap();
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(100, 100, ImageDetail::High);
        assert_eq!(toks, 258);
    }

    #[test]
    fn test_vision_gemini_tile_960x540_matches_official_example() {
        // From ai.google.dev/gemini-api/docs/image-understanding:
        // 960×540 → 1548 tokens (6 tiles × 258).
        let m = get_model("gemini-2.5-flash").unwrap();
        let toks = m
            .extended
            .vision
            .unwrap()
            .image_tokens(960, 540, ImageDetail::High);
        assert_eq!(toks, 1548);
    }

    #[test]
    fn test_estimate_image_cost_haiku() {
        let m = get_model("claude-haiku-4.5").unwrap();
        // 1000×750 → 1000 tokens × $1/M = $0.001
        let cost = m.estimate_image_cost(1000, 750, ImageDetail::High).unwrap();
        assert!((cost - 0.001).abs() < 1e-6, "expected ~0.001, got {cost}",);
    }

    #[test]
    fn test_estimate_image_cost_none_for_pixtral_large() {
        // pixtral-large has no verified vision formula — should return None.
        let m = get_model("pixtral-large").unwrap();
        assert!(
            m.estimate_image_cost(1024, 1024, ImageDetail::High)
                .is_none()
        );
    }

    // ── GPT-5.x + Flex / Priority tiers ─────────────────────

    #[test]
    fn test_gpt5_family_registered() {
        for id in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
        ] {
            assert!(get_model(id).is_some(), "missing GPT-5.x model {id}");
        }
    }

    #[test]
    fn test_gpt55_priority_cost_matches_doc_rates() {
        let m = get_model("gpt-5.5").unwrap();
        // Priority: $12.50 input + $75.00 output per 1M.
        let cost = m.estimate_priority_cost(1_000_000, 500_000).unwrap();
        assert!(
            (cost - (12.50 + 37.50)).abs() < 1e-6,
            "expected 50.00, got {cost}",
        );
    }

    #[test]
    fn test_gpt55_flex_equals_batch_token_rates() {
        // OpenAI Flex guide: Flex token rates equal Batch token rates.
        let m = get_model("gpt-5.5").unwrap();
        let f = m.estimate_flex_cost(1_000_000, 500_000).unwrap();
        let b = m.estimate_batch_cost(1_000_000, 500_000).unwrap();
        assert!((f - b).abs() < 1e-9, "flex {f} should equal batch {b}");
    }

    #[test]
    fn test_gpt5_has_no_flex_or_priority() {
        let m = get_model("gpt-5").unwrap();
        assert!(m.extended.flex.is_none());
        assert!(m.extended.priority.is_none());
        // Estimators must surface that absence.
        assert!(m.estimate_flex_cost(1, 1).is_none());
        assert!(m.estimate_priority_cost(1, 1).is_none());
    }

    #[test]
    fn test_gpt54_long_context_auto_applies_high_tier() {
        let m = get_model("gpt-5.4").unwrap();
        // Standard input $2.50; long-context >272K input $5.00.
        let standard = m.estimate_cost(100_000, 0);
        let long = m.estimate_cost(300_000, 0);
        // 100k × $2.50/M = $0.25; 300k × $5.00/M = $1.50.
        assert!((standard - 0.25).abs() < 1e-9);
        assert!((long - 1.50).abs() < 1e-9);
    }

    #[test]
    fn test_gpt54_flex_cached_is_none_per_docs() {
        // Aggregate pricing page shows "—" for gpt-5.4 Flex cached. Keep None.
        let m = get_model("gpt-5.4").unwrap();
        let flex = m.extended.flex.unwrap();
        assert_eq!(flex.cached_input_per_1m, None);
    }

    #[test]
    fn test_get_model_openai() {
        let m = get_model("gpt-4o").unwrap();
        assert_eq!(m.provider, Provider::OpenAI);
        assert!(m.pricing.input_per_1m > 0.0);
    }

    #[test]
    fn test_get_model_claude() {
        let m = get_model("claude-opus-4").unwrap();
        assert_eq!(m.provider, Provider::Anthropic);
    }

    #[test]
    fn test_get_model_gemini() {
        let m = get_model("gemini-2.5-pro").unwrap();
        assert_eq!(m.provider, Provider::Google);
    }

    #[test]
    fn test_get_model_deepseek() {
        let m = get_model("deepseek-v3").unwrap();
        assert_eq!(m.provider, Provider::DeepSeek);
        assert!(m.pricing.cached_input_per_1m.is_some());
    }

    #[test]
    fn test_get_model_llama() {
        let m = get_model("llama-3.1-70b").unwrap();
        assert_eq!(m.provider, Provider::Meta);
    }

    #[test]
    fn test_get_model_qwen() {
        let m = get_model("qwen2.5-72b").unwrap();
        assert_eq!(m.provider, Provider::Alibaba);
    }

    #[test]
    fn test_get_model_mistral() {
        let m = get_model("mistral-large").unwrap();
        assert_eq!(m.provider, Provider::Mistral);
    }

    #[test]
    fn test_get_model_unknown() {
        assert!(get_model("nonexistent").is_none());
    }

    #[test]
    fn test_estimate_cost_gpt4o() {
        let cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000).unwrap();
        // $2.50 input + $10.00 output = $12.50
        assert!((cost - 12.50).abs() < 0.001);
    }

    #[test]
    fn test_estimate_cost_zero_tokens() {
        let cost = estimate_cost("gpt-4o", 0, 0).unwrap();
        assert!((cost).abs() < 0.0001);
    }

    #[test]
    fn test_estimate_cost_with_cache() {
        let m = get_model("gpt-4o").unwrap();
        let cost = m.estimate_cost_with_cache(500_000, 500_000, 1_000_000);
        // input: 500k * 2.50/1M = 1.25
        // cached: 500k * 1.25/1M = 0.625
        // output: 1M * 10.00/1M = 10.00
        let expected = 1.25 + 0.625 + 10.00;
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn test_models_by_provider() {
        let openai = models_by_provider(Provider::OpenAI);
        assert!(openai.len() >= 10);
        let claude = models_by_provider(Provider::Anthropic);
        assert!(claude.len() >= 4);
        let google = models_by_provider(Provider::Google);
        assert!(google.len() >= 4);
        let meta = models_by_provider(Provider::Meta);
        assert!(meta.len() >= 3);
        let deepseek = models_by_provider(Provider::DeepSeek);
        assert!(deepseek.len() >= 2);
        let alibaba = models_by_provider(Provider::Alibaba);
        assert!(alibaba.len() >= 3);
        let mistral = models_by_provider(Provider::Mistral);
        assert!(mistral.len() >= 3);
    }

    #[test]
    fn test_embed_model_zero_output_price() {
        let m = get_model("text-embedding-3-small").unwrap();
        assert!(m.pricing.output_per_1m == 0.0);
        assert_eq!(m.max_output, 0);
    }

    #[test]
    fn test_all_models_have_positive_context() {
        for m in ALL_MODELS {
            assert!(m.context_window > 0, "{} has zero context window", m.id);
        }
    }

    #[test]
    fn test_case_insensitive_lookup() {
        assert!(get_model("GPT-4o").is_some());
        assert!(get_model("Claude-Opus-4").is_some());
        assert!(get_model("GEMINI-2.5-PRO").is_some());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::OpenAI.to_string(), "OpenAI");
        assert_eq!(Provider::Anthropic.to_string(), "Anthropic");
        assert_eq!(Provider::Google.to_string(), "Google");
        assert_eq!(Provider::Meta.to_string(), "Meta");
        assert_eq!(Provider::DeepSeek.to_string(), "DeepSeek");
        assert_eq!(Provider::Alibaba.to_string(), "Alibaba");
        assert_eq!(Provider::Mistral.to_string(), "Mistral");
    }

    #[test]
    fn test_estimate_cost_with_cache_no_cache_support() {
        let m = get_model("gpt-4").unwrap();
        assert!(m.pricing.cached_input_per_1m.is_none());
        // without cache support, cached tokens charged at normal rate
        let cost = m.estimate_cost_with_cache(500_000, 500_000, 500_000);
        let expected = m.estimate_cost(1_000_000, 500_000);
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn test_estimate_cost_unknown_model() {
        assert!(estimate_cost("nonexistent-model", 1000, 1000).is_none());
    }

    #[test]
    fn test_deepseek_cache_pricing() {
        let m = get_model("deepseek-v3").unwrap();
        assert!(m.pricing.cached_input_per_1m.is_some());
        let cached = m.pricing.cached_input_per_1m.unwrap();
        assert!(cached < m.pricing.input_per_1m);
        let cost = m.estimate_cost_with_cache(500_000, 500_000, 100_000);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_deepseek_r1_pricing() {
        let m = get_model("deepseek-r1").unwrap();
        assert_eq!(m.provider, Provider::DeepSeek);
        assert!(m.pricing.input_per_1m > 0.0);
        assert!(m.pricing.output_per_1m > m.pricing.input_per_1m);
    }

    #[test]
    fn test_llama_models_no_cache() {
        for id in [
            "llama-3.1-405b",
            "llama-3.1-70b",
            "llama-3.1-8b",
            "llama-3.3-70b",
            "llama-4-scout",
            "llama-4-maverick",
        ] {
            let m = get_model(id).unwrap();
            assert_eq!(m.provider, Provider::Meta, "wrong provider for {id}");
            assert!(
                m.pricing.cached_input_per_1m.is_none(),
                "unexpected cache for {id}"
            );
        }
    }

    #[test]
    fn test_qwen_models() {
        for id in [
            "qwen2.5-72b",
            "qwen2.5-32b",
            "qwen2.5-7b",
            "qwen3-max",
            "qwen3-plus",
            "qwen3-coder",
            "qwen3-8b",
        ] {
            let m = get_model(id).unwrap();
            assert_eq!(m.provider, Provider::Alibaba, "wrong provider for {id}");
        }
    }

    #[test]
    fn test_mistral_models() {
        for id in [
            "mistral-large",
            "mistral-medium",
            "mistral-small",
            "mistral-nemo",
            "codestral",
            "pixtral-large",
            "mixtral-8x7b",
        ] {
            let m = get_model(id).unwrap();
            assert_eq!(m.provider, Provider::Mistral, "wrong provider for {id}");
        }
    }

    #[test]
    fn test_new_openai_models() {
        for id in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3-pro"] {
            let m = get_model(id).unwrap();
            assert_eq!(m.provider, Provider::OpenAI, "wrong provider for {id}");
        }
    }

    #[test]
    fn test_new_claude_models() {
        for id in [
            "claude-opus-4.6",
            "claude-sonnet-4.6",
            "claude-haiku-4.5",
            "claude-opus-4.5",
            "claude-sonnet-4.5",
        ] {
            let m = get_model(id).unwrap();
            assert_eq!(m.provider, Provider::Anthropic, "wrong provider for {id}");
        }
    }

    #[test]
    fn test_all_models_unique_ids() {
        let mut ids: Vec<&str> = ALL_MODELS.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), ALL_MODELS.len(), "duplicate model ids found");
    }

    #[test]
    fn test_all_models_non_negative_pricing() {
        for m in ALL_MODELS {
            assert!(
                m.pricing.input_per_1m >= 0.0,
                "{} has negative input price",
                m.id
            );
            assert!(
                m.pricing.output_per_1m >= 0.0,
                "{} has negative output price",
                m.id
            );
            if let Some(cached) = m.pricing.cached_input_per_1m {
                assert!(cached >= 0.0, "{} has negative cached price", m.id);
                assert!(
                    cached <= m.pricing.input_per_1m,
                    "{} cached price exceeds input price",
                    m.id
                );
            }
        }
    }

    #[test]
    fn test_estimate_cost_every_model() {
        for m in ALL_MODELS {
            let cost = estimate_cost(m.id, 1000, 1000).unwrap();
            assert!(cost >= 0.0, "{} produced negative cost", m.id);
        }
    }

    #[test]
    fn test_all_providers_have_models() {
        let providers = [
            Provider::OpenAI,
            Provider::Anthropic,
            Provider::Google,
            Provider::Meta,
            Provider::DeepSeek,
            Provider::Alibaba,
            Provider::Mistral,
        ];
        for p in providers {
            assert!(!models_by_provider(p).is_empty(), "{p} has no models");
        }
    }

    #[test]
    fn test_max_output_within_context() {
        for m in ALL_MODELS {
            // embedding models have max_output = 0, skip those
            if m.max_output == 0 {
                continue;
            }
            assert!(
                m.max_output <= m.context_window,
                "{}: max_output {} > context_window {}",
                m.id,
                m.max_output,
                m.context_window,
            );
        }
    }

    #[test]
    fn test_cache_price_leq_normal() {
        for m in ALL_MODELS {
            if let Some(cached) = m.pricing.cached_input_per_1m {
                assert!(
                    cached <= m.pricing.input_per_1m,
                    "{}: cached_input {cached} > input {}",
                    m.id,
                    m.pricing.input_per_1m,
                );
            }
        }
    }

    #[test]
    fn test_models_by_provider_exhaustive() {
        let total: usize = [
            Provider::OpenAI,
            Provider::Anthropic,
            Provider::Google,
            Provider::Meta,
            Provider::DeepSeek,
            Provider::Alibaba,
            Provider::Mistral,
            Provider::Moonshot,
            Provider::Zhipu,
            Provider::MiniMax,
        ]
        .iter()
        .map(|p| models_by_provider(*p).len())
        .sum();
        assert_eq!(
            total,
            ALL_MODELS.len(),
            "provider counts don't sum to total"
        );
    }
}
