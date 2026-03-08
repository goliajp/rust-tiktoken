//! Per-model pricing data and cost estimation for OpenAI, Anthropic, Google, Meta, DeepSeek, Alibaba, and Mistral.
//!
//! Prices are in USD per 1M tokens. Updated as of 2026-03.
//! Pricing changes frequently — verify against official docs before production billing.

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

/// Model metadata including pricing, context window, and output limits.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// model identifier (e.g. `"gpt-4o"`, `"claude-opus-4"`)
    pub id: &'static str,
    /// which provider this model belongs to
    pub provider: Provider,
    /// per-token pricing
    pub pricing: Pricing,
    /// maximum input context window in tokens
    pub context_window: u32,
    /// maximum output tokens per request
    pub max_output: u32,
}

impl Model {
    /// Estimate cost in USD for a given number of input and output tokens.
    ///
    /// # Examples
    ///
    /// ```
    /// let model = tiktoken::pricing::get_model("gpt-4o").unwrap();
    /// let cost = model.estimate_cost(1_000_000, 500_000);
    /// assert!((cost - 7.50).abs() < 0.001); // $2.50 input + $5.00 output
    /// ```
    pub fn estimate_cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = input_tokens as f64 * self.pricing.input_per_1m / 1_000_000.0;
        let output_cost = output_tokens as f64 * self.pricing.output_per_1m / 1_000_000.0;
        input_cost + output_cost
    }

    /// Estimate cost with cached input tokens.
    ///
    /// `input_tokens` are charged at the normal rate, `cached_tokens` at the
    /// discounted cached rate (falls back to normal rate if caching is not available).
    pub fn estimate_cost_with_cache(
        &self,
        input_tokens: u64,
        cached_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let cached_rate = self
            .pricing
            .cached_input_per_1m
            .unwrap_or(self.pricing.input_per_1m);
        let input_cost = input_tokens as f64 * self.pricing.input_per_1m / 1_000_000.0;
        let cached_cost = cached_tokens as f64 * cached_rate / 1_000_000.0;
        let output_cost = output_tokens as f64 * self.pricing.output_per_1m / 1_000_000.0;
        input_cost + cached_cost + output_cost
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
/// assert!(models.len() >= 57);
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
        context_window: ctx,
        max_output: max_out,
    }
}

// ── OpenAI ──────────────────────────────────────────────

const OPENAI_GPT41: Model = model(
    "gpt-4.1",
    Provider::OpenAI,
    2.00,
    8.00,
    Some(0.50),
    1_000_000,
    32_768,
);

const OPENAI_GPT41_MINI: Model = model(
    "gpt-4.1-mini",
    Provider::OpenAI,
    0.40,
    1.60,
    Some(0.10),
    1_000_000,
    32_768,
);

const OPENAI_GPT41_NANO: Model = model(
    "gpt-4.1-nano",
    Provider::OpenAI,
    0.10,
    0.40,
    Some(0.025),
    1_000_000,
    32_768,
);

const OPENAI_GPT4O: Model = model(
    "gpt-4o",
    Provider::OpenAI,
    2.50,
    10.00,
    Some(1.25),
    128_000,
    16_384,
);

const OPENAI_GPT4O_MINI: Model = model(
    "gpt-4o-mini",
    Provider::OpenAI,
    0.15,
    0.60,
    Some(0.075),
    128_000,
    16_384,
);

const OPENAI_O1: Model = model(
    "o1",
    Provider::OpenAI,
    15.00,
    60.00,
    Some(7.50),
    200_000,
    100_000,
);

const OPENAI_O1_MINI: Model = model(
    "o1-mini",
    Provider::OpenAI,
    3.00,
    12.00,
    Some(1.50),
    128_000,
    65_536,
);

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
);

const OPENAI_O3_PRO: Model = model(
    "o3-pro",
    Provider::OpenAI,
    20.00,
    80.00,
    None,
    200_000,
    100_000,
);

const OPENAI_O3_MINI: Model = model(
    "o3-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.55),
    200_000,
    100_000,
);

const OPENAI_O4_MINI: Model = model(
    "o4-mini",
    Provider::OpenAI,
    1.10,
    4.40,
    Some(0.275),
    200_000,
    100_000,
);

const OPENAI_GPT4_TURBO: Model = model(
    "gpt-4-turbo",
    Provider::OpenAI,
    10.00,
    30.00,
    None,
    128_000,
    4_096,
);

const OPENAI_GPT4: Model = model("gpt-4", Provider::OpenAI, 30.00, 60.00, None, 8_192, 8_192);

const OPENAI_GPT35_TURBO: Model = model(
    "gpt-3.5-turbo",
    Provider::OpenAI,
    0.50,
    1.50,
    None,
    16_385,
    4_096,
);

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

const CLAUDE_OPUS_46: Model = model(
    "claude-opus-4.6",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    200_000,
    128_000,
);

const CLAUDE_SONNET_46: Model = model(
    "claude-sonnet-4.6",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    200_000,
    64_000,
);

const CLAUDE_HAIKU_45: Model = model(
    "claude-haiku-4.5",
    Provider::Anthropic,
    1.00,
    5.00,
    Some(0.10),
    200_000,
    64_000,
);

const CLAUDE_OPUS_45: Model = model(
    "claude-opus-4.5",
    Provider::Anthropic,
    5.00,
    25.00,
    Some(0.50),
    200_000,
    64_000,
);

const CLAUDE_SONNET_45: Model = model(
    "claude-sonnet-4.5",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(0.30),
    200_000,
    64_000,
);

const CLAUDE_OPUS_4: Model = model(
    "claude-opus-4",
    Provider::Anthropic,
    15.00,
    75.00,
    Some(7.50),
    200_000,
    32_000,
);

const CLAUDE_SONNET_4: Model = model(
    "claude-sonnet-4",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(1.50),
    200_000,
    64_000,
);

const CLAUDE_HAIKU_35: Model = model(
    "claude-3.5-haiku",
    Provider::Anthropic,
    0.80,
    4.00,
    Some(0.40),
    200_000,
    8_192,
);

const CLAUDE_SONNET_35: Model = model(
    "claude-3.5-sonnet",
    Provider::Anthropic,
    3.00,
    15.00,
    Some(1.50),
    200_000,
    8_192,
);

const CLAUDE_OPUS_3: Model = model(
    "claude-3-opus",
    Provider::Anthropic,
    15.00,
    75.00,
    Some(7.50),
    200_000,
    4_096,
);

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

const GEMINI_25_PRO: Model = model(
    "gemini-2.5-pro",
    Provider::Google,
    1.25,
    10.00,
    Some(0.3125),
    1_048_576,
    65_536,
);

const GEMINI_25_FLASH: Model = model(
    "gemini-2.5-flash",
    Provider::Google,
    0.30,
    2.50,
    Some(0.075),
    1_048_576,
    65_536,
);

const GEMINI_20_FLASH: Model = model(
    "gemini-2.0-flash",
    Provider::Google,
    0.10,
    0.40,
    Some(0.025),
    1_048_576,
    8_192,
);

const GEMINI_15_PRO: Model = model(
    "gemini-1.5-pro",
    Provider::Google,
    1.25,
    5.00,
    Some(0.3125),
    2_097_152,
    8_192,
);

const GEMINI_15_FLASH: Model = model(
    "gemini-1.5-flash",
    Provider::Google,
    0.075,
    0.30,
    Some(0.01875),
    1_048_576,
    8_192,
);

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
// pricing based on common API providers (Together, Fireworks, etc.)

const META_LLAMA_3_1_405B: Model = model(
    "llama-3.1-405b",
    Provider::Meta,
    3.00,
    3.00,
    None,
    128_000,
    4_096,
);

const META_LLAMA_3_1_70B: Model = model(
    "llama-3.1-70b",
    Provider::Meta,
    0.88,
    0.88,
    None,
    128_000,
    4_096,
);

const META_LLAMA_3_1_8B: Model = model(
    "llama-3.1-8b",
    Provider::Meta,
    0.18,
    0.18,
    None,
    128_000,
    4_096,
);

const META_LLAMA_3_3_70B: Model = model(
    "llama-3.3-70b",
    Provider::Meta,
    0.88,
    0.88,
    None,
    128_000,
    4_096,
);

const META_LLAMA_4_SCOUT: Model = model(
    "llama-4-scout",
    Provider::Meta,
    0.11,
    0.34,
    None,
    10_000_000,
    8_192,
);

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

const DEEPSEEK_V3: Model = model(
    "deepseek-v3",
    Provider::DeepSeek,
    0.27,
    1.10,
    Some(0.07),
    128_000,
    8_192,
);

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

const QWEN_2_5_72B: Model = model(
    "qwen2.5-72b",
    Provider::Alibaba,
    0.90,
    0.90,
    None,
    128_000,
    8_192,
);

const QWEN_2_5_32B: Model = model(
    "qwen2.5-32b",
    Provider::Alibaba,
    0.40,
    0.40,
    None,
    128_000,
    8_192,
);

const QWEN_2_5_7B: Model = model(
    "qwen2.5-7b",
    Provider::Alibaba,
    0.15,
    0.15,
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
    1.20,
    None,
    128_000,
    8_192,
);

const QWEN_3_CODER: Model = model(
    "qwen3-coder",
    Provider::Alibaba,
    0.22,
    1.00,
    None,
    262_144,
    8_192,
);

const QWEN_3_8B: Model = model(
    "qwen3-8b",
    Provider::Alibaba,
    0.05,
    0.40,
    None,
    128_000,
    8_192,
);

// ── Mistral ─────────────────────────────────────────────

const MISTRAL_LARGE: Model = model(
    "mistral-large",
    Provider::Mistral,
    2.00,
    6.00,
    None,
    128_000,
    4_096,
);

const MISTRAL_SMALL: Model = model(
    "mistral-small",
    Provider::Mistral,
    0.10,
    0.30,
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
    0.40,
    2.00,
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
    0.60,
    0.60,
    None,
    32_768,
    4_096,
);

// ── Master list ─────────────────────────────────────────

static ALL_MODELS: &[Model] = &[
    // OpenAI
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
    GEMINI_25_PRO,
    GEMINI_25_FLASH,
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
    DEEPSEEK_V3,
    DEEPSEEK_R1,
    // Alibaba
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
    PIXTRAL_LARGE,
    MIXTRAL_8X7B,
];

#[cfg(test)]
mod tests {
    use super::*;

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
