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
    for m in all_models() {
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
    let mut ids: Vec<&str> = all_models().iter().map(|m| m.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), all_models().len(), "duplicate model ids found");
}

#[test]
fn test_all_models_non_negative_pricing() {
    for m in all_models() {
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
    for m in all_models() {
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
    for m in all_models() {
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
    for m in all_models() {
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
        Provider::Voyage,
    ]
    .iter()
    .map(|p| models_by_provider(*p).len())
    .sum();
    assert_eq!(
        total,
        all_models().len(),
        "provider counts don't sum to total"
    );
}

#[test]
fn gemini_embedding_001_prices() {
    let m = get_model("gemini-embedding-001").unwrap();
    assert_eq!(m.provider, Provider::Google);
    assert_eq!(m.context_window, 2_048);
    // $0.15 / 1M input tokens
    assert!((estimate_cost("gemini-embedding-001", 1_000_000, 0).unwrap() - 0.15).abs() < 1e-9);
}
