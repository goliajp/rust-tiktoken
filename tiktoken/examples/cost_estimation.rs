// cost estimation across multiple providers and models
//
// demonstrates:
// - looking up model pricing with `get_model`
// - estimating cost for input/output tokens
// - estimating cost with cached input tokens
// - listing models by provider
// - combining token counting with cost estimation

use tiktoken::pricing::{self, Provider};

fn main() {
    // estimate cost for a single model
    println!("=== single model cost estimation ===\n");

    let model = pricing::get_model("gpt-4o").expect("model not found");
    println!("model:          {}", model.id);
    println!("provider:       {}", model.provider);
    println!("context window: {} tokens", model.context_window);
    println!("max output:     {} tokens", model.max_output);
    println!(
        "pricing:        ${}/1M input, ${}/1M output",
        model.pricing.input_per_1m, model.pricing.output_per_1m,
    );
    if let Some(cached) = model.pricing.cached_input_per_1m {
        println!("cached input:   ${cached}/1M");
    }

    let input_tokens = 10_000_u64;
    let output_tokens = 2_000_u64;
    let cost = model.estimate_cost(input_tokens, output_tokens);
    println!("\n{input_tokens} input + {output_tokens} output = ${cost:.6}",);

    // estimate with cached tokens
    let cached_tokens = 8_000_u64;
    let fresh_tokens = 2_000_u64;
    let cost_cached = model.estimate_cost_with_cache(fresh_tokens, cached_tokens, output_tokens);
    println!(
        "{fresh_tokens} fresh + {cached_tokens} cached + {output_tokens} output = ${cost_cached:.6}",
    );
    println!("savings from cache: ${:.6}", cost - cost_cached);

    // compare the same prompt across providers
    println!("\n=== cross-provider comparison (10k input, 2k output) ===\n");

    let comparison_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3.5-haiku",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "deepseek-v3",
        "deepseek-r1",
        "llama-3.3-70b",
        "qwen2.5-72b",
        "mistral-large",
        "mistral-small",
    ];

    println!("{:<22} {:<12} {:>12}", "model", "provider", "cost (USD)");
    println!("{}", "-".repeat(48));

    for model_id in comparison_models {
        let m = pricing::get_model(model_id).expect("model not found");
        let cost = m.estimate_cost(input_tokens, output_tokens);
        println!("{:<22} {:<12} ${cost:>10.6}", m.id, m.provider);
    }

    // use the convenience function for quick estimates
    println!("\n=== convenience function ===\n");

    let quick_cost =
        pricing::estimate_cost("gpt-4o-mini", 50_000, 10_000).expect("model not found");
    println!("gpt-4o-mini: 50k input + 10k output = ${quick_cost:.6}");

    // list all models for a provider
    println!("\n=== all models by provider ===\n");

    let providers = [
        Provider::OpenAI,
        Provider::Anthropic,
        Provider::Google,
        Provider::Meta,
        Provider::DeepSeek,
        Provider::Alibaba,
        Provider::Mistral,
    ];

    for provider in providers {
        let models = pricing::models_by_provider(provider);
        let ids: Vec<&str> = models.iter().map(|m| m.id).collect();
        println!("{provider:<10} ({} models): {}", ids.len(), ids.join(", "));
    }

    // real-world scenario: count tokens then estimate cost
    println!("\n=== real-world: count tokens then estimate cost ===\n");

    let prompt = "Explain the theory of relativity in simple terms, \
                  covering both special and general relativity. \
                  Include examples that a high school student would understand.";

    let enc = tiktoken::encoding_for_model("gpt-4o").expect("encoding not found");
    let input_count = enc.count(prompt) as u64;

    // assume the model generates roughly 500 output tokens
    let estimated_output = 500_u64;

    let m = pricing::get_model("gpt-4o").unwrap();
    let estimated_cost = m.estimate_cost(input_count, estimated_output);

    println!("prompt tokens:     {input_count}");
    println!("estimated output:  {estimated_output}");
    println!("estimated cost:    ${estimated_cost:.6}");

    println!("\ndone.");
}
