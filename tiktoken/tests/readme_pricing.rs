//! The pricing block in `tiktoken/README.md` (and its zh / ja translations),
//! kept compiling.
//!
//! README code is not doctested, so an example can rot into something that no
//! longer builds or no longer holds without anything failing. This is that
//! example verbatim — if the API moves under it, this breaks here rather than
//! in a reader's editor.

#[test]
fn readme_pricing_snippet_compiles_and_holds() {
    use tiktoken::pricing;

    let _cost = pricing::estimate_cost("gpt-4o", 1_000_000, 500_000).unwrap();

    let model = pricing::get_model("claude-opus-4").unwrap();
    let _cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

    let _models = pricing::models_by_provider(pricing::Provider::DeepSeek);

    let r = pricing::resolve_model("us.anthropic.claude-opus-5").unwrap();
    assert_eq!(r.model.id, "claude-opus-5");
    assert!(matches!(r.matched, pricing::Match::Normalized { .. }));

    assert!(pricing::estimate_cost("claude-haiku-4-5-20251001", 1_000, 1_000).is_some());
}
