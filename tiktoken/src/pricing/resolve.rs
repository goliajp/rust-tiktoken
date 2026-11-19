//! Resolving the model ids callers actually send (dated, dash-spelled,
//! platform-decorated) to a table entry.

use super::*;

/// How [`resolve_model`] reached a table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Match {
    /// The id is a table id (case-insensitively).
    Exact,
    /// The id was normalized before it matched. `as_id` is the table id it
    /// reached — log it if you want to know which of your model names are not
    /// spelled the way the table spells them.
    Normalized { as_id: &'static str },
}

/// A model, plus how its id was matched.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Resolved {
    /// The matched table entry.
    pub model: &'static Model,
    /// Whether the id matched as given, or only after normalization.
    pub matched: Match,
}

/// Look up a model by id, normalizing the id if it does not match as given.
///
/// [`get_model`] is an exact match, but the ids the provider APIs actually
/// accept differ from the ids this table is keyed by, in three systematic ways:
///
/// - **Dot versus dash.** The table writes a generation with a dot
///   (`claude-haiku-4.5`); the API addresses it with a dash
///   (`claude-haiku-4-5`).
/// - **Release dates.** `claude-sonnet-4-20250514`, `gpt-4o-2024-08-06`.
/// - **Platform decoration.** Amazon Bedrock prefixes the vendor and,
///   for cross-region inference, the region (`us.anthropic.claude-opus-4-5`)
///   and suffixes a version (`-v1:0`); Vertex AI separates a dated snapshot
///   with `@` (`claude-opus-4-5@20251101`).
///
/// Each is mechanical, so this resolves them rather than requiring an alias per
/// model per snapshot date. Candidates are tried most-literal first, so an id
/// that is in the table always matches itself before any normalization runs.
///
/// # Examples
///
/// ```
/// use tiktoken::pricing::{resolve_model, Match};
///
/// let r = resolve_model("claude-haiku-4-5-20251001").unwrap();
/// assert_eq!(r.model.id, "claude-haiku-4.5");
/// assert_eq!(r.matched, Match::Normalized { as_id: "claude-haiku-4.5" });
///
/// // an id the table already carries matches as given
/// assert_eq!(resolve_model("claude-opus-5").unwrap().matched, Match::Exact);
/// ```
pub fn resolve_model(id: &str) -> Option<Resolved> {
    if let Some(model) = get_model(id) {
        return Some(Resolved {
            model,
            matched: Match::Exact,
        });
    }

    normalizations(id).find_map(|candidate| {
        get_model(&candidate).map(|model| Resolved {
            model,
            matched: Match::Normalized { as_id: model.id },
        })
    })
}

/// Progressively normalized forms of `id`, least aggressive first.
///
/// Each step builds on the previous one, so a Bedrock id carrying a vendor
/// prefix, a version suffix and a release date is peeled one layer per step.
fn normalizations(id: &str) -> impl Iterator<Item = String> {
    let platform = strip_platform_decoration(id);
    let dated = strip_release_date(&platform);
    let merged = merge_version_segments(&dated);
    let unsuffixed = merged.strip_suffix(".0").map(str::to_string);

    let mut seen: Vec<String> = Vec::with_capacity(4);
    [Some(platform), Some(dated), Some(merged), unsuffixed]
        .into_iter()
        .flatten()
        .filter(move |candidate| {
            if candidate.is_empty() || seen.contains(candidate) {
                return false;
            }
            seen.push(candidate.clone());
            true
        })
}

/// Strip Bedrock and Vertex decoration: a leading `[region.]vendor.` prefix, a
/// trailing Bedrock `-vN:M` version, and a trailing Vertex `@snapshot`.
///
/// The leading prefix is matched against known region and vendor tokens rather
/// than "everything before the first dot", because a model id may legitimately
/// begin with a dotted version (`gpt-5.6-terra`).
fn strip_platform_decoration(id: &str) -> String {
    const REGIONS: &[&str] = &["us", "eu", "apac", "global"];
    const VENDORS: &[&str] = &[
        "anthropic",
        "amazon",
        "meta",
        "mistral",
        "cohere",
        "deepseek",
        "ai21",
        "writer",
        "openai",
        "google",
        "qwen",
    ];

    let mut out = id;

    // Vertex dated snapshot: `claude-opus-4-5@20251101`
    if let Some((head, _)) = out.split_once('@') {
        out = head;
    }
    // Bedrock version: `-v1:0` → `-v1` → ``
    if let Some((head, tail)) = out.rsplit_once(':')
        && tail.bytes().all(|b| b.is_ascii_digit())
    {
        out = head;
    }
    if let Some((head, tail)) = out.rsplit_once("-v")
        && !tail.is_empty()
        && tail.bytes().all(|b| b.is_ascii_digit())
    {
        out = head;
    }
    // Bedrock `[region.]vendor.` prefix
    if let Some((head, tail)) = out.split_once('.')
        && REGIONS.iter().any(|r| r.eq_ignore_ascii_case(head))
    {
        out = tail;
    }
    if let Some((head, tail)) = out.split_once('.')
        && VENDORS.iter().any(|v| v.eq_ignore_ascii_case(head))
    {
        out = tail;
    }

    out.to_string()
}

/// Drop a trailing release date: `-20250514` or `-2024-08-06`.
fn strip_release_date(id: &str) -> String {
    let parts: Vec<&str> = id.split('-').collect();
    let digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());

    if parts.len() >= 2 && digits(parts[parts.len() - 1], 8) {
        return parts[..parts.len() - 1].join("-");
    }
    if parts.len() >= 4
        && digits(parts[parts.len() - 3], 4)
        && digits(parts[parts.len() - 2], 2)
        && digits(parts[parts.len() - 1], 2)
    {
        return parts[..parts.len() - 3].join("-");
    }

    id.to_string()
}

/// Join runs of adjacent numeric segments with dots, which is how this table
/// spells a generation: `claude-haiku-4-5` → `claude-haiku-4.5`,
/// `claude-3-5-sonnet` → `claude-3.5-sonnet`. A lone number is left alone, so
/// `claude-3-opus` and `gpt-5.6-terra` pass through unchanged.
fn merge_version_segments(id: &str) -> String {
    let parts: Vec<&str> = id.split('-').collect();
    let numeric = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());

    let mut out: Vec<String> = Vec::with_capacity(parts.len());
    let mut i = 0;
    while i < parts.len() {
        let mut j = i;
        while j + 1 < parts.len() && numeric(parts[j]) && numeric(parts[j + 1]) {
            j += 1;
        }
        out.push(parts[i..=j].join("."));
        i = j + 1;
    }

    out.join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_the_id_as_given() {
        let r = resolve_model("claude-opus-5").unwrap();
        assert_eq!(r.model.id, "claude-opus-5");
        assert_eq!(r.matched, Match::Exact);
    }

    #[test]
    fn resolve_reports_the_id_it_reached() {
        let r = resolve_model("claude-haiku-4-5").unwrap();
        assert_eq!(r.model.id, "claude-haiku-4.5");
        assert_eq!(
            r.matched,
            Match::Normalized {
                as_id: "claude-haiku-4.5"
            }
        );
    }

    #[test]
    fn resolve_handles_dash_spelled_generations() {
        for (api_id, table_id) in [
            ("claude-haiku-4-5", "claude-haiku-4.5"),
            ("claude-opus-4-6", "claude-opus-4.6"),
            ("claude-sonnet-4-6", "claude-sonnet-4.6"),
            ("claude-3-5-sonnet", "claude-3.5-sonnet"),
            ("claude-sonnet-4-0", "claude-sonnet-4"),
        ] {
            assert_eq!(
                resolve_model(api_id).unwrap().model.id,
                table_id,
                "{api_id}"
            );
        }
    }

    #[test]
    fn resolve_handles_release_dates() {
        for (api_id, table_id) in [
            ("claude-sonnet-4-20250514", "claude-sonnet-4"),
            ("claude-3-5-sonnet-20241022", "claude-3.5-sonnet"),
            ("claude-haiku-4-5-20251001", "claude-haiku-4.5"),
            ("gpt-4o-2024-08-06", "gpt-4o"),
        ] {
            assert_eq!(
                resolve_model(api_id).unwrap().model.id,
                table_id,
                "{api_id}"
            );
        }
    }

    #[test]
    fn resolve_handles_bedrock_and_vertex_decoration() {
        for (api_id, table_id) in [
            ("anthropic.claude-opus-5", "claude-opus-5"),
            ("us.anthropic.claude-opus-5", "claude-opus-5"),
            ("eu.anthropic.claude-sonnet-4-6", "claude-sonnet-4.6"),
            (
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "claude-3.5-sonnet",
            ),
            ("claude-opus-4-5@20251101", "claude-opus-4.5"),
        ] {
            assert_eq!(
                resolve_model(api_id).unwrap().model.id,
                table_id,
                "{api_id}"
            );
        }
    }

    #[test]
    fn resolve_leaves_dotted_ids_alone() {
        // `gpt-5.6-terra` starts with a dotted version, not a vendor prefix
        assert_eq!(strip_platform_decoration("gpt-5.6-terra"), "gpt-5.6-terra");
        assert_eq!(merge_version_segments("gpt-5.6-terra"), "gpt-5.6-terra");
        assert_eq!(
            resolve_model("gpt-5.6-terra").unwrap().matched,
            Match::Exact
        );
    }

    #[test]
    fn resolve_does_not_invent_models() {
        assert!(resolve_model("totally-made-up-9-9").is_none());
        assert!(resolve_model("").is_none());
        assert!(resolve_model("anthropic.").is_none());
    }

    /// The property that makes a normalizing lookup safe to rely on: every id
    /// in the table resolves to itself, and none normalizes onto a *different*
    /// entry. Without it, a future entry could collide with another model's
    /// normalized form and silently price at the wrong rate.
    #[test]
    fn normalization_never_maps_one_model_onto_another() {
        for m in all_models() {
            let r = resolve_model(m.id).unwrap_or_else(|| panic!("{} no longer resolves", m.id));
            assert_eq!(r.model.id, m.id, "{} resolved to {}", m.id, r.model.id);
            assert_eq!(r.matched, Match::Exact, "{} should match as given", m.id);
        }
    }

    /// Normalization must not let two different table entries be reached by the
    /// same normalized id — that would make the winner depend on table order.
    #[test]
    fn normalized_forms_are_unique_across_the_table() {
        let mut seen: Vec<(String, &str)> = Vec::new();
        for m in all_models() {
            let normalized =
                merge_version_segments(&strip_release_date(&strip_platform_decoration(m.id)));
            if let Some((_, other)) = seen.iter().find(|(n, _)| *n == normalized) {
                panic!(
                    "{} and {other} share the normalized form {normalized}",
                    m.id
                );
            }
            seen.push((normalized, m.id));
        }
    }

    #[test]
    fn estimate_cost_accepts_api_spellings() {
        let table = estimate_cost("claude-haiku-4.5", 1_000, 1_000).unwrap();
        let api = estimate_cost("claude-haiku-4-5", 1_000, 1_000).unwrap();
        assert_eq!(table, api);
    }
}
