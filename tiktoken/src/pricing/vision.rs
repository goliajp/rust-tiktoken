//! How an image becomes billable input tokens, per provider.

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
///
/// `#[non_exhaustive]`: providers change their image formulas, and each new
/// formula is a new variant. Match with a `_` arm.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
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
    /// Anthropic's earlier published approximation: tokens ≈
    /// `(width * height) / divisor`, capped at `cap_tokens`.
    ///
    /// No table entry uses it any more; Anthropic now documents
    /// [`Self::AnthropicPatch`] for every model. Kept so code naming it builds.
    AnthropicDivisor { divisor: u32, cap_tokens: u32 },
    /// Anthropic Claude vision: one token per 28×28-pixel patch,
    /// `ceil(w/28) × ceil(h/28)`.
    ///
    /// An image that exceeds either limit is first scaled down to the largest
    /// aspect-preserving size whose padded edges stay within `max_edge_px` and
    /// whose patch count stays within `max_tokens` (Anthropic's reference
    /// resize: a binary search over the long edge, short edge rounded half to
    /// even). Claude 4.7 and later: 2576 px / 4784 tokens; earlier models:
    /// 1568 px / 1568 tokens.
    AnthropicPatch { max_edge_px: u32, max_tokens: u32 },
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
            VisionPricing::AnthropicPatch {
                max_edge_px,
                max_tokens,
            } => {
                let (w, h) = anthropic_resized_size(width, height, max_edge_px, max_tokens);
                anthropic_patches(w, h)
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

const ANTHROPIC_PATCH_PX: u32 = 28;

fn anthropic_patches(w: u32, h: u32) -> u64 {
    w.div_ceil(ANTHROPIC_PATCH_PX) as u64 * h.div_ceil(ANTHROPIC_PATCH_PX) as u64
}

/// The size Claude scales an image to before patching, ported from the
/// reference implementation in Anthropic's vision documentation. Exact `.5`
/// short edges round to even, as the live API does.
fn anthropic_resized_size(width: u32, height: u32, max_edge: u32, max_tokens: u32) -> (u32, u32) {
    let fits = |w: u32, h: u32| {
        w.div_ceil(ANTHROPIC_PATCH_PX) * ANTHROPIC_PATCH_PX <= max_edge
            && h.div_ceil(ANTHROPIC_PATCH_PX) * ANTHROPIC_PATCH_PX <= max_edge
            && anthropic_patches(w, h) <= max_tokens as u64
    };
    if fits(width, height) {
        return (width, height);
    }
    if height > width {
        let (h, w) = anthropic_resized_size(height, width, max_edge, max_tokens);
        return (w, h);
    }
    let aspect = width as f64 / height as f64;
    let short = |long: u32| ((long as f64 / aspect).round_ties_even() as u32).max(1);
    // lo always fits, hi never does
    let (mut lo, mut hi) = (1, width);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if fits(mid, short(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, short(lo))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::get_model;

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

    fn claude_tokens(model: &str, w: u32, h: u32) -> u64 {
        get_model(model)
            .unwrap()
            .extended
            .vision
            .unwrap()
            .image_tokens(w, h, ImageDetail::High)
    }

    // every expected value below is from the table and examples in Anthropic's
    // vision and vision-coordinates documentation
    #[test]
    fn anthropic_standard_tier_matches_documented_sizes() {
        for model in ["claude-haiku-4.5", "claude-sonnet-4.6"] {
            assert_eq!(claude_tokens(model, 200, 200), 64);
            assert_eq!(claude_tokens(model, 1000, 1000), 1296);
            assert_eq!(claude_tokens(model, 1092, 1092), 1521);
            assert_eq!(claude_tokens(model, 1920, 1080), 1560);
            assert_eq!(claude_tokens(model, 2000, 1500), 1564);
            assert_eq!(claude_tokens(model, 3840, 2160), 1560);
        }
    }

    #[test]
    fn anthropic_high_resolution_tier_matches_documented_sizes() {
        for model in ["claude-opus-4.7", "claude-opus-5.5", "claude-fable-5.2"] {
            assert_eq!(claude_tokens(model, 1000, 1000), 1296);
            assert_eq!(claude_tokens(model, 1920, 1080), 2691);
            assert_eq!(claude_tokens(model, 2000, 1500), 3888);
            assert_eq!(claude_tokens(model, 3840, 2160), 4784);
            // the A4 scan the standard tier resizes is left alone here
            assert_eq!(claude_tokens(model, 1075, 1520), 39 * 55);
        }
    }

    #[test]
    fn anthropic_resize_matches_reference_implementation() {
        assert_eq!(anthropic_resized_size(1920, 1080, 1568, 1568), (1456, 819));
        // the docs table prints 1269x952, but their own reference code gives
        // 1270x952; both cost 1564 tokens
        assert_eq!(anthropic_resized_size(2000, 1500, 1568, 1568), (1270, 952));
        assert_eq!(anthropic_resized_size(1075, 1520, 1568, 1568), (924, 1307));
        assert_eq!(anthropic_resized_size(3840, 2160, 2576, 4784), (2576, 1449));
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
        // 1000×1000 → 1296 tokens × $1/M; Anthropic quotes ~$1.30 per thousand
        let cost = m
            .estimate_image_cost(1000, 1000, ImageDetail::High)
            .unwrap();
        assert!(
            (cost - 0.001296).abs() < 1e-9,
            "expected 0.001296, got {cost}"
        );
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
}
