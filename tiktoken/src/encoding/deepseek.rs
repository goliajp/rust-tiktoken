//! DeepSeek encodings: deepseek_v3 (V3, R1) and deepseek_v4 (V4, V4.1).

use super::*;

/// Construct the deepseek_v3 encoding (DeepSeek V3, R1).
///
/// Vocabulary size: 128,000 regular tokens + 818 added tokens — 3 sentence
/// markers, 800 placeholders (`<｜place▁holder▁no▁0｜>`..=`no▁799｜>`, ids
/// 128000..=128799), and 15 named tokens (ids 128800..=128814).
///
/// Note the named tokens mostly use fullwidth pipes (`｜`, U+FF5C), not ASCII
/// `|`; `<|EOT|>` is the one exception and really is ASCII.
#[cfg(feature = "vocab-deepseek_v3")]
pub fn deepseek_v3() -> CoreBpe {
    let encoder = parse_tkv(DEEPSEEK_V3_CHAIN);
    CoreBpe::new(
        encoder,
        deepseek_v3_special_tokens(),
        DEEPSEEK_V3_PATTERN,
        FastPath::Deepseek,
        WhitespaceRules::NewlineFirstSplitOnNumCjk,
    )
}

/// DeepSeek V3's 818-entry added-token table, shared as the base of
/// [`deepseek_v4`]'s table.
fn deepseek_v3_special_tokens() -> FxHashMap<Vec<u8>, u32> {
    let mut special = special_tokens(&[
        ("<｜begin▁of▁sentence｜>", 0),
        ("<｜end▁of▁sentence｜>", 1),
        ("<｜▁pad▁｜>", 2),
        ("<｜fim▁hole｜>", 128800),
        ("<｜fim▁begin｜>", 128801),
        ("<｜fim▁end｜>", 128802),
        ("<｜User｜>", 128803),
        ("<｜Assistant｜>", 128804),
        ("<|EOT|>", 128805),
        ("<｜tool▁calls▁begin｜>", 128806),
        ("<｜tool▁calls▁end｜>", 128807),
        ("<｜tool▁call▁begin｜>", 128808),
        ("<｜tool▁call▁end｜>", 128809),
        ("<｜tool▁outputs▁begin｜>", 128810),
        ("<｜tool▁outputs▁end｜>", 128811),
        ("<｜tool▁output▁begin｜>", 128812),
        ("<｜tool▁output▁end｜>", 128813),
        ("<｜tool▁sep｜>", 128814),
    ]);
    for id in 128000..=128799_u32 {
        let n = id - 128000;
        special.insert(format!("<｜place▁holder▁no▁{n}｜>").into_bytes(), id);
    }
    special
}

/// Construct the deepseek_v4 encoding (DeepSeek V4 Pro / Flash, 2026).
///
/// Same 128,000-token vocabulary, merges, and split pattern as
/// [`deepseek_v3`]; the delta is the added-token table, which grows from
/// V3's 818 entries to 1,283 — 50 new named tokens (`<think>`, the DSML
/// markup markers, vision/grounding tags) plus 415 multimodal span
/// placeholders (`<|place_holder_mm_span_0021|>`..=`_0435|>`).
#[cfg(feature = "vocab-deepseek_v3")]
pub fn deepseek_v4() -> CoreBpe {
    let encoder = parse_tkv(DEEPSEEK_V3_CHAIN);
    let mut special = deepseek_v3_special_tokens();
    for (name, id) in [
        ("<｜begin▁of▁repo▁name｜>", 128815_u32),
        ("<｜end▁of▁repo▁name｜>", 128816),
        ("<｜begin▁of▁file▁name｜>", 128817),
        ("<｜end▁of▁file▁name｜>", 128818),
        ("<｜begin▁of▁file｜>", 128819),
        ("<｜end▁of▁file｜>", 128820),
        ("<think>", 128821),
        ("</think>", 128822),
        ("<｜place▁holder▁for▁copy｜>", 128823),
        ("<｜place▁holder▁for▁pointer▁replace｜>", 128824),
        ("｜DSML｜", 128825),
        ("<｜begin▁sys｜>", 128826),
        ("<｜end▁sys｜>", 128827),
        ("<｜latest_reminder｜>", 128828),
        ("<｜action｜>", 128829),
        ("<｜query｜>", 128830),
        ("<｜authority｜>", 128831),
        ("<｜domain｜>", 128832),
        ("<｜task｜>", 128833),
        ("<｜political｜>", 128834),
        ("<｜entity｜>", 128835),
        ("<｜title｜>", 128836),
        ("<｜safety｜>", 128837),
        ("<｜answer｜>", 128838),
        ("<｜search｜>", 128839),
        ("<dsml:", 128840),
        ("</dsml:", 128841),
        ("<｜search▁begin｜>", 128842),
        ("<｜search▁end｜>", 128843),
        ("<｜extracted_url｜>", 128844),
        ("<｜read_url｜>", 128845),
        ("<｜end_of_query｜>", 128846),
        ("<｜rl_image_pad｜>", 129262),
        ("<｜rl_image_start｜>", 129263),
        ("<｜image2｜>", 129264),
        ("<｜/table>｜", 129265),
        ("<｜table｜>", 129266),
        ("<｜/td｜>", 129267),
        ("<｜td｜>", 129268),
        ("<｜/tr｜>", 129269),
        ("<｜tr｜>", 129270),
        ("<｜/polygon｜>", 129271),
        ("<｜polygon｜>", 129272),
        ("<｜/point｜>", 129273),
        ("<｜point｜>", 129274),
        ("<｜/box｜>", 129275),
        ("<｜box｜>", 129276),
        ("<｜/ref｜>", 129277),
        ("<｜ref｜>", 129278),
        ("<｜image｜>", 129279),
    ] {
        special.insert(name.as_bytes().to_vec(), id);
    }
    // 415 multimodal span placeholders: number 0021..=0435 map to contiguous
    // ids 128847..=129261 (id = 128847 + (n - 21)).
    for n in 21..=435_u32 {
        special.insert(
            format!("<|place_holder_mm_span_{n:04}|>").into_bytes(),
            128847 + (n - 21),
        );
    }
    CoreBpe::new(
        encoder,
        special,
        DEEPSEEK_V3_PATTERN,
        FastPath::Deepseek,
        WhitespaceRules::NewlineFirstSplitOnNumCjk,
    )
}
