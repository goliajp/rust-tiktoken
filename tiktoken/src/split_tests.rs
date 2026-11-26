use super::*;
use crate::pretokenize::FastPath;

#[test]
fn cuts_cover_the_text() {
    let rules = Rules::of(FastPath::Cl100k);
    let text = "word ".repeat(10_000);
    let c = cuts(text.as_bytes(), 4096, 8, rules);
    assert_eq!(c.len(), 9);
    assert_eq!((c[0], c[8]), (0, text.len()));
    assert!(c.windows(2).all(|w| w[0] < w[1]));
    assert!(
        c[1..8]
            .iter()
            .all(|&i| cut::is_cut(text.as_bytes(), i, rules))
    );
    // nothing to cut at
    let digits = "1".repeat(50_000);
    assert_eq!(cuts(digits.as_bytes(), 4096, 8, rules), [0, 50_000]);
    assert_eq!(cuts(b"", 4096, 8, rules), [0, 0]);
}
