use super::generate::{MAX_CP, members};
use super::*;

/// The category table against the regex crate, code point by code point.
#[test]
fn table_matches_regex_classes() {
    type Pred = fn(u8) -> bool;
    let checks: [(&str, Pred); 12] = [
        (r"\p{L}", |c| {
            matches!(c & GROUP_MASK, LU | LL | LT | LM | LO)
        }),
        (r"\p{Lu}", |c| c & GROUP_MASK == LU),
        (r"\p{Ll}", |c| c & GROUP_MASK == LL),
        (r"\p{Lt}", |c| c & GROUP_MASK == LT),
        (r"\p{Lm}", |c| c & GROUP_MASK == LM),
        (r"\p{Lo}", |c| c & GROUP_MASK == LO),
        (r"\p{M}", |c| c & GROUP_MASK == MARK),
        (r"\p{N}", |c| c & GROUP_MASK == NUM),
        (r"\p{P}", |c| c & GROUP_MASK == PUNCT),
        (r"\p{S}", |c| c & GROUP_MASK == SYM),
        (r"\s", |c| c & WS != 0),
        (r"\p{Han}", |c| c & HAN != 0),
    ];
    for (class, pred) in checks {
        let expect = members(class);
        for cp in 0..=MAX_CP {
            if char::from_u32(cp).is_none() {
                continue;
            }
            assert_eq!(
                pred(category(cp)),
                expect[cp as usize],
                "U+{cp:04X} {class}"
            );
        }
    }
    for cp in 0..128u32 {
        assert_eq!(tables::ASCII[cp as usize], category(cp));
    }
}

/// Cross-check of `members` itself with a differently shaped matcher: an
/// anchored per-char match on a sample of code points, plus known values.
#[test]
fn members_agrees_with_per_char_match() {
    for class in [r"\p{L}", r"\p{M}", r"\p{N}", r"\s", r"\p{Han}"] {
        let all = members(class);
        assert!(all.iter().filter(|&&x| x).count() > 20, "{class}");
        let re = regex::Regex::new(&format!("^{class}$")).unwrap();
        let mut buf = [0u8; 4];
        for cp in (0..=MAX_CP).step_by(97) {
            if let Some(c) = char::from_u32(cp) {
                assert_eq!(re.is_match(c.encode_utf8(&mut buf)), all[cp as usize]);
            }
        }
    }
    assert_eq!(category('A' as u32), LU);
    assert_eq!(category('ǅ' as u32), LT);
    assert_eq!(category('ʰ' as u32), LM);
    assert_eq!(category('\u{4E00}' as u32), LO | HAN);
    assert_eq!(category('\u{0301}' as u32), MARK);
    assert_eq!(category('²' as u32), NUM);
    assert_eq!(category('\u{3000}' as u32), SEP | WS);
    assert_eq!(category('\u{0085}' as u32), OTHER | WS);
    assert_eq!(category('\n' as u32), NL | WS);
    assert_eq!(category('\u{200D}' as u32), OTHER);
}

/// `char::is_numeric` follows the standard library's Unicode version, which
/// can be newer than the regex crate's. The two may only disagree on code
/// points the table's version leaves unassigned.
#[test]
fn std_numeric_and_whitespace_differ_only_on_newer_code_points() {
    for cp in 0..=MAX_CP {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let cat = category(cp);
        assert_eq!(c.is_whitespace(), cat & WS != 0, "U+{cp:04X}");
        if c.is_numeric() != (cat & GROUP_MASK == NUM) {
            assert_eq!(cat, OTHER, "U+{cp:04X}");
        }
    }
}
