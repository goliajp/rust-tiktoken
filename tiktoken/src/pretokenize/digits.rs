//! Tekken v15's digit rule, `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}`:
//! a digit run is cut into groups of three counted from the right, so the
//! leading group holds `len % 3` digits (three when that is zero). The
//! lookahead is beyond the `regex` crate, so the run is cut here instead.

/// If a `\p{N}` char starts at `pos`, the end of the digit group that starts
/// there; `None` otherwise.
///
/// A match only begins inside a digit run when the previous match was a digit
/// group cut by this function, which leaves the remainder a multiple of three
/// long. So a preceding digit means "take three" without rescanning the run,
/// keeping a long run linear rather than quadratic.
#[inline]
pub(super) fn tekken15_digit_group_end(text: &str, pos: usize) -> Option<usize> {
    let b = text.as_bytes();
    let first = *b.get(pos)?;
    if first < 0x80 {
        if !first.is_ascii_digit() {
            return None;
        }
    } else if !text[pos..].chars().next().is_some_and(char::is_numeric) {
        return None;
    }
    let mid_run = text[..pos]
        .chars()
        .next_back()
        .is_some_and(char::is_numeric);
    let take = if mid_run {
        3
    } else {
        let run = text[pos..].chars().take_while(|c| c.is_numeric()).count();
        match run % 3 {
            0 => 3,
            r => r,
        }
    };
    let len: usize = text[pos..].chars().take(take).map(char::len_utf8).sum();
    Some(pos + len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            match tekken15_digit_group_end(text, pos) {
                Some(end) => {
                    out.push(&text[pos..end]);
                    pos = end;
                }
                None => pos += text[pos..].chars().next().unwrap().len_utf8(),
            }
        }
        out
    }

    #[test]
    fn groups_count_from_the_right() {
        assert_eq!(groups("1"), ["1"]);
        assert_eq!(groups("123"), ["123"]);
        assert_eq!(groups("1234"), ["1", "234"]);
        assert_eq!(groups("12345"), ["12", "345"]);
        assert_eq!(groups("1234567"), ["1", "234", "567"]);
        assert_eq!(groups("x1234y56"), ["1", "234", "56"]);
    }

    #[test]
    fn non_ascii_digits_count_as_digits() {
        // arabic-indic and fullwidth digits are \p{N}, two and three bytes wide
        assert_eq!(groups("١٢٣٤"), ["١", "٢٣٤"]);
        assert_eq!(groups("12３4"), ["1", "2３4"]);
    }

    #[test]
    fn non_digit_start_is_none() {
        assert_eq!(tekken15_digit_group_end("a1", 0), None);
        assert_eq!(tekken15_digit_group_end("你1", 0), None);
        assert_eq!(tekken15_digit_group_end("", 0), None);
    }
}
