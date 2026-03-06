use regex::Regex;
use rustc_hash::FxHashMap;

/// A Byte Pair Encoding tokenizer engine.
///
/// Instances are created via [`get_encoding`](crate::get_encoding) or
/// [`encoding_for_model`](crate::encoding_for_model) and cached globally.
///
/// # Examples
///
/// ```
/// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
///
/// // encode
/// let tokens = enc.encode("hello world");
/// assert_eq!(tokens, vec![15339, 1917]);
///
/// // decode
/// let text = enc.decode_to_string(&tokens).unwrap();
/// assert_eq!(text, "hello world");
///
/// // count without allocating
/// assert_eq!(enc.count("hello world"), 2);
/// ```
pub struct CoreBpe {
    encoder: FxHashMap<Vec<u8>, u32>,
    decoder: FxHashMap<u32, Vec<u8>>,
    special_encoder: FxHashMap<Vec<u8>, u32>,
    special_decoder: FxHashMap<u32, Vec<u8>>,
    regex: Regex,
    special_regex: Option<Regex>,
}

impl CoreBpe {
    pub(crate) fn new(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_encoder: FxHashMap<Vec<u8>, u32>,
        pattern: &str,
    ) -> Self {
        let regex = Regex::new(pattern).expect("invalid regex pattern");

        let special_regex = if special_encoder.is_empty() {
            None
        } else {
            let pat = special_encoder
                .keys()
                .map(|k| regex::escape(&String::from_utf8_lossy(k)))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&pat).expect("invalid special regex"))
        };

        let decoder = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();
        let special_decoder = special_encoder
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        Self {
            encoder,
            decoder,
            special_encoder,
            special_decoder,
            regex,
            special_regex,
        }
    }

    /// Encode text into token ids.
    ///
    /// Special tokens like `<|endoftext|>` are treated as ordinary text.
    /// Use [`encode_with_special_tokens`](Self::encode_with_special_tokens) to recognize them.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// let tokens = enc.encode("hello world");
    /// assert_eq!(tokens, vec![15339, 1917]);
    /// ```
    #[inline]
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::with_capacity(text.len() / 3);
        self.encode_into(text, &mut result);
        result
    }

    /// Encode text into token ids, recognizing special tokens.
    ///
    /// Special tokens (e.g. `<|endoftext|>`) are encoded as their designated ids
    /// instead of being split into sub-word pieces.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// let tokens = enc.encode_with_special_tokens("hello<|endoftext|>world");
    /// assert!(tokens.contains(&100257)); // <|endoftext|> token id
    /// ```
    #[must_use]
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        let special_regex = match &self.special_regex {
            Some(r) => r,
            None => return self.encode(text),
        };

        let mut result = Vec::with_capacity(text.len() / 3);
        let mut start = 0;

        for mat in special_regex.find_iter(text) {
            if mat.start() > start {
                self.encode_into(&text[start..mat.start()], &mut result);
            }
            let piece = &text.as_bytes()[mat.start()..mat.end()];
            if let Some(&token) = self.special_encoder.get(piece) {
                result.push(token);
            }
            start = mat.end();
        }

        if start < text.len() {
            self.encode_into(&text[start..], &mut result);
        }

        result
    }

    /// Decode token ids back to raw bytes.
    ///
    /// Use [`decode_to_string`](Self::decode_to_string) if you want a `String` directly.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// let bytes = enc.decode(&[15339, 1917]);
    /// assert_eq!(&bytes, b"hello world");
    /// ```
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let mut result = Vec::with_capacity(tokens.len() * 4);
        for &token in tokens {
            if let Some(bytes) = self.decoder.get(&token) {
                result.extend_from_slice(bytes);
            } else if let Some(bytes) = self.special_decoder.get(&token) {
                result.extend_from_slice(bytes);
            }
        }
        result
    }

    /// Decode token ids back to a UTF-8 string.
    ///
    /// Returns `Err` if the decoded bytes are not valid UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// let text = enc.decode_to_string(&[15339, 1917]).unwrap();
    /// assert_eq!(text, "hello world");
    /// ```
    pub fn decode_to_string(&self, tokens: &[u32]) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.decode(tokens))
    }

    /// Count tokens without allocating a token vector.
    ///
    /// This is faster than `encode(text).len()` because it avoids building
    /// the full token id vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("o200k_base").unwrap();
    /// let count = enc.count("The quick brown fox.");
    /// assert_eq!(count, enc.encode("The quick brown fox.").len());
    /// ```
    #[inline]
    #[must_use]
    pub fn count(&self, text: &str) -> usize {
        let mut count = 0;
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < text.len() {
            let mat = match self.regex.find_at(text, pos) {
                Some(m) => m,
                None => break,
            };
            let start = mat.start();
            let end = adjust_whitespace_end(bytes, start, mat.end());
            let piece = &bytes[start..end];
            if self.encoder.contains_key(piece) {
                count += 1;
            } else {
                count += bpe_count(piece, &self.encoder);
            }
            pos = end;
        }
        count
    }

    /// Main encoding hot loop. Iterates regex matches over the input, then for each
    /// matched piece: (1) try direct HashMap lookup (fast path for known tokens),
    /// (2) fall back to BPE merge for unknown multi-byte pieces.
    ///
    /// The `adjust_whitespace_end` call emulates the `\s+(?!\S)` negative lookahead
    /// from original tiktoken patterns without needing a backtracking regex engine.
    fn encode_into(&self, text: &str, result: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < text.len() {
            let mat = match self.regex.find_at(text, pos) {
                Some(m) => m,
                None => break,
            };
            let start = mat.start();
            // trim trailing whitespace char when followed by non-whitespace (lookahead emulation)
            let end = adjust_whitespace_end(bytes, start, mat.end());
            let piece = &bytes[start..end];
            if let Some(&token) = self.encoder.get(piece) {
                // direct lookup hit — most common path (~95% of tokens)
                result.push(token);
            } else {
                // rare path: piece not in vocabulary, apply BPE merging
                bpe_encode(piece, &self.encoder, result);
            }
            pos = end;
        }
    }
}

/// BPE merge with rank cache — only recomputes 2 neighbor ranks per merge step.
/// Uses separate parts (Vec<usize>) and rank_cache (Vec<u32>) for cache-friendly layout.
fn byte_pair_merge(piece: &[u8], ranks: &FxHashMap<Vec<u8>, u32>) -> Vec<usize> {
    let n = piece.len() + 1;

    // fast path: 2-byte piece
    if n == 3 {
        if ranks.contains_key(piece) {
            return vec![0, piece.len()];
        }
        return vec![0, 1, piece.len()];
    }

    let mut parts: Vec<usize> = (0..n).collect();
    let mut rank_cache: Vec<u32> = (0..n)
        .map(|i| {
            if i + 2 < n {
                ranks.get(&piece[i..i + 2]).copied().unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        })
        .collect();

    loop {
        if parts.len() <= 2 {
            break;
        }

        // find minimum rank
        let mut min_rank = u32::MAX;
        let mut min_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..parts.len() - 1 {
            if rank_cache[i] < min_rank {
                min_rank = rank_cache[i];
                min_idx = i;
            }
        }

        if min_rank == u32::MAX {
            break;
        }

        // remove the merged partition point
        parts.remove(min_idx + 1);
        rank_cache.remove(min_idx + 1);

        // recompute rank for the merged element
        rank_cache[min_idx] = if min_idx + 2 < parts.len() {
            ranks
                .get(&piece[parts[min_idx]..parts[min_idx + 2]])
                .copied()
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };

        // recompute rank for predecessor
        if min_idx > 0 {
            rank_cache[min_idx - 1] = if min_idx + 1 < parts.len() {
                ranks
                    .get(&piece[parts[min_idx - 1]..parts[min_idx + 1]])
                    .copied()
                    .unwrap_or(u32::MAX)
            } else {
                u32::MAX
            };
        }
    }

    parts
}

/// BPE-encode a piece, writing tokens directly to result
fn bpe_encode(piece: &[u8], ranks: &FxHashMap<Vec<u8>, u32>, result: &mut Vec<u32>) {
    if piece.len() == 1 {
        result.push(*ranks.get(piece).expect("single byte not in ranks"));
        return;
    }

    let parts = byte_pair_merge(piece, ranks);

    for i in 0..parts.len() - 1 {
        let key = &piece[parts[i]..parts[i + 1]];
        result.push(*ranks.get(key).expect("merged token not in ranks"));
    }
}

/// BPE-count a piece without allocating a token vector
fn bpe_count(piece: &[u8], ranks: &FxHashMap<Vec<u8>, u32>) -> usize {
    if piece.len() == 1 {
        return 1;
    }
    byte_pair_merge(piece, ranks).len() - 1
}

/// Emulates `\s+(?!\S)|\s+` from original tiktoken patterns.
/// Pure byte-level fast path for ASCII whitespace, char-level fallback for Unicode.
#[inline]
fn adjust_whitespace_end(bytes: &[u8], start: usize, end: usize) -> usize {
    if end - start <= 1 || end >= bytes.len() {
        return end;
    }

    // fast reject: if first byte is printable ASCII (0x21..0x7E), not whitespace
    let first = bytes[start];
    if first > 0x20 && first < 0x7F {
        return end;
    }

    // ASCII fast path
    let piece = &bytes[start..end];
    if piece.iter().all(|&b| is_ascii_ws(b)) {
        let next = bytes[end];
        if is_ascii_ws(next) {
            return end;
        }
        return end - 1;
    }

    // unicode slow path
    let matched = std::str::from_utf8(&bytes[start..end]).unwrap();
    if !matched.chars().all(|c| c.is_whitespace()) {
        return end;
    }
    let tail = std::str::from_utf8(&bytes[end..]).unwrap();
    let next_char = match tail.chars().next() {
        Some(c) => c,
        None => return end,
    };
    if next_char.is_whitespace() {
        return end;
    }
    let last_len = matched.chars().next_back().unwrap().len_utf8();
    end - last_len
}

#[inline(always)]
const fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bpe() -> CoreBpe {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"b".to_vec(), 1);
        encoder.insert(b"c".to_vec(), 2);
        encoder.insert(b"ab".to_vec(), 3);
        encoder.insert(b"bc".to_vec(), 4);
        encoder.insert(b"abc".to_vec(), 5);

        let special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        CoreBpe::new(encoder, special, r"\w+|\S")
    }

    #[test]
    fn test_byte_pair_merge_single_byte() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"x".to_vec(), 0);
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        let tokens = bpe.encode("x");
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn test_byte_pair_merge_known_pair() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        ranks.insert(b"ab".to_vec(), 2);
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        let tokens = bpe.encode("ab");
        assert_eq!(tokens, vec![2]);
    }

    #[test]
    fn test_byte_pair_merge_unknown_pair() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w|\S");
        let tokens = bpe.encode("ab");
        assert_eq!(tokens, vec![0, 1]);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let bpe = make_test_bpe();
        let tokens = bpe.encode("abc");
        let decoded = bpe.decode(&tokens);
        assert_eq!(&decoded, b"abc");
    }

    #[test]
    fn test_count_matches_encode_len() {
        let bpe = make_test_bpe();
        let text = "abc";
        assert_eq!(bpe.count(text), bpe.encode(text).len());
    }

    #[test]
    fn test_empty_input() {
        let bpe = make_test_bpe();
        assert_eq!(bpe.encode(""), Vec::<u32>::new());
        assert_eq!(bpe.count(""), 0);
        assert_eq!(bpe.decode(&[]), Vec::<u8>::new());
    }

    #[test]
    fn test_multi_step_bpe_merge() {
        // ranks: d=0, e=1, f=2, de=3, ef=4, def=5
        // "def" is NOT in the vocab, so BPE must merge d+e→de, then de+f→def won't work
        // because def(5) > ef(4), so it tries ef first → d + ef
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"d".to_vec(), 0);
        ranks.insert(b"e".to_vec(), 1);
        ranks.insert(b"f".to_vec(), 2);
        ranks.insert(b"de".to_vec(), 3);
        ranks.insert(b"ef".to_vec(), 4);
        // no "def" in ranks — forces multi-step merge
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        let tokens = bpe.encode("def");
        // BPE picks lowest rank pair: de(3) < ef(4), so merge d+e → de, then de + f
        assert_eq!(tokens, vec![3, 2]); // de=3, f=2
    }

    // adjust_whitespace_end

    #[test]
    fn test_adjust_whitespace_single_byte() {
        // single byte piece — should not trim
        assert_eq!(adjust_whitespace_end(b"a b", 0, 1), 1);
    }

    #[test]
    fn test_adjust_whitespace_at_end_of_input() {
        // end == bytes.len() — no next byte to check
        assert_eq!(adjust_whitespace_end(b"  ", 0, 2), 2);
    }

    #[test]
    fn test_adjust_whitespace_non_ws_piece() {
        // piece starts with printable ASCII — fast reject
        assert_eq!(adjust_whitespace_end(b"hello world", 0, 5), 5);
    }

    #[test]
    fn test_adjust_whitespace_trim_before_nonws() {
        // "  " followed by 'x' — should trim trailing space
        let bytes = b"  x";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2), 1);
    }

    #[test]
    fn test_adjust_whitespace_no_trim_before_ws() {
        // "  " followed by ' ' — no trim needed
        let bytes = b"   ";
        assert_eq!(adjust_whitespace_end(bytes, 0, 2), 2);
    }

    #[test]
    fn test_adjust_whitespace_unicode_slow_path() {
        // U+3000 (IDEOGRAPHIC SPACE) = 3 bytes: E3 80 80
        // two ideographic spaces followed by ASCII 'x'
        let input = "\u{3000}\u{3000}x";
        let bytes = input.as_bytes(); // [E3,80,80, E3,80,80, 78]
        // piece is first 6 bytes (two ideographic spaces), end=6, next byte is 'x' (non-ws)
        // should trim last unicode ws char (3 bytes)
        assert_eq!(adjust_whitespace_end(bytes, 0, 6), 3);
    }

    #[test]
    fn test_adjust_whitespace_unicode_followed_by_unicode_ws() {
        // two ideographic spaces followed by another ideographic space → no trim
        let input = "\u{3000}\u{3000}\u{3000}";
        let bytes = input.as_bytes(); // [E3,80,80, E3,80,80, E3,80,80]
        // piece is first 6 bytes, next bytes start another ws char → no trim
        assert_eq!(adjust_whitespace_end(bytes, 0, 6), 6);
    }

    #[test]
    fn test_adjust_whitespace_mixed_not_all_ws() {
        // first byte is high (0xE3) but the piece isn't all whitespace
        // "日x" — not whitespace, should return end unchanged
        let input = "日x";
        let bytes = input.as_bytes(); // [E6,97,A5, 78]
        assert_eq!(adjust_whitespace_end(bytes, 0, 3), 3);
    }

    // decode_to_string error path

    #[test]
    fn test_decode_to_string_invalid_utf8() {
        // construct a BPE where token 0 maps to invalid UTF-8 bytes
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(vec![0xFF, 0xFE], 0);
        let bpe = CoreBpe::new(encoder, FxHashMap::default(), r"[\s\S]+");
        assert!(bpe.decode_to_string(&[0]).is_err());
    }

    // special token encoding

    #[test]
    fn test_encode_with_special_tokens_no_specials() {
        // when no special tokens are defined, should behave like encode
        let bpe = make_test_bpe();
        assert_eq!(bpe.encode("abc"), bpe.encode_with_special_tokens("abc"));
    }

    #[test]
    fn test_encode_with_special_tokens_mixed() {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"x".to_vec(), 0);
        encoder.insert(b"y".to_vec(), 1);
        let mut special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        special.insert(b"<|end|>".to_vec(), 99);
        let bpe = CoreBpe::new(encoder, special, r"\w|\S");
        let tokens = bpe.encode_with_special_tokens("x<|end|>y");
        assert_eq!(tokens, vec![0, 99, 1]);
    }

    #[test]
    fn test_decode_includes_special_tokens() {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"hi".to_vec(), 0);
        let mut special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        special.insert(b"<|end|>".to_vec(), 99);
        let bpe = CoreBpe::new(encoder, special, r"\w+|\S");
        // decode should include special token bytes
        let decoded = bpe.decode(&[0, 99]);
        assert_eq!(&decoded, b"hi<|end|>");
    }

    #[test]
    fn test_bpe_merge_full_collapse() {
        // piece "abc" merges all the way down: a+b→ab, ab+c→abc
        // but "abc" is NOT in the direct encoder; only its merge chain is
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 10);
        ranks.insert(b"b".to_vec(), 20);
        ranks.insert(b"c".to_vec(), 30);
        ranks.insert(b"ab".to_vec(), 5); // lowest rank → merged first
        ranks.insert(b"abc".to_vec(), 3); // then ab+c → abc
        // use a per-char regex so "abcd" matches as one piece
        let bpe = CoreBpe::new(ranks.clone(), FxHashMap::default(), r"\w+|\S");
        let tokens = bpe.encode("abc");
        // direct lookup hits since "abc" IS in the encoder
        assert_eq!(tokens, vec![3]);

        // now test through bpe_encode by using a 4-char word where
        // only sub-pieces are in vocab
        ranks.insert(b"d".to_vec(), 40);
        ranks.insert(b"cd".to_vec(), 7);
        ranks.insert(b"abcd".to_vec(), 1); // full merge rank
        let bpe2 = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        let tokens2 = bpe2.encode("abcd");
        // "abcd" is in encoder → direct lookup
        assert_eq!(tokens2, vec![1]);
    }

    #[test]
    fn test_bpe_merge_to_two_parts() {
        // force byte_pair_merge to merge until parts.len() == 2
        // ranks: a=10, b=20, c=30, ab=5, abc=2
        // "abc" not in direct regex lookup won't help since it IS in encoder
        // so we use a 4-byte piece: "abcx" not in encoder
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 10);
        ranks.insert(b"b".to_vec(), 20);
        ranks.insert(b"c".to_vec(), 30);
        ranks.insert(b"x".to_vec(), 40);
        ranks.insert(b"ab".to_vec(), 5);
        ranks.insert(b"abc".to_vec(), 2);
        ranks.insert(b"cx".to_vec(), 15);
        ranks.insert(b"abcx".to_vec(), 1); // full collapse
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        // "abcx" in encoder → direct hit
        assert_eq!(bpe.encode("abcx"), vec![1]);
    }

    #[test]
    fn test_bpe_count_single_byte_fallback() {
        // create a case where count() and encode() go through bpe path for multi-byte
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        // no "ab" in ranks — regex matches "ab" but encoder lookup fails
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        assert_eq!(bpe.count("ab"), 2);
        assert_eq!(bpe.encode("ab"), vec![0, 1]);
    }

    #[test]
    fn test_encode_with_special_tokens_trailing_text() {
        // text after the last special token
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"b".to_vec(), 1);
        let mut special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        special.insert(b"<|s|>".to_vec(), 99);
        let bpe = CoreBpe::new(encoder, special, r"\w|\S");
        let tokens = bpe.encode_with_special_tokens("a<|s|>b");
        assert_eq!(tokens, vec![0, 99, 1]);
    }
}
