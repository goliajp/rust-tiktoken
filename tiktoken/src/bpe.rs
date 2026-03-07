use regex::Regex;
use rustc_hash::FxHashMap;

use crate::merge;
use crate::pretokenize::{PreTokenizer, RegexPreTokenizer};
use crate::vocab::Vocab;

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
    vocab: Vocab,
    special_encoder: FxHashMap<Vec<u8>, u32>,
    special_decoder: FxHashMap<u32, Vec<u8>>,
    pre_tokenizer: RegexPreTokenizer,
    special_regex: Option<Regex>,
}

impl CoreBpe {
    pub(crate) fn new(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_encoder: FxHashMap<Vec<u8>, u32>,
        pattern: &str,
    ) -> Self {
        let pre_tokenizer = RegexPreTokenizer::new(pattern);

        let special_regex = if special_encoder.is_empty() {
            None
        } else {
            debug_assert!(
                special_encoder.keys().all(|k| !k.is_empty()),
                "special token keys must not be empty"
            );
            let pat = special_encoder
                .keys()
                .map(|k| regex::escape(&String::from_utf8_lossy(k)))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&pat).expect("invalid special regex"))
        };

        let special_decoder = special_encoder
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        let entries: Vec<(Vec<u8>, u32)> = encoder.into_iter().collect();
        let vocab = Vocab::from_entries(entries);

        Self {
            vocab,
            special_encoder,
            special_decoder,
            pre_tokenizer,
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
            } else {
                debug_assert!(
                    false,
                    "special regex matched {:?} but no encoder entry found",
                    String::from_utf8_lossy(piece)
                );
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
            if let Some(bytes) = self.vocab.try_decode(token) {
                result.extend_from_slice(bytes);
                continue;
            }
            if let Some(bytes) = self.special_decoder.get(&token) {
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
    /// Note: special tokens (e.g. `<|endoftext|>`) are treated as ordinary text,
    /// matching the behavior of [`encode`](Self::encode). Use
    /// [`count_with_special_tokens`](Self::count_with_special_tokens) if you
    /// need special tokens to be recognized.
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
        let mut pos = 0;

        while let Some((start, end)) = self.pre_tokenizer.next_match(text, pos) {
            let piece = &text.as_bytes()[start..end];
            if self.vocab.contains_key(piece) {
                count += 1;
            } else {
                count += merge::bpe_count(piece, &self.vocab);
            }
            pos = end;
        }
        count
    }

    /// Count tokens, recognizing special tokens.
    ///
    /// This is the counting equivalent of
    /// [`encode_with_special_tokens`](Self::encode_with_special_tokens).
    #[must_use]
    pub fn count_with_special_tokens(&self, text: &str) -> usize {
        let special_regex = match &self.special_regex {
            Some(r) => r,
            None => return self.count(text),
        };

        let mut total = 0;
        let mut start = 0;

        for mat in special_regex.find_iter(text) {
            if mat.start() > start {
                total += self.count(&text[start..mat.start()]);
            }
            total += 1; // special token = 1 token
            start = mat.end();
        }

        if start < text.len() {
            total += self.count(&text[start..]);
        }

        total
    }

    /// Encode text using multiple threads for large inputs.
    ///
    /// Falls back to single-threaded encoding for texts smaller than 4 KB.
    /// Results are identical to [`encode`](Self::encode).
    ///
    /// Requires the `parallel` feature.
    #[cfg(feature = "parallel")]
    #[must_use]
    pub fn encode_parallel(&self, text: &str) -> Vec<u32> {
        use rayon::prelude::*;

        const THRESHOLD: usize = 4096;
        if text.len() < THRESHOLD {
            return self.encode(text);
        }

        // collect all pre-tokenizer match ranges
        let mut ranges = Vec::new();
        let mut pos = 0;
        while let Some((start, end)) = self.pre_tokenizer.next_match(text, pos) {
            ranges.push((start, end));
            pos = end;
        }

        // encode each piece in parallel
        let bytes = text.as_bytes();
        let chunks: Vec<Vec<u32>> = ranges
            .par_iter()
            .map(|&(start, end)| {
                let piece = &bytes[start..end];
                if let Some(token) = self.vocab.get(piece) {
                    vec![token]
                } else {
                    let mut tokens = Vec::new();
                    merge::bpe_encode(piece, &self.vocab, &mut tokens);
                    tokens
                }
            })
            .collect();

        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut result = Vec::with_capacity(total);
        for chunk in chunks {
            result.extend_from_slice(&chunk);
        }
        result
    }

    /// Number of regular (non-special) tokens in the vocabulary.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// assert_eq!(enc.vocab_size(), 100256);
    /// ```
    #[inline]
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Number of special tokens (e.g. `<|endoftext|>`).
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// assert_eq!(enc.num_special_tokens(), 5);
    /// ```
    #[inline]
    #[must_use]
    pub fn num_special_tokens(&self) -> usize {
        self.special_encoder.len()
    }

    /// Main encoding hot loop. Iterates pre-tokenizer matches over the input,
    /// then for each matched piece: (1) try direct vocab lookup (fast path for
    /// known tokens), (2) fall back to BPE merge for unknown multi-byte pieces.
    fn encode_into(&self, text: &str, result: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let mut pos = 0;

        while let Some((start, end)) = self.pre_tokenizer.next_match(text, pos) {
            let piece = &bytes[start..end];
            if let Some(token) = self.vocab.get(piece) {
                result.push(token);
            } else {
                merge::bpe_encode(piece, &self.vocab, result);
            }
            pos = end;
        }
    }
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
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"d".to_vec(), 0);
        ranks.insert(b"e".to_vec(), 1);
        ranks.insert(b"f".to_vec(), 2);
        ranks.insert(b"de".to_vec(), 3);
        ranks.insert(b"ef".to_vec(), 4);
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        let tokens = bpe.encode("def");
        assert_eq!(tokens, vec![3, 2]); // de=3, f=2
    }

    #[test]
    fn test_decode_to_string_invalid_utf8() {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(vec![0xFF, 0xFE], 0);
        let bpe = CoreBpe::new(encoder, FxHashMap::default(), r"[\s\S]+");
        assert!(bpe.decode_to_string(&[0]).is_err());
    }

    #[test]
    fn test_encode_with_special_tokens_no_specials() {
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
        let decoded = bpe.decode(&[0, 99]);
        assert_eq!(&decoded, b"hi<|end|>");
    }

    #[test]
    fn test_bpe_merge_full_collapse() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 10);
        ranks.insert(b"b".to_vec(), 20);
        ranks.insert(b"c".to_vec(), 30);
        ranks.insert(b"ab".to_vec(), 5);
        ranks.insert(b"abc".to_vec(), 3);
        let bpe = CoreBpe::new(ranks.clone(), FxHashMap::default(), r"\w+|\S");
        let tokens = bpe.encode("abc");
        assert_eq!(tokens, vec![3]);

        ranks.insert(b"d".to_vec(), 40);
        ranks.insert(b"cd".to_vec(), 7);
        ranks.insert(b"abcd".to_vec(), 1);
        let bpe2 = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        let tokens2 = bpe2.encode("abcd");
        assert_eq!(tokens2, vec![1]);
    }

    #[test]
    fn test_bpe_merge_to_two_parts() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 10);
        ranks.insert(b"b".to_vec(), 20);
        ranks.insert(b"c".to_vec(), 30);
        ranks.insert(b"x".to_vec(), 40);
        ranks.insert(b"ab".to_vec(), 5);
        ranks.insert(b"abc".to_vec(), 2);
        ranks.insert(b"cx".to_vec(), 15);
        ranks.insert(b"abcx".to_vec(), 1);
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        assert_eq!(bpe.encode("abcx"), vec![1]);
    }

    #[test]
    fn test_bpe_count_single_byte_fallback() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        let bpe = CoreBpe::new(ranks, FxHashMap::default(), r"\w+|\S");
        assert_eq!(bpe.count("ab"), 2);
        assert_eq!(bpe.encode("ab"), vec![0, 1]);
    }

    #[test]
    fn test_count_with_special_tokens() {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"x".to_vec(), 0);
        encoder.insert(b"y".to_vec(), 1);
        let mut special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        special.insert(b"<|end|>".to_vec(), 99);
        let bpe = CoreBpe::new(encoder, special, r"\w|\S");

        assert_eq!(
            bpe.count_with_special_tokens("x<|end|>y"),
            bpe.encode_with_special_tokens("x<|end|>y").len()
        );
    }

    #[test]
    fn test_count_with_special_tokens_no_specials() {
        let bpe = make_test_bpe();
        assert_eq!(bpe.count_with_special_tokens("abc"), bpe.count("abc"));
    }

    #[test]
    fn test_encode_with_special_tokens_trailing_text() {
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
