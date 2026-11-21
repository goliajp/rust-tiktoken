use rustc_hash::FxHashMap;

use crate::encode_loop::Engine;
use crate::piece_cache;
use crate::pretokenize::{FastPath, ScalarPreTokenizer, WhitespaceRules};
use crate::special::SpecialMatcher;
use crate::vocab::Vocab;
#[cfg(not(target_arch = "wasm32"))]
use crate::{pool, split};

/// Texts at least this long are cut into chunks and encoded on the worker
/// pool (see [`set_parallelism`](crate::set_parallelism)).
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_MIN: usize = 64 * 1024;

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
    /// Identity for the thread-local piece cache — a nonce, never an address.
    cache_id: u64,
    special_encoder: FxHashMap<Vec<u8>, u32>,
    special_decoder: FxHashMap<u32, Vec<u8>>,
    pre_tokenizer: ScalarPreTokenizer,
    special_matcher: Option<SpecialMatcher>,
}

impl CoreBpe {
    pub(crate) fn new(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_encoder: FxHashMap<Vec<u8>, u32>,
        pattern: &str,
        fast: FastPath,
        ws: WhitespaceRules,
    ) -> Self {
        let pre_tokenizer = ScalarPreTokenizer::new(pattern, fast, ws);
        let special_matcher = SpecialMatcher::new(&special_encoder);

        let special_decoder = special_encoder
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        let entries: Vec<(Vec<u8>, u32)> = encoder.into_iter().collect();
        let vocab = Vocab::from_entries(entries);

        Self {
            vocab,
            cache_id: piece_cache::new_cache_id(),
            special_encoder,
            special_decoder,
            pre_tokenizer,
            special_matcher,
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
        #[cfg(not(target_arch = "wasm32"))]
        if text.len() >= PARALLEL_MIN {
            let mut result = Vec::new();
            self.encode_chunked(text, split::CHUNK, None, &mut result);
            return result;
        }
        let mut result = Vec::with_capacity(text.len() / 3);
        self.engine().encode_into(text, &mut result);
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
        let special = match &self.special_matcher {
            Some(m) => m,
            None => return self.encode(text),
        };

        let mut result = Vec::with_capacity(text.len() / 3);
        let mut start = 0;

        while let Some((s, e, token)) = special.find(text.as_bytes(), start) {
            if s > start {
                self.encode_into(&text[start..s], &mut result);
            }
            result.push(token);
            start = e;
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
        #[cfg(not(target_arch = "wasm32"))]
        if text.len() >= PARALLEL_MIN {
            return self.count_chunked(text, split::CHUNK, None);
        }
        self.engine().count(text)
    }

    /// Count tokens, recognizing special tokens.
    ///
    /// This is the counting equivalent of
    /// [`encode_with_special_tokens`](Self::encode_with_special_tokens).
    #[must_use]
    pub fn count_with_special_tokens(&self, text: &str) -> usize {
        let special = match &self.special_matcher {
            Some(m) => m,
            None => return self.count(text),
        };

        let mut total = 0;
        let mut start = 0;

        while let Some((s, e, _)) = special.find(text.as_bytes(), start) {
            if s > start {
                total += self.count(&text[start..s]);
            }
            total += 1; // special token = 1 token
            start = e;
        }

        if start < text.len() {
            total += self.count(&text[start..]);
        }

        total
    }

    /// Same as [`encode`](Self::encode), which already spreads long texts
    /// over several threads; kept for code written against older versions.
    #[must_use]
    pub fn encode_parallel(&self, text: &str) -> Vec<u32> {
        self.encode(text)
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

    #[cfg(test)]
    pub(crate) fn special_tokens(&self) -> &FxHashMap<Vec<u8>, u32> {
        &self.special_encoder
    }

    #[cfg(test)]
    pub(crate) fn pre_tokenizer(&self) -> &ScalarPreTokenizer {
        &self.pre_tokenizer
    }

    fn engine(&self) -> Engine<'_> {
        Engine {
            vocab: &self.vocab,
            pre: &self.pre_tokenizer,
            cache_id: self.cache_id,
        }
    }

    fn encode_into(&self, text: &str, result: &mut Vec<u32>) {
        #[cfg(not(target_arch = "wasm32"))]
        if text.len() >= PARALLEL_MIN {
            self.encode_chunked(text, split::CHUNK, None, result);
            return;
        }
        self.engine().encode_into(text, result);
    }

    /// Thread count and chunk boundaries for a parallel run, or `None` to
    /// stay on this thread. At most `max_chunks` chunks, three per thread
    /// by default.
    #[cfg(not(target_arch = "wasm32"))]
    fn plan(text: &str, chunk: usize, max_chunks: Option<usize>) -> Option<(usize, Vec<usize>)> {
        let threads = pool::threads();
        if threads <= 1 {
            return None;
        }
        let max_chunks = max_chunks.unwrap_or(threads * 3);
        let cuts = split::cuts(text.as_bytes(), chunk, max_chunks);
        (cuts.len() > 2).then_some((threads, cuts))
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline(never)]
    fn encode_chunked(
        &self,
        text: &str,
        chunk: usize,
        max_chunks: Option<usize>,
        result: &mut Vec<u32>,
    ) {
        let engine = self.engine();
        if let Some((threads, cuts)) = Self::plan(text, chunk, max_chunks) {
            let parts = pool::run(cuts.len() - 1, threads, |i| {
                let part = &text[cuts[i]..cuts[i + 1]];
                let mut out = Vec::with_capacity(part.len() / 3);
                engine.encode_into(part, &mut out);
                out
            });
            if let Some(parts) = parts {
                pool::concat_into(&parts, threads, result);
                return;
            }
        }
        result.reserve(text.len() / 3);
        engine.encode_into(text, result);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline(never)]
    fn count_chunked(&self, text: &str, chunk: usize, max_chunks: Option<usize>) -> usize {
        let engine = self.engine();
        if let Some((threads, cuts)) = Self::plan(text, chunk, max_chunks) {
            let parts = pool::run(cuts.len() - 1, threads, |i| {
                engine.count(&text[cuts[i]..cuts[i + 1]])
            });
            if let Some(parts) = parts {
                return parts.iter().sum();
            }
        }
        engine.count(text)
    }

    /// [`encode`](Self::encode)'s parallel path at any length, with chunks of
    /// about `chunk` bytes and no cap on their number. For tests and
    /// benchmarks; not a stable API.
    #[doc(hidden)]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn __encode_chunks(&self, text: &str, chunk: usize) -> Vec<u32> {
        let mut result = Vec::new();
        self.encode_chunked(text, chunk, Some(usize::MAX), &mut result);
        result
    }

    /// The counting twin of [`__encode_chunks`](Self::__encode_chunks).
    #[doc(hidden)]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn __count_chunks(&self, text: &str, chunk: usize) -> usize {
        self.count_chunked(text, chunk, Some(usize::MAX))
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
        CoreBpe::new(
            encoder,
            special,
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        )
    }

    #[test]
    fn test_byte_pair_merge_single_byte() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"x".to_vec(), 0);
        let bpe = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        let tokens = bpe.encode("x");
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn test_byte_pair_merge_known_pair() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        ranks.insert(b"ab".to_vec(), 2);
        let bpe = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        let tokens = bpe.encode("ab");
        assert_eq!(tokens, vec![2]);
    }

    #[test]
    fn test_byte_pair_merge_unknown_pair() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        let bpe = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
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
        let bpe = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        let tokens = bpe.encode("def");
        assert_eq!(tokens, vec![3, 2]); // de=3, f=2
    }

    #[test]
    fn test_decode_to_string_invalid_utf8() {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(vec![0xFF, 0xFE], 0);
        let bpe = CoreBpe::new(
            encoder,
            FxHashMap::default(),
            r"[\s\S]+",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
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
        let bpe = CoreBpe::new(
            encoder,
            special,
            r"\w|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        let tokens = bpe.encode_with_special_tokens("x<|end|>y");
        assert_eq!(tokens, vec![0, 99, 1]);
    }

    #[test]
    fn test_decode_includes_special_tokens() {
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"hi".to_vec(), 0);
        let mut special: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        special.insert(b"<|end|>".to_vec(), 99);
        let bpe = CoreBpe::new(
            encoder,
            special,
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
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
        let bpe = CoreBpe::new(
            ranks.clone(),
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        let tokens = bpe.encode("abc");
        assert_eq!(tokens, vec![3]);

        ranks.insert(b"d".to_vec(), 40);
        ranks.insert(b"cd".to_vec(), 7);
        ranks.insert(b"abcd".to_vec(), 1);
        let bpe2 = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
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
        let bpe = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        assert_eq!(bpe.encode("abcx"), vec![1]);
    }

    #[test]
    fn test_bpe_count_single_byte_fallback() {
        let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        let bpe = CoreBpe::new(
            ranks,
            FxHashMap::default(),
            r"\w+|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
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
        let bpe = CoreBpe::new(
            encoder,
            special,
            r"\w|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );

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
        let bpe = CoreBpe::new(
            encoder,
            special,
            r"\w|\S",
            FastPath::P50k,
            WhitespaceRules::Generic,
        );
        let tokens = bpe.encode_with_special_tokens("a<|s|>b");
        assert_eq!(tokens, vec![0, 99, 1]);
    }
}
