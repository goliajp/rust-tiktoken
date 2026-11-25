use crate::CoreBpe;

impl CoreBpe {
    /// The bytes of one token id, special tokens included; `None` for an id
    /// the encoding does not have.
    ///
    /// Borrowed from the vocabulary, so it costs no allocation: summing the
    /// lengths over the ids of [`encode`](Self::encode) gives each token's
    /// byte offset in the text. A token can end inside a multi-byte
    /// character, so check `str::is_char_boundary` before slicing the text at
    /// such an offset.
    ///
    /// # Examples
    ///
    /// ```
    /// let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    /// assert_eq!(enc.token_bytes(15339), Some(&b"hello"[..]));
    /// assert_eq!(enc.token_bytes(100257), Some(&b"<|endoftext|>"[..]));
    /// assert_eq!(enc.token_bytes(u32::MAX), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        match self.vocab.try_decode(id) {
            Some(bytes) => Some(bytes),
            None => self.special_decoder.get(&id).map(Vec::as_slice),
        }
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
            if let Some(bytes) = self.token_bytes(token) {
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
}

#[cfg(test)]
mod tests {
    #[test]
    fn token_bytes_rebuild_the_text() {
        for name in ["cl100k_base", "o200k_base"] {
            let enc = crate::get_encoding(name).unwrap();
            let text = "Hello, 世界! naïve café — 12345 <|endoftext|>";
            let ids = enc.encode_with_special_tokens(text);
            let mut rebuilt = Vec::new();
            for &id in &ids {
                rebuilt.extend_from_slice(enc.token_bytes(id).unwrap());
            }
            assert_eq!(rebuilt, text.as_bytes(), "{name}");
            assert_eq!(enc.decode(&ids), text.as_bytes(), "{name}");
        }
    }
}
