//! Mistral AI tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE and NOTICE.

#![no_std]

/// mistral_v3 (Tekken, 131k).
#[cfg(feature = "mistral_v3")]
pub const MISTRAL_V3: &[u8] = include_bytes!("../data/mistral_v3.tkv.zst");

/// mistral_v15 (Tekken v15, Mistral Large 4).
#[cfg(feature = "mistral_v15")]
pub const MISTRAL_V15: &[u8] = include_bytes!("../data/mistral_v15.tkv.zst");
