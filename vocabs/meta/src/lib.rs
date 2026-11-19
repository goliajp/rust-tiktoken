//! Meta tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE and NOTICE.

#![no_std]

/// the tokens llama3 adds beyond cl100k_base.
#[cfg(feature = "llama3")]
pub const LLAMA3_TAIL: &[u8] = include_bytes!("../data/llama3.tkv.zst");

/// llama4 (Llama 4 and Meta's Muse models).
#[cfg(feature = "llama4")]
pub const LLAMA4: &[u8] = include_bytes!("../data/llama4.tkv.zst");
