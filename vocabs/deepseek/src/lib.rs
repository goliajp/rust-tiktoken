//! DeepSeek tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE and NOTICE.

#![no_std]

/// deepseek_v3 and deepseek_v4.
#[cfg(feature = "deepseek_v3")]
pub const DEEPSEEK_V3: &[u8] = include_bytes!("../data/deepseek_v3.tkv.zst");
