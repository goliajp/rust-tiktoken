//! Alibaba Qwen tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE.

#![no_std]

/// qwen2 (Qwen 2 through Qwen 3).
#[cfg(feature = "qwen2")]
pub const QWEN2: &[u8] = include_bytes!("../data/qwen2.tkv.zst");

/// qwen3_5 (Qwen 3.5 through 3.8).
#[cfg(feature = "qwen3_5")]
pub const QWEN3_5: &[u8] = include_bytes!("../data/qwen3_5.tkv.zst");
