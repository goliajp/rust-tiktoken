//! Moonshot AI Kimi tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE.

#![no_std]

/// kimi_k2 and kimi_k3.
#[cfg(feature = "kimi_k2")]
pub const KIMI_K2: &[u8] = include_bytes!("../data/kimi_k2.tkv.zst");
