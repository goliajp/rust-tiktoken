//! OpenAI tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE.

#![no_std]

/// cl100k_base, also llama3's base.
#[cfg(feature = "cl100k_base")]
pub const CL100K_BASE: &[u8] = include_bytes!("../data/cl100k_base.tkv.zst");

/// o200k_base and o200k_harmony.
#[cfg(feature = "o200k_base")]
pub const O200K_BASE: &[u8] = include_bytes!("../data/o200k_base.tkv.zst");

/// r50k_base (gpt2), also p50k_base's base.
#[cfg(feature = "r50k_base")]
pub const R50K_BASE: &[u8] = include_bytes!("../data/r50k_base.tkv.zst");

/// the tokens p50k_base adds beyond r50k_base.
#[cfg(feature = "p50k_base")]
pub const P50K_BASE_TAIL: &[u8] = include_bytes!("../data/p50k_base.tkv.zst");
