//! Zhipu AI GLM tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE.

#![no_std]

/// glm4, also glm5's base.
#[cfg(feature = "glm4")]
pub const GLM4: &[u8] = include_bytes!("../data/glm4.tkv.zst");

/// the tokens glm5 adds beyond glm4.
#[cfg(feature = "glm5")]
pub const GLM5_TAIL: &[u8] = include_bytes!("../data/glm5.tkv.zst");
