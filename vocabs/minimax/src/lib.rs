//! MiniMax tokenizer vocabularies for the [`tiktoken`](https://docs.rs/tiktoken)
//! crate, as compressed TKV1 frames (see `tiktoken::encoding::parse_tkv`).
//!
//! This crate only carries data; depend on `tiktoken` and enable its `vocab-*`
//! features instead of using it directly. Each vocabulary is behind a feature of
//! its own name and stays under its publisher's license: see LICENSE.

#![no_std]

/// minimax_m2 (MiniMax M2 and M3).
#[cfg(feature = "minimax_m2")]
pub const MINIMAX_M2: &[u8] = include_bytes!("../data/minimax_m2.tkv.zst");
