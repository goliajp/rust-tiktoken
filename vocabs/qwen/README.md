# tiktoken-vocab-qwen

Alibaba Qwen tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `qwen2` | qwen2 (Qwen 2 through Qwen 3) |
| `qwen3_5` | qwen3_5 (Qwen 3.5 through 3.8) |

Source: Qwen/Qwen2.5-7B and Qwen/Qwen3.5-27B on Hugging Face. Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data is licensed Apache-2.0 by its publisher (see LICENSE).
