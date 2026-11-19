# tiktoken-vocab-deepseek

DeepSeek tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `deepseek_v3` | deepseek_v3 and deepseek_v4 |

Source: deepseek-ai/DeepSeek-V3 on Hugging Face. Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data stays under its publisher's license: see LICENSE and NOTICE.
