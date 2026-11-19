# tiktoken-vocab-openai

OpenAI tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `cl100k_base` | cl100k_base, also llama3's base |
| `o200k_base` | o200k_base and o200k_harmony |
| `r50k_base` | r50k_base (gpt2), also p50k_base's base |
| `p50k_base` | the tokens p50k_base adds beyond r50k_base |

Source: openai/tiktoken (https://github.com/openai/tiktoken). Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data is licensed MIT by its publisher (see LICENSE).
