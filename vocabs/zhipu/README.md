# tiktoken-vocab-zhipu

Zhipu AI GLM tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `glm4` | glm4, also glm5's base |
| `glm5` | the tokens glm5 adds beyond glm4 |

Source: zai-org/GLM-4.5 and zai-org/GLM-5.2 on Hugging Face. Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data is licensed MIT by its publisher (see LICENSE).
