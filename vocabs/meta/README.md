# tiktoken-vocab-meta

Meta tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `llama3` | the tokens llama3 adds beyond cl100k_base |
| `llama4` | llama4 (Llama 4 and Meta's Muse models) |

Source: meta-llama/Meta-Llama-3-8B and meta-llama/Llama-4-Scout-17B-16E-Instruct on Hugging Face. Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data stays under its publisher's license: see LICENSE and NOTICE.
