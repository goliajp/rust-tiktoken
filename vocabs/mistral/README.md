# tiktoken-vocab-mistral

Mistral AI tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `mistral_v3` | mistral_v3 (Tekken, 131k) |
| `mistral_v15` | mistral_v15 (Tekken v15, Mistral Large 4) |

Source: mistralai/Mistral-Nemo-Base-2407 and mistralai/Mistral-Large-4-1T-A52B on Hugging Face. Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data stays under its publisher's license: see LICENSE and NOTICE.
