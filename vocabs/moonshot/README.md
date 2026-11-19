# tiktoken-vocab-moonshot

Moonshot AI Kimi tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `kimi_k2` | kimi_k2 and kimi_k3 |

Source: moonshotai/Kimi-K2-Instruct on Hugging Face (tiktoken.model). Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data stays under its publisher's license: see LICENSE.
