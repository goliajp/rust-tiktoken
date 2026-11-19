# tiktoken-vocab-minimax

MiniMax tokenizer vocabularies for the [`tiktoken`](https://crates.io/crates/tiktoken)
crate. This crate only carries data: depend on `tiktoken` and enable its
`vocab-*` features, which pull in the data crates they need.

| Feature | Contents |
|---|---|
| `minimax_m2` | minimax_m2 (MiniMax M2 and M3) |

Source: MiniMaxAI/MiniMax-M2 on Hugging Face. Converted from the published format into tiktoken's
compressed TKV1 format; token bytes and ranks are unchanged.

## License

The data stays under its publisher's license: see LICENSE.
