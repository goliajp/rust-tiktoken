// basic encoding and decoding across all 9 supported encodings
//
// demonstrates:
// - looking up encodings by name (`get_encoding`)
// - looking up encodings by model name (`encoding_for_model`)
// - encode → decode roundtrip
// - token inspection

fn main() {
    // a selection of the supported encodings; `list_encodings()` has them all
    let encodings = [
        "cl100k_base", // openai: gpt-4, gpt-3.5-turbo, embeddings
        "o200k_base",  // openai: gpt-4o, o1, o3, o4-mini
        "p50k_base",   // openai: text-davinci-002/003, code-davinci
        "p50k_edit",   // openai: text-davinci-edit (with FIM tokens)
        "r50k_base",   // openai: gpt-3 era (davinci, curie, babbage, ada)
        "llama3",      // meta: llama 3.x
        "llama4",      // meta: llama 4, muse
        "deepseek_v3", // deepseek: v3, r1
        "qwen2",       // alibaba: qwen 2.5, qwen 3
        "qwen3_5",     // alibaba: qwen 3.5 - 3.8
        "mistral_v3",  // mistral: mistral, mixtral (tekken tokenizer)
        "mistral_v15", // mistral: large 4 (tekken v15)
    ];

    let sample = "Hello, world! This is a test of BPE tokenization. 你好世界 🚀";

    println!("=== encode/decode across these encodings ===\n");
    println!("input: {sample:?}\n");

    for name in encodings {
        let enc = tiktoken::get_encoding(name).expect("encoding not found");
        let tokens = enc.encode(sample);
        let decoded = enc.decode_to_string(&tokens).expect("invalid utf-8");

        println!(
            "{name:<14} → {count:>3} tokens: {tokens:?}",
            count = tokens.len()
        );

        // verify roundtrip correctness
        assert_eq!(decoded, sample, "roundtrip failed for {name}");
    }

    // look up encoding by model name instead of encoding name
    println!("\n=== encoding_for_model examples ===\n");

    let models = [
        "gpt-6-sol",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5-turbo",
        "llama-4-scout",
        "llama-3.1-70b",
        "deepseek-v3",
        "qwen3.8-max",
        "qwen2.5-72b",
        "mistral-large-4",
        "mistral-large",
    ];

    for model in models {
        let encoding_name = tiktoken::model_to_encoding(model).expect("unknown model");
        let enc = tiktoken::encoding_for_model(model).expect("unknown model");
        let tokens = enc.encode(sample);
        println!(
            "{model:<20} uses {encoding_name:<14} → {count} tokens",
            count = tokens.len(),
        );
    }

    // inspect individual tokens
    println!("\n=== token-level inspection (cl100k_base) ===\n");

    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let text = "Hello, world!";
    let tokens = enc.encode(text);

    for &token_id in &tokens {
        let piece = enc
            .decode_to_string(&[token_id])
            .unwrap_or_else(|_| "<bytes>".to_string());
        println!("  token {token_id:>6} → {piece:?}");
    }

    println!("\ndone.");
}
