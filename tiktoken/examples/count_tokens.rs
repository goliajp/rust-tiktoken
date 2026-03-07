// token counting with and without special tokens
//
// demonstrates:
// - zero-alloc `count()` vs `encode().len()` — same result, different performance
// - `count_with_special_tokens()` for text containing special token sequences
// - how special tokens affect token count
// - counting across different encodings for the same text

fn main() {
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();

    // basic counting: count() is the zero-alloc fast path
    println!("=== zero-alloc count() vs encode().len() ===\n");

    let texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "fn main() { println!(\"hello\"); }",
        "你好世界！这是一个测试。",
        "🚀🎉🌍 emoji test 🤖",
        &"word ".repeat(1000), // longer text to show the difference matters more at scale
    ];

    for text in &texts {
        let count = enc.count(text);
        let encode_len = enc.encode(text).len();

        // these always produce the same result
        assert_eq!(count, encode_len);

        let display = if text.len() > 60 {
            format!("{}... ({} chars)", &text[..57], text.len())
        } else {
            format!("{text:?}")
        };
        println!("  {display:<65} → {count} tokens");
    }

    println!("\n  note: count() avoids allocating a Vec<u32>, making it faster");
    println!("  for cases where you only need the token count, not the token ids");

    // special tokens: the key difference between count and count_with_special_tokens
    println!("\n=== special tokens: count() vs count_with_special_tokens() ===\n");

    let text_with_special = "hello<|endoftext|>world";

    let count_normal = enc.count(text_with_special);
    let count_special = enc.count_with_special_tokens(text_with_special);

    println!("  text: {text_with_special:?}");
    println!(
        "  count()                      → {count_normal} tokens (special tokens treated as text)"
    );
    println!("  count_with_special_tokens()  → {count_special} tokens (special tokens recognized)");
    println!();

    // show what happens at the token level
    let tokens_normal = enc.encode(text_with_special);
    let tokens_special = enc.encode_with_special_tokens(text_with_special);

    println!("  encode()                     → {tokens_normal:?}");
    println!("  encode_with_special_tokens() → {tokens_special:?}");
    println!("  (100257 is the <|endoftext|> special token id)");

    // verify consistency between count and encode for both modes
    assert_eq!(count_normal, tokens_normal.len());
    assert_eq!(count_special, tokens_special.len());

    // multiple special tokens in sequence
    println!("\n=== multiple special tokens ===\n");

    let multi_special = "start<|endoftext|>middle<|endoftext|>end";
    let c_normal = enc.count(multi_special);
    let c_special = enc.count_with_special_tokens(multi_special);
    println!("  text: {multi_special:?}");
    println!("  count()                     → {c_normal} tokens");
    println!("  count_with_special_tokens() → {c_special} tokens");

    // compare token counts across different encodings for the same text
    println!("\n=== same text, different encodings ===\n");

    let sample =
        "Rust is a systems programming language focused on safety, speed, and concurrency.";
    println!("  text: {sample:?}\n");

    let encoding_names = [
        ("cl100k_base", "gpt-4, gpt-3.5"),
        ("o200k_base", "gpt-4o, o1, o3"),
        ("p50k_base", "text-davinci"),
        ("r50k_base", "gpt-3 era"),
        ("llama3", "llama 3.x"),
        ("deepseek_v3", "deepseek"),
        ("qwen2", "qwen 2.5/3"),
        ("mistral_v3", "mistral/mixtral"),
    ];

    for (name, models) in encoding_names {
        let e = tiktoken::get_encoding(name).unwrap();
        let count = e.count(sample);
        println!("  {name:<14} ({models:<16}) → {count:>3} tokens");
    }

    // llama3 special tokens (different from openai)
    println!("\n=== provider-specific special tokens ===\n");

    let llama_enc = tiktoken::get_encoding("llama3").unwrap();
    let llama_text = "hello<|begin_of_text|>world";
    let c1 = llama_enc.count(llama_text);
    let c2 = llama_enc.count_with_special_tokens(llama_text);
    println!("  llama3 text: {llama_text:?}");
    println!("  count()                     → {c1} tokens");
    println!("  count_with_special_tokens() → {c2} tokens");

    let qwen_enc = tiktoken::get_encoding("qwen2").unwrap();
    let qwen_text = "<|im_start|>user\nWhat is Rust?<|im_end|>";
    let c1 = qwen_enc.count(qwen_text);
    let c2 = qwen_enc.count_with_special_tokens(qwen_text);
    println!("\n  qwen2 text: {qwen_text:?}");
    println!("  count()                     → {c1} tokens");
    println!("  count_with_special_tokens() → {c2} tokens");

    let mistral_enc = tiktoken::get_encoding("mistral_v3").unwrap();
    let mistral_text = "[INST]What is Rust?[/INST]";
    let c1 = mistral_enc.count(mistral_text);
    let c2 = mistral_enc.count_with_special_tokens(mistral_text);
    println!("\n  mistral_v3 text: {mistral_text:?}");
    println!("  count()                     → {c1} tokens");
    println!("  count_with_special_tokens() → {c2} tokens");

    println!("\ndone.");
}
