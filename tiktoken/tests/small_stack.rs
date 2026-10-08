// The per-thread piece cache is about 1.2 MB. It must be allocated on the heap
// directly; building it on the stack first overflows threads with small stacks
// (async runtime workers, rayon pools) the first time they hit a cache miss.

#[test]
fn first_cache_miss_on_small_stack() {
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let count = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || enc.count("zxqvbnmlkj unbelievablyxyzzy naïveté"))
        .unwrap()
        .join()
        .unwrap();
    assert!(count > 0);
}
