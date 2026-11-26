# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

tiktoken は Rust 製の BPE トークナイザーで最速の実装です。ネイティブ版の速度は tiktoken-rs の 22〜83 倍（中国語と日本語の文章では 50〜82 倍）、fastokens の 1.5〜12 倍です。ブラウザで動く wasm 版は gpt-tokenizer の 5.8〜7.0 倍です。ASCII と CJK はいずれも手書きのスキャナーで分割します。語彙はキーの長さ別に格納し、同じ文字列が分割結果に再び現れたときは、その文字列全体の処理結果を再利用します。OpenAI の [tiktoken](https://github.com/openai/tiktoken) と互換性があり、主要な LLM のトークナイザー（OpenAI、Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）すべてに対応しています。

## 特徴

- **マルチベンダー**：8 社（OpenAI、Meta、DeepSeek、Alibaba、Mistral、Moonshot、Zhipu、MiniMax）の 20 種類のエンコーディングに対応
- **高速**：前処理では全エンコーディングの分割規則を表引きで処理し、実行時に正規表現を使わない。cl100k 系と o200k 系の分割位置は SIMD で求める。語彙はキーの長さ別に格納し、分割した文字列ごとに処理結果をスレッド間で共有する。BPE のマージには線形走査とヒープを使い分ける
- **並列エンコード**：32 KiB 以上のテキストは自動で複数のスレッドに分けてエンコードする。内蔵のスレッドプールは Rust 標準ライブラリだけで動作する
- **料金見積もり**：11 社が提供する 153 モデルの料金を見積もれる
- **コンパクト**：全 20 種類の語彙データは合計 7.8 MB で、不要な語彙は個別に除外できる。cl100k だけを組み込む場合は 373 KB。Cargo がダウンロードするのは有効にした提供元のデータだけ
- **メモリ確保なしのカウント**：`count()` は token 列を格納するメモリを確保せずに token 数を数える

## パフォーマンス

どの測定でも、まず各実装が返す token 列の完全一致を確認します。各値はコーパス全体を 1 回処理する時間で、ウォームアップ後に 9 回測定した中央値です。各測定環境（`bench-compare/`、`benches/bench_python.py`、`web/bench/`）では、バイト単位で同一のコーパスを使っています。

#### ネイティブ版の `encode`（Apple M4 Pro Mac mini、単一スレッド）

`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`

| コーパス | Python tiktoken 0.14 | tiktoken-rs 0.12 | **tiktoken** | 対 tiktoken-rs | 対 Python 版 |
|---|---|---|---|---|---|
| 短文（13 B） | 1.3 µs | 602 ns | **28 ns** | **22x** | **46x** |
| 中程度の長さの文章（900 B） | 23.2 µs | 19.8 µs | **431 ns** | **46x** | **54x** |
| 英語の文章（45 KB） | 1,100 µs | 955 µs | **19.9 µs** | **48x** | **55x** |
| 中国語の文章（4.3 KB） | 114 µs | 97.8 µs | **2.0 µs** | **50x** | **58x** |
| 日本語の文章（4.6 KB） | 131 µs | 119 µs | **1.4 µs** | **82x** | **91x** |
| CJK 混在 ×50（4.5 KB） | 122 µs | 108 µs | **2.5 µs** | **43x** | **49x** |
| 繰り返しのない敵対的 CJK 入力（3.9 KB） | 114 µs | 94.0 µs | **2.6 µs** | **36x** | **43x** |
| コード（3.9 KB） | 179 µs | 163 µs | **1.9 µs** | **83x** | **92x** |

o200k_base の処理速度は単一スレッドで tiktoken-rs の 12〜80 倍です。`count()` は ID 列のメモリを確保しません。短文では `encode` より最大 36% 速く、長文ではほぼ同じ速度です。既定のスレッド数では 45 KB のテキストを 6.2 µs で処理します。

#### 高速な実装との比較（Apple M4 Pro Mac mini、単一スレッド、cl100k_base、`encode`）

調査した中で tiktoken を上回る可能性のある実装をすべて比較しました。`bench-compare` とは別の測定環境を使い、同一プロセスで各実装に同じ入力を与え、ID 列の完全一致を確認してから測定しました。数値は 3 回の中央値です。`tiktoken` は crates.io で公開している 5.2.0 を使いました。

| コーパス | **tiktoken** | [fastokens](https://crates.io/crates/fastokens) 0.3.6 | [HF tokenizers](https://crates.io/crates/tokenizers) 1.0-rc.5 | [toks](https://github.com/actual-computer/toks) 0.3.3 | [toktok-rs](https://crates.io/crates/toktok-rs) 0.5 | [magi-tokenizer](https://crates.io/crates/magi-tokenizer) 0.1 | [bpe-openai](https://crates.io/crates/bpe-openai) 0.3 |
|---|---|---|---|---|---|---|---|
| 短文（13 B） | **25 ns** | 61 ns | 112 ns | 42 ns | 32 ns | 27 ns | 147 ns |
| 中程度の長さの文章（900 B） | **431 ns** | 750 ns | 733 ns | 606 ns | 1.3 µs | 1.0 µs | 8.2 µs |
| 英語の文章（45 KB） | **19.9 µs** | 35.6 µs | 32.3 µs | 29.9 µs | 61.2 µs | 49.5 µs | 405 µs |
| 中国語の文章（4.3 KB） | **2.0 µs** | 8.0 µs | 4.3 µs | 12.9 µs | 10.1 µs | 12.2 µs | 37.7 µs |
| 日本語の文章（4.6 KB） | **1.4 µs** | 17.2 µs | 4.1 µs | 18.8 µs | 15.8 µs | 22.8 µs | 41.1 µs |
| CJK 混在 ×50（4.5 KB） | **2.6 µs** | 3.9 µs | 5.8 µs | 17.8 µs | 12.7 µs | 18.9 µs | 40.0 µs |
| 繰り返しのない敵対的 CJK 入力（3.9 KB） | **2.7 µs** | 5.3 µs | 5.4 µs | 17.2 µs | 15.3 µs | 19.4 µs | 56.1 µs |
| コード（3.9 KB） | **1.9 µs** | 3.9 µs | 3.4 µs | 2.8 µs | 6.1 µs | 6.6 µs | 46.1 µs |
| 英語の書籍（200 KB） | **111 µs** | 184 µs | 205 µs | 144 µs | 341 µs | 715 µs | 2,854 µs |
| Rust ソース（100 KB） | **56.1 µs** | 95.8 µs | 97.6 µs | 79.8 µs | 181 µs | 263 µs | 1,412 µs |

- 単一スレッドでの tiktoken の処理速度は fastokens の 1.5〜12 倍。既定のスレッド数では、書籍の処理時間が tiktoken で 16.7 µs、fastokens で 50.3 µs（速度比 3.0 倍）。Rust ソースではそれぞれ 12.5 µs、36.3 µs（2.9 倍）
- 13 バイトの短文では magi-tokenizer との速度差は 4〜12%
- キャッシュを消去して単一スレッドで初めて処理する場合、tiktoken の速度は fastokens の 1.7〜6.2 倍（書籍では 228 µs と 392 µs）。13 バイトを超える入力では、ほかのどのライブラリと比べても 1.17 倍以上。13 バイトでは複数の実装がタイマーの分解能である 42 ns で並ぶ。既定のスレッド数で書籍を処理し、fastokens の worker がキャッシュを保持している場合は、tiktoken が 8〜20% 速い
- 表の toks の値は、テキスト全体のキャッシュを無効にして測定したもの。有効にすると、同じテキストには保存済みの結果を返すため、tiktoken で再エンコードする場合の 3〜20 倍速い。tiktoken で同じ用途に使えるのが [`CoreBpe::memo`](#同じテキストの再利用)。今回のコーパス（cl100k と o200k）では、保存済みの ID 列をそのまま返すと toks のキャッシュの 1.3〜6 倍の速度になる。toks と同様に ID 列をバッファにコピーする場合は、45 KB で 10% 遅く、13 バイトで 5 倍速い
- ほかに測定した実装（tiktoken-rs、tokensim、riptoken、kitoken、basetenkenizer、Hugging Face tokenizers 0.23）では、どの入力でも tiktoken の 3.4 倍以上の時間がかかった。Hugging Face 0.23 と kitoken は Rust ソースで ID 列が一致しなかったため、その入力を測定から除いた。Python からのみ呼び出せる Gigatoken の処理時間は、呼び出しのオーバーヘッドを含めて tiktoken の 1.4〜11 倍

#### ブラウザでの比較（Apple M4 Pro Mac mini、Chromium）

`web/` で `npm run bench` を実行し、この crate の wasm 版を、別の wasm トークナイザーおよび主要な JavaScript トークナイザー 2 種と比較しました。各列は `encode` の処理時間です。

| コーパス | tiktoken-fast 0.1 (wasm) | gpt-tokenizer 4.1 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|---|
| 中国語の文章（4.3 KB） | 6.4 µs | 31.9 µs | 6,625 µs | **4.8 µs** |
| 日本語の文章（4.6 KB） | 6.6 µs | 29.2 µs | 12,825 µs | **4.2 µs** |
| CJK 混在 ×50（4.5 KB） | 7.0 µs | 40.3 µs | 3,975 µs | **6.3 µs** |
| 繰り返しのない敵対的 CJK 入力（3.9 KB） | 6.7 µs | 42.2 µs | 2,756 µs | **6.1 µs** |
| 英語の文章（45 KB） | 118 µs | 277 µs | 9,638 µs | **47.7 µs** |
| コード（3.9 KB） | 8.9 µs | 30.4 µs | 1,120 µs | **5.1 µs** |

複数の文字種が混在する 2 つのコーパスでは、処理済みの入力なら tiktoken-fast と tiktoken の速度差は約 10% まで縮まります。初めて処理する入力では tiktoken-fast は tiktoken の 1.8〜6.6 倍の時間がかかります。Safari での tiktoken の処理速度は gpt-tokenizer の 3.3〜7.9 倍、tiktoken-fast の 1.5〜2.2 倍です。

敵対的コーパスでは、分割後の文字列が一度も重複しません。どの実装も保存済みの結果を再利用できないこの条件でも、tiktoken が最速です。

<details>
<summary>なぜ速いのか</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 前処理での分割 | 全エンコーディングで、各分割パターンが想定するバージョンの Unicode から表を生成し、スキャナーがその表を参照する。cl100k 系と o200k 系では SIMD（NEON、SSE2 または AVX2、wasm simd128）で 64 バイトずつ読み、分割位置を求める | 毎回正規表現を実行 | 毎回正規表現を実行 |
| 正規表現エンジン | 実行時は不使用（`regex` はスキャナーのテストで照合に使う開発用依存関係） | `fancy-regex`（バックトラッキング） | PyO3 経由で `regex` を使うため、FFI 呼び出しの負荷がかかる |
| 語彙の検索 | キーの長さ別に格納する。15 バイト以下のキーは詰めて 1 つのオープンアドレス方式の表に格納し、キャッシュラインあたり 2 スロットを使う。それより長いキーにはハッシュ値でタグ付けしたアリーナ内のスロットを使う。1〜2 バイトのキーには直接添字で引ける表も使う | `HashMap<Vec<u8>>`（メモリ確保 20 万回） | PyO3 経由で Rust の `HashMap` を使う |
| 同じ文字列の再出現 | 語彙にない文字列のマージ結果を検索表に追加し、全スレッドで再利用する。長い文字列には専用のスレッドローカルキャッシュと共有キャッシュを使う | なし | なし |
| BPE のマージ | 短い文字列はスタック上で線形走査し、長い文字列にはヒープを使う。表にない短い文字列は token ID ごとに複数まとめてマージする | O(n*m) の線形走査 | O(n*m) の線形走査 |
| メモリ確保を伴わない `count()` | 対応 | 非対応 | 非対応 |

毎回数十万件のランダム入力を使うプロパティテストと、短い文字列を網羅するテストで、スキャナーの分割結果が `regex` crate と一致するかを確認します。また、53,275 件の差分テストで、各エンコーディングの結果を提供元のトークナイザーと照合します。

測定用のソースコードは [`benches/`](benches/) と [`../bench-compare/`](../bench-compare/) にあります。`cargo bench` と `cargo run -p bench-compare` で測定を再現できます。

</details>

## インストール

```toml
[dependencies]
tiktoken = "5"
```

実行時に依存するのは `ruzstd` と、有効にした語彙データの crate だけです。従来の `parallel` feature は既存のマニフェストをそのまま使えるように残しています。この指定は動作に影響せず、長文は指定の有無にかかわらず並列にエンコードします。

### 語彙を選ぶ

既定では 20 種類のエンコーディングすべてが有効で、語彙データは合計 7.8 MB です。必要な語彙だけを組み込むには、既定の feature を無効にして、使う語彙を指定します。

```toml
# GPT-4o / GPT-5 だけ — 語彙データは 7.8 MB ではなく 815 KB
tiktoken = { version = "5", default-features = false, features = ["vocab-o200k_base"] }

# ベンダー単位でまとめて
tiktoken = { version = "5", default-features = false, features = ["vocab-openai"] }
```

Linux x86_64 で `examples/count_tokens` をリリースビルドしたときのバイナリサイズは、全語彙を含む場合が 9,463,280 バイト、`vocab-cl100k_base` だけの場合が 2,550,256 バイトです。

語彙データは提供元ごとに別の crate（`tiktoken-vocab-openai`、`-meta`、`-deepseek`、`-qwen`、`-mistral`、`-moonshot`、`-zhipu`、`-minimax`）に収録しています。上記の feature を指定すると、Cargo は有効にした提供元のデータだけをダウンロードします。ダウンロード量は OpenAI だけなら約 1.5 MB、全社分では 8.0 MB です。

語彙を組み込まなかったエンコーディングは `list_encodings()` の一覧に含まれず、`get_encoding()` は `None` を返します。`pricing` の料金表は語彙から独立しているので、料金の見積もりだけなら `vocab-*` をすべて無効にしてビルドできます。

提供元ごとに語彙をまとめて指定する feature は `vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax` です。既定の `vocabs-all` はすべての語彙を含みます。語彙を個別に指定する feature は下表に示します。

## クイックスタート

```rust
// エンコーディング名で取得
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// モデル名で取得 — 全プロバイダ対応
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // ゼロアロケーション高速パス

let enc = tiktoken::encoding_for_model("llama-3.1-70b").unwrap();
let enc = tiktoken::encoding_for_model("deepseek-v3").unwrap();
let enc = tiktoken::encoding_for_model("qwen2.5-72b").unwrap();
```

## 対応エンコーディング

`データ` 列は、各エンコーディングの feature を有効にするとバイナリに追加される語彙データのサイズです。同じファイルを共有する語彙を複数選んでも、データは一度だけ組み込まれます。`+ ベース` と記した 3 種類では、元の語彙と順位が一致する部分を共有し、末尾の追加分だけを格納します。

| エンコーディング | プロバイダ | Feature | データ | 対応モデル |
|---|---|---|---|---|
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o、GPT-4.1、GPT-4.5、GPT-5〜5.6（Sol/Terra/Luna を含む）、GPT-6.x（推定、公式未公表）、o1、o3、o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss（harmony チャット形式） |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4、GPT-4 Turbo、GPT-3.5 Turbo、text-embedding-*、davinci-002、babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + ベース | text-davinci-002/003、code-davinci-*、code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*、code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 世代：davinci、curie、babbage、ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2（`r50k_base` の別名） |
| `llama3` | Meta | `vocab-llama3` | 111 KB + ベース | Llama 3、3.1、3.2、3.3 |
| `llama4` | Meta | `vocab-llama4` | 816 KB | Llama 4 Scout / Maverick、Muse |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3、R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash、V4.1 Flash（V3 語彙 + V4 特殊 token） |
| `qwen2` | Alibaba | `vocab-qwen2` | 564 KB | Qwen 2.5、Qwen 3 |
| `qwen3_5` | Alibaba | `vocab-qwen3_5` | 989 KB | Qwen 3.5〜3.8、Qwen 4（推定、公式未公表） |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral、Mixtral、Large 3、Small 4（Tekken トークナイザー） |
| `mistral_v15` | Mistral | `vocab-mistral_v15` | 853 KB | Mistral Large 4（Tekken v15） |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3 / K3.1（K2 語彙 + K3 特殊 token） |
| `glm4` | Zhipu | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | Zhipu | `vocab-glm5` | 6 KB + ベース | GLM-5 / 5.1 / 5.2 / 5.3 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 / M3 / M3.1 |

## API

### エンコードとデコード

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();

let tokens = enc.encode("hello world");           // Vec<u32>
let text = enc.decode_to_string(&tokens).unwrap(); // String
let bytes = enc.decode(&tokens);                   // Vec<u8>

// 1 つのバッファを複数回の呼び出しで使い回し、呼び出しごとの確保をなくす
let mut ids = Vec::new();
enc.encode_into("hello world", &mut ids);         // `ids` の末尾に追加

// 1 トークン分のバイト列。語彙から借用するので割り当てはない。長さを足し合わせると原文でのバイト位置がわかる
let hello: Option<&[u8]> = enc.token_bytes(ids[0]);

// エンコードと同時に、各トークンが原文のどのバイト位置で終わるかを返す。token_bytes の長さを
// 後から足し合わせるより速い。マルチバイト文字の途中で終わるトークンもある。
// encode_with_ends_into(text, &mut ids, &mut ends) は encode_into と同じく末尾に追加する
let (ids, ends) = enc.encode_with_ends("hello world"); // ends: [5, 11]

// カット位置：ここで 2 つに分けて別々にエンコードしても、text[..b] と text[b..] の
// token ID を並べると text 全体の token ID と一致する
let text = "Hello world, again.";
let b = enc.next_cut(text, 1).unwrap();           // 5（" world" の前）
let last = enc.prev_cut(text, text.len());        // Some(12)（" again." の前）
```

### 特殊 token

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_with_special_tokens("hello<|endoftext|>world");
// tokens に <|endoftext|> の特殊トークン ID が含まれます
```

### token 数の計算

```rust
let enc = tiktoken::get_encoding("o200k_base").unwrap();
let count = enc.count("素早い茶色の狐が怠けた犬を飛び越えた。");
// encode().len() より高速 — トークンベクタを割り当てません

// 特殊トークン認識付きカウント
let count = enc.count_with_special_tokens("hello<|endoftext|>world");
```

### 同じテキストの再利用

`TextMemo` はテキスト全体とその token ID 列を保存します。入力が保存済みのテキストとバイト単位で一致すれば、ID 列をスライスとして返します。それ以外の入力は通常どおりエンコードして保存します。初回に余分にかかる時間は、3 KB 以上の入力では最大で約 4%、900 バイトではタイマーの 1 目盛り（42 ns）以内です。テキストと ID 列を保存するのは、このキャッシュを作成した場合だけです。

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let mut memo = enc.memo(64 << 20); // テキストと id の合計で最大約 64 MiB
let first = memo.encode("a long system prompt").to_vec();
assert_eq!(memo.encode("a long system prompt"), first); // memo から結果を返す
```

### 並列エンコード

`encode`、`count`、`encode_with_special_tokens`、`count_with_special_tokens` は、32 KiB 以上のテキストを分割し、プロセス全体で共有する worker スレッドプールでエンコードします。このサイズの入力を初めて処理するときにプールを作成します。結果は単一スレッドでエンコードした場合と完全に一致します。

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let text = "word ".repeat(20_000);
let tokens = enc.encode(&text); // 100 KB：複数スレッドでエンコード

// 1 回の呼び出しで使うスレッド数（呼び出し元スレッドを含む）；
// 1 = シングルスレッド、0 = デフォルトに戻す
tiktoken::set_parallelism(1);
assert_eq!(enc.encode(&text), tokens);
tiktoken::set_parallelism(0);
```

- 既定のスレッド数は高性能コアの数。システムがコアの種類を区別しない場合は全コア数を使う。環境変数 `TIKTOKEN_THREADS` に正の整数を指定すると、その値を使う
- `TIKTOKEN_THREADS=1` または `set_parallelism(1)` で並列処理を無効にできる。この場合、すべての処理を呼び出し元のスレッドで実行し、worker スレッドは起動しない
- 同時にプールを使える呼び出しは 1 つだけ。アプリケーションが複数のスレッドからエンコードする場合など、プールの使用中に別の呼び出しがあれば、空きを待たずに呼び出し元のスレッドで処理する。これによりエンコードに使うスレッド数を設定値以内に抑える
- `encode_parallel` は既存コードとの互換性のために残しており、現在の動作は `encode` と同じ。`parallel` feature の指定は不要
- `wasm32` はスレッドをサポートしないため、すべての呼び出しを単一スレッドで処理する

### 料金の見積もり

```rust
use tiktoken::pricing;

// 簡易見積もり
let cost = pricing::estimate_cost("gpt-4o", 1_000_000, 500_000).unwrap();

// プロンプトキャッシュ付き
let model = pricing::get_model("claude-opus-5.5").unwrap();
let cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

// キャッシュ書き込みは保持時間ごとの価格（ここでは 1 時間）。高速ティアと
// バックグラウンド枠にはそれぞれ見積もり関数があります
assert_eq!(model.cache_write_per_1m(60), Some(8.0));
let fast = pricing::get_model("gpt-6-sol").unwrap().estimate_priority_cost(1_000, 1_000);
let background = pricing::get_model("gpt-6-luna").unwrap().estimate_background_cost(1_000, 1_000, 8);

// プロバイダ別のモデル一覧
let models = pricing::models_by_provider(pricing::Provider::DeepSeek);

// 各社 API が実際に受け付ける表記も解決します — 本表はドット、API はハイフン、
// リリース日サフィックス、Bedrock / Vertex の装飾
let r = pricing::resolve_model("us.anthropic.claude-opus-5").unwrap();
assert_eq!(r.model.id, "claude-opus-5");
assert!(matches!(r.matched, pricing::Match::Normalized { .. }));

// `estimate_cost` は同じ解決を通ります。`get_model` は完全一致のままです
assert!(pricing::estimate_cost("claude-haiku-4-5-20251001", 1_000, 1_000).is_some());
```

OpenAI、Anthropic、Google、Meta、DeepSeek、Alibaba、Mistral、Moonshot、Zhipu、MiniMax、Voyage が提供する 153 モデルに対応しています。

## WebAssembly

npm パッケージ [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) をブラウザや Node.js で利用できます。WebAssembly SIMD を使ってビルドしているため、ブラウザでの実行には Chrome 91、Firefox 89、Safari 16.4 以降が必要です。

```bash
npm install @goliapkg/tiktoken-wasm
```

```typescript
import init, { getEncoding, encodingForModel, estimateCost } from '@goliapkg/tiktoken-wasm'

await init()

const enc = getEncoding('cl100k_base')
const tokens = enc.encode('hello world')    // Uint32Array
const text = enc.decode(tokens)             // "hello world"
const count = enc.count('hello world')      // 2

enc.free() // 使用後に WASM メモリを解放
```

| バンドラー | プラグイン |
|---------|--------|
| Vite | [vite-plugin-wasm](https://www.npmjs.com/package/vite-plugin-wasm) + [vite-plugin-top-level-await](https://www.npmjs.com/package/vite-plugin-top-level-await) |
| webpack 5 | 組み込みの実験的機能 `asyncWebAssembly` |
| Next.js | [next.config.js `webpack.experiments`](https://nextjs.org/docs/app/api-reference/next-config-js/webpack) |

デモアプリ一式は [`examples/react-app`](examples/react-app/) にあります。

<details>
<summary>ソースからビルド</summary>

```bash
# wasm-pack が必要: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

</details>

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## エコシステム

**tiktoken** · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

コードのライセンスは [MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のどちらかを選べます。

組み込みの語彙には、コードとは別に各公開元のライセンスが適用されます。各語彙の出典とライセンスは [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY)、ライセンス全文は [`licenses/`](licenses/)、各ライセンスが求める帰属表示は [NOTICE](NOTICE) に掲載しています。

Llama のライセンスでは、月間アクティブユーザー数が 7 億人を超える製品には Meta からのライセンスが必要です。Mistral の `mistral_v15` を Model-as-a-Service 事業で利用する場合は商用ライセンスが必要です。Kimi と MiniMax では一定の規模を超えると表示義務が生じます。DeepSeek には用途に応じた制限があります。ビルド時に各語彙の `vocab-*` feature を無効にすると、その語彙は組み込まれません。

Built with Llama.
