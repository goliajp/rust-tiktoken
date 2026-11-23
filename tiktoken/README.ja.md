# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

最速の Rust BPE トークナイザ — ネイティブで tiktoken-rs の 22〜83 倍（日本語・中国語の文章で 50〜82 倍）、fastokens の 1.5〜12 倍、ブラウザ内（wasm）で gpt-tokenizer の 5.8〜7.0 倍。手書きスキャナが ASCII と CJK の両方を扱い、語彙はキー長で階層化、繰り返し断片は丸ごとメモ化。OpenAI [tiktoken](https://github.com/openai/tiktoken) 互換で、**主要な全 LLM トークナイザ**をサポート — OpenAI、Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax。

## 特徴

- **マルチプロバイダ**: 8 社 20 エンコーディング（OpenAI、Meta、DeepSeek、Alibaba、Mistral、Moonshot、Zhipu、MiniMax）
- **高速**: 全エンコーディングにテーブル駆動のプリトークナイザ（実行時に正規表現を使わない）、cl100k 系と o200k 系は SIMD で断片の開始位置を判定、キー長で階層化した語彙、スレッド間で共有される断片の丸ごとメモ化、ハイブリッド BPE マージ
- **並列エンコード**: 32 KiB 以上のテキストは、内蔵の std のみのスレッドプールで自動的に複数スレッドでエンコード
- **料金見積もり**: 11 プロバイダ 153 モデルのコスト推定
- **コンパクト**: 20 語彙で合計 7.8 MB を埋め込み、語彙ごとにオプトアウト可能 — cl100k のみのビルドなら 373 KB、ダウンロードも有効にしたベンダー分のみ
- **ゼロアロケーションカウント**: `count()` パスはトークンベクタを割り当てません

## パフォーマンス

計測前に各実装のトークン出力が完全に一致することを確認し、各数値は 1 パス、
ウォームアップ後 9 回の中央値です。コーパスはすべてのハーネスでバイト単位に
同一（`bench-compare/`・`benches/bench_python.py`・`web/bench/`）。

#### ネイティブ — Apple M4 Pro Mac mini・シングルスレッド・`encode`

`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`

| コーパス | Python tiktoken 0.14 | tiktoken-rs 0.12 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| 短文 (13 B) | 1.3 µs | 602 ns | **28 ns** | **22x** | **46x** |
| 中程度 (900 B) | 23.2 µs | 19.8 µs | **431 ns** | **46x** | **54x** |
| 英語の文章 (45 KB) | 1,100 µs | 955 µs | **19.9 µs** | **48x** | **55x** |
| 中国語の文章 (4.3 KB) | 114 µs | 97.8 µs | **2.0 µs** | **50x** | **58x** |
| 日本語の文章 (4.6 KB) | 131 µs | 119 µs | **1.4 µs** | **82x** | **91x** |
| CJK 混在 ×50 (4.5 KB) | 122 µs | 108 µs | **2.5 µs** | **43x** | **49x** |
| 敵対的 CJK・繰り返しなし (3.9 KB) | 114 µs | 94.0 µs | **2.6 µs** | **36x** | **43x** |
| コード (3.9 KB) | 179 µs | 163 µs | **1.9 µs** | **83x** | **92x** |

o200k_base はシングルスレッドで tiktoken-rs の 12〜80 倍です。`count()` は
短いテキストで `encode` より最大 36% 高速、長いテキストではほぼ同等です（id
ベクタを一切確保しません）。既定のスレッド数では 45 KB のテキストが 6.2 µs です。

#### 最速クラスの他実装との比較 — Apple M4 Pro Mac mini・シングルスレッド・cl100k_base・`encode`

より速い可能性のある実装を見つけた限りすべて、同一プロセス・同一入力で、id の
完全一致を確認してから計測しました（`bench-compare` とは別のハーネス）。数値は
3 回の中央値、`tiktoken` は crates.io の 5.2.0 です。

| コーパス | **tiktoken** | [fastokens](https://crates.io/crates/fastokens) 0.3.6 | [HF tokenizers](https://crates.io/crates/tokenizers) 1.0-rc.5 | [toks](https://github.com/actual-computer/toks) 0.3.3 | [toktok-rs](https://crates.io/crates/toktok-rs) 0.5 | [magi-tokenizer](https://crates.io/crates/magi-tokenizer) 0.1 | [bpe-openai](https://crates.io/crates/bpe-openai) 0.3 |
|---|---|---|---|---|---|---|---|
| 短文 (13 B) | **25 ns** | 61 ns | 112 ns | 42 ns | 32 ns | 27 ns | 147 ns |
| 中程度 (900 B) | **431 ns** | 750 ns | 733 ns | 606 ns | 1.3 µs | 1.0 µs | 8.2 µs |
| 英語の文章 (45 KB) | **19.9 µs** | 35.6 µs | 32.3 µs | 29.9 µs | 61.2 µs | 49.5 µs | 405 µs |
| 中国語の文章 (4.3 KB) | **2.0 µs** | 8.0 µs | 4.3 µs | 12.9 µs | 10.1 µs | 12.2 µs | 37.7 µs |
| 日本語の文章 (4.6 KB) | **1.4 µs** | 17.2 µs | 4.1 µs | 18.8 µs | 15.8 µs | 22.8 µs | 41.1 µs |
| CJK 混在 ×50 (4.5 KB) | **2.6 µs** | 3.9 µs | 5.8 µs | 17.8 µs | 12.7 µs | 18.9 µs | 40.0 µs |
| 敵対的 CJK・繰り返しなし (3.9 KB) | **2.7 µs** | 5.3 µs | 5.4 µs | 17.2 µs | 15.3 µs | 19.4 µs | 56.1 µs |
| コード (3.9 KB) | **1.9 µs** | 3.9 µs | 3.4 µs | 2.8 µs | 6.1 µs | 6.6 µs | 46.1 µs |
| 英語の書籍 (200 KB) | **111 µs** | 184 µs | 205 µs | 144 µs | 341 µs | 715 µs | 2,854 µs |
| Rust ソース (100 KB) | **56.1 µs** | 95.8 µs | 97.6 µs | 79.8 µs | 181 µs | 263 µs | 1,412 µs |

- **fastokens** — シングルスレッドで 1.5〜12 倍遅い。既定のスレッド数では書籍 50.3 µs、
  Rust ソース 36.3 µs に対し、本クレートは 16.7 µs と 12.5 µs（3.0 倍、2.9 倍）
- **短文**: 13 バイトの行では magi-tokenizer との差は 4〜12%
- **初めて見るテキスト**（キャッシュなし・シングルスレッド）: fastokens の 1.7〜6.2 倍高速、
  書籍は 392 µs に対して 228 µs。13 バイトを超えるテキストでは他のどのライブラリより
  1.17 倍以上高速。13 バイトでは複数のライブラリがタイマーの分解能 42 ns で並ぶ。既定の
  スレッド数の書籍では、fastokens のワーカーがキャッシュを保持している場合の差は 8〜20%
- **toks** は全文メモを無効にした値。有効にすると、*同じ*テキストの再エンコードでは保存済みの
  結果を返すため、本クレートの再エンコードより 3〜20 倍速い
- そのほかに試したライブラリ（tiktoken-rs、tokensim、riptoken、kitoken、basetenkenizer、
  Hugging Face tokenizers 0.23）はすべての行で 3.4 倍以上遅い（Hugging Face 0.23 と kitoken は Rust ソースで
  id が一致せず、その行は計測していない）。Python からのみ呼べる
  Gigatoken は、呼び出しのオーバーヘッド込みで 1.4〜11 倍遅い

#### ブラウザ内 — Apple M4 Pro Mac mini・Chromium

`web/` で `npm run bench` — 本クレートの wasm ビルドを、別の wasm トークナイザーと
主要な 2 つの JavaScript トークナイザーと比較。すべて `encode` の値です。

| コーパス | tiktoken-fast 0.1 (wasm) | gpt-tokenizer 4.1 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|---|
| 中国語の文章 (4.3 KB) | 6.4 µs | 31.9 µs | 6,625 µs | **4.8 µs** |
| 日本語の文章 (4.6 KB) | 6.6 µs | 29.2 µs | 12,825 µs | **4.2 µs** |
| CJK 混在 ×50 (4.5 KB) | 7.0 µs | 40.3 µs | 3,975 µs | **6.3 µs** |
| 敵対的 CJK・繰り返しなし (3.9 KB) | 6.7 µs | 42.2 µs | 2,756 µs | **6.1 µs** |
| 英語の文章 (45 KB) | 118 µs | 277 µs | 9,638 µs | **47.7 µs** |
| コード (3.9 KB) | 8.9 µs | 30.4 µs | 1,120 µs | **5.1 µs** |

tiktoken-fast は、一度見たテキストなら混在文字の 2 コーパスで 10% ほどの差まで迫り
ますが、初めて見るテキストでは 1.8〜6.6 倍遅くなります。Safari では gpt-tokenizer の
3.3〜7.9 倍、tiktoken-fast の 1.5〜2.2 倍高速です。

敵対的コーパスは断片の繰り返しを一切含まず、各実装のメモ化を無効にしますが、
優位は変わりません。

<details>
<summary>なぜ速いのか</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 前処理分割 | 全エンコーディングでテーブル駆動のスキャナ。Unicode テーブルはパターンが前提とする Unicode バージョンから生成；cl100k 系と o200k 系は SIMD（NEON / SSE2 / wasm simd128）で 64 バイト単位に断片の開始位置を判定 | 常に正規表現 | 常に正規表現 |
| 正規表現エンジン | 実行時は不使用（`regex` は開発用依存で、スキャナを検証する参照実装） | `fancy-regex`（バックトラック） | `regex`＋PyO3/FFI オーバーヘッド |
| 語彙引き | キー長で階層化：15 バイト以下のキーは詰めて 1 つの開放アドレス表に格納（キャッシュラインあたり 2 スロット）、それ以上はハッシュタグ付き arena スロット、1〜2 バイトは別に直接表 | `HashMap<Vec<u8>>`（20 万回のアロケーション） | PyO3 背後の Rust `HashMap` |
| 繰り返し断片 | 語彙にない断片のマージ結果を共有の検索表に追加し、全スレッドで再利用；長い断片には専用のスレッドローカル・共有キャッシュ | なし | なし |
| BPE マージ | ハイブリッド：スタック線形走査（短い断片）＋ヒープ（長い断片）；表になかった短い断片はトークン id 単位でまとめてマージ | O(n*m) 線形走査 | O(n*m) 線形走査 |
| アロケーションなしの `count()` | あり | なし | なし |

スキャナはプロパティテスト（毎回数十万件のランダム入力）と短い文字列の網羅
テストで `regex` クレートの語義に固定され、各エンコーディングは 53,275 件の差分フィクスチャでベンダー自身の
トークナイザーに固定されています。

ベンチマークのソース：[`benches/`](benches/)・[`../bench-compare/`](../bench-compare/)。
`cargo bench` / `cargo run -p bench-compare` で再現できます。

</details>

## インストール

```toml
[dependencies]
tiktoken = "5"
```

実行時の依存は `ruzstd` と、有効にした語彙データクレートだけです。以前のバージョンの `parallel` feature は、それを指定したマニフェストがそのままビルドできるよう残してありますが、何もしません。指定しなくても長いテキストは並列にエンコードされます。

### 語彙を選ぶ

20 のエンコーディングは既定ですべて有効です（語彙データ 7.8 MB）。既定を外して使うものだけを指定すると、その分だけを含めます：

```toml
# GPT-4o / GPT-5 だけ — 語彙データは 7.8 MB ではなく 815 KB
tiktoken = { version = "5", default-features = false, features = ["vocab-o200k_base"] }

# ベンダー単位でまとめて
tiktoken = { version = "5", default-features = false, features = ["vocab-openai"] }
```

`examples/count_tokens`（release、Linux x86_64）での実測：全語彙で 9,463,280 バイト、`vocab-cl100k_base` だけなら 2,550,256 バイト。

語彙データはベンダーごとのクレート（`tiktoken-vocab-openai`、`-meta`、`-deepseek`、`-qwen`、`-mistral`、`-moonshot`、`-zhipu`、`-minimax`）にあり、上の feature から取り込まれます。cargo は有効にしたベンダーの分だけをダウンロードします：OpenAI だけなら約 1.5 MB、全部なら 8.0 MB。

組み込まれていないエンコーディングは存在しない扱いになります：`list_encodings()` に出ず、`get_encoding()` は `None` を返します。`pricing` の表は語彙と無関係なので、料金見積もりだけなら `vocab-*` を一つも有効にしないビルドでも使えます。

ベンダー単位の feature：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`、既定の `vocabs-all`。語彙ごとの feature は下の表にあります。

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

`データ` 列は、そのエンコーディングの機能がバイナリに加える語彙バイト数です。同じデータファイルを共有するエンコーディングは二度目のコストがかからず、`+ ベース` の 3 つはランク整列した拡張で末尾のみを保持します。

| エンコーディング | プロバイダ | Feature | データ | 対応モデル |
|---|---|---|---|---|
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), GPT-6.x（推定、公式未公表）, o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss（harmony チャットフォーマット） |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + ベース | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 世代: davinci, curie, babbage, ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2（`r50k_base` のエイリアス） |
| `llama3` | Meta | `vocab-llama3` | 111 KB + ベース | Llama 3, 3.1, 3.2, 3.3 |
| `llama4` | Meta | `vocab-llama4` | 816 KB | Llama 4 Scout / Maverick, Muse |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3, R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash, V4.1 Flash（V3 語彙 + V4 特殊トークン） |
| `qwen2` | Alibaba | `vocab-qwen2` | 564 KB | Qwen 2.5, Qwen 3 |
| `qwen3_5` | Alibaba | `vocab-qwen3_5` | 989 KB | Qwen 3.5 – 3.8, Qwen 4（推定、公式未公表） |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral, Mixtral, Large 3, Small 4（Tekken トークナイザ） |
| `mistral_v15` | Mistral | `vocab-mistral_v15` | 853 KB | Mistral Large 4 (Tekken v15) |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3 / K3.1（K2 語彙 + K3 特殊トークン） |
| `glm4` | Zhipu | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | Zhipu | `vocab-glm5` | 6 KB + ベース | GLM-5 / 5.1 / 5.2 / 5.3 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 / M3 / M3.1 |

## API

### エンコード / デコード

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();

let tokens = enc.encode("hello world");           // Vec<u32>
let text = enc.decode_to_string(&tokens).unwrap(); // String
let bytes = enc.decode(&tokens);                   // Vec<u8>
```

### 特殊トークン

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_with_special_tokens("hello<|endoftext|>world");
// tokens に <|endoftext|> の特殊トークン ID が含まれます
```

### トークンカウント

```rust
let enc = tiktoken::get_encoding("o200k_base").unwrap();
let count = enc.count("素早い茶色の狐が怠けた犬を飛び越えた。");
// encode().len() より高速 — トークンベクタを割り当てません

// 特殊トークン認識付きカウント
let count = enc.count_with_special_tokens("hello<|endoftext|>world");
```

### 並列エンコード

`encode`、`count`、`encode_with_special_tokens`、`count_with_special_tokens`
は、32 KiB 以上のテキストをチャンクに分け、プロセス全体で共有するワーカースレッドプールでエンコードします。プールは最初のそうした呼び出しで作られます。出力はシングルスレッドの場合と完全に同一です。

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

- デフォルトのスレッド数は高性能コアの数（コアの種類を区別しないシステムでは全コア数）です。環境変数 `TIKTOKEN_THREADS` が正の整数ならそれに従います。
- `TIKTOKEN_THREADS=1` または `set_parallelism(1)` で無効になります。すべての呼び出しが呼び出し元スレッドで実行され、ワーカースレッドは一切起動しません。
- 並列で走る呼び出しは同時に 1 つだけです。プールが使用中だった呼び出し（たとえばアプリケーション自身が複数スレッドからエンコードしている場合）は待たずに自分のスレッドで実行するため、エンコードに使うスレッドが設定数を超えることはありません。
- `encode_parallel` は `encode` と同じになり、既存コードのために残してあります。`parallel` feature はもう不要です。
- `wasm32` にはスレッドがなく、すべての呼び出しがシングルスレッドで実行されます。

### コスト見積もり

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

OpenAI、Anthropic、Google、Meta、DeepSeek、Alibaba、Mistral、Moonshot、Zhipu、
MiniMax、Voyage の 153 モデルに対応。

## WebAssembly

npm パッケージ [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) として公開中。ブラウザと Node.js で利用可能。WebAssembly SIMD でビルドしているため、Chrome 91、Firefox 89、Safari 16.4 以降が必要です。

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

| バンドラ | プラグイン |
|---------|--------|
| Vite | [vite-plugin-wasm](https://www.npmjs.com/package/vite-plugin-wasm) + [vite-plugin-top-level-await](https://www.npmjs.com/package/vite-plugin-top-level-await) |
| webpack 5 | 組み込み `asyncWebAssembly` 実験機能 |
| Next.js | [next.config.js `webpack.experiments`](https://nextjs.org/docs/app/api-reference/next-config-js/webpack) |

完全なデモは [`examples/react-app`](examples/react-app/) を参照。

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

コードは [MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のいずれか、あなたの選択でライセンスされます。

組み込みの語彙はこれに含まれず、それぞれ公開元のライセンスに従います。[LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) に各語彙の出典とライセンス、[`licenses/`](licenses/) に全文、[NOTICE](NOTICE) に各ライセンスが求める表示をまとめています。一部のライセンスには利用条件があります：Llama のライセンス（月間アクティブユーザー 7 億超の製品は Meta への別途ライセンス申請が必要）、`mistral_v15` の Mistral ライセンス（Model as a Service 事業者は商用ライセンスが必要）、一定規模を超えた場合の Kimi・MiniMax の表示条項、DeepSeek の利用制限条項。語彙の `vocab-*` フィーチャを有効にしないビルドには、その語彙は含まれません。

Built with Llama.
