# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語** · **[tiktoken.golia.jp](https://tiktoken.golia.jp)** — ブラウザで試す

最速の Rust BPE トークナイザーと、その WebAssembly バインディング。OpenAI [tiktoken](https://github.com/openai/tiktoken) 互換で、主要なオープンモデル（Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）もサポート。手書きスキャナが ASCII と CJK の両方を扱い、キー長で階層化した語彙と断片の丸ごとメモ化により、**ネイティブで tiktoken-rs の 5〜49 倍**、**ブラウザ内で gpt-tokenizer の 2〜4 倍** 高速——日本語・中国語の文章でも優位です。

## このワークスペースの crate

| パス | Crate / Package | 説明 | バージョン |
|:-----|:----------------|:-----|:-----------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE トークナイザー — 17 エンコーディング、107 モデル、各社料金 | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | 上記の WASM バインディング crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | 同じものを `wasm-pack` で npm へ公開 | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> 2 つの crate は同一ワークスペースにあり、**バージョンは常に同期（lockstep）** — リリースのたびに両方を同じバージョン番号で発行します。

## 特長

- **ASCII 高速パス（事前トークン化）** — よくある ASCII の断片（英字・数字・記号・短縮形）を正規表現エンジンを使わずに解決。cl100k / o200k / qwen2 / deepseek の ASCII テキストで `encode` / `count` が **2.3〜5.5 倍高速**。Unicode/CJK は自動的に正規表現へフォールバック。
- **17 エンコーディング・107 モデル・10 プロバイダ** — OpenAI（GPT-4/4o/4.1/4.5、GPT-5.x、o1/o3/o4-mini、gpt-oss）、Llama 3/4、DeepSeek V3/V4、Qwen、Mistral、Kimi K2/K3、GLM-4/5、MiniMax M2。さらに USD のコスト見積もり（料金は Anthropic・Google も含む）。
- **軽量・移植性** — Arena ベースの語彙、線形/ヒープのハイブリッド BPE マージ、オプションの rayon 並列、ゼロアロケーションの `count()`、C 依存ゼロの純 Rust、小さな wasm ビルド、zstd 圧縮語彙をコンパイル時に埋め込み。

API・対応モデル表・ベンチマークは各 crate の README を参照：**[`tiktoken/`](tiktoken/README.md)** ·  **[`tiktoken-wasm/`](tiktoken-wasm/README.md)**。

## クイックスタート

### Rust

```toml
[dependencies]
tiktoken = "3.5"
```

```rust
// エンコーディング名で取得
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
assert_eq!(enc.decode_to_string(&tokens).unwrap(), "hello world");

// トークンベクタを割り当てずにカウント
let n = enc.count("The quick brown fox.");

// モデル名で解決
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
```

### WebAssembly（ブラウザ / Node.js）

```bash
npm install @goliapkg/tiktoken-wasm
```

```js
import init, { getEncoding } from '@goliapkg/tiktoken-wasm'
await init()
const enc = getEncoding('o200k_base')
const tokens = enc.encode('hello world')
```

## パフォーマンス

Apple M4 Mac mini での測定で、`encode` は **tiktoken-rs の 5〜49 倍**、**Python tiktoken の 5〜29 倍** 高速：ASCII で 29〜49 倍、日本語・中国語の文章で 15〜17 倍、繰り返しのない敵対的 CJK コーパスでも 5 倍。ブラウザ内（wasm）では gpt-tokenizer の 2〜4 倍です。詳細表と方法論は [`tiktoken/README.md#performance`](tiktoken/README.md#performance) を参照。

## ビルド

```bash
cargo test -p tiktoken
cargo fmt --all --check
cargo clippy --workspace --lib -- -D warnings

# WASM (wasm-pack が必要: cargo install wasm-pack)
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## リリース

`tiktoken` と `tiktoken-wasm` はバージョンを同期し、git-flow で一緒にリリースします（PR なし）：

```bash
git flow release start X.Y.Z
# バージョンを X.Y.Z に更新: tiktoken/Cargo.toml、tiktoken-wasm/Cargo.toml
#（およびその tiktoken パス依存）。両方の CHANGELOG を確定。
git flow release finish X.Y.Z                       # master へマージ、vX.Y.Z タグ、develop へ戻しマージ
git tag -a tiktoken-wasm-vX.Y.Z vX.Y.Z^{commit} -m "tiktoken-wasm X.Y.Z"
git push origin master develop vX.Y.Z tiktoken-wasm-vX.Y.Z
# タグ `v*` が tiktoken crate を、`tiktoken-wasm-v*` が wasm crate + npm を公開
```

## ライセンス

[MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のいずれか、あなたの選択でライセンスされます。
