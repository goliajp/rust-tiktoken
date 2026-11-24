# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語** · **[tiktoken.golia.jp](https://tiktoken.golia.jp)** — ブラウザで試す

> **3.5.x からのアップグレード：** 3.6.0、3.8.0 と 5.0.0 は参照実装から乖離していたトークン id を修正しました。キャッシュ済みのトークン数や保存済み id 列は再計算してください。詳細は [CHANGELOG](tiktoken/CHANGELOG.md#upgrading-from-35x--token-id-changes)。
>
> **4.0 へのアップグレード：** トークン id と API は変わりません。唯一の破壊的変更は、語彙が既定で有効な `vocab-*` 機能の下に移ったことです。`default-features = false` と書いている依存は、使う語彙を明示してください。
>
> **5.2 へのアップグレード：** API の変更はありません。32 KiB 以上のテキストはデフォルトで複数スレッドでエンコードされるようになりました（`tiktoken::set_parallelism(1)` または `TIKTOKEN_THREADS=1` で無効化）。wasm パッケージは WebAssembly SIMD が必要です（Chrome 91、Firefox 89、Safari 16.4）。また、新しい Unicode バージョンでのみ数字とされる 336 個のコードポイントは、`mistral_v15` と `deepseek_v3` / `deepseek_v4` での分割が変わります。詳細は [CHANGELOG](tiktoken/CHANGELOG.md#520---2026-11-23)。
>
> **5.0 へのアップグレード：** 破壊的変更は 2 つです。料金関連の構造体と `VisionPricing` が `#[non_exhaustive]` になったため、`match` には `_` の分岐が必要で、構造体をリテラルで作ることもできなくなりました。また Llama 4、Qwen 3.5〜3.8、Mistral Large 4、`deepseek-flash` は実際に使う語彙に対応付けられるようになったため、トークン数が変わります。5.1 は語彙データをベンダーごとの `tiktoken-vocab-*` クレートに移しただけで、ほかは変わりません。

最速の Rust BPE トークナイザーと、その WebAssembly バインディング。OpenAI [tiktoken](https://github.com/openai/tiktoken) 互換で、主要なオープンモデル（Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）もサポート。手書きスキャナが ASCII と CJK の両方を扱い、キー長で階層化した語彙と断片の丸ごとメモ化により、**ネイティブで tiktoken-rs の 22〜83 倍、fastokens の 1.5〜12 倍**、**ブラウザ内で gpt-tokenizer の 5.8〜7.0 倍** 高速——日本語・中国語の文章でも優位です。

## このワークスペースの crate

| パス | Crate / Package | 説明 | バージョン |
|:-----|:----------------|:-----|:-----------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE トークナイザー — 20 エンコーディング、153 モデル、各社料金 | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | 上記の WASM バインディング crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | 同じものを `wasm-pack` で npm へ公開 | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> 2 つの crate は同一ワークスペースにあり、**バージョンは常に同期（lockstep）** — リリースのたびに両方を同じバージョン番号で発行します。

## 特長

- **実行時に正規表現を使わない** — 全エンコーディングの分割パターンを、生成した Unicode テーブルに基づくテーブル駆動のスキャナで再現。cl100k 系と o200k 系は SIMD（NEON、SSE2、wasm simd128）で 64 バイト単位に断片の開始位置を判定します。`regex` クレートはスキャナを検証するテスト用の参照実装としてのみ使います。
- **20 エンコーディング・153 モデル・11 プロバイダ** — OpenAI（GPT-4/4o/4.1/4.5、GPT-5.x、GPT-6.x、o1/o3/o4-mini、gpt-oss）、Llama 3/4、DeepSeek V3/V4、Qwen 2〜3.8、Mistral（Large 4 を含む）、Kimi K2/K3、GLM-4/5、MiniMax M2/M3。さらに USD のコスト見積もり（料金は Anthropic・Google も含む）。
- **繰り返しのテキスト** — `enc.memo(max_bytes)` が返すオプトインの `TextMemo` は、既出のテキストに保存済みの結果を返す（200 KB の書籍で 3.5 µs、再エンコードは 110 µs）。`encode` 自体は何も保持しない。
- **軽量・移植性** — 20 語彙を合計 7.8 MB で埋め込み、語彙ごとにオプトアウト可能（cl100k のみなら 373 KB、cargo は有効にしたベンダーのデータだけをダウンロード）、キー長で階層化した語彙と断片のメモ化、線形/ヒープのハイブリッド BPE マージ、長いテキストは内蔵の std のみのスレッドプールで自動的に並列エンコード、ゼロアロケーションの `count()`、C 依存ゼロの純 Rust（実行時の依存は `ruzstd` と語彙データクレートのみ）、自己完結の wasm ビルド。

API・対応モデル表・ベンチマークは各 crate の README を参照：**[`tiktoken/`](tiktoken/README.md)** ·  **[`tiktoken-wasm/`](tiktoken-wasm/README.md)**。

## クイックスタート

### Rust

```toml
[dependencies]
tiktoken = "5"
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

Apple M4 Pro Mac mini での測定で、`encode` は シングルスレッドで **tiktoken-rs の 22〜83 倍**、**Python tiktoken の 43〜92 倍** 高速：ASCII で 22〜83 倍、日本語・中国語の文章で 50〜82 倍、繰り返しのない敵対的 CJK コーパスでも 36 倍。fastokens に対してはシングルスレッドで 1.5〜12 倍、既定のスレッド数では 200 KB の書籍で 3 倍。ブラウザ内（wasm）では gpt-tokenizer の 5.8〜7.0 倍、tiktoken-fast の 1.1〜2.5 倍です。詳細表と方法論は [`tiktoken/README.md#performance`](tiktoken/README.md#performance) を参照。

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
# リリースゲート: 完全なプロパティスイープ（約 35 分。CI はサンプルのみ）
PROPTEST_CASES=25000 cargo test --workspace --all-targets

git flow release start X.Y.Z
# バージョンを X.Y.Z に更新: tiktoken/Cargo.toml、tiktoken-wasm/Cargo.toml
#（およびその tiktoken パス依存）。両方の CHANGELOG を確定。
git flow release finish X.Y.Z                       # master へマージ、vX.Y.Z タグ、develop へ戻しマージ
git tag -a tiktoken-wasm-vX.Y.Z vX.Y.Z^{commit} -m "tiktoken-wasm X.Y.Z"
git push origin master develop vX.Y.Z tiktoken-wasm-vX.Y.Z
# タグ `v*` が tiktoken crate を、`tiktoken-wasm-v*` が wasm crate + npm を公開
```

7 個のプリトークナイザ用プロパティテスト（それぞれ全パターン系統で実行）、短い文字列の網羅テスト、12 個のラウンドトリップ用プロパティテストが、スキャナを `regex` クレートに対して固定しています。`PROPTEST_CASES` は 1 回の実行が新たに踏む範囲を決めるもので、`proptest-regressions/` のシードはどの設定でもランダムケースより先に再生されるため、過去に失敗した入力は常に再確認されます。

## ライセンス

コードは [MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のいずれか、あなたの選択でライセンスされます。

組み込みの語彙はこれに含まれず、それぞれ公開元のライセンスに従います。[LICENSE-3RD-PARTY](tiktoken/LICENSE-3RD-PARTY) に各語彙の出典とライセンス、[`licenses/`](tiktoken/licenses/) に全文、[NOTICE](tiktoken/NOTICE) に各ライセンスが求める表示をまとめています。一部のライセンスには利用条件があります：Llama のライセンス（月間アクティブユーザー 7 億超の製品は Meta への別途ライセンス申請が必要）、`mistral_v15` の Mistral ライセンス（Model as a Service 事業者は商用ライセンスが必要）、一定規模を超えた場合の Kimi・MiniMax の表示条項、DeepSeek の利用制限条項。語彙の `vocab-*` フィーチャを有効にしないビルドには、その語彙は含まれません。

Built with Llama.
