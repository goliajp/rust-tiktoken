# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語** · [ブラウザで試す（tiktoken.golia.jp）](https://tiktoken.golia.jp)

> **3.5.x からの更新**：キャッシュした token 数と保存済みの token ID 列を再計算してください。3.6.0、3.8.0、5.0.0 で、参照実装と異なる token ID を返していた問題を修正しました。詳細は [CHANGELOG](tiktoken/CHANGELOG.md#upgrading-from-35x--token-id-changes) を参照してください。
>
> **4.0 への更新**：token ID と API は従来どおりです。互換性に関わる変更は語彙の指定方法だけです。各語彙は既定で有効な `vocab-*` feature に移りました。依存関係に `default-features = false` を指定している場合は、使用する語彙を明示してください。
>
> **5.0 への更新**：互換性に関わる変更が 2 点あります。料金関連の構造体と `VisionPricing` に `#[non_exhaustive]` が付いたため、`match` には `_` の分岐が必要です。構造体リテラルでの生成もできなくなりました。また、Llama 4、Qwen 3.5〜3.8、Mistral Large 4、`deepseek-flash` の語彙を各モデルが実際に使うものに修正したため、token 数が変わります。5.1 での変更は、語彙データを提供元ごとの `tiktoken-vocab-*` crate に移したことだけです。
>
> **5.2 への更新**：API に変更はありません。32 KiB 以上のテキストは既定で複数のスレッドを使ってエンコードします。並列処理は `tiktoken::set_parallelism(1)` または `TIKTOKEN_THREADS=1` で無効にできます。wasm パッケージには WebAssembly SIMD が必要です（Chrome 91、Firefox 89、Safari 16.4 以降）。新しいバージョンの Unicode でのみ数字に分類される 336 個の符号位置については、`mistral_v15`、`deepseek_v3`、`deepseek_v4` で分割結果が変わります。詳細は [CHANGELOG](tiktoken/CHANGELOG.md#520---2026-11-23) を参照してください。

rust-tiktoken は Rust 製の BPE トークナイザーで最速の実装です。WebAssembly バインディングも提供しています。OpenAI の [tiktoken](https://github.com/openai/tiktoken) や主要なオープンモデル（Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）のトークナイザーと互換性があり、そのまま置き換えられます。ASCII と CJK の分割にはそれぞれ手書きのスキャナーを使います。語彙はキーの長さ別に格納し、分割した文字列ごとに処理結果を再利用します。CJK の文章を含む測定で、処理速度はネイティブ版が tiktoken-rs の 22〜83 倍、fastokens の 1.5〜12 倍、ブラウザ版が gpt-tokenizer の 5.8〜7.0 倍です。

## workspace 内の crate

| パス | crate／パッケージ | 説明 | バージョン |
|:-----|:----------------|:-----|:-----------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | 20 種類のエンコーディング、153 モデル、各社の料金見積もりに対応する Rust 製 BPE トークナイザー | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm)（Rust） | Rust 版 tiktoken を wasm から利用するための crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)（npm） | 同じバインディングを `wasm-pack` で npm 向けに公開 | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> 2 つの crate は同じ workspace で管理しています。リリースのたびに両方のバージョンを上げ、同じ番号で公開します。

## 特長

- **実行時に正規表現を使わない**：各エンコーディングの分割規則は、生成した Unicode 表を参照するスキャナーで再現する。cl100k 系と o200k 系では SIMD（NEON、SSE2 または AVX2、wasm simd128）で 64 バイトずつ読み、分割位置を求める。実行時に正規表現は使わず、`regex` crate はスキャナーのテストで結果を照合するためだけに使う
- **20 エンコーディング・153 モデル・11 社**：OpenAI（GPT-4/4o/4.1/4.5、GPT-5.x、GPT-6.x、o1/o3/o4-mini、gpt-oss）、Llama 3/4、DeepSeek V3/V4、Qwen 2〜3.8、Mistral（Large 4 を含む）、Kimi K2/K3、GLM-4/5、MiniMax M2/M3。このほか米ドルでの料金見積もりにも対応し、見積もりは Anthropic と Google のモデルも対象
- **繰り返しのテキスト**：同じテキストを何度も処理する場合は、`enc.memo(max_bytes)` で `TextMemo` を作ると、保存済みの結果を再利用できる。200 KB の書籍では再エンコードに 110 µs かかるところ、保存済みの結果は 3.5 µs で返る。`encode` 自体はテキストも結果も保存しない
- **軽量で移植しやすい**：全 20 種類の語彙データは合計 7.8 MB で、不要な語彙は個別に除外できる。cl100k だけなら 373 KB で、Cargo がダウンロードするのも有効にした提供元のデータだけ。語彙をキーの長さ別に格納し、分割した文字列ごとに結果を再利用する。BPE のマージには線形走査とヒープを使い分ける。長文は Rust 標準ライブラリだけで実装したスレッドプールで並列に処理し、`count()` はメモリを確保せずに token 数を数える。実装はすべて Rust で、C への依存はなく、実行時に依存するのは `ruzstd` と語彙データの crate だけ。wasm 版は単体で動作する

API の詳細、対応モデルの一覧、ベンチマークは各 crate の README に掲載しています（[`tiktoken/`](tiktoken/README.md)、[`tiktoken-wasm/`](tiktoken-wasm/README.md)）。

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

// 一つのバッファを使い回す。encode_into は末尾に追加する
let mut ids = Vec::new();
enc.encode_into("hello world", &mut ids);

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

Apple M4 Pro Mac mini の単一スレッドで測定した `encode` の速度は、tiktoken-rs の 22〜83 倍、Python 版 tiktoken の 43〜92 倍です。tiktoken-rs 比では、ASCII で 22〜83 倍、中国語と日本語の文章で 50〜82 倍、繰り返しを排した CJK の敵対的コーパスで 36 倍です。fastokens 比では単一スレッドで 1.5〜12 倍、既定のスレッド数で 200 KB の書籍を処理すると 3 倍です。ブラウザでの wasm 版の速度は、gpt-tokenizer の 5.8〜7.0 倍、tiktoken-fast の 1.1〜2.5 倍です。測定結果の一覧と測定方法は [`tiktoken/README.md#performance`](tiktoken/README.md#performance) を参照してください。

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

`tiktoken` と `tiktoken-wasm` は毎回同じバージョンで同時にリリースします。リリースには git-flow を使い、PR は作成しません。

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

スキャナーの動作が `regex` crate と一致するかを、前処理の分割に関する 7 件のプロパティテスト、短い文字列を網羅するテスト、エンコードとデコードの往復に関する 12 件のプロパティテストで確認します。前処理の 7 件は、それぞれすべての分割パターン系統で実行します。`PROPTEST_CASES` で、実行ごとに新しく試すケースの数を設定できます。設定値にかかわらず、ランダムな入力を試す前に `proptest-regressions/` のシードを再現し、過去に失敗した入力を毎回検証します。

## ライセンス

コードのライセンスは [MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のどちらかを選べます。

組み込みの語彙には、コードとは別に各公開元のライセンスが適用されます。各語彙の出典とライセンスは [LICENSE-3RD-PARTY](tiktoken/LICENSE-3RD-PARTY)、ライセンス全文は [`licenses/`](tiktoken/licenses/)、各ライセンスが求める帰属表示は [NOTICE](tiktoken/NOTICE) に掲載しています。

Llama のライセンスでは、月間アクティブユーザー数が 7 億人を超える製品には Meta からのライセンスが必要です。Mistral の `mistral_v15` を Model-as-a-Service 事業で利用する場合は商用ライセンスが必要です。Kimi と MiniMax では一定の規模を超えると表示義務が生じます。DeepSeek には用途に応じた制限があります。ビルド時に各語彙の `vocab-*` feature を無効にすると、その語彙は組み込まれません。

Built with Llama.
