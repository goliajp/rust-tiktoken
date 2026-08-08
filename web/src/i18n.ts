// Trilingual dictionary + a tiny hook. No i18n library: three locales, one
// page, a flat key space — a Record and a context are the whole machinery.

import { createContext, useContext } from 'react'

export type Lang = 'en' | 'zh' | 'ja'

export const LANGS: { id: Lang; label: string }[] = [
  { id: 'en', label: 'EN' },
  { id: 'zh', label: '中文' },
  { id: 'ja', label: '日本語' },
]

export function detectLang(): Lang {
  const saved = localStorage.getItem('lang')
  if (saved === 'en' || saved === 'zh' || saved === 'ja') return saved
  const nav = navigator.language.toLowerCase()
  if (nav.startsWith('zh')) return 'zh'
  if (nav.startsWith('ja')) return 'ja'
  return 'en'
}

type Dict = Record<string, { en: string; zh: string; ja: string }>

const dict: Dict = {
  // top nav
  'nav.playground': { en: 'Playground', zh: '在线体验', ja: 'プレイグラウンド' },
  'nav.encodings': { en: 'Encodings', zh: '编码支持', ja: 'エンコーディング' },
  'nav.performance': { en: 'Performance', zh: '性能', ja: 'パフォーマンス' },
  'nav.install': { en: 'Install', zh: '安装', ja: 'インストール' },

  // hero
  'hero.title.pre': { en: 'The fastest Rust BPE tokenizer, ', zh: '最快的 Rust BPE 分词器，', ja: '最速の Rust 製 BPE トークナイザー。' },
  'hero.title.accent': { en: 'running right here in your browser.', zh: '就在你的浏览器里运行。', ja: 'いま、このブラウザ上で動いています。' },
  'hero.sub': {
    en: 'Drop-in compatible with OpenAI tiktoken, byte-exact against every reference tokenizer it reimplements — verified over 167,000+ differential comparisons. 17 encodings across 8 vendors, embedded vocabularies, zero C dependencies.',
    zh: '与 OpenAI tiktoken 完全兼容，对每一个重实现的参考分词器做到字节级一致 —— 经 167,000+ 次差分对照验证。17 套编码、8 家厂商、词表内嵌、零 C 依赖。',
    ja: 'OpenAI tiktoken と完全互換。再実装したすべての参照トークナイザーとバイト単位で一致 — 167,000 回超の差分照合で検証済み。17 エンコーディング・8 ベンダー・語彙内蔵・C 依存ゼロ。',
  },
  'hero.stat.speed': { en: 'vs tiktoken-rs (ASCII)', zh: '对比 tiktoken-rs（ASCII）', ja: 'tiktoken-rs 比（ASCII）' },
  'hero.stat.encodings': { en: 'encodings', zh: '套编码', ja: 'エンコーディング' },
  'hero.stat.models': { en: 'models priced', zh: '个模型价格', ja: 'モデル料金対応' },
  'hero.stat.short': { en: 'short-text encode', zh: '短文本编码', ja: '短文エンコード' },
  'hero.cta.github': { en: 'GitHub', zh: 'GitHub', ja: 'GitHub' },
  'hero.cta.crates': { en: 'crates.io', zh: 'crates.io', ja: 'crates.io' },
  'hero.cta.npm': { en: 'npm', zh: 'npm', ja: 'npm' },
  'hero.cta.try': { en: 'Try it below ↓', zh: '在下方直接体验 ↓', ja: 'すぐ下で試す ↓' },

  // playground
  'pg.tag': { en: 'Live playground', zh: '在线体验', ja: 'ライブ・プレイグラウンド' },
  'pg.heading': { en: 'Tokenize anything, locally', zh: '任意文本，本地分词', ja: 'どんなテキストも、ローカルで' },
  'pg.blurb': {
    en: 'This is the actual Rust crate compiled to WebAssembly — the same code that runs 15–40x faster than tiktoken-rs — executing in your browser. Nothing you type leaves this page.',
    zh: '这就是编译成 WebAssembly 的 Rust crate 本体 —— 与比 tiktoken-rs 快 15–40 倍的是同一份代码 —— 直接在你的浏览器里执行。你输入的内容不会离开这个页面。',
    ja: 'これは WebAssembly にコンパイルされた Rust クレートそのもの — tiktoken-rs より 15〜40 倍速いのと同じコード — がブラウザ内で実行されています。入力したテキストがこのページの外に出ることはありません。',
  },
  'pg.title': { en: 'wasm tokenizer', zh: 'wasm 分词器', ja: 'wasm トークナイザー' },
  'pg.placeholder': {
    en: 'Type or paste text here — English, 中文, 日本語, code, emoji 🚀 …',
    zh: '在这里输入或粘贴文本 —— 中文、English、日本語、代码、emoji 🚀 ……',
    ja: 'ここにテキストを入力・貼り付け — 日本語、English、中文、コード、絵文字 🚀 …',
  },
  'pg.example': {
    en: 'Try the example',
    zh: '填入示例',
    ja: 'サンプルを入力',
  },
  'pg.tokens': { en: 'tokens', zh: 'tokens', ja: 'トークン' },
  'pg.chars': { en: 'characters', zh: '字符', ja: '文字' },
  'pg.bytes': { en: 'utf-8 bytes', zh: 'UTF-8 字节', ja: 'UTF-8 バイト' },
  'pg.view.text': { en: 'text', zh: '文本', ja: 'テキスト' },
  'pg.view.ids': { en: 'token ids', zh: 'token ID', ja: 'トークン ID' },
  'pg.loading': {
    en: 'Loading the WebAssembly module — 17 embedded vocabularies make it ~11 MB, worth the wait…',
    zh: '正在加载 WebAssembly 模块 —— 内嵌 17 套词表共约 11 MB，值得等待……',
    ja: 'WebAssembly モジュールを読み込み中 — 17 の内蔵語彙で約 11 MB。少々お待ちください…',
  },
  'pg.error': {
    en: 'Failed to load the wasm module. Check the console, or try a hard refresh.',
    zh: 'wasm 模块加载失败。请查看控制台，或强制刷新重试。',
    ja: 'wasm モジュールの読み込みに失敗しました。コンソールを確認するか、強制リロードしてください。',
  },
  'pg.foot': {
    en: 'runs entirely in-browser · nothing is uploaded',
    zh: '完全在浏览器内运行 · 不上传任何内容',
    ja: 'すべてブラウザ内で実行 · 何もアップロードされません',
  },

  // features
  'feat.tag': { en: 'Why this one', zh: '为什么选它', ja: 'この実装を選ぶ理由' },
  'feat.heading': { en: 'Correct first. Then very, very fast.', zh: '先做到正确，再做到非常非常快。', ja: 'まず正確に。それから、圧倒的に速く。' },
  'feat.exact.h': { en: 'Byte-exact', zh: '字节级一致', ja: 'バイト単位で一致' },
  'feat.exact.p': {
    en: 'Every encoding is verified against its reference implementation — Python tiktoken for the OpenAI family, the official HuggingFace or vendor tokenizers for the rest — over adversarial differential corpora. 167,849 comparisons, 0 divergences.',
    zh: '每套编码都对照其参考实现验证 —— OpenAI 系对照 Python tiktoken，其余对照官方 HuggingFace / 厂商分词器 —— 使用对抗性差分语料。167,849 次对照，0 分歧。',
    ja: 'すべてのエンコーディングを参照実装と照合 — OpenAI 系は Python tiktoken、その他は公式 HuggingFace / ベンダーのトークナイザー — 敵対的差分コーパスで検証。167,849 回の比較、相違ゼロ。',
  },
  'feat.fast.h': { en: 'ASCII fast path', zh: 'ASCII 快速路径', ja: 'ASCII ファストパス' },
  'feat.fast.p': {
    en: 'A hand-written scanner resolves common ASCII pieces without touching the regex engine, and a hybrid stack-based BPE merge avoids allocation for word-sized pieces. Result: 43 ns to encode a short string on Apple Silicon.',
    zh: '手写扫描器不经过正则引擎直接解析常见 ASCII 片段，混合式栈上 BPE 合并让词级片段零分配。结果：Apple Silicon 上短文本编码仅 43 ns。',
    ja: '手書きスキャナが正規表現エンジンを介さず一般的な ASCII 片を解決し、ハイブリッドなスタックベース BPE マージが単語サイズの片をゼロアロケーションで処理。結果、Apple Silicon で短文のエンコードは 43 ns。',
  },
  'feat.everywhere.h': { en: 'Everywhere', zh: '随处可用', ja: 'どこでも動く' },
  'feat.everywhere.p': {
    en: 'Pure Rust with zero C dependencies, vocabularies zstd-compressed and embedded at compile time. Ships as a crate and as a wasm npm package — this very page is the npm package running unmodified.',
    zh: '纯 Rust、零 C 依赖，词表经 zstd 压缩在编译期内嵌。以 crate 和 wasm npm 包双形态发布 —— 你眼前这个页面就是那个 npm 包原样运行的样子。',
    ja: '純 Rust・C 依存ゼロ。語彙は zstd 圧縮でコンパイル時に埋め込み。クレートと wasm npm パッケージの両形態で配布 — このページ自体が、その npm パッケージが無改変で動いている姿です。',
  },

  // encodings
  'enc.tag': { en: 'Encodings', zh: '编码支持', ja: 'エンコーディング' },
  'enc.heading': { en: '17 encodings, 8 vendors', zh: '17 套编码，8 家厂商', ja: '17 エンコーディング・8 ベンダー' },
  'enc.blurb': {
    en: 'From GPT-2 to GPT-5.6, and first-class coverage of the open-weights ecosystem — including Kimi, GLM, MiniMax, DeepSeek V4, Qwen and Llama. Vendor-shared vocabularies (Kimi K2/K3, DeepSeek V3/V4) are deduplicated internally.',
    zh: '从 GPT-2 到 GPT-5.6，并对开源权重生态一等公民级支持 —— 包括 Kimi、GLM、MiniMax、DeepSeek V4、Qwen 与 Llama。同厂共享词表（Kimi K2/K3、DeepSeek V3/V4）在内部去重。',
    ja: 'GPT-2 から GPT-5.6 まで。さらにオープンウェイト系 — Kimi・GLM・MiniMax・DeepSeek V4・Qwen・Llama — をファーストクラスでカバー。ベンダー内で共有される語彙（Kimi K2/K3、DeepSeek V3/V4）は内部で重複排除されます。',
  },
  'enc.col.encoding': { en: 'Encoding', zh: '编码', ja: 'エンコーディング' },
  'enc.col.vendor': { en: 'Vendor', zh: '厂商', ja: 'ベンダー' },
  'enc.col.models': { en: 'Models', zh: '模型', ja: 'モデル' },
  'enc.col.vocab': { en: 'Vocab', zh: '词表', ja: '語彙数' },

  // performance
  'perf.tag': { en: 'Performance', zh: '性能', ja: 'パフォーマンス' },
  'perf.heading': { en: 'Measured, not promised', zh: '实测数字，不是承诺', ja: '約束ではなく、実測' },
  'perf.blurb': {
    en: 'cl100k_base encode on an Apple M4 Mac mini, single-threaded, criterion n=100. Token output verified identical across all three implementations. Full benchmark tables live in the repository README.',
    zh: 'Apple M4 Mac mini 上的 cl100k_base 编码，单线程，criterion n=100。三个实现的 token 输出经验证完全一致。完整基准表见仓库 README。',
    ja: 'Apple M4 Mac mini での cl100k_base エンコード。シングルスレッド、criterion n=100。3 実装のトークン出力が同一であることを検証済み。完全なベンチマーク表はリポジトリの README に。',
  },
  'perf.col.input': { en: 'Input', zh: '输入', ja: '入力' },
  'perf.col.python': { en: 'Python tiktoken', zh: 'Python tiktoken', ja: 'Python tiktoken' },
  'perf.col.rs': { en: 'tiktoken-rs', zh: 'tiktoken-rs', ja: 'tiktoken-rs' },
  'perf.col.ours': { en: 'this crate', zh: '本 crate', ja: '本クレート' },
  'perf.col.speedup': { en: 'speedup', zh: '加速比', ja: '高速化率' },

  // install
  'inst.tag': { en: 'Install & use', zh: '安装与使用', ja: 'インストールと使用' },
  'inst.heading': { en: 'Two lines to tokens', zh: '两行代码得到 token', ja: '2 行でトークンに' },
  'inst.rust.blurb': {
    en: 'In Rust — encodings are cached globally, so repeated lookups are free:',
    zh: 'Rust 侧 —— 编码实例全局缓存，重复获取零成本：',
    ja: 'Rust では — エンコーディングはグローバルにキャッシュされ、再取得はコストゼロ：',
  },
  'inst.js.blurb': {
    en: 'In the browser or Node.js via the wasm package (the one running on this page):',
    zh: '浏览器或 Node.js 侧用 wasm 包（本页面正在运行的就是它）：',
    ja: 'ブラウザ / Node.js では wasm パッケージで（このページで動いているものと同じ）：',
  },
  'inst.docs': {
    en: 'Full API docs: docs.rs for Rust, the package README for JavaScript/TypeScript. Cost estimation for 107 models across 10 providers is included in both.',
    zh: '完整 API 文档：Rust 见 docs.rs，JavaScript/TypeScript 见包内 README。两侧都内置了 10 家厂商 107 个模型的成本估算。',
    ja: '完全な API ドキュメント：Rust は docs.rs、JavaScript/TypeScript はパッケージの README へ。両方に 10 プロバイダ・107 モデルのコスト見積もりを内蔵。',
  },

  // footer
  'foot.license': { en: 'MIT OR Apache-2.0 · GOLIA Inc.', zh: 'MIT OR Apache-2.0 · GOLIA株式会社', ja: 'MIT OR Apache-2.0 · GOLIA株式会社' },
  'foot.tagline': {
    en: 'tokens measured in nanoseconds',
    zh: '以纳秒计量 token',
    ja: 'ナノ秒で測るトークン',
  },
}

export function t(lang: Lang, key: string): string {
  const e = dict[key]
  if (!e) return key
  return e[lang]
}

export const LangContext = createContext<Lang>('en')
export function useT() {
  const lang = useContext(LangContext)
  return (key: string) => t(lang, key)
}
