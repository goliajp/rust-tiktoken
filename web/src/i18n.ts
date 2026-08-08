// Trilingual dictionary + a tiny hook. No i18n library: three locales, one
// page, a flat key space — a Record and a context are the whole machinery.
//
// Register: this is a research-project page, so the copy states what was
// measured and how, and avoids marketing superlatives that the repository
// cannot back with a number.

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
  // masthead
  'brand.lab': { en: 'Golia Lab', zh: 'Golia Lab', ja: 'Golia Lab' },
  'nav.playground': { en: 'Playground', zh: '在线试用', ja: 'プレイグラウンド' },
  'nav.encodings': { en: 'Encodings', zh: '编码', ja: 'エンコーディング' },
  'nav.performance': { en: 'Performance', zh: '性能', ja: '性能' },
  'nav.install': { en: 'Install', zh: '安装', ja: '導入' },

  // front matter
  'front.eyebrow': {
    en: 'A Golia Lab project · Open source',
    zh: 'Golia Lab 研究项目 · 开源',
    ja: 'Golia Lab のプロジェクト · オープンソース',
  },
  'front.title.a': { en: 'A ', zh: '', ja: '' },
  'front.title.b': { en: 'high-performance', zh: '高性能', ja: '高性能' },
  'front.title.c': { en: ' BPE tokenizer.', zh: ' BPE 分词器', ja: ' BPE トークナイザー' },
  'front.abstract': {
    en: 'Tokenization decides what a language model actually reads, and a tokenizer that is merely close is a silent source of error. This implementation reproduces 17 encodings from 8 vendors and is checked against each vendor’s own tokenizer over adversarial differential corpora — currently 167,849 comparisons with no divergence. It is also, as a consequence of how it is built, considerably faster than the alternatives.',
    zh: '分词决定了语言模型真正读到的内容，而一个「差不多对」的分词器是一种静默的错误来源。本实现复刻了 8 家厂商的 17 套编码，并使用对抗性差分语料逐一对照各厂商自己的分词器 —— 目前 167,849 次对照，零分歧。同时，由于其实现方式，它也显著快于现有方案。',
    ja: 'トークン化は言語モデルが実際に読む内容を決めます。「ほぼ正しい」トークナイザーは、静かな誤りの源です。本実装は 8 ベンダー 17 エンコーディングを再現し、敵対的差分コーパスで各ベンダー自身のトークナイザーと照合しています — 現在 167,849 回の比較で相違ゼロ。またその実装方式の帰結として、既存の選択肢より大幅に高速です。',
  },
  'front.fig.comparisons': {
    en: 'differential comparisons, 0 divergences',
    zh: '次差分对照，0 分歧',
    ja: '回の差分照合、相違ゼロ',
  },
  'front.fig.encodings': { en: 'encodings, 8 vendors', zh: '套编码，8 家厂商', ja: 'エンコーディング・8 ベンダー' },
  'front.fig.short': { en: 'to encode a short string', zh: '短字符串编码耗时', ja: '短い文字列のエンコード' },
  'front.fig.speed': { en: 'vs tiktoken-rs on ASCII', zh: 'ASCII 文本对比 tiktoken-rs', ja: 'ASCII で tiktoken-rs 比' },
  'front.cta.try': { en: 'Run it in your browser', zh: '在浏览器中运行', ja: 'ブラウザで実行' },

  // playground
  'pg.heading': {
    en: 'Run the tokenizer, here, on your own text',
    zh: '就在此处，用你自己的文本运行分词器',
    ja: 'このページで、自分のテキストを分かち書きする',
  },
  'pg.blurb': {
    en: 'The Rust crate compiled to WebAssembly and executed by this page. Every encoding listed below is available; nothing you type is sent anywhere.',
    zh: '这是编译为 WebAssembly 并由本页面直接执行的 Rust crate。下表所有编码均可选用；你输入的内容不会发送到任何地方。',
    ja: 'WebAssembly にコンパイルされた Rust クレートを、このページが直接実行しています。下表のすべてのエンコーディングが利用可能で、入力内容はどこにも送信されません。',
  },
  'pg.cap': { en: 'Encoding', zh: '编码', ja: 'エンコーディング' },
  'pg.placeholder': {
    en: 'Type or paste text — English, 中文, 日本語, source code, emoji …',
    zh: '输入或粘贴文本 —— 中文、English、日本語、源代码、emoji ……',
    ja: 'テキストを入力・貼り付け — 日本語、English、中文、ソースコード、絵文字 …',
  },
  'pg.example': { en: 'Load sample text', zh: '载入示例文本', ja: 'サンプルを読み込む' },
  'pg.tokens': { en: 'tokens', zh: 'token 数', ja: 'トークン数' },
  'pg.chars': { en: 'characters', zh: '字符数', ja: '文字数' },
  'pg.bytes': { en: 'UTF-8 bytes', zh: 'UTF-8 字节', ja: 'UTF-8 バイト' },
  'pg.view.text': { en: 'Segments', zh: '分段', ja: '分割' },
  'pg.view.ids': { en: 'Token IDs', zh: 'Token ID', ja: 'トークン ID' },
  'pg.loading': {
    en: 'Loading the WebAssembly module — it carries all 17 vocabularies, about 11 MB',
    zh: '正在加载 WebAssembly 模块 —— 其中内嵌全部 17 套词表，约 11 MB',
    ja: 'WebAssembly モジュールを読み込み中 — 17 の語彙をすべて内蔵、約 11 MB',
  },
  'pg.error': {
    en: 'The WebAssembly module failed to load. A hard refresh usually resolves it.',
    zh: 'WebAssembly 模块加载失败。通常强制刷新即可恢复。',
    ja: 'WebAssembly モジュールの読み込みに失敗しました。強制リロードで解消することが多いです。',
  },
  'pg.empty': {
    en: 'Tokens appear here as you type.',
    zh: '输入后，token 会显示在这里。',
    ja: '入力すると、ここにトークンが表示されます。',
  },
  'pg.foot': {
    en: 'Executed locally in this browser',
    zh: '在本浏览器内本地执行',
    ja: 'このブラウザ内でローカル実行',
  },
  'pg.caption.label': { en: 'Figure 1.', zh: '图 1.', ja: '図 1.' },
  'pg.caption': {
    en: 'Each shaded run is one token; hover a segment to read its id. A token whose bytes are not valid UTF-8 on their own — the halves of an emoji, for instance — displays as a replacement character, though its id is exact.',
    zh: '每一个底色片段即一个 token，悬停可查看其 id。若某个 token 的字节本身不是合法 UTF-8（例如 emoji 被拆开的一半），则显示为替换字符，但其 id 是精确的。',
    ja: '網掛けされた各区間が 1 トークンです。ホバーすると id を表示します。単独では正しい UTF-8 にならないトークン（絵文字の断片など）は置換文字として表示されますが、id は正確です。',
  },

  // method
  'feat.heading': {
    en: 'Correctness is established by differential testing, not by assertion',
    zh: '正确性由差分测试确立，而非声称',
    ja: '正しさは主張ではなく、差分テストによって確立する',
  },
  'feat.exact.h': { en: 'Checked against the reference', zh: '对照参考实现校验', ja: '参照実装との照合' },
  'feat.exact.p': {
    en: 'The OpenAI encodings are compared against Python tiktoken; the rest against each vendor’s own HuggingFace tokenizer, or in Kimi’s case the vocabulary Moonshot ships. Corpora deliberately target the axes where these patterns disagree — whitespace runs, newline boundaries, digits, CJK, slashes.',
    zh: 'OpenAI 系编码对照 Python tiktoken；其余对照各厂商自己的 HuggingFace 分词器 —— Kimi 则对照 Moonshot 自身发布的词表。语料刻意针对这些 pattern 产生分歧的位置：空白串、换行边界、数字、CJK、斜杠。',
    ja: 'OpenAI 系は Python tiktoken と、その他は各ベンダー自身の HuggingFace トークナイザー（Kimi は Moonshot 配布の語彙）と比較します。コーパスは、これらのパターンが食い違う軸 — 空白列・改行境界・数字・CJK・スラッシュ — を意図的に突きます。',
  },
  'feat.fast.h': { en: 'A hand-written ASCII path', zh: '手写的 ASCII 路径', ja: '手書きの ASCII パス' },
  'feat.fast.p': {
    en: 'Common ASCII pieces are resolved by a scanner that never enters the regex engine, and word-sized pieces merge on the stack without allocating. The regex remains the authority: property tests assert the two paths agree for arbitrary input.',
    zh: '常见的 ASCII 片段由一个从不进入正则引擎的扫描器解析，词级片段在栈上合并、不做分配。正则仍是权威：属性测试断言两条路径对任意输入结果一致。',
    ja: '一般的な ASCII 片は、正規表現エンジンに入らないスキャナが解決し、単語サイズの片はスタック上でアロケーションなしにマージされます。正規表現が正解であり続けます — プロパティテストが任意入力で両経路の一致を保証します。',
  },
  'feat.everywhere.h': { en: 'One implementation, two artefacts', zh: '同一实现，两种产物', ja: '一つの実装、二つの成果物' },
  'feat.everywhere.p': {
    en: 'Pure Rust with no C dependencies and vocabularies embedded at compile time, published both as a crate and as a WebAssembly package. The playground above is that package, unmodified.',
    zh: '纯 Rust、无 C 依赖，词表在编译期内嵌；以 crate 与 WebAssembly 包两种形态发布。上方的试用区就是该包本身，未经改动。',
    ja: '純 Rust・C 依存なし、語彙はコンパイル時に埋め込み。クレートと WebAssembly パッケージの両方で公開しています。上のプレイグラウンドは、そのパッケージそのものです。',
  },

  // encodings
  'enc.heading': { en: 'Supported encodings', zh: '支持的编码', ja: '対応エンコーディング' },
  'enc.blurb': {
    en: 'From GPT-2 through GPT-5.6, alongside the open-weights models that most implementations omit — Kimi, GLM, MiniMax, DeepSeek V4, Qwen, Llama and Mistral.',
    zh: '从 GPT-2 到 GPT-5.6，并覆盖多数实现所忽略的开源权重模型 —— Kimi、GLM、MiniMax、DeepSeek V4、Qwen、Llama 与 Mistral。',
    ja: 'GPT-2 から GPT-5.6 まで。加えて、多くの実装が省略するオープンウェイト系 — Kimi・GLM・MiniMax・DeepSeek V4・Qwen・Llama・Mistral — も対象です。',
  },
  'enc.col.encoding': { en: 'Encoding', zh: '编码', ja: 'エンコーディング' },
  'enc.col.vendor': { en: 'Vendor', zh: '厂商', ja: 'ベンダー' },
  'enc.col.models': { en: 'Models', zh: '模型', ja: 'モデル' },
  'enc.col.vocab': { en: 'Vocabulary', zh: '词表规模', ja: '語彙数' },
  'enc.caption.label': { en: 'Table 1.', zh: '表 1.', ja: '表 1.' },
  'enc.caption': {
    en: 'Where a vendor ships one vocabulary across generations — Kimi K2 and K3, DeepSeek V3 and V4 — the data is stored once and the entries differ only in their special-token tables.',
    zh: '若某厂商跨代共用同一词表 —— 如 Kimi K2 与 K3、DeepSeek V3 与 V4 —— 数据仅存储一份，各条目之间只有特殊 token 表不同。',
    ja: 'ベンダーが世代をまたいで同一の語彙を用いる場合（Kimi K2 と K3、DeepSeek V3 と V4）、データは一度だけ保持し、エントリ間の差は特殊トークン表のみです。',
  },

  // performance
  'perf.heading': { en: 'Measured performance', zh: '实测性能', ja: '実測性能' },
  'perf.blurb': {
    en: 'Encoding with cl100k_base on an Apple M4 Mac mini, single-threaded, criterion with n = 100. Token output was verified identical across all three implementations before timing them.',
    zh: 'Apple M4 Mac mini 上使用 cl100k_base 编码，单线程，criterion n = 100。计时前已验证三个实现的 token 输出完全一致。',
    ja: 'Apple M4 Mac mini 上で cl100k_base によるエンコード。シングルスレッド、criterion n = 100。計測前に 3 実装のトークン出力が同一であることを確認しています。',
  },
  'perf.col.input': { en: 'Input', zh: '输入', ja: '入力' },
  'perf.col.python': { en: 'Python tiktoken', zh: 'Python tiktoken', ja: 'Python tiktoken' },
  'perf.col.rs': { en: 'tiktoken-rs', zh: 'tiktoken-rs', ja: 'tiktoken-rs' },
  'perf.col.ours': { en: 'This implementation', zh: '本实现', ja: '本実装' },
  'perf.col.speedup': { en: 'Speedup', zh: '加速比', ja: '高速化率' },
  'perf.caption.label': { en: 'Table 2.', zh: '表 2.', ja: '表 2.' },
  'perf.caption': {
    en: 'Speedup is stated against tiktoken-rs. Unicode-heavy input gains least, which is expected: it falls through to the regex engine, and that path was never the target of the optimisation.',
    zh: '加速比以 tiktoken-rs 为基准。Unicode 密集的输入提升最小，这符合预期 —— 该情形会回落到正则引擎，而这条路径本就不是优化目标。',
    ja: '高速化率は tiktoken-rs を基準としています。Unicode 主体の入力で伸びが最も小さいのは想定どおりで、その場合は正規表現エンジンに委ねられ、そこは最適化の対象ではありません。',
  },

  // install
  'inst.heading': { en: 'Installation', zh: '安装', ja: '導入' },
  'inst.rust.blurb': {
    en: 'In Rust. Encodings are cached globally, so repeated lookups cost nothing.',
    zh: 'Rust 侧。编码实例全局缓存，重复获取没有开销。',
    ja: 'Rust の場合。エンコーディングはグローバルにキャッシュされ、再取得のコストはありません。',
  },
  'inst.js.blurb': {
    en: 'In the browser or Node.js, through the WebAssembly package used on this page.',
    zh: '浏览器或 Node.js 侧，使用本页面所用的 WebAssembly 包。',
    ja: 'ブラウザまたは Node.js で、本ページが使用している WebAssembly パッケージ経由。',
  },
  'inst.docs': {
    en: 'Both distributions also carry cost estimation for 107 models across 10 providers. Full API reference:',
    zh: '两种分发形态均内置 10 家厂商 107 个模型的成本估算。完整 API 参考：',
    ja: 'いずれの配布形態にも、10 プロバイダ・107 モデルのコスト見積もりを同梱しています。完全な API リファレンス：',
  },

  // colophon
  'foot.org': { en: 'Golia Lab · GOLIA K.K.', zh: 'Golia Lab · GOLIA株式会社', ja: 'Golia Lab · GOLIA株式会社' },
  'foot.license': {
    en: 'Released under MIT OR Apache-2.0.',
    zh: '以 MIT OR Apache-2.0 双许可发布。',
    ja: 'MIT OR Apache-2.0 のデュアルライセンスで公開。',
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
