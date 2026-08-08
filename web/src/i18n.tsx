// Trilingual dictionary + a tiny hook. No i18n library: three locales, one
// page, a flat key space — a Record and a context are the whole machinery.
//
// Each locale is written, not translated. The register is a technical
// project page: say what the thing does and what was measured, in the
// shortest form that stays precise. No defending against claims nobody made
// ("correctness by testing, not by assertion" — of course not by assertion),
// no hedging, no filler connectives. Those read as translationese in Chinese
// and Japanese and as padding in English.

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
  'nav.playground': { en: 'Playground', zh: '在线试用', ja: 'プレイグラウンド' },
  'nav.encodings': { en: 'Encodings', zh: '编码', ja: 'エンコーディング' },
  'nav.performance': { en: 'Performance', zh: '性能', ja: '性能' },
  'nav.install': { en: 'Install', zh: '安装', ja: '導入' },

  // front matter
  'front.eyebrow': {
    en: 'Open source · Maintained by Golia Lab',
    zh: '开源 · 由 Golia Lab 承诺保持维护',
    ja: 'オープンソース · Golia Lab が継続的にメンテナンス',
  },
  'front.title.a': { en: 'A ', zh: '', ja: '' },
  'front.title.b': { en: 'high-performance', zh: '高性能', ja: '高性能' },
  'front.title.c': { en: ' BPE tokenizer.', zh: ' BPE 分词器', ja: ' BPE トークナイザー' },
  'front.abstract': {
    en: 'Token counts drive billing, context limits and truncation, so a tokenizer that is close but not exact costs real money and drops real text. This one covers 17 encodings from 8 vendors, each checked byte-for-byte against that vendor’s own tokenizer — 167,849 comparisons so far, no divergence. The ASCII path is hand-written, which is where the speed comes from.',
    zh: 'token 数直接决定计费、上下文上限和截断位置，分词只要差一点，账单和内容就跟着错。本实现覆盖 8 家厂商的 17 套编码，每一套都与厂商自己的分词器逐字节比对，至今 167,849 次对照无一处不同。ASCII 路径为手写实现，速度即由此而来。',
    ja: 'トークン数は課金・コンテキスト上限・打ち切り位置を直接左右します。分割が少しずれるだけで、請求も本文も狂います。本実装は 8 ベンダー 17 エンコーディングを収録し、いずれもベンダー自身のトークナイザーとバイト単位で照合済み。現時点で 167,849 件、相違はありません。速度は手書きの ASCII 経路によるものです。',
  },
  'front.fig.comparisons': {
    en: 'comparisons against reference implementations',
    zh: '次与参考实现逐字节对照',
    ja: '件の参照実装との照合',
  },
  'front.fig.encodings': { en: 'encodings, 8 vendors', zh: '套编码，8 家厂商', ja: 'エンコーディング・8 ベンダー' },
  'front.fig.short': { en: 'to encode a short string', zh: '编码一个短字符串', ja: '短い文字列のエンコード' },
  'front.fig.speed': {
    en: 'faster than tiktoken-rs on ASCII',
    zh: 'ASCII 文本快于 tiktoken-rs',
    ja: 'ASCII で tiktoken-rs より高速',
  },
  'front.cta.try': { en: 'Run it in your browser', zh: '在浏览器中运行', ja: 'ブラウザで実行' },

  // playground
  'pg.heading': {
    en: 'Tokenize your own text, right here',
    zh: '在这里分词你自己的文本',
    ja: 'ここで、自分のテキストを分割する',
  },
  'pg.blurb': {
    en: 'The Rust crate compiled to WebAssembly, running in this page. All 17 encodings are available, and nothing you type leaves the browser.',
    zh: '这是编译为 WebAssembly 的 Rust crate，运行在本页面内。17 套编码全部可选，输入内容不会离开浏览器。',
    ja: 'Rust クレートを WebAssembly にコンパイルし、このページ上で実行しています。17 エンコーディングすべてを選択でき、入力内容がブラウザの外に出ることはありません。',
  },
  'pg.cap': { en: 'Encoding', zh: '编码', ja: 'エンコーディング' },
  'pg.placeholder': {
    en: 'Type or paste text — English, 中文, 日本語, source code, emoji …',
    zh: '输入或粘贴文本 —— 中文、English、日本語、源代码、emoji ……',
    ja: 'テキストを入力・貼り付け（日本語、English、中文、ソースコード、絵文字 …）',
  },
  'pg.example': { en: 'Load sample text', zh: '载入示例文本', ja: 'サンプルを読み込む' },
  'pg.tokens': { en: 'tokens', zh: 'token 数', ja: 'トークン数' },
  'pg.chars': { en: 'characters', zh: '字符数', ja: '文字数' },
  'pg.bytes': { en: 'UTF-8 bytes', zh: 'UTF-8 字节', ja: 'UTF-8 バイト' },
  'pg.view.text': { en: 'Segments', zh: '分段', ja: '分割' },
  'pg.view.ids': { en: 'Token IDs', zh: 'Token ID', ja: 'トークン ID' },
  'pg.loading': {
    en: 'Loading the WebAssembly module — 17 vocabularies, about 11 MB',
    zh: '正在加载 WebAssembly 模块 —— 内含 17 套词表，约 11 MB',
    ja: 'WebAssembly モジュールを読み込み中。17 の語彙を内蔵、約 11 MB',
  },
  'pg.error': {
    en: 'The WebAssembly module failed to load. A hard refresh usually fixes it.',
    zh: 'WebAssembly 模块加载失败，强制刷新通常即可恢复。',
    ja: 'WebAssembly モジュールの読み込みに失敗しました。強制リロードで解消することがほとんどです。',
  },
  'pg.empty': {
    en: 'Tokens appear here as you type.',
    zh: '输入后这里会显示分词结果。',
    ja: '入力するとここに分割結果が出ます。',
  },
  'pg.foot': {
    en: 'Runs entirely in this browser',
    zh: '完全在本浏览器内运行',
    ja: 'すべてこのブラウザ内で実行',
  },
  'pg.caption.label': { en: 'Figure 1.', zh: '图 1.', ja: '図 1.' },
  'pg.caption': {
    en: 'Each shaded run is one token; hover it to read its id. A token whose bytes are not valid UTF-8 on their own — half an emoji, say — shows as a replacement character, but its id is exact.',
    zh: '每个底色片段是一个 token，悬停可看它的 id。若某个 token 的字节单独不构成合法 UTF-8（例如半个 emoji），会显示为替换字符，但 id 是准确的。',
    ja: '網掛けされた各区間が 1 トークンで、ホバーすると id が出ます。単体では正しい UTF-8 にならないトークン（絵文字の半分など）は置換文字で表示されますが、id は正確です。',
  },

  // method
  'feat.heading': {
    en: 'Seventeen encodings, each checked byte-for-byte against its reference',
    zh: '17 套编码，每一套都与参考实现逐字节比对',
    ja: '17 のエンコーディングを、すべて参照実装とバイト単位で照合',
  },
  'feat.exact.h': { en: 'Checked against the vendor', zh: '对照厂商实现', ja: 'ベンダー実装と照合' },
  'feat.exact.p': {
    en: 'OpenAI encodings are compared against Python tiktoken, the rest against each vendor’s published HuggingFace tokenizer, and Kimi against the vocabulary Moonshot ships. The corpora target where these patterns actually disagree: whitespace runs, newline boundaries, digits, CJK, slashes.',
    zh: 'OpenAI 系比对 Python tiktoken，其余比对各厂商发布的 HuggingFace 分词器，Kimi 比对 Moonshot 自带词表。语料专挑各家 pattern 真正会分歧的位置：连续空白、换行边界、数字、CJK、斜杠。',
    ja: 'OpenAI 系は Python tiktoken、その他は各ベンダー公開の HuggingFace トークナイザー、Kimi は Moonshot 同梱の語彙と比較します。コーパスは、実際に食い違う箇所（連続する空白、改行境界、数字、CJK、スラッシュ）を狙って作っています。',
  },
  'feat.fast.h': { en: 'A hand-written ASCII path', zh: '手写 ASCII 路径', ja: '手書きの ASCII 経路' },
  'feat.fast.p': {
    en: 'Common ASCII pieces are cut by a hand-written scanner that never enters the regex engine, and word-sized pieces merge on the stack without allocating. The regex stays the arbiter: property tests hold both paths to the same output for arbitrary input.',
    zh: '常见 ASCII 片段由手写扫描器直接切分，不进正则引擎；词级片段在栈上合并，零分配。正则仍是判准 —— 属性测试要求两条路径对任意输入给出相同结果。',
    ja: '一般的な ASCII 片は手書きスキャナが直接切り出し、正規表現エンジンを通しません。単語サイズの片はスタック上でマージし、アロケーションは発生しません。正解は正規表現側にあり、任意の入力で両経路が一致することをプロパティテストで担保しています。',
  },
  'feat.everywhere.h': { en: 'One implementation, two artefacts', zh: '同一实现，两种产物', ja: '一つの実装、二つの成果物' },
  'feat.everywhere.p': {
    en: 'Pure Rust, no C dependencies, vocabularies embedded at compile time. Published as a crate and as a WebAssembly package — the playground above is that package, unmodified.',
    zh: '纯 Rust，无 C 依赖，词表在编译期内嵌。以 crate 和 WebAssembly 包两种形态发布 —— 上方试用区用的就是该包本身，未作改动。',
    ja: '純 Rust、C 依存なし、語彙はコンパイル時に埋め込みます。クレートと WebAssembly パッケージの二形態で公開しており、上のプレイグラウンドはそのパッケージそのものです。',
  },

  // encodings
  'enc.heading': { en: 'Supported encodings', zh: '支持的编码', ja: '対応エンコーディング' },
  'enc.blurb': {
    en: 'GPT-2 through GPT-5.6, plus the open-weights models most implementations leave out: Kimi, GLM, MiniMax, DeepSeek V4, Qwen, Llama and Mistral.',
    zh: '从 GPT-2 到 GPT-5.6，并覆盖多数实现不做的开源权重模型：Kimi、GLM、MiniMax、DeepSeek V4、Qwen、Llama、Mistral。',
    ja: 'GPT-2 から GPT-5.6 まで。加えて、多くの実装が対象外とするオープンウェイト系（Kimi・GLM・MiniMax・DeepSeek V4・Qwen・Llama・Mistral）にも対応します。',
  },
  'enc.col.encoding': { en: 'Encoding', zh: '编码', ja: 'エンコーディング' },
  'enc.col.vendor': { en: 'Vendor', zh: '厂商', ja: 'ベンダー' },
  'enc.col.models': { en: 'Models', zh: '模型', ja: 'モデル' },
  'enc.col.vocab': { en: 'Vocabulary', zh: '词表规模', ja: '語彙数' },
  'enc.caption.label': { en: 'Table 1.', zh: '表 1.', ja: '表 1.' },
  'enc.caption': {
    en: 'Where one vocabulary spans generations — Kimi K2 and K3, DeepSeek V3 and V4 — it is stored once; the entries differ only in their special-token tables.',
    zh: '同一词表跨代复用时（Kimi K2 与 K3、DeepSeek V3 与 V4），数据只存一份，两个条目之间仅特殊 token 表不同。',
    ja: '同一の語彙が世代をまたぐ場合（Kimi K2 と K3、DeepSeek V3 と V4）、データは一度だけ保持し、エントリの違いは特殊トークン表のみです。',
  },

  // performance
  'perf.heading': { en: 'Measured performance', zh: '实测性能', ja: '実測性能' },
  'perf.blurb': {
    en: 'cl100k_base encode on an Apple M4 Mac mini, single-threaded, criterion at n = 100. All three implementations were confirmed to produce identical tokens before timing.',
    zh: 'Apple M4 Mac mini，cl100k_base 编码，单线程，criterion n = 100。计时前先确认三个实现的 token 输出完全一致。',
    ja: 'Apple M4 Mac mini、cl100k_base によるエンコード、シングルスレッド、criterion n = 100。計測前に 3 実装のトークン出力が完全に一致することを確認しています。',
  },
  'perf.col.input': { en: 'Input', zh: '输入', ja: '入力' },
  'perf.col.python': { en: 'Python tiktoken', zh: 'Python tiktoken', ja: 'Python tiktoken' },
  'perf.col.rs': { en: 'tiktoken-rs', zh: 'tiktoken-rs', ja: 'tiktoken-rs' },
  'perf.col.ours': { en: 'This implementation', zh: '本实现', ja: '本実装' },
  'perf.col.speedup': { en: 'Speedup', zh: '加速比', ja: '高速化率' },
  'perf.caption.label': { en: 'Table 2.', zh: '表 2.', ja: '表 2.' },
  'perf.caption': {
    en: 'Speedup is against tiktoken-rs. Unicode-heavy input gains least: it falls through to the regex engine, which the ASCII path was never meant to replace.',
    zh: '加速比以 tiktoken-rs 为基准。Unicode 密集的输入提升最小 —— 这类输入会落到正则引擎，而 ASCII 路径本就不是用来取代它的。',
    ja: '高速化率は tiktoken-rs 比です。Unicode 主体の入力で伸びが最も小さいのは、処理が正規表現エンジンに渡るためで、ASCII 経路はそこを置き換えるものではありません。',
  },

  // install
  'inst.heading': { en: 'Installation', zh: '安装', ja: '導入' },
  'inst.rust.blurb': {
    en: 'In Rust. Encodings are cached globally, so looking one up again costs nothing.',
    zh: 'Rust。编码实例全局缓存，重复取用没有开销。',
    ja: 'Rust の場合。エンコーディングはグローバルにキャッシュされ、取り直してもコストはかかりません。',
  },
  'inst.js.blurb': {
    en: 'In the browser or Node.js, through the same WebAssembly package this page uses.',
    zh: '浏览器或 Node.js，用的是本页面同一个 WebAssembly 包。',
    ja: 'ブラウザまたは Node.js の場合。このページと同じ WebAssembly パッケージを使います。',
  },
  'inst.docs': {
    en: 'Both also carry cost estimation for 107 models across 10 providers. Full API reference:',
    zh: '两者都内置 10 家厂商 107 个模型的成本估算。完整 API 文档：',
    ja: 'いずれにも 10 プロバイダ・107 モデルのコスト見積もりを同梱しています。API リファレンス：',
  },

  // colophon
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
  return (key: string) => phrase(t(lang, key), lang)
}

const CJK = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f]/
const LATIN = /[0-9A-Za-z]/

/**
 * Chinese and Japanese line-breaking is the browser's job, and it does it
 * correctly: CJK folds at any character boundary — that is how CJK is
 * typeset, in books and on paper — and `line-break: strict` in the stylesheet
 * enforces kinsoku, so a line never begins with 、 or ends with 「.
 *
 * The one thing the browser gets wrong is the space around embedded Latin
 * (盘古之白). That space is typographic, not lexical: "ASCII 路径" and
 * "特殊 token 表" are single terms, and a fold there reads as a mistake. Make
 * exactly that space non-breaking and leave everything else alone.
 *
 * A space between two Latin words ("Apple M4 Mac mini") is a real separator
 * and stays breakable.
 *
 * This is a string transform on purpose. An earlier version segmented the
 * text and emitted `<wbr>` between phrases, which shredded a sentence into
 * dozens of nodes, split "UTF-8", and — because `<wbr>` overrides
 * `line-break: strict` — put commas at the start of lines.
 */
export function phrase(text: string, lang: Lang): string {
  if (lang === 'en') return text
  return text.replace(/ /g, (_m, i: number) => {
    const a = text[i - 1]
    const b = text[i + 1]
    // A space at the edge of a fragment — the heading is assembled from three
    // dictionary entries around an <em>. Chinese and Japanese do not separate
    // words with spaces, so a space there exists only to set off the Latin on
    // the other side of the seam.
    if (!a || !b) return '\u00a0'
    const mixed = (CJK.test(a) && LATIN.test(b)) || (LATIN.test(a) && CJK.test(b))
    return mixed ? '\u00a0' : ' '
  })
}

/** Dictionary lookup, with the CJK/Latin spacing fix applied. */
export function T({ k }: { k: string }) {
  const lang = useContext(LangContext)
  return <>{phrase(t(lang, k), lang)}</>
}
