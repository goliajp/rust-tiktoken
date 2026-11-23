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
  'front.title.c': { en: ' BPE tokenizer', zh: ' BPE 分词器', ja: ' BPE トークナイザー' },
  'front.abstract': {
    en: 'Token counts drive billing, context limits and truncation, so a tokenizer has to be fast and exact at once. This one beats the fastest JavaScript tokenizers by 5.8–7.0× in the browser and tiktoken-rs by 22–83× natively, and each of its 20 encodings is checked byte-for-byte against the vendor’s own tokenizer — 53,275 differential cases, zero divergence. The playground below runs the published WebAssembly package itself, so both claims can be verified on this page.',
    zh: 'token 数直接决定计费、上下文上限和截断位置，分词器必须又快又准。浏览器里，它比最快的 JavaScript 分词器快 5.8–7.0 倍；原生比 tiktoken-rs 快 22–83 倍。20 套编码逐一与厂商自己的分词器逐字节比对，53,275 组差分对照零分歧。下方试用区运行的就是发布的 WebAssembly 包本体 —— 这两件事都可以当场验证。',
    ja: 'トークン数は課金・コンテキスト上限・打ち切り位置を直接決めるため、トークナイザーには速さと正確さの両方が要ります。本実装はブラウザ内で最速の JavaScript トークナイザーの 5.8〜7.0 倍、ネイティブでは tiktoken-rs の 22〜83 倍高速。20 のエンコーディングはすべてベンダー自身のトークナイザーとバイト単位で照合済みで、53,275 件の差分照合に相違はありません。下のプレイグラウンドは公開中の WebAssembly パッケージそのもの——どちらの主張もこのページで確かめられます。',
  },
  'front.fig.comparisons': {
    en: 'differential cases, zero divergence',
    zh: '组差分对照，零分歧',
    ja: '件の差分照合・相違ゼロ',
  },
  'front.fig.encodings': { en: 'encodings, 8 vendors', zh: '套编码，8 家厂商', ja: 'エンコーディング・8 ベンダー' },
  'front.fig.browser': {
    en: 'faster than gpt-tokenizer, in the browser',
    zh: '快于 gpt-tokenizer（浏览器实测）',
    ja: 'ブラウザ内で gpt-tokenizer より高速',
  },
  'front.fig.speed': {
    en: 'faster than tiktoken-rs, native',
    zh: '快于 tiktoken-rs（原生实测）',
    ja: 'ネイティブで tiktoken-rs より高速',
  },
  'front.cta.try': { en: 'Run it in your browser', zh: '在浏览器中运行', ja: 'ブラウザで実行' },

  // playground
  'pg.heading': {
    en: 'Tokenize your own text, right here',
    zh: '在这里分词你自己的文本',
    ja: 'ここで、自分のテキストを分割する',
  },
  'pg.blurb': {
    en: 'This is the npm package itself, running in this page. All 20 encodings are available, and nothing you type leaves the browser.',
    zh: '这就是发布到 npm 的 WebAssembly 包，原样运行在本页。20 套编码全部可选，输入内容不会离开浏览器。',
    ja: 'npm に公開している WebAssembly パッケージが、そのままこのページで動いています。20 エンコーディングすべてを選択でき、入力内容がブラウザの外に出ることはありません。',
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
    en: 'Loading the WebAssembly module — 20 vocabularies, about 9 MB',
    zh: '正在加载 WebAssembly 模块 —— 内含 20 套词表，约 9 MB',
    ja: 'WebAssembly モジュールを読み込み中 — 20 の語彙を内蔵、約 9 MB',
  },
  'pg.error': {
    en: 'The WebAssembly module failed to load — a hard refresh usually fixes it',
    zh: 'WebAssembly 模块加载失败，强制刷新通常即可恢复',
    ja: 'WebAssembly モジュールの読み込みに失敗 — 強制リロードで解消します',
  },
  'pg.empty': {
    en: 'Tokens appear here as you type',
    zh: '输入后这里实时显示分词结果',
    ja: '入力するとここに分割結果が表示されます',
  },
  'pg.foot': {
    en: 'Runs entirely in this browser',
    zh: '完全在本浏览器内运行',
    ja: 'すべてこのブラウザ内で実行',
  },
  'pg.caption.label': { en: 'Figure 1.', zh: '图 1.', ja: '図 1.' },
  'pg.caption': {
    en: 'Each shaded run is one token; hover it to read its id. A token whose bytes are not valid UTF-8 on their own — half an emoji, say — shows as a replacement character, but its id is exact.',
    zh: '每个底色片段是一个 token，悬停查看 id。若某个 token 的字节单独不构成合法 UTF-8（例如半个 emoji），会显示为替换字符，但 id 是准确的。',
    ja: '網掛けされた各区間が 1 トークンで、ホバーすると id が出ます。単体では正しい UTF-8 にならないトークン（絵文字の半分など）は置換文字で表示されますが、id は正確です。',
  },

  // encodings
  'enc.heading': { en: 'Supported encodings', zh: '支持的编码', ja: '対応エンコーディング' },
  'enc.blurb': {
    en: 'GPT-2 through GPT-5.6, and GPT-6 by assumption, plus the open-weights models most implementations leave out: Kimi, GLM, MiniMax, DeepSeek V4, Qwen, Llama and Mistral.',
    zh: '从 GPT-2 到 GPT-5.6（GPT-6 按推定支持），并覆盖多数实现不做的开源权重模型：Kimi、GLM、MiniMax、DeepSeek V4、Qwen、Llama、Mistral。',
    ja: 'GPT-2 から GPT-5.6 まで（GPT-6 は推定で対応）。加えて、多くの実装が対象外とするオープンウェイト系（Kimi・GLM・MiniMax・DeepSeek V4・Qwen・Llama・Mistral）にも対応します。',
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
    en: 'One machine, two sets of rivals, identical corpora. Token outputs are asserted identical across implementations before anything is timed; each figure is one full pass, median of 9 rounds after warmup, cl100k_base. Both tables reproduce from commands in the repository, except the fastokens column, which comes from a separate same-process harness.',
    zh: '同一台机器、两组对手、同一批语料。计时前先断言各实现的 token 输出完全一致；每个数字是一次完整处理，预热后取 9 轮中位数，cl100k_base 编码。两张表都可以用仓库里的命令复现；fastokens 一列除外，它来自另一套同进程测试程序。',
    ja: '同一マシン、2 組の比較相手、同一のコーパス。計測前に各実装のトークン出力が完全に一致することを確認し、各数値はウォームアップ後 9 回の中央値です（cl100k_base）。どちらの表もリポジトリのコマンドで再現できます（fastokens の列のみ、同一プロセスで測る別のハーネスの値です）。',
  },
  'perf.browser.h': {
    en: 'In the browser — Apple M4 Pro Mac mini, Chromium · npm run bench',
    zh: '浏览器内 —— Apple M4 Pro Mac mini、Chromium · npm run bench',
    ja: 'ブラウザ内 — Apple M4 Pro Mac mini・Chromium · npm run bench',
  },
  'perf.native.h': {
    en: 'Native — Apple M4 Pro Mac mini, single thread · cargo run -p bench-compare',
    zh: '原生 —— Apple M4 Pro Mac mini、单线程 · cargo run -p bench-compare',
    ja: 'ネイティブ — Apple M4 Pro Mac mini・シングルスレッド · cargo run -p bench-compare',
  },
  'perf.corpus.zh': { en: 'Chinese prose (4.3 KB)', zh: '中文散文（4.3 KB）', ja: '中国語の文章（4.3 KB）' },
  'perf.corpus.ja': { en: 'Japanese prose (4.6 KB)', zh: '日文散文（4.6 KB）', ja: '日本語の文章（4.6 KB）' },
  'perf.corpus.uni': { en: 'mixed CJK ×50 (4.5 KB)', zh: '多语混排 ×50（4.5 KB）', ja: 'CJK 混在 ×50（4.5 KB）' },
  'perf.corpus.varied': {
    en: 'adversarial CJK, no repeats (3.9 KB)',
    zh: '对抗语料：CJK 无重复（3.9 KB）',
    ja: '敵対的 CJK・繰り返しなし（3.9 KB）',
  },
  'perf.corpus.ascii': { en: 'English prose (45 KB)', zh: '英文文本（45 KB）', ja: '英語の文章（45 KB）' },
  'perf.corpus.code': { en: 'source code (3.9 KB)', zh: '源代码（3.9 KB）', ja: 'ソースコード（3.9 KB）' },
  'perf.col.input': { en: 'Corpus', zh: '语料', ja: 'コーパス' },
  'perf.caption.label': { en: 'Table 2.', zh: '表 2.', ja: '表 2.' },
  'perf.caption.browser.label': { en: 'Table 1.', zh: '表 1.', ja: '表 1.' },
  'perf.caption.browser': {
    en: 'encode(). tiktoken-fast is another wasm build, gpt-tokenizer the fastest JavaScript tokenizer, js-tiktoken the most downloaded. tiktoken-fast comes within about 10% on the two mixed-script rows once it has seen the text, and is 1.8–6.6× slower on a first encounter. The adversarial corpus never repeats a piece, which disables every implementation’s memoisation; the lead holds there too.',
    zh: 'encode()。tiktoken-fast 是另一个 wasm 分词器，gpt-tokenizer 是浏览器里最快的 JavaScript 分词器，js-tiktoken 是下载量最大的。两行混排语料在 tiktoken-fast 见过之后差距约 10%，首次遇到时它慢 1.8–6.6 倍。对抗语料不含任何重复片段，各实现的缓存全部失效，领先依然成立。',
    ja: 'encode()。tiktoken-fast は別の wasm 実装、gpt-tokenizer は最速の JavaScript トークナイザー、js-tiktoken はダウンロード数首位のものです。混在文字の 2 行では、tiktoken-fast は一度見たテキストなら約 10% の差まで迫り、初見では 1.8〜6.6 倍遅くなります。敵対的コーパスは断片の繰り返しを一切含まず、各実装のメモ化を無効にしますが、優位は変わりません。',
  },
  'perf.caption': {
    en: 'Rust rivals, same corpora, encode(). Against tiktoken-rs the rows here are 36–83×; against fastokens, 1.5–12×. Texts of 32 KiB and more also run on several threads by default: 3× ahead of fastokens on a 200 KB book.',
    zh: 'Rust 同行对比，同一批语料，encode()。这几行比 tiktoken-rs 快 36–83 倍，比 fastokens 快 1.5–12 倍。32 KiB 及以上的文本默认还会用多个线程：200 KB 书籍比 fastokens 快 3 倍。',
    ja: 'Rust 実装との比較、同一コーパス、encode()。ここの各行は tiktoken-rs の 36〜83 倍、fastokens の 1.5〜12 倍。32 KiB 以上のテキストは既定で複数スレッドでも処理され、200 KB の書籍では fastokens の 3 倍です。',
  },

  // install
  'inst.heading': { en: 'Installation', zh: '安装', ja: '導入' },
  'inst.rust.blurb': {
    en: 'In Rust — encodings are cached globally, so looking one up again costs nothing. All 20 ship by default; name only the vocabularies you use and the rest stay out of your binary.',
    zh: 'Rust —— 编码实例全局缓存，重复取用没有开销。20 套词表默认全带；只点名用到的，其余不会进你的二进制。',
    ja: 'Rust の場合 — エンコーディングはグローバルにキャッシュされ、取り直してもコストはかかりません。既定では 20 語彙すべてを同梱しますが、使うものだけを指定すれば残りはバイナリに入りません。',
  },
  'inst.js.blurb': {
    en: 'In the browser or Node.js, through the same WebAssembly package this page uses.',
    zh: '浏览器或 Node.js，用的是本页面同一个 WebAssembly 包。',
    ja: 'ブラウザまたは Node.js の場合 — このページと同じ WebAssembly パッケージを使います。',
  },
  'inst.docs': {
    en: 'Both also carry cost estimation for 153 models across 11 providers. Full API reference:',
    zh: '两者都内置 11 家厂商 153 个模型的成本估算。完整 API 文档：',
    ja: 'いずれにも 11 プロバイダ・153 モデルのコスト見積もりを同梱しています。API リファレンス：',
  },

  // colophon
  'foot.license': {
    en: 'Code released under MIT OR Apache-2.0',
    zh: '代码以 MIT OR Apache-2.0 双许可发布',
    ja: 'コードは MIT OR Apache-2.0 のデュアルライセンスで公開',
  },
  'foot.vocab': {
    en: 'Vocabularies stay under their publishers’ licenses',
    zh: '词表沿用各发布方的许可证',
    ja: '語彙はそれぞれ公開元のライセンスに従います',
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
 * A space between two Latin words ("Apple M4 Pro Mac mini") is a real separator
 * and stays breakable.
 *
 * Deliberately a plain string transform: emitting `<wbr>` or splitting into
 * nodes would override `line-break: strict` — `<wbr>` is an explicit break
 * opportunity honoured even where kinsoku forbids one — and shred the text
 * into dozens of DOM nodes.
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
