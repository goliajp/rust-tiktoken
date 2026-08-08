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
    en: 'Token counts drive billing, context limits and truncation, so a tokenizer has to be fast and exact at once. This one beats the fastest JavaScript tokenizers by 2–4× in the browser and tiktoken-rs by 5–49× natively, and each of its 17 encodings is checked byte-for-byte against the vendor’s own tokenizer — 44,518 differential cases, zero divergence. The playground below runs the published WebAssembly package itself, so both claims can be verified on this page.',
    zh: 'token 数直接决定计费、上下文上限和截断位置，分词器必须又快又准。本实现在浏览器里快于最快的 JavaScript 分词器 2–4 倍，原生快于 tiktoken-rs 5–49 倍；17 套编码逐一与厂商自己的分词器逐字节比对，44,518 组差分对照零分歧。下方试用区运行的就是发布的 WebAssembly 包本体，这两件事都可以当场验证。',
    ja: 'トークン数は課金・コンテキスト上限・打ち切り位置を直接決めるため、トークナイザーには速さと正確さの両方が要ります。本実装はブラウザ内で最速の JavaScript トークナイザーの 2〜4 倍、ネイティブでは tiktoken-rs の 5〜49 倍高速。17 のエンコーディングはすべてベンダー自身のトークナイザーとバイト単位で照合済みで、44,518 件の差分照合に相違はありません。下のプレイグラウンドは公開中の WebAssembly パッケージそのもの——どちらの主張もこのページで確かめられます。',
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
    en: 'This is the npm package itself, running in this page. All 17 encodings are available, and nothing you type leaves the browser.',
    zh: '这就是发布到 npm 的 WebAssembly 包，原样运行在本页。17 套编码全部可选，输入内容不会离开浏览器。',
    ja: 'npm に公開している WebAssembly パッケージが、そのままこのページで動いています。17 エンコーディングすべてを選択でき、入力内容がブラウザの外に出ることはありません。',
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
  'feat.fast.h': { en: 'Hand-written scanners, ASCII and CJK', zh: '手写扫描：ASCII 与 CJK', ja: '手書きスキャナ：ASCII と CJK' },
  'feat.fast.p': {
    en: 'ASCII and CJK pieces alike are cut by hand-written scanners that never enter the regex engine; the vocabulary serves each key size its own structure, and repeated pieces are memoised whole. The regex stays the arbiter — property tests hold the scanners to its exact output.',
    zh: 'ASCII 与汉字、假名、谚文的常见片段都由手写扫描器直接切分，不进正则引擎；词表按 key 长度分层，重复片段整片记忆。正则仍是判准 —— 属性测试以每轮数十万条随机输入，要求两者切分完全一致。',
    ja: 'ASCII も漢字・かな・ハングルも、一般的な断片は手書きスキャナが直接切り出し、正規表現エンジンを通しません。語彙はキー長ごとに最適な構造で引き、繰り返す断片は丸ごとメモ化。正解は正規表現側にあり、毎回数十万件のランダム入力で一致を担保します。',
  },
  'feat.everywhere.h': { en: 'Pure Rust, embeds anywhere', zh: '纯 Rust，随处可嵌', ja: '純 Rust、どこへでも' },
  'feat.everywhere.p': {
    en: 'No C dependencies, no runtime, no external data files — vocabularies compile into one self-contained artifact that drops into servers, IoT and edge devices, and browsers alike, as a crate or a WebAssembly package. The playground above is that package, unmodified.',
    zh: '零 C 依赖、无运行时、无外部数据文件 —— 词表在编译期内嵌成单一自足产物，服务器、IoT 与边缘设备、浏览器都能直接嵌入，以 crate 和 WebAssembly 包发布。上方试用区就是该包本身，未作改动。',
    ja: 'C 依存なし・ランタイム不要・外部データなし。語彙を埋め込んだ自己完結の単一成果物は、サーバーでも IoT・エッジ機器でもブラウザでも、クレートか WebAssembly パッケージでそのまま動きます。上のプレイグラウンドはそのパッケージ本体です。',
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
    en: 'Two machines, two sets of rivals, identical corpora. Token outputs are asserted identical across implementations before anything is timed; each figure is one full pass, median of 9 rounds after warmup, cl100k_base. Both tables reproduce from commands in the repository.',
    zh: '两台机器、两组对手、同一批语料。计时前先断言各实现的 token 输出完全一致；每个数字是一次完整处理，预热后取 9 轮中位数，cl100k_base 编码。两张表都可以用仓库里的命令复现。',
    ja: '2 台のマシン、2 組の比較相手、同一のコーパス。計測前に各実装のトークン出力が完全に一致することを確認し、各数値はウォームアップ後 9 回の中央値です（cl100k_base）。どちらの表もリポジトリのコマンドで再現できます。',
  },
  'perf.browser.h': {
    en: 'In the browser — Mac Studio (M4 Max), Chromium · npm run bench',
    zh: '浏览器内 —— Mac Studio（M4 Max）、Chromium · npm run bench',
    ja: 'ブラウザ内 — Mac Studio（M4 Max）・Chromium · npm run bench',
  },
  'perf.native.h': {
    en: 'Native — Apple M4 Mac mini, single thread · cargo run -p bench-compare',
    zh: '原生 —— Apple M4 Mac mini、单线程 · cargo run -p bench-compare',
    ja: 'ネイティブ — Apple M4 Mac mini・シングルスレッド · cargo run -p bench-compare',
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
    en: 'gpt-tokenizer is the fastest JavaScript tokenizer; js-tiktoken the most downloaded. The adversarial corpus never repeats a piece, which disables every implementation’s memoisation — it is the floor, and the lead holds there too.',
    zh: 'gpt-tokenizer 是浏览器里最快的 JavaScript 分词器，js-tiktoken 是下载量最大的。对抗语料不含任何重复片段，各实现的缓存全部失效 —— 那一行是下界，领先在下界处依然成立。',
    ja: 'gpt-tokenizer は最速の JavaScript トークナイザー、js-tiktoken は最もダウンロードされているものです。敵対的コーパスは断片の繰り返しを一切含まず、各実装のメモ化を無効にします——その行が下限であり、下限でも優位は変わりません。',
  },
  'perf.caption': {
    en: 'Rust rivals, same corpora, encode(). The old weakness is gone: CJK used to fall to the regex engine at ≈2×; the scanners and the layered vocabulary now put it at 5–18×.',
    zh: 'Rust 同行对比，同一批语料，encode()。旧短板已经不在：CJK 过去落回正则引擎、只快约 2 倍；现在由扫描器和分层词表接住，快 5–18 倍。',
    ja: 'Rust 実装との比較、同一コーパス、encode()。かつての弱点は解消済みです：以前の CJK は正規表現エンジン頼みで約 2 倍止まりでしたが、現在はスキャナと階層化した語彙で 5〜18 倍です。',
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
