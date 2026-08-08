import { encodingForModel } from 'js-tiktoken'
import { encode as gptEncode } from 'gpt-tokenizer/encoding/cl100k_base'

import init, { getEncoding } from '@goliapkg/tiktoken-wasm'


// Deterministic varied-CJK corpus: an LCG walk over Han/Kana/Hangul/Arabic
// blocks, so pre-token pieces rarely repeat. gpt-tokenizer memoises whole
// pieces in a 100k-entry LRU, which the ×50-repetition corpus turns into a
// 98% hit rate; this one measures the merge itself.
function varied(n: number): string {
  let s = '', x = 12345
  const blocks: [number, number][] = [
    [0x4e00, 0x9fa0],
    [0x3040, 0x30a0],
    [0xac00, 0xd780],
    [0x620, 0x650],
  ]
  for (let i = 0; i < n; i++) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0
    const [lo, hi] = blocks[(x >>> 7) % 4]!
    s += String.fromCodePoint(lo + ((x >>> 9) % (hi - lo)))
    if (i % 7 === 6) s += '\uff0c'
    if (i % 23 === 22) s += ' '
  }
  return s
}

const CASES: [string, string][] = [
  ['short_13b', 'Hello, world!'],
  ['medium_900b', 'The quick brown fox jumps over the lazy dog. '.repeat(20)],
  ['long_45kb', 'The quick brown fox jumps over the lazy dog. '.repeat(1000)],
  ['unicode_4kb', '你好世界！こんにちは世界！안녕하세요 세계！مرحبا بالعالم '.repeat(50)],
  ['unicode_varied_4kb', varied(1200)],
  [
    'zh_prose_4kb',
    '分词器把文本切成 token，模型按 token 计费。同一段话在不同词表下的 token 数可能相差一倍以上，因此计费、上下文上限和截断位置都取决于分词是否准确。本实现覆盖多家厂商的编码，每一套都与参考实现逐字节比对，至今没有发现分歧。速度来自手写的扫描器：常见片段不进正则引擎，词级片段在栈上合并，零分配。'.repeat(
      10,
    ),
  ],
  [
    'ja_prose_4kb',
    'トークナイザーはテキストをトークンへ分割し、モデルはトークン単位で課金します。同じ文章でも語彙が違えばトークン数は大きく変わるため、分割の正確さは請求額と文脈上限に直結します。本実装は各ベンダーのエンコーディングを収録し、いずれも参照実装とバイト単位で照合済みです。速度は手書きスキャナによるもので、一般的な断片は正規表現エンジンを通しません。'.repeat(
      9,
    ),
  ],
  [
    'code_3kb',
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# compute first 100 fibonacci numbers\nresults = [fibonacci(i) for i in range(100)]\nprint(results)\n'.repeat(20),
  ],
]

const t0 = performance.now()
await init()
const initMs = performance.now() - t0
const enc = getEncoding('cl100k_base')

const t1 = performance.now()
const js = encodingForModel('gpt-4')
const jsInitMs = performance.now() - t1

function timeIt(text: string, fn: (t: string) => unknown) {
  for (let i = 0; i < 20; i++) fn(text)
  const runs: number[] = []
  for (let r = 0; r < 9; r++) {
    const n = text.length < 100 ? 5000 : text.length < 2000 ? 500 : 50
    const s = performance.now()
    for (let i = 0; i < n; i++) fn(text)
    runs.push(((performance.now() - s) * 1e6) / n)
  }
  runs.sort((a, b) => a - b)
  return Math.round(runs[4])
}

function bench(text: string) {
  for (let i = 0; i < 50; i++) enc.count(text)
  const runs: number[] = []
  for (let r = 0; r < 9; r++) {
    const n = text.length < 100 ? 20000 : text.length < 2000 ? 2000 : 200
    const s = performance.now()
    for (let i = 0; i < n; i++) enc.count(text)
    runs.push(((performance.now() - s) * 1e6) / n) // ns per op
  }
  runs.sort((a, b) => a - b)
  return runs[4]
}

// Their pipeline, split the same way ours is: regex pre-tokenize, then
// everything after it. Lets the two implementations be compared stage by
// stage instead of end to end.
// Verbatim from gpt-tokenizer/src/encodingParams/constants.ts — inlined so
// the stage timing does not depend on their build layout.
const CL100K_SPLIT =
  /'(?:[sS]|[dD]|[mM]|[tT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s/gu

function gptRegexOnly(text: string): number {
  let acc = 0
  for (const [m] of text.matchAll(CL100K_SPLIT)) acc += m.length
  return acc
}

// Sanity: all three must agree on the token count, or the timings compare
// different work.
const out = CASES.map(([name, text]) => {
  const a = enc.count(text)
  const b = js.encode(text).length
  const c = gptEncode(text).length
  return {
    name,
    bytes: new TextEncoder().encode(text).length,
    tokens: a,
    agree: a === b && a === c,
    wasm: Math.round(bench(text)),
    jsTiktoken: timeIt(text, (t) => js.encode(t)),
    gptTokenizer: timeIt(text, (t) => gptEncode(t)),
    gptRegex: timeIt(text, (t) => gptRegexOnly(t)),
  }
})
;const result = { initMs: Math.round(initMs), jsInitMs: Math.round(jsInitMs), out }
;(globalThis as unknown as { RESULT: unknown }).RESULT = result
console.log('RESULT ' + JSON.stringify(result))
