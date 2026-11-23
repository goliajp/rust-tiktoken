import { useT } from '../i18n'

// Static spec-sheet data. Single source of truth is the repository README /
// encoding.rs — keep in lockstep when encodings change.
const ENCODINGS: { name: string; vendor: string; models: string; vocab: string }[] = [
  { name: 'o200k_base', vendor: 'OpenAI', models: 'GPT-4o, GPT-4.1, GPT-5–5.6 (Sol/Terra/Luna), GPT-6.x (unconfirmed), o1, o3, o4-mini', vocab: '199,998' },
  { name: 'o200k_harmony', vendor: 'OpenAI', models: 'gpt-oss (harmony chat format)', vocab: '199,998' },
  { name: 'cl100k_base', vendor: 'OpenAI', models: 'GPT-4, GPT-3.5 Turbo, text-embedding-*', vocab: '100,256' },
  { name: 'p50k_base / p50k_edit', vendor: 'OpenAI', models: 'text-davinci-002/003, code-davinci', vocab: '50,256' },
  { name: 'r50k_base / gpt2', vendor: 'OpenAI', models: 'GPT-3 era, GPT-2', vocab: '50,256' },
  { name: 'llama3', vendor: 'Meta', models: 'Llama 3.x', vocab: '128,000' },
  { name: 'llama4', vendor: 'Meta', models: 'Llama 4, Muse', vocab: '200,000' },
  { name: 'deepseek_v3', vendor: 'DeepSeek', models: 'DeepSeek V3, R1', vocab: '128,000' },
  { name: 'deepseek_v4', vendor: 'DeepSeek', models: 'DeepSeek V4 Pro / Flash, V4.1 Flash', vocab: '128,000' },
  { name: 'qwen2', vendor: 'Alibaba', models: 'Qwen 2.5 / 3', vocab: '151,643' },
  { name: 'qwen3_5', vendor: 'Alibaba', models: 'Qwen 3.5 – 3.8, Qwen 4 (unconfirmed)', vocab: '248,044' },
  { name: 'mistral_v3', vendor: 'Mistral', models: 'Mistral / Mixtral, Large 3, Small 4 (Tekken)', vocab: '131,072' },
  { name: 'mistral_v15', vendor: 'Mistral', models: 'Mistral Large 4 (Tekken v15)', vocab: '204,800' },
  { name: 'kimi_k2', vendor: 'Moonshot', models: 'Kimi K2 / K2.5 / K2.6', vocab: '163,584' },
  { name: 'kimi_k3', vendor: 'Moonshot', models: 'Kimi K3 / K3.1', vocab: '163,584' },
  { name: 'glm4', vendor: 'Zhipu', models: 'GLM-4.5 / 4.6 / 4.7', vocab: '151,329' },
  { name: 'glm5', vendor: 'Zhipu', models: 'GLM-5 / 5.1 / 5.2 / 5.3', vocab: '154,820' },
  { name: 'minimax_m2', vendor: 'MiniMax', models: 'MiniMax M2.x / M3 / M3.1', vocab: '200,000' },
]

export function EncodingTable() {
  const t = useT()
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>
            <th>{t('enc.col.encoding')}</th>
            <th>{t('enc.col.vendor')}</th>
            <th>{t('enc.col.models')}</th>
            <th>{t('enc.col.vocab')}</th>
          </tr>
        </thead>
        <tbody>
          {ENCODINGS.map((e) => (
            <tr key={e.name}>
              <td className="mono">{e.name}</td>
              <td>{e.vendor}</td>
              <td>{e.models}</td>
              <td className="mono">{e.vocab}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Benchmark data — one full pass over the corpus, cl100k_base, warmup then
// median of 9 rounds, token outputs asserted identical across implementations
// before anything is timed. Corpora are byte-identical in every harness.
//   browser — Apple M4 Pro Mac mini, Chromium: `npm run bench` in web/
//   native  — Apple M4 Pro Mac mini, single thread: `TIKTOKEN_THREADS=1 cargo run -p bench-compare`;
//             fastokens from a separate same-process harness
// Corpus labels are i18n'd; the numbers are not.

const BROWSER: { key: string; ours: string; fast: string; gpt: string; js: string }[] = [
  { key: 'zh', ours: '4.8 µs', fast: '6.4 µs', gpt: '31.9 µs', js: '6,625 µs' },
  { key: 'ja', ours: '4.2 µs', fast: '6.6 µs', gpt: '29.2 µs', js: '12,825 µs' },
  { key: 'uni', ours: '6.3 µs', fast: '7.0 µs', gpt: '40.3 µs', js: '3,975 µs' },
  { key: 'varied', ours: '6.1 µs', fast: '6.7 µs', gpt: '42.2 µs', js: '2,756 µs' },
  { key: 'ascii', ours: '47.7 µs', fast: '118 µs', gpt: '277 µs', js: '9,638 µs' },
  { key: 'code', ours: '5.1 µs', fast: '8.9 µs', gpt: '30.4 µs', js: '1,120 µs' },
]

const NATIVE: { key: string; ours: string; fastokens: string; rs: string; py: string }[] = [
  { key: 'zh', ours: '2.0 µs', fastokens: '8.0 µs', rs: '97.8 µs', py: '114 µs' },
  { key: 'ja', ours: '1.4 µs', fastokens: '17.2 µs', rs: '119 µs', py: '131 µs' },
  { key: 'uni', ours: '2.5 µs', fastokens: '3.9 µs', rs: '108 µs', py: '122 µs' },
  { key: 'varied', ours: '2.6 µs', fastokens: '5.3 µs', rs: '94.0 µs', py: '114 µs' },
  { key: 'ascii', ours: '19.9 µs', fastokens: '35.6 µs', rs: '955 µs', py: '1,100 µs' },
  { key: 'code', ours: '1.9 µs', fastokens: '3.9 µs', rs: '163 µs', py: '179 µs' },
]

export function BrowserPerfTable() {
  const t = useT()
  return (
    <div className="tablewrap">
      <table className="perftable">
        <thead>
          <tr>
            <th>{t('perf.col.input')}</th>
            <th>tiktoken (wasm)</th>
            <th>tiktoken-fast 0.1</th>
            <th>gpt-tokenizer 4.1</th>
            <th>js-tiktoken 1.0</th>
          </tr>
        </thead>
        <tbody>
          {BROWSER.map((r) => (
            <tr key={r.key}>
              <td>{t(`perf.corpus.${r.key}`)}</td>
              <td>
                <span className="hl">{r.ours}</span>
              </td>
              <td className="mono">{r.fast}</td>
              <td className="mono">{r.gpt}</td>
              <td className="mono">{r.js}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function NativePerfTable() {
  const t = useT()
  return (
    <div className="tablewrap">
      <table className="perftable">
        <thead>
          <tr>
            <th>{t('perf.col.input')}</th>
            <th>tiktoken</th>
            <th>fastokens 0.3</th>
            <th>tiktoken-rs 0.12</th>
            <th>Python tiktoken 0.14</th>
          </tr>
        </thead>
        <tbody>
          {NATIVE.map((r) => (
            <tr key={r.key}>
              <td>{t(`perf.corpus.${r.key}`)}</td>
              <td>
                <span className="hl">{r.ours}</span>
              </td>
              <td className="mono">{r.fastokens}</td>
              <td className="mono">{r.rs}</td>
              <td className="mono">{r.py}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
