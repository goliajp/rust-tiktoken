import { useT } from '../i18n'

// Static spec-sheet data. Single source of truth is the repository README /
// encoding.rs — keep in lockstep when encodings change.
const ENCODINGS: { name: string; vendor: string; models: string; vocab: string }[] = [
  { name: 'o200k_base', vendor: 'OpenAI', models: 'GPT-4o, GPT-4.1, GPT-5–5.6 (Sol/Terra/Luna), o1, o3, o4-mini', vocab: '199,998' },
  { name: 'o200k_harmony', vendor: 'OpenAI', models: 'gpt-oss (harmony chat format)', vocab: '199,998' },
  { name: 'cl100k_base', vendor: 'OpenAI', models: 'GPT-4, GPT-3.5 Turbo, text-embedding-*', vocab: '100,256' },
  { name: 'p50k_base / p50k_edit', vendor: 'OpenAI', models: 'text-davinci-002/003, code-davinci', vocab: '50,256' },
  { name: 'r50k_base / gpt2', vendor: 'OpenAI', models: 'GPT-3 era, GPT-2', vocab: '50,256' },
  { name: 'llama3', vendor: 'Meta', models: 'Llama 3.x / 4', vocab: '128,000' },
  { name: 'deepseek_v3', vendor: 'DeepSeek', models: 'DeepSeek V3, R1', vocab: '128,000' },
  { name: 'deepseek_v4', vendor: 'DeepSeek', models: 'DeepSeek V4 Pro / Flash', vocab: '128,000' },
  { name: 'qwen2', vendor: 'Alibaba', models: 'Qwen 2.5 / 3', vocab: '151,643' },
  { name: 'mistral_v3', vendor: 'Mistral', models: 'Mistral / Mixtral (Tekken)', vocab: '131,072' },
  { name: 'kimi_k2', vendor: 'Moonshot', models: 'Kimi K2 / K2.5 / K2.6', vocab: '163,584' },
  { name: 'kimi_k3', vendor: 'Moonshot', models: 'Kimi K3', vocab: '163,584' },
  { name: 'glm4', vendor: 'Zhipu', models: 'GLM-4.5 / 4.6 / 4.7', vocab: '151,329' },
  { name: 'glm5', vendor: 'Zhipu', models: 'GLM-5 / 5.2', vocab: '154,820' },
  { name: 'minimax_m2', vendor: 'MiniMax', models: 'MiniMax M2 / M2.1 / M2.5 / M2.7', vocab: '200,000' },
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
//   browser — Mac Studio (M4 Max), Chromium: `npm run bench` in web/
//   native  — Apple M4 Mac mini, single thread: `cargo run -p bench-compare`
// Corpus labels are i18n'd; the numbers are not.

const BROWSER: { key: string; ours: string; gpt: string; js: string }[] = [
  { key: 'zh', ours: '13.4 µs', gpt: '36.8 µs', js: '8,029 µs' },
  { key: 'ja', ours: '13.5 µs', gpt: '27.4 µs', js: '15,862 µs' },
  { key: 'uni', ours: '24.2 µs', gpt: '41.2 µs', js: '4,665 µs' },
  { key: 'varied', ours: '40.3 µs', gpt: '49.6 µs', js: '3,832 µs' },
  { key: 'ascii', ours: '112.5 µs', gpt: '478 µs', js: '7,010 µs' },
  { key: 'code', ours: '19.5 µs', gpt: '76.0 µs', js: '916 µs' },
]

const NATIVE: { key: string; ours: string; rs: string; py: string }[] = [
  { key: 'zh', ours: '8.1 µs', rs: '135 µs', py: '120 µs' },
  { key: 'ja', ours: '8.6 µs', rs: '145 µs', py: '131 µs' },
  { key: 'uni', ours: '15.2 µs', rs: '160 µs', py: '139 µs' },
  { key: 'varied', ours: '25.9 µs', rs: '141 µs', py: '132 µs' },
  { key: 'ascii', ours: '51.5 µs', rs: '2,498 µs', py: '1,500 µs' },
  { key: 'code', ours: '11.1 µs', rs: '318 µs', py: '264 µs' },
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
            <th>gpt-tokenizer 3.4</th>
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
            <th>tiktoken-rs 0.9</th>
            <th>Python tiktoken 0.12</th>
          </tr>
        </thead>
        <tbody>
          {NATIVE.map((r) => (
            <tr key={r.key}>
              <td>{t(`perf.corpus.${r.key}`)}</td>
              <td>
                <span className="hl">{r.ours}</span>
              </td>
              <td className="mono">{r.rs}</td>
              <td className="mono">{r.py}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
