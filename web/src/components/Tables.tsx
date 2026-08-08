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

// cl100k_base encode, Apple M4 Mac mini — mirrors the README benchmark table.
const PERF: { input: string; python: string; rs: string; ours: string; speedup: string }[] = [
  { input: 'short (13 B)', python: '1,700 ns', rs: '1,248 ns', ours: '43 ns', speedup: '29x' },
  { input: 'medium (900 B)', python: '32.2 µs', rs: '53.8 µs', ours: '1.5 µs', speedup: '35x' },
  { input: 'long (45 KB)', python: '1,500 µs', rs: '2,611 µs', ours: '74 µs', speedup: '35x' },
  { input: 'unicode (4.5 KB)', python: '141 µs', rs: '164 µs', ours: '91 µs', speedup: '1.8x' },
  { input: 'code (3.9 KB)', python: '247 µs', rs: '264 µs', ours: '17 µs', speedup: '16x' },
]

export function PerfTable() {
  const t = useT()
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>
            <th>{t('perf.col.input')}</th>
            <th>{t('perf.col.python')}</th>
            <th>{t('perf.col.rs')}</th>
            <th>{t('perf.col.ours')}</th>
            <th>{t('perf.col.speedup')}</th>
          </tr>
        </thead>
        <tbody>
          {PERF.map((r) => (
            <tr key={r.input}>
              <td className="mono">{r.input}</td>
              <td className="mono">{r.python}</td>
              <td className="mono">{r.rs}</td>
              <td>
                <span className="hl">{r.ours}</span>
              </td>
              <td className="mono">{r.speedup}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
