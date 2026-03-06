import { useState, useEffect, useCallback, useRef } from 'react'
import init, { getEncoding, estimateCost, getModelInfo, type Encoding } from 'tiktoken-wasm'

const ENCODINGS = ['cl100k_base', 'o200k_base', 'p50k_base', 'p50k_edit', 'r50k_base'] as const

const MODELS = [
  'gpt-4o', 'gpt-4o-mini', 'o1', 'o3', 'o4-mini',
  'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',
  'claude-opus-4', 'claude-sonnet-4', 'claude-3.5-haiku',
  'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash',
] as const

interface ModelInfo {
  id: string
  provider: string
  input_per_1m: number
  output_per_1m: number
  cached_input_per_1m: number | null
  context_window: number
  max_output: number
}

function App() {
  const [ready, setReady] = useState(false)
  const [text, setText] = useState('Hello, world! 你好世界 🚀')
  const [encodingName, setEncodingName] = useState<string>('cl100k_base')
  const [tokens, setTokens] = useState<number[]>([])
  const [count, setCount] = useState(0)
  const [decoded, setDecoded] = useState('')
  const [encodeTime, setEncodeTime] = useState(0)
  const [selectedModel, setSelectedModel] = useState<string>('gpt-4o')
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [cost, setCost] = useState<number | null>(null)
  const encodingRef = useRef<Encoding | null>(null)

  useEffect(() => {
    init().then(() => setReady(true))
  }, [])

  useEffect(() => {
    if (!ready) return
    try {
      encodingRef.current?.free()
      encodingRef.current = getEncoding(encodingName)
    } catch (e) {
      console.error('failed to get encoding:', e)
    }
  }, [ready, encodingName])

  const handleEncode = useCallback(() => {
    const enc = encodingRef.current
    if (!enc) return

    const start = performance.now()
    const tokenIds = enc.encode(text)
    const elapsed = performance.now() - start

    const tokenArray = Array.from(tokenIds)
    setTokens(tokenArray)
    setCount(enc.count(text))
    setDecoded(enc.decode(tokenIds))
    setEncodeTime(elapsed)
  }, [text])

  useEffect(() => {
    if (ready && encodingRef.current) {
      handleEncode()
    }
  }, [ready, encodingName, handleEncode])

  useEffect(() => {
    if (!ready) return
    try {
      const info = getModelInfo(selectedModel) as ModelInfo
      setModelInfo(info)
      const c = estimateCost(selectedModel, count, count)
      setCost(c)
    } catch {
      setModelInfo(null)
      setCost(null)
    }
  }, [ready, selectedModel, count])

  if (!ready) {
    return <div style={styles.container}><p>Loading WASM module...</p></div>
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>tiktoken-wasm Demo</h1>
      <p style={styles.subtitle}>High-performance BPE tokenizer in your browser via WebAssembly</p>

      <div style={styles.section}>
        <label style={styles.label}>Encoding</label>
        <select
          value={encodingName}
          onChange={e => setEncodingName(e.target.value)}
          style={styles.select}
        >
          {ENCODINGS.map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      <div style={styles.section}>
        <label style={styles.label}>Input Text</label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          rows={6}
          style={styles.textarea}
          placeholder="Type or paste text here..."
        />
        <button onClick={handleEncode} style={styles.button}>
          Encode
        </button>
      </div>

      <div style={styles.statsRow}>
        <div style={styles.statCard}>
          <div style={styles.statValue}>{count}</div>
          <div style={styles.statLabel}>tokens</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statValue}>{text.length}</div>
          <div style={styles.statLabel}>characters</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statValue}>{encodeTime.toFixed(2)}ms</div>
          <div style={styles.statLabel}>encode time</div>
        </div>
      </div>

      <div style={styles.section}>
        <label style={styles.label}>Token IDs ({tokens.length})</label>
        <div style={styles.tokenBox}>
          {tokens.map((t, i) => (
            <span key={i} style={styles.token}>{t}</span>
          ))}
        </div>
      </div>

      <div style={styles.section}>
        <label style={styles.label}>Decoded</label>
        <div style={styles.decoded}>{decoded}</div>
        <div style={{ ...styles.match, color: decoded === text ? '#22c55e' : '#ef4444' }}>
          {decoded === text ? '✓ roundtrip match' : '✗ mismatch'}
        </div>
      </div>

      <hr style={styles.divider} />

      <h2 style={styles.sectionTitle}>Cost Estimation</h2>

      <div style={styles.section}>
        <label style={styles.label}>Model</label>
        <select
          value={selectedModel}
          onChange={e => setSelectedModel(e.target.value)}
          style={styles.select}
        >
          {MODELS.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {modelInfo && (
        <div style={styles.infoGrid}>
          <div><strong>Provider:</strong> {modelInfo.provider}</div>
          <div><strong>Context:</strong> {modelInfo.context_window.toLocaleString()} tokens</div>
          <div><strong>Max Output:</strong> {modelInfo.max_output.toLocaleString()} tokens</div>
          <div><strong>Input:</strong> ${modelInfo.input_per_1m}/1M tokens</div>
          <div><strong>Output:</strong> ${modelInfo.output_per_1m}/1M tokens</div>
          {modelInfo.cached_input_per_1m != null && (
            <div><strong>Cached Input:</strong> ${modelInfo.cached_input_per_1m}/1M tokens</div>
          )}
          {cost != null && (
            <div>
              <strong>Estimated Cost:</strong> ${cost.toFixed(6)}
              <span style={styles.costNote}> ({count} input + {count} output tokens)</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 720,
    margin: '0 auto',
    padding: '2rem',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: '#1a1a1a',
  },
  title: {
    fontSize: '2rem',
    fontWeight: 700,
    margin: 0,
  },
  subtitle: {
    color: '#666',
    marginTop: '0.5rem',
    marginBottom: '2rem',
  },
  section: {
    marginBottom: '1.5rem',
  },
  sectionTitle: {
    fontSize: '1.3rem',
    fontWeight: 600,
    marginBottom: '1rem',
  },
  label: {
    display: 'block',
    fontWeight: 600,
    marginBottom: '0.5rem',
    fontSize: '0.9rem',
  },
  select: {
    width: '100%',
    padding: '0.5rem',
    fontSize: '1rem',
    borderRadius: 6,
    border: '1px solid #ccc',
  },
  textarea: {
    width: '100%',
    padding: '0.75rem',
    fontSize: '1rem',
    borderRadius: 6,
    border: '1px solid #ccc',
    fontFamily: 'monospace',
    resize: 'vertical' as const,
    boxSizing: 'border-box' as const,
  },
  button: {
    marginTop: '0.5rem',
    padding: '0.5rem 1.5rem',
    fontSize: '1rem',
    fontWeight: 600,
    color: '#fff',
    backgroundColor: '#0066ff',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
  },
  statsRow: {
    display: 'flex',
    gap: '1rem',
    marginBottom: '1.5rem',
  },
  statCard: {
    flex: 1,
    padding: '1rem',
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    textAlign: 'center' as const,
  },
  statValue: {
    fontSize: '1.5rem',
    fontWeight: 700,
  },
  statLabel: {
    fontSize: '0.8rem',
    color: '#666',
    marginTop: '0.25rem',
  },
  tokenBox: {
    display: 'flex',
    flexWrap: 'wrap' as const,
    gap: '0.25rem',
    padding: '0.75rem',
    backgroundColor: '#f9f9f9',
    borderRadius: 6,
    border: '1px solid #eee',
    maxHeight: 200,
    overflowY: 'auto' as const,
    fontFamily: 'monospace',
  },
  token: {
    display: 'inline-block',
    padding: '2px 6px',
    backgroundColor: '#e0e7ff',
    borderRadius: 4,
    fontSize: '0.85rem',
  },
  decoded: {
    padding: '0.75rem',
    backgroundColor: '#f9f9f9',
    borderRadius: 6,
    border: '1px solid #eee',
    fontFamily: 'monospace',
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-all' as const,
  },
  match: {
    marginTop: '0.25rem',
    fontSize: '0.85rem',
  },
  divider: {
    border: 'none',
    borderTop: '1px solid #eee',
    margin: '2rem 0',
  },
  infoGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '0.5rem',
    padding: '1rem',
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    fontSize: '0.9rem',
  },
  costNote: {
    fontSize: '0.8rem',
    color: '#888',
  },
}

export default App
