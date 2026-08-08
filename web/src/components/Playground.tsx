import { useEffect, useMemo, useState } from 'react'

import { useT } from '../i18n'
import { encodingNames, initWasm, segments, utf8Len, type Segment } from '../tokenizer'

const EXAMPLE = `Many words map to one token, but some don't: indivisible.

分词器把文本切成 token：模型眼中的最小单位。
トークナイザーはテキストをトークンへ分割します。

Sequences of characters commonly found next to each other may be grouped: 1234567890
Emoji are split into their underlying bytes: 🤚🏾`

// Encodings we surface first in the picker; the rest follow alphabetically.
const FEATURED = ['o200k_base', 'cl100k_base', 'kimi_k3', 'glm5', 'deepseek_v4', 'qwen2', 'minimax_m2', 'llama3', 'mistral_v3']

type WasmState = 'loading' | 'ready' | 'error'

export function Playground() {
  const t = useT()
  const [state, setState] = useState<WasmState>('loading')
  const [names, setNames] = useState<string[]>([])
  const [enc, setEnc] = useState('o200k_base')
  const [text, setText] = useState('')
  const [view, setView] = useState<'text' | 'ids'>('text')

  useEffect(() => {
    initWasm()
      .then(() => {
        const all = encodingNames()
        const featured = FEATURED.filter((n) => all.includes(n))
        const rest = all.filter((n) => !featured.includes(n)).sort()
        setNames([...featured, ...rest])
        setState('ready')
      })
      .catch(() => setState('error'))
  }, [])

  const result = useMemo(() => {
    if (state !== 'ready' || !text) return { ids: [] as number[], segs: [] as Segment[] }
    try {
      return segments(enc, text)
    } catch {
      return { ids: [], segs: [] }
    }
  }, [state, enc, text])

  return (
    <div className="playground" id="playground">
      <div className="pg-head">
        <span className="title">{t('pg.title')}</span>
        <div className="pg-controls">
          <button className="cb-copy" onClick={() => setText(EXAMPLE)} style={{ fontSize: '0.78rem' }}>
            {t('pg.example')}
          </button>
          <select value={enc} onChange={(e) => setEnc(e.target.value)} disabled={state !== 'ready'} aria-label="encoding">
            {(names.length ? names : [enc]).map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>
      </div>

      {state === 'loading' && (
        <div className="pg-loading">
          <div className="bar">
            <i />
          </div>
          {t('pg.loading')}
        </div>
      )}
      {state === 'error' && <div className="pg-loading">{t('pg.error')}</div>}

      {state === 'ready' && (
        <>
          <div className="pg-body">
            <div className="pg-input">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={t('pg.placeholder')}
                spellCheck={false}
              />
            </div>
            <div className="pg-output">
              <div className="pg-meters">
                <div className="meter">
                  <div className="n">{result.ids.length.toLocaleString()}</div>
                  <div className="l">{t('pg.tokens')}</div>
                </div>
                <div className="meter">
                  <div className="n">{[...text].length.toLocaleString()}</div>
                  <div className="l">{t('pg.chars')}</div>
                </div>
                <div className="meter">
                  <div className="n">{utf8Len(text).toLocaleString()}</div>
                  <div className="l">{t('pg.bytes')}</div>
                </div>
              </div>
              <div className="pg-tokens">
                {view === 'text' ? (
                  result.segs.map((s, i) => (
                    <span
                      key={i}
                      className={`tok c${i % 7}`}
                      style={{ animationDelay: `${Math.min(i * 6, 400)}ms` }}
                      title={`id ${s.id}`}
                    >
                      {s.text}
                    </span>
                  ))
                ) : (
                  <div className="pg-ids">
                    [{result.ids.map((id, i) => (
                      <span key={i}>
                        {i > 0 && ', '}
                        <b>{id}</b>
                      </span>
                    ))}]
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="pg-foot">
            <span>{t('pg.foot')}</span>
            <div className="viewswitch">
              <button className={view === 'text' ? 'on' : ''} onClick={() => setView('text')}>
                {t('pg.view.text')}
              </button>
              <button className={view === 'ids' ? 'on' : ''} onClick={() => setView('ids')}>
                {t('pg.view.ids')}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
