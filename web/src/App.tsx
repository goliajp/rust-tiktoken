import { useEffect, useState } from 'react'

import { CodeBlock } from './components/CodeBlock'
import { Playground } from './components/Playground'
import { EncodingTable, PerfTable } from './components/Tables'
import { detectLang, LANGS, LangContext, t as tr, type Lang } from './i18n'

const GITHUB = 'https://github.com/goliajp/rust-tiktoken'
const CRATES = 'https://crates.io/crates/tiktoken'
const NPM = 'https://www.npmjs.com/package/@goliapkg/tiktoken-wasm'
const DOCSRS = 'https://docs.rs/tiktoken'

const RUST_SNIPPET = `let enc = tiktoken::get_encoding("o200k_base").unwrap();
let tokens = enc.encode("hello world");        // → [24912, 2375]
let count  = enc.count("hello world");         // zero-alloc counting

// by model name — GPT, Kimi, GLM, DeepSeek, Qwen, …
let enc = tiktoken::encoding_for_model("kimi-k3").unwrap();`

const JS_SNIPPET = `import init, { getEncoding, encodingForModel } from '@goliapkg/tiktoken-wasm'

await init()                                   // load wasm once
const enc = getEncoding('o200k_base')
enc.encode('hello world')                      // → Uint32Array [24912, 2375]
encodingForModel('glm-5.2').count('你好世界')   // → 2`

export function App() {
  const [lang, setLang] = useState<Lang>('en')
  useEffect(() => {
    setLang(detectLang())
  }, [])
  useEffect(() => {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : lang
  }, [lang])
  const t = (k: string) => tr(lang, k)

  return (
    <LangContext.Provider value={lang}>
      <div className="shell">
        <header className="topbar">
          <a className="brand" href="/">
            <span className="dot">
              <i />
              <i />
              <i />
            </span>
            tiktoken
          </a>
          <nav className="topnav">
            <a href="#playground">{t('nav.playground')}</a>
            <a href="#encodings">{t('nav.encodings')}</a>
            <a href="#performance">{t('nav.performance')}</a>
            <a href="#install">{t('nav.install')}</a>
            <div className="langswitch" role="group" aria-label="language">
              {LANGS.map((l) => (
                <button
                  key={l.id}
                  className={lang === l.id ? 'on' : ''}
                  onClick={() => {
                    setLang(l.id)
                    localStorage.setItem('lang', l.id)
                  }}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </nav>
        </header>

        <section className="hero reveal">
          <h1>
            {t('hero.title.pre')}
            <span className="accent">{t('hero.title.accent')}</span>
          </h1>
          <p className="sub">{t('hero.sub')}</p>

          <div className="hero-stats">
            <div className="stat">
              <div className="n">15–40x</div>
              <div className="l">{t('hero.stat.speed')}</div>
            </div>
            <div className="stat">
              <div className="n">43 ns</div>
              <div className="l">{t('hero.stat.short')}</div>
            </div>
            <div className="stat">
              <div className="n">17</div>
              <div className="l">{t('hero.stat.encodings')}</div>
            </div>
            <div className="stat">
              <div className="n">107</div>
              <div className="l">{t('hero.stat.models')}</div>
            </div>
          </div>

          <div className="hero-links">
            <a className="btn primary" href="#playground">
              {t('hero.cta.try')}
            </a>
            <a className="btn" href={GITHUB} target="_blank" rel="noreferrer">
              {t('hero.cta.github')}
            </a>
            <a className="btn" href={CRATES} target="_blank" rel="noreferrer">
              {t('hero.cta.crates')}
            </a>
            <a className="btn" href={NPM} target="_blank" rel="noreferrer">
              {t('hero.cta.npm')}
            </a>
          </div>
        </section>

        <section id="playground-section" className="reveal" style={{ animationDelay: '90ms' }}>
          <div className="tag">{t('pg.tag')}</div>
          <h2>{t('pg.heading')}</h2>
          <p className="prose" style={{ marginBottom: '1.6rem' }}>
            {t('pg.blurb')}
          </p>
          <Playground />
        </section>

        <section id="features" className="reveal" style={{ animationDelay: '160ms' }}>
          <div className="tag">{t('feat.tag')}</div>
          <h2>{t('feat.heading')}</h2>
          <div className="grid3" style={{ marginTop: '1.6rem' }}>
            <div className="cell">
              <h3>
                <span className="k">01</span> {t('feat.exact.h')}
              </h3>
              <p>{t('feat.exact.p')}</p>
            </div>
            <div className="cell">
              <h3>
                <span className="k">02</span> {t('feat.fast.h')}
              </h3>
              <p>{t('feat.fast.p')}</p>
            </div>
            <div className="cell">
              <h3>
                <span className="k">03</span> {t('feat.everywhere.h')}
              </h3>
              <p>{t('feat.everywhere.p')}</p>
            </div>
          </div>
        </section>

        <section id="encodings">
          <div className="tag">{t('enc.tag')}</div>
          <h2>{t('enc.heading')}</h2>
          <p className="prose" style={{ marginBottom: '1.6rem' }}>
            {t('enc.blurb')}
          </p>
          <EncodingTable />
        </section>

        <section id="performance">
          <div className="tag">{t('perf.tag')}</div>
          <h2>{t('perf.heading')}</h2>
          <p className="prose" style={{ marginBottom: '1.6rem' }}>
            {t('perf.blurb')}
          </p>
          <PerfTable />
        </section>

        <section id="install">
          <div className="tag">{t('inst.tag')}</div>
          <h2>{t('inst.heading')}</h2>
          <div className="install-grid">
            <div>
              <p className="prose">{t('inst.rust.blurb')}</p>
              <CodeBlock label="cargo add tiktoken" copy={'cargo add tiktoken'}>
                <span className="cm"># Cargo.toml → tiktoken = "3"</span>
                {'\n\n'}
                {RUST_SNIPPET}
              </CodeBlock>
            </div>
            <div>
              <p className="prose">{t('inst.js.blurb')}</p>
              <CodeBlock label="npm install @goliapkg/tiktoken-wasm" copy={'npm install @goliapkg/tiktoken-wasm'}>
                {JS_SNIPPET}
              </CodeBlock>
            </div>
          </div>
          <p className="prose" style={{ marginTop: '1.6rem' }}>
            {t('inst.docs')}{' '}
            <a href={DOCSRS} target="_blank" rel="noreferrer">
              docs.rs/tiktoken
            </a>
          </p>
        </section>

        <footer>
          <span>{t('foot.license')}</span>
          <span style={{ color: 'var(--phosphor-dim)' }}>{t('foot.tagline')}</span>
          <div className="links">
            <a href={GITHUB} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <a href={CRATES} target="_blank" rel="noreferrer">
              crates.io
            </a>
            <a href={NPM} target="_blank" rel="noreferrer">
              npm
            </a>
            <a href={DOCSRS} target="_blank" rel="noreferrer">
              docs.rs
            </a>
          </div>
        </footer>
      </div>
    </LangContext.Provider>
  )
}
