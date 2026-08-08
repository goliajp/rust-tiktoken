import { useEffect, useState } from 'react'

import { Code, CodeBlock } from './components/CodeBlock'
import { Playground } from './components/Playground'
import { EncodingTable, PerfTable } from './components/Tables'
import { detectLang, LANGS, LangContext, t as tr, type Lang } from './i18n'

const GITHUB = 'https://github.com/goliajp/rust-tiktoken'
const CRATES = 'https://crates.io/crates/tiktoken'
const NPM = 'https://www.npmjs.com/package/@goliapkg/tiktoken-wasm'
const DOCSRS = 'https://docs.rs/tiktoken'
const GOLIA = 'https://golia.jp'

const RUST_SNIPPET = `// Cargo.toml → tiktoken = "3"

let enc = tiktoken::get_encoding("o200k_base").unwrap();
let ids = enc.encode("hello world");  // [24912, 2375]
let n = enc.count("hello world");     // zero-alloc

// or by model name — GPT, Kimi, GLM, DeepSeek, Qwen, …
tiktoken::encoding_for_model("kimi-k3").unwrap();`

const JS_SNIPPET = `import init, { getEncoding, encodingForModel }
  from '@goliapkg/tiktoken-wasm'

await init()                     // load wasm once
const enc = getEncoding('o200k_base')
enc.encode('hello world')        // [24912, 2375]
encodingForModel('glm-5.2').count('你好世界')   // 2`

function Section({
  id,
  title,
  lede,
  children,
}: {
  id: string
  title: string
  lede?: string
  children: React.ReactNode
}) {
  return (
    <section id={id}>
      <div className="sechead">
        <h2>{title}</h2>
        {lede && <p className="lede">{lede}</p>}
      </div>
      {children}
    </section>
  )
}

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
      <header className="masthead">
        <div className="masthead-inner">
          <a className="brand" href="/">
            <img src="/golia-logo.png" alt="GOLIA" width={26} height={26} />
            <span className="wordmark">tiktoken</span>
            <span className="lab">{t('brand.lab')}</span>
          </a>
          <nav className="topnav">
            <a className="navlink" href="#playground">
              {t('nav.playground')}
            </a>
            <a className="navlink" href="#encodings">
              {t('nav.encodings')}
            </a>
            <a className="navlink" href="#performance">
              {t('nav.performance')}
            </a>
            <a className="navlink" href="#install">
              {t('nav.install')}
            </a>
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
        </div>
      </header>

      <div className="shell">
        <section className="frontmatter reveal">
          <div className="eyebrow">{t('front.eyebrow')}</div>
          <h1>
            {t('front.title.a')}
            <em>{t('front.title.b')}</em>
            {t('front.title.c')}
          </h1>
          <p className="abstract">{t('front.abstract')}</p>

          <div className="figures">
            <div className="figure">
              <div className="v">167,849</div>
              <div className="k">{t('front.fig.comparisons')}</div>
            </div>
            <div className="figure">
              <div className="v">17</div>
              <div className="k">{t('front.fig.encodings')}</div>
            </div>
            <div className="figure">
              <div className="v">43 ns</div>
              <div className="k">{t('front.fig.short')}</div>
            </div>
            <div className="figure">
              <div className="v">15–40×</div>
              <div className="k">{t('front.fig.speed')}</div>
            </div>
          </div>

          <div className="actions">
            <a className="btn primary" href="#playground">
              {t('front.cta.try')}
            </a>
            <a className="btn" href={GITHUB} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <a className="btn" href={CRATES} target="_blank" rel="noreferrer">
              crates.io
            </a>
            <a className="btn" href={NPM} target="_blank" rel="noreferrer">
              npm
            </a>
          </div>
        </section>

        <Section id="playground" title={t('pg.heading')} lede={t('pg.blurb')}>
          <Playground />
          <p className="caption">
            <b>{t('pg.caption.label')}</b> {t('pg.caption')}
          </p>
        </Section>

        <Section id="method" title={t('feat.heading')}>
          <div className="claims">
            <div className="claim">
              <h3>{t('feat.exact.h')}</h3>
              <p>{t('feat.exact.p')}</p>
            </div>
            <div className="claim">
              <h3>{t('feat.fast.h')}</h3>
              <p>{t('feat.fast.p')}</p>
            </div>
            <div className="claim">
              <h3>{t('feat.everywhere.h')}</h3>
              <p>{t('feat.everywhere.p')}</p>
            </div>
          </div>
        </Section>

        <Section id="encodings" title={t('enc.heading')} lede={t('enc.blurb')}>
          <EncodingTable />
          <p className="caption">
            <b>{t('enc.caption.label')}</b> {t('enc.caption')}
          </p>
        </Section>

        <Section id="performance" title={t('perf.heading')} lede={t('perf.blurb')}>
          <PerfTable />
          <p className="caption">
            <b>{t('perf.caption.label')}</b> {t('perf.caption')}
          </p>
        </Section>

        <Section id="install" title={t('inst.heading')}>
          {/* Direct grid children, in row order: both blurbs share row 1 and
              both code blocks share row 2, so the columns line up top and
              bottom even when one blurb wraps to more lines than the other.
              On one column, `order` restores blurb → code pairing. */}
          <div className="install-grid">
            <p className="prose i-blurb-a">{t('inst.rust.blurb')}</p>
            <p className="prose i-blurb-b">{t('inst.js.blurb')}</p>
            <CodeBlock
              className="i-code-a"
              label="cargo add tiktoken"
              copy="cargo add tiktoken"
            >
              <Code src={RUST_SNIPPET} lang="rust" />
            </CodeBlock>
            <CodeBlock
              className="i-code-b"
              label="npm install @goliapkg/tiktoken-wasm"
              copy="npm install @goliapkg/tiktoken-wasm"
            >
              <Code src={JS_SNIPPET} lang="js" />
            </CodeBlock>
          </div>
          <p className="caption">
            {t('inst.docs')}{' '}
            <a href={DOCSRS} target="_blank" rel="noreferrer">
              docs.rs/tiktoken
            </a>
          </p>
        </Section>

        <footer>
          <div>
            <a className="org" href={GOLIA} target="_blank" rel="noreferrer" style={{ borderBottom: 'none' }}>
              <img src="/golia-logo.png" alt="GOLIA" width={20} height={20} />
              <span>{t('foot.org')}</span>
            </a>
            <div>{t('foot.license')}</div>
          </div>
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
