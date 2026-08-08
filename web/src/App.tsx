import { ArrowDown, ArrowUpRight } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Code, CodeBlock } from './components/CodeBlock'
import { Playground } from './components/Playground'
import { EncodingTable, PerfTable } from './components/Tables'
import { detectLang, LANGS, LangContext, T, type Lang } from './i18n'

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
  title: React.ReactNode
  lede?: React.ReactNode
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
  return (
    <LangContext.Provider value={lang}>
      <header className="masthead">
        <div className="masthead-inner">
          <a className="brand" href="/">
            <img src="/golia-logo.png" alt="GOLIA" width={26} height={26} />
            <span className="wordmark">tiktoken</span>
          </a>
          <nav className="topnav">
            <a className="navlink" href="#playground">
              <T k="nav.playground" />
            </a>
            <a className="navlink" href="#encodings">
              <T k="nav.encodings" />
            </a>
            <a className="navlink" href="#performance">
              <T k="nav.performance" />
            </a>
            <a className="navlink" href="#install">
              <T k="nav.install" />
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
          <div className="eyebrow"><T k="front.eyebrow" /></div>
          <h1>
            <T k="front.title.a" />
            <em><T k="front.title.b" /></em>
            <T k="front.title.c" />
          </h1>
          <p className="abstract"><T k="front.abstract" /></p>

          <div className="figures">
            <div className="figure">
              <div className="v">167,849</div>
              <div className="k"><T k="front.fig.comparisons" /></div>
            </div>
            <div className="figure">
              <div className="v">17</div>
              <div className="k"><T k="front.fig.encodings" /></div>
            </div>
            <div className="figure">
              <div className="v">43 ns</div>
              <div className="k"><T k="front.fig.short" /></div>
            </div>
            <div className="figure">
              <div className="v">15–40×</div>
              <div className="k"><T k="front.fig.speed" /></div>
            </div>
          </div>

          <div className="actions">
            <a className="btn primary" href="#playground">
              <T k="front.cta.try" />
              <ArrowDown size={15} strokeWidth={2.25} />
            </a>
            {[
              ['GitHub', GITHUB],
              ['crates.io', CRATES],
              ['npm', NPM],
            ].map(([label, href]) => (
              <a key={label} className="btn" href={href} target="_blank" rel="noreferrer">
                {label}
                <ArrowUpRight size={14} strokeWidth={2} className="ext" />
              </a>
            ))}
          </div>
        </section>

        <Section id="playground" title={<T k="pg.heading" />} lede={<T k="pg.blurb" />}>
          <Playground />
          <p className="caption">
            <b><T k="pg.caption.label" /></b> <T k="pg.caption" />
          </p>
        </Section>

        <Section id="method" title={<T k="feat.heading" />}>
          <div className="claims">
            <div className="claim">
              <h3><T k="feat.exact.h" /></h3>
              <p><T k="feat.exact.p" /></p>
            </div>
            <div className="claim">
              <h3><T k="feat.fast.h" /></h3>
              <p><T k="feat.fast.p" /></p>
            </div>
            <div className="claim">
              <h3><T k="feat.everywhere.h" /></h3>
              <p><T k="feat.everywhere.p" /></p>
            </div>
          </div>
        </Section>

        <Section id="encodings" title={<T k="enc.heading" />} lede={<T k="enc.blurb" />}>
          <EncodingTable />
          <p className="caption">
            <b><T k="enc.caption.label" /></b> <T k="enc.caption" />
          </p>
        </Section>

        <Section id="performance" title={<T k="perf.heading" />} lede={<T k="perf.blurb" />}>
          <PerfTable />
          <p className="caption">
            <b><T k="perf.caption.label" /></b> <T k="perf.caption" />
          </p>
        </Section>

        <Section id="install" title={<T k="inst.heading" />}>
          {/* Direct grid children, in row order: both blurbs share row 1 and
              both code blocks share row 2, so the columns line up top and
              bottom even when one blurb wraps to more lines than the other.
              On one column, `order` restores blurb → code pairing. */}
          <div className="install-grid">
            <p className="prose i-blurb-a"><T k="inst.rust.blurb" /></p>
            <p className="prose i-blurb-b"><T k="inst.js.blurb" /></p>
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
            <T k="inst.docs" />
            {/* The Chinese and Japanese strings end in a full-width colon,
                which already carries its own trailing space. */}
            {lang === 'en' ? ' ' : ''}
            <a href={DOCSRS} target="_blank" rel="noreferrer">
              docs.rs/tiktoken
            </a>
          </p>
        </Section>

        <footer>
          <div>
            <a className="org" href={GOLIA} target="_blank" rel="noreferrer" aria-label="GOLIA">
              <img src="/golia-wordmark.png" alt="GOLIA" width={92} height={20} />
            </a>
            <div><T k="foot.license" /></div>
          </div>
          <div className="links">
            {[
              ['GitHub', GITHUB],
              ['crates.io', CRATES],
              ['npm', NPM],
              ['docs.rs', DOCSRS],
            ].map(([label, href]) => (
              <a key={label} href={href} target="_blank" rel="noreferrer">
                {label}
                <ArrowUpRight size={12} strokeWidth={2} className="ext" />
              </a>
            ))}
          </div>
        </footer>
      </div>
    </LangContext.Provider>
  )
}
