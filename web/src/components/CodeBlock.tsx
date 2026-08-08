import { Check, Copy } from 'lucide-react'
import { Fragment, useState, type ReactNode } from 'react'

/**
 * A deliberately small syntax highlighter.
 *
 * The page shows two short, fixed snippets in two languages, so pulling in
 * Prism or Shiki would cost more (bundle, theme plumbing) than it returns.
 * This tokenizes with one ordered alternation — comments and strings are
 * matched before anything else, so a `//` inside a string literal and a quote
 * inside a comment both behave. Anything unmatched is emitted verbatim, so a
 * gap in the grammar degrades to plain text rather than to dropped source.
 */

export type Lang = 'rust' | 'js'

const KEYWORDS: Record<Lang, Set<string>> = {
  rust: new Set([
    'let', 'mut', 'const', 'fn', 'pub', 'use', 'impl', 'struct', 'enum', 'trait',
    'match', 'if', 'else', 'for', 'in', 'while', 'loop', 'return', 'self', 'crate',
    'move', 'ref', 'as', 'where', 'type', 'true', 'false',
  ]),
  js: new Set([
    'import', 'from', 'export', 'default', 'const', 'let', 'var', 'function',
    'await', 'async', 'return', 'new', 'if', 'else', 'for', 'of', 'in', 'while',
    'class', 'extends', 'true', 'false', 'null', 'undefined',
  ]),
}

// Ordered alternation. Comment and string come first so their contents are
// never re-scanned; the macro rule (`name!`) precedes the identifier rule.
const TOKEN = new RegExp(
  [
    '(?<comment>\\/\\/[^\\n]*)',
    '(?<string>"(?:\\\\.|[^"\\\\])*"|\'(?:\\\\.|[^\'\\\\])*\'|`(?:\\\\.|[^`\\\\])*`)',
    '(?<macro>[A-Za-z_][A-Za-z0-9_]*!)',
    '(?<number>\\b\\d[\\d_]*(?:\\.\\d+)?\\b)',
    '(?<ident>[A-Za-z_$][A-Za-z0-9_$]*)',
  ].join('|'),
  'g',
)

interface Piece {
  text: string
  kind?: 'comment' | 'string' | 'macro' | 'number' | 'keyword'
}

function tokenize(src: string, lang: Lang): Piece[] {
  const kw = KEYWORDS[lang]
  const out: Piece[] = []
  let last = 0

  for (const m of src.matchAll(TOKEN)) {
    const i = m.index!
    if (i > last) out.push({ text: src.slice(last, i) })
    const g = m.groups!
    if (g.comment) out.push({ text: g.comment, kind: 'comment' })
    else if (g.string) out.push({ text: g.string, kind: 'string' })
    else if (g.macro) out.push({ text: g.macro, kind: 'macro' })
    else if (g.number) out.push({ text: g.number, kind: 'number' })
    else if (g.ident) {
      out.push(kw.has(g.ident) ? { text: g.ident, kind: 'keyword' } : { text: g.ident })
    }
    last = i + m[0].length
  }
  if (last < src.length) out.push({ text: src.slice(last) })
  return out
}

/**
 * Offer the browser break opportunities at code punctuation.
 *
 * Without this, a long unbroken run like
 * `tiktoken::get_encoding("o200k_base").unwrap();` has no spaces, so a narrow
 * column splits it mid-identifier (`.u` / `nwrap();`). Breaking after `::`,
 * `.`, `(`, `,` and `/` instead puts the fold where a reader would put it.
 */
function withBreaks(text: string): ReactNode[] {
  const parts = text.split(/(::|[.(,/])/)
  const out: ReactNode[] = []
  parts.forEach((part, i) => {
    if (!part) return
    out.push(<Fragment key={i}>{part}</Fragment>)
    if (/^(::|[.(,/])$/.test(part)) out.push(<wbr key={`w${i}`} />)
  })
  return out
}

/**
 * Render one source line. Lines are block elements with a hanging indent so
 * that a wrapped line is visually subordinate to the line it belongs to —
 * the code column wraps rather than scrolling sideways.
 */
function Line({ src, lang }: { src: string; lang: Lang }) {
  if (src === '') return <span className="ln" />
  return (
    <span className="ln">
      {tokenize(src, lang).map((p, i) => (
        <span key={i} className={p.kind ? `tk-${p.kind}` : undefined}>
          {withBreaks(p.text)}
        </span>
      ))}
    </span>
  )
}

export function Code({ src, lang }: { src: string; lang: Lang }) {
  return (
    <>
      {src.split('\n').map((line, i) => (
        <Line key={i} src={line} lang={lang} />
      ))}
    </>
  )
}

export function CodeBlock({
  label,
  copy,
  className,
  children,
}: {
  label: string
  copy: string
  className?: string
  children: ReactNode
}) {
  const [copied, setCopied] = useState(false)
  return (
    <div className={className ? `codeblock ${className}` : 'codeblock'}>
      <div className="cb-head">
        <span>{label}</span>
        <button
          className="cb-copy"
          aria-label={copied ? 'copied' : 'copy'}
          onClick={() => {
            navigator.clipboard.writeText(copy).then(() => {
              setCopied(true)
              setTimeout(() => setCopied(false), 1400)
            })
          }}
        >
          {copied ? <Check size={14} strokeWidth={2.25} /> : <Copy size={14} strokeWidth={2} />}
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  )
}
