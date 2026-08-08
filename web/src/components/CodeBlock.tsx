import { useState, type ReactNode } from 'react'

export function CodeBlock({ label, copy, children }: { label: string; copy: string; children: ReactNode }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="codeblock">
      <div className="cb-head">
        <span>{label}</span>
        <button
          className="cb-copy"
          onClick={() => {
            navigator.clipboard.writeText(copy).then(() => {
              setCopied(true)
              setTimeout(() => setCopied(false), 1400)
            })
          }}
        >
          {copied ? '✓ copied' : 'copy'}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  )
}
