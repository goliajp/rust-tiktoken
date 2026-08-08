// Thin wrapper over @goliapkg/tiktoken-wasm: one-time async init, per-encoding
// instance cache, and token → display-segment resolution for the playground.

import init, { getEncoding, listEncodings, type Encoding } from '@goliapkg/tiktoken-wasm'

let ready: Promise<void> | null = null

/** Kick off (or join) wasm initialization. */
export function initWasm(): Promise<void> {
  if (!ready) ready = init().then(() => undefined)
  return ready
}

const cache = new Map<string, Encoding>()

export function encoding(name: string): Encoding {
  let e = cache.get(name)
  if (!e) {
    e = getEncoding(name)
    cache.set(name, e)
  }
  return e
}

export function encodingNames(): string[] {
  return listEncodings()
}

export interface Segment {
  /** Rendered text for this token (may contain U+FFFD for byte fragments). */
  text: string
  id: number
}

/**
 * Encode text and resolve each token id back to its surface text.
 *
 * Tokens whose bytes are not self-contained UTF-8 (emoji split across
 * tokens, CJK bytes, …) decode to replacement characters individually; the
 * ids are still exact — only the per-token display is approximate, which the
 * UI communicates via the id tooltip.
 */
export function segments(name: string, text: string): { ids: number[]; segs: Segment[] } {
  const enc = encoding(name)
  const ids = Array.from(enc.encode(text))
  const one = new Uint32Array(1)
  const segs = ids.map((id) => {
    one[0] = id
    return { text: enc.decode(one), id }
  })
  return { ids, segs }
}

/** UTF-8 byte length without allocating an encoded copy per call site. */
const byteCounter = new TextEncoder()
export function utf8Len(text: string): number {
  return byteCounter.encode(text).length
}
