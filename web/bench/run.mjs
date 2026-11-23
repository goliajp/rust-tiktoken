#!/usr/bin/env node
// Serves .bench-dist and drives bench.ts in a real browser, because two wasm
// and two JIT'd JS tokenizers are being compared and only a browser runs them
// the way a user would.
//
//   npm run bench
//
// Prints one row per corpus: ns/op, median of 9 timed rounds after warmup.

import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync, readdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { extname, join } from 'node:path'

import { chromium } from 'playwright-core'

const DIST = new URL('../.bench-dist/', import.meta.url)
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' }

const server = createServer(async (req, res) => {
  const path = req.url === '/' ? '/index.html' : req.url.split('?')[0]
  try {
    const body = await readFile(new URL('.' + path, DIST))
    res.writeHead(200, { 'content-type': TYPES[extname(path)] ?? 'application/octet-stream' })
    res.end(body)
  } catch {
    res.writeHead(404).end()
  }
})
await new Promise((r) => server.listen(0, r))
const url = `http://localhost:${server.address().port}/`

function findBrowser() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH
  const cache = join(homedir(), 'Library/Caches/ms-playwright')
  for (const d of readdirSync(cache).filter((x) => x.startsWith('chromium'))) {
    for (const r of [
      'chrome-mac/Chromium.app/Contents/MacOS/Chromium',
      'chrome-headless-shell-mac-arm64/chrome-headless-shell',
      'chrome-linux/chrome',
    ]) {
      const p = join(cache, d, r)
      if (existsSync(p)) return p
    }
  }
  throw new Error('no Chromium found — set CHROME_PATH')
}

const browser = await chromium.launch({ executablePath: findBrowser() })
const page = await browser.newPage()
page.on('pageerror', (e) => {
  console.error('page error:', e.message)
  process.exitCode = 1
})
await page.goto(url, { waitUntil: 'networkidle', timeout: 180_000 })
const r = await page.waitForFunction(() => globalThis.RESULT, { timeout: 600_000 })
const { initMs, jsInitMs, out } = await r.jsonValue()
await browser.close()
server.close()

const us = (ns) => (ns / 1000).toFixed(2).padStart(9) + ' µs'
console.log(`\nwasm init ${initMs} ms · js-tiktoken init ${jsInitMs} ms\n`)
console.log(
  'corpus'.padEnd(20) +
    ['ours encode', 'ours count', 'tiktoken-fast', 'gpt-tokenizer', 'js-tiktoken'].map((h) => h.padStart(15)).join('') +
    '  tokens',
)
for (const o of out) {
  if (!o.agree) throw new Error(`${o.name}: the tokenizers disagree — timings are meaningless`)
  console.log(
    o.name.padEnd(20) +
      [o.wasm, o.wasmCount, o.tiktokenFast, o.gptTokenizer, o.jsTiktoken].map((v) => us(v).padStart(15)).join('') +
      '  ' +
      o.tokens,
  )
}
