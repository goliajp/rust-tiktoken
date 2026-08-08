#!/usr/bin/env node
// Render the site in a real browser and assert the things a build cannot:
// no horizontal overflow at three viewports, no console/page errors, the
// GOLIA mark actually loads, the light treatment survives a dark colour-scheme
// preference, and the wasm playground returns tokens for real input.
//
// This is not decoration — every check here guards a class of defect a build
// cannot see (grid items default to min-width:auto and silently widen the
// page; font stacks reflow text differently per engine).
//
//   node verify.mjs [url]        default: http://localhost:6040
//   node verify.mjs https://tiktoken.golia.jp
//
// Needs a Chromium. Resolution order: $CHROME_PATH, the Playwright browser
// cache, then the system Google Chrome.

import { existsSync, readdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { chromium } from 'playwright-core'

const URL_ = process.argv[2] ?? 'http://localhost:6040'

function findBrowser() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH
  const cache = join(homedir(), 'Library/Caches/ms-playwright')
  if (existsSync(cache)) {
    for (const dir of readdirSync(cache).filter((d) => d.startsWith('chromium'))) {
      for (const rel of [
        'chrome-headless-shell-mac-arm64/chrome-headless-shell',
        'chrome-mac/Chromium.app/Contents/MacOS/Chromium',
        'chrome-linux/chrome',
      ]) {
        const p = join(cache, dir, rel)
        if (existsSync(p)) return p
      }
    }
  }
  const sys = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  if (existsSync(sys)) return sys
  throw new Error('no Chromium found — set CHROME_PATH')
}

const failures = []
const browser = await chromium.launch({ executablePath: findBrowser() })

for (const [width, height, label] of [
  [1440, 1000, 'desktop'],
  [834, 1112, 'tablet'],
  [390, 844, 'mobile'],
]) {
  const page = await browser.newPage({ viewport: { width, height } })
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e.message)))
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))

  await page.goto(URL_, { waitUntil: 'networkidle', timeout: 120_000 })
  await page.waitForTimeout(1200)

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  if (overflow > 1) {
    // name the widest offender so the report is actionable
    const who = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth
      let worst = null
      for (const el of document.querySelectorAll('*')) {
        if (el.scrollWidth > vw + 1 && getComputedStyle(el).overflowX === 'visible') {
          if (!worst || el.scrollWidth > worst.w)
            worst = { sel: el.tagName + '.' + (el.className || ''), w: el.scrollWidth }
        }
      }
      return worst
    })
    failures.push(`${label}: ${overflow}px horizontal overflow — widest: ${who?.sel} (${who?.w}px)`)
  }
  if (errors.length) failures.push(`${label}: ${errors.length} console/page errors — ${errors[0]}`)

  console.log(`${label.padEnd(8)} overflow=${overflow}px errors=${errors.length}`)
  await page.close()
}

// brand mark + playground, on desktop
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  await page.goto(URL_, { waitUntil: 'networkidle', timeout: 120_000 })
  await page.waitForTimeout(1200)

  const logoOk = await page
    .locator('.brand img')
    .first()
    .evaluate((i) => i.naturalWidth > 0)
  if (!logoOk) failures.push('the GOLIA mark did not load')

  const wordmarkOk = await page
    .locator('footer .org img')
    .first()
    .evaluate((i) => i.naturalWidth > 0)
  if (!wordmarkOk) failures.push('the GOLIA wordmark did not load')

  await page.locator('.pg-head .linkbtn').first().click()
  await page.waitForTimeout(1200)
  const tokens = Number((await page.locator('.meter.lead .v').first().textContent())?.replace(/,/g, ''))
  if (!Number.isFinite(tokens) || tokens <= 0) failures.push(`playground produced no tokens (${tokens})`)

  // The two panes are read across, so they must start and end level and share
  // type metrics; the install columns must line up the same way. Copy of
  // differing lengths above either pair is what pushes them apart.
  const align = await page.evaluate(() => {
    const r = (s) => document.querySelector(s)?.getBoundingClientRect()
    const cs = (s) => getComputedStyle(document.querySelector(s))
    const ta = r('.pg-input textarea')
    const tk = r('.pg-tokens')
    const ca = r('.i-code-a')
    const cb = r('.i-code-b')
    const a = cs('.pg-input textarea')
    const c = cs('.pg-tokens')
    return {
      paneTop: Math.round(ta.top - tk.top),
      paneBottom: Math.round(ta.bottom - tk.bottom),
      sameType:
        a.fontSize === c.fontSize && a.lineHeight === c.lineHeight && a.paddingTop === c.paddingTop,
      codeTop: Math.round(ca.top - cb.top),
      codeBottom: Math.round(ca.bottom - cb.bottom),
    }
  })
  for (const [k, v] of [
    ['playground panes top', align.paneTop],
    ['playground panes bottom', align.paneBottom],
    ['install columns top', align.codeTop],
    ['install columns bottom', align.codeBottom],
  ]) {
    if (Math.abs(v) > 1) failures.push(`${k} misaligned by ${v}px`)
  }
  if (!align.sameType) failures.push('input and token panes do not share type metrics')

  // Everything shares one left edge: headings, ledes and the content they
  // introduce. A heading indented away from its own table reads as a
  // mistake, and it was one.
  const edges = await page.evaluate(() => {
    const L = (s) => Math.round(document.querySelector(s).getBoundingClientRect().left)
    const base = L('.playground')
    return {
      base,
      h1: L('.frontmatter h1') - base,
      eyebrow: L('.eyebrow') - base,
      h2: L('.sechead h2') - base,
      lede: L('.sechead .lede') - base,
      table: L('.tablewrap') - base,
    }
  })
  for (const [k, v] of Object.entries(edges)) {
    if (k !== 'base' && Math.abs(v) > 1) failures.push(`${k} left edge off by ${v}px`)
  }

  // A caption belongs to the table or figure above it, so it spans that
  // width. Holding it to the prose measure made it narrower than its own
  // subject and orphaned its last words onto a short second line.
  const captions = await page.evaluate(() =>
    [...document.querySelectorAll('.caption')].map((c) => {
      const subject = c.previousElementSibling
      return {
        id: c.closest('section')?.id ?? '?',
        delta: Math.round(c.getBoundingClientRect().width - subject.getBoundingClientRect().width),
      }
    }),
  )
  for (const c of captions) {
    if (Math.abs(c.delta) > 1) failures.push(`#${c.id} caption is ${c.delta}px off its subject's width`)
  }
  console.log(`captions=${captions.map((c) => `${c.id}${c.delta >= 0 ? '+' : ''}${c.delta}`).join(' ')}`)

  // The two benchmark tables are read as one comparison: every column edge
  // must sit at the same x in both, or the eye re-reads the header per table.
  const perfCols = await page.evaluate(() => {
    const tables = [...document.querySelectorAll('.perftable')]
    if (tables.length !== 2) return { error: `${tables.length} perftables` }
    const xs = tables.map((t) =>
      [...t.querySelectorAll('thead th')].map((th) => Math.round(th.getBoundingClientRect().left)),
    )
    const drift = xs[0].map((x, i) => Math.abs(x - xs[1][i]))
    return { drift: Math.max(...drift) }
  })
  if (perfCols.error) failures.push(`perf tables: ${perfCols.error}`)
  else if (perfCols.drift > 1) failures.push(`perf table columns misaligned by ${perfCols.drift}px`)
  console.log(`perfTableColumnDrift=${perfCols.drift ?? 'n/a'}px`)


  console.log(
    `logo=${logoOk} wordmark=${wordmarkOk} sampleTokens=${tokens} paneΔ=${align.paneTop}/${align.paneBottom}px ` +
      `codeΔ=${align.codeTop}/${align.codeBottom}px sameType=${align.sameType}`,
  )
  await page.close()
}

// CJK line-breaking. Folding between two Chinese or Japanese characters is
// ordinary typesetting, not a defect — that is how CJK is set in books — so
// there are only two rules worth asserting:
//
//   - kinsoku: no line begins with closing punctuation. `line-break: strict`
//     does this natively; this checks it is actually in effect.
//   - the space around embedded Latin is typographic, not lexical, so
//     "ASCII 路径" and "特殊 token 表" must not fold. That one is ours.
//
// The second is checked structurally rather than by looking at rendered
// lines: a plain space at a CJK↔Latin boundary is a break opportunity whether
// or not this viewport happens to land on it. A rendered-line-only check
// missed a reported defect across 126 width×locale combinations here, because
// the reporter's Safari resolves different CJK fonts than headless does.
{
  const FORBIDDEN = '。、，．：；！？」』）］｝〕〉》”’·・…—～'
  const PROSE = '.abstract, .lede, .caption, h1, h2, h3, .prose'
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  await page.goto(URL_, { waitUntil: 'networkidle', timeout: 120_000 })
  await page.waitForTimeout(1000)

  for (const [label, name] of [
    ['zh', '中文'],
    ['ja', '日本語'],
  ]) {
    await page.getByRole('button', { name, exact: true }).click()
    await page.waitForTimeout(500)

    const kinsoku = await page.evaluate(
      ({ forbidden, prose }) => {
        const bad = []
        for (const el of document.querySelectorAll(prose)) {
          const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
          const nodes = []
          while (walker.nextNode()) nodes.push(walker.currentNode)
          for (const node of nodes) {
            const text = node.textContent
            const range = document.createRange()
            let prevTop = null
            for (let i = 0; i < text.length; i++) {
              range.setStart(node, i)
              range.setEnd(node, i + 1)
              const rect = range.getBoundingClientRect()
              if (!rect.width && !rect.height) continue
              if (prevTop !== null && Math.abs(rect.top - prevTop) > 2 && forbidden.includes(text[i]))
                bad.push(`"${text[i]}" …${text.slice(Math.max(0, i - 8), i)} ⏎ ${text.slice(i, i + 8)}…`)
              prevTop = rect.top
            }
          }
        }
        return bad
      },
      { forbidden: FORBIDDEN, prose: PROSE },
    )
    if (kinsoku.length)
      failures.push(`${label}: ${kinsoku.length} line(s) start with closing punctuation — ${kinsoku[0]}`)

    const glue = await page.evaluate((prose) => {
      const CJK = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f]/
      const LATIN = /[0-9A-Za-z]/
      const bad = []
      let glued = 0
      for (const el of document.querySelectorAll(prose)) {
        const text = el.textContent
        for (let i = 1; i < text.length - 1; i++) {
          if (text[i] === '\u00a0') glued++
          if (text[i] !== ' ') continue
          const a = text[i - 1]
          const b = text[i + 1]
          if ((CJK.test(a) && LATIN.test(b)) || (LATIN.test(a) && CJK.test(b)))
            bad.push(`breakable space in "${text.slice(Math.max(0, i - 6), i + 7)}"`)
        }
      }
      return { glued, bad }
    }, PROSE)
    if (glue.bad.length)
      failures.push(`${label}: ${glue.bad.length} splittable CJK↔Latin term(s) — ${glue.bad[0]}`)

    // The markup should be plain text. The <wbr>-per-phrase experiment this
    // replaced shredded each sentence into dozens of nodes and split "UTF-8".
    const shredded = await page.evaluate((prose) => {
      let worst = 0
      for (const el of document.querySelectorAll(prose)) {
        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
        let n = 0
        while (walker.nextNode()) n++
        worst = Math.max(worst, n)
      }
      return { worst, wbr: document.querySelectorAll(`${prose.split(',').join(' wbr,')} wbr`).length }
    }, PROSE)
    if (shredded.wbr) failures.push(`${label}: ${shredded.wbr} <wbr> in prose — text should be plain`)

    console.log(
      `${label} kinsoku=${kinsoku.length} splittableTerms=${glue.bad.length} ` +
        `glued=${glue.glued} maxTextNodes=${shredded.worst} wbr=${shredded.wbr}`,
    )
  }
  await page.close()
}

// the page is a light design: a dark UA preference must not invert it
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' })
  await page.goto(URL_, { waitUntil: 'networkidle', timeout: 120_000 })
  await page.waitForTimeout(800)
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
  const [r, g, b] = bg.match(/\d+/g).map(Number)
  if ((r + g + b) / 3 < 200) failures.push(`body background is dark under a dark preference: ${bg}`)
  console.log(`dark-preference body background=${bg}`)
  await page.close()
}

await browser.close()

if (failures.length) {
  console.error('\n✗ ' + failures.join('\n✗ '))
  process.exit(1)
}
console.log('\n✓ all checks passed')
