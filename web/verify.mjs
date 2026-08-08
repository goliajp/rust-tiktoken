#!/usr/bin/env node
// Render the site in a real browser and assert the things a build cannot:
// no horizontal overflow at three viewports, no console/page errors, the
// GOLIA mark actually loads, the light treatment survives a dark colour-scheme
// preference, and the wasm playground returns tokens for real input.
//
// This is not decoration — the mobile-overflow check caught a live bug (grid
// items default to min-width:auto, so a long package name in a code block
// header widened the page).
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
  // type metrics; the install columns must line up the same way. Both drifted
  // once already when copy of different lengths sat above them.
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

  console.log(
    `logo=${logoOk} wordmark=${wordmarkOk} sampleTokens=${tokens} paneΔ=${align.paneTop}/${align.paneBottom}px ` +
      `codeΔ=${align.codeTop}/${align.codeBottom}px sameType=${align.sameType}`,
  )
  await page.close()
}

// CJK line-breaking, two rules the browser will not enforce on its own:
//   - no line may begin with closing punctuation (kinsoku)
//   - no line may break inside a word — Chinese and Japanese have no spaces,
//     so the default breaks anywhere: 各厂商发 / 布的
{
  const FORBIDDEN = '。、，．：；！？」』）］｝〕〉》”’·・…—～'
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  await page.goto(URL_, { waitUntil: 'networkidle', timeout: 120_000 })
  await page.waitForTimeout(1000)

  for (const [label, name] of [
    ['zh', '中文'],
    ['ja', '日本語'],
  ]) {
    await page.getByRole('button', { name, exact: true }).click()
    await page.waitForTimeout(500)
    const bad = await page.evaluate(
      ({ forbidden, locale }) => {
        // Intl.Segmenter is the same segmentation the page itself uses to place
        // its <wbr> hints, so "inside a word" means the same thing to both.
        const seg = new Intl.Segmenter(locale, { granularity: 'word' })
        const kinsoku = []
        const midWord = []
        for (const el of document.querySelectorAll(
          '.abstract, .lede, .caption, .claim p, h1, h2, .prose',
        )) {
          const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
          const nodes = []
          while (walker.nextNode()) nodes.push(walker.currentNode)
          for (const node of nodes) {
            const text = node.textContent
            const inside = new Set()
            for (const s of seg.segment(text)) {
              if (!s.isWordLike) continue
              for (let k = s.index + 1; k < s.index + s.segment.length; k++) inside.add(k)
            }
            const range = document.createRange()
            let prevTop = null
            for (let i = 0; i < text.length; i++) {
              range.setStart(node, i)
              range.setEnd(node, i + 1)
              const rect = range.getBoundingClientRect()
              if (!rect.width && !rect.height) continue
              const broke = prevTop !== null && Math.abs(rect.top - prevTop) > 2
              if (broke) {
                const ctx = `…${text.slice(Math.max(0, i - 8), i)} ⏎ ${text.slice(i, i + 8)}…`
                if (forbidden.includes(text[i])) kinsoku.push(`"${text[i]}" ${ctx}`)
                else if (inside.has(i)) midWord.push(ctx)
              }
              prevTop = rect.top
            }
          }
        }
        return { kinsoku, midWord }
      },
      { forbidden: FORBIDDEN, locale: label === 'zh' ? 'zh-Hans' : 'ja' },
    )
    if (bad.kinsoku.length)
      failures.push(`${label}: ${bad.kinsoku.length} line(s) start with closing punctuation — ${bad.kinsoku[0]}`)
    if (bad.midWord.length)
      failures.push(`${label}: ${bad.midWord.length} mid-word line break(s) — ${bad.midWord[0]}`)
    // Rendered-line checks only catch a bad break opportunity if some width
    // happens to land on it — the reported 、-at-line-start was invisible
    // across 126 width×locale combinations here, because the reporter's Safari
    // has different font metrics. So also check the opportunities themselves,
    // which is deterministic and font-independent: a <wbr> immediately before
    // a character that cannot start a line is a defect whether or not this
    // viewport reveals it.
    const wbr = await page.evaluate(
      ({ noStart, noEnd }) => {
        const bad = []
        let total = 0
        const edgeText = (node, dir) => {
          let n = dir > 0 ? node.nextSibling : node.previousSibling
          while (n && n.nodeType !== 3) n = dir > 0 ? (n.firstChild ?? n.nextSibling) : (n.lastChild ?? n.previousSibling)
          const t = n?.textContent
          return dir > 0 ? t?.[0] : t?.at(-1)
        }
        for (const el of document.querySelectorAll('.phrase wbr')) {
          total++
          const after = edgeText(el, 1)
          const before = edgeText(el, -1)
          if (after && noStart.includes(after)) bad.push(`break offered before "${after}"`)
          else if (before && noEnd.includes(before)) bad.push(`break offered after "${before}"`)
        }
        return { total, bad }
      },
      {
        noStart: '。、，．：；！？）］｝」』〕〉》・…ー〜～々%‰℃',
        noEnd: '（［｛「『〔〈《',
      },
    )
    if (wbr.bad.length)
      failures.push(`${label}: ${wbr.bad.length} kinsoku-violating break opportunities — ${wbr.bad[0]}`)

    console.log(
      `${label} kinsoku=${bad.kinsoku.length} midWord=${bad.midWord.length} ` +
        `badBreakOpportunities=${wbr.bad.length}/${wbr.total}`,
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
