// Vega-Lite figure rendering for docs pages.
//
// Figures are emitted by `vega_figure` in docs/tables.jl as a fenced
// ```vega-lite block holding the spec as JSON. VitePress renders that into
//
//     <div class="language-vega-lite …"><button …><span …><pre><code>…</code></pre></div>
//
// and this module swaps a chart in for the whole wrapper.
//
// WHY THE SPEC ARRIVES IN A CODE FENCE
//
// Because it is the only carrier that survives. A generated `<div data-spec>`
// is destroyed twice over: `Markdown.parse` reads `$schema` as math and
// `ns_const` as emphasis, and DocumenterVitepress escapes `<`/`>` in every
// text node, so the div reaches the browser as `&lt;div&gt;`. Raw HTML gets in
// only via `@raw html`, which is static and cannot carry a computed value.
// Markdown does not interpret inside a fence and VitePress marks code blocks
// `v-pre`, so a fence is the one path a generated string crosses untouched.
//
// The consequence for this file: read `textContent`, not an attribute. Shiki
// splits the JSON across dozens of styled <span>s, but textContent
// reconstitutes the source exactly — that is what makes the round-trip exact
// rather than merely close.
//
// The `vega` / `vega-lite` / `vega-embed` runtimes are CDN <script> tags in
// config.mts, pinned to the same major versions AlgebraOfVega's
// `vega_cdn_urls()` returns, so a spec that renders in the web app renders
// here. They are blocking head scripts, so `vegaEmbed` is defined before any
// page script runs — but `waitForVega` still guards the cold-load case where a
// route settles while the CDN is in flight.
//
// If the runtime never arrives the fence is LEFT ALONE, still showing the
// spec. That is deliberate: the numbers are in it, so an offline reader sees
// the figure's data rather than a blank gap where a figure was promised.

declare const vegaEmbed: any

// Vega has no access to CSS custom properties, so the docs' light/dark tokens
// cannot be inherited — the palette has to be handed over explicitly and
// re-applied when the user toggles the theme.
function vegaConfig(dark: boolean) {
  const fg = dark ? '#c9d1d9' : '#3c3c43'
  const muted = dark ? '#8b949e' : '#67676c'
  const grid = dark ? 'rgba(255,255,255,0.10)' : 'rgba(60,60,67,0.12)'
  return {
    background: 'transparent',
    axis: {
      labelColor: muted, titleColor: fg,
      domainColor: grid, tickColor: grid, gridColor: grid,
    },
    legend: { labelColor: muted, titleColor: fg },
    title: { color: fg, subtitleColor: muted },
    view: { stroke: 'transparent' },
  }
}

const isDark = () => document.documentElement.classList.contains('dark')

// Every ```vega-lite block on the page. VitePress puts the info string in the
// wrapper's class; the `[class*=]` form tolerates the other classes it adds
// (vp-adaptive-theme, and a fallback-highlight class when Shiki does not know
// the language, which it does not know for vega-lite).
const blocks = () =>
  Array.from(
    document.querySelectorAll<HTMLElement>('div[class*="language-vega-lite"]'))

function renderOne(wrapper: HTMLElement) {
  const code = wrapper.querySelector('code')
  if (!code) return

  let spec: any
  try {
    spec = JSON.parse(code.textContent || '')
  } catch (err) {
    // A malformed spec is a build-side bug. Leave the fence in place — it is
    // the evidence — and say what happened above it.
    if (!wrapper.previousElementSibling?.classList.contains('vega-figure-error')) {
      const note = document.createElement('p')
      note.className = 'vega-figure-error'
      note.textContent = 'Figure unavailable: the embedded Vega-Lite spec did not parse.'
      wrapper.parentElement?.insertBefore(note, wrapper)
    }
    return
  }

  // Replace the whole wrapper — button, language label and all — with a host
  // div. Keeping the source fence around under a `display:none` would leave it
  // in the local search index and in the page's copy-all text, describing a
  // figure the reader can already see.
  const host = document.createElement('div')
  host.className = 'vega-figure'
  wrapper.replaceWith(host)
  ;(host as any).__vegaSpec = spec

  vegaEmbed(host, spec, {
    actions: false,
    renderer: 'svg',
    config: vegaConfig(isDark()),
  }).catch((err: any) => {
    host.textContent = 'Figure unavailable: ' + String(err)
  })
}

// Re-embed figures already swapped in — the palette is baked into the rendered
// SVG, so a theme toggle would otherwise leave dark-mode axes unreadable.
function rerenderExisting() {
  document.querySelectorAll<HTMLElement>('.vega-figure').forEach(host => {
    const spec = (host as any).__vegaSpec
    if (!spec) return
    vegaEmbed(host, spec, {
      actions: false,
      renderer: 'svg',
      config: vegaConfig(isDark()),
    }).catch(() => {})
  })
}

function waitForVega(attempt = 0) {
  if (typeof vegaEmbed !== 'undefined') {
    blocks().forEach(renderOne)
    return
  }
  // ~5 s of 100 ms polls. Past that the CDN is genuinely unreachable; stop, and
  // leave every fence exactly as it is (see the header — the JSON is the
  // fallback, not a placeholder to overwrite).
  if (attempt > 50) return
  setTimeout(() => waitForVega(attempt + 1), 100)
}

export function setupVegaFigures(router: any) {
  if (typeof window === 'undefined') return // SSR build: nothing to render

  const schedule = () => requestAnimationFrame(() => waitForVega())

  // Initial paint, then every SPA navigation — VitePress swaps page content
  // without reloading, so a figure on the next page has never been embedded.
  schedule()
  const onAfterRouteChange = router.onAfterRouteChange
  router.onAfterRouteChange = (to: string) => {
    onAfterRouteChange?.(to)
    schedule()
  }

  new MutationObserver(() => rerenderExisting()).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })
}
