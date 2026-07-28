// .vitepress/theme/index.ts
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import 'virtual:mathjax-styles.css';

import { 
  NolebaseEnhancedReadabilitiesMenu, 
  NolebaseEnhancedReadabilitiesScreenMenu, 
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
import AuthorBadge from '@/AuthorBadge.vue'
import Authors from '@/Authors.vue'
import Banner from '@/Banner.vue'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

// Synced from HTMXObjects/assets/vitepress/htmxo-embed.ts by
// `HTMXObjects.vitepress_theme_install` in make.jl. Don't edit in place
// — edit upstream and re-run make.jl.
import { setupHtmxoEmbed } from './htmxo-embed'

// Vega-Lite figures emitted by `vega_figure` in docs/tables.jl. Written for
// this repo — not synced from anywhere, so edit it in place.
import { setupVegaFigures } from './vega-figure'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css' // You could setup your own, or else a default will be copied.
import './docstrings.css' // You could setup your own, or else a default will be copied.

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(Banner),
      'nav-bar-content-after': () => [
        h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
      ],
      // A enhanced readabilities menu for narrower screens (usually smaller than iPad Mini)
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
    app.component('AuthorBadge', AuthorBadge)
    app.component('Authors', Authors)
    // HTMXObjects embed wiring: data-hx-base resolution + SPA route
    // re-process + .htmxo-embed link rewriting. WHMC defaults the proxy
    // prefix to `/live-whmc` (matches the Vite proxy in config.mts and
    // the committed recordings under public/live-whmc/).
    setupHtmxoEmbed(router, { proxyPrefix: '/live-whmc' });
    // Vega-Lite figure wiring: swaps each ```vega-lite fence for a rendered
    // chart. Must be CALLED, not merely imported — Rollup tree-shakes an
    // unused import, and the build stays green while the figure silently
    // stays a JSON code block. Checked by grepping the built theme chunk for
    // the selector string, not by the build exiting 0.
    setupVegaFigures(router);
  }
}
export default Theme