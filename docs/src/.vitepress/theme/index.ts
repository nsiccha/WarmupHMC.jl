import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { setupVegaFigures } from './vega-figure'
import 'virtual:mathjax-styles.css'
import './style.css'
import './docstrings.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ router }) {
    setupVegaFigures(router)
  },
} satisfies Theme
