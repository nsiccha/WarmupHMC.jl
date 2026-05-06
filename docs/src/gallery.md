---
htmxo-embed-fullwidth: true
---

# Gallery

The WarmupHMC dashboard runs every PosteriorDB model through four
samplers (WarmupHMC, DynamicHMC, AdvancedHMC, plus an optional
reparametrised pass) and reports compile / sample / ESS / time per
posterior. The view below is the live dashboard during development; in
the deployed docs it's the most recent recording committed under
`docs/src/public/live-whmc/`.

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-whmc/gallery" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading WHMC gallery…</em>
</div>
</div>
```

## Refresh the recording

To re-record the dashboard for the docs:

```julia
# from the WHMC web server, in a browser:
GET /record_gallery               # uses defaults
GET /record_gallery?force=true    # invalidate cache and re-record
```

The recording dumps both full-page and HX-shape variants of `/`,
`/gallery`, and every `/model/<pn>` into `docs/src/public/live-whmc/`.
Commit the result and CI deploys it.
