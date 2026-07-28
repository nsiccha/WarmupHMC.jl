# Gallery

The WarmupHMC dashboard runs every PosteriorDB model through four
samplers (WarmupHMC, DynamicHMC, AdvancedHMC, plus an optional
reparametrised pass) and reports compile / sample / ESS / time per
posterior. The views below are the live dashboard during development; in the
deployed docs they are a **static recording made during the docs build**. The
recording is never committed — `*.html` is ignored, so nothing under
`docs/src/public/live-whmc/` is tracked, and CI regenerates it on every build.

## Runnable examples

The dashboard below is a recording of runs, not something you can step through.
The examples that you *can* run are, smallest first:

| Example | Where | Executed at build time? |
| --- | --- | --- |
| Sample a 3-D Gaussian, read the draws back | [Quickstart](@ref) | **yes** |
| Checkpoint a run and resume it | [Checkpoints, Callbacks and Resume](@ref) | **yes** |
| Fit a nonlinear reparametrization on Neal's funnel | [A complete worked example](@ref) | no — needs an AD backend the docs environment does not carry |

"Executed at build time" means the outputs on those pages are what the code
returned during this build, and that a change breaking the example fails the
build rather than leaving a page that reads correctly and is wrong. The funnel
example is complete and self-contained but is not run here; see
[A note on the code blocks](@ref).

## Overview table

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-whmc/" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading WHMC overview…</em>
</div>
</div>
```

## Card grid

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-whmc/gallery" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading WHMC gallery…</em>
</div>
</div>
```

## What the recording does and does not include

The recorder writes full-page and HX-shape variants of exactly two routes — `/`
and `/gallery` — into `docs/src/public/live-whmc/`. That is the whole export.

The live app serves more than that: per-posterior routes under
`/posteriors/<name>`, plus `/benchmarks` and `/benchmark/<key>`. **None of those
are recorded**, and the export deliberately removes the links that would lead to
them, so the two views above are complete rather than a surface with dead ends
behind it. Recording is done in a static mode that also turns the per-posterior
detail panels into inert placeholders and disables the compute controls; the
overview and card surfaces are intact, but nothing in a recording computes.

The static benchmark evidence is not here at all — it is native Markdown on
[Benchmark evidence](@ref), generated from the same JSON.

## Refresh the recording

CI runs the recorder before building the docs, so a deployed page is always a
recording of the revision it was built from. To reproduce it locally, after
instantiating the web environment:

```sh
julia --project=web docs/record_gallery.jl
```

That is the supported path. It records the two docs routes, writes both shapes,
and fails loudly if any expected file is missing.

!!! warning "`GET /record_gallery` is not the docs recipe"
    The running app still exposes that route, but without the static-recording
    mode the script sets it uses the app's **full** path list — including the
    per-posterior routes, which are expensive. Use the script above for anything
    aimed at the docs.
