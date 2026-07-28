# Nonlinear weighting evidence

The adaptive nonlinear path collects evidence from the leaves of each
trajectory. Two independent choices govern that: **which leaves contribute**, and
**how their contribution is scaled**. Both are exposed as keyword arguments, and
this page is the measured answer to whether the second one matters.

Every figure below is generated at docs-build time from
`docs/benchmark/results/nonlinear_weighting/rows.json`, which holds one object
per run plus the configuration that produced them — and no stored summaries or
conclusion. The summaries, the conclusion included, are computed by
`docs/benchmark/nonlinear_weighting.jl`, the same derivation the report uses; the
web app reads the same JSON through its own generic evidence renderer. Nothing on
this page is a second copy of a number, and if the rows go missing the docs build
fails rather than rendering a stale table.

## What was varied

| axis | keyword | arms measured | shipped default |
|---|---|---|---|
| which leaves contribute | `nonlinear_evidence` | `:all_good_leaves`, `:nuts_weighted` | `:linear_pool` |
| how they are scaled | `nonlinear_trajectory_weighting` | `:unit`, `:stepsize` | `:unit` |

!!! warning "Neither leaf-policy arm is the default"
    `nonlinear_evidence` ships as `:linear_pool`, and the study did not measure
    it. The leaf-policy axis therefore compares two **opt-in** modes against each
    other, not against what you get by default. Read the rows below as "given
    that you have already opted into one of these two, does the scaling matter",
    not as a verdict on the shipped configuration.

The scaling axis is different: `:unit` **is** the default, so the `unit`-vs-`stepsize`
comparison does speak directly to a choice you already have.

## Can the target tell the arms apart?

This is the gate, and it belongs before any efficiency number. If two policy arms
produce a bit-identical run on a target, then every comparison on that target is
a tie by construction — and a tie reported without this check reads as evidence
that the policies are equivalent, which it is not.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("nonlinear_weighting.jl")
nw_discrimination_table(load_results("nonlinear_weighting/rows.json")["rows"])
```

Read that column-by-column before trusting anything downstream. A target where
every arm is identical contributes ties that measure the target, not the policy.

## Efficiency per cell

Minimum-coordinate ESS per thousand gradient evaluations — the portable measure;
wall time is host-specific and is not reported here.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("nonlinear_weighting.jl")
nw_aggregate_table(load_results("nonlinear_weighting/rows.json")["rows"])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
d = load_results("nonlinear_weighting/rows.json")
Markdown.parse("*" * provenance(d; harness = "docs/benchmark/nonlinear_weighting_run.jl") *
               " $(length(d["rows"])) runs; summaries derived by " *
               "`docs/benchmark/nonlinear_weighting.jl` at build time.*")
```

## Paired comparison

Seeds are paired, so each ratio compares `stepsize` against `unit` on the *same*
seed. Above 1 favours `stepsize`. The five-number summary is here for the reason
a median alone is misleading on this data: when most seeds are exact ties, the
median is pinned to 1 by the ties and says nothing about the seeds that did move.
The last column counts how many pairs are within ±5% of parity.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("nonlinear_weighting.jl")
nw_paired_median_table(load_results("nonlinear_weighting/rows.json")["rows"])
```

## What the rows conclude

The sentence below is **assembled from the numbers** by `nw_conclusion`, not
written here. It cannot disagree with the tables above, because it is computed
from the same rows in the same build.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
load_harness("nonlinear_weighting.jl")
Markdown.parse(nw_conclusion(load_results("nonlinear_weighting/rows.json")["rows"]))
```

## Reproduce it

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
rep = get(load_results("nonlinear_weighting/rows.json")["config"], "reproduction", Dict())
bits = String[]
haskey(rep, "driver") && push!(bits, "Driver: `$(rep["driver"])`.")
haskey(rep, "n_rows") && push!(bits, "$(rep["n_rows"]) rows.")
haskey(rep, "result") && push!(bits, "Independent re-run: $(rep["result"]).")
haskey(rep, "reproduced_by") && haskey(rep, "reproduced_at_sha") &&
    push!(bits, "Reproduced by `$(rep["reproduced_by"])` at `$(first(string(rep["reproduced_at_sha"]), 7))`.")
Markdown.parse(join(bits, " "))
```

A re-run regenerates the rows and its own run-scoped configuration. It does not
regenerate the fields describing the original measurement — `warmuphmc_sha`,
`nonlinear_impl_sha`, `measured_on`, `measured_by`, `verification` — which is why
those are recorded once, in the artifact, rather than restated here.

## See also

- [Linear restart evidence](@ref) — the same treatment for the linear path.
- [Benchmark evidence](@ref) — the appendix rendering every checked-in result file.
