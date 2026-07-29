# BRM inventory-generated posteriors

This benchmark asks whether nonlinear centering improves sampling on real
hierarchical models that BayesianRegressionModels (BRM) generates from its
historical-model inventory.

The distinction is load-bearing: the earlier benchmark on this page used
locally written `@brm` blocks transcribed from catalogue formulas. It exercised
BRM and WarmupHMC, but it bypassed the catalogue-to-model translation path. That
output is retained only as accessor-regression evidence. Every result below
instead starts with the exact executable body stored in BRM's
`research/historical_model_inventory/translations.tsv`.

The matrix now spans a deliberate coverage tranche rather than only the cheapest
rows: crossed high-dimensional hierarchy, correlated random slopes, Poisson and
Bernoulli and binomial GLMMs, a known-standard-error meta-analysis, and a
slope-only hierarchy with no group intercept. Rows were selected for the
structures they exercise. Runtime, block width, crossed structure, and known
upstream quirks are recorded as annotations, not used as silent exclusion
filters, so a row that is expensive or awkward is present and labelled rather
than missing and unexplained.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_inventory_generated/rows.json")
models = brmc_models(d)
rows = brmc_rows(d)
expected = length(models) * 3 * 2 * d["config"]["n_seeds"]
length(rows) == expected ||
    error("generated BRM artifact has $(length(rows)) rows; expected $expected")
isempty(brmc_failures(d)) ||
    error("generated BRM artifact contains $(length(brmc_failures(d))) failed rows")
brmc_controls_clean(d) ||
    error("generated BRM bare-arm controls are not seedwise identical")
all(m -> m["translation_status"] == "ready" &&
         m["surface_support_class"] == "already-expressible-verbatim" &&
         m["capability_tier"] == "bridgestan-finite-density-gradient", models) ||
    error("generated BRM artifact contains a row without the advertised inventory receipts")
get(d["config"], "timing_preflight_draws", 0) > 0 ||
    error("generated BRM artifact lacks the untimed timing preflight")
Markdown.parse(
    "**Checked artifact:** $(length(models)) generated real-data models, " *
    "$(d["config"]["n_seeds"]) seeds, $(d["config"]["n_draws"]) retained draws, " *
    "$(length(rows))/$(expected) successful rows. The generated adaptive path " *
    "has $(sum(r["n_divergent"] for r in rows if r["arm"] == "adaptive_centering")) " *
    "divergences.")
```

## What BRM generates

For each selected inventory row the runner:

1. reads the `inferred-family` body from `translations.tsv`;
2. requires `ready` and `already-expressible-verbatim`;
3. checks the same body and its finite BridgeStan-gradient receipt in
   `model_matrix.tsv`;
4. evaluates that string through BRM's `_brm` generator and lowers it with
   `SBBRMI`; and
5. asks BRM for the default non-centered model, static-centered model, and
   `adaptive_centering_problem`.

No model formula is copied into the runner. BRM does not yet provide a generic
real-data catalogue loader, so the consumer still supplies a small, recorded
column adapter: continuous columns become `Float64`, and grouping labels are
densely recoded in sorted order. The raw responses are not rescaled.

Each adapter also pins the SHA-256 of the upstream file it read. The pin is on
the raw bytes rather than on the URL, because several receipts are OSF, figshare
and `ndownloader` links that redirect, so a URL alone cannot tell a rerun that it
silently picked up different data.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
d = load_results("brm_inventory_standard/rows.json")
md_table(
    ["inventory row", "source fidelity", "historical formula", "data", "n",
     "parameters", "probe evidence", "data adapter"],
    [[brmc_spec_cell(m["spec"]), m["source_fidelity_verdict"],
      "`" * m["historical_formula"] * "`", m["dataset"],
      m["n_obs"], m["dim_noncentered"], m["probe_evidence_kind"],
      m["data_adapter"]] for m in brmc_models(d)],
)
```

## What each row is here to cover

Coverage is stated per row, next to the structure that justifies it. The
grouping-factor level counts are counted on the adapted data actually sampled,
and the block widths are parsed from the generated body, so neither is a claim
about the upstream dataset that this benchmark did not check.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
d = load_results("brm_inventory_standard/rows.json")
models = brmc_models(d)
missing_coverage = [m["spec"] for m in models if isempty(get(m, "coverage", ""))]
isempty(missing_coverage) ||
    error("generated BRM models without a recorded coverage rationale: " *
          join(missing_coverage, ", "))
md_table(
    ["inventory row", "inferred family", "grouping factors (levels)",
     "random-effect blocks", "n", "unconstrained dimension", "why this row"],
    [[brmc_spec_cell(m["spec"]), m["inferred_family"], brmc_group_summary(m),
      brmc_block_summary(m), m["n_obs"], m["dim_noncentered"], m["coverage"]]
     for m in models],
)
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
load_harness("brm_plots.jl")
import Markdown
d = load_results("brm_inventory_standard/rows.json")
models = brmc_models(d)
ks = [brmc_max_k(m) for m in models]
unlabelled = brmc_plot_unlabelled_specs(d)
families = sort(unique(m["inferred_family"] for m in models))
Markdown.parse(
    "**Structural span of the published matrix:** " *
    "$(length(models)) models over " *
    "$(length(unique(m["dataset"] for m in models))) real datasets; inferred " *
    "response families " * join("`" .* families .* "`", ", ") * "; " *
    "$(count(m -> length(get(m, "grouping_factors", String[])) > 1, models)) " *
    "row(s) with more than one grouping factor; widest random-effect block " *
    "K=$(maximum(ks)); observation counts " *
    "$(minimum(m["n_obs"] for m in models))–$(maximum(m["n_obs"] for m in models)); " *
    "unconstrained dimensions " *
    "$(minimum(m["dim_noncentered"] for m in models))–" *
    "$(maximum(m["dim_noncentered"] for m in models))." *
    (isempty(unlabelled) ? "" :
     " Figures below render these rows by raw inventory key because they have " *
     "no short axis label yet: " *
     join("`" .* unlabelled .* "`", ", ") * "."))
```

### Where the generated block is narrower than the historical one

`lme4` and `brms` read `(x | g)` as a correlated random intercept *and* slope.
BRM's verbatim surface takes the terms as written, so the same text generates one
random coefficient per group and no correlation. The published dimensions are the
proof rather than the assumption: `loc ~ Days + (Days | Subject)` over 18 subjects
lowers to 21 unconstrained coordinates — one fixed slope, one `log(sigma)`, 18
subject coefficients, one group scale.

This is a real fidelity gap in the row, not a bug in the measurement, and it
predates the coverage tranche: it affects rows that were already published here.
It is reported per row because it changes what "random slopes" means on this page.
A block written `(1 + x | g)` is unaffected — it is width two in both readings,
and it is why the widest block below is two rather than one.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_inventory_standard/rows.json")
dep = brmc_block_width_departures(d)
isempty(dep) ? Markdown.parse(
    "Every generated random-effect block has the same width as the historical " *
    "formula's."
) : md_table(
    ["inventory row", "departure", "detail"],
    [[brmc_spec_cell(x.spec), x.kind, x.detail] for x in dep],
)
```

### Where an adapter departs from the historical coding

BRM's verbatim surface takes one column per model term, so a multi-level factor
predictor cannot yet be expanded into contrast columns inside the generated body.
Where the historical fit used a factor, the adapter therefore supplies a single
monotone integer column and the row below says so. This changes what the
coefficient means relative to the historical publication; it does not change what
the sampler is being asked to do, which is what this benchmark measures.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_inventory_standard/rows.json")
dep = brmc_categorical_departures(d)
isempty(dep) ? Markdown.parse(
    "No adapter in this matrix departs from its historical categorical coding."
) : md_table(
    ["inventory row", "departure from the historical coding"],
    [[brmc_spec_cell(m["spec"]), m["categorical_departures"]] for m in dep],
)
```

The non-centered and centered programs have equal unconstrained dimension on
every row; the docs build asserts that invariant below. Their reported
constrained-name sets can differ, so ESS is reduced only over names both
programs report. That shared set includes the random effects themselves while
excluding parameterization-specific scaffolding. Structurally constant
coordinates are dropped and counted before taking the minimum.

Dropping them is necessary — a coordinate that never moves has no effective
sample size, and keeping it would drive every headline ESS to zero — but it does
narrow what the published minimum is a minimum over, so the size of that
narrowing is shown rather than promised.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown, Printf
d = load_results("brm_inventory_standard/rows.json")
dropped = brmc_dropped_coordinates(d)
isempty(dropped) ? Markdown.parse(
    "No row dropped a constrained coordinate: every shared name contributed to " *
    "the reported minimum."
) : md_table(
    ["inventory row", "shared constrained names", "most dropped in any cell",
     "share", "cells with a drop"],
    [[brmc_spec_cell(x.spec), string(x.shared), string(x.max_dropped),
      Printf.@sprintf("%.1f%%", 100x.share), "$(x.cells)/$(x.n_cells)"]
     for x in dropped],
)
```

## Gradient sampling efficiency

The bare non-centered and centered rows are negative controls:
`nonlinear_adapt=true` must do nothing because those problems carry no
reparametrizer. The table therefore shows each bare arm once. The adaptive arm
appears with the flag off and on.

Each entry is the median across seeds. “min ESS / 1k gradients” is computed per
seed before taking the median; it is not a ratio of two independently aggregated
columns.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_inventory_generated/rows.json")
bad_dim = [m["spec"] for m in brmc_models(d)
           if m["dim_noncentered"] != m["dim_centered"]]
isempty(bad_dim) ||
    error("generated centered/non-centered dimensions differ: " * join(bad_dim, ", "))
s = brmc_summary(d)
shown = (("noncentered", false), ("centered", false),
         ("adaptive_centering", false), ("adaptive_centering", true))
table_rows = []
for m in brmc_models(d), (arm, adapt) in shown
    x = only(v for v in s if v.spec == m["spec"] && v.arm == arm && v.adapt == adapt)
    push!(table_rows, [
        brmc_spec_cell(m["spec"]), brmc_arm_label(arm), adapt ? "on" : "off",
        string(round(Int, x.grad)), Printf.@sprintf("%.1f", x.ess),
        Printf.@sprintf("%.2f", 1000x.per_grad), string(x.ndiv),
    ])
end
md_table(
    ["model", "generated arm", "nonlinear adapt", "gradient evaluations",
     "min shared ESS", "min ESS / 1k gradients", "divergences"],
    table_rows,
)
```

## Runtime sampling efficiency

The wall timer surrounds `adaptive_warmup_mcmc` only; model generation,
BridgeStan compilation, and a dedicated preflight of every arm/flag path happen
outside it. Recorded flag order alternates by seed, so one flag does not
systematically run earlier on the shared host.

“min ESS / second” is again computed per seed and then aggregated. It measures
end-to-end sampling throughput for this exact host and package stack. Unlike
gradient counts, it is not portable across machines.

These are the timings from BRM's corrected bit-exact adaptive-centering
accessor. A superseded generated run had a last-bit arithmetic drift that made
the wrapper appear much faster; it is retained only as rejected regression
evidence and is excluded from every table on this page.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_inventory_generated/rows.json")
s = brmc_summary(d)
shown = (("noncentered", false), ("centered", false),
         ("adaptive_centering", false), ("adaptive_centering", true))
table_rows = []
for m in brmc_models(d), (arm, adapt) in shown
    x = only(v for v in s if v.spec == m["spec"] && v.arm == arm && v.adapt == adapt)
    push!(table_rows, [
        brmc_spec_cell(m["spec"]), brmc_arm_label(arm), adapt ? "on" : "off",
        Printf.@sprintf("%.3f s", x.wall), Printf.@sprintf("%.1f", x.per_s),
    ])
end
md_table(
    ["model", "generated arm", "nonlinear adapt",
     "sampling wall time", "min shared ESS / second"],
    table_rows,
)
```

## Paired effect of turning adaptation on

This is the direct A/B on the only generated arm that carries a
reparametrizer. Every ratio is paired by model and seed before taking the
median. Ratios above one are better for the two efficiency columns; a wall-time
ratio below one is faster.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_inventory_generated/rows.json")
p = brmc_paired_efficiency(d)
md_table(
    ["model", "seeds", "trajectories changed", "grad eval ratio on/off",
     "ESS/grad ratio on/off", "wall ratio on/off", "ESS/s ratio on/off"],
    [[x.spec, x.n, "$(x.changed)/$(x.n)",
      Printf.@sprintf("%.2f×", x.grad_ratio),
      Printf.@sprintf("%.2f×", x.ess_per_grad_ratio),
      Printf.@sprintf("%.2f×", x.wall_ratio),
      Printf.@sprintf("%.2f×", x.ess_per_s_ratio)] for x in p],
)
```

Do not read a trajectory that stayed unchanged as a failed adaptation. It means
the window scorer kept the generated non-centered endpoint. Conversely, a
changed trajectory is not automatically an improvement; the two efficiency
ratios report whether it paid off.

## Against DynamicHMC's standard warmup

The comparison below runs DynamicHMC's default `mcmc_with_warmup`—its
Stan-style 1,000-transition warmup—on the same inventory-generated densities.
It also reruns WarmupHMC in the same process. Every arm is measured through one
external `count_and_time` wrapper around the generated BRM density; for
adaptive centering, the reparametrizer is built *over* that counted inner
density so its WarmupHMC hooks remain active.

This is a defaults comparison, not a fixed-budget ablation: each sampler
chooses its own warmup budget. Six arms make the attribution explicit:
WarmupHMC and DynamicHMC each run BRM's generated non-centered and centered
targets; WarmupHMC also runs BRM's nonlinear wrapper both fixed at its generated
`c=0` endpoint and with centering adaptation enabled. The fixed-wrapper arm
separates wrapper cost from the effect of fitting nonlinear centerings, while
the two bare WarmupHMC arms expose its ordinary linear-transformation adaptation.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d0 = load_results("brm_inventory_generated/rows.json")
d = load_results("brm_inventory_standard/rows.json")
rows = brmc_rows(d)
expected_arms = Set(brmc_standard_arm_order())
expected = length(brmc_models(d)) * length(expected_arms) * d["config"]["n_seeds"]
d["config"]["mode"] == "standard" ||
    error("generated BRM standard-warmup artifact has the wrong mode")
length(rows) == expected ||
    error("generated BRM standard-warmup artifact has $(length(rows)) rows; expected $expected")
failures = brmc_failures(d)
all(r -> !isempty(r["error"]), failures) ||
    error("generated BRM standard-warmup artifact has an unlabelled failed row")
Set(r["arm"] for r in rows) == expected_arms ||
    error("generated BRM standard-warmup artifact has an unexpected arm set")
all(r -> r["n_draws_actual"] >= d["config"]["n_draws"], rows) ||
    error("a standard-warmup arm returned fewer draws than requested")
d["config"]["translations_sha256"] == d0["config"]["translations_sha256"] &&
d["config"]["model_matrix_sha256"] == d0["config"]["model_matrix_sha256"] ||
    error("standard-warmup and nonlinear artifacts do not use the same BRM inventory")
standard_bodies = Dict(m["spec"] => m["current_brm_body_sha256"]
                       for m in brmc_models(d))
focused_bodies = Dict(m["spec"] => m["current_brm_body_sha256"]
                      for m in brmc_models(d0))
all(pair -> get(standard_bodies, first(pair), nothing) == last(pair),
    focused_bodies) ||
    error("the expanded standard artifact changed a focused generated body")
Markdown.parse(
    "**Checked standard-warmup artifact:** $(length(brmc_models(d))) generated " *
    "models × $(length(expected_arms)) arms × $(d["config"]["n_seeds"]) seeds = " *
    "$(length(rows) - length(failures))/$(expected) successful rows and " *
    "$(length(failures)) explicitly reported statistical/runtime failure(s); DynamicHMC " *
    "$(d["config"]["dynamichmc_version"]).")
```

Failed trajectories remain part of the experimental design and the divergence
total, but do not contribute invented zeroes to ESS medians or plot marks.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_inventory_standard/rows.json")
bad = brmc_failures(d)
isempty(bad) ? Markdown.parse("No failed trajectories were recorded.") : md_table(
    ["model", "arm", "seed", "divergences", "diagnostic"],
    [[brmc_spec_cell(r["spec"]), brmc_standard_arm_label(r["arm"]), r["seed"],
      r["n_divergent"], r["error"]] for r in bad],
)
```

Both tables use minimum constrained-space ESS over the parameter names shared
by BRM's generated centered and non-centered programs. Medians are across 12
seeds; divergences are totals over those seed trajectories.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_inventory_standard/rows.json")
s = brmc_standard_summary(d)
table_rows = []
for m in brmc_models(d), arm in brmc_standard_arm_order()
    x = only(v for v in s if v.spec == m["spec"] && v.arm == arm)
    push!(table_rows, [
        brmc_spec_cell(m["spec"]), brmc_standard_arm_label(arm), "$(x.n)/$(x.n_total)",
        string(round(Int, x.draws)),
        string(round(Int, x.grad)), Printf.@sprintf("%.1f", x.ess),
        Printf.@sprintf("%.2f", 1000x.per_grad),
        Printf.@sprintf("%.3f s", x.wall), Printf.@sprintf("%.1f", x.per_s),
        string(x.ndiv),
    ])
end
md_table(
    ["model", "default-warmup arm", "successful seeds", "draws", "gradients", "min shared ESS",
     "min ESS / 1k gradients", "wall time", "min ESS / second", "divergences"],
    table_rows,
)
```

The table compresses each arm to a median. The figures below are actual
AlgebraOfVega layers over every seed row. HTMXObjects' public `SemanticPlot`
provides each layer's accessible text summary; AoV's public Vega-Lite MIME
projection supplies the embedded figure. There is no hand-authored Vega spec
and no second stored summary.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
load_harness("brm_plots.jl")
d = load_results("brm_inventory_standard/rows.json")
aov_figure(brmc_gradient_efficiency_plot(d);
    caption="Points and nested intervals summarise the 12 seed-level minimum " *
            "shared constrained-space ESS values per 1,000 gradients. One band " *
            "row per model, six arms dodged within it; the metric runs along " *
            "the horizontal axis, which is logarithmic.")
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
load_harness("brm_plots.jl")
d = load_results("brm_inventory_standard/rows.json")
aov_figure(brmc_runtime_efficiency_plot(d);
    caption="The same seed-level runs expressed as minimum shared constrained-" *
            "space ESS per measured sampling second. Same layout; the " *
            "horizontal axis is logarithmic.")
```

The ratios below are paired by seed before taking the median. Above one is
better for ESS/gradient and ESS/second; below one is faster for wall time. The
first two comparisons isolate the warmup algorithm on identical generated
targets. The fixed-wrapper comparison measures wrapper overhead at unchanged
geometry; the fitted/fixed comparison isolates nonlinear adaptation; the final
two compare the fitted nonlinear path with DynamicHMC's static endpoints.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_inventory_standard/rows.json")
pairs = [
    ("warmuphmc_noncentered", "dynamichmc_noncentered"),
    ("warmuphmc_centered", "dynamichmc_centered"),
    ("warmuphmc_fixed_centering", "warmuphmc_noncentered"),
    ("warmuphmc_adaptive_centering", "warmuphmc_fixed_centering"),
    ("warmuphmc_adaptive_centering", "dynamichmc_noncentered"),
    ("warmuphmc_adaptive_centering", "dynamichmc_centered"),
]
table_rows = []
for (num, den) in pairs, x in brmc_standard_ratios(d, num, den)
    push!(table_rows, [
        x.spec,
        brmc_standard_arm_label(num) * " / " * brmc_standard_arm_label(den),
        string(x.n), Printf.@sprintf("%.2f×", x.ess_per_grad_ratio),
        Printf.@sprintf("%.2f×", x.wall_ratio),
        Printf.@sprintf("%.2f×", x.ess_per_s_ratio),
    ])
end
md_table(
    ["model", "numerator / denominator", "paired seeds",
     "ESS/gradient ratio", "wall ratio", "ESS/second ratio"],
    table_rows,
)
```

The external gradient counter includes each sampler's initialization and
warmup calls into the generated density. Wall time excludes model generation,
BridgeStan compilation, and one dedicated preflight per arm; arm order rotates
cyclically by seed. As elsewhere on this page, per-gradient results are the
portable comparison and wall-clock results describe this exact host.

## How wide a random-effect block the inventory can reach

The widest block in the published matrix is two coefficients sharing one
group-level covariance. That is a property of the inventory, not a rule this
benchmark applied: no row was excluded for its block width. The question was
answered by scanning every catalogue card under both width readings and then
taking each qualifying candidate as far as it would go on its real data.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_inventory_high_k/rows.json")
c = d["config"]
cands = brmc_high_k_candidates(d)
sampled = [x for x in cands if x["sampled"]]
Markdown.parse(
    "**K>2 probe:** $(c["n_cards_scanned"]) catalogue cards scanned; " *
    "$(c["n_high_k_cards"]) carry a block of three or more coefficients under at " *
    "least one reading; $(length(sampled)) reached a real sample. " *
    "Among rows that pass the publishable gate (`ready` + " *
    "`already-expressible-verbatim` + finite BridgeStan gradient) the widest " *
    "block is K=$(c["max_historical_k_among_publishable_rows"]) under the " *
    "historical reading and K=$(c["max_generated_k_among_publishable_rows"]) as " *
    "BRM actually lowers it.\n\n" *
    "*Method.* " * c["method"])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
d = load_results("brm_inventory_high_k/rows.json")
md_table(
    ["catalogue card", "historical K", "generated K",
     "furthest stage reached on real data", "what stopped it",
     "the inventory's own note"],
    [[brmc_spec_cell(c["spec"]), string(c["historical_max_k"]), string(c["generated_max_k"]),
      "`" * brmc_high_k_reached(c) * "`",
      isempty(c["blocker"]) ? "nothing" : c["blocker"],
      "`" * c["translation_status"] * "`" *
      (isempty(c["translation_note"]) ? "" : " — " * c["translation_note"])]
     for c in brmc_high_k_candidates(d)],
)
```

The one candidate with a fetchable historical dataset was carried onto that data
and stopped at a specific, reported exception rather than at a status field: it
downloaded, adapted, parsed through `BRM._brm`, and then failed to lower because
the Student-t degrees-of-freedom symbol its translation leaves open has no value.
That is the same limitation the inventory records for it, confirmed by execution
instead of quoted.

The finding is a statement about the inventory rather than about cost: the
historical gallery does contain wider blocks, and each one currently sits behind
an unresolved translation, a synthetic dataset receipt, or a surface gap — not
behind a runtime ceiling or an eligibility rule imposed here. When one becomes
`ready` and verbatim-expressible it belongs in the matrix above, and nothing in
the publishing runner would keep it out.

## Scope and provenance

The three-model nonlinear flag A/B is the first real-data consumer receipt for
BRM's generated historical inventory bodies. The standard-warmup matrix expands
that receipt across the coverage tranche listed above: multiple response
families, crossed and nested grouping structures, correlated slopes, known
observation standard errors, and a slope-only hierarchy. It remains a benchmark
of `ready`, verbatim-expressible rows with a finite BridgeStan gradient — not a
claim that every historical card has an executable translation or that BRM has a
generic real-data catalogue loader.

Two limits are worth stating plainly rather than leaving to be inferred. Row
selection is not a random sample of the inventory, so the ratios on this page
describe these structures and should not be read as an expected effect over the
gallery as a whole. And the adapters are consumer code checked in here, not BRM
exports; the inventory pins what model is run, while this repository pins what
data it is run on.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
using SHA
d = load_results("brm_inventory_standard/rows.json")
c = d["config"]
d0 = load_results("brm_inventory_generated/rows.json")
c0 = d0["config"]
standard_path = joinpath(results_dir(), "brm_inventory_standard", "rows.json")
standard_sha = bytes2hex(open(sha256, standard_path))
sampling_seconds = sum(r["wall_s"] for r in d["rows"])
Markdown.parse(
    "**Expanded standard-warmup matrix**\n\n" *
    "```\n" * c["reproduction"] * "\n```\n\n" *
    "* WarmupHMC `" * c["warmuphmc_sha"][1:10] * "`\n" *
    "* BayesianRegressionModels `" * c["brm_sha"][1:10] * "` (unregistered)\n" *
    "* StanBlocks `" * c["stanblocks_sha"][1:10] * "` (unregistered)\n" *
    "* inventory translations `" * c["translations_sha256"][1:12] * "…`\n" *
    "* inventory model matrix `" * c["model_matrix_sha256"][1:12] * "…`\n" *
    "* host `" * c["host"] * "`, Julia " * c["julia"] * ", BLAS threads " *
      string(c["blas_threads"]) * "\n" *
    "* process elapsed: " * string(c["total_elapsed_s"]) * " s; summed timed " *
      "sampling: " * string(round(sampling_seconds; digits=3)) * " s\n" *
    "* result SHA-256: `" * standard_sha * "`\n" *
    "* untimed preflight draws per arm: " *
      string(c["timing_preflight_draws"]) * "\n" *
    "* runner + adapters SHA-256: `" *
      get(c, "runner_sha256", "not recorded")[1:12] * "…`\n" *
    "* upstream data files pinned by SHA-256: " *
      string(count(m -> !isempty(get(m, "data_sha256_pinned", "")),
                   brmc_models(d))) * "/" * string(length(brmc_models(d))) * "\n\n" *
    "**Earlier focused nonlinear-flag receipt**\n\n" *
    "```\n" * c0["reproduction"] * "\n```\n\n" *
    "* WarmupHMC `" * c0["warmuphmc_sha"][1:10] * "`\n" *
    "* BayesianRegressionModels `" * c0["brm_sha"][1:10] * "` (unregistered)\n" *
    "* StanBlocks `" * c0["stanblocks_sha"][1:10] * "` (unregistered)\n" *
    "* host `" * c0["host"] * "`, Julia " * c0["julia"] *
      ", BLAS threads " * string(c0["blas_threads"]) * "\n")
```

The focused receipt predates both the runner SHA-256 pin and the coverage
tranche, so its reproduction line describes the spec list as it stood when it ran
— three models — rather than the sixteen the runner now carries. Its inventory
translation and model-matrix checksums are byte-identical to the expanded
matrix's, which is the part the two artifacts have to agree on for the comparison
above to be about the same generated bodies; the docs build asserts that equality
rather than assuming it.

The documentation build imports neither BRM nor StanBlocks. It reads the
checked-in JSON and computes every table from the raw rows; missing fields,
failed controls, failed runs, stale translation tiers, or a row-count mismatch
make the build fail instead of rendering a reassuring partial table.
