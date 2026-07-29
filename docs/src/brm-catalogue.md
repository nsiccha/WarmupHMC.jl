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

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
d = load_results("brm_inventory_generated/rows.json")
md_table(
    ["inventory row", "historical formula", "data", "n", "parameters",
     "probe evidence", "data adapter"],
    [[m["spec"], "`" * m["historical_formula"] * "`", m["dataset"],
      m["n_obs"], m["dim_noncentered"], m["probe_evidence_kind"],
      m["data_adapter"]] for m in brmc_models(d)],
)
```

The non-centered and centered programs have equal unconstrained dimension on
every row; the docs build asserts that invariant below. Their reported
constrained-name sets can differ, so ESS is reduced only over names both
programs report. That shared set includes the random effects themselves while
excluding parameterization-specific scaffolding. Structurally constant
coordinates are dropped and counted before taking the minimum.

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
        m["spec"], brmc_arm_label(arm), adapt ? "on" : "off",
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
        m["spec"], brmc_arm_label(arm), adapt ? "on" : "off",
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

## Scope and provenance

This is the first real-data consumer receipt for BRM's generated historical
inventory bodies. It is intentionally a focused benchmark of cheap, ready,
verbatim hierarchical rows—not a claim that every historical card has an
executable translation or a generic real-data loader.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
d = load_results("brm_inventory_generated/rows.json")
c = d["config"]
Markdown.parse(
    "```\n" * c["reproduction"] * "\n```\n\n" *
    "* WarmupHMC `" * c["warmuphmc_sha"][1:10] * "`\n" *
    "* BayesianRegressionModels `" * c["brm_sha"][1:10] * "` (unregistered)\n" *
    "* StanBlocks `" * c["stanblocks_sha"][1:10] * "` (unregistered)\n" *
    "* inventory translations `" * c["translations_sha256"][1:12] * "…`\n" *
    "* inventory model matrix `" * c["model_matrix_sha256"][1:12] * "…`\n" *
    "* host `" * c["host"] * "`, Julia " * c["julia"] * ", BLAS threads " *
      string(c["blas_threads"]) * "\n" *
    "* untimed preflight draws per arm/flag: " *
      string(c["timing_preflight_draws"]) * "\n")
```

The documentation build imports neither BRM nor StanBlocks. It reads the
checked-in JSON and computes every table from the raw rows; missing fields,
failed controls, failed runs, stale translation tiers, or a row-count mismatch
make the build fail instead of rendering a reassuring partial table.
