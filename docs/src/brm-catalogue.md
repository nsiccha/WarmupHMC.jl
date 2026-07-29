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

## Against DynamicHMC's standard warmup

The comparison below runs DynamicHMC's default `mcmc_with_warmup`—its
Stan-style 1,000-transition warmup—on the same inventory-generated densities.
It also reruns WarmupHMC in the same process. Every arm is measured through one
external `count_and_time` wrapper around the generated BRM density; for
adaptive centering, the reparametrizer is built *over* that counted inner
density so its WarmupHMC hooks remain active.

This is a defaults comparison, not a fixed-budget ablation: each sampler
chooses its own warmup budget. Four arms make the attribution explicit:
WarmupHMC and DynamicHMC on the identical generated non-centered target,
WarmupHMC with adaptive centering, and DynamicHMC on BRM's separately generated
centered target.

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
isempty(brmc_failures(d)) ||
    error("generated BRM standard-warmup artifact contains failed rows")
Set(r["arm"] for r in rows) == expected_arms ||
    error("generated BRM standard-warmup artifact has an unexpected arm set")
all(r -> r["n_draws_actual"] >= d["config"]["n_draws"], rows) ||
    error("a standard-warmup arm returned fewer draws than requested")
d["config"]["brm_sha"] == d0["config"]["brm_sha"] &&
d["config"]["translations_sha256"] == d0["config"]["translations_sha256"] &&
d["config"]["model_matrix_sha256"] == d0["config"]["model_matrix_sha256"] ||
    error("standard-warmup and nonlinear artifacts do not use the same BRM inventory")
Dict(m["spec"] => m["current_brm_body_sha256"] for m in brmc_models(d)) ==
Dict(m["spec"] => m["current_brm_body_sha256"] for m in brmc_models(d0)) ||
    error("standard-warmup and nonlinear artifacts do not use the same generated bodies")
Markdown.parse(
    "**Checked standard-warmup artifact:** $(length(brmc_models(d))) generated " *
    "models × $(length(expected_arms)) arms × $(d["config"]["n_seeds"]) seeds = " *
    "$(length(rows))/$(expected) successful rows; DynamicHMC " *
    "$(d["config"]["dynamichmc_version"]).")
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
        m["spec"], brmc_standard_arm_label(arm), string(round(Int, x.draws)),
        string(round(Int, x.grad)), Printf.@sprintf("%.1f", x.ess),
        Printf.@sprintf("%.2f", 1000x.per_grad),
        Printf.@sprintf("%.3f s", x.wall), Printf.@sprintf("%.1f", x.per_s),
        string(x.ndiv),
    ])
end
md_table(
    ["model", "default-warmup arm", "draws", "gradients", "min shared ESS",
     "min ESS / 1k gradients", "wall time", "min ESS / second", "divergences"],
    table_rows,
)
```

The ratios below are paired by seed before taking the median. Above one is
better for ESS/gradient and ESS/second; below one is faster for wall time. The
first comparison isolates the warmup algorithm on the identical non-centered
target. The other two show what adaptive centering costs relative to each
standard static parameterization.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_inventory_standard/rows.json")
pairs = [
    ("warmuphmc_noncentered", "dynamichmc_noncentered"),
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
