# Adaptive centering at fixed `c`

Adaptive centering chooses a per-coordinate centering `c` during warm-up. This
page measures the sampler at a **fixed** `c` instead, against a reference whose
`c` is exact, so that the adaptation and the parametrisation it is aiming at can
be judged separately.

Every figure below is generated at docs-build time from
`docs/benchmark/results/adaptive_centering_fixed_c/rows.json`, which holds one
object per run plus the configuration that produced them, and no stored
summaries. The summaries and the conclusion are computed by
`docs/benchmark/adaptive_centering_fixed_c.jl` from those rows in the same
build, so nothing on this page is a second copy of a number, and if the rows go
missing or change shape the docs build fails rather than rendering a stale
table.

## What is being compared

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
import Markdown
d = load_results("adaptive_centering_fixed_c/rows.json")
rows = ace_summary_rows(d)
fams = unique(r.family for r in rows)
arms = unique(r.arm for r in rows)
Markdown.parse(
    "**Families** (" * string(length(fams)) * "): " * join(fams, ", ") * ". " *
    "**Arms** (" * string(length(arms)) * "): " * join(arms, ", ") * ".")
```

The artifact also records what these targets are and how the arms were run.
These are the caveats that decide what the numbers below can be used for, so
they are rendered from the configuration rather than paraphrased:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
cfg = load_results("adaptive_centering_fixed_c/rows.json")["config"]
labels = ["representativeness" => "What the targets are",
          "cstar_role" => "What `c*` is",
          "efficiency_denominator" => "What the efficiency denominator counts",
          "ad_backend" => "Where the gradient comes from",
          "arm_execution" => "How the arms were run"]
bits = ["- **$(lab):** $(cfg[k])" for (k, lab) in labels if haskey(cfg, k)]
Markdown.parse(isempty(bits) ?
    "*The artifact records no scope metadata.*" : join(bits, "\n"))
```

## The exactness boundary

`c*` is exact **by construction of the target**, not because a search found it
to maximise ESS. That distinction is the whole reason this page can separate the
two questions: an arm at `c*` is sampling the parametrisation the target is
built to have, so comparing against it measures how far the other arms sit from
a known-correct answer — not which arm happened to win a tuning contest.

Read every efficiency number below in that light. A gap to `c*` is a distance
from exactness. It is not evidence that `c*` is the ESS optimum, and this page
does not claim it is.

## Does the common frame validate?

This is the gate, and it belongs before any efficiency number. The arms are only
comparable if they are being scored on a common frame; where that fails, the
rows below it measure the frame rather than the arm.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
rows = ace_validation_rows(load_results("adaptive_centering_fixed_c/rows.json"))
md_table(
    ["family", "common points", "max common-frame Δ", "max coordinate round-trip Δ",
     "max transport log-density Δ", "max gradient vs finite-difference Δ"],
    [[r.family, string(r.n_points),
      num(r.max_common_frame_abs; sig = 3),
      num(r.max_coordinate_roundtrip_abs; sig = 3),
      num(r.max_transport_logdensity_abs; sig = 3),
      num(r.max_gradient_fd_abs; sig = 3)] for r in rows])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
import Markdown
d = load_results("adaptive_centering_fixed_c/rows.json")
rows = ace_validation_rows(d)
narms = length(unique(r.arm for r in ace_summary_rows(d)))
vals = Float64[Float64(v) for r in rows
               for v in (r.max_common_frame_abs, r.max_coordinate_roundtrip_abs,
                         r.max_transport_logdensity_abs, r.max_gradient_fd_abs)
               if v !== nothing]
Markdown.parse(isempty(vals) ?
    "*The table above records no numeric deviation.*" :
    "*The largest deviation anywhere in the table above is " *
    "$(num(maximum(vals); sig = 3)), over " *
    "$(sum(r.n_points for r in rows)) common model-frame points across " *
    "$(length(rows)) families — each point checked in all $(narms) arms, so " *
    "the maxima are taken over $(narms)× that many evaluations.*")
```

The gradient column is checked against finite differences rather than against a
second AD backend. Two backends differentiate the same implementation, so they
can agree on a wrong derivative; only a finite-difference reference can falsify
one.

## Efficiency per family and arm

Minimum-coordinate ESS per thousand gradient evaluations — the portable measure.
Wall time is host-specific and is not reported here. The per-chain view comes
first because it carries the spread; the pooled view follows.

Each median carries the number of chains it was taken over, because that count
is not always the same in both columns — a chain that diverges catastrophically
can return a non-finite tail ESS, and dropping it from one median while keeping
it in the other would otherwise be invisible.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
rows = ace_summary_rows(load_results("adaptive_centering_fixed_c/rows.json"))
md_table(
    ["family", "arm", "bulk ESS/1k grad (median)", "range", "chains",
     "tail ESS/1k grad (median)", "chains", "grad/draw (median)", "range"],
    [[r.family, r.arm,
      num(r.chain_bulk_eff_median), string(r.chain_bulk_eff_range),
      string(r.chain_bulk_eff_n),
      num(r.chain_tail_eff_median), string(r.chain_tail_eff_n),
      num(r.chain_gradients_per_draw_median),
      string(r.chain_gradients_per_draw_range)] for r in rows])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
import Markdown
rows = ace_summary_rows(load_results("adaptive_centering_fixed_c/rows.json"))
short = [r for r in rows if r.chain_tail_eff_n != r.chain_bulk_eff_n]
Markdown.parse(isempty(short) ?
    "*Every median above summarises the same number of chains in both columns.*" :
    "*The two `chains` columns differ for " *
    join(["$(r.family) / $(r.arm) " *
          "($(r.chain_tail_eff_n) of $(r.chain_bulk_eff_n))" for r in short],
         ", ") *
    ". Those chains returned a non-finite tail ESS, which the artifact stores " *
    "as `null`; they are dropped from the tail median rather than imputed, so " *
    "read that median as describing the chains that finished.*")
```

The medians above compress each arm to one number. The per-chain values behind
them are what the boxes show, so an arm whose median looks competitive but whose
spread straddles another's is visible as such rather than having to be inferred
from the `range` column:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
load_harness("plots.jl")
d = load_results("adaptive_centering_fixed_c/rows.json")
vega_figure(ace_bulk_ess_spec(d);
    caption = "Per-chain minimum-coordinate bulk ESS per thousand gradient " *
              "evaluations, by arm, faceted by family. Log scale, shared " *
              "across facets. Boxes span min to max. Built from the same " *
              "chain rows as the table above, at build time.")
```

The scale is logarithmic, so equal vertical distances are equal *ratios*, not
equal differences. The spec refuses zero, negative, missing and non-numeric
values rather than plotting them, because a log axis drops those silently and a
chain that failed would leave no mark on the figure at all.

Pooled across chains, with the sampler diagnostics that decide whether any of it
is admissible:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
rows = ace_summary_rows(load_results("adaptive_centering_fixed_c/rows.json"))
md_table(
    ["family", "arm", "min bulk ESS", "bulk ESS/1k grad",
     "min tail ESS", "tail ESS/1k grad", "divergences", "max rank `Rhat`"],
    [[r.family, r.arm,
      num(r.pooled_min_bulk_ess), num(r.pooled_bulk_ess_per_1000_grad),
      num(r.pooled_min_tail_ess), num(r.pooled_tail_ess_per_1000_grad),
      string(r.divergences), num(r.max_rank_rhat)] for r in rows])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
import Markdown
d = load_results("adaptive_centering_fixed_c/rows.json")
n = length(d["rows"])
Markdown.parse("*" * provenance(d; harness = "adaptive_centering_fixed_c.jl") *
               " $(n) rows; summaries derived by " *
               "`docs/benchmark/adaptive_centering_fixed_c.jl` at build time.*")
```

## Paired comparison

The strict-online invariant proxy against the exact-score reference, paired
within seed on minimum-coordinate bulk ESS per thousand gradient evaluations.
The `seeds below 0.8` column shows how many of the paired seeds fall strictly
below `0.8`, so neither a favourable nor an unfavourable median stands on its
own. The verdict is not read off the median — it applies the rule the harness
predeclared, which the artifact stores and which is rendered below the table.
A median can sit below the threshold while too few individual seeds do for the
rule to fire, and where that happens the verdict says so rather than rounding
the median up into a finding.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
rows = ace_comparison_rows(load_results("adaptive_centering_fixed_c/rows.json"))
md_table(
    ["family", "median proxy ÷ reference", "range", "seeds below 0.8", "verdict"],
    [[r.family,
      num(r.median_proxy_reference_ratio), string(r.ratio_range),
      string(r.seeds_below_0_8), string(r.verdict)] for r in rows])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
mr = get(load_results("adaptive_centering_fixed_c/rows.json")["config"],
         "materiality_rule", Dict())
order = ["metric", "materially_better", "materially_worse", "otherwise"]
ks = [k for k in order if haskey(mr, k)]
append!(ks, sort([k for k in keys(mr) if !(k in order)]))
Markdown.parse(isempty(ks) ?
    "*The artifact declares no materiality rule.*" :
    "**The predeclared rule**\n\n" *
    join(["- **$(replace(k, "_" => " ")):** $(mr[k])" for k in ks], "\n"))
```

Every paired seed is drawn below against that rule, rather than only the median
and the count that the table reports. This is the view in which "the median sits
below `0.8` but too few individual seeds do for the rule to fire" stops being a
sentence to take on trust:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
load_harness("plots.jl")
d = load_results("adaptive_centering_fixed_c/rows.json")
vega_figure(ace_proxy_ratio_spec(d);
    caption = "Per-seed ratio of the strict-online invariant proxy to the " *
              "exact-score reference, on minimum-coordinate bulk ESS per " *
              "thousand gradient evaluations, faceted by family. The `1.0` " *
              "line is parity; the `0.8` line is the threshold the artifact " *
              "predeclares; the third line is the family median.")
```

Read it against the boundary the section above sets out: a ratio is a distance
from a reference that is exact by construction of the target, not a score in a
tuning contest. The `0.8` line is the artifact's own predeclared threshold and
is drawn from the same `materiality_rule` rendered above — it is not a level
this page chose.

## What the rows conclude

The sentence below is **assembled from the numbers** by `ace_conclusion`, not
written here. It cannot disagree with the tables above, because it is computed
from the same rows in the same build.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("adaptive_centering_fixed_c.jl")
import Markdown
Markdown.parse(ace_conclusion(load_results("adaptive_centering_fixed_c/rows.json")))
```

## The raw record

The per-chain rows are not reproduced here — there are too many to read as a
table, and they are already published. The checked-in JSON is the raw record,
and [Benchmark evidence](@ref) renders every file under
`docs/benchmark/results/` directly from it.

That appendix enumerates the directory, so it shows what is checked in; it
cannot notice a file being removed. What gates this artifact is that the page
above **names** it — a `load_results` call fails the build if the file goes
missing or changes shape, which is the guarantee the appendix by itself does not
provide.

## Reproduce it

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
cfg = load_results("adaptive_centering_fixed_c/rows.json")["config"]
rep = get(cfg, "reproduction", "")
Markdown.parse(isempty(rep) ?
    "*The artifact records no reproduction command.*" :
    "```bash\n" * rep * "\n```")
```

## See also

- [Nonlinear weighting evidence](@ref) — the same treatment for the nonlinear path.
- [Linear restart evidence](@ref) — and for the linear path.
- [Benchmark evidence](@ref) — the appendix rendering every checked-in result file.
