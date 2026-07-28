# Linear restart evidence

WarmupHMC can base its linear restart condition on either the legacy thinned
halo or a running transformed marginal-scale estimate. The running estimate can
use all acceptable NUTS leaves or the NUTS multinomial weights, with either unit
trajectory exposure or a whole-trajectory step-size multiplier.

## Current recommendation

Keep the default `linear_restart_source=:halo` with
`linear_trajectory_weighting=:unit`.

The running estimators are useful opt-in experiments, but this benchmark does
not identify a policy that wins reliably enough to replace the default:

- On the clean diagonal and block-correlated Gaussian controls, the policies
  usually make the same restart decision. The correlated control does select a
  genuinely non-diagonal successive-reflection metric, so agreement there is a
  real control rather than another diagonal target.
- On the forced diagonal Student-t fallback control, running estimates can
  improve sampling efficiency, but the advantage does not pair with a
  consistent improvement in covariance accuracy. Unit and step-size exposure
  also rank differently for the all-leaf and NUTS-weighted sources.
- On `kilpisjarvi` and `diamonds`, changed restart decisions are rare and can
  help or hurt substantially. That is evidence for target-specific tuning, not
  for a new global default.

In particular, step-size weighting is not a general improvement. Treat
`:all_good_leaves` and `:nuts_weighted` as configurable policies to benchmark on
the target at hand.

## Median summary

This table is generated during the documentation build from
`docs/benchmark/results/linear_restart.json`. A missing file or changed schema
therefore fails the docs build instead of leaving stale figures on this page.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
d = load_results("linear_restart.json")
md_table(
    ["target", "policy", "restarts", "min ESS/1k gradients",
     "metric", "reflections", "fallbacks"],
    [["`" * r["target"] * "`", "`" * r["arm"] * "`",
      num(r["restarts_median"]), num(r["min_ess_per_kgrad_median"]),
      "`" * r["active_transformations"] * "`",
      num(r["adaptive_reflections_median"]),
      num(r["linear_metric_fallbacks_median"])] for r in d["summary"]])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
Markdown.parse("*" * provenance(load_results("linear_restart.json");
                                harness = "run_linear_restart_benchmark.jl") * "*")
```

Paired ratios below are test arm ÷ baseline. Larger is better for ESS
efficiency; smaller is better for the accuracy-error column. The win columns
count strict seed-level improvements, so exact ties remain visible as neither
arm winning.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
d = load_results("linear_restart.json")
md_table(
    ["target", "comparison", "test ÷ baseline", "ESS ratio", "ESS wins",
     "accuracy metric", "error ratio", "accuracy wins", "restart count differs"],
    [["`" * r["target"] * "`", r["comparison"],
      "`" * r["arm"] * "` ÷ `" * r["baseline_arm"] * "`",
      num(r["ess_efficiency_ratio_median"]),
      "$(r["ess_efficiency_wins"])/$(r["n_pairs"])",
      "`" * r["accuracy_metric"] * "`",
      num(r["accuracy_error_ratio_median"]),
      "$(r["accuracy_wins"])/$(r["n_pairs"])",
      num(r["restart_count_differs"])] for r in d["comparisons"]])
```

## Recorded experiment

The view below is a static recording of the WarmupHMC web app's generic
benchmark renderer. Its tables are generated directly from the checked-in JSON;
the prose above does not carry a second copy of the numbers.

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-whmc/benchmark/linear_restart" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading linear restart evidence…</em>
</div>
</div>
```

The JSON contains the per-policy medians, paired comparisons, all individual
runs, and the exact WarmupHMC revision and runtime provenance. Wall time is
host-specific; the portable efficiency measure is minimum ESS per thousand
gradient evaluations. The
[raw per-seed JSON](https://github.com/nsiccha/WarmupHMC.jl/blob/dev/docs/benchmark/results/linear_restart.json)
remains available even when the recorded app view is not being served.

## Reproduce it

The full matrix uses 32 fixed seeds, balanced arm order, single-threaded BLAS,
three analytic controls, and the correlated PosteriorDB targets `kilpisjarvi`
and `diamonds`. Nonlinear adaptation is disabled in every arm.

```bash
julia --project=docs/benchmark docs/benchmark/run_linear_restart_benchmark.jl
```

The benchmark environment needs the same one-resolve setup described in
[`docs/benchmark/README.md`](https://github.com/nsiccha/WarmupHMC.jl/blob/dev/docs/benchmark/README.md).
The checked-in full study is a deliberate quiet-host run: shared CI timing is
too noisy to make a useful performance gate. GitHub Actions nevertheless runs
the exact driver and schema on the three analytic controls with the quick
configuration below, checks the reflection and fallback controls, and uploads
its per-seed JSON artifact:

```bash
WHMC_LINEAR_BENCH_SEEDS=2 \
WHMC_LINEAR_BENCH_DRAWS=250 \
WHMC_LINEAR_BENCH_EVALS=250 \
WHMC_LINEAR_BENCH_TARGETS=diag_gaussian,diag_fallback_probe,correlated_gaussian \
WHMC_LINEAR_BENCH_OUT=/tmp/linear-restart-smoke \
julia --project=docs/benchmark docs/benchmark/run_linear_restart_benchmark.jl
```
