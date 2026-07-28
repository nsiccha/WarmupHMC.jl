# WarmupHMC vs other samplers

The README says results "should come in faster than with 'standard' methods, and
should often be better". This page is what that sentence is worth.

Every number below is recomputed from
[`docs/benchmark/results/sampler_comparison.json`](https://github.com/nsiccha/WarmupHMC.jl/tree/main/docs/benchmark/results)
during the docs build — including the verdict, which is *counted* from the rows
rather than written down. If a re-measurement inverts the result, this page
inverts with it.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
Markdown.parse("*" * provenance(load_results("sampler_comparison.json");
                                harness = "bench/sampler_comparison.jl") * "*")
```

## The three arms

| Arm | What it runs |
| --- | --- |
| **WarmupHMC** | `adaptive_warmup_mcmc(rng, problem; n_draws)` — defaults throughout |
| **DynamicHMC** | `DynamicHMC.mcmc_with_warmup(rng, problem, n_draws)` — defaults throughout |
| **AdvancedHMC** | NUTS (multinomial, generalised no-U-turn, max depth 10) with Stan's adaptor — diagonal mass matrix, dual averaging to 0.8, 1000 adaptation iterations, warm-up dropped |

These are the same three arms the [Gallery](@ref) dashboard runs, so a figure
here and a cell there mean the same thing. Gradient evaluations are counted for
all three by the *same* counter, wrapped around the problem itself
(`WarmupHMC.count_and_time`), rather than read from each sampler's own
bookkeeping — those count different things.

## Effective sample per gradient evaluation

This is the portable number and the one to quote: it does not depend on the
machine, the BLAS build, or what else was running.

`min ESS` is taken over the sampler's own unconstrained coordinates, which
invites the obvious objection — you declared `tau`, the table scored `log tau`.
It does not bite here, and not by luck: `MCMCDiagnosticTools.ess` defaults to
**bulk** ESS, which rank-normalizes first and is therefore invariant under any
strictly monotone reparametrization of a coordinate. Measured on an AR(1)
series, `ess(x)`, `ess(exp(x))` and `ess(x^3)` agree to 14 digits, while
`ess(x^2)` — not monotone — does not. Every result file in
`docs/benchmark/results/` records `ess_con_min` beside `ess_min` for exactly
this check, and across the whole corpus the two are byte-identical in every
row; the *medians* separate, because a model's transformed parameters are
functions of several coordinates at once and have no unconstrained
counterpart to be invariant to.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
d = load_results("sampler_comparison.json")
md_table(
    ["target", "sampler", "min ESS", "ESS / gradient", "gradients", "divergences", "runs ok"],
    [[r.target, COMPARISON_ARM_LABELS[r.arm], num(r.ess_min), num(r.ess_per_grad),
      num(r.grad_evals), string(r.n_divergent), string(r.n_ok, "/", r.n_run)]
     for r in comparison_summary(d)],
)
```

Ratios, WarmupHMC ÷ the reference sampler. Above 1 means WarmupHMC extracted
more effective sample from the same number of gradient evaluations:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
d = load_results("sampler_comparison.json")
md_table(
    ["target", "vs", "ESS/gradient ratio"],
    [[r.target, COMPARISON_ARM_LABELS[r.versus], num(r.ratio)]
     for r in comparison_ratios(d; key = "ess_min_per_grad")],
)
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
import Markdown
d = load_results("sampler_comparison.json")
Markdown.parse("**" * comparison_verdict_sentence(d; key = "ess_min_per_grad") * "** " *
               comparison_stability_sentence(d; key = "ess_min_per_grad"))
```

## Effective sample per wall-clock second

"Faster" is a wall-clock word, so this is here too — but it is
machine-specific, and it does not have to agree with the table above. WarmupHMC
does strictly more work per gradient evaluation (a Pathfinder initialization,
several linear transformations fitted in parallel and scored against each
other), so a lead in gradient count is not automatically a lead in seconds.

**This is the weaker of the two tables, and the artifact says so rather than
this paragraph asserting it.** Every figure is measured once per (target, arm,
seed, repeat); the repeats re-run identical seeds, so the draws are
bit-identical and only the clock moves. That makes the verdict below checkable
against itself — see the sentence under it, and contrast it with the one under
the gradient verdict.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
d = load_results("sampler_comparison.json")
md_table(
    ["target", "vs", "ESS/second ratio"],
    [[r.target, COMPARISON_ARM_LABELS[r.versus], num(r.ratio)]
     for r in comparison_ratios(d; key = "ess_min_per_s")],
)
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
import Markdown
d = load_results("sampler_comparison.json")
Markdown.parse("**" * comparison_verdict_sentence(d; key = "ess_min_per_s") * "** " *
               comparison_stability_sentence(d; key = "ess_min_per_s"))
```

Both verdicts, recounted inside each repeat:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
d = load_results("sampler_comparison.json")
wlt(v) = string(v.n_wins, "–", v.n_losses, "–", v.n_ties)
grad = comparison_verdict_by_repeat(d; key = "ess_min_per_grad")
secs = comparison_verdict_by_repeat(d; key = "ess_min_per_s")
md_table(
    ["repeat", "ESS/gradient (win–loss–tie)", "ESS/second (win–loss–tie)"],
    [[string(g.repeat), wlt(g), wlt(s)] for (g, s) in zip(grad, secs)],
)
```

## Arms that failed

An arm that could not run on a target is a result about that arm, not a reason
to drop the row. If the table below is empty, every arm completed on every
target and seed.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("sampler_comparison_summary.jl")
import Markdown
d = load_results("sampler_comparison.json")
f = comparison_failures(d)
isempty(f) ? Markdown.parse("*No arm failed on any target or seed in this run.*") :
md_table(["target", "sampler", "failed seeds", "error"],
         [[r.target, COMPARISON_ARM_LABELS[r.arm], string(r.n_failed), r.error] for r in f])
```

## What this does not measure

Read this section before quoting anything above.

* **Correctness.** These are efficiency measurements. Nothing here verifies the
  draws are from the right distribution — there is no reference-posterior
  comparison on this page. **An efficiency win on incorrect draws is worse than
  no claim.** Divergence counts are the one cheap smell test, and they are in
  the first table rather than in a footnote.
* **One chain per run.** `min ESS` is within-chain, computed from the returned
  draws by `MCMCDiagnosticTools`. There is no R-hat here and no between-chain
  diagnostic.
* **A tuned AdvancedHMC.** Its arm is a reasonable default configuration, not an
  expert's — the initial step size and starting point are the dashboard's
  choices, kept for comparability. Read its numbers as "AdvancedHMC out of the
  box", never as "AdvancedHMC at its best".
* **Warm-up budget parity.** Each sampler decides for itself how much warm-up to
  spend; only AdvancedHMC's is stated explicitly, because it has no default and
  the number therefore had to be chosen. This is the honest comparison for
  someone picking a sampler and the *wrong* comparison for attributing a win to
  any one mechanism.
* **Wall-clock portability.** The second table is `strato2`, single-threaded
  BLAS. Only the per-gradient figures travel.
* **Few targets.** Four posteriordb posteriors — every one with a ready
  reparametrization spec in `web/src/posteriordb_reparametrizations.jl` — plus
  one synthetic funnel, because posteriordb ships none. That is enough to refute
  a universal claim and **not enough to establish one**. The verdict sentences
  above say "on this target set" for that reason.
* **The default configuration only.** No reparametrization is fitted in any arm;
  `nonlinear_adapt` adapts nothing without a spec. What partial centering buys on
  top of this is a different measurement, on
  [Nonlinear reparametrization](@ref).

## Reproducing it

```sh
julia --project=bench -e 'using Pkg; Pkg.develop([
    PackageSpec(path=pwd()), PackageSpec(path="../Treebars.jl")]); Pkg.instantiate()'
julia --project=bench bench/sampler_comparison.jl
```

The `Pkg.develop` line is needed because Treebars.jl is not registered and
Julia 1.10 ignores `[sources]`; the same step appears in `.github/workflows/`.
Knobs: `WHMC_CMP_SEEDS`, `WHMC_CMP_DRAWS`, `WHMC_CMP_REPEATS`, `WHMC_CMP_TARGETS` (comma-separated,
`funnel` included), `WHMC_CMP_OUT`.
