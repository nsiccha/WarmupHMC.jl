# Gaussian calibration

The target is a standard bivariate Gaussian with independent, dimensionless
coordinates. There are no observations or fitted hyperparameters: its analytic
mean and variance are known. That makes it a useful calibration case for the
target interface, draw orientation, and interpretation of diagnostics before
using a model whose posterior is unknown. This example examines one recorded
run, not a comparison of sampling algorithms or a claim about difficult posterior
geometry. The fit is cached as a small summary; the code receipt below binds that
summary to the target, random seed, and sampler source.

## Target and run

The object declares its dimension and first-order capability, then supplies a
log density and gradient. Dropping the Gaussian normalization constant leaves
the target distribution unchanged. The gradient has the same length and
coordinate order as the input vector. No automatic differentiation package or
nonlinear coordinate map is needed for this target. For a hierarchical target,
the [reparametrization guide](../guides/reparametrization.md) explains how to add
a map while keeping the model and sampler coordinate conventions explicit.

The following program is the exact source executed for the cached fit. The
documentation build checks its hash and the package source hash before showing
its saved output. Sampling is deliberately excluded from the build and CI;
the remaining calculations and figure construction execute at build time.

```@eval
using WarmupHMC
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl"))
ManualCache.program("gaussian")
```

```@example gaussian
using WarmupHMC
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl")) # hide
target_source = first(split(read(joinpath(pkgdir(WarmupHMC), "bench/docs/gaussian.jl"), String), "\nresult =")) # hide
include_string(@__MODULE__, target_source) # hide
cache = ManualCache.load_summary(); # hide
Tuple(cache["shape"]) # hide
```

The rows are model coordinates and the columns are retained draws. The output
reports the actual shape rather than assuming the requested floor is the exact
count. With a nonlinear wrapper, returned positions would already be in model
coordinates; applying the map again would change the sample. Raw warm-up state
and gradients have a different coordinate contract, described in
[Sampling](../guides/sampling.md).

## Coordinate view

A fixed scale transformation illustrates the distinction between the sampler's
position and the model's position. This evaluates a density without fitting or
requiring an AD backend. The Boolean output checks that the wrapper includes the
Jacobian for a non-centered first coordinate:

```@example gaussian
rule = Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), 0.0, log(2.0))
wrapped = ReparametrizedProblem(IndexedReparametrization([1 => rule]), Gaussian())
x = [0.3, -0.4]
y = [2x[1], x[2]]
isapprox(LogDensityProblems.logdensity(wrapped, x),
    log(2.0) + LogDensityProblems.logdensity(Gaussian(), y))
```

The sampled run above uses the original Gaussian directly. The view demonstrates
coordinate bookkeeping; it is not an additional fit or a performance comparison.

## Coordinate summaries

Compare each estimated mean and standard deviation with the analytic target.
The table also shows a Monte Carlo standard error for the mean, estimated as the
sample standard deviation divided by the square root of the reported effective
sample size. That estimate describes simulation uncertainty under the chain's
dependence structure. It is not posterior uncertainty about the coordinate, and
it is not evidence that all modes of an unknown target have been explored.

```@example gaussian
rows = cache["coordinates"]
ManualCache.table(
    ["Coordinate", "Mean", "SD", "Mean MCSE", "ESS (draws)"],
    [[r["coordinate"], round(r["mean"]; digits=4), round(r["sd"]; digits=4),
      round(r["mean_mcse"]; digits=4), round(r["ess"]; digits=1)] for r in rows])
```

The figure places the estimated moments beside the analytic mean and standard
deviation. Both axes are dimensionless. A point near the analytic marker shows
agreement of those two marginal summaries; it does not check the joint
distribution, tail probabilities, or independence between coordinates. The
table's MCSE helps put the horizontal separation into context. Standard
deviation has its own simulation uncertainty, which is not estimated by the
mean MCSE column.

```@eval
using WarmupHMC, AlgebraOfVega, Markdown
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl"))
rows = ManualCache.load_summary()["coordinates"]
points = (
    mean=vcat(0.0, [r["mean"] for r in rows]),
    sd=vcat(1.0, [r["sd"] for r in rows]),
    series=vcat("Analytic target", ["Coordinate $(r["coordinate"])" for r in rows]),
)
figure = data(points) * mapping(:mean => "Mean (dimensionless)",
    :sd => "Standard deviation (dimensionless)"; color=:series => "Summary") *
    visual(Scatter; markersize=120) *
    config(width=380, height=240,
        scales=scales(Color=(; palette=["#2a78d6", "#eb6834", "#eb6834"])))
Markdown.MD(Markdown.Code("vega-lite", AlgebraOfVega.to_json(figure)))
```

Blue marks the analytic target; orange marks the two sampled coordinates.
Distance from the blue point shows the discrepancy in the marginal mean and
standard deviation for this recorded run.

## Diagnostics

The following values describe the retained sample and the work the sampler
recorded. Inspect divergences alongside moments rather than treating an accurate
mean as a substitute for trajectory diagnostics. A chain can have a plausible
mean while missing important regions of its target. Conversely, a finite run
need not reproduce an analytic moment exactly even when its transitions behave
as intended. For an unfamiliar model, repeat the analysis with independent
initializations and compare chains before combining their draws.

```@example gaussian
diagnostics = cache["diagnostics"]
ManualCache.table(["Diagnostic", "Recorded value"], [
    ["Divergent retained draws", diagnostics["divergent_draws"]],
    ["Retained-sampling gradient evaluations", diagnostics["sampling_gradient_evaluations"]],
    ["Run gradient evaluations", diagnostics["run_gradient_evaluations"]],
    ["Step size", round(diagnostics["stepsize"]; digits=4)],
    ["Selected linear transformation", diagnostics["transformation"]],
])
```

Gradient evaluations are counts, not seconds. The retained-sampling counter
belongs to the transitions behind the returned draw matrix. The run counter
includes discarded epochs and step-size adaptation but excludes Pathfinder and
the initial step-size search. Dividing the run counter by the retained draw
count therefore answers a different cost question from dividing the sampling
counter by that count. Neither ratio alone measures the accuracy of a posterior
estimate; compare simulation error as well as work when assessing a sampler.

## Continuation

The same cached run collection includes a checkpoint demonstration. Its program
and visible before/after result are in [Checkpoints](../guides/checkpoints.md).
It saves an initial sample, restores the same target with a larger draw floor,
and compares the prefix of the continued matrix with the original draws.
That comparison checks continuity of the retained sample, not the validity of
resuming after changing data or a likelihood. Such changes define a different
target and are not established by a matching dimension.

For a production analysis, keep the model inputs and sampler configuration
alongside the checkpoint. Select a persistent path and preserve the dependency
environment needed to read its serialized state. The temporary directory in the
demonstration is appropriate only because the program checks its result before
the directory is removed. The saved summary below is sufficient to rebuild this
page, but intentionally cannot resume the fitted chain because it contains no
draws, RNG state, or checkpoint payload.

::: details Reproduce

Run these commands from the repository root after installing the docs
dependencies. Regenerate only when the example or sampler source changes;
CI reads the committed summary and checks its receipt.

```sh
julia --project=docs bench/docs/run.jl
julia --project=docs docs/make.jl
```

```@example gaussian
receipt = cache["receipt"]
ManualCache.table(["Receipt field", "Value"],
    [[key, receipt[key]] for key in
     ["created_utc", "julia", "warmuphmc", "base_revision", "program_sha256", "source_sha256", "command"]])
```

The [cache generator README](https://github.com/nsiccha/WarmupHMC.jl/blob/dev/bench/docs/README.md)
describes the hash inputs and saved dependency versions. Broader experiments and
historical benchmark regeneration belong in the
[research README](https://github.com/nsiccha/WarmupHMC.jl/blob/dev/bench/README.md).
The base revision identifies the checkout before documentation edits; the
source hash identifies the package files that actually executed.

:::
