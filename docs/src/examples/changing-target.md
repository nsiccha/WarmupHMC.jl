# Changing Gaussian target

The data for this example are a synthetic sequence of Gaussian locations. Each
location defines a different univariate target with unit standard deviation;
all values are dimensionless. There are no observations, estimated parameters,
or external datasets. A chain carries its position and random-number state from
one target to the next while its kernel settings remain fixed. This makes the
state transition visible without introducing an optimizer or an application
model. The resulting sequence illustrates an interface for changing targets;
it is not a sample from one posterior distribution.

## Target

The object stores the current location and implements the first-order
`LogDensityProblems` interface. Changing the location changes both the density
and its gradient. The dimension and coordinate frame remain the same, which is
required when reusing a saved state. The omitted normalization constant is
independent of the position, so it has no effect on a NUTS transition. A target
with constrained support would additionally need to handle invalid positions
according to the target interface.

```@example moving
using WarmupHMC, LogDensityProblems, Random
struct ShiftedGaussian
    location::Float64
end
LogDensityProblems.dimension(::ShiftedGaussian) = 1
LogDensityProblems.capabilities(::Type{ShiftedGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::ShiftedGaussian, x) = -sum(abs2, x .- p.location) / 2
LogDensityProblems.logdensity_and_gradient(p::ShiftedGaussian, x) =
    (LogDensityProblems.logdensity(p, x), p.location .- x)

locations = collect(range(0.0, 1.0; length=12))
(; dimension=LogDensityProblems.dimension(ShiftedGaussian(first(locations))),
   targets=length(locations), first_location=first(locations), last_location=last(locations))
```

The generated view reports the sequence that the following loop actually uses.
The location moves gradually, but that does not establish that one transition
per target is sufficient for any particular inference procedure. An outer
algorithm must determine its own transition budget and convergence criteria.
Keeping that distinction explicit prevents a plausible looking trajectory from
being mistaken for evidence about an estimator that this page does not run.

## State and transitions

Construct a state with a reproducible RNG, an initial position, a scale operator,
and a step size. Here the identity scale matches the coordinate units. The
`metric` input is a scale operator, so it should not be interpreted as an
arbitrary covariance matrix. For a model with a different coordinate system,
obtain suitable settings from warm-up or choose them with the intended scale
convention in mind. This example supplies its settings directly and performs
no adaptation or posterior fit during the documentation build.

```@example moving
function follow_targets(locations)
    state = FixedKernelNUTSState(Xoshiro(18), [0.0]; metric=[1.0], stepsize=0.2)
    positions, divergences, stepsizes = Float64[], Bool[], Float64[]
    for location in locations
        step = fixed_kernel_nuts_step(state, ShiftedGaussian(location))
        state = step.state
        push!(positions, only(state.position))
        push!(divergences, step.divergent)
        push!(stepsizes, state.stepsize)
    end
    (; positions, divergences, stepsizes)
end
trajectory = follow_targets(locations)
using JSON # hide
write(joinpath(pkgdir(WarmupHMC), "docs/build/changing-target.json"), JSON.json((; locations, trajectory.positions))) # hide
(; transitions=length(trajectory.positions),
   unchanged_step_size=all(==(0.2), trajectory.stepsizes),
   divergent_transitions=count(trajectory.divergences))
```

The returned `state` becomes the input to the next call. Reusing the original
state instead would discard the intended progression. Each call evaluates the
new target at the saved position before constructing its transition; a cached
gradient from the previous target is not a valid substitute. The unchanged
step-size output checks one part of the fixed-kernel contract directly. The
scale and tree-depth limit are also held fixed by this API, although they are
not separate measurements in the printed result.

## Diagnostics

Read the divergence flags before interpreting the trajectory. A divergence is
a transition diagnostic, while agreement with a location is a statement about
a particular position. Neither is a replacement for the other. A draw need not
equal the target mean, even for this Gaussian with known moments. The table
shows the target and saved position at each transition so the result can be
inspected without inferring values from the plot.

```@example moving
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl")) # hide
ManualCache.table(["Transition", "Target location", "Saved position", "Divergent"],
    [Any[i, round(locations[i]; digits=3), round(trajectory.positions[i]; digits=3),
      trajectory.divergences[i]] for i in eachindex(locations)])
```

A saved position outside the next target's support is rejected before the
transition. Leaving support during a leapfrog trajectory is handled as a
divergence. This Gaussian has unrestricted support, so it isolates the state
handoff from those boundary cases. In a constrained model, check that an outer
parameter update preserves a meaningful starting position before calling the
transition function. Changing the target dimension would require a compatible
new state rather than the handoff illustrated here.

## State sequence

The figure compares the changing target location with the position returned
after each transition. The horizontal axis counts transitions, not elapsed
seconds or retained draws from a fixed target. Connecting the positions makes
their order visible; the connecting segments do not represent intermediate
leapfrog states. The vertical separation describes the realized trajectory for
this seed and these settings. It is not a tracking-error estimate averaged over
repeated runs.

```@eval
using WarmupHMC, AlgebraOfVega, Markdown, JSON
values = JSON.parsefile(joinpath(pkgdir(WarmupHMC), "docs/build/changing-target.json"))
locations, positions = values["locations"], values["positions"]
plot_data = (
    transition=repeat(collect(eachindex(locations)), 2),
    value=vcat(locations, positions),
    series=vcat(fill("Target location", length(locations)),
                fill("Saved position", length(locations))),
)
figure = data(plot_data) * mapping(:transition => "Transition (count)",
    :value => "Position (dimensionless)"; color=:series => "Sequence") *
    visual(Lines) * config(width=420, height=240, font_scale=1.3,
        scales=scales(Color=(; categories=["Target location", "Saved position"],
            palette=["#2a78d6", "#eb6834"])))
Markdown.MD(Markdown.Code("vega-lite", AlgebraOfVega.to_json(figure)))
```

Blue shows the supplied target location; orange shows the saved position.
Their separation shows the random displacement after each transition, including
the effect of carrying the previous position into a changed target.

## Interpretation

Do not pool this sequence to estimate a stationary posterior mean or effective
sample size. Each position was produced under a different target. If an outer
procedure eventually fixes the target, its subsequent draws and diagnostics
need a separate interpretation. For an unchanged target with persisted draws,
the stream interface provides a disk-backed alternative; close its mappings
after copying any values that must survive removal of the stream files.

If the geometry changes enough to require continued adaptation, use
`ContinuousNUTSState` and `continuous_nuts_step`. That interface updates its
metric and step size from decaying evidence, whereas the example here preserves
the selected kernel. The [changing-target guide](../guides/changing-targets.md)
describes that choice, and the [transition reference](../reference/streams.md)
lists the constructors and returned fields. For calibration against one fixed
distribution, use the [Gaussian example](gaussian.md), whose moment summaries
have a stationary target to compare with.

::: details Reproduce

The transitions, tables, and figure run during the documentation build. No
sampling result or figure from this example is committed. From the repository
root, after installing the documentation dependencies, run:

```sh
julia --project=docs docs/make.jl
```

```@example moving
ManualCache.table(["Receipt field", "Value"], [
    ["Julia", string(VERSION)],
    ["WarmupHMC", string(pkgversion(WarmupHMC))],
    ["Source revision", readchomp(`git -C $(pkgdir(WarmupHMC)) rev-parse HEAD`)],
    ["RNG", "Xoshiro(18)"],
])
```

The [research README](https://github.com/nsiccha/WarmupHMC.jl/blob/dev/bench/README.md)
describes the separate experiment drivers. This page uses only its own synthetic
target; it does not reproduce an external study or retain a benchmark history.

:::
