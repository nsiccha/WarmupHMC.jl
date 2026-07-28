# WarmupHMC.jl

Adaptive NUTS warm-up for linear transformations and step size.

## Quickstart

Everything below this heading runs during the docs build. The numbers on this
page are what the code returned, not what someone typed next to it — see
[A note on the code blocks](@ref) at the foot of the page for which blocks that
covers and which it does not.

WarmupHMC samples anything implementing the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface.
The target here supplies its own gradient, so no AD backend is involved:

```@example quickstart
using WarmupHMC, LogDensityProblems, Random, Statistics

struct DiagGaussian
    mu::Vector{Float64}
    sigma::Vector{Float64}
end

LogDensityProblems.dimension(p::DiagGaussian) = length(p.mu)
LogDensityProblems.capabilities(::Type{DiagGaussian}) =
    LogDensityProblems.LogDensityOrder{1}()

# Up to a constant — MCMC never needs the normalization.
LogDensityProblems.logdensity(p::DiagGaussian, x) =
    -sum(abs2, (x .- p.mu) ./ p.sigma) / 2

function LogDensityProblems.logdensity_and_gradient(p::DiagGaussian, x)
    z = (x .- p.mu) ./ p.sigma
    (-sum(abs2, z) / 2, -z ./ p.sigma)
end

lpdf = DiagGaussian([1.0, -2.0, 0.5], [0.5, 2.0, 1.0])
nothing # hide
```

That is the whole setup. Sampling is one call:

```@example quickstart
result = adaptive_warmup_mcmc(Xoshiro(20260728), lpdf; n_draws=400, progress=nothing)
nothing # hide
```

The result is a `NamedTuple`; `posterior_position` holds the draws, one column
per draw:

```@example quickstart
draws = result.posterior_position          # dimension × n_draws
size(draws)
```

```@example quickstart
(; mean = round.(vec(mean(draws; dims=2)); digits=2),
   std  = round.(vec(std(draws; dims=2)); digits=2))
```

Compare that against the `mu = [1.0, -2.0, 0.5]` and `sigma = [0.5, 2.0, 1.0]`
the target was built with.

`progress=nothing` turns the progress bar off, which is what you want in a
script or a docs build; drop it and, with
[Treebars.jl](https://github.com/nsiccha/Treebars.jl) loaded, you get a live
one.

## Method

WarmupHMC adaptively:

* Learns a **linear transformation** of the posterior that simplifies MCMC sampling,
* Optionally learns **nonlinear reparametrizations** (adaptive partial centering for hierarchical models),
* Learns a NUTS step size (using standard Dual Averaging), and
* Returns samples from the posterior.

The warm-up procedure is windowed and inspired by [Stan](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)'s and [nutpie](https://github.com/pymc-devs/nutpie)'s warm-up procedures, but differs in several important ways:

* We initialize using Pathfinder.
* Our warm-up windows aim to reach a certain number of **gradient evaluations**, instead of a certain number of MCMC transitions (Stan). We start with a default target of 1000 gradient evaluations, and double that target after each warm-up window.
* Instead of only using the posterior positions (Stan), we use the posterior **positions and gradients** (like e.g. nutpie).
* Instead of only using the MCMC/posterior positions and gradients (Stan and nutpie), we also store and use the **intermediate positions and gradients**, i.e. the ones that MCMC visits before returning the "final" new position. We store up to a default of 1000 intermediate positions and gradients, selected pseudo-randomly, and only if the Hamiltonian error is small enough.
* Instead of only learning a single linear transformation and updating that one repeatedly, we learn **several transformations in parallel** and at the end of each warm-up window select the one that minimizes a loss function. Currently, we learn three different linear transformations:
    * Pathfinder's initial transformation, enriched by an updated additional diagonal scaling,
    * A standard diagonal "mass matrix",
    * A novel, adaptive sequence of Householder transformations followed by diagonal scaling.
* Instead of running the warm-up for a fixed number of windows, we try to **estimate when continuing warming up is harmful/useless** and stop warming up then.

## Nonlinear Reparametrizations

WarmupHMC supports adaptive **partial centering** for hierarchical models. For parameters
with a location-scale hierarchy (e.g. `x ~ Normal(mu, sigma)`), the centering parameter
`c ∈ [0, 1]` interpolates between the centered (`c = 1`) and non-centered (`c = 0`)
parametrizations, and is re-fitted per coordinate at warm-up window boundaries against
the recorded intermediate positions and gradients.

This is **opt-in and entirely caller-driven**: `nonlinear_adapt=true` is the default, but
it adapts nothing unless you wrap your problem in a [`ReparametrizedProblem`](@ref)
carrying a non-empty [`IndexedReparametrization`](@ref) that names the coordinates and
their location/log-scale accessors. Nothing is detected automatically.

```julia
using WarmupHMC, DifferentiationInterface, Enzyme

# Eight schools example: dims 1:8 are group effects, dim 9 = location, dim 10 = log-scale
ir = IndexedReparametrization(
    1:8 .=> Ref(Reparametrization(
        PartiallyCentered(1.0), PartiallyCentered(1.0),
        x -> x[9], x -> x[10]
    ))
)
rp = ReparametrizedProblem(ir, my_problem,
    AutoEnzyme(; function_annotation=Enzyme.Const))
result = adaptive_warmup_mcmc(rng, rp)
```

Posterior samples are transformed back to the wrapped problem's own parametrization
before being returned.

The third argument is a DifferentiationInterface.jl backend and is **not** optional:
the two-argument form constructs fine and then fails on the first gradient. A
**reverse-mode** backend is the reasonable default — what is differentiated is a
scalar objective in the full parameter vector, so reverse mode costs one pass
whatever the dimension, while forward mode costs one per coordinate. That is an
operation count, though, and it is not wall-clock; measure your own target.
[`ReparametrizedProblem`](@ref) carries the per-target table and is explicit
about which rows it can currently stand behind.

With Enzyme, `function_annotation=Enzyme.Const` is not decoration: a bare
`AutoEnzyme()` also constructs fine and then fails on the first gradient. See
[Nonlinear reparametrization](@ref).

DifferentiationInterface is a hard dependency, so the interface is always available;
the **backend package** is not, and you load it yourself (`AutoEnzyme` needs
`using Enzyme`). WarmupHMC constructs no backend of its own and depends on no
particular AD implementation — which one you use is your choice.

See [Nonlinear reparametrization](@ref) for a runnable end-to-end example, how the
centering is fitted, and the constraints that bite (coordinate order across a resume,
what a checkpoint does and does not store, and the cost on the gradient hot path).
If the model comes from BayesianRegressionModels.jl, use
[A BRM random-intercept example](@ref) instead: BRM discovers the random-effect
coordinates and constructs the candidate-scoring plan, so you do not write an
`IndexedReparametrization` by hand.

## Checkpoints, Callbacks and Resume

Warm-up has two checkpoint boundaries: **CP-0**, right after initialization
(Pathfinder), and **CP-N**, after each outer warm-up window. Three opt-in,
independent mechanisms hang off these boundaries. All default to off, and with
them off the run is byte-for-byte identical to the plain sampler.

**Observing progress.** `callback=(state, stage) -> should_stop` fires at each
boundary with `stage ∈ (:init, :window)`. It may read `state` and request an
early stop by returning `true`, but it is *observational* — it must not mutate
`state`. Collecting the `stage` it is handed is enough to show the boundaries
above are the boundaries you actually get:

```@example quickstart
boundaries = Symbol[]
adaptive_warmup_mcmc(Xoshiro(20260728), lpdf; n_draws=400, progress=nothing,
    callback = (state, stage) -> (push!(boundaries, stage); false))
boundaries
```

**Writing checkpoints.** `checkpoint_dir=path` serializes a resumable snapshot at
each boundary as `cp_init.jls`, `cp_window_<n>.jls`, and an overwritten
`cp_latest.jls` — one `cp_window_<n>.jls` per `:window` above:

```@example quickstart
dir = mktempdir()
adaptive_warmup_mcmc(Xoshiro(20260728), lpdf;
                     n_draws=400, progress=nothing, checkpoint_dir=dir)
sort(readdir(dir))
```

The multi-chain method writes chain `i` under `path/chain_<i>/`. The snapshot
deliberately excludes the (possibly non-serializable) inner problem and stores
the reparametrizer only as its scalar centering values — so the `lpdf` you
resume with is one **you** reconstruct, typically the same way you built it for
the original call.

**Resuming.** Point the sampler at the same directory with `resume=true`:

```@example quickstart
resumed = adaptive_warmup_mcmc(Xoshiro(20260728), lpdf;
                               checkpoint_dir=dir, resume=true,
                               n_draws=400, progress=nothing)
size(resumed.posterior_position)
```

Resuming this way takes its configuration from the **call**, so asking for a
larger `n_draws` than the original run keeps sampling rather than re-running a
fixed-length batch.

!!! warning "`resume_warmup_mcmc` is deprecated"
    [`resume_warmup_mcmc`](@ref)`(lpdf, "checkpoints/cp_latest.jls")` still
    works and is still exported, but resuming is no longer a separate function.
    It reads no configuration from the checkpoint either — a caller that relied
    on the payload carrying the original run's `n_draws` must now pass it
    explicitly. Prefer the `resume=true` form above.

## A note on the code blocks

The blocks on this page come in two kinds, and the rendered page does not
distinguish them, so it is said here instead.

**Executed.** The Quickstart and the checkpoint/resume blocks are Documenter
`@example` blocks. They run in a shared session during every docs build, and
the outputs shown are what they returned. If a field is renamed, a kwarg
dropped, or `resume=true` stops working, the build fails and this page cannot
be published saying otherwise.

**Not executed.** The reparametrization block under
[Nonlinear Reparametrizations](@ref) is a plain fence — it needs an AD backend
(`Enzyme`), which the docs environment deliberately does not carry, and it
refers to a `my_problem` you supply. It is illustration, not a transcript. The
end-to-end version that *is* complete lives on
[Nonlinear reparametrization](@ref); it is likewise not executed here.

The distinction matters because an unexecuted example makes exactly the same
visual claim as an executed one while nothing checks it. This manual had no
executed blocks at all until the Quickstart above; the code in them was
plausible, and being plausible is not the same as having run.

## See Also

- [Stan's warm-up documentation](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)
- [nutpie](https://github.com/pymc-devs/nutpie) -- Python NUTS sampler with similar ideas
- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) -- Julia HMC implementation
