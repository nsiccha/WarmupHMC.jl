# WarmupHMC.jl

Adaptive NUTS warm-up for linear transformations and step size.

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

## Checkpoints, Callbacks and Resume

Warm-up has two checkpoint boundaries: **CP-0**, right after initialization
(Pathfinder), and **CP-N**, after each outer warm-up window. Three opt-in,
independent mechanisms hang off these boundaries. All default to off, and with
them off the run is byte-for-byte identical to the plain sampler.

**Observing progress.** `callback=(state, stage) -> should_stop` fires at each
boundary with `stage ∈ (:init, :window)`. It may read `state` and request an
early stop by returning `true`, but it is *observational* — it must not mutate
`state`:

```julia
result = adaptive_warmup_mcmc(rng, lpdf; callback=(state, stage) -> begin
    @info "boundary" stage stepsize=state.stepsize
    false  # keep going
end)
```

**Writing checkpoints.** `checkpoint_dir=path` serializes a resumable snapshot at
each boundary as `cp_init.jls`, `cp_window_<n>.jls`, and an overwritten
`cp_latest.jls`. The multi-chain method writes chain `i` under `path/chain_<i>/`:

```julia
result = adaptive_warmup_mcmc(rng, lpdf; checkpoint_dir="checkpoints/")
```

The snapshot deliberately excludes the (possibly non-serializable) inner problem
and stores the reparametrizer only as its scalar centering values.

**Resuming.** [`resume_warmup_mcmc`](@ref) re-supplies `lpdf` and continues from a
checkpoint. For a fixed seed the result is identical to an uninterrupted run:

```julia
# single chain: point at a specific checkpoint file
result = resume_warmup_mcmc(lpdf, "checkpoints/cp_latest.jls")

# multi-chain: point at the parent directory; chain i resumes from chain_<i>/
results = resume_warmup_mcmc(lpdfs, "checkpoints/")
```

Because the checkpoint does not carry the inner problem, the `lpdf` you pass to
`resume_warmup_mcmc` must be reconstructed by you — typically the same way you
built it for the original call.

## See Also

- [Stan's warm-up documentation](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)
- [nutpie](https://github.com/pymc-devs/nutpie) -- Python NUTS sampler with similar ideas
- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) -- Julia HMC implementation
