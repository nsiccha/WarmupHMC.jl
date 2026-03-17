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
parametrizations.

During warmup, the system tries multiple candidate centering values and selects the one
minimizing a correlation-based loss. This happens at the end of each warmup window using
the accumulated intermediate positions and gradients.

To use reparametrizations, wrap your log density problem in a
[`ReparametrizedProblem`](@ref) with an [`IndexedReparametrization`](@ref):

```julia
using WarmupHMC, DifferentiationInterface, Mooncake

# Eight schools example: dims 1:8 are group effects, dim 9 = location, dim 10 = log-scale
ir = IndexedReparametrization(
    1:8 .=> Ref(Reparametrization(
        PartiallyCentered(1.0), PartiallyCentered(1.0),
        x -> x[9], x -> x[10]
    ))
)
rp = ReparametrizedProblem(ir, my_problem, AutoMooncake())
result = adaptive_warmup_mcmc(rng, rp)
```

Reparametrization is controlled by the `nonlinear_adapt` keyword (default `true`).
Posterior samples are automatically transformed back to the original parametrization.

## See Also

- [Stan's warm-up documentation](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)
- [nutpie](https://github.com/pymc-devs/nutpie) -- Python NUTS sampler with similar ideas
- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) -- Julia HMC implementation
