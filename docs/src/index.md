# WarmupHMC.jl

Adaptive NUTS warm-up for linear transformations and step size.

## Motivation

A previous approach to do automatic non-linear reparametrizations using MCMC draws has been found lacking, mainly for two reasons:

* **High start-up cost**: Especially in the early stages of warm-up, a lot of effort is spent to get just a handful of MCMC draws. These are usually strongly correlated, and are thus usually less than helpful in learning the linear transformation, and even less so for the non-linear transformation.
* **Limited convergence due to statistical "fluctuations"**: Even in the later stages of warm-up, the number of MCMC draws will inherently be limited. If there are many linear and non-linear parameters that have to be learned, the inherent fluctuations in estimating the "ideal" parameters may be so large that the linear and/or non-linear transformation are not helpful.

## Method

WarmupHMC adaptively:

* Learns a linear transformation of the posterior that simplifies MCMC sampling,
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

## See Also

- [Stan's warm-up documentation](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)
- [nutpie](https://github.com/pymc-devs/nutpie) -- Python NUTS sampler with similar ideas
- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) -- Julia HMC implementation
