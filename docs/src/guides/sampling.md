# Sampling

Choose an initial state, run independent chains, and interpret the returned
positions without confusing them with warm-up state.

## Target

The installation command on [Start](../index.md) adds the checked-out revision
and its unregistered dependencies to Julia's active environment. Use
`julia --project=path/to/your/environment WarmupHMC.jl/scripts/install.jl` to
choose a project explicitly. The script reads dependency revisions from the
package's `Project.toml` rather than relying on Julia's transitive handling of
`[sources]` tables.

A target implements `LogDensityProblems.dimension`, `capabilities`, `logdensity`,
and `logdensity_and_gradient`. The last method returns a scalar log density and
its gradient in the same coordinates. Additive constants in the density cancel.
The [Gaussian example](../examples/gaussian.md) supplies an analytic gradient;
an automatic differentiation wrapper can provide the same interface.

The dimension is the length of the unconstrained parameter vector. Transform
constrained parameters before sampling and include the corresponding Jacobian
in the log density. A finite starting log density and gradient are required.

## Initialization

`adaptive_warmup_mcmc(rng, problem; ...)` uses Pathfinder by default. Use `init`
to supply a known position or a position with a known scale. A scale is an
operator mapping a standard normal vector into a displacement in parameter
space; it is not a covariance matrix. See the `init` argument in
[`adaptive_warmup_mcmc`](@ref) for the accepted forms and stationary-start
controls.

Request retained draws with `n_draws`. Warm-up may restart and discard an earlier
sampling epoch; the returned positions belong to the retained epoch. Treat the
requested count as a floor, and inspect the actual matrix shape. Set
`progress=nothing` in batch scripts and `monitor_ess=true` when the run should
compute ESS without a progress display. The random number generator is explicit,
which allows independent seeds and avoids reliance on the process-global RNG.

## Independent chains

The vector-of-RNGs overload runs independent adaptive chains and returns a vector
of results. Give each chain its own RNG. A shared problem is copied where needed
for adaptation; an explicitly supplied vector of problems remains the caller's
responsibility to make independent. Mutable reparametrization state must not be
shared between concurrently running chains.

[`cooperative_warmup_mcmc`](@ref) instead shares adaptation evidence across chains.
[`completion_warmup_mcmc`](@ref) adds a completion policy to independent chains.
These are different sampling policies, not interchangeable ways to display a
progress bar. In particular, an early completion rule can select chains by
runtime; assess that selection when runtime depends on posterior location. The
completion sampler warns, and records the caveat in
`completion.selection_warning`, only when its returned chains are such a
selection. Requiring every chain, the default `min_completed`, never selects.

## Returned coordinates

`posterior_position` stores coordinates in rows and draws in columns. For a
`ReparametrizedProblem`, these positions have already been transformed into the
wrapped model's coordinates. Do not transform them again. `posterior_gradient`,
`halo_position`, `halo_gradient`, and the saved active position retain working
coordinates, so they cannot be paired with returned model-frame positions as
though they shared a frame. The reported `ess` belongs to the returned positions.

Use `n_divergent_samples` and effective sample size to inspect the retained
sample. Compare multiple independently initialized chains for convergence;
single-chain ESS cannot establish agreement between chains or discovery of
separated modes. `sampling_evaluation_counter` counts gradient evaluations for
retained sampling transitions. `total_evaluation_counter` also includes work
from discarded epochs and adaptation, but excludes Pathfinder and initial
step-size search. Neither counter is elapsed time.

The [Gaussian calibration example](../examples/gaussian.md) reads the matrix
shape, moments, ESS, divergence count, and evaluation counters from one run.
