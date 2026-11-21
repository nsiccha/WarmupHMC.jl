# Sampling

Choose an initial state, run independent chains, and interpret the returned
positions without confusing them with warm-up state.

## Target

The installation command on [Start](../index.md) adds the checked-out revision
and the dependency revisions its `Project.toml` pins under `[sources]`, our
Pathfinder fork among them (see Initialization), to Julia's active environment. Use
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

The log density and gradient must depend on the position alone. A target that
keeps work buffers between evaluations must not let one evaluation, for example
a `NaN` at an extreme leapfrog point, change its later answers. At every warm-up
window boundary of `adaptive_warmup_mcmc`, and at every round barrier of
`cooperative_warmup_mcmc`, the sampler evaluates each chain's current point
once more. It raises an error when the target no longer answers finite there,
because every transition from that point would diverge whatever the step size.

## Initialization

`adaptive_warmup_mcmc(rng, problem; ...)` uses Pathfinder by default. Use `init`
to supply a known position or a position with a known scale. A scale is an
operator mapping a standard normal vector into a displacement in parameter
space; it is not a covariance matrix. See the `init` argument in
[`adaptive_warmup_mcmc`](@ref) for the accepted forms and stationary-start
controls.

A Pathfinder release keeps its whole L-BFGS trajectory, the Gaussian fitted at
every iterate and their ELBO draws until it returns, about 34 words per
coordinate and iterate, so the start's memory grows with `maxiters` in
`pathfinder_kw`. WarmupHMC pins the fork `nsiccha/Pathfinder.jl` (branch
`save-trace-false`, proposed upstream as mlcolab/Pathfinder.jl#437), with which
Pathfinder keeps only the fit it selects: the start then holds memory
proportional to the dimension, whatever `maxiters` is. Julia does not apply a
dependency's `[sources]` to your environment, so the fork reaches it through the
installation script or your own `[sources]` entry; with a Pathfinder release the
start keeps the whole trajectory, as before.

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
The completion sampler also leaves out chains that are stuck: a chain whose
step size collapses far below its siblings' is stopped while it is still
warming up, and a finished chain must pass its step-size, divergence and mass
checks to count. WarmupHMC does not move such a chain back; rerun it from
another start if it is needed.

### Calibrating a log-mass cutoff

The completion sampler compares a finished chain with the best eligible
sibling at the time of admission. To see how a different cutoff would treat a
recorded comparison, take the four fields below from an entry in
`outcome.completion.chains`. Skip entries with a missing estimate or reference
(`nothing`). Here the reference scores 20 nats higher, and the gap exceeds
four combined standard errors.

```@example admission
chain = (log_mass=-120.0, reference_log_mass=-100.0,
         log_mass_se=1.2, reference_log_mass_se=1.6)
gap = chain.reference_log_mass - chain.log_mass
four_se = 4 * hypot(chain.log_mass_se, chain.reference_log_mass_se)
for cutoff in (25, 15)
    low_mass = gap > cutoff && gap > four_se
    println("cutoff = $cutoff nats, low_mass = $low_mass")
end
```

The default cutoff of 25 nats admits this chain on the mass check. At 15 nats
the same comparison would exclude it. To choose a cutoff, check what it does
to modes whose weights you know independently. For example, consider two
well-separated Gaussian components of equal weight in 226 dimensions. Both
have unit marginal variances. One has independent coordinates and the other
has correlation 0.55 within each of 113 independent pairs. The score adds the
mean log density to the sum of the log marginal standard deviations. The
marginal terms agree, but the correlated component's mean log density is
higher by `-113 * log(1 - 0.55^2) / 2`, about 20.35 nats. A 15-nat cutoff
can therefore exclude half the target mass even with very precise estimates.

Choose a cutoff that preserves known modes and flags chains independently
identified as trapped, for example by comparison with a reference fit.
If the evidence on your target supports a tighter cutoff, pass it as
`max_admitted_log_mass_gap` to a fresh `completion_warmup_mcmc` run. Use a
new checkpoint directory if saving the fit. Changing admission can change
which siblings finish and which comparisons take place. Record how many chains
passed and whether they reached the requested quorum. Check maximum R-hat,
minimum bulk and tail ESS, and the count and percentage of divergent draws
across the retained chains. If the run throws a
`WarmupHMC.CompletionQuorumError`, its `outcome.results` holds the passing chains.

## Cooperative draws from conditional targets

Pass one target to share adaptation across chains sampling the same posterior.
Pass an ordered vector of targets to sample a family of conditionals. For
example, if `targets[k]` conditions on upstream draw `k`, supply its start
at `starts[k]`. Calling
`out = cooperative_warmup_mcmc(rng, targets; init=starts, n_draws=1)`
requests one draw per target. Its column in `out.results[k].posterior_position`
belongs to upstream draw `k`. Check
`out.stop_reason` and the number of columns in each matrix before using the
draws in a downstream calculation. An evaluation budget can expire before
every target qualifies for its first draw. Dropping an empty matrix would also
drop its upstream draw and change the weighting of the calculation.

A target that no chain can sample, even after one cold replacement, ends with
`out.results[k].status === :recovery_failed` and no draw. The other targets
still sample, and the run then returns `stop_reason === :recovery_failed`.
Each failed result lists its attempts in `abandoned`, with the reason each one
was given up. Find them with
`findall(r -> r.status === :recovery_failed, out.results)`, fit each such
target on its own, for example with
`adaptive_warmup_mcmc(rng, targets[k]; init=starts[k], n_draws=100)`, and check
that fit before you use its draw.

With `features` (one column per target), first-wave donors take 30 pilot
transitions before seeding nearby chains. Each chain checks warm-up drift and
rescores up to eight nearby endpoints on its own target. A large discrepancy
triggers further adaptation. Unavailable peers or persistent disagreement
trigger an independently tuned check. Production begins on the next transition
after qualification. Each slot reports `metric_reviews`, `independent_probes`,
`peer_evaluations` and `probe_evaluations` in `accounting`.
Pooled geometry uses variation within each conditional, so
moving their means farther apart does not by itself increase the estimated
scale. Assess each conditional against draws or known quantities for that
conditional. The variation across upstream draws also contributes to the
spread of the combined sample.

## Returned coordinates

`posterior_position` stores coordinates in rows and draws in columns. For a
`ReparametrizedProblem`, these positions have already been transformed into the
wrapped model's coordinates. Do not transform them again. `posterior_log_density`
is the wrapped model's own log density at those positions, without the
reparametrization's log-Jacobian. `posterior_gradient`,
`halo_position`, `halo_gradient`, and the saved active position retain working
coordinates, so they cannot be paired with returned model-frame positions as
though they shared a frame. The reported `ess` belongs to the returned positions.

Use `n_divergent_samples` and effective sample size to inspect the retained
sample. `n_nonfinite_leaves` counts the run's leapfrog points where the target
answered `+Inf` or `NaN`, or a finite log density with a non-finite gradient.
The sampler treats such a point as a divergence, like a point outside the
support (`-Inf`), but it is a defect of the target, such as an underflow, so
the run also warns at the first one. Compare multiple independently initialized chains for convergence;
single-chain ESS cannot establish agreement between chains or discovery of
separated modes. `sampling_evaluation_counter` counts gradient evaluations for
retained sampling transitions. `total_evaluation_counter` also includes work
from discarded epochs and adaptation, but excludes Pathfinder and initial
step-size search. Neither counter is elapsed time.

The [Gaussian calibration example](../examples/gaussian.md) reads the matrix
shape, moments, ESS, divergence count, and evaluation counters from one run.
