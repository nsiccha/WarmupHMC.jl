# Transformations

The signatures below use placeholder arguments and are not standalone programs.
The [reparametrization guide](../guides/reparametrization.md) executes the constructors.

## Scale operators

```@docs
WarmupHMC.scale_data
WarmupHMC.scale_from_data
WarmupHMC.fit_gaussian_moments
```

## Partial centering

`PartiallyCentered` is re-exported from ReparametrizableDistributions.

### [PartiallyCentered](@id PartiallyCentered)

```@eval
using Markdown, WarmupHMC
Markdown.parse(string(@doc WarmupHMC.PartiallyCentered))
```

```@docs
WarmupHMC.ScaleCentered
WarmupHMC.Reparametrization
WarmupHMC.SharedAccessor
WarmupHMC.IndexedReparametrization
WarmupHMC.MarkovChain
WarmupHMC.ReparametrizedProblem
```

## Adaptation rules

```@docs
WarmupHMC.CandidateScoringPlan
WarmupHMC.candidate_scoring_losses
WarmupHMC.WindowSelectionPlan
```

## Continuous centering and diagonal adaptation

Pass `centering_schedule=:continuous` to `completion_warmup_mcmc` or
`adaptive_warmup_mcmc`. On each adapting tree, the sampler uses the NUTS
proposal probabilities to fit a diagonal inverse mass and update eligible
scalar centering controls for the next tree. Each control can take any value
in `[0, 1]`. A plain target, or a reparametrized target with
`nonlinear_adapt=false`, goes directly through the diagonal update.
The default schedule is `:window`.

For an adjustable effect, write `x` and `g` for its candidate position and
gradient, and `a` and `b` for their estimated means. A stochastic
Gauss–Newton step fits the control, both means and the log inverse mass
`r` to minimize the weighted mean of
`((x-a)^2 * exp(-r) + (g-b)^2 * exp(r)) / 2`. The inverse mass entry is
`exp(r)`. The ordinary diagonal estimator uses the standard deviation of
position divided by the standard deviation of gradient, regularized by the
starting scale. Dual averaging carries its count and acceptance history
across updates. Its log step size and running average shift with the geometric
mean change in the diagonal scale factor.

A `CandidateScoringPlan` scores real-valued `PartiallyCentered(c)`. The
direct scalar map has an analytic derivative. Every adapting tree discounts
earlier observations with a 20-tree half-life. Each installation transports
the current physical point and refreshes its full gradient. Shared-parent
gradients include the Jacobian term. Invariant marginals keep their discounted
moments. Changed marginals carry those moments through an affine map estimated
from paired positions and gradients. Dependent cells carry their curvature
through the corresponding old and new Cholesky factors. This transport is
approximate after the first tree. Up to 16 initializer representatives and 16
recent NUTS proposals calibrate it. All trees contribute to the running fit.

Pass `centering_initial_ensemble=(; positions, gradients, source_controls)`.
Columns of both matrices are in starting source coordinates, shared with
`init.position`. Supply full transformed log-density gradients and centeredness
in `ir.pairs` order (empty for a plain target). Before tree 1 the sampler fits
each control on all columns, profiles out both means and the diagonal scale,
and transports every column into the installed frame. Reusing gradients costs
no target calls. Each changed current point needs one fresh gradient.

With `stepsize_adaptation_limit=L`, trees 1 through `L-1` adapt and tree
`L` finalizes step size. Retained draws use the frozen kernel. Results count
`initializer_rows` and `adaptation_trees` separately. Completion counts actual
`gradient_calls`. Checkpoints restore the fit without re-consuming cached rows.

Each chain returns `h = result.centering_handoff`. Rebuild the same model with
`h.ensemble.source_controls` at `source_indices`. An initializer can use
`h.init`, its diagonal `squared_scale`, and the paired source-frame
`h.ensemble.positions` and `gradients`. Another warm-up accepts
`init=h.init, centering_initial_ensemble=h.ensemble`. The handoff also records
`stepsize` and `frozen`. Before any retained draw it contains the current
point. Custom fixed frames expose native `reparam_sources` and may lack a
portable ensemble.

## Centering between NUTS trees

Use `centering_schedule=:online` in `completion_warmup_mcmc` or
`adaptive_warmup_mcmc` for per-chain updates during step-size adaptation.
The default `:window` updates at restarting linear windows.

Each tree contributes one group with the exact NUTS proposal probabilities,
including the root. A failed doubling contributes zero. The online schedule
uses `nonlinear_evidence=:nuts_weighted`. Its `CandidateScoringPlan` reads
borrowed node positions and gradients during traversal, including speculative
nodes. `nonlinear_trajectory_weighting=:stepsize` weights groups by step size.

After three groups, a finite winner must persist for `centering_persistence`
consecutive checks (default 3). An accepted change transports the halo,
refreshes the current gradient, resets the active metric to identity and
restarts dual averaging. All candidate histories and streaks restart because
a parent's source can alter a child's score. A rejected change preserves them.

`centering_max_updates` optionally caps installations. Centering freezes at
the cap or at the start of terminal step-size tuning. The positive
`centering_terminal_buffer` reserves 16 full dual-averaging updates by
default. With a step-size adaptation limit of 96 and one uninterrupted
window, the last eligible installation is after tree 79. Trees 80 through 95
tune the final frame, and tree 96 finalizes the step size. With a limit of
17 or fewer, the default keeps the initial controls from the start. Callers
can reserve fewer updates explicitly. The buffer uses the existing budget.

Each online chain reports `centering_terminal_buffer`, the installation
transition numbers in `centering_installations` and the actual freeze boundary
in `centering_frozen_after`. An interrupted run can report `nothing` for the freeze.
Completion's `gradient_calls` includes frame refreshes.

Use the direct scorer or a `CandidateScoringPlan` with indexed constant
`PartiallyCentered` sources. Whole-window selectors, Markov-chain controls
and two-pass scale centering use `:window`. Resume with the same target,
scoring plan and sampler configuration. Omitting the schedule and its
stabilization settings restores them from the checkpoint.
