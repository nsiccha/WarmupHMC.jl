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
