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
WarmupHMC.Reparametrization
WarmupHMC.IndexedReparametrization
WarmupHMC.ReparametrizedProblem
```

## Adaptation rules

```@docs
WarmupHMC.CandidateScoringPlan
WarmupHMC.candidate_scoring_losses
WarmupHMC.WindowSelectionPlan
```
