# Transformations

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
