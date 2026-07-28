# API Reference

## Sampling

```@docs
WarmupHMC.adaptive_warmup_mcmc
WarmupHMC.cooperative_warmup_mcmc
WarmupHMC.clustered_warmup_mcmc
WarmupHMC.resume_warmup_mcmc
```

## Reparametrizations

```@docs
WarmupHMC.ReparametrizedProblem
WarmupHMC.IndexedReparametrization
WarmupHMC.PartiallyCentered
WarmupHMC.Reparametrization
```

## Internals

Not part of the public API — no compatibility is promised for anything below.
They are listed because the exported docstrings above refer to them.

```@autodocs
Modules = [WarmupHMC]
Public = false
```
