# Adaptive sampling

The signatures below use placeholder arguments and are not standalone programs.
The [Gaussian example](../examples/gaussian.md) supplies a complete target and run.

Arguments, return fields, initialization, and checkpoint controls for the
adaptive entry points. For a worked call, see
[Gaussian calibration](../examples/gaussian.md).

## Independent chains

```@docs
WarmupHMC.adaptive_warmup_mcmc
WarmupHMC.resume_warmup_mcmc
WarmupHMC.rescore_warmup_checkpoint
```

## Cooperative chains

```@docs
WarmupHMC.cooperative_warmup_mcmc
```

## Completion policy

```@docs
WarmupHMC.completion_warmup_mcmc
```
