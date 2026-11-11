# Streams and transitions

The signatures below use placeholder arguments and are not standalone programs.
The [changing-target example](../examples/changing-target.md) executes the fixed-kernel interface.

## Persistent streams

```@docs
WarmupHMC.stream_mcmc
WarmupHMC.open_stream
WarmupHMC.close_stream
```

## Fixed kernel

```@docs
WarmupHMC.FixedKernelNUTSState
WarmupHMC.fixed_kernel_nuts_step
```

## Continuous adaptation

```@docs
WarmupHMC.ContinuousNUTSState
WarmupHMC.continuous_nuts_step
WarmupHMC.reposition!
```
