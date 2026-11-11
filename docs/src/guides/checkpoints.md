# Checkpoints

Save an adaptive run and continue it with a reconstructed target while preserving
the retained draws and learned sampler state.

## Continuation

Pass `checkpoint_dir` to save the run. On a later call, use the same directory
and `resume=true`, supply the same target and configuration, and increase
`n_draws` to extend the retained sample. The restored RNG state drives the
continuation; a new RNG argument does not reseed an existing checkpoint.

This program uses the `Gaussian` target from [Start](../index.md). Sampling runs
outside the docs build; the displayed program and result share a checked receipt.

```@eval
using WarmupHMC
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl"))
ManualCache.program("checkpoint")
```

```@example checkpoints
using WarmupHMC
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl")) # hide
c = ManualCache.load_summary()["checkpoint"] # hide
(; before=c["before"], after=c["after"], prefix_preserved=c["prefix_preserved"]) # hide
```

`prefix_preserved` compares the continued result's leading columns with the
original result. The temporary directory is removed after this demonstration;
use a persistent directory for a real run.

## Compatibility

Checkpoints do not serialize the target's model definition or data. Reconstruct
the same target before restoring. A dimension check cannot detect changed data
or a changed likelihood, so keep that identity with the checkpoint yourself.
The schema and sampler kind must be compatible. Reconstruct a custom scoring or
selection plan when the original run used one; adding or removing a plan on
restore is rejected. Omit `recording_target` to inherit the saved recorder
capacity. Supplying a different capacity is rejected.

Use `resume_warmup_mcmc` when a specific checkpoint path is already available.
The primary entry point's `resume=true` form is useful when the caller manages a
checkpoint directory. Consult the [sampling reference](../reference/sampling.md)
for the path layouts and options inherited from the saved state.

## Interruption

`stop_requested` is a cheap predicate polled at safe points. With parallel
chains it may run concurrently, so any shared state it reads must be thread-safe.
`checkpoint_interval` requests periodic checkpoints within a window. Neither
option interrupts initialization, a transition, window finalization, or I/O;
an interval is not a bound on checkpoint age, and a deadline predicate is not a
hard process deadline. Work after the last completed save must be repeated after
a crash.

The `callback` option observes initialization and window boundaries. Treat its
state as read-only. It is not a substitute for the finer-grained stop predicate.

The [Gaussian calibration example](../examples/gaussian.md) carries the receipt
for this continuation and explains which diagnostics describe retained draws.
