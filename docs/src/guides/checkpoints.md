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

## Re-scoring before continuation

Ordinary resume restores cached values and gradients. To check and refresh an
adaptive checkpoint on another CPU, call
`WarmupHMC.rescore_warmup_checkpoint(lpdf, payload; rtol=1e-10)` explicitly.
It returns an independent payload; serialize that payload to a new path and
pass the path to `resume_warmup_mcmc`. For a checkpoint nested inside another
sampler, replace the nested payload with the returned value through that
sampler's supported checkpoint interface.

Supply the same target, data and reparametrization structure. Saved controls
are restored on a private reparametrizer; the caller's controls are unchanged.
The operation refreshes the live log density and gradient, retained posterior
and halo gradients, and occupied nonlinear leaf-sample gradients. Every stored
value or component compared must satisfy
`abs(new - old) <= rtol * max(abs(old), 1)`, and all evaluated positions,
values and gradients must be finite. A finite nonnegative `rtol` is required;
zero requires numerical equality. Excessive drift reports the largest scaled
deviation and its field, column and coordinate. The input remains unchanged
on both success and failure.

Random streams, sampler counters, the current window, metric and step size
are preserved. Accumulated moments, dual averaging and candidate losses are
also preserved: the full history needed to rebuild them is no longer retained.
Dropped-epoch snapshots remain unchanged in their original coordinates.
Consequently this provides tolerance-checked continuation, with no guarantee
of the same trajectory across CPUs. Passing these checks also cannot establish
that two arbitrary model definitions or datasets are identical.

The returned `rescored` audit records the policy, tolerance, number of target
evaluations, changed-component counts, maximum absolute and scaled deviations,
and the location of the largest scaled deviation. Later checkpoints preserve
the audit. Its evaluations are separate from sampler cost counters. Work is
one target evaluation per retained point, linear vector work in the stored
coordinates, and an independent copy of the payload.

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
