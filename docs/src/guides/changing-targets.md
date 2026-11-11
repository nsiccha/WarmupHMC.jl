# Changing targets

Advance a NUTS chain under successive target densities while choosing whether
the kernel remains fixed or continues to adapt.

## Fixed kernel

Construct a `FixedKernelNUTSState` from a warm-up checkpoint or from an RNG and
position with a scale operator and step size. `fixed_kernel_nuts_step` evaluates
the supplied target at the saved position before taking the transition. Carry
its returned `state` into the next call. The target may change, but its dimension
and coordinate frame must remain compatible with the state.

A transition is not a posterior fit or a convergence check. A fixed
kernel preserves its scale, step size, and tree-depth limit across transitions.
When the target remains fixed and draws should be persisted, `stream_mcmc`
provides a disk-backed alternative. Release its mappings with `close_stream`
after copying needed values and before deleting its files.

## Continuous adaptation

`ContinuousNUTSState` and `continuous_nuts_step` carry decaying position-gradient
evidence and update the metric and step size after each transition. Use
`adapt=false` to take a transition with the current kernel held fixed.
`reposition!` changes the saved position; the next transition reevaluates that
position under the supplied target.

A saved position outside the new target's support is rejected. Leaving support
during a leapfrog trajectory is handled as a divergence. Positions collected
under different targets do not form one stationary posterior sample, so a pooled
mean or ESS of that sequence does not describe a single posterior.

See the [streams and transitions reference](../reference/streams.md) for state
construction and return fields. The [changing Gaussian example](../examples/changing-target.md)
executes a sequence of fixed-kernel transitions and shows its positions and diagnostics.
