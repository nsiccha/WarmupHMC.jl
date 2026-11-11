# Compatibility

The public interface comprises exported names, their documented behavior, the
accepted keywords of `adaptive_warmup_mcmc`, and the names and order of its
returned `NamedTuple` fields. Changes to these contracts require the package's
breaking-change policy. `test/public_api.jl` checks the exported surface and the
accepted keyword list.

Numerical outputs are not fixed across releases: draws, adaptation paths,
evaluation counts, and learned step sizes may change for the same seed.
Returned field types, including array representation and lazy versus materialized
storage, are not covered by the field-name contract. Dependency interfaces and
unexported `WarmupHMC` names are outside this public boundary.

Checkpoint compatibility is also checked at restoration. Preserve the original
model inputs, configuration, and dependency environment when a run must remain
resumable. See [Checkpoints](../guides/checkpoints.md).
