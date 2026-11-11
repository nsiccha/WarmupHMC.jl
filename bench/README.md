# Research and regeneration

These programs are outside the reader-facing manual. They run experiments,
compare sampler variants, or investigate implementation costs. They are not
executed by the documentation build.

| Directory or driver | Purpose |
| --- | --- |
| `docs/` | Generate the manual's small Gaussian summary and content receipt. |
| `research/` | Reparametrization, restart, weighting, and derivative experiments. |
| `sampler_comparison.jl` | Compare sampler costs under matched targets. |
| `continuous_nuts_saem.jl`, `changing_target_transition.jl` | Study changing-target transitions. |
| `moment_scale.jl` | Check weighted Gaussian moment approximation. |
| `horseshoe_divergences.jl`, `stationary_start.jl` | Investigate initialization and divergence behaviour. |

Run a driver with its own project, for example
`julia --project=bench/research bench/research/run_reparam_benchmark.jl`.
Read the driver's header for its inputs and environment variables. Research
outputs belong in ignored `bench/research/results/` or an external directory set
with `WHMC_BENCH_OUT`; the web application's optional benchmark routes read that
same location. A fresh checkout has no research measurements.

Historical dumps and their prose interpretation were removed from the current
tree. The original files remain in Git history before this manual rewrite.
Regenerate a measurement against the source being investigated before drawing a
new performance conclusion. The manual makes no cross-sampler speed claim.
