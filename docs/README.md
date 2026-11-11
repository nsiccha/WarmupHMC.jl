# Manual structure

The manual has four navigation groups. This list owns its scope; `make.jl`
owns the order and `lint.jl` checks their agreement.

| Group | Page | Purpose |
| --- | --- | --- |
| Start | `index.md` | Install the package and read a complete sampling call and its result. |
| Guides | `guides/sampling.md` | Choose an initializer, run chains, and interpret returned draws. |
| Guides | `guides/checkpoints.md` | Save, stop, and resume adaptive sampling. |
| Guides | `guides/reparametrization.md` | Specify partial centering and its derivative backend. |
| Guides | `guides/changing-targets.md` | Choose fixed or continuously adapting transitions. |
| Examples | `examples/gaussian.md` | Compare a cached Gaussian run with analytic moments and inspect its diagnostics. |
| Examples | `examples/changing-target.md` | Follow a moving Gaussian target with fixed-kernel transitions and inspect the state sequence. |
| Reference | `reference/sampling.md` | Document adaptive samplers and checkpoint continuation. |
| Reference | `reference/streams.md` | Document streams and individual transitions. |
| Reference | `reference/transformations.md` | Document scales, moments, reparametrizations, and selection plans. |
| Reference | `reference/compatibility.md` | Define the public compatibility boundary. |

## Build

Install dependencies with `.github/scripts/install-docs-deps.sh`, then run
`julia --project=docs docs/make.jl`. The build treats warnings as errors, checks
all exports, lints the page structure, verifies the cached example receipt, and
generates its figure from the saved summary.

The sampler runs outside the documentation build. `bench/docs/run.jl` executes
the exact displayed program in `bench/docs/gaussian.jl` and writes a small JSON
summary and receipt. It never saves draws. The build displays that program with
its recorded output; subsequent computations and plots execute during the build.
CI must not regenerate this cache. See `bench/docs/README.md`.

Research drivers live under `bench/`. Historical benchmark dumps, the evidence
catalogue, and dashboard recordings are excluded from the manual.
