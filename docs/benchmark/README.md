# Nonlinear-reparametrization benchmark

Measures WarmupHMC's adaptive partial centering against the fixed centering
endpoints on hierarchical targets, and measures what the transform costs on the
gradient hot path.

The question: **on a model written in its centered parametrization, does the
adaptive method reach noncentered-like performance without being told to?**

`RESULTS.md` has the answer and the numbers. This file is how to reproduce them.

## Rerun it

```bash
# from the repository root
julia --project=docs/benchmark docs/benchmark/run_reparam_benchmark.jl
```

Knobs, all optional:

| variable | default | meaning |
|---|---|---|
| `WHMC_BENCH_SEEDS` | `8` | pinned seeds per (target, arm) — seeds are `1:N` |
| `WHMC_BENCH_DRAWS` | `1000` | `n_draws` floor per run |
| `WHMC_BENCH_TARGETS` | all | comma-separated posteriordb names, or `funnel` |
| `WHMC_BENCH_OUT` | `docs/benchmark/results` | output directory |

A quick pass over one cheap target:

```bash
WHMC_BENCH_SEEDS=2 WHMC_BENCH_DRAWS=200 \
  WHMC_BENCH_TARGETS=eight_schools-eight_schools_centered,funnel \
  julia --project=docs/benchmark docs/benchmark/run_reparam_benchmark.jl
```

## Environment

`Project.toml` is checked in; `Manifest.toml` is not (the repository gitignores
every manifest and resolves at startup instead). Resolve this one the same way
every other environment in the fleet is resolved — through the canonical
`resolve_script` generator, which develops the host-local checkouts rather than
cloning anything:

```bash
cd ~/github/nsiccha/Claude
source lib-repos.sh && source lib-resolve.sh
julia --project=<repo>/docs/benchmark \
  -e "$(resolve_script <repo>/docs/benchmark WarmupHMC <repo>)"
```

A bare `Pkg.instantiate()` does **not** work here: `WarmupHMC` depends on
`Treebars`, which is unregistered, and the published copy is behind the local
one.

`HTMX`, `HTMXObjects` and `DynamicObjects` appear in `[deps]` because the
resolve develops them; nothing in this benchmark uses them directly.

## What is reproducible, and what is not

- **Reproducible**: for a fixed seed, a rerun on the same machine reproduces the
  draws. `common.jl` sets `BLAS.set_num_threads(1)` at load — without it the
  transformation update's multithreaded reduction order is not deterministic and
  a fixed seed no longer pins the result.
- **Not portable**: wall-clock and therefore ESS/second are machine-specific.
  ESS per gradient evaluation is the metric to compare across machines; it is
  reported alongside, and the results table leads with it for that reason.
- **Chaotic, not systematic**: `plain` and `fixed_centered` are mathematically
  the same sampler — the wrapper's transform is an exact identity when source
  and target centerings agree. They still diverge on Stan targets, because the
  wrapper adds an AD-computed correction term that is zero only up to rounding,
  and NUTS amplifies a 1e-16 gradient difference into a different trajectory.
  On the analytic funnel, where the correction is exactly zero, the two arms
  agree to the last gradient evaluation — that agreement is the check that the
  no-op wrapper really is a no-op.

## Layout

| path | what |
|---|---|
| `common.jl` | targets, arm construction, metrics, gradient-overhead timing |
| `run_reparam_benchmark.jl` | driver — runs everything, writes `results/`, prints the table |
| `results/runs.json` | one record per run, every measurement kept |
| `results/gradient_overhead.json` | per-call cost of the transform on the gradient path |
| `RESULTS.md` | the write-up |

## Prior art

`docs/benchmark/common.jl` and `index.qmd` existed once on the abandoned
`cooperative-clusters` branch (`0937482`, Sep 2025). That harness covered
step-size and metric adaptation on Gaussian targets only — nothing hierarchical,
no reparametrization — and imports internals (`BayesianOptimizationStepsizeAdaptation`,
`IPGPRegression`, …) that no longer exist. It was mined for shape, not extended.
