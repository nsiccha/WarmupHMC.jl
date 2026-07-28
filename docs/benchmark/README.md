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

**AD backend — under review.** The harness passes `AutoForwardDiff()` to
`ReparametrizedProblem` at three sites (`common.jl:234`, `:309`, `:310`),
matching the shipped consumer. That is forward mode over an objective that is
scalar in `d` inputs, which the package's own docstring
(`src/Reparametrizations.jl:50-51`) notes is the case reverse mode handles
better — and the standing instruction is DifferentiationInterface with
**Enzyme**, never Mooncake or ForwardDiff. The backend already arrives as an
ADTypes object through DI, so changing it here is one argument in three places;
doing it *consistently* is a WarmupHMC-wide change, not a benchmark-local one.
Until that lands, every overhead and wall-clock number in `RESULTS.md` is
ForwardDiff-specific — the per-gradient results are not.

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
  and target centerings agree. They agree to the last gradient evaluation on
  every seed for `seeds_data` and the funnel, and diverge on eight_schools and
  both radon models. The predictor is whether the spec's **location** is a
  constant: it is for those two targets and a closure over the position vector
  for the rest, and reconstructing `loc + exp(s)·((x − loc)/exp(s))` is the
  identity only up to rounding once `loc` moves with the position. NUTS then
  amplifies a 1e-16 gradient difference into a different trajectory. The exact
  agreement where the arithmetic permits it is the check that the no-op wrapper
  really is a no-op; elsewhere the two arms stay within seed noise.

## Always record the SHA you measured on

`RESULTS.md` names the WarmupHMC commit its numbers were produced on, in the
lede. This is not bookkeeping. The first full run of this benchmark — 184 runs,
every arm — was measured against `34ce034`, a halo-recording regression that
collapsed the pool the metric adaptation reads from, and which was fixed a few
hours later. Nothing in the numbers looked wrong, and nothing could have flagged
it, because the affected code path is upstream of every arm including the
unwrapped control. **A measurement with no revision attached is a measurement
that rots silently.** The `results/before` and `results/after` split exists so
that episode stays visible rather than being overwritten.

When you rerun, write to a new directory rather than over an old one:

```bash
WHMC_BENCH_OUT=docs/benchmark/results/<name> \
  julia --project=docs/benchmark docs/benchmark/run_reparam_benchmark.jl
julia --project=docs/benchmark docs/benchmark/compare.jl \
  docs/benchmark/results/<old> docs/benchmark/results/<name>
```

`compare.jl` reports ESS per gradient evaluation and gradient spend per arm, the
count of seeds whose adaptation never moved, and where the centering settled. It
deliberately omits wall-clock: two runs that compiled different package sources
are not comparable in seconds.

## Layout

| path | what |
|---|---|
| `common.jl` | targets, arm construction, metrics, gradient-overhead timing |
| `run_reparam_benchmark.jl` | driver — runs everything, writes a results dir, prints the table |
| `summarize.jl` | regenerates `RESULTS.md`'s tables from one results dir |
| `compare.jl` | before/after diff across two results dirs |
| `results/<run>/runs.json` | one record per run, every measurement kept |
| `results/<run>/gradient_overhead.json` | per-call cost of the transform on the gradient path |
| `RESULTS.md` | the write-up |

## Prior art

`docs/benchmark/common.jl` and `index.qmd` existed once on the abandoned
`cooperative-clusters` branch (`0937482`, Sep 2025). That harness covered
step-size and metric adaptation on Gaussian targets only — nothing hierarchical,
no reparametrization — and imports internals (`BayesianOptimizationStepsizeAdaptation`,
`IPGPRegression`, …) that no longer exist. It was mined for shape, not extended.
