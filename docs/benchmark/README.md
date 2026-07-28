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

**AD backend.** The harness selects the backend for `ReparametrizedProblem` in
one place and it is settable per run:

| variable | default | meaning |
|---|---|---|
| `WHMC_BENCH_AD` | `enzyme` | `enzyme` or `forwarddiff` — the DI backend the wrapper differentiates with |

Reverse mode is the rule; ForwardDiff is retained *only* so the two can be
compared on one base, which is what `results/enzyme-b5c7dee` and
`results/forwarddiff-b5c7dee` document. Note the backend package must be loaded
for the ADTypes object to work: `AutoEnzyme()` needs `using Enzyme`.

Two things about that default are easy to get wrong:

- **It must be `AutoEnzyme(; function_annotation = Enzyme.Const)`.** A bare
  `AutoEnzyme()` raises `EnzymeMutabilityException` here, because the
  differentiated objective is a closure capturing the reparametrizer and the
  frozen inner gradient. Enzyme's own error text suggests `Duplicated`, which is
  correct but costs up to 13× more on small targets.
- **Reverse mode is not uniformly faster on these targets, and the theoretical
  argument for it does not survive measurement.** The objective is scalar in `d`
  inputs with the inner gradient held fixed, so forward mode should cost
  `ceil(d / chunksize)` sweeps where reverse costs one, with the gap widening in
  `d`. Measured, Enzyme/`Const` wins only on the two `d = 10` targets and loses
  on all three larger ones. See `RESULTS.md` § *Which AD backend*. Pick the
  backend per target from the numbers, not from the argument.

### Backend probes

Three standalone scripts, each answering one question and writing one JSON. They
are separate from the driver because none of them samples — they time the
gradient path directly and finish in minutes.

```bash
julia --project=docs/benchmark docs/benchmark/replicate_backends.jl   # ROUNDS, NCALLS
julia --project=docs/benchmark docs/benchmark/prep_cost.jl            # NCALLS
julia --project=docs/benchmark docs/benchmark/typical_positions.jl    # ROUNDS
```

| script | question | output |
|---|---|---|
| `replicate_backends.jl` | ForwardDiff vs Enzyme/`Const` vs Enzyme/`Duplicated` per gradient, with rounds interleaved and the backend order rotated so drift is not charged to one backend | `results/backend_replication.json` |
| `prep_cost.jl` | how much of the per-gradient cost is DI preparation, which the hot path redoes on every call | `results/prep_cost.json` |
| `typical_positions.jl` | whether the verdict depends on evaluating at `randn(d)` rather than where the sampler actually goes | `results/typical_positions.json` |
| `annotation_sweep.jl` | superseded first pass at `Const` vs `Duplicated`, one shot per configuration; kept because `replicate_backends.jl` was written to check it | `results/annotation_sweep.json` |

**Run these in a process that has not just sampled.** Running a full sampler
first warms the ForwardDiff path enough to make its subsequent microbenchmark
~2× faster, which silently biases a backend comparison in ForwardDiff's favour.

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
count of seeds whose adaptation never moved, and where the centering settled.

Wall-clock is reported **only when both runs recorded the same WarmupHMC SHA** —
two runs that compiled different package sources are not comparable in seconds,
but two runs of the same source under different AD backends differ in seconds for
exactly the reason being compared, and omitting it there would throw the result
away. When it does report wall-clock it also checks that the runs are
sampling-identical, and names any arm whose gradient count drifted ≥2% so a
verdict can be checked against that list rather than against a global maximum.

## Layout

| path | what |
|---|---|
| `common.jl` | targets, arm construction, metrics, gradient-overhead timing |
| `run_reparam_benchmark.jl` | driver — runs everything, writes a results dir, prints the table |
| `summarize.jl` | regenerates `RESULTS.md`'s tables from one results dir |
| `compare.jl` | before/after diff across two results dirs |
| `replicate_backends.jl`, `prep_cost.jl`, `typical_positions.jl`, `annotation_sweep.jl` | backend probes (above) |
| `results/<run>/runs.json` | one record per run, every measurement kept |
| `results/<run>/gradient_overhead.json` | per-call cost of the transform on the gradient path |
| `results/*.json` | backend-probe outputs, not tied to a sampling run |
| `RESULTS.md` | the write-up |

Result directories, newest base last:

| dir | base | backend | note |
|---|---|---|---|
| `results/before` | `05aed41` | ForwardDiff | carried the halo-recording regression `34ce034` |
| `results/after` | `c8fed88` | ForwardDiff | the fix; reproduced exactly by `forwarddiff-b5c7dee` |
| `results/forwarddiff-b5c7dee` | `b5c7dee` | ForwardDiff | backend comparison baseline |
| `results/enzyme-b5c7dee` | `b5c7dee` | Enzyme/`Const` | **current default** |

## Prior art

`docs/benchmark/common.jl` and `index.qmd` existed once on the abandoned
`cooperative-clusters` branch (`0937482`, Sep 2025). That harness covered
step-size and metric adaptation on Gaussian targets only — nothing hierarchical,
no reparametrization — and imports internals (`BayesianOptimizationStepsizeAdaptation`,
`IPGPRegression`, …) that no longer exist. It was mined for shape, not extended.
