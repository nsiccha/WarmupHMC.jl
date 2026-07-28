# Nonlinear-reparametrization benchmark

Measures WarmupHMC's adaptive partial centering against the fixed centering
endpoints on hierarchical targets, and measures what the transform costs on the
gradient hot path.

The question: **on a model written in its centered parametrization, does the
adaptive method reach noncentered-like performance without being told to?**

`RESULTS.md` has the answer and the numbers. This file is how to reproduce them.

The same environment also carries `run_linear_restart_benchmark.jl`, the
linear-only comparison of the legacy halo restart criterion against the running
all-good-leaf and NUTS-weighted estimators. Its checked-in output is
`results/linear_restart.json`; the generic WarmupHMC web renderer and the
VitePress [Linear restart evidence](../src/linear-restart.md) page both read that
one file. The default command runs the full 32-seed matrix:

```bash
julia --project=docs/benchmark docs/benchmark/run_linear_restart_benchmark.jl
```

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
canonical resolve develops them; nothing in this benchmark uses them directly.
The GitHub Actions evidence job develops those packages from their explicitly
named active branches before instantiating this environment.

**Every workflow that instantiates this environment needs that same four-clone
step** — `test.yml`'s `linear-evidence` job and `nonlinear_weighting.ci.yml`
both carry it verbatim, so keep them in step. A runner has no `Manifest.toml`
to fall back on (the repository gitignores every one), and three of the four
packages are unregistered, so nothing else supplies them. `--branch dev` on the
Treebars clone is the one that fails loudest if dropped: `Treebars.round2`,
which WarmupHMC's `src/progress.jl` adds a method to at load time, does not
exist on `origin/main`, so a default-branch clone dies at precompile with
`UndefVarError: round2` before any benchmark code runs.

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
- **Enzyme/`Const` loses on three of five targets here — because of a defect in
  the spec table, not because of reverse mode.** `reparametrization()` in
  `web/src/posteriordb_reparametrizations.jl` closes over indices assigned in
  several branches of one long `if`/`elseif`, so Julia captures a `Core.Box`
  instead of an `Int` on `radon_partially_pooled`, `radon_variable_intercept`
  and `seeds`. That costs ForwardDiff ~4.8× and Enzyme ~15× per wrapped
  gradient, which is enough to invert which backend looks faster (1.21× slower
  → 0.39×, i.e. 2.6× faster, on de-boxed `radon_partially_pooled`, with
  bit-identical gradients). `funnel` and `eight_schools` close over literals and
  are unaffected — they are the only two rows that currently say anything about
  the backend, and both favour Enzyme. Run `capture_boxing.jl` (below) before
  drawing any backend conclusion, and treat the `d`-scaling question as **open**:
  dimension and boxing are perfectly confounded across these five targets.

### Backend probes

Standalone scripts, each answering one question and writing one JSON. They are
separate from the driver because none of them samples — they time the gradient
path directly and finish in minutes. **Start with `capture_boxing.jl`**: it
explains the others' headline result.

```bash
julia --project=docs/benchmark docs/benchmark/replicate_backends.jl   # ROUNDS, NCALLS
julia --project=docs/benchmark docs/benchmark/prep_cost.jl            # NCALLS
julia --project=docs/benchmark docs/benchmark/typical_positions.jl    # ROUNDS
julia --project=docs/benchmark docs/benchmark/capture_boxing.jl       # ROUNDS, NCALLS
julia --project=docs/benchmark docs/benchmark/frame_check.jl
```

| script | question | output |
|---|---|---|
| `replicate_backends.jl` | ForwardDiff vs Enzyme/`Const` vs Enzyme/`Duplicated` per gradient, with rounds interleaved and the backend order rotated so drift is not charged to one backend | `results/backend_replication.json` |
| `prep_cost.jl` | how much of the per-gradient cost is DI preparation, which the hot path redoes on every call | `results/prep_cost.json` |
| `typical_positions.jl` | whether the verdict depends on evaluating at `randn(d)` rather than where the sampler actually goes | `results/typical_positions.json` |
| `annotation_sweep.jl` | superseded first pass at `Const` vs `Duplicated`, one shot per configuration; kept because `replicate_backends.jl` was written to check it | `results/annotation_sweep.json` |
| `capture_boxing.jl` | **why the backend comparison says what it says**, and a guard so it cannot say it again — which specs capture a `Core.Box`, read off `fieldtypes`, plus an A/B between a boxed and an unboxed control with gradients bit-identical to the shipped spec's | `results/capture_boxing.json` |

| `frame_check.jl` | **which frame `nonlinear_adapt=false` returns draws in** — the assumption every fixed-`c` arm's numbers rest on, and one that has already flipped once | `results/frame_check.json` |

`capture_boxing.jl` and `frame_check.jl` are the two probes with an **exit code**:
non-zero, saying what to do about it, when the property they pin does not hold.
`capture_boxing.jl` names every shipped spec that captures a `Core.Box`, and builds
both A/B controls inside the script rather than taking one from the spec table, so
the cost stays measurable — and the guard stays honest — regardless of how the
shipped table is currently written. `frame_check.jl` discriminates by a wide margin
(1.6 against 103.3 in max relative sd deviation), so a pass is not a near miss.

**Run both before trusting a results directory.** Each pins a sampler property the
harness depends on and cannot detect the loss of from the numbers alone — a
double-transformed fixed arm and a boxed spec are both perfectly plausible-looking
rows.

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
| `replicate_backends.jl`, `prep_cost.jl`, `typical_positions.jl`, `annotation_sweep.jl`, `capture_boxing.jl`, `frame_check.jl` | probes (above) |
| `results/<run>/runs.json` | one record per run, every measurement kept |
| `results/<run>/gradient_overhead.json` | per-call cost of the transform on the gradient path |
| `results/*.json` | backend-probe outputs, not tied to a sampling run |
| `RESULTS.md` | the write-up |
| `nonlinear_weighting.jl` | the trajectory-weighting study's derivation — definitions only, see below |
| `nonlinear_weighting_run.jl` | that study's measurement driver |
| `nonlinear_weighting_report.jl` | prints its tables, writes a derived summary |
| `nonlinear_weighting.ci.yml` | ready-to-move GitHub Actions workflow for it |
| `results/nonlinear_weighting/rows.json` | its rows — the only stored fact of that study |

## The nonlinear trajectory-weighting study

A separate experiment sharing this directory: two policy axes on the adaptive
nonlinear path — which leaves contribute evidence (`all_good_leaves` vs
`nuts_weighted`) and how that evidence is scaled (`unit` vs `stepsize`) — over
240 runs, measured by `WarmupHMC:nonlinear-online`.

It is split three ways for a reason:

- **`nonlinear_weighting.jl` is definitions only.** The docs build
  `load_harness`es it and calls `nw_conclusion(rows)` and the table functions at
  build time, so nothing in it may run at include time — no `ARGS`, no reads, no
  writes — and it may depend only on JSON, Markdown, Printf and Statistics.
- **`nonlinear_weighting_report.jl` is the part that acts** — reads `ARGS`,
  prints, writes.
- **`nonlinear_weighting_run.jl` re-measures**, and needs BridgeStan, PosteriorDB
  and hours of quiet CPU that reading the rows does not.

**The rows are the artifact.** Every ratio, table and the conclusion itself is
derived from `results/nonlinear_weighting/rows.json` at read time; nothing
derived is stored beside it (the report's summary JSON is gitignored there for
exactly that reason). A stored summary is a second representation of one fact
with nothing keeping the two in step.

Its `config` block is deliberately wider than "defaults": it records the
resolved sampler settings, each target's exact constructor, spec and AD backend,
`julia`, and `blas_threads`. Guessing a target from its name cost a full round of
mismatched rows — `funnel` there is the analytic `Funnel(5)` from
`web/src/test/targets.jl`, not this directory's `Funnel(10)`, and
`eight_schools` is the analytic `EightSchools(true)`, not the posteriordb Stan
model of that name. The `reproduction` block records that the committed driver
regenerates all 240 rows bit-for-bit; the keys naming the original measurement
(`measured_by`, `measured_on`, `warmuphmc_sha`, `nonlinear_impl_sha`,
`linear_estimator_base_sha`, `verification`) describe that event and are not
regenerated by a later run — a re-run records its own.

Re-measure it, or a slice of it:

```bash
WHMC_NW_TARGETS=funnel,eight_schools WHMC_NW_SEEDS=1:2 WHMC_NW_DRAWS=200 \
  WHMC_NW_OUT=/tmp/nw julia --project=docs/benchmark \
  docs/benchmark/nonlinear_weighting_run.jl
julia --project=docs/benchmark docs/benchmark/nonlinear_weighting_report.jl /tmp/nw/rows.json
```

That slice is also the CI smoke configuration. `nonlinear_weighting.ci.yml` is
inert where it sits — GitHub reads only `.github/workflows/` — and is meant to
be moved there rather than copied. Its `derive` job is a real gate (pure
arithmetic over the committed rows); its `smoke` job uploads raw rows and
asserts nothing about their values, because a benchmark that goes red on an
unfavourable result teaches everyone to stop reading it.

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
