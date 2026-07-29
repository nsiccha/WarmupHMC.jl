# Changelog

Notable changes to WarmupHMC.jl. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org) as scoped by
[What semver covers](https://nsiccha.github.io/WarmupHMC.jl/dev/api#What-semver-covers).

<!--
WHAT THIS FILE MAY AND MAY NOT CONTAIN.

A changelog is a census in prose, and this repository treats those as defects
unless something can contradict them. Two rules keep this one honest:

1. A RELEASED section is immutable. It describes what shipped under that
   version number and is never edited afterwards, so it cannot go stale — the
   version number is the thing it is a census of, and that never changes again.
   The Unreleased section is the only one that drifts, and it drifts for at most
   one release.

2. DO NOT RE-STATE THE API CONTRACT HERE. The exported surface, the accepted
   keywords and the returned field names are pinned by `test/public_api.jl` and
   stated in `docs/src/api.md`. A second copy in this file would be a copy that
   nothing checks, and the failure mode is specific: it would keep reading as a
   complete answer while silently describing an older surface. Name what CHANGED
   in a release; point at the pinned contract for what IS.

Numbers quoted here must come from a checked-in artifact under
`docs/benchmark/results/` and name the page that renders them — same rule as
`docs/src/evidence.md` states for the documentation.
-->

## 1.0.0 — unreleased

First release with a stated compatibility contract. The code has been usable for
a long time; what 1.0 adds is a promise about which parts of it will not move.

### The contract

- **Semantic versioning now covers** the exported names, the keywords
  `adaptive_warmup_mcmc` accepts, and the field names *and order* of the
  `NamedTuple` it returns. Stated in full under
  [What semver covers](https://nsiccha.github.io/WarmupHMC.jl/dev/api#What-semver-covers),
  and enforced by `test/public_api.jl`, which pins each of them against a
  literal list — deliberately literal, so that changing the public surface
  requires editing the test that describes it.
- **It deliberately does not cover the numbers.** Draws, step sizes and the
  adaptation path change whenever the sampler improves: a minor release may
  return different draws for the same seed. Anything that needs bit-stability
  should pin an exact version.
- Non-exported names are not covered, including those that appear in the API
  reference. They are listed there because the exported docstrings link to them
  — a consequence of `checkdocs = :exports`, not a promise.

### What 0.2.1 actually was

Worth stating before the delta, because it decides how to read it: 0.2.1's
`src/WarmupHMC.jl` **commented out most of its own `include` lines**. Twelve
files sat in `src/` without being loaded — the reparametrization machinery, the
recorder, the cooperative sampler, the NUTS implementation. Four were live:
`MatrixExpressions.jl`, `WrappedLogDensityProblems.jl`, `adaptive_warmup_mcmc.jl`
and `progress.jl`. 1.0.0 loads eleven files and comments out none.

So the honest reading of everything below is that most of it is not a
modification of code a 0.2.1 user was running — it is code that was dark.

### Added since 0.2.1

0.2.1 exported a single name, `adaptive_warmup_mcmc`. Everything below is new
public surface since then; `README.md` lists what the total now is, and
`test/public_api.jl` is what enforces it.

- `cooperative_warmup_mcmc` — multi-chain scheduler with per-chain
  continue/resume/start/abandon, a stable `chain_index` surfaced in results, and
  write-once `run_manifest.json` / `run_summary.json`.
- `clustered_warmup_mcmc` — pooled-scale cooperative sampler with greedy
  compatibility clustering and a look-behind strategy over a swappable
  criterion.
- `resume_warmup_mcmc` — byte-identical resume from an on-disk checkpoint.
  **Added and deprecated within this same window**: `resume=true` on the
  samplers themselves supersedes it. It ships exported and documented, carrying
  a deprecation warning, so 1.0's compatibility promise covers it — that is why
  it is listed rather than quietly dropped.
- `CandidateScoringPlan` — steers online candidate adaptation.
- The reparametrization types `ReparametrizedProblem`,
  `IndexedReparametrization`, `PartiallyCentered` and `Reparametrization`.
- **On-disk checkpointing** across all three samplers, on a shared contract:
  `checkpoint_dir`, `resume`, `overwrite`, plus an observational `callback` at
  window boundaries. Checkpoints are written atomically, hold pure sampler
  state, and keep the draws a restart would otherwise discard. A payload's
  positions are in the sampler's working frame, and `WarmupHMC.back_transform`
  maps them into the model's own — so a consumer can materialize a running
  reparametrized fit's partial results, which previously had no entry point.
- **Keyword validation.** Unknown keywords raise `ArgumentError` instead of
  being silently forwarded and ignored.
- Online estimators for the linear restart source and for nonlinear leaf
  evidence; NUTS leaves weighted by proposal probability.

### Changed

- **No more package extensions.** 0.2.1 carried two (`SerializationExt` on
  `Serialization`, `TermExt` on `Term`). 1.0.0 has no `[weakdeps]` and no
  `[extensions]`: `Serialization` is a plain dependency, and `Term` is gone.
- **Progress reporting moved to `Treebars`** (from `Term`), which is what
  renders the tree-shaped progress bar.
- **`DifferentiationInterface` is a hard dependency** — new in this window,
  not a promoted extension. `InverseFunctions` was added; `StatsBase` dropped.
- **Minimum Julia is 1.10** (was 1.9), and every dependency now carries a
  `[compat]` bound.
- The multi-chain method gives every chain its own log-density problem.

### Fixed — in code 0.2.1 actually loaded

Only two, and the scope is deliberate: a fix to a file 0.2.1 never `include`d
could not have affected anyone running 0.2.1, and listing it here would imply
otherwise.

- **Pathfinder initialization uses the supplied gradient** (`0ccdcec`).
  Optimization's default `AutoForwardDiff` path was synthesizing its own
  gradient and calling `logdensity` with `Dual`-valued parameters, which
  native-backed targets such as BridgeStan cannot accept. Now passes
  `NoAD()` and the target's own `logdensity_and_gradient`.
- **`AbstractMatrixExpression` has structural `hash`/`isequal`/`==`**
  (`3b3c05d`). These are matrix-free operators: they subtype `AbstractMatrix`
  for `mul!`/`ldiv!` but define no `getindex`, so Base's generic fallbacks
  iterated via `getindex` and threw `CanonicalIndexError` the moment one became
  a `Dict` key.

Everything else repaired during this window — the draw back-transform direction,
joint halo transport, the halo recording rate, pooled joint ESS — was in code
0.2.1 did not load. Those are in the git history, not here.

### Evidence

Every performance claim in the documentation is rendered at build time from a
checked-in JSON file under `docs/benchmark/results/`; none is typed in. The
cross-sampler comparison behind the README's "faster than standard methods"
sentence is
[WarmupHMC vs other samplers](https://nsiccha.github.io/WarmupHMC.jl/dev/sampler-comparison),
which counts its own verdict from the rows — including the finding that the
per-gradient verdict reproduces across repeated runs and the wall-clock one does
not. See [Evidence](https://nsiccha.github.io/WarmupHMC.jl/dev/evidence) for the
full artifact index.

**No page in this documentation measures correctness of the draws.** The
comparison is an efficiency measurement, and says so before its own tables.
