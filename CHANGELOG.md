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

## 0.2.7

### Fixed

- A vector `init` at a stationary point of the log density (a mode: a MAP
  estimate, an empirical-Bayes estimate) initializes. L-BFGS declares
  convergence there before its first step, so Pathfinder returned no fit and
  `initialize_mcmc` raised "Pathfinder produced no ELBO estimates" from every
  sampler, `ContinuousNUTSState` included (snag `pathfinder-start-890fde27`,
  reporter NLMEEstimation: its Louis information and `saem(…; η0 = modes)`
  start every subject's chain at its FOCE(I) mode). Pathfinder now runs from
  the start plus a Gaussian jitter (standard deviation 0.05 per coordinate,
  then 1.0), and the chain starts from that run's draw with its metric; the
  same retry serves a finite start from which Pathfinder's first line search
  fails. If no jittered run fits either, the chain starts exactly at the given
  position with a unit diagonal metric, with a warning. A start Pathfinder
  fits is initialized exactly as before, and WarmupHMC's own random starts
  still redraw on a failure.

## 0.2.6

### Fixed

- A vector `init` on a target with a bounded support starts the chain inside
  it. Pathfinder's fit is a Gaussian, so the draw it starts the chain from
  could land where the log density is `-Inf` (or the model answers `NaN`), and
  every sampler then refused a start the caller never chose:
  `ContinuousNUTSState` with "the log density is -Inf at the initial position",
  `adaptive_warmup_mcmc` with DynamicHMC's "Starting point has non-finite
  density." (snag `pathfinder-draw-c1e43cb7`, reporter NLMEEstimation: a SAEM
  conditional whose first random effect is bounded at 0.6 in magnitude; 12 of
  40 seeds on a Gaussian whose mode sits 0.2 SD inside such a bound). The
  initializer now redraws from the same fit, up to ten times, and failing that
  starts at the L-BFGS iterate the fit was made at (then the other iterates,
  from the trajectory's end back to the start), with a warning; the fit's
  covariance gives the first metric either way. A run whose first draw is
  finite is unchanged, draw for draw.

## 0.2.5

### Added

- `reposition!(state::ContinuousNUTSState, position)`: translates the chain to
  a caller-supplied position, moving the mean of its position evidence by the
  same vector and nothing else, so the adapted metric, step size, evidence
  spread, κ fit, counters and random stream carry on and a translated target
  sampled by the translated chain is the same chain (snag
  `continuous-nuts-a09d94e2`, reporter NLMEEstimation: SAEM's mu-referenced
  re-centring shifts each subject's η after every M-step so that φ = Dθ + η
  stays where it was, and under a minibatch replaces an excluded chain by a
  prior draw; a translation of the target leaves the kernel valid). The
  position is copied and not evaluated; the next `continuous_nuts_step`
  evaluates it under its target. Rebuilding the state through
  `init = (; position, squared_scale)` instead restarts the metric evidence
  from the pseudo-count and runs the step-size search again.

## 0.2.4

### Added

- `ContinuousNUTSState` and `continuous_nuts_step`: a NUTS chain for a target
  that changes between transitions, whose diagonal metric and step size keep
  adapting on every transition. `FixedKernelNUTSState` / `fixed_kernel_nuts_step`
  advance a changing target under a kernel learned once, which mixes worse the
  further the target moves from where it was learned (snag
  `continuous-nuts-15212ad7`, reporter NLMEEstimation: SAEM's conditional
  posteriors move at every iteration). The new state starts as
  `adaptive_warmup_mcmc` does (`init` takes every form documented there;
  Pathfinder by default), and each `continuous_nuts_step` feeds every good
  leapfrog leaf of its transition into exponentially forgotten position and
  gradient moments (half-life `half_life` transitions, default 20; the start
  scale enters as `prior_weight` pseudo-states, default 10), takes the Nutpie
  diagonal scale of those moments as the next metric, carries the step size
  through the metric change, and moves it toward `target_acceptance_rate`
  through the cooperative sampler's κ fit over the recent transitions.
  `adapt=false` takes the transition under the current kernel and is the fixed
  kernel's draw. The state is advanced in place and holds O(d) memory; the
  leaves are not stored. `bench/continuous_nuts_saem.jl` compares it with the
  frozen kernel on SAEM-shaped changing targets.

### Changed

- `_fit_log_kappa` (internal) takes the sizes of its two search grids as
  keywords; the cooperative sampler's defaults are unchanged.

## 0.2.3

### Changed

- Every sampler now runs with single-threaded BLAS by default, and gives the
  caller's BLAS thread count back when it returns or throws. A new keyword on
  every entry point (`adaptive_warmup_mcmc`, `resume_warmup_mcmc`,
  `completion_warmup_mcmc`, `cooperative_warmup_mcmc`, `stream_mcmc`,
  `fixed_kernel_nuts_step`), `blas_threads` (default `1`), sets the count;
  `blas_threads = nothing` leaves BLAS untouched, which is the previous
  behaviour. Every leapfrog step takes `dot` products of posterior-length
  vectors, which OpenBLAS runs on all its threads above 10,000 entries; its
  idle workers then busy-wait while the gradient is computed, and under a CPU
  quota or with chains on Julia threads they take the gradient's CPU (snag
  `openblas-threads-2f1bf9e9`). The count is process-global: nested and
  concurrent sampler calls share one setting, and other BLAS work in the same
  process runs single-threaded while a sampler runs. A fixed seed now
  reproduces its draws without the caller setting BLAS threads.

## 0.2.2

The first patch bump under the per-landing version policy: a landing that adds
or changes public API bumps the patch version, so a dependant can state it as
its `[compat]` floor. 0.2.2 also contains everything listed under 1.0.0 below,
which is written against 0.2.1.

### Changed

- `completion_warmup_mcmc` no longer admits a chain merely because it reached
  `n_draws`. A chain whose step size collapsed, or whose transitions diverge,
  takes short, cheap trajectories and can finish before every healthy chain,
  and the quorum kept it. Such a chain is now left out as `:stuck`: it does not
  count toward `min_completed`, so the quorum waits for chains that pass. Two
  new keywords set the checks: `max_admitted_divergent_fraction` (default
  `0.1`) and `stuck_stepsize_ratio` (default `10`, against the median latest
  step size of the sibling chains that are keeping draws). `completion` reports
  `stuck_chain_indices`, `n_stuck` and both thresholds, and each chain's
  `stepsize`, `reference_stepsize`, `divergent_fraction` and `stuck_reason`; its
  `schema_version` is now 3. A finished run recorded before this change reopens
  as recorded.

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
and `progress.jl`. 1.0.0 loads every file in `src/` and comments out none.

So the honest reading of everything below is that most of it is not a
modification of code a 0.2.1 user was running — it is code that was dark.

### Added since 0.2.1

0.2.1 exported a single name, `adaptive_warmup_mcmc`. Everything below is new
public surface since then; `README.md` lists what the total now is, and
`test/public_api.jl` is what enforces it.

- `cooperative_warmup_mcmc` — multi-chain sampler that pools every chain's
  warm-up evidence through soft clusters, from one seed RNG, with a returned
  `stop_reason`, a stable `chain_index` in results, and write-once
  `run_manifest.json` / `run_summary.json`. It replaced, before any release, a
  scheduler that parked chains and the separate `clustered_warmup_mcmc`, which
  is gone.
- `resume_warmup_mcmc` — byte-identical resume from an on-disk checkpoint.
  **Added and deprecated within this same window**: `resume=true` on the
  samplers themselves supersedes it. It ships exported and documented, carrying
  a deprecation warning, so 1.0's compatibility promise covers it — that is why
  it is listed rather than quietly dropped.
- `CandidateScoringPlan` — steers online candidate adaptation.
- `WindowSelectionPlan` — replaces the candidate grid with a user rule that
  chooses the controls at every restarting window from `recording_target`
  equally weighted states: the retained pool, or a weight-proportional sample
  of the window's NUTS leaves. It also adapts reparametrizers that are not an
  `IndexedReparametrization`, through a small controls interface
  (`reparam_controls`, `restore_reparam_controls!`,
  `snapshot_reparametrization`).
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
- **A divergence gate on the window that ends the run** (`max_divergent_fraction`,
  default `0.5`). A run ends as soon as its retained draws reach `n_draws`,
  without the restart test on that window. A chain that had moved on from the
  region its kernel was adapted in could therefore return mostly divergent draws.
  When more than that fraction of the retained draws diverged, warm-up now
  restarts where the chain is instead. It does so at most twice per run, then
  returns the draws with a warning, because rerunning is the caller's job.
- Online estimators for the linear restart source and for nonlinear leaf
  evidence; NUTS leaves weighted by proposal probability.

### Changed

- **Neither of 0.2.1's package extensions survives; one new one is added.**
  0.2.1 carried two (`SerializationExt` on `Serialization`, `TermExt` on
  `Term`). In 1.0.0 `Serialization` is a plain dependency and `Term` is gone.
  The new `WarmupHMCEnzymeExt` on `Enzyme` changes no API (`b14ee88`): with a
  reverse-mode `AutoEnzyme` backend and an `IndexedReparametrization` it
  differentiates the reparametrization pair by pair instead of in one pass,
  which removes a per-accessor-call allocation from every gradient and every
  window-end transport on accessors that read shared mutable state. The
  gradient agrees with the one-pass one to rounding.
- **Progress reporting moved to `Treebars`** (from `Term`), which is what
  renders the tree-shaped progress bar.
- **`DifferentiationInterface` is a hard dependency** — new in this window,
  not a promoted extension. `InverseFunctions` was added; `StatsBase` dropped.
- **Minimum Julia is 1.10** (was 1.9), and every dependency now carries a
  `[compat]` bound.
- The multi-chain method gives every chain its own log-density problem.

### Fixed — in code 0.2.1 actually loaded

Only three, and the scope is deliberate: a fix to a file 0.2.1 never `include`d
could not have affected anyone running 0.2.1, and listing it here would imply
otherwise.

- **Pathfinder initialization uses the supplied gradient** (`0ccdcec`).
  Optimization's default `AutoForwardDiff` path was synthesizing its own
  gradient and calling `logdensity` with `Dual`-valued parameters, which
  native-backed targets such as BridgeStan cannot accept. Now passes
  `NoAD()` and the target's own `logdensity_and_gradient`.
- **Pathfinder initialization survives a trajectory its fit cannot
  factorize** (`c1750fd`). On a log density with a kink at its mode, L-BFGS
  ends on the kink, and Pathfinder's inverse-Hessian fit throws a
  `PosDefException`. A vector `init` then ended the sampler call, and a random
  start could use up all ten retries. The initializer now refits on the
  longest prefix of the trajectory that factorizes, with a warning; a run that
  Pathfinder completes is unchanged.
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
