# Measured: adaptive reparametrization vs the fixed centering endpoints

**Measured on WarmupHMC `c8fed88`.** 8 pinned seeds per arm, `n_draws` floor
1000, Julia 1.10.11, single-threaded BLAS, on `strato2`. 184 runs, 0 failed.
Raw records in `results/after/`; `summarize.jl` regenerates every table below
from them.

An earlier full run of this benchmark was measured on `05aed41`, which carried
the halo-recording regression `34ce034`. Those numbers are superseded but kept,
in `results/before/` — the comparison is a measured result of its own and is
reported in [Effect of the halo-recording regression](#effect-of-the-halo-recording-regression).

## Verdict

**Adaptive partial centering, started from the centered parametrization, beats
the plain sampler on every target measured** — 1.8× to 51× per gradient
evaluation. It gets there without being told which parametrization to use.

**Against the best available *fixed* parametrization the picture splits, and the
split is the interesting part:**

| target | adaptive | best fixed alternative | |
|---|---|---|---|
| radon partially_pooled | **53.08** | 22.61 (noncentered endpoint) | **2.3× win** |
| radon variable_intercept | **65.92** | 22.65 (centered endpoint) | **2.9× win** |
| eight_schools centered | 48.40 | 51.56 (hand-written noncentered) | ties, −6% |
| seeds centered | 15.45 | 15.05 (noncentered endpoint) | ties, +3% |
| funnel (synthetic) | 78.45 | 107.05 (noncentered endpoint) | **loses, −27%** |

(min ESS per 1000 gradient evaluations, median over 8 seeds.)

Read as one sentence: **where a good hand-written parametrization exists,
adaptive finds something as good as it, automatically. Where none of the fixed
options is good, adaptive beats all of them.** Both radon models are the second
case — centered, noncentered and posteriordb's own hand-written noncentered
model all land within seed noise of each other around 14–23, and adaptive gets
53–66 by settling at an *interior* `c ≈ 0.4–0.5` that no hand-written model
offers. That interior optimum is the strongest result here.

**It is not free, and on three of five targets it is a net wall-clock loss.**
The `ReparametrizedProblem` wrapper costs ×4.9–×15.8 per
`logdensity_and_gradient` call. Where the pathology is severe (eight_schools,
funnel) the sampling gain dwarfs that and adaptive wins by 20×. Where it is mild
(both radon models, seeds) the per-gradient gain is real but smaller than the
per-call cost, and adaptive is 1.6×–3.4× **slower in wall-clock than doing
nothing at all**.

So the claim the package can support today is about gradient evaluations, not
seconds:

> Starting from a centered parametrization, adaptive partial centering matches or
> exceeds the best fixed parametrization's sampling efficiency per gradient
> evaluation on four of five targets, and beats every fixed option on the two
> where no fixed option is good — without being told to. Whether that converts
> into wall-clock depends on how expensive the target's own gradient is relative
> to the wrapper.

**The overhead is not the transform.** A wrapper configured as an exact identity
costs the same as a live one (×5.05 vs ×4.87, ×12.44 vs ×13.39, ×5.18 vs ×4.98,
×7.87 vs ×7.64, ×14.96 vs ×15.78 — five independent target pairs, and the same
result held on the pre-fix run). The cost is carrying the AD re-differentiation
of `ljac_(x_) + dot(g_y, y_)` at all, not the arithmetic of the
reparametrization. Every wall-clock loss above is addressable there; none of it
is intrinsic to the method.

## What each arm is

Every arm samples the **same** Stan model, so all draws are in the same
coordinates and their ESS is directly comparable. The arms differ only in what
the sampler may do with the partial-centering parameter `c`
(`PartiallyCentered(1.0)` is centered, `(0.0)` is noncentered).

| arm | what it is |
|---|---|
| `plain` | bare `StanProblem`, no wrapper. What a user gets today. |
| `fixed c = centered` | wrapped, source `c` pinned to the model's own value, `nonlinear_adapt=false`. An exact identity transform, so it isolates the cost of merely carrying the wrapper. |
| `adaptive` | wrapped, source `c` starts at the model's own value, `nonlinear_adapt=true`. The method under test. |
| `fixed c = noncentered` | wrapped, source `c` pinned to the opposite endpoint. The hand-written noncentered parametrization, reached by transform. |
| `hand-written noncentered model` | posteriordb's separately written noncentered member, sampled plain. The external reference for "noncentered-like". |

All arms use `AutoForwardDiff()`, matching the shipped consumer at
`web/src/WarmupHMCWeb.jl:148`, so the overhead figures are what a user pays
rather than an artifact of a backend choice made here.

`min ESS` is the minimum over coordinates — the number that governs how long you
must run. Sub-figures are the min–max across the 8 seeds. `final source c` is
read back off the mutated `IndexedReparametrization` after the run, which is
valid because every run here is single-chain `adaptive_warmup_mcmc`; the
cooperative and clustered samplers `deepcopy` the problem per chain and would
silently return the *initial* `c` instead.

<!-- tables generated by docs/benchmark/summarize.jl from results/after/runs.json -->

### `eight_schools-eight_schools_centered`

*8 group effects — dimension 10. The textbook hierarchical funnel.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 51.1 <sub>25.2–86.3</sub> | 300.0 <sub>71.0–626.8</sub> | 1.55 | 33320 | 4.5 | — |
| fixed c = centered | 46.8 <sub>8.8–102.8</sub> | 324.7 <sub>40.0–436.2</sub> | 2.15 | 22504 | 8.0 | 1.0 |
| **adaptive** (starts centered) | 510.7 <sub>450.9–780.4</sub> | 6317.4 <sub>4649.2–8434.0</sub> | 48.40 | 10512 | 0.0 | 0.0 |
| fixed c = noncentered | 500.8 <sub>348.3–658.2</sub> | 8097.8 <sub>2020.5–11351.3</sub> | 46.22 | 10312 | 0.0 | 0.0 |
| hand-written noncentered model | 569.9 <sub>345.5–627.4</sub> | 10627.6 <sub>2677.0–28125.6</sub> | 51.56 | 9482 | 0.0 | — |

Min ESS over the 10 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 51.1 | 1.54 |
| fixed c = centered | 46.8 | 2.08 |
| **adaptive** (starts centered) | 510.7 | 48.58 |
| fixed c = noncentered | 500.8 | 48.56 |
| hand-written noncentered model | 569.9 | 60.11 |

Adaptive drives `c` to exactly 0.0 on all 8 coordinates on all 8 seeds — the
known right answer for this model, found without being told. It lands level with
the noncentered endpoint reached by transform (48.58 vs 48.56 matched) and about
6% below posteriordb's hand-written noncentered model. Divergences go from a
median of 4.5 on plain to 0.

This is the target where the pre-fix run over-stated the method: on `05aed41`
adaptive appeared to *beat* the hand-written model by 41%. It does not.

### `radon_mn-radon_partially_pooled_centered`

*85 counties — dimension 88.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 189.9 <sub>100.1–255.9</sub> | 379.7 <sub>215.2–494.5</sub> | 14.12 | 14971 | 0.0 | — |
| fixed c = centered | 209.4 <sub>134.6–300.9</sub> | 35.4 <sub>31.0–43.9</sub> | 14.91 | 14828 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 507.8 <sub>391.4–795.5</sub> | 113.3 <sub>86.7–152.9</sub> | 53.08 | 8644 | 0.0 | 0.4 |
| fixed c = noncentered | 361.3 <sub>201.4–452.1</sub> | 54.3 <sub>31.2–66.8</sub> | 22.61 | 16008 | 0.0 | 0.0 |
| hand-written noncentered model | 314.4 <sub>178.5–439.6</sub> | 499.1 <sub>377.7–701.3</sub> | 18.52 | 16456 | 0.0 | — |

Min ESS over the 88 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 189.9 | 12.68 |
| fixed c = centered | 209.4 | 14.12 |
| **adaptive** (starts centered) | 507.8 | 58.74 |
| fixed c = noncentered | 361.3 | 22.57 |
| hand-written noncentered model | 314.4 | 19.11 |

**The headline case for the method.** No fixed parametrization helps: centered
(14.12), noncentered (22.61) and posteriordb's hand-written noncentered model
(18.52) are all in the same band. Adaptive settles at `c = 0.4` on seven of eight
seeds (0.5 on the eighth) and gets 53.08 — 2.3× the best fixed option — using 46%
fewer gradient evaluations
than the noncentered endpoint. **And it is still 3.4× slower in wall-clock than
the plain sampler** (113.3 vs 379.7 ESS/sec), because this target has the worst
measured wrapper overhead (×13.4). That is the split verdict in one row.

### `radon_mn-radon_variable_intercept_centered`

*85 counties + floor slope — dimension 89.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 273.7 <sub>173.6–408.5</sub> | 368.9 <sub>223.2–505.0</sub> | 19.37 | 14960 | 0.0 | — |
| fixed c = centered | 339.7 <sub>208.5–445.1</sub> | 46.0 <sub>40.2–56.9</sub> | 22.65 | 15016 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 627.6 <sub>409.0–766.7</sub> | 227.5 <sub>158.8–331.7</sub> | 65.92 | 8654 | 0.0 | 0.5 |
| fixed c = noncentered | 315.3 <sub>200.7–377.4</sub> | 74.7 <sub>47.0–96.9</sub> | 18.26 | 16818 | 0.0 | 0.0 |
| hand-written noncentered model | 287.0 <sub>220.2–444.0</sub> | 416.5 <sub>273.1–535.0</sub> | 21.36 | 15837 | 0.0 | — |

Min ESS over the 89 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 273.7 | 18.30 |
| fixed c = centered | 339.7 | 22.62 |
| **adaptive** (starts centered) | 627.6 | 72.52 |
| fixed c = noncentered | 315.3 | 18.75 |
| hand-written noncentered model | 287.0 | 18.12 |

Same shape as its sibling, more strongly: an interior optimum at `c = 0.5` on
all eight seeds worth 2.9× the best fixed alternative, and still a 1.6×
wall-clock loss. Note the noncentered endpoint here is *worse* than centered —
this is a model where the conventional advice would actively mislead, and the
adaptive method is unaffected by that.

### `seeds_data-seeds_centered_model`

*21 plates — dimension 26. posteriordb ships no noncentered sibling.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 101.7 <sub>88.3–161.0</sub> | 852.4 <sub>645.7–1324.3</sub> | 8.40 | 13068 | 0.0 | — |
| fixed c = centered | 101.7 <sub>88.3–161.0</sub> | 257.6 <sub>166.3–372.7</sub> | 8.40 | 13068 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 223.2 <sub>140.2–310.4</sub> | 457.6 <sub>249.4–640.5</sub> | 15.45 | 15454 | 0.0 | 0.3 |
| fixed c = noncentered | 295.7 <sub>144.2–414.4</sub> | 598.9 <sub>253.5–862.0</sub> | 15.05 | 17673 | 0.0 | 0.0 |

Min ESS over the 47 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 101.7 | 7.78 |
| fixed c = centered | 101.7 | 7.78 |
| **adaptive** (starts centered) | 223.2 | 14.45 |
| fixed c = noncentered | 295.7 | 16.73 |

Adaptive and the noncentered endpoint are within noise of each other, and which
one is ahead depends on which ESS you take: unconstrained per-gradient favours
adaptive (15.45 vs 15.05), the matched constrained comparison favours the
endpoint (14.45 vs 16.73). Treat this as a tie, not a win. Both roughly double
plain. Interior optimum again, `c ≈ 0.3`.

### `funnel` (synthetic)

*Neal's funnel, `v ~ Normal(0, 3)`, `theta_i ~ Normal(0, exp(v/2))`, K = 9 —
dimension 10. Not a posteriordb posterior; posteriordb ships none. Defined in
`common.jl` with an analytic gradient. It is the extreme case — the centered
parametrization is pathological and the noncentered one is exact — so it bounds
what partial centering can possibly buy. Reported alongside the real targets,
never as a headline.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 31.7 <sub>19.6–56.4</sub> | 677.7 <sub>109.4–1821.2</sub> | 1.53 | 26799 | 0.5 | — |
| fixed c = centered | 31.7 <sub>19.6–56.4</sub> | 538.5 <sub>202.4–983.8</sub> | 1.53 | 26799 | 0.5 | 1.0 |
| **adaptive** (starts centered) | 823.1 <sub>746.4–936.8</sub> | 13523.6 <sub>12002.8–14725.2</sub> | 78.45 | 10054 | 0.0 | 0.0 |
| fixed c = noncentered | 823.2 <sub>718.3–1017.9</sub> | 43315.8 <sub>36386.5–52876.3</sub> | 107.05 | 7963 | 0.0 | 0.0 |

The one target where a fixed endpoint beats adaptive on gradient efficiency: 78.5
vs 107.1 per 1000 gradients. That is the expected and correct result — noncentered
is *exactly* right here, so there is nothing to discover and adaptation is pure
overhead. Note the min ESS is identical (823.1 vs 823.2): adaptive reaches the
same sampling quality, it just spends 26% more gradients getting there. Against
the parametrization a user would actually have written, it is 51× better.

## Effect of the halo-recording regression

`34ce034` collapsed the halo pool the adaptation reads from. It was fixed in
`095efb0`, landed as `c8fed88`. **No arm was a clean control**: the nonlinear
reparametrization fit at `src/adaptive_warmup_mcmc.jl:379` is gated on
`nonlinear_adapt`, but the linear metric/scale selection at `:381-383` reads the
same pool unconditionally, so `plain` and `fixed_centered` were degraded too.
Full diff in `compare.jl`'s output; the two effects that matter:

**1. The stuck-adaptation defect was a regression artifact, and it is gone.**
The pre-fix run found 2 of 40 adaptive runs where `c` finished at exactly its
starting value on every coordinate — the signature of `optimize!`'s per-index
early return (`nobs(or) > 2 || return idx => value`, `src/Reparametrizations.jl:197`)
firing on a pool too thin to use. After the fix there are none.

| target | before | after |
|---|---|---|
| `eight_schools` | 0 / 8 | 0 / 8 |
| `radon` partially_pooled | 0 / 8 | 0 / 8 |
| `radon` variable_intercept | 0 / 8 | 0 / 8 |
| `seeds_data` | 1 / 8 (seed 8) | **0 / 8** |
| `funnel` | 1 / 8 (seed 7) | **0 / 8** |

The tail it produced is gone with it. The funnel's worst adaptive seed went from
**16.8 to 746.4 min ESS** against a median of 823.1 — a 44× recovery of the worst
case, and the adaptive arm's spread on that target narrowed from 16.8–950.0 to
746.4–936.8. I reported that one-in-eight failure rate as "the single most
actionable defect these measurements surface". It was not a property of the
method; it was this bug.

**2. It flattered the adaptive arm relative to the noncentered arms.** The fix
helped the well-parametrized arms more than the adaptive one, because they were
the ones whose good trajectories the thin pool was discarding:

| target | arm | before | after | |
|---|---|---|---|---|
| `eight_schools` | hand-written noncentered | 39.25 | 51.56 | +31% |
| `eight_schools` | fixed noncentered | 41.37 | 46.22 | +12% |
| `eight_schools` | **adaptive** | 55.21 | 48.40 | **−12%** |
| `funnel` | plain | 0.52 | 1.53 | +194% |

That reversal is why this document no longer claims adaptive beats the
hand-written noncentered model on eight_schools. On the regressed code it
appeared to, by 41%. On correct code it trails by 6%. The conclusion that
survives unchanged is the radon one, where adaptive's margin over every fixed
option is 2.3–2.9× before and after.

## Cost on the gradient hot path

`ReparametrizedProblem` re-differentiates `ljac_(x_) + dot(g_y, y_)` at every
`logdensity_and_gradient`. Nanoseconds per call, 2000 calls at random positions:

| target | bare | wrapped, no-op | wrapped, live | no-op cost | live cost |
|---|---|---|---|---|---|
| `eight_schools-eight_schools_centered` | 544 | 2746 | 2651 | ×5.05 | ×4.87 |
| `radon_mn-radon_partially_pooled_centered` | 33223 | 413266 | 445001 | ×12.44 | ×13.39 |
| `radon_mn-radon_variable_intercept_centered` | 44180 | 228920 | 220168 | ×5.18 | ×4.98 |
| `seeds_data-seeds_centered_model` | 2795 | 21985 | 21349 | ×7.87 | ×7.64 |
| `funnel` | 55 | 822 | 867 | ×14.96 | ×15.78 |

**The no-op and live columns agree to within noise on every target, on both
runs.** An identity reparametrization — source `c` equal to target `c`, so the
transform provably does nothing — pays the same ×4.9–×15.8 as a live one. The
cost is the AD machinery itself, not the reparametrization.

The two radon rows are also the two largest wall-clock losses; the ×13.4 on
`radon_partially_pooled` is what turns a 2.3× per-gradient win into a 3.4×
wall-clock loss. The `plain` arm calls BridgeStan's own gradient with no Julia AD
in the loop at all, so this comparison is what a user actually experiences, not a
like-for-like AD comparison.

## Harness correctness

Two things this benchmark had to get right that are not obvious from the API, and
which would each have silently corrupted the fixed-parametrization arms:

- **`nonlinear_adapt=false` skips the finalization back-transform.** The flag
  gates both the reparametrization fit (`adaptive_warmup_mcmc.jl:379`) and the
  back-transform (`:397`), so a fixed-`c` run returns draws in the *source*
  frame, not the model's. Every fixed arm here therefore applies
  `WarmupHMC.reparametrize!` exactly once itself (`common.jl:252`); the adaptive
  arms must not, because the sampler has already done it. Without that, the
  fixed-noncentered arm would have been scored on draws that were never mapped
  back — measurably wrong, and wrong in the direction that would have made
  adaptive look better.
- **The no-op arm really is a no-op.** `plain` and `fixed c = centered` are
  mathematically the same sampler, and agree to the last gradient evaluation on
  every seed for `seeds_data` (101.7 min ESS, 13068 grads, both arms) and the
  funnel (31.7, 26799) — while differing on eight_schools and both radon models.
  The predictor is whether the spec's **location** is a constant: it is for
  exactly those two targets and a closure over the position vector for the rest,
  and reconstructing `loc + exp(s)·((x − loc)/exp(s))` is the identity only up to
  rounding once `loc` moves with the position. NUTS then amplifies a 1e-16
  gradient difference into a different trajectory. Where the arithmetic permits
  an exact identity the wrapper delivers one; elsewhere the arms stay within seed
  noise. Same split on both runs.

## What this does not measure

- **Correctness.** These are efficiency measurements. Nothing here verifies the
  draws are from the right distribution — no reference-posterior comparison. That
  is `WarmupHMC:reparam-verify`'s scope, and **no performance claim should ship
  before the two are read together.** An efficiency win on incorrect draws is
  worse than no claim.
- **One chain per run.** No R-hat, no between-chain diagnostics. `min ESS` is
  within-chain, computed from the draws by `MCMCDiagnosticTools`, not read from
  `result.ess` (which is all zeros unless `monitor_ess=true`).
- **Wall-clock is machine-specific** (`strato2`, single-threaded BLAS). ESS per
  gradient evaluation is the portable number, which is why the verdict is stated
  in it.
- **Five targets.** Every posteriordb posterior with a ready non-empty spec in
  `web/src/posteriordb_reparametrizations.jl`, plus one synthetic funnel because
  posteriordb ships none.
- **One sampler.** Single-chain `adaptive_warmup_mcmc` only.
  `clustered_warmup_mcmc` has no reparametrization hooks at all and
  `cooperative_warmup_mcmc` was not measured.
