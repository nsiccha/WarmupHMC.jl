# BRM inventory-generated standard-warmup comparison

This artifact compares WarmupHMC with DynamicHMC's default, Stan-style warmup
on sixteen real-data models generated from BRM's landed historical inventory.
It also separates static generated parameterizations, the cost of wrapping a
non-centered target, and the effect of fitting nonlinear centerings.

Every figure below is printed by `docs/benchmark/brm_inventory_report.jl`.
Regenerate the rows, re-run that script, and paste what it prints; do not
transcribe from a rendered table. The cross-pin comparison further down is the
one section that needs a second input — the previously published `rows.json`,
which the script takes as an optional second argument and which comes out of
git rather than off anyone's disk:

```sh
git show 06cd053:docs/benchmark/results/brm_inventory_standard/rows.json > "$TMPDIR/base.json"
julia --startup-file=no --project=docs docs/benchmark/brm_inventory_report.jl \
  docs/benchmark/results/brm_inventory_standard/rows.json "$TMPDIR/base.json"
```

The base is the artifact this one replaced (measured at WarmupHMC `6adc5ea6`,
checked in on `dev` at `06cd053`). Name the version being replaced, not the
commit an artifact records — those are different files.

## Six arms

Every model runs:

- WarmupHMC on BRM's generated non-centered density (conventional representation:
  the runner passes `total_groups=()`, opting out of BRM's automatic exact totals);
- WarmupHMC on BRM's separately generated centered density;
- WarmupHMC on the nonlinear wrapper fixed at the generated `c=0` endpoint;
- WarmupHMC with nonlinear centering adaptation enabled;
- DynamicHMC defaults on the generated non-centered density; and
- DynamicHMC defaults on the generated centered density.

DynamicHMC uses `mcmc_with_warmup` unchanged, including its default 1,000-step
warmup. WarmupHMC uses its own defaults and gradient-targeted windows. This is
a defaults comparison, not a fixed-budget mechanism study.

All six arms are measured through `WarmupHMC.count_and_time` at the generated
BRM logdensity-and-gradient boundary. For adaptive centering, the
reparametrizer is built over the counted inner density so its hooks remain
active. The gradient column therefore does not mix sampler-specific internal
counters.

## Recorded run

- Logical compute host: `strato2`
- Julia: 1.10.11; BLAS threads: 1
- WarmupHMC: `b5c0b956e3373ee35070200f24e490b56cc43ea4`
- BayesianRegressionModels: `a5e118b0119e29c137d80d887b571db2d77af8e1`
- StanBlocks: `d520980f98cc90141967a47dd6520fb5bb6e3f31`
- DynamicHMC: 3.6.1
- Runner + adapters: `d184d83db9985f7160ddc00da1e85be6b4d0f3c895a1bcd4603803d94d28d849`
- Inventory `translations.tsv`: `3abb3199a194f74754c8e02b725dfac42081bc7e302c796cdf097ad1504b633d`
- Inventory `model_matrix.tsv`: `edeed76f24931af8f53e296538d4d4b98c6768c9c5665f3f4dec55919ae78aeb`
- 12 seeds × 500 retained draws × 6 arms × 16 models = 1152 rows
- One 50-draw untimed preflight per arm; recorded arm order rotates by seed
- Runner elapsed: 18737.439 seconds
- Summed recorded sampling time: 14493.394 seconds
- Result: 1152/1152 usable trajectories
- Total divergences, including any failed trajectory: 1420
- `rows.json` SHA-256: `b86dcafba1afffd5a5cf0f451bf68246c922bb070721b2d53f2d958ba1bd7c53`
- Run from one environment developed against fixed clones (BRM and StanBlocks
  above, MutatingFunctions `18dd9e5`, OutputSignatures `7e16ea9`, Treebars
  `8bde866`), so a shared checkout moving mid-campaign cannot change what runs.

Every model/arm/seed cell returned a usable trajectory, and no 50-draw untimed
preflight in this run was statistically degenerate. The runner logs and passes
over one that is: it aborts only on an arm that raises, because a degenerate
preflight says nothing the per-seed rows do not record as data. An earlier
attempt at this same `src/` was stopped by the host's out-of-memory guard
during its last spec; this artifact is a clean full re-run, not a splice of
that partial one.

## Divergences by arm

| default-warmup arm | divergences | trajectories with any |
| --- | ---: | ---: |
| WarmupHMC — generated non-centered | 69 | 11/192 |
| WarmupHMC — generated centered | 459 | 28/192 |
| WarmupHMC — nonlinear wrapper fixed at c=0 | 69 | 11/192 |
| WarmupHMC — adaptive centering | 25 | 9/192 |
| DynamicHMC — generated non-centered | 12 | 4/192 |
| DynamicHMC — generated centered | 786 | 30/192 |

The WarmupHMC non-centered and adaptive totals fell from 248 and 449 in the
artifact this replaced to 69 and 25. `mixed_models_jl:penicillin_crossed`, which
carried almost all of them, fell from 921 divergences across its arms to 112;
`vasishth:meta_sbi` rose from 30 to 71. The centered totals are dominated by
`kruschke:fruitfly_anhecova` (848 across its arms, from 940), and
DynamicHMC's centered arm still records 312 on `bambi:radon_slopes`.

The DynamicHMC arms are an exact control here. Only the WarmupHMC pin moved
between the two artifacts (BRM, StanBlocks, inventory, host, Julia and BLAS
threads are unchanged, below), DynamicHMC runs no WarmupHMC code, and its two
arms reproduce the replaced artifact's divergence totals (12 and 786) and every
one of their 32 model × arm median gradient counts. Every change on the
WarmupHMC arms is therefore WarmupHMC's own, `6adc5ea6` → `b5c0b956`.

## Seed-paired headline ratios

WarmupHMC on the identical generated non-centered target, divided by
DynamicHMC on that target:

| model | ESS/gradient | gradients | wall time | ESS/second | paired seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| dyestuff | 3.14× | 0.31× | 0.81× | 1.34× | 12 |
| sleepstudy slope | 6.03× | 0.19× | 0.31× | 3.39× | 12 |
| sleepstudy (Bambi) | 3.81× | 0.18× | 0.33× | 2.10× | 12 |
| penicillin crossed | 2.51× | 0.11× | 0.13× | 2.22× | 12 |
| radon partial pooling | 2.76× | 0.24× | 0.26× | 2.72× | 12 |
| radon floor | 2.85× | 0.25× | 0.25× | 3.11× | 12 |
| radon slopes | 3.40× | 0.25× | 0.27× | 3.16× | 12 |
| dietox | 3.13× | 0.34× | 0.35× | 2.94× | 12 |
| SBI meta-analysis | 2.65× | 0.37× | 1.69× | 0.71× | 12 |
| fruitfly ANCOVA | 3.91× | 0.16× | 0.31× | 2.07× | 12 |
| epilepsy counts | 5.07× | 0.16× | 0.25× | 3.00× | 12 |
| therapeutic touch | 3.32× | 0.34× | 0.45× | 2.20× | 12 |
| baseball binomial | 2.22× | 0.28× | 0.49× | 1.32× | 12 |
| contraception | 9.16× | 0.08× | 0.09× | 8.27× | 12 |
| pulmonary slopes | 4.72× | 0.21× | 0.25× | 4.25× | 12 |
| N400 crossed | 4.71× | 0.11× | 0.11× | 4.35× | 12 |

ESS/gradient favours WarmupHMC on all sixteen (2.22×–9.16×). ESS/second favours
it on fifteen (1.32×–8.27×) and trails on the SBI meta-analysis (0.71×; 1.29× in
the replaced artifact): a millisecond-scale density on which WarmupHMC's wall
time per gradient more than doubled between the pins, while DynamicHMC's did not
(next section).

On the generated *centered* target the same comparison ranges 0.45×–52.65× in
ESS/gradient: WarmupHMC leads on fifteen of sixteen and trails only on
pulmonary slopes (`bambi:predict_new_groups`, 0.45×). Every row pairs 12 seeds.

## The nonlinear wrapper's cost, across the pins

The fixed wrapper still preserves seed-level ESS/gradient **exactly**: the
ratio is 1.00× with identical gradient counts on all sixteen models, which is
the invariant this arm exists to check. Its wall-time cost over the unwrapped
non-centered arm is 1.04×–2.09× (N400 crossed to baseball binomial), against
1.01×–2.97× in the replaced artifact.

Per gradient, against the replaced artifact. Only WarmupHMC moved
(`6adc5ea6` → `b5c0b956`; BRM `a5e118b0` and StanBlocks `d520980f` on both), so
the DynamicHMC column — identical gradient counts on both sides — measures the
host, not the code:

| median µs per gradient | non-centered | centered | fixed wrapper | adaptive | DynamicHMC non-centered |
| --- | ---: | ---: | ---: | ---: | ---: |
| dyestuff | 5.6 → 10.0 | 4.7 → 4.7 | 12.5 → 16.4 | 19.8 → 24.9 | 3.5 → 3.3 |
| sleepstudy slope | 14.1 → 21.4 | 20.9 → 22.1 | 22.0 → 30.3 | 45.7 → 53.0 | 12.0 → 12.3 |
| SBI meta-analysis | 4.2 → 9.9 | 3.6 → 5.4 | 10.3 → 15.3 | 25.7 → 31.2 | 1.9 → 2.1 |
| fruitfly ANCOVA | 13.0 → 21.4 | 12.5 → 12.7 | 21.0 → 26.3 | 39.4 → 38.6 | 11.3 → 10.7 |
| penicillin crossed | 14.1 → 14.1 | 11.6 → 13.6 | 17.0 → 17.7 | 59.6 → 60.0 | 13.3 → 13.3 |
| radon slopes | 71.6 → 71.1 | 103.4 → 101.4 | 93.9 → 90.6 | 376.7 → 353.8 | 69.5 → 63.8 |
| contraception | 79.9 → 80.5 | 70.2 → 71.7 | 88.1 → 87.9 | 171.3 → 185.2 | 71.6 → 74.3 |
| N400 crossed | 1853.5 → 2421.7 | 2443.8 → 3326.1 | 1894.0 → 2960.3 | 2565.3 → 4677.5 | 1772.6 → 3144.5 |

(Selected rows; the report prints all sixteen and all six arms.) Two different
things moved. On the small densities, where sampler bookkeeping and per-run
fixed costs are a visible share of each gradient, the WarmupHMC non-centered
and fixed-wrapper arms got slower per gradient (dyestuff, both sleepstudy
models, the SBI meta-analysis and fruitfly: +42% to +138% non-centered), while
DynamicHMC on the same models, interleaved in the same run, stayed within 11%.
That is a WarmupHMC-side change between the pins; on the larger densities
(radon, penicillin, contraception, pulmonary) it disappears into the density
cost. On N400 crossed, every arm got slower, DynamicHMC's included (1.77×
non-centered, 1.32× centered) — that is host load, covered below. The per-arm
gradient counts that do differ between the two artifacts are all WarmupHMC's
(64 of 96 model × arm cells; the report lists them).

## Fitting centerings

Isolated from the wrapper — adaptive over fixed, so the wrapper's overhead
divides out — fitting centerings improves ESS/gradient on nine of sixteen models
and degrades it on one (`bambi:dietox`, 0.98×). The large wins are penicillin
crossed (4.15×), epilepsy counts (1.90×) and radon slopes (1.85×);
contraception (1.70×), radon partial pooling (1.56×), baseball binomial
(1.50×), radon floor (1.39×), N400 crossed (1.26×) and therapeutic touch
(1.06×) follow. `bambi:radon_floor`, the one degradation in the replaced
artifact (0.89×), now gains. The other six sit at exactly 1.00×, i.e. the
fitted centering returns the generated endpoint.

Its extra differentiation and adaptation work means those gradient
improvements do not become ESS/second improvements except on penicillin
crossed (1.28×); N400 crossed comes closest otherwise (0.97×).

## N400 wall times and box load

`vasishth:n400_crossed` is 13111.5 s of the run's 14493.4 s of recorded sampling
(90%). This run's `strato2` was far more loaded than the replaced artifact's
(1-minute load 61 on 8 cores when it started), and this time the load does
show in the rows. Normalising each cell's `wall_s` by its own gradient count
leaves the per-gradient cost, which now spans up to 3.2× across the twelve
seeds of an arm (it was within 7% before):

| arm | s/gradient range | max/median | median wall s | max wall s |
| --- | --- | ---: | ---: | ---: |
| WarmupHMC — generated non-centered | 0.00191–0.00531 | 2.19 | 32.38 | 83.55 |
| WarmupHMC — generated centered | 0.00247–0.00719 | 2.16 | 117.03 | 418.43 |
| WarmupHMC — nonlinear wrapper fixed at c=0 | 0.00194–0.00559 | 1.89 | 36.60 | 87.96 |
| WarmupHMC — adaptive centering | 0.00256–0.01026 | 2.19 | 51.71 | 118.55 |
| DynamicHMC — generated non-centered | 0.00177–0.00575 | 1.83 | 318.87 | 583.14 |
| DynamicHMC — generated centered | 0.00240–0.00612 | 1.91 | 416.54 | 807.19 |

Two things confirm load rather than sampler behaviour. The DynamicHMC arms,
whose gradient counts and ESS reproduce the replaced artifact exactly, cost
1.77× (non-centered) and 1.32× (centered) more per gradient than they did
there. And seed 7 accounts for three of the five widest raw cells: WarmupHMC
generated-centered seed 7 is 3.58× its arm's median wall time (418.43 s), but
only 2.2× its arm's median gradient count (59,602 against 27,121); the rest is
load. Read this spec's wall times and ESS/second as load-inflated. ESS/gradient
and gradient counts are unaffected, and the seed-paired wall-time ratios above
pair arms that ran interleaved under the same load.

## Figures and provenance

The documentation derives the complete absolute and seed-paired tables plus
the AlgebraOfVega figures from `rows.json`; no plotted summary is stored beside
the raw rows.

`rows.json` is the source of truth. Its SHA-256 is
`b86dcafba1afffd5a5cf0f451bf68246c922bb070721b2d53f2d958ba1bd7c53`.

## Reproduction

```sh
KB_COMPACT_KEEP_LOG=1 \
kb-run-compact env \
  KB_HOST=strato2 \
  BRMI_MODE=standard \
  BRMI_SEEDS=12 BRMI_DRAWS=500 BRMI_PREFLIGHT_DRAWS=50 \
  BRMI_OUT=docs/benchmark/results/brm_inventory_standard/rows.json \
  BRMI_DATA_CACHE="$TMPDIR/brm-inventory-data" \
  julia --startup-file=no --project=/path/to/pinned/environment \
    docs/benchmark/run_brm_inventory_benchmark.jl
```

Acceptance is `docs/benchmark/verify_brm_inventory_standard.jl`, which names
all sixteen expected specs explicitly so a silently dropped model fails the
gate instead of shrinking the matrix.

This is a real-data comparison over sixteen audited, ready, verbatim inventory
rows. It does not establish a universal sampler ranking or equalize the
samplers' warmup budgets. The two radon fixed-effect rows retain their separate
`adapted-but-defensible` historical-source label; translation readiness is not
being presented as exact historical-source identity. The widest random-effect
block reachable from the inventory is `K=2` (`kruschke:fruitfly_anhecova`, on
`CompanionNumber`); `results/brm_inventory_high_k/` records that limit as a
measured negative rather than an untried direction.
