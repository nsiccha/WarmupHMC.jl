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
git show 0d654a92:docs/benchmark/results/brm_inventory_standard/rows.json > "$TMPDIR/base.json"
julia --startup-file=no --project=docs docs/benchmark/brm_inventory_report.jl \
  docs/benchmark/results/brm_inventory_standard/rows.json "$TMPDIR/base.json"
```

The base is the artifact this one replaced (measured at WarmupHMC `dc9636f8`,
checked in on `dev` at `0d654a92`). Name the version being replaced, not the
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
- WarmupHMC: `6adc5ea679722d3b7b13a5b5d6c12d497b10c2d7`
- BayesianRegressionModels: `a5e118b0119e29c137d80d887b571db2d77af8e1`
- StanBlocks: `d520980f98cc90141967a47dd6520fb5bb6e3f31`
- DynamicHMC: 3.6.1
- Runner + adapters: `d184d83db9985f7160ddc00da1e85be6b4d0f3c895a1bcd4603803d94d28d849`
- Inventory `translations.tsv`: `3abb3199a194f74754c8e02b725dfac42081bc7e302c796cdf097ad1504b633d`
- Inventory `model_matrix.tsv`: `edeed76f24931af8f53e296538d4d4b98c6768c9c5665f3f4dec55919ae78aeb`
- 12 seeds × 500 retained draws × 6 arms × 16 models = 1152 rows
- One 50-draw untimed preflight per arm; recorded arm order rotates by seed
- Runner elapsed: 10594.485 seconds
- Summed recorded sampling time: 9090.667 seconds
- Result: 1152/1152 usable trajectories
- Total divergences, including any failed trajectory: 2329
- `rows.json` SHA-256: `321fa5b4d274672cbefa93ebb34437ef26a7f5a121a09505f64c4902139d34a0`
- Run from one environment developed against fixed clones (BRM and StanBlocks
  above, MutatingFunctions `18dd9e5`, OutputSignatures `7e16ea9`, Treebars
  `8bde866`), so a shared checkout moving mid-campaign cannot change what runs.

Every model/arm/seed cell returned a usable trajectory. One 50-draw untimed
preflight in this run was statistically degenerate and was logged and passed
over (an earlier, stopped run at the same `src/` showed the degenerate
preflights on `lme4:dyestuff_re`'s centered and fixed-wrapper arms). The runner
aborts only on an arm that raises: a degenerate preflight says nothing the
per-seed rows do not record as data.

## Divergences by arm

| default-warmup arm | divergences | trajectories with any |
| --- | ---: | ---: |
| WarmupHMC — generated non-centered | 248 | 9/192 |
| WarmupHMC — generated centered | 586 | 30/192 |
| WarmupHMC — nonlinear wrapper fixed at c=0 | 248 | 9/192 |
| WarmupHMC — adaptive centering | 449 | 10/192 |
| DynamicHMC — generated non-centered | 12 | 4/192 |
| DynamicHMC — generated centered | 786 | 30/192 |

`mixed_models_jl:penicillin_crossed` carries almost all non-centered and
adaptive divergences: 242 and 437 here, against 66 and 0 in the artifact this
replaced. The centered totals are dominated by `kruschke:fruitfly_anhecova` (518
WarmupHMC, 422 DynamicHMC); DynamicHMC's centered arm also records 312 on
`bambi:radon_slopes`, where the replaced artifact had none. DynamicHMC does not
run any WarmupHMC code, so that change — and the others on the DynamicHMC arms —
comes from the generated models: BRM and StanBlocks moved too (below).

## Seed-paired headline ratios

WarmupHMC on the identical generated non-centered target, divided by
DynamicHMC on that target:

| model | ESS/gradient | gradients | wall time | ESS/second | paired seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| dyestuff | 3.13× | 0.32× | 0.52× | 2.02× | 12 |
| sleepstudy slope | 6.21× | 0.19× | 0.22× | 4.96× | 12 |
| sleepstudy (Bambi) | 4.08× | 0.18× | 0.22× | 3.50× | 12 |
| penicillin crossed | 1.67× | 0.11× | 0.11× | 1.60× | 12 |
| radon partial pooling | 3.33× | 0.24× | 0.25× | 3.28× | 12 |
| radon floor | 3.48× | 0.25× | 0.27× | 3.12× | 12 |
| radon slopes | 4.26× | 0.26× | 0.27× | 4.00× | 12 |
| dietox | 3.37× | 0.33× | 0.33× | 2.92× | 12 |
| SBI meta-analysis | 2.37× | 0.37× | 0.68× | 1.29× | 12 |
| fruitfly ANCOVA | 3.81× | 0.17× | 0.21× | 3.40× | 12 |
| epilepsy counts | 5.22× | 0.14× | 0.19× | 3.83× | 12 |
| therapeutic touch | 2.63× | 0.38× | 0.42× | 2.17× | 12 |
| baseball binomial | 2.68× | 0.30× | 0.54× | 1.38× | 12 |
| contraception | 12.67× | 0.08× | 0.09× | 10.27× | 12 |
| pulmonary slopes | 4.88× | 0.21× | 0.25× | 4.11× | 12 |
| N400 crossed | 5.00× | 0.11× | 0.11× | 4.85× | 12 |

ESS/gradient favours WarmupHMC on all sixteen (1.67×–12.67×), and so does
ESS/second (1.29×–10.27×) — including the SBI meta-analysis, which trailed at
0.74× in the replaced artifact.

On the generated *centered* target the same comparison ranges 0.53×–50.52× in
ESS/gradient: WarmupHMC leads on fifteen of sixteen and trails only on
`vasishth:pulmonary_slopes` (0.53×). Every row pairs 12 seeds.

## The nonlinear wrapper's cost, across the pins

The fixed wrapper still preserves seed-level ESS/gradient **exactly**: the
ratio is 1.00× with identical gradient counts on all sixteen models, which is
the invariant this arm exists to check. Its wall-time cost over the unwrapped
non-centered arm is 1.01×–2.97× (N400 crossed to the SBI meta-analysis), down
from the 1.24×–4.86× the replaced artifact recorded.

Per gradient, against the replaced artifact (`dc9636f8` / BRM `d452e97a` /
StanBlocks `7a02d30f` → `6adc5ea6` / `a5e118b0` / `d520980f`):

| median µs per gradient | non-centered | centered | fixed wrapper | adaptive |
| --- | ---: | ---: | ---: | ---: |
| dyestuff | 8.6 → 5.6 | 5.1 → 4.7 | 25.9 → 12.5 | 28.2 → 19.8 |
| sleepstudy slope | 23.4 → 14.1 | 22.3 → 20.9 | 50.5 → 22.0 | 65.6 → 45.7 |
| penicillin crossed | 13.9 → 14.1 | 12.2 → 11.6 | 51.0 → 17.0 | 82.5 → 59.6 |
| radon partial pooling | 50.7 → 49.7 | 43.2 → 46.5 | 127.2 → 60.5 | 245.5 → 189.6 |
| radon slopes | 58.7 → 71.6 | 90.7 → 103.4 | 146.1 → 93.9 | 233.0 → 376.7 |
| dietox | 54.4 → 50.6 | 87.1 → 84.9 | 126.8 → 58.8 | 222.9 → 162.1 |
| N400 crossed | 2249.6 → 1853.5 | 2284.6 → 2443.8 | 2789.9 → 1894.0 | 3359.2 → 2565.3 |

(Selected rows; the report prints all sixteen.) The fixed wrapper's
per-gradient cost roughly halved on most models, and the adaptive arm fell too
except on `radon_slopes`. **This cannot be attributed to WarmupHMC alone**: all
three pins moved, only 25 of 96 shared model × arm median gradient counts are
identical across the two artifacts (the report lists the rest), and the
DynamicHMC arms moved as well. Read the column as "what the published stack
costs now", not as a WarmupHMC change.

## Fitting centerings

Isolated from the wrapper — adaptive over fixed, so the wrapper's overhead
divides out — fitting centerings improves ESS/gradient on ten of sixteen models
and degrades it on one (`bambi:radon_floor`, 0.89×). The large wins are
penicillin crossed (4.87×) and epilepsy counts (2.06×); contraception (1.31×),
dietox (1.21×), radon partial pooling and radon slopes (1.18× each), baseball
binomial and N400 crossed (1.17× each), therapeutic touch (1.16×) and fruitfly
(1.01×) follow. The other five sit at exactly 1.00×, i.e. the fitted centering
returns the generated endpoint.

Its extra differentiation and adaptation work means those gradient
improvements do not become ESS/second improvements except on penicillin
crossed (1.27×).

## N400 wall times and box load

`vasishth:n400_crossed` is 7777.2 s of the run's 9090.7 s of recorded sampling
(86%). Unlike the replaced artifact's single contention window, `strato2`
carried other agents' jobs throughout this run (load 11–30 on 8 cores), so the
question is whether that load shows in the rows. Normalising each cell's
`wall_s` by its own gradient count leaves the per-gradient cost, which varies by
at most 7% across the twelve seeds in every arm:

| arm | s/gradient range | max/median | median wall s | max wall s |
| --- | --- | ---: | ---: | ---: |
| WarmupHMC — generated non-centered | 0.00178–0.00198 | 1.07 | 20.65 | 29.65 |
| WarmupHMC — generated centered | 0.00239–0.00261 | 1.07 | 65.09 | 136.50 |
| WarmupHMC — nonlinear wrapper fixed at c=0 | 0.00181–0.00197 | 1.04 | 20.59 | 30.03 |
| WarmupHMC — adaptive centering | 0.00248–0.00268 | 1.05 | 25.21 | 41.13 |
| DynamicHMC — generated non-centered | 0.00174–0.00189 | 1.06 | 188.44 | 206.01 |
| DynamicHMC — generated centered | 0.00237–0.00255 | 1.05 | 312.59 | 353.71 |

The widest raw cell is WarmupHMC generated-centered seed 1 at 2.10× its arm
median: 56,199 gradients, so geometry, not load. The large per-gradient
spreads elsewhere on this page are on millisecond-scale densities, where a
shared host shows; the N400 medians are not.

## Figures and provenance

The documentation derives the complete absolute and seed-paired tables plus
the AlgebraOfVega figures from `rows.json`; no plotted summary is stored beside
the raw rows.

`rows.json` is the source of truth. Its SHA-256 is
`321fa5b4d274672cbefa93ebb34437ef26a7f5a121a09505f64c4902139d34a0`.

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
