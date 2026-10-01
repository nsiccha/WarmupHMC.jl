# BRM historical-inventory generated-model benchmark

This is the focused real-data rerun that consumes BayesianRegressionModels'
landed historical-inventory translations. It is deliberately separate from
`../brm_catalogue/`, whose builders were transcribed by hand and therefore do
not test catalogue-to-BRM generation.

## Exact scope

`run_brm_inventory_benchmark.jl` runs every spec in its `SPECS` list — the eight
published controls plus the historical-gallery tranche, 16 rows from BRM's
checked-in `research/historical_model_inventory/translations.tsv`:
`lme4:dyestuff_re`, `lme4:sleepstudy_slope`, `bambi:sleepstudy`,
`mixed_models_jl:penicillin_crossed`, `bambi:radon_partial`, `bambi:radon_floor`,
`bambi:radon_slopes`, `bambi:dietox`, `vasishth:meta_sbi`,
`kruschke:fruitfly_anhecova`, `burkner_papers:epilepsy_simple`,
`kruschke:therapeutic_touch`, `bambi:hierarchical_binomial_partial`,
`mixed_models_jl:contraception_glmm`, `bambi:predict_new_groups`,
`vasishth:n400_crossed`.

For every row, the runner asserts the translation is ready, cross-checks the
body and finite-gradient capability receipt against `model_matrix.tsv`, and
evaluates the row's `current_brm_body` through BRM's `_brm` and `SBBRMI` path. No
formula is copied into the runner. The inventory has no generic real-data
loader, so the runner owns only the documented column adapters and records each
input CSV's URL and SHA-256.

Each generated model is exercised as BRM's conventional non-centered model
(`total_groups=()`, opting out of BRM's automatic exact totals), BRM's static
centered model, and `BRM.adaptive_centering_problem`. Both values of
WarmupHMC's `nonlinear_adapt` flag are run so the bare generated models act as
negative controls.

## Recorded run

- Host: `strato2`
- Julia: 1.10.11; BLAS threads: 1
- WarmupHMC: `b5c0b956e3373ee35070200f24e490b56cc43ea4` (`src/` clean)
- BRM: `a5e118b0119e29c137d80d887b571db2d77af8e1`
- StanBlocks: `d520980f98cc90141967a47dd6520fb5bb6e3f31`
- Run from one environment developed against fixed clones of those two and of
  MutatingFunctions `18dd9e5`, OutputSignatures `7e16ea9` and Treebars
  `8bde866`, so a shared checkout moving mid-campaign cannot change what runs
- 12 seeds × 500 retained draws × 3 arms × 2 flag values × 16 models = 1152 rows
- Every arm/flag path received a 50-draw untimed preflight; recorded flag order
  alternated by seed to remove systematic JIT/order bias from wall comparisons
- Detached run, so no process exit status was captured; `run_finished_at` is
  recorded and the runner logged its final write of all 1152 rows. Elapsed:
  6262 seconds; summed in-row sampling time: 4416.45 seconds
- Result: 1152/1152 rows usable; zero exceptions or crashes
- Controls: all 384 `(model, bare arm, seed)` flag pairs were identical on
  gradient count and constrained-space minimum ESS
- Non-centered and centered programs have equal unconstrained dimension on all
  16 models (the docs build asserts it)

Divergences over distinct trajectories (flag off): 69 non-centered, 459 static
centered.

`nonlinear_adapt=true` on the adaptive wrapper, paired against the flag-off run
of the same seed (medians over seeds):

| spec | seeds whose trajectory changed | ESS/gradient, on ÷ off | ESS/second, on ÷ off |
|---|---|---|---|
| `lme4:dyestuff_re` | 0/12 | 1.00 | 0.49 |
| `lme4:sleepstudy_slope` | 0/12 | 1.00 | 0.45 |
| `bambi:sleepstudy` | 0/12 | 1.00 | 0.42 |
| `mixed_models_jl:penicillin_crossed` | 12/12 | 4.17 | 1.25 |
| `bambi:radon_partial` | 12/12 | 1.56 | 0.44 |
| `bambi:radon_floor` | 12/12 | 1.39 | 0.41 |
| `bambi:radon_slopes` | 12/12 | 1.85 | 0.52 |
| `bambi:dietox` | 12/12 | 0.98 | 0.31 |
| `vasishth:meta_sbi` | 0/12 | 1.00 | 0.35 |
| `kruschke:fruitfly_anhecova` | 5/12 | 1.00 | 0.62 |
| `burkner_papers:epilepsy_simple` | 12/12 | 1.91 | 0.52 |
| `kruschke:therapeutic_touch` | 9/12 | 1.06 | 0.38 |
| `bambi:hierarchical_binomial_partial` | 12/12 | 1.50 | 0.29 |
| `mixed_models_jl:contraception_glmm` | 12/12 | 1.71 | 0.79 |
| `bambi:predict_new_groups` | 0/12 | 1.00 | 0.34 |
| `vasishth:n400_crossed` | 12/12 | 1.26 | 0.94 |

On five models adaptation keeps the generated non-centered endpoint on every
seed (ratio exactly 1.00); where it moves, it pays per gradient on all but
`dietox` (0.98) and `fruitfly_anhecova` (1.00, with 5 of 12 seeds moved), most on
`penicillin_crossed` (4.17), `epilepsy_simple` (1.91) and `radon_slopes` (1.85).
In wall-clock it wins only on `penicillin_crossed` (1.25), because the wrapper's
runtime is paid on every call whether or not the trajectory changes.

Against the artifact this one replaced (WarmupHMC `6adc5ea6`, the same BRM and
StanBlocks pins), the per-gradient ratio rose on six of the eleven models where
adaptation moves and fell on five; `radon_floor`, the one model that paid there
(0.89), now gains (1.39), and `dietox` went from 1.22 to 0.98. Only WarmupHMC
moved between the two artifacts. Its `src/` changes in that span include
`7b1d694`, which restored Pathfinder's low-rank initial factor; nothing here
isolates which change moved which row. `dyestuff_re` and `sleepstudy_slope`
read exactly 1.0 at every pin, including the three-spec run before that
(WarmupHMC `9039668`, BRM `aed667cf`). An earlier
draft of the `6adc5ea6` revision showed 2–7× gains on `dyestuff_re`, `sleepstudy_slope`,
`meta_sbi` and `predict_new_groups`; those came from BRM's new default exact-
totals representation having replaced the non-centered program (one coordinate
fewer), not from adaptation, and the runner now opts out of it
(`total_groups=()`). The docs page derives the full gradient and runtime
comparison tables from the raw rows.

The earlier generated run against BRM tree `79cc4a7` is preserved under
`regression_79cc4a7/`, not as supported performance evidence. That accessor was
stable in this run but changed the legacy floating-point operation tree by one
ULP. The corrected bit-exact accessor kept every non-timing benchmark field
identical here, while making the adaptive wrapper 3.7×–7.4× slower than the
invalid implementation. Keeping that rejected artifact makes the performance
cost auditable without allowing its timings into the headline tables.

`rows.json` is the source of truth. Its SHA-256 is
`3c0890d9d60094dadf08ebf03c2ea43016412cb97ae77b8758032e4330a2f495`.

## Reproduction

Use a Julia environment developed at the three package commits recorded above
(the six-path `develop` call is in `docs/benchmark/brm/Project.toml`'s header):

```sh
KB_COMPACT_KEEP_LOG=1 \
BRMI_SEEDS=12 BRMI_DRAWS=500 BRMI_PREFLIGHT_DRAWS=50 \
BRMI_OUT=docs/benchmark/results/brm_inventory_generated/rows.json \
BRMI_DATA_CACHE=/tmp/brm-inventory-data \
kb-run-compact nice -n 19 ionice -c 3 \
julia --startup-file=no --project=/path/to/pinned/environment \
  docs/benchmark/run_brm_inventory_benchmark.jl
```

This is evidence for the sixteen inventory rows above. It is not a claim that
all 359 historical catalogue rows have real-data loaders or runnable
translations.
