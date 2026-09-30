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
- WarmupHMC: `da7e99cc38bdde6c05d9040de8662a42e1c0ce7a` (`src/` clean)
- BRM: `a5e118b0119e29c137d80d887b571db2d77af8e1`
- StanBlocks: `d520980f98cc90141967a47dd6520fb5bb6e3f31`
- 12 seeds × 500 retained draws × 3 arms × 2 flag values × 16 models = 1152 rows
- Every arm/flag path received a 50-draw untimed preflight; recorded flag order
  alternated by seed to remove systematic JIT/order bias from wall comparisons
- Process exit: 0; elapsed: 9498 seconds; summed in-row sampling time: 6431.32 seconds
- Result: 1149/1152 rows usable; zero exceptions or crashes. The three failed
  rows are one trajectory: `bambi:sleepstudy` seed 9 on the non-centered model
  (both flag values) and the adaptive wrapper with the flag off, which follows
  the same path — all draws constant (902 coordinates), recorded as an explicit
  statistical failure
- Controls: all 384 `(model, bare arm, seed)` flag pairs were identical on
  gradient count and constrained-space minimum ESS

Divergences over distinct trajectories (flag off): 865 non-centered, 586 static
centered.

`nonlinear_adapt=true` on the adaptive wrapper, paired against the flag-off run
of the same seed (medians over seeds with both rows usable):

| spec | seeds whose trajectory changed | ESS/gradient, on ÷ off | ESS/second, on ÷ off |
|---|---|---|---|
| `lme4:dyestuff_re` | 12/12 | 2.74 | 1.61 |
| `lme4:sleepstudy_slope` | 12/12 | 2.25 | 1.15 |
| `bambi:sleepstudy` | 11/12 | 3.04 | 1.16 |
| `mixed_models_jl:penicillin_crossed` | 12/12 | 4.89 | 1.25 |
| `bambi:radon_partial` | 12/12 | 1.05 | 0.46 |
| `bambi:radon_floor` | 12/12 | 0.89 | 0.29 |
| `bambi:radon_slopes` | 12/12 | 1.18 | 0.30 |
| `bambi:dietox` | 6/12 | 1.00 | 0.57 |
| `vasishth:meta_sbi` | 12/12 | 1.91 | 0.70 |
| `kruschke:fruitfly_anhecova` | 9/12 | 1.02 | 0.54 |
| `burkner_papers:epilepsy_simple` | 12/12 | 2.06 | 0.59 |
| `kruschke:therapeutic_touch` | 2/12 | 1.00 | 0.42 |
| `bambi:hierarchical_binomial_partial` | 8/12 | 1.00 | 0.30 |
| `mixed_models_jl:contraception_glmm` | 12/12 | 0.69 | 0.50 |
| `bambi:predict_new_groups` | 12/12 | 7.44 | 3.73 |
| `vasishth:n400_crossed` | 12/12 | 1.17 | 0.89 |

Adaptation pays per gradient on seven rows (1.9–7.4×), is roughly neutral on
the three radon rows and five others (0.89–1.18), and costs 31% on
`contraception_glmm` (0.69); in wall-clock it wins only where the per-gradient
gain is large (five rows), because the wrapper's runtime is paid on every
call. The previous checked-in run (three specs, WarmupHMC
`9039668`, BRM `aed667cf`) found adaptation left `dyestuff_re` and
`sleepstudy_slope` at the non-centered endpoint (ratio exactly 1.0) and gained
1.25× on `dietox`; on this base it moves both of the former and not the
latter. Sampler and BRM both changed in between, so this table does not
attribute the difference. The docs page derives the full gradient and runtime
comparison tables from the raw rows.

The earlier generated run against BRM tree `79cc4a7` is preserved under
`regression_79cc4a7/`, not as supported performance evidence. That accessor was
stable in this run but changed the legacy floating-point operation tree by one
ULP. The corrected bit-exact accessor kept every non-timing benchmark field
identical here, while making the adaptive wrapper 3.7×–7.4× slower than the
invalid implementation. Keeping that rejected artifact makes the performance
cost auditable without allowing its timings into the headline tables.

`rows.json` is the source of truth. Its SHA-256 is
`e228147ef11b557ad3d22ae11818dd4e7be735ac93d11f85d0c0cc2fc99e7aef`.

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
