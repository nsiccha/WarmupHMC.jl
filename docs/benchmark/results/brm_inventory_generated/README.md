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
- WarmupHMC: `6adc5ea679722d3b7b13a5b5d6c12d497b10c2d7` (`src/` clean)
- BRM: `a5e118b0119e29c137d80d887b571db2d77af8e1`
- StanBlocks: `d520980f98cc90141967a47dd6520fb5bb6e3f31`
- Run from one environment developed against fixed clones of those two and of
  MutatingFunctions `18dd9e5`, OutputSignatures `7e16ea9` and Treebars
  `8bde866`, so a shared checkout moving mid-campaign cannot change what runs
- 12 seeds × 500 retained draws × 3 arms × 2 flag values × 16 models = 1152 rows
- Every arm/flag path received a 50-draw untimed preflight; recorded flag order
  alternated by seed to remove systematic JIT/order bias from wall comparisons
- Process exit: 0; elapsed: 6639 seconds; summed in-row sampling time: 5037.05 seconds
- Result: 1152/1152 rows usable; zero exceptions or crashes
- Controls: all 384 `(model, bare arm, seed)` flag pairs were identical on
  gradient count and constrained-space minimum ESS
- Non-centered and centered programs have equal unconstrained dimension on all
  16 models (the docs build asserts it)

Divergences over distinct trajectories (flag off): 248 non-centered, 586 static
centered.

`nonlinear_adapt=true` on the adaptive wrapper, paired against the flag-off run
of the same seed (medians over seeds):

| spec | seeds whose trajectory changed | ESS/gradient, on ÷ off | ESS/second, on ÷ off |
|---|---|---|---|
| `lme4:dyestuff_re` | 0/12 | 1.00 | 0.31 |
| `lme4:sleepstudy_slope` | 0/12 | 1.00 | 0.30 |
| `bambi:sleepstudy` | 0/12 | 1.00 | 0.39 |
| `mixed_models_jl:penicillin_crossed` | 12/12 | 4.89 | 1.38 |
| `bambi:radon_partial` | 12/12 | 1.20 | 0.40 |
| `bambi:radon_floor` | 12/12 | 0.89 | 0.27 |
| `bambi:radon_slopes` | 12/12 | 1.18 | 0.32 |
| `bambi:dietox` | 12/12 | 1.22 | 0.41 |
| `vasishth:meta_sbi` | 0/12 | 1.00 | 0.20 |
| `kruschke:fruitfly_anhecova` | 9/12 | 1.02 | 0.47 |
| `burkner_papers:epilepsy_simple` | 12/12 | 2.06 | 0.42 |
| `kruschke:therapeutic_touch` | 7/12 | 1.16 | 0.39 |
| `bambi:hierarchical_binomial_partial` | 12/12 | 1.17 | 0.25 |
| `mixed_models_jl:contraception_glmm` | 12/12 | 1.32 | 0.68 |
| `bambi:predict_new_groups` | 0/12 | 1.00 | 0.36 |
| `vasishth:n400_crossed` | 12/12 | 1.17 | 0.89 |

On five models adaptation keeps the generated non-centered endpoint on every
seed (ratio exactly 1.00); where it moves, it pays per gradient on all but
`radon_floor` (0.89), most on `penicillin_crossed` (4.89) and
`epilepsy_simple` (2.06). In wall-clock it wins only on `penicillin_crossed`
(1.38), because the wrapper's runtime is paid on every call whether or not the
trajectory changes. The three specs of the previous checked-in run (WarmupHMC
`9039668`, BRM `aed667cf`) read the same way — `dyestuff_re` and
`sleepstudy_slope` exactly 1.0, `dietox` 1.25 then and 1.22 now. An earlier
draft of this revision showed 2–7× gains on `dyestuff_re`, `sleepstudy_slope`,
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
`ab03898f37e97eeb983c286675ea8f9ded219796b485a1d38f5720bd0f5ce2f9`.

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
