# BRM inventory-generated standard-warmup comparison

This artifact compares WarmupHMC with DynamicHMC's default, Stan-style warmup
on eight real-data models generated from BRM's landed historical inventory. It
also separates static generated parameterizations, the cost of wrapping a
non-centered target, and the effect of fitting nonlinear centerings.

## Six arms

Every model runs:

- WarmupHMC on BRM's generated non-centered density;
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
- WarmupHMC: `605f4b83c65730087fd05bae249d673315345a91`
- BayesianRegressionModels: `16c1c46ab27ac4681871048d8555b36c4a432212`
- StanBlocks: `0eaebfae904d3bffab150dfa2c59632ac783b992`
- DynamicHMC: 3.6.1
- 12 seeds × 500 retained draws × 6 arms × 8 models = 576 rows
- One 50-draw untimed preflight per arm; recorded arm order rotates by seed
- Process exit: 0; runner elapsed: 775.372 seconds; wrapper elapsed: 805 seconds
- Summed recorded sampling time: 559.541 seconds
- Result: 575/576 usable trajectories; every arm returned 500 draws
- Total divergences, including the failed trajectory: 395

One trajectory is an explicit statistical failure rather than a hidden null:
`lme4:dyestuff_re`, WarmupHMC generated-centered, seed 11 returned 500
identical draws (189 constant constrained coordinates) with 207 divergences.
It remains in the design and divergence total, but contributes neither an
invented zero nor a missing value to ESS medians and AoV marks.

The adaptive-centering arm recorded zero divergences over all eight models.
WarmupHMC/DynamicHMC generated-non-centered recorded 66/1 divergences;
generated-centered recorded 221/41; the fixed wrapper recorded 66.

## Seed-paired headline ratios

WarmupHMC on the identical generated non-centered target, divided by
DynamicHMC on that target:

| model | ESS/gradient | wall time | ESS/second |
| --- | ---: | ---: | ---: |
| dyestuff | 3.06× | 0.79× | 1.19× |
| sleepstudy slope | 4.55× | 0.29× | 2.88× |
| sleepstudy (Bambi) | 4.47× | 0.30× | 2.78× |
| penicillin crossed | 1.77× | 0.13× | 1.57× |
| radon partial pooling | 2.97× | 0.24× | 2.97× |
| radon floor | 2.46× | 0.25× | 2.35× |
| radon slopes | 3.47× | 0.21× | 3.16× |
| dietox | 2.23× | 0.36× | 2.06× |

WarmupHMC's generated-centered arm also leads DynamicHMC's generated-centered
arm in ESS/gradient on every model (2.27×–10.00×; 11 paired seeds for dyestuff,
12 elsewhere).

The fixed nonlinear wrapper preserves seed-level ESS/gradient exactly while
adding about 6–13% median wall time. Fitting centerings improves ESS/gradient
materially for penicillin, radon partial pooling, radon floor, and dietox; it
keeps the endpoint result on dyestuff and both sleepstudy rows, and is nearly
neutral on radon slopes. Its extra differentiation/adaptation work means those
gradient improvements do not automatically become ESS/second improvements.

The documentation derives the complete absolute and seed-paired tables plus
two AlgebraOfVega figures from `rows.json`; no plotted summary is stored beside
the raw rows.

`rows.json` is the source of truth. Its SHA-256 is
`829950855f4a367f8fd3c4c9d976ebba5d605286584ec7882eb4a96fea2259c9`.

## Reproduction

```sh
KB_COMPACT_KEEP_LOG=1 \
kb-run-compact env \
  KB_HOST=strato2 \
  BRMI_MODE=standard \
  BRMI_SEEDS=12 BRMI_DRAWS=500 BRMI_PREFLIGHT_DRAWS=50 \
  BRMI_OUT=docs/benchmark/results/brm_inventory_standard/rows.json \
  BRMI_DATA_CACHE=/tmp/brm-inventory-data \
  julia --startup-file=no --project=/path/to/pinned/environment \
    docs/benchmark/run_brm_inventory_benchmark.jl
```

This is a real-data comparison over eight audited, ready, verbatim inventory
rows. It does not establish a universal sampler ranking or equalize the
samplers' warmup budgets. The two radon fixed-effect rows retain their separate
`adapted-but-defensible` historical-source label; translation readiness is not
being presented as exact historical-source identity.
