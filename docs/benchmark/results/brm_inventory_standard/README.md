# BRM inventory-generated standard-warmup comparison

This artifact compares WarmupHMC with DynamicHMC's default, Stan-style warmup
on the same three real-data models generated from BRM's landed historical
inventory. It complements `../brm_inventory_generated/`: that artifact isolates
WarmupHMC's nonlinear-adaptation flag, while this one answers which default
warmup a consumer gets on the generated target.

## Four arms

Every model runs:

- WarmupHMC on BRM's generated default non-centered density;
- WarmupHMC with BRM's adaptive-centering wrapper;
- DynamicHMC defaults on that identical generated non-centered density; and
- DynamicHMC defaults on BRM's separately generated centered density.

DynamicHMC uses `mcmc_with_warmup` unchanged, including its default 1,000-step
warmup. WarmupHMC uses its own defaults and gradient-targeted windows. This is
a defaults comparison, not a fixed-budget mechanism study.

All four arms are wrapped by `WarmupHMC.count_and_time` at the generated BRM
logdensity-and-gradient boundary. For adaptive centering, the wrapper is built
over the counted inner density so BRM's reparametrization hooks remain active.
Thus the gradient column never compares sampler-specific internal counters.

## Recorded run

- Host: `strato2`
- Julia: 1.10.11; BLAS threads: 1
- WarmupHMC: `9039668d8ce1f1f62c7eb3d235ac7dac19a12d5a`
- BayesianRegressionModels: `aed667cfdb304718978750251547819bf5120bda`
- StanBlocks: `329a178a7ad7877da0b58ad2c360d417ddd663f9`
- DynamicHMC: 3.6.1
- 12 seeds × 500 retained draws × 4 arms × 3 models = 144 rows
- One 50-draw untimed preflight per arm; recorded arm order rotates by seed
- Process exit: 0; elapsed: 523 seconds; summed recorded sampling time: 244.49 seconds
- Per-model process blocks: dyestuff 212.8 s, sleepstudy 60.6 s, dietox 219.2 s
- Result: 144/144 rows completed; zero exceptions or crashes
- Every arm returned 500 draws

WarmupHMC and DynamicHMC both had zero divergences on every generated
non-centered trajectory. WarmupHMC adaptive centering also had zero. The
DynamicHMC generated-centered arm recorded 30 dyestuff and 6 sleepstudy
divergences; dietox had zero.

## Seed-paired headline ratios

WarmupHMC on the identical generated non-centered target, divided by DynamicHMC
on that target:

| model | ESS/gradient | wall time | ESS/second |
| --- | ---: | ---: | ---: |
| dyestuff | 3.06× | 0.98× | 1.09× |
| sleepstudy | 4.54× | 0.31× | 2.61× |
| dietox | 2.23× | 0.37× | 1.99× |

WarmupHMC adaptive centering remained 2.66×–4.54× better in ESS/gradient than
DynamicHMC on the default non-centered target, but the corrected bit-exact
Enzyme wrapper reduced ESS/second to 0.10×–0.22×. The docs page derives the
complete absolute and paired tables from `rows.json`; no result number is
stored only in this prose.

`rows.json` is the source of truth. Its SHA-256 is
`0ce2309b22b89b2a25efdc714ca1bfddb962fa2a0b33f25927ade7077b6ef453`.

## Reproduction

```sh
KB_COMPACT_KEEP_LOG=1 \
BRMI_MODE=standard \
BRMI_SEEDS=12 BRMI_DRAWS=500 BRMI_PREFLIGHT_DRAWS=50 \
BRMI_OUT=docs/benchmark/results/brm_inventory_standard/rows.json \
BRMI_DATA_CACHE=/tmp/brm-inventory-data \
kb-run-compact nice -n 19 ionice -c 3 \
julia --startup-file=no --project=/path/to/pinned/environment \
  docs/benchmark/run_brm_inventory_benchmark.jl
```

This remains a focused real-data comparison over three cheap, ready, verbatim
inventory rows. It does not establish a universal sampler ranking or equalize
the samplers' warmup budgets.
