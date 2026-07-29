# BRM `68edfeab` adaptive-centering crash receipt

This directory preserves the evidence that blocked replacement of `rows.json`
after the BRM accessor-speed fix. It is a regression receipt, not a benchmark
result: the candidate process crashes, so no post-fix performance claim is
published from it.

## Fixed inputs

- WarmupHMC: `9039668d8ce1f1f62c7eb3d235ac7dac19a12d5a`
- StanBlocks: `329a178a7ad7877da0b58ad2c360d417ddd663f9`
- Treebars: `c02aa16ab1b08e4f5283597fe678a88e69555cd1`
- Julia: `1.10.11`
- BLAS threads: `1`
- Draws per chain: `500`
- Host: `strato2`

Only BRM changes between the candidate and positive control:

- Candidate: `68edfeab8ec465364488e37ed5c2c8fb6be94346`
- Positive control: `784712998ea67f6429d0a3b5a3241fe9cb690e64`

The resolved manifests otherwise differ only in the absolute checkout paths.

## Results

1. `full_sweep_crash.log`: the canonical 576-row candidate command exits 139
   after 175 seconds during the first `sleepstudy` model. The stack enters
   `BayesianRegressionModelsWarmupHMCExt._recursive_block_location` while
   Enzyme computes the gradient of the reparameterized log density. The process
   reports 242,808,854 allocations and 257 GC cycles.
2. `sleepstudy_crash.log`: row-traced reproduction completes all 48 bare
   noncentered/centered rows and adaptive-centering flag-off seeds 1 and 2,
   then segfaults during seed 3. It reports 237,289,391 allocations and 203 GC
   cycles.
3. `postfix_seed1.json`: the candidate completes a one-seed smoke, 6/6 rows.
   This is the false green that a single-gradient or single-seed check permits.
4. `prefix_sleepstudy.json` and `prefix_sleepstudy.log`: with only BRM restored
   to `7847129`, the identical 12-seed process completes 72/72 rows with zero
   failures in 588 seconds.
5. `postfix_full_79cc4a7_rejected.json`: the non-recursive replacement at
   integrated tree `79cc4a7` completes the hand-authored 576-row process with
   zero failures, but changes 91 adaptive-on gradient counts, 92 ESS values,
   and 19 divergence counts relative to the stable baseline. BRM subsequently
   isolated a last-bit accessor/Enzyme arithmetic drift that can redirect the
   discrete window winner. This file is preserved as a rejected regression
   receipt, not installed as the canonical benchmark and not presented as an
   inventory-generated catalogue result.

Every non-wall field in the 72-row positive control is bit-identical to the
corresponding `sleepstudy` rows in the checked-in 576-row artifact: `arm`,
`error`, all three ESS fields, `ess_min_per_grad`, `grad_evals`, `n_constant`,
`n_divergent`, `n_draws_constrained`, `nonlinear_adapt`, `ok`, `seed`, and
`spec` each have zero mismatches.

The checked-in `rows.json` was compared before and after both candidate crashes
and remained byte-identical at the time:

```text
e1a3201d42298689bdffbfac44e39314aa3fb7f4b16ef9b3c9d394a9fc284870  rows.json
```

Relocating the later rejected receipt normalized the legacy file's missing
final newline. Its JSON payload is unchanged; the post-normalization SHA-256 is
`155f987007af5aac9f9bdc9ac4fb12d571e6a94999f0f4867faad22de33d9fed`.

## Commands

Canonical full candidate sweep:

```sh
BRMB_SEEDS=12 BRMB_DRAWS=500 \
BRM_DATA_CACHE=/tmp/kb-whmc-brm-data-cache \
BRMB_OUT=docs/benchmark/results/brm_catalogue/rows.json \
julia --startup-file=no --project=/tmp/kb-whmc-brm-rerun-env-68edfeab \
  docs/benchmark/run_brm_catalogue_benchmark.jl
```

The sleepstudy diagnostic used the same runner and environment with
`BRMB_SPECS=sleepstudy`. It inserted an `@info "row start"` immediately before
the existing `push!(ROWS, run_arm(...))` call through `include_string`; no
benchmark source or measurement logic changed.

Positive control:

```sh
BRMB_SPECS=sleepstudy BRMB_SEEDS=12 BRMB_DRAWS=500 \
BRM_DATA_CACHE=/tmp/kb-whmc-brm-data-cache \
BRMB_OUT=/tmp/kb-whmc-brm-sleepstudy-prefix-seeds12.json \
julia --startup-file=no --project=/tmp/kb-whmc-brm-control-env-7847129 \
  docs/benchmark/run_brm_catalogue_benchmark.jl
```

All long commands ran under `nice -n 19` and `ionice -c 3`.

## Source-file checksums

```text
753148c6ca7e3bea188f11a4ce049f15c64197e878ddb56f9bd592045b1eb18d  full_sweep_crash.log
32bf15489bdac55297ee81170ade23153d019b701dde4c472d72dcdab1e1059f  sleepstudy_crash.log
c4367e8118852040e0c1103761f0faf656fbe0ad38ed46fe14c36092a696175d  prefix_sleepstudy.log
4d3493ed2f3bc393cc5c5e9065675fe5b1c30864b110da5b89302227eb744778  postfix_seed1.json
2a1981dec7a8e3208db7d639fb6c340a64aebd88a3c97d777be32cd93ff75090  prefix_sleepstudy.json
c81446e42ce3cecd143fe6cb319f87dfc42fcbeda83e56ad8f995fee11a2de1d  postfix_full_79cc4a7_rejected.json
```

Upstream tracking: BayesianRegressionModels snag `adaptive-centeri-298729a2`.
