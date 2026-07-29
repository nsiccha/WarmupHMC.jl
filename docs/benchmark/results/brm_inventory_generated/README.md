# BRM historical-inventory generated-model benchmark

This is the focused real-data rerun that consumes BayesianRegressionModels'
landed historical-inventory translations. It is deliberately separate from
`../brm_catalogue/`, whose builders were transcribed by hand and therefore do
not test catalogue-to-BRM generation.

## Exact scope

`run_brm_inventory_benchmark.jl` selects these three `inferred-family` rows
from BRM's checked-in `research/historical_model_inventory/translations.tsv`:

- `lme4:dyestuff_re`
- `lme4:sleepstudy_slope`
- `bambi:dietox`

For every row, the runner asserts `translation_status=ready` and
`surface_support_class=already-expressible-verbatim`, cross-checks the body and
finite-gradient capability receipt against `model_matrix.tsv`, and evaluates
the row's `current_brm_body` through BRM's `_brm` and `SBBRMI` path. No formula
is copied into the runner. The inventory has no generic real-data loader, so
the runner owns only the documented column adapters and records each input
CSV's URL and SHA-256.

Each generated model is exercised as BRM's default non-centered model, BRM's
static centered model, and `BRM.adaptive_centering_problem`. Both values of
WarmupHMC's `nonlinear_adapt` flag are run so the bare generated models act as
negative controls.

## Recorded run

- Host: `strato2`
- Julia: 1.10.11; BLAS threads: 1
- WarmupHMC: `9039668d8ce1f1f62c7eb3d235ac7dac19a12d5a`
- BRM integrated fix tree: `79cc4a75906445d925db810c45c3384c262fb313`
  (the same Git tree subsequently merged into canonical `7052605a...`)
- StanBlocks: `329a178a7ad7877da0b58ad2c360d417ddd663f9`
- 3 seeds × 500 retained draws × 3 arms × 2 flag values × 3 models = 54 rows
- Process exit: 0; elapsed: 269 seconds; summed in-row sampling time: 57.95 seconds
- Result: 54/54 rows completed; zero exceptions or crashes
- Controls: all 6 `(model, bare arm)` flag pairs were identical for all 3 seeds
  on gradient count and constrained-space minimum ESS

The generated adaptive path had zero divergences. The generated static-centered
dyestuff path had one divergence in seeds 2 and 3; the flag-on rows repeat those
same control trajectories rather than representing two additional distinct
failures.

Median adaptive-on sampling time was 0.094 s for dyestuff, 0.346 s for
sleepstudy, and 2.612 s for dietox. Adaptive centering selected a different
trajectory only for dietox, improving median constrained-space ESS per gradient
by 1.46× over its flag-off pair. Dyestuff and sleepstudy remained at the
non-centered endpoint, so their paired efficiency ratio was exactly 1.0.

`rows.json` is the source of truth. Its SHA-256 is
`1859cb9655a4fa3732bafb22600bc053914937a3ba33fb721ce16d0fc9268470`.

## Reproduction

Use a Julia environment developed at the three package commits recorded above:

```sh
KB_COMPACT_KEEP_LOG=1 \
BRMI_SEEDS=3 BRMI_DRAWS=500 \
BRMI_OUT=docs/benchmark/results/brm_inventory_generated/rows.json \
BRMI_DATA_CACHE=/tmp/brm-inventory-data \
kb-run-compact nice -n 19 ionice -c 3 \
julia --startup-file=no --project=/path/to/pinned/environment \
  docs/benchmark/run_brm_inventory_benchmark.jl
```

This is evidence for three cheap, ready, verbatim inventory rows. It is not a
claim that all 359 historical catalogue rows have real-data loaders or runnable
translations.
