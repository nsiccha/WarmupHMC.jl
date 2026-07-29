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
- BRM canonical bit-exact fix merge: `aed667cfdb304718978750251547819bf5120bda`
- StanBlocks: `329a178a7ad7877da0b58ad2c360d417ddd663f9`
- 12 seeds × 500 retained draws × 3 arms × 2 flag values × 3 models = 216 rows
- Every arm/flag path received a 50-draw untimed preflight; recorded flag order
  alternated by seed to remove systematic JIT/order bias from wall comparisons
- Process exit: 0; elapsed: 641 seconds; summed in-row sampling time: 353.86 seconds
- Result: 216/216 rows completed; zero exceptions or crashes
- Controls: all 6 `(model, bare arm)` flag pairs were identical for all 12 seeds
  on gradient count and constrained-space minimum ESS

The generated adaptive path had zero divergences. The generated static-centered
dyestuff path had 219 divergences across its 12 distinct seed trajectories and
static-centered sleepstudy had one; the flag-on control rows repeat those same
trajectories rather than representing additional distinct failures.

Adaptive centering selected a different trajectory only for dietox, in all 12
seeds. Its paired median constrained-space ESS per gradient improved by 1.25×,
but the added runtime made paired ESS per second 0.87× the flag-off value.
Dyestuff and sleepstudy remained at the non-centered endpoint: their ESS per
gradient ratios were exactly 1.0, while wrapper/adaptation runtime reduced ESS
per second to 0.97× and 0.93× respectively. The docs page derives the full
gradient and runtime comparison tables from the raw rows.

The earlier generated run against BRM tree `79cc4a7` is preserved under
`regression_79cc4a7/`, not as supported performance evidence. That accessor was
stable in this run but changed the legacy floating-point operation tree by one
ULP. The corrected bit-exact accessor kept every non-timing benchmark field
identical here, while making the adaptive wrapper 3.7×–7.4× slower than the
invalid implementation. Keeping that rejected artifact makes the performance
cost auditable without allowing its timings into the headline tables.

`rows.json` is the source of truth. Its SHA-256 is
`3e92a1e66112592c0b70ae92a9e438ff3da7b0a5afe5a7ffb5421269dcaa0875`.

## Reproduction

Use a Julia environment developed at the three package commits recorded above:

```sh
KB_COMPACT_KEEP_LOG=1 \
BRMI_SEEDS=12 BRMI_DRAWS=500 BRMI_PREFLIGHT_DRAWS=50 \
BRMI_OUT=docs/benchmark/results/brm_inventory_generated/rows.json \
BRMI_DATA_CACHE=/tmp/brm-inventory-data \
kb-run-compact nice -n 19 ionice -c 3 \
julia --startup-file=no --project=/path/to/pinned/environment \
  docs/benchmark/run_brm_inventory_benchmark.jl
```

This is evidence for three cheap, ready, verbatim inventory rows. It is not a
claim that all 359 historical catalogue rows have real-data loaders or runnable
translations.
