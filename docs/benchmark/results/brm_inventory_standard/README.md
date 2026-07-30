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
git show dc9636f8:docs/benchmark/results/brm_inventory_standard/rows.json > /tmp/base.json
julia --startup-file=no --project=docs docs/benchmark/brm_inventory_report.jl \
  docs/benchmark/results/brm_inventory_standard/rows.json /tmp/base.json
```

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
- WarmupHMC: `dc9636f8e8cdf6437e69980b3336d2e5659fecbf`
- BayesianRegressionModels: `d452e97a90974d5bef5472978f3d2379255fbfbc`
- StanBlocks: `7a02d30ffb28215e79470ce9689c0f65902b10df`
- DynamicHMC: 3.6.1
- Runner + adapters: `492cbc14c7ff7a69f4658b667dd7c02d1c1280baae7d8037679384d3e8d8453c`
- Inventory `translations.tsv`: `8443c0a15bfe22bee55bd241dbf9b26c4ffe3de6596ddb71d2213d632a597f25`
- Inventory `model_matrix.tsv`: `1e76f72dd7ec518e29c7e6fd0875a51f959842059453228c1d0146be4a6bc9d3`
- 12 seeds × 500 retained draws × 6 arms × 16 models = 1152 rows
- One 50-draw untimed preflight per arm; recorded arm order rotates by seed
- Runner elapsed: 11284.408 seconds
- Summed recorded sampling time: 9655.859 seconds
- Result: 1151/1152 usable trajectories; every arm returned 500 draws
- Total divergences, including the failed trajectory: 1379

One trajectory is an explicit statistical failure rather than a hidden null:
`lme4:dyestuff_re`, WarmupHMC generated-centered, seed 11 returned 500
identical draws (189 constant constrained coordinates) with 207 divergences.
It remains in the design and divergence total, but contributes neither an
invented zero nor a missing value to ESS medians and AoV marks. This is the
same cell that failed in the eight-model predecessor, at the same seed and
with the same constant-coordinate profile.

## Divergences by arm

| default-warmup arm | divergences | trajectories with any |
| --- | ---: | ---: |
| WarmupHMC — generated non-centered | 75 | 6/192 |
| WarmupHMC — generated centered | 742 | 25/192 |
| WarmupHMC — nonlinear wrapper fixed at `c=0` | 75 | 6/192 |
| WarmupHMC — adaptive centering | 10 | 5/192 |
| DynamicHMC — generated non-centered | 3 | 3/192 |
| DynamicHMC — generated centered | 474 | 29/192 |

The adaptive-centering arm is the least divergent WarmupHMC arm but is no
longer at exactly zero, as it was over the eight published models. Its ten
divergences are nine on `vasishth:meta_sbi` and one on `vasishth:n400_crossed`.
Neither is a regression against the arm it is meant to improve on: the SBI
meta-analysis is a twelve-study funnel on which *every* arm diverges, and the
non-centered and fixed-wrapper arms record the same nine there; the single
n400 divergence is matched by one in DynamicHMC's non-centered arm. Meanwhile
adaptive centering removes all 66 of `mixed_models_jl:penicillin_crossed`'s
divergences, which the non-centered and fixed arms both carry.

Two models dominate the centered-arm totals: `kruschke:fruitfly_anhecova`
(516 WarmupHMC, 422 DynamicHMC) and `lme4:dyestuff_re` (219 and 30).

## Seed-paired headline ratios

WarmupHMC on the identical generated non-centered target, divided by
DynamicHMC on that target:

| model | ESS/gradient | wall time | ESS/second |
| --- | ---: | ---: | ---: |
| dyestuff | 3.06× | 0.76× | 1.43× |
| sleepstudy slope | 4.54× | 0.31× | 2.68× |
| sleepstudy (Bambi) | 4.47× | 0.30× | 2.70× |
| penicillin crossed | 1.77× | 0.14× | 1.53× |
| radon partial pooling | 2.97× | 0.26× | 2.95× |
| radon floor | 2.46× | 0.27× | 2.29× |
| radon slopes | 3.46× | 0.22× | 3.26× |
| dietox | 2.23× | 0.36× | 2.16× |
| SBI meta-analysis | 3.05× | 1.37× | 0.74× |
| fruitfly ANCOVA | 3.01× | 0.32× | 1.64× |
| epilepsy counts | 5.14× | 0.27× | 2.85× |
| therapeutic touch | 3.15× | 0.53× | 2.31× |
| baseball binomial | 2.12× | 0.49× | 1.17× |
| contraception | 8.82× | 0.09× | 8.39× |
| pulmonary slopes | 5.12× | 0.26× | 4.48× |
| N400 crossed | 5.36× | 0.11× | 5.15× |

ESS/gradient favours WarmupHMC on all sixteen (1.77×–8.82×). ESS/second
follows on fifteen; the exception is the SBI meta-analysis, a 14-dimensional
target on which WarmupHMC's per-gradient overhead is not amortized.

On the generated *centered* target the same comparison ranges 0.70×–60.32× in
ESS/gradient, so WarmupHMC leads there on fourteen of sixteen models rather
than on all of them: `kruschke:fruitfly_anhecova` ties at 1.00× and
`vasishth:pulmonary_slopes` trails at 0.70×. Dyestuff pairs 11 seeds there
because of the failed cell; every other row pairs 12.

## The nonlinear wrapper's cost has grown since the published pins

The fixed wrapper still preserves seed-level ESS/gradient **exactly**: the
ratio is 1.00× with identical gradient counts on all sixteen models, which is
the invariant this arm exists to check. Its wall-time cost, however, is now
1.24×–4.86× the unwrapped non-centered arm — not the 6–13% the eight-model
predecessor recorded.

That is a change in WarmupHMC, not in the measurement. Re-measuring the eight
shared control models on the same host and the same Julia 1.10.11 isolates it
to the two arms that go through the wrapper:

| median µs per gradient | non-centered | centered | fixed wrapper | adaptive |
| --- | ---: | ---: | ---: | ---: |
| dyestuff | 8.8 → 8.6 | 4.4 → 5.1 | 8.9 → 25.9 | 13.9 → 28.2 |
| sleepstudy slope | 21.7 → 23.4 | 21.8 → 22.3 | 23.8 → 50.5 | 37.7 → 65.6 |
| sleepstudy (Bambi) | 22.8 → 23.6 | 21.3 → 21.7 | 23.4 → 52.5 | 36.4 → 65.3 |
| penicillin crossed | 13.4 → 13.9 | 12.1 → 12.2 | 15.5 → 51.0 | 46.4 → 82.5 |
| radon partial pooling | 49.7 → 50.7 | 43.8 → 43.2 | 54.5 → 127.2 | 146.9 → 245.5 |
| radon floor | 56.4 → 55.9 | 52.1 → 51.0 | 62.6 → 130.9 | 153.1 → 238.6 |
| radon slopes | 57.9 → 58.7 | 91.0 → 90.7 | 63.4 → 146.1 | 143.6 → 233.0 |
| dietox | 55.2 → 54.4 | 86.7 → 87.1 | 59.1 → 126.8 | 131.9 → 222.9 |

Left of each arrow is `605f4b83c65730087fd05bae249d673315345a91`, right is
`dc9636f8e8cdf6437e69980b3336d2e5659fecbf`. Both DynamicHMC arms and both
unwrapped WarmupHMC arms are unchanged to within ~2%; the wrapped arms cost
2.0–3.3× more per gradient. The per-model ratio is largest where the density
itself is cheapest (4.86× on baseball binomial, 1.24× on N400 crossed), which
is the signature of a fixed per-call overhead rather than a cost that scales
with the target.

Nothing above is inferred from ESS. Gradient counts are exact integers from
`WarmupHMC.count_and_time`, and 47 of the 48 shared model × arm median counts
are identical across the two pins — so the per-gradient column really is
comparing like with like. The single exception is penicillin crossed on the
adaptive arm, whose median moved 38,049 → 36,668 (−3.6%), which does not
account for that cell's 46.4 → 82.5 µs. The report script prints this check
alongside the table; do not read the comparison without it.

The ESS/second columns below inherit this overhead, and the honest reading is
that the published wall-clock figures for the two wrapped arms are a floor on
what the method can do, not a measurement of the method's intrinsic cost.

## Fitting centerings

Isolated from the wrapper — adaptive over fixed, so the overhead above divides
out — fitting centerings improves ESS/gradient on nine of sixteen models and
degrades it on none. The large win is penicillin crossed (5.57×); radon
partial pooling (1.74×), radon floor (1.57×), baseball binomial (1.41×),
therapeutic touch (1.30×), contraception (1.27×), dietox (1.24×), epilepsy
counts (1.17×) and N400 crossed (1.06×) follow. The other seven sit at exactly
1.00×, i.e. the fitted centering returns the generated endpoint.

Its extra differentiation and adaptation work means those gradient
improvements mostly do not become ESS/second improvements: only penicillin
crossed (3.32×) and radon partial pooling (1.03×) come out ahead on wall time.

## N400 wall times and box load

`vasishth:n400_crossed` is the one spec whose timings could have been
perturbed by unrelated work. It ran last, from 22:49:22 to 01:22:23, and
between roughly 23:00 and 23:12 the host also carried two `Pkg.instantiate()`
precompiles, a toolchain download with two resolve-only runs, and two short
git-plus-Julia invocations — on the order of 40 seconds of genuine multi-core
contention inside a spec that ran for two and a half hours.

**Nothing in the rows stands out.** Normalising each cell's `wall_s` by its own
exact gradient count removes the geometry and leaves the per-gradient cost,
which varies across the twelve seeds by at most 5% in every arm:

| arm | s/gradient range over 12 seeds | max/median |
| --- | --- | ---: |
| WarmupHMC — generated non-centered | 0.00217–0.00235 | 1.04 |
| WarmupHMC — generated centered | 0.00223–0.00233 | 1.02 |
| WarmupHMC — nonlinear wrapper fixed at `c=0` | 0.00270–0.00293 | 1.05 |
| WarmupHMC — adaptive centering | 0.00325–0.00340 | 1.01 |
| DynamicHMC — generated non-centered | 0.00213–0.00219 | 1.02 |
| DynamicHMC — generated centered | 0.00221–0.00228 | 1.02 |

Reconstructing the cell timeline from the recorded durations puts six cells in
the contention window — seed 1's DynamicHMC centered cell, then the first five
of seed 2's six cells — at 0.83×, 0.91×, 0.99×, 1.01×, 0.97× and 1.15× their
respective arm medians. The largest raw outlier in the whole spec is elsewhere
and is not a timing artefact at all: WarmupHMC generated-centered seed 10, at
2.03× its arm median, ran 58,828 gradients against that arm's median of
28,603, hours after the window closed. So the widest `wall_s` spread in this
spec is geometry, not load, and the published medians over twelve seeds are
unaffected either way.

## Figures and provenance

The documentation derives the complete absolute and seed-paired tables plus
the AlgebraOfVega figures from `rows.json`; no plotted summary is stored beside
the raw rows.

`rows.json` is the source of truth. Its SHA-256 is
`1ec8dc808ed51242bdc30f543266aad9eab9882b2866e4c92a8f0ca980eb9c21`.

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
