# Measured: adaptive reparametrization vs the fixed centering endpoints

**Measured on WarmupHMC `b5c7dee`.** 8 pinned seeds per arm, `n_draws` floor
1000, Julia 1.10.11, single-threaded BLAS, on `strato2`. 184 runs per backend,
0 failed.

This base was measured **twice, under both AD backends**, changing nothing else:
`results/enzyme-b5c7dee/` (the default, `AutoEnzyme(; function_annotation =
Enzyme.Const)`) and `results/forwarddiff-b5c7dee/` (`AutoForwardDiff()`). The
runs were sequential, never concurrent, because the wall-clock comparison is the
point and two concurrent runs would contend for CPU. `summarize.jl` regenerates
every table below from those records; `compare.jl` regenerates the backend diff.

Two superseded runs are kept: `results/after/` (base `c8fed88`, ForwardDiff) and
`results/before/` (base `05aed41`, which carried the halo-recording regression
`34ce034`). The before/after comparison is a measured result of its own and is
reported in [Effect of the halo-recording regression](#effect-of-the-halo-recording-regression).

> **Re-baselining check.** `c8fed88 → b5c7dee` touched `src/`, so the older
> tables could not simply be assumed to carry over. Re-running ForwardDiff on
> the new base reproduces `results/after/` **exactly** — every arm, every
> target, 0.0% change in min ESS per 1000 gradients and in gradient counts. The
> `src/` drift did not alter sampling, so the sampling verdict below is
> continuous across all three bases.

> ⚠ **Every wall-clock and per-gradient number in this document was measured
> against reparametrization specs whose closures capture a `Core.Box`** on
> `radon_partially_pooled`, `radon_variable_intercept` and `seeds`. That costs
> ForwardDiff ~4× and Enzyme ~15× on the gradient path, and it is enough to
> invert which backend looks faster. The **sampling** results — ESS per gradient
> evaluation, where adaptation settles, divergences — are unaffected, because the
> defect changes only the cost of a gradient and not its value (gradients agree
> to 0.0). The **backend** and **wall-clock** sections are affected and are
> flagged in place. Details and the fix in
> [Which AD backend](#which-ad-backend).

## Verdict

**Adaptive partial centering, started from the centered parametrization, beats
the plain sampler on every target measured** — 1.8× to 51× per gradient
evaluation. It gets there without being told which parametrization to use.

**Against the best available *fixed* parametrization the picture splits, and the
split is the interesting part:**

| target | adaptive | best fixed alternative | |
|---|---|---|---|
| radon partially_pooled | **53.08** | 22.07 (noncentered endpoint) | **2.4× win** |
| radon variable_intercept | **65.92** | 24.46 (noncentered endpoint) | **2.7× win** |
| eight_schools centered | 47.23 | 51.56 (hand-written noncentered) | ties, −8% |
| seeds centered | 15.09 | 11.28 (noncentered endpoint) | ties (see below) |
| funnel (synthetic) | 78.45 | 107.05 (noncentered endpoint) | **loses, −27%** |

(min ESS per 1000 gradient evaluations, median over 8 seeds, Enzyme run.)

Read as one sentence: **where a good hand-written parametrization exists,
adaptive finds something as good as it, automatically. Where none of the fixed
options is good, adaptive beats all of them.** Both radon models are the second
case — centered, noncentered and posteriordb's own hand-written noncentered
model all land within seed noise of each other around 15–25, and adaptive gets
53–66 by settling at an *interior* `c ≈ 0.4–0.5` that no hand-written model
offers. That interior optimum is the strongest result here.

**This half of the verdict does not depend on the AD backend, and that is now
measured rather than assumed.** ESS per 1000 gradient evaluations is identical
across the two backend runs on every `plain` and `adaptive` arm, to 0%. The
backend sets the cost of a gradient; it does not change how many the sampler
asks for, nor which parametrization the adaptation finds — the final `c` and the
stuck-adaptation counts are identical in both runs.

### It is not free, and switching to reverse mode did not fix that

On three of five targets adaptive is a net **wall-clock** loss against doing
nothing, and switching to reverse mode does not recover any of them — it makes
two of the three **worse** and leaves the third where it was:

| target | adaptive vs plain, ForwardDiff | adaptive vs plain, Enzyme/Const |
|---|---|---|
| funnel (synthetic) | **14.7× faster** | **17.9× faster** |
| eight_schools centered | **19.7× faster** | **21.0× faster** |
| radon variable_intercept | 1.5× slower | **2.3× slower** |
| radon partially_pooled | 3.5× slower | 3.5× slower |
| seeds centered | 1.8× slower | **6.1× slower** |

(min ESS/sec, median over 8 seeds, same base, same seeds, sequential runs.)

Reverse mode helps where the wrapper was already winning (the two `d = 10`
targets) and does not help where it was already losing. On `seeds` it more than
triples the loss.

**Do not read that as a verdict on reverse mode.** The three targets it fails on
are exactly the three whose reparametrization specs capture a `Core.Box`, which
costs Enzyme ~15× on the gradient path and is enough to invert the ordering by
itself — see [Which AD backend](#which-ad-backend). These wall-clock losses are
measured correctly and they are real *for the specs as currently written*; what
causes them is a defect in the spec table, and the rows above should be
re-measured once it is fixed.

So the claim the package can support is about gradient evaluations, not seconds:

> Starting from a centered parametrization, adaptive partial centering matches or
> exceeds the best fixed parametrization's sampling efficiency per gradient
> evaluation on four of five targets, and beats every fixed option on the two
> where no fixed option is good — without being told to. Whether that converts
> into wall-clock depends on how expensive the target's own gradient is relative
> to the wrapper, and under both backends measured it does not, on three of five.

**The overhead is not the transform.** A wrapper configured as an exact identity
costs the same as a live one, under **both** backends and on every target (e.g.
on `radon_partially_pooled`: Enzyme ×14.47 no-op vs ×14.38 live, ForwardDiff
×15.94 vs ×14.30). The cost is carrying the AD re-differentiation of
`ljac_(x_) + dot(g_y, y_)` at all, not the arithmetic of the reparametrization.
Every wall-clock loss above is addressable there; none of it is intrinsic to the
method.

## Which AD backend

The differentiated objective `x -> ljac(x) + dot(g_y, y(x))` is scalar in the
full parameter vector with the inner gradient `g_y` frozen, so it is the
textbook reverse-mode shape: forward mode costs `ceil(d / chunksize)` tangent
sweeps of the transform per gradient where reverse mode costs one. The
prediction that follows is that reverse mode should win, and win *hardest* at
large `d`.

**Measured, it does not.** Per wrapped `logdensity_and_gradient` call, ratio of
Enzyme/`Const` to ForwardDiff — below 1.0 means Enzyme is faster:

| target | `d` | Enzyme/Const ÷ ForwardDiff | | spec closure captures |
|---|---|---|---|---|
| `funnel` | 10 | **0.35–0.42×** | Enzyme ~2.5× faster | `()` — a literal |
| `eight_schools` | 10 | **0.44–0.92×** | Enzyme faster, margin varies | `()` — a literal |
| `radon_variable_intercept` | 89 | 1.21–2.37× | Enzyme slower | `(Core.Box,)` |
| `radon_partially_pooled` | 88 | 1.25–2.67× | Enzyme slower | `(Core.Box,)` |
| `seeds` | 26 | 2.31–5.42× | Enzyme 2–5× slower | `(Core.Box,)` |

Ranges span **every measurement taken**: four independent Julia processes at 3
rounds × 1000 calls, plus two at 7 rounds × 1000 calls, each with both a
centering endpoint and an interior `c = 0.5`, and with the backend order rotated
per round so no backend keeps the warm slot. Within a process the spread is
8–14% on the radon targets and much wider on the two `d = 10` targets, where the
whole call is ~1 µs and GC dominates — which is why `eight_schools` spans 0.44
to 0.92. The direction never changes on any target, in any process, at either
`c`.

*Between* processes the two radon ratios are much less stable than that
within-process spread suggests: a later replication put them at 2.31–2.67×
against the 1.21–1.52× of the earlier ones, roughly doubling the apparent
penalty on the same base with the same script. Only the direction replicates;
the magnitude on the boxed targets should not be quoted to two significant
figures. The clean targets (`funnel`, `eight_schools`) reproduced within their
stated ranges every time.

### These numbers measure a defect in the spec table, not the backend

**Read the last column before reading the ratios.** It predicts the verdict
perfectly, five targets out of five, and `d` does not.

`reparametrization()` in `web/src/posteriordb_reparametrizations.jl` is one long
`if`/`elseif` chain, and `l`, `s`, `o` are assigned in many of its branches
inside that single scope. The per-pair closures capture them:

```julia
(l, s, o) = (J+1, J+2, 0)              # :41, radon_partially_pooled
map(1:J) do i
    idx => Reparametrization(..., x->x[l], x->x[s])
end
```

Julia's closure conversion cannot prove single assignment across those branches,
so it captures a **`Core.Box`** — a mutable heap cell read as `Any` — rather than
an `Int`. `funnel` and `eight_schools` close over literals (`x->x[1]`,
`x->x[9]`) and capture nothing. Confirmed with `fieldtypes`, not inferred from
timings.

Rebuilding the identical spec so the captures are plain `Int`s — same 85 pairs,
same indices, same centerings, closures reading the same `x[86]`/`x[87]`, and
**gradients agreeing to exactly 0.0** — gives, on `radon_partially_pooled`:

| spec | ForwardDiff | Enzyme/`Const` |
|---|---|---|
| as shipped (boxed) | 339590 ns | 429477 ns |
| de-boxed | 79567 ns | **27911 ns** |
| speedup | 4.3× | **15.4×** |

Both backends are hurt; Enzyme is hurt ~4× harder, which is enough to **invert
the ordering**. De-boxed, Enzyme is 2.85× *faster* than ForwardDiff on the target
this document reports it 1.25–1.52× slower on.

So the dimension argument is not refuted in the opposite direction, as an earlier
revision of this section claimed — it is **untested**. Dimension and boxing are
perfectly confounded in the current spec table: the three boxed specs are the
three larger models. Nothing here separates them, and no claim about how this
backend scales in `d` should be drawn from this table until the captures are
fixed. The two clean rows (`funnel`, `eight_schools`) are the only ones that
currently say anything about the backend, and both favour Enzyme.

This is a fixable performance bug worth ~15× on the wrapper's gradient path, not
a property of reverse mode. `web/src/posteriordb_reparametrizations.jl` is
outside this directory's ownership; it has been reported rather than edited here.

Three other things were checked, and none of them explains the ratios either:

- **Correctness.** All three backends agree on the gradient to ≤ 9.1e-13 (max
  absolute deviation, every target × both endpoints × interior `c = 0.5`). This
  is a cost difference, not a wrong answer. Enzyme never differentiates through
  BridgeStan's FFI — only the pure-Julia transform is differentiated, and the
  inner problem's own `logdensity_and_gradient` is reused as a frozen constant.
- **Evaluation position.** The microbenchmarks evaluate at `randn(d)`, which is
  not where a sampler spends its time. Re-timing at the positions the sampler
  *actually visited* — draws captured in the source frame from the
  `fixed_noncentered` arm itself — moves the ratio by a few percent and changes
  no verdict (`radon_partially_pooled` 2.01× at `randn` vs 2.05× typical;
  `funnel` 0.41× vs 0.39×). Recorded in `results/typical_positions.json`.
- **DI preparation.** The hot path calls `value_and_gradient` with **no prep
  object** (`src/Reparametrizations.jl:148`), so DifferentiationInterface
  re-prepares on every gradient evaluation. On the two `d ≈ 88` targets an
  explicit `prepare_gradient` is only 10–16% of the call and costs the same
  under both backends (37–55 µs), against a backend gap of ~110 µs; on `seeds`
  it is 6–10 µs against a 115 µs gap. Reusing a prep object was neutral-to-worse
  in this probe, so it is not an available speedup either. Recorded in
  `results/prep_cost.json`.

### `function_annotation` is required, and how much it costs is target-dependent

A bare `AutoEnzyme()` **fails outright** on this objective with
`EnzymeMutabilityException` — the objective is a closure capturing the
reparametrizer and the frozen `g_y`. Enzyme's own error text suggests
`Duplicated`; `Const` is the correct annotation here, since the closure is not
something we differentiate with respect to.

The cost of getting that wrong is not a constant, and this matters because the
package docstring's recommendation rests on a single 11-dimensional funnel:

| target | `d` | `Duplicated` ÷ `Const` |
|---|---|---|
| `funnel` | 10 | 11.3–13.6× |
| `eight_schools` | 10 | 3.8–4.9× |
| `seeds` | 26 | 0.96–1.16× |
| `radon_variable_intercept` | 89 | 1.01–1.05× |
| `radon_partially_pooled` | 88 | 1.02–1.06× |

The `Duplicated` penalty is roughly a fixed per-call cost (~5 µs at `d = 10`,
~8 µs at `d = 26`, ~33 µs at `d = 88`), so it dominates exactly when the call is
otherwise cheap and vanishes when it is not. On the funnel it is a 13× disaster;
on the three larger targets it is free — on `seeds` one run put `Duplicated`
marginally *ahead*, which is the signature of a difference below the noise floor.
Both gradients are equally correct.

So the honest statement is: **`Const` is the right annotation on correctness
grounds everywhere, and its measured speedup is large on the two targets whose
specs are clean.** A recommendation calibrated on the funnel alone overstates the
stakes by an order of magnitude.

Note what this table cannot currently tell you. The three targets where
`Duplicated` looks free are exactly the three whose specs capture a `Core.Box`
(above) — where a boxed capture already costs Enzyme ~15×, so a further shadow
copy is lost in the noise. "Free on large targets" is therefore not established;
it is what a fixed per-call cost looks like when it is added to a much larger
defect. Re-measure after the captures are fixed before quoting this table as a
property of `Duplicated`.

### One thing the two methods disagree about

On the two radon targets the microbenchmark and the end-to-end runs do not
agree, and it is not resolved:

- the microbenchmark (stable across four processes) makes Enzyme 1.2–1.5×
  **slower** per wrapped gradient;
- the 184-run benchmark, at gradient counts identical to 0%, makes Enzyme 2–14%
  **faster** per second on the same target and arms.

An additive per-iteration sampler cost cannot invert an ordering, so this is not
simply "the sampler does other work too". The `seeds` and `d = 10` results are
**not** affected — there both methods agree in direction and roughly in
magnitude, which is why those carry the verdict above and radon does not.

There is now a candidate explanation, untested: radon is one of the boxed specs,
and reading a `Core.Box` is a dynamic load whose cost depends on what else is
resident. A tight 1000-call loop and a running sampler stress that differently,
which is exactly the shape of defect that can disagree between the two methods
without either being mismeasured. If the disagreement dissolves once the captures
are fixed, that was the cause; if it survives, it is cleanly separated from it.

One measurement artifact was identified and excluded while chasing this: running
a full sampler before timing warms the ForwardDiff path enough to make its
subsequent microbenchmark ~2× faster (185 µs vs 331–351 µs on
`radon_partially_pooled`). Absolute timings are therefore only comparable within
a process that did the same warm-up; the typical-vs-`randn` contrast above is
unaffected because both position sets were timed under the same warmed state.

## What each arm is

Every arm samples the **same** Stan model, so all draws are in the same
coordinates and their ESS is directly comparable. The arms differ only in what
the sampler may do with the partial-centering parameter `c`
(`PartiallyCentered(1.0)` is centered, `(0.0)` is noncentered).

| arm | what it is |
|---|---|
| `plain` | bare `StanProblem`, no wrapper. What a user gets today. |
| `fixed c = centered` | wrapped, source `c` pinned to the model's own value, `nonlinear_adapt=false`. An exact identity transform, so it isolates the cost of merely carrying the wrapper. |
| `adaptive` | wrapped, source `c` starts at the model's own value, `nonlinear_adapt=true`. The method under test. |
| `fixed c = noncentered` | wrapped, source `c` pinned to the opposite endpoint. The hand-written noncentered parametrization, reached by transform. |
| `hand-written noncentered model` | posteriordb's separately written noncentered member, sampled plain. The external reference for "noncentered-like". |

The tables below are the **Enzyme** run, the current default. Where a number is
backend-sensitive it is the `ESS/sec` column and only that column; `ESS per 1k
grad`, `grad evals`, `divergences` and `final source c` are identical in the
ForwardDiff run except on the three `fixed_noncentered` arms noted below.

`min ESS` is the minimum over coordinates — the number that governs how long you
must run. Sub-figures are the min–max across the 8 seeds. `final source c` is
read back off the mutated `IndexedReparametrization` after the run, which is
valid because every run here is single-chain `adaptive_warmup_mcmc`; the
cooperative and clustered samplers `deepcopy` the problem per chain and would
silently return the *initial* `c` instead.

> **Where the two backend runs are not sampling-identical.** Three arms drift by
> ≥2% in gradient count between backends — `eight_schools`/fixed_noncentered
> −5%, `radon_variable_intercept`/fixed_noncentered −3%,
> `seeds`/fixed_noncentered −2%. Every other arm is 0–1%. The cause is the same
> mechanism described under [Harness correctness](#harness-correctness): the
> backends' gradients differ at the 1e-13 level and NUTS amplifies that into a
> different trajectory. **No arm quoted in the verdict is affected** — every
> `plain` and `adaptive` arm drifts 0% — but the `fixed_noncentered` rows are
> not a like-for-like wall-clock comparison across backends and are not used as
> one.

<!-- tables generated by docs/benchmark/summarize.jl from results/enzyme-b5c7dee/runs.json -->

### `eight_schools-eight_schools_centered`

*8 group effects — dimension 10. The textbook hierarchical funnel.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 51.1 <sub>25.2–86.3</sub> | 365.3 <sub>75.5–1277.4</sub> | 1.55 | 33320 | 4.5 | — |
| fixed c = centered | 46.8 <sub>8.8–102.8</sub> | 504.3 <sub>65.0–715.0</sub> | 2.15 | 22504 | 8.0 | 1.0 |
| **adaptive** (starts centered) | 492.7 <sub>428.1–780.4</sub> | 7668.3 <sub>7162.5–13771.7</sub> | 47.23 | 10496 | 0.0 | 0.0 |
| fixed c = noncentered | 498.0 <sub>384.3–609.6</sub> | 15684.6 <sub>3502.8–23132.3</sub> | 46.73 | 9846 | 0.0 | 0.0 |
| hand-written noncentered model | 569.9 <sub>345.5–627.4</sub> | 18353.8 <sub>2712.8–28537.3</sub> | 51.56 | 9482 | 0.0 | — |

Min ESS over the 10 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 51.1 | 1.54 |
| fixed c = centered | 46.8 | 2.08 |
| **adaptive** (starts centered) | 492.7 | 46.95 |
| fixed c = noncentered | 498.0 | 50.57 |
| hand-written noncentered model | 569.9 | 60.11 |

Adaptive drives `c` to exactly 0.0 on all 8 coordinates on all 8 seeds — the
known right answer for this model, found without being told. It lands level with
the noncentered endpoint reached by transform and about 8% below posteriordb's
hand-written noncentered model. Divergences go from a median of 4.5 on plain to 0.

This is the target where the pre-fix run over-stated the method: on `05aed41`
adaptive appeared to *beat* the hand-written model by 41%. It does not.

### `radon_mn-radon_partially_pooled_centered`

*85 counties — dimension 88.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 189.9 <sub>100.1–255.9</sub> | 454.0 <sub>245.2–575.0</sub> | 14.12 | 14971 | 0.0 | — |
| fixed c = centered | 209.4 <sub>134.6–300.9</sub> | 38.5 <sub>31.8–46.4</sub> | 14.91 | 14828 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 507.8 <sub>391.4–795.5</sub> | 128.0 <sub>99.5–175.6</sub> | 53.08 | 8644 | 0.0 | 0.4 |
| fixed c = noncentered | 364.3 <sub>193.4–428.1</sub> | 56.2 <sub>39.4–63.8</sub> | 22.07 | 15858 | 0.0 | 0.0 |
| hand-written noncentered model | 314.4 <sub>178.5–439.6</sub> | 563.1 <sub>391.3–827.2</sub> | 18.52 | 16456 | 0.0 | — |

Min ESS over the 88 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 189.9 | 12.68 |
| fixed c = centered | 209.4 | 14.12 |
| **adaptive** (starts centered) | 507.8 | 58.74 |
| fixed c = noncentered | 364.3 | 22.98 |
| hand-written noncentered model | 314.4 | 19.11 |

**The headline case for the method.** No fixed parametrization helps: centered
(14.91), noncentered (22.07) and posteriordb's hand-written noncentered model
(18.52) are all in the same band. Adaptive settles at `c = 0.4` on seven of eight
seeds (0.5 on the eighth) and gets 53.08 — 2.4× the best fixed option — using 45%
fewer gradient evaluations than the noncentered endpoint. **And it is still 3.5×
slower in wall-clock than the plain sampler** (128.0 vs 454.0 ESS/sec), because
this target has the worst measured wrapper overhead (×14.4 under Enzyme, ×14.3
under ForwardDiff). That is the split verdict in one row.

### `radon_mn-radon_variable_intercept_centered`

*85 counties + floor slope — dimension 89.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 273.7 <sub>173.6–408.5</sub> | 346.4 <sub>190.2–535.3</sub> | 19.37 | 14960 | 0.0 | — |
| fixed c = centered | 339.7 <sub>208.5–445.1</sub> | 55.7 <sub>47.1–66.0</sub> | 22.65 | 15016 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 627.6 <sub>409.0–766.7</sub> | 151.2 <sub>103.3–208.7</sub> | 65.92 | 8654 | 0.0 | 0.5 |
| fixed c = noncentered | 403.6 <sub>188.9–452.9</sub> | 61.1 <sub>30.7–67.9</sub> | 24.46 | 16274 | 0.0 | 0.0 |
| hand-written noncentered model | 287.0 <sub>220.2–444.0</sub> | 405.1 <sub>215.6–520.6</sub> | 21.36 | 15837 | 0.0 | — |

Min ESS over the 89 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 273.7 | 18.30 |
| fixed c = centered | 339.7 | 22.62 |
| **adaptive** (starts centered) | 627.6 | 72.52 |
| fixed c = noncentered | 403.6 | 24.80 |
| hand-written noncentered model | 287.0 | 18.12 |

Same shape as its sibling: an interior optimum at `c = 0.5` on all eight seeds
worth 2.7× the best fixed alternative, and still a 2.3× wall-clock loss. The
`fixed_noncentered` row is one of the three arms that drifted between backends
(−3% gradients), so read its 24.46 as approximate; the ForwardDiff run puts it
at 18.26, which would make the margin 2.9× instead. The adaptive and plain rows
are identical across backends.

### `seeds_data-seeds_centered_model`

*21 plates — dimension 26. posteriordb ships no noncentered sibling.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 101.7 <sub>88.3–161.0</sub> | 858.7 <sub>622.2–1234.6</sub> | 8.40 | 13068 | 0.0 | — |
| fixed c = centered | 101.7 <sub>88.3–161.0</sub> | 82.7 <sub>49.2–131.5</sub> | 8.40 | 13068 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 213.2 <sub>140.2–310.4</sub> | 141.4 <sub>89.6–219.1</sub> | 15.09 | 15454 | 0.0 | 0.3 |
| fixed c = noncentered | 206.4 <sub>172.2–262.9</sub> | 122.0 <sub>84.6–205.0</sub> | 11.28 | 17274 | 0.0 | 0.0 |

Min ESS over the 47 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 101.7 | 7.78 |
| fixed c = centered | 101.7 | 7.78 |
| **adaptive** (starts centered) | 213.2 | 13.79 |
| fixed c = noncentered | 206.4 | 11.95 |

Adaptive is ahead of the noncentered endpoint on both measures here (15.09 vs
11.28 unconstrained, 13.79 vs 11.95 matched), where the ForwardDiff run had them
tied. That reversal is not a backend effect on the method: the
`fixed_noncentered` arm is one of the three that drifted (−2% gradients, and its
min ESS moved 295.7 → 206.4 on a seed-noise range of 144–414). Treat this target
as **adaptive ties or modestly beats** the endpoint, not as a clean 1.3× win.
Both roughly double plain. Interior optimum again, `c ≈ 0.3`.

This is also the target where reverse mode costs the most: adaptive goes from
1.8× slower than plain under ForwardDiff to **6.1× slower** under Enzyme.

### `funnel` (synthetic)

*Neal's funnel, `v ~ Normal(0, 3)`, `theta_i ~ Normal(0, exp(v/2))`, K = 9 —
dimension 10. Not a posteriordb posterior; posteriordb ships none. Defined in
`common.jl` with an analytic gradient. It is the extreme case — the centered
parametrization is pathological and the noncentered one is exact — so it bounds
what partial centering can possibly buy. Reported alongside the real targets,
never as a headline.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 31.7 <sub>19.6–56.4</sub> | 837.1 <sub>271.3–1802.4</sub> | 1.53 | 26799 | 0.5 | — |
| fixed c = centered | 31.7 <sub>19.6–56.4</sub> | 626.3 <sub>229.2–1381.5</sub> | 1.53 | 26799 | 0.5 | 1.0 |
| **adaptive** (starts centered) | 823.1 <sub>746.4–936.8</sub> | 14964.4 <sub>14052.0–15993.5</sub> | 78.45 | 10054 | 0.0 | 0.0 |
| fixed c = noncentered | 829.9 <sub>718.3–1017.9</sub> | 53386.8 <sub>37826.9–71595.2</sub> | 107.05 | 7963 | 0.0 | 0.0 |

The one target where a fixed endpoint beats adaptive on gradient efficiency: 78.5
vs 107.1 per 1000 gradients. That is the expected and correct result — noncentered
is *exactly* right here, so there is nothing to discover and adaptation is pure
overhead. Note the min ESS is nearly identical (823.1 vs 829.9): adaptive reaches
the same sampling quality, it just spends 26% more gradients getting there.
Against the parametrization a user would actually have written, it is 51× better.

## Effect of the halo-recording regression

`34ce034` collapsed the halo pool the adaptation reads from. It was fixed in
`095efb0`, landed as `c8fed88`. **No arm was a clean control**: the nonlinear
reparametrization fit at `src/adaptive_warmup_mcmc.jl:379` is gated on
`nonlinear_adapt`, but the linear metric/scale selection at `:381-383` reads the
same pool unconditionally, so `plain` and `fixed_centered` were degraded too.
Full diff in `compare.jl`'s output; the two effects that matter:

**1. The stuck-adaptation defect was a regression artifact, and it is gone.**
The pre-fix run found 2 of 40 adaptive runs where `c` finished at exactly its
starting value on every coordinate — the signature of `optimize!`'s per-index
early return (`nobs(or) > 2 || return idx => value`, `src/Reparametrizations.jl:197`)
firing on a pool too thin to use. After the fix there are none, in either
backend run.

| target | before | after |
|---|---|---|
| `eight_schools` | 0 / 8 | 0 / 8 |
| `radon` partially_pooled | 0 / 8 | 0 / 8 |
| `radon` variable_intercept | 0 / 8 | 0 / 8 |
| `seeds_data` | 1 / 8 (seed 8) | **0 / 8** |
| `funnel` | 1 / 8 (seed 7) | **0 / 8** |

The tail it produced is gone with it. The funnel's worst adaptive seed went from
**16.8 to 746.4 min ESS** against a median of 823.1 — a 44× recovery of the worst
case, and the adaptive arm's spread on that target narrowed from 16.8–950.0 to
746.4–936.8. I reported that one-in-eight failure rate as "the single most
actionable defect these measurements surface". It was not a property of the
method; it was this bug.

**2. It flattered the adaptive arm relative to the noncentered arms.** The fix
helped the well-parametrized arms more than the adaptive one, because they were
the ones whose good trajectories the thin pool was discarding:

| target | arm | before | after | |
|---|---|---|---|---|
| `eight_schools` | hand-written noncentered | 39.25 | 51.56 | +31% |
| `eight_schools` | fixed noncentered | 41.37 | 46.22 | +12% |
| `eight_schools` | **adaptive** | 55.21 | 48.40 | **−12%** |
| `funnel` | plain | 0.52 | 1.53 | +194% |

That reversal is why this document no longer claims adaptive beats the
hand-written noncentered model on eight_schools. On the regressed code it
appeared to, by 41%. On correct code it trails by 6–8%. The conclusion that
survives unchanged is the radon one, where adaptive's margin over every fixed
option is 2.3–2.9× on every run of every backend.

## Cost on the gradient hot path

`ReparametrizedProblem` re-differentiates `ljac_(x_) + dot(g_y, y_)` at every
`logdensity_and_gradient`. Nanoseconds per call, 2000 calls at random positions,
same base, both backends:

| target | bare | ForwardDiff live | ×bare | Enzyme/Const live | ×bare |
|---|---|---|---|---|---|
| `eight_schools` | 453 / 473 | 1654 | ×3.65 | 899 | **×1.90** |
| `radon_partially_pooled` | 28084 / 26513 | 401676 | ×14.30 | 381380 | ×14.38 |
| `radon_variable_intercept` | 43947 / 42084 | 214698 | ×4.89 | 378160 | ×8.99 |
| `seeds` | 2715 / 2796 | 23444 | ×8.64 | 89292 | **×31.93** |
| `funnel` | 46 / 44 | 1016 | ×21.98 | 338 | **×7.62** |

(`bare` is given for each run; the two differ by 3–6%, which is the run-to-run
drift of this single-shot probe.)

**The no-op and live configurations cost the same, in both backend runs.** An
identity reparametrization — source `c` equal to target `c`, so the transform
provably does nothing — pays the same multiple as a live one: within 1–4% on
every Enzyme row and on three of five ForwardDiff rows. The two exceptions are
the ForwardDiff radon rows, which differ by 11% and 16% *in the direction of the
no-op being more expensive* — which the live transform doing strictly more work
cannot produce, so it is the probe's noise rather than a real effect, and it is
bounded by the same 3–6% drift visible in the `bare` column.

The cost is therefore the AD machinery itself, not the reparametrization. This
is the one result here that holds regardless of backend, and it is the one that
says the overhead is addressable.

> **This table is a single-shot probe and the `radon_variable_intercept`
> ForwardDiff cell is an outlier.** Its 214698 is ~40% below what two
> independent multi-round sweeps measure for the same quantity (373613 and
> 376145), which is what makes its ×4.89 look anomalous next to its sibling's
> ×14.30 at the same dimension. The stable cross-backend ratios are the ones in
> [Which AD backend](#which-ad-backend), measured over four processes with the
> backend order rotated; this table is retained because it is what the benchmark
> driver itself records alongside each run, not because it is the better
> measurement. `gradient_overhead` timing each configuration once, in a fixed
> order, is a real weakness of the harness.

## Harness correctness

Two things this benchmark had to get right that are not obvious from the API, and
which would each have silently corrupted the fixed-parametrization arms:

- **`nonlinear_adapt=false` skips the finalization back-transform.** The flag
  gates both the reparametrization fit (`adaptive_warmup_mcmc.jl:379`) and the
  back-transform (`:397`), so a fixed-`c` run returns draws in the *source*
  frame, not the model's. Every fixed arm here therefore applies
  `WarmupHMC.reparametrize!` exactly once itself (`common.jl:252`); the adaptive
  arms must not, because the sampler has already done it. Without that, the
  fixed-noncentered arm would have been scored on draws that were never mapped
  back — measurably wrong, and wrong in the direction that would have made
  adaptive look better.
- **The no-op arm really is a no-op.** `plain` and `fixed c = centered` are
  mathematically the same sampler, and agree to the last gradient evaluation on
  every seed for `seeds_data` (101.7 min ESS, 13068 grads, both arms) and the
  funnel (31.7, 26799) — while differing on eight_schools and both radon models.
  The predictor is whether the spec's **location** is a constant: it is for
  exactly those two targets and a closure over the position vector for the rest,
  and reconstructing `loc + exp(s)·((x − loc)/exp(s))` is the identity only up to
  rounding once `loc` moves with the position. NUTS then amplifies a 1e-16
  gradient difference into a different trajectory. Where the arithmetic permits
  an exact identity the wrapper delivers one; elsewhere the arms stay within seed
  noise. Same split on every run.

  The same mechanism is why three `fixed_noncentered` arms drift 2–5% between
  the two backend runs: Enzyme and ForwardDiff agree only to ~1e-13, and that is
  above the threshold NUTS amplifies.

## What this does not measure

- **Correctness.** These are efficiency measurements. Nothing here verifies the
  draws are from the right distribution — no reference-posterior comparison. That
  is `WarmupHMC:reparam-verify`'s scope, and **no performance claim should ship
  before the two are read together.** An efficiency win on incorrect draws is
  worse than no claim. (The AD backends agreeing with each other to 1e-13, above,
  is a self-consistency check, not a correctness check — three backends can be
  wrong the same way.)
- **One chain per run.** No R-hat, no between-chain diagnostics. `min ESS` is
  within-chain, computed from the draws by `MCMCDiagnosticTools`, not read from
  `result.ess` (which is all zeros unless `monitor_ess=true`).
- **Wall-clock is machine-specific** (`strato2`, single-threaded BLAS). ESS per
  gradient evaluation is the portable number, which is why the verdict is stated
  in it.
- **Five targets.** Every posteriordb posterior with a ready non-empty spec in
  `web/src/posteriordb_reparametrizations.jl`, plus one synthetic funnel because
  posteriordb ships none. Five targets is enough to refute a universal claim and
  not enough to establish one — and for the *backend* question the effective
  count is **two**, since the other three specs carry the `Core.Box` defect.
- **How either backend scales in `d`.** Dimension and the boxed-capture defect
  are perfectly confounded across these five targets: the three boxed specs are
  the three larger models. No claim about backend scaling in `d` — in either
  direction — can be drawn from this document until the captures are fixed and
  the comparison re-run.
- **One sampler.** Single-chain `adaptive_warmup_mcmc` only.
  `clustered_warmup_mcmc` has no reparametrization hooks at all and
  `cooperative_warmup_mcmc` was not measured.
- **Two AD backends, and only one annotation choice explored in depth.** Both
  arrive through DifferentiationInterface, per the standing instruction to use DI
  where possible and Enzyme where not; Mooncake and ForwardDiff are excluded as
  package defaults, and ForwardDiff appears here only as the comparison baseline.
  `AutoForwardDiff()` in `web/src/test/` is a deliberate frozen-baseline harness
  pin, documented at the point of use in `web/src/test/ad_backend.jl` — not a
  recommendation and not a site to change. The one place still selecting forward
  mode for real work is the shipped consumer, `web/src/WarmupHMCWeb.jl:148`.
  Measured against the specs as they stand today, switching it makes
  `seeds`-shaped targets **slower** — but that is the boxed-capture defect
  talking, not reverse mode, and the sensible order is to fix the captures first
  and then measure the switch rather than to decide it on these numbers.

  **Two claims in `src/Reparametrizations.jl`'s docstring are not supported by
  these measurements**, and both are load-bearing for a 1.0 manual:

  1. That reverse mode is the right default because "forward mode costs
     `ceil(n / chunksize)` sweeps … and the gap opens up exactly where
     reparametrization is worth doing — high-dimensional hierarchical models."
     Measured, Enzyme/`Const` is faster only on the two `d = 10` targets and
     slower on all three larger ones, `seeds` by 2.3–5.4×. The mechanism may
     well be real; it is not what dominates the measured cost.
  2. That "`AutoEnzyme()` is what this project benchmarks against." This is the
     benchmark. It now runs Enzyme by default — but as
     `AutoEnzyme(; function_annotation = Enzyme.Const)`, because a bare
     `AutoEnzyme()` raises `EnzymeMutabilityException` on this objective and
     cannot be benchmarked against at all.

  Both are `src/` matters and belong to WarmupHMC, not to this document; they are
  recorded here because this is where the evidence is.
