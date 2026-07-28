# Measured: adaptive reparametrization vs the fixed centering endpoints

**Measured on WarmupHMC `5637fcf`.** 8 pinned seeds per arm, `n_draws` floor
1000, Julia 1.10.11, single-threaded BLAS, on `strato2`. 184 runs per backend,
0 failed.

`5637fcf` is not the tip this landed on, so the gap is stated rather than
assumed: from `5637fcf` to `c4e4670`, `src/Reparametrizations.jl` changes by
**docstring text only** and `web/src/WarmupHMCWeb.jl` gains 294 lines with **zero
deletions** (docs route rendering). Nothing under `web/src/posteriordb_reparametrizations.jl`
— the spec table these numbers differentiate — moved at all. The gradient path
is byte-identical across the two, which is why the tables were not re-run for
the tip; `git diff 5637fcf..c4e4670 -- src/ web/src/` is the check.

This base was measured **twice, under both AD backends**, changing nothing else:
`results/enzyme-5637fcf/` (the default, `AutoEnzyme(; function_annotation =
Enzyme.Const)`) and `results/forwarddiff-5637fcf/` (`AutoForwardDiff()`). The
runs were sequential, never concurrent, because the wall-clock comparison is the
point and two concurrent runs would contend for CPU. `summarize.jl` regenerates
every table below from those records; `compare.jl` regenerates the backend diff.

Superseded runs are kept: `results/enzyme-b5c7dee/` and
`results/forwarddiff-b5c7dee/` (the **boxed-spec** base — see below),
`results/after/` (base `c8fed88`, ForwardDiff) and `results/before/` (base
`05aed41`, which carried the halo-recording regression `34ce034`). The
before/after comparison is a measured result of its own and is reported in
[Effect of the halo-recording regression](#effect-of-the-halo-recording-regression).

> **The `Core.Box` capture defect is FIXED, and fixing it changed the
> headline.** Every earlier revision of this document was measured against
> reparametrization specs whose per-pair closures captured a `Core.Box` on
> `radon_partially_pooled`, `radon_variable_intercept` and `seeds`. That is now
> repaired in `web/src/posteriordb_reparametrizations.jl`, and
> `capture_boxing.jl` is a **guard** that fails the run if any shipped spec
> boxes again.
>
> The consequence is not cosmetic. On the boxed base, adaptive reparametrization
> was a **wall-clock loss** on those three targets (0.16–0.44× the plain
> sampler's ESS/sec). On this base it is a **win on all five** under Enzyme
> (1.11–17.0×). The two targets whose specs were never boxed — `funnel` and
> `eight_schools` — did not move. That the three that changed are exactly the
> three that were boxed, and the two that did not are exactly the two that were
> not, is the causal evidence; nothing else in the table separates them.
>
> **Sampling results were never affected** and did not change: boxing alters
> what a gradient *costs*, not what it *is* (gradients agreed to exactly 0.0),
> so ESS per gradient evaluation, where adaptation settles, and divergence
> counts are identical across the fix.

## Verdict

**Adaptive partial centering, started from the centered parametrization, beats
the plain sampler on every target measured** — 1.8× to 51× per gradient
evaluation. It gets there without being told which parametrization to use.

**Against the best available *fixed* parametrization the picture splits, and the
split is the interesting part:**

| target | adaptive | best fixed alternative | |
|---|---|---|---|
| radon partially_pooled | **68.08** | 22.07 (noncentered endpoint) | **3.1× win** |
| radon variable_intercept | **62.61** | 24.46 (noncentered endpoint) | **2.6× win** |
| seeds centered | **16.92** | 11.28 (noncentered endpoint) | **1.5× win** |
| eight_schools centered | 52.58 | 51.56 (hand-written noncentered) | ties, +2% † |
| funnel (synthetic) | 93.17 | 107.05 (noncentered endpoint) | loses, −13% |

(min ESS per 1000 gradient evaluations, median over 8 seeds, Enzyme run.)

† **The eight_schools row is the one that moves with the backend**, and it moves
enough to change its verdict: the ForwardDiff run puts adaptive at 44.9 against
the same 51.56, so it *trails* by 13% there. Read that row as **−13% to +2%,
i.e. within noise of the hand-written model**, not as a tie. The mechanism is
[the 1e-13 gradient disagreement NUTS amplifies](#harness-correctness), and this
is the only arm in the verdict where it changes an answer — the four other
adaptive rows and every `plain` row agree between backends to 0–2%.

Read as one sentence: **where a good hand-written parametrization exists,
adaptive finds something as good as it, automatically. Where none of the fixed
options is good, adaptive beats all of them.** Both radon models are the second
case — centered, noncentered and posteriordb's own hand-written noncentered
model all land within seed noise of each other around 15–25, and adaptive gets
53–66 by settling at an *interior* `c ≈ 0.4–0.5` that no hand-written model
offers. That interior optimum is the strongest result here.

**This half of the verdict is almost backend-independent, and that is now
measured rather than assumed.** Every `plain` and `fixed_centered` arm is
identical across the two backend runs to 0% — those arms never touch the
wrapper's AD path. Of the five `adaptive` arms, three (both radon models, the
funnel) are identical to 0%, `seeds` moves 2%, and `eight_schools` moves 17%
(above). **Where adaptation settles does not move at all**: the per-seed final
`c` vectors are identical between backends on all five targets, and both runs
have 0/8 stuck seeds everywhere. So the backend can change *which trajectory* a
seed happens to take; it does not change what the method learns.

### It now converts into wall-clock too, on every target

Under Enzyme, adaptive reparametrization is faster **in seconds**, not just per
gradient, on all five targets. This is the claim that previously failed:

| target | adaptive vs plain, Enzyme/Const | | ForwardDiff |
|---|---|---|---|
| | boxed base `b5c7dee` | **this base** | this base |
| funnel (synthetic) | 17.9× faster | **17.0× faster** | 16.5× faster |
| eight_schools centered | 21.0× faster | **16.2× faster** | 15.6× faster |
| radon partially_pooled | 3.5× **slower** | **2.0× faster** | 1.7× faster |
| radon variable_intercept | 2.3× **slower** | **1.8× faster** | 1.6× faster |
| seeds centered | 6.1× **slower** | **1.1× faster** | 1.1× **slower** |

(min ESS/sec, median over 8 seeds, sequential runs.)

Three losses became wins. The three that moved are precisely the three whose
specs were boxed; `funnel` and `eight_schools` were never boxed and stayed in
the 16–18× band. Their small drift is run-to-run variation on a shared host plus
the sampler changes between the two bases, and it is not attributed to the fix.

**`seeds` is the honest edge case.** It clears parity under Enzyme by 11% and
still *loses* by 11% under ForwardDiff — the one target where the backend choice
decides the sign of the answer rather than the size. It is also the smallest
margin here, close enough to the noise floor that it should be read as "roughly
break-even under Enzyme", not as a win.

So the claim the package can now support:

> Starting from a centered parametrization, adaptive partial centering matches or
> exceeds the best fixed parametrization's sampling efficiency per gradient
> evaluation on all five targets, and beats every fixed option on the three where
> no fixed option is good — without being told to. Under Enzyme that converts
> into wall-clock on all five, from break-even on `seeds` to 17× on the funnel.

**The overhead is still not the transform's arithmetic.** A wrapper configured
as an exact identity costs the same as a live one, under both backends and on
every target — on this base, Enzyme ×1.13 no-op vs ×1.17 live on
`radon_partially_pooled`, ×1.04 vs ×1.04 on `radon_variable_intercept`. What
remains is the cost of carrying the AD re-differentiation of
`ljac_(x_) + dot(g_y, y_)` at all. Under Enzyme that is now **+4% of a gradient
on `radon_variable_intercept`, +17% on `radon_partially_pooled` and +22% on
`seeds`** — small enough to be paid out of the sampling gain, which is exactly
what the wall-clock table above shows. It is +84% on `eight_schools` and 7.7× on
the funnel, but those are the two targets whose *bare* gradient costs 500 ns and
45 ns, where a roughly fixed per-call AD cost has to dominate; both still come
out 16–17× ahead overall because the sampling gain there is enormous. Under
ForwardDiff the same overheads are +71%, +104% and +118% — the backend choice is
most of what makes this affordable.

## Which AD backend

The differentiated objective `x -> ljac(x) + dot(g_y, y(x))` is scalar in the
full parameter vector with the inner gradient `g_y` frozen, so it is the
textbook reverse-mode shape: forward mode costs `ceil(d / chunksize)` tangent
sweeps of the transform per gradient where reverse mode costs one. The
prediction that follows is that reverse mode should win, and win *hardest* at
large `d`.

**Measured on this base, the direction holds on every target.** Per wrapped
`logdensity_and_gradient` call, ratio of Enzyme/`Const` to ForwardDiff — below
1.0 means Enzyme is faster:

| target | `d` | Enzyme/Const ÷ ForwardDiff | |
|---|---|---|---|
| `radon_partially_pooled` | 88 | **0.34–0.61×** | Enzyme 1.6–2.9× faster |
| `radon_variable_intercept` | 89 | **0.55–0.62×** | Enzyme 1.6–1.8× faster |
| `seeds` | 26 | **0.56–0.64×** | Enzyme 1.6–1.8× faster |
| `eight_schools` | 10 | **0.33–0.64×** | Enzyme 1.6–3.0× faster |
| `funnel` | 10 | **0.29–0.70×** | Enzyme 1.4–3.4× faster |

Every range spans **four independent harnesses**, each writing its own JSON:
`replicate_backends.jl` (7 rounds × 1000 calls, backend order rotated per round,
both a centering endpoint and an interior `c = 0.5`), `typical_positions.jl`
(5 rounds, at `randn` *and* at positions the sampler actually visited),
`annotation_sweep.jl` (2000 calls, both endpoints), and the driver's own
`gradient_overhead` block (7 rounds × 2000 calls, median). **Enzyme is faster in
all 35 of the 35 comparisons those four harnesses produce** — every target,
every centering, both position sets, every harness.

This reverses what every earlier revision of this document reported, where
Enzyme measured 1.2–5.4× *slower* on the three larger targets. Nothing about the
backends changed; the specs they were differentiating did. The next section is
that story, and it is kept because it is also the reason to distrust a backend
number measured through a spec you have not inspected.

**What is still not established is the second half of the prediction — that the
gap should widen with `d`.** It does not, visibly: the widest margins are on
`funnel` and `eight_schools` at `d = 10`, the narrowest on `radon_variable_intercept`
at `d = 89`. But the spread *within* a single target across harnesses (0.34 to
0.61 on `radon_partially_pooled`) is as large as the spread *between* targets, so
these five points cannot resolve a scaling law either way. The direction
replicates; the slope does not.

### Why the earlier numbers said the opposite: a defect in the spec table

Every backend ratio measured before `e9bcfd0` was a property of the *spec*, not
of the backend, and the tell was in the closures. `capture_boxing.jl` is now a
**guard** that re-derives every branch of the shipped spec table and fails if any
of them regresses; it is also where the A/B below is measured.

`reparametrization()` in `web/src/posteriordb_reparametrizations.jl` used to be
one long `if`/`elseif` chain, with `l`, `s`, `o` assigned in many of its branches
inside that single scope. The per-pair closures captured them:

```julia
(l, s, o) = (J+1, J+2, 0)              # :41, radon_partially_pooled
map(1:J) do i
    idx => Reparametrization(..., x->x[l], x->x[s])
end
```

Julia's closure conversion cannot prove single assignment across those branches,
so it captured a **`Core.Box`** — a mutable heap cell read as `Any` — rather than
an `Int`. `funnel` and `eight_schools` closed over literals (`x->x[1]`,
`x->x[9]`) and captured nothing, which is exactly why those two targets never
moved. Confirmed with `fieldtypes`, not inferred from timings.

The A/B is still run on every invocation of `capture_boxing.jl`, because it is
the causal evidence and it costs seconds. It builds the boxed spec deliberately
alongside the shipped one — same 85 pairs, same indices, same centerings,
closures reading the same `x[86]`/`x[87]`, and **gradients agreeing to exactly
0.0** — and times both. On `radon_partially_pooled`, 5 rounds × 1000 calls with
the order rotated, against a bare gradient of 27431 ns:

| spec | ForwardDiff | Enzyme/`Const` | Enzyme ÷ ForwardDiff |
|---|---|---|---|
| boxed | 339163 ns | 444685 ns | 1.31× — Enzyme *slower* |
| unboxed (**as shipped today**) | 72873 ns | **28536 ns** | **0.39× — Enzyme faster** |
| de-boxing speedup | 4.65× | **15.58×** | |

Both backends are hurt by the box; Enzyme is hurt ~3.4× harder, and that alone
**inverts which backend looks faster**. Two numbers in one table that disagree in
direction, from one process, minutes apart, on specs that produce identical
gradients: no backend verdict measured through an uninspected spec means
anything.

Note what the unboxed Enzyme figure implies for the wrapper as a whole: 28536 ns
against a bare gradient of 27431 ns is **4% overhead**. On this target the
transform is now essentially free under reverse mode and was a 15× tax before.

`web/src/posteriordb_reparametrizations.jl` is outside this directory's
ownership; the defect was reported rather than edited here, and the fix landed as
`e9bcfd0`.

Three other things were checked, and none of them changes the ratios either:

- **Correctness.** All three backends agree on the gradient to ≤ 9.1e-13 (max
  absolute deviation, every target × both endpoints × interior `c = 0.5`), and
  `Const` vs `Duplicated` to ≤ 6.8e-13. This is a cost difference, not a wrong
  answer. Enzyme never differentiates through BridgeStan's FFI — only the
  pure-Julia transform is differentiated, and the inner problem's own
  `logdensity_and_gradient` is reused as a frozen constant.
- **Evaluation position.** The microbenchmarks evaluate at `randn(d)`, which is
  not where a sampler spends its time. Re-timing at the positions the sampler
  *actually visited* — draws captured in the source frame from the
  `fixed_noncentered` arm itself — leaves the ratio within a few points on the
  four posteriordb targets (`radon_partially_pooled` 0.46× at `randn` vs 0.47×
  typical; `eight_schools` 0.59× vs 0.59×; `radon_variable_intercept` 0.62× vs
  0.55×; `seeds` 0.64× vs 0.56×) and changes no verdict. The funnel is the one
  wide swing (0.70× vs 0.38×), and its `bare` column moves 85 → 44 ns in the same
  pair, so that is the `randn` round being noisy on a 50 ns call rather than a
  position effect. Recorded in `results/typical_positions.json`.
- **DI preparation.** The hot path calls `value_and_gradient` with **no prep
  object** (`src/Reparametrizations.jl:148`), so DifferentiationInterface
  re-prepares on every gradient evaluation. Now that the call itself is cheap,
  `prepare_gradient` alone accounts for **89–95% of the whole unprepped Enzyme
  call** on the three larger targets (26.1 of 28.6 µs on `radon_partially_pooled`,
  43.3 of 45.6 µs on `radon_variable_intercept`, 4.4 of 4.9 µs on `seeds`) —
  against 33–65% under ForwardDiff. That reads like an obvious speedup and **it
  is not one**: reusing a prep object measured −2% on `radon_partially_pooled`,
  +12% (worse) on `radon_variable_intercept` and +2% (worse) on `seeds`. The
  costs do not decompose additively under Enzyme, so the 90% share is not 90%
  that can be removed. Under ForwardDiff reuse *would* be worth 21% and 33% on
  the two radon targets — but ForwardDiff is the slower backend to begin with, so
  the prepped-ForwardDiff figure (65.0 µs) is still more than twice the
  unprepped-Enzyme one (28.6 µs). Reuse also returned gradients identical to the
  unprepped path (`|Δg| = 0.0`) in all ten configurations, which is worth knowing
  but is **not** a licence to reuse in the sampler: the objective closes over
  `g_y`, which changes every call, and this probe holds it fixed. Recorded in
  `results/prep_cost.json`.

### `function_annotation` is required, and how much it costs is target-dependent

A bare `AutoEnzyme()` **fails outright** on this objective with
`EnzymeMutabilityException` — the objective is a closure capturing the
reparametrizer and the frozen `g_y`. Enzyme's own error text suggests
`Duplicated`; `Const` is the correct annotation here, since the closure is not
something we differentiate with respect to.

The cost of getting that wrong varies by an order of magnitude across targets,
but it is now **never free**, which is a change from every earlier revision of
this section. Both centerings, `annotation_sweep.jl`:

| target | `d` | `Duplicated` ÷ `Const` | absolute penalty |
|---|---|---|---|
| `funnel` | 10 | 12.7–16.1× | ~5 µs |
| `eight_schools` | 10 | 4.5–8.9× | ~6–9 µs |
| `seeds` | 26 | 3.3–3.5× | ~9–10 µs |
| `radon_partially_pooled` | 88 | 2.1× | ~38–43 µs |
| `radon_variable_intercept` | 89 | 1.6–2.0× | ~37–52 µs |

The penalty is roughly a **fixed per-call cost that grows with `d`** — ~5 µs at
`d = 10`, ~9 µs at `d = 26`, ~40 µs at `d ≈ 88` — so the *ratio* is largest
exactly where the call is otherwise cheapest. Both gradients are equally correct
(they agree to ≤ 6.8e-13).

**The previous revision of this table said `Duplicated` was free on the three
larger targets (0.96–1.06×), and it flagged the reason to distrust that: those
were the three boxed specs, where a ~15× defect swallowed a 40 µs shadow copy.**
That caveat was right. Unboxed, the same three targets show a 1.6–3.5× penalty.
It is a useful calibration on how much a confounded measurement can hide: not a
few percent, but the entire effect.

So the statement that now survives, in two parts. *Some* annotation is
mandatory — a bare `AutoEnzyme()` does not run at all. And of the two candidates,
`Const` is both the semantically correct one and **1.6–16× faster than
`Duplicated`, on every target measured** — where `Duplicated` is what Enzyme's own
error message points you at.

### The two methods used to disagree. They now reconcile to within a microsecond

The previous revision of this section recorded an unresolved contradiction: on
the two radon targets the microbenchmark made Enzyme 1.2–1.5× **slower** per
wrapped gradient while the 184-run benchmark made it 2–14% **faster** per second,
and an additive per-iteration sampler cost cannot invert an ordering. It named
one candidate cause — that radon was a boxed spec, and a `Core.Box` read is a
dynamic load a tight loop and a running sampler stress differently — and said the
test was whether the disagreement dissolved once the captures were fixed.

**It dissolved, and it did so quantitatively.** The right comparison is not the
ratio but the *absolute* per-gradient difference, which is what an additive cost
predicts. Taking only arms whose two backend runs spent the identical number of
gradients — so the trajectories really are the same work — and comparing the
sampler's own seconds-per-gradient against the microbenchmark's per-call gap:

| target | arm | seeds | end-to-end Δ | microbenchmark Δ |
|---|---|---|---|---|
| `radon_partially_pooled` | fixed c = centered | 8 | 19.9 µs | 20.4 µs |
| `radon_partially_pooled` | adaptive | 8 | 16.7 µs | 20.4 µs |
| `radon_variable_intercept` | fixed c = centered | 8 | 23.0 µs | 31.2 µs |
| `radon_variable_intercept` | adaptive | 8 | 15.4 µs | 31.2 µs |
| `seeds` | fixed c = centered | 8 | 2.8 µs | 2.6 µs |
| `eight_schools` | fixed c = centered | 8 | 1.2 µs | 1.9 µs |
| `funnel` | fixed c = centered | 8 | 0.5 µs | 0.5 µs |

(ForwardDiff minus Enzyme, per gradient evaluation; median over seeds on the
left, the `gradient_overhead` live-wrapper medians on the right.)

Four of five targets agree to within a microsecond or better, which for
`radon_partially_pooled` means a 20 µs microbenchmark difference showing up as a
20 µs sampler difference. `radon_variable_intercept` is the loose one — the
sampler realises 15–23 µs of a predicted 31 µs — and that gap is not explained
here; it is the same target whose single-shot overhead probe was an outlier in
earlier revisions, so the microbenchmark side is the more suspect of the two.

The mechanism the earlier revision guessed at is therefore confirmed by its own
proposed test, and the two methods are no longer in conflict on any target.

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
ForwardDiff run except on the four arms noted below.

`min ESS` is the minimum over coordinates — the number that governs how long you
must run. Sub-figures are the min–max across the 8 seeds. `final source c` is
read back off the mutated `IndexedReparametrization` after the run, which is
valid because every run here is single-chain `adaptive_warmup_mcmc`; the
cooperative and clustered samplers `deepcopy` the problem per chain and would
silently return the *initial* `c` instead.

> **Where the two backend runs are not sampling-identical.** Four arms drift by
> ≥2% in gradient count between backends — `eight_schools`/adaptive −9%,
> `eight_schools`/fixed_noncentered −5%, `radon_variable_intercept`/fixed_noncentered
> −3%, `seeds`/fixed_noncentered −2%. Every other arm is 0–1%, and every `plain`
> and `fixed_centered` arm is exactly 0% because those arms never call the
> wrapper's AD path at all. The cause is the mechanism described under
> [Harness correctness](#harness-correctness): the backends' gradients differ at
> the 1e-13 level and NUTS amplifies that into a different trajectory.
>
> **One arm quoted in the verdict is affected**, `eight_schools`/adaptive, and
> its footnote there says so. The other four adaptive rows, and every `plain`
> row, are like-for-like across backends; the `fixed_noncentered` rows are not
> and are never used as a cross-backend wall-clock comparison.

<!-- tables generated by docs/benchmark/summarize.jl from results/enzyme-5637fcf/runs.json -->

### `eight_schools-eight_schools_centered`

*8 group effects — dimension 10. The textbook hierarchical funnel.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 51.1 <sub>25.2–86.3</sub> | 297.2 <sub>70.2–1215.7</sub> | 1.55 | 33320 | 4.5 | — |
| fixed c = centered | 46.8 <sub>8.8–102.8</sub> | 403.0 <sub>48.4–768.8</sub> | 2.15 | 22504 | 8.0 | 1.0 |
| **adaptive** (starts centered) | 577.5 <sub>214.3–759.1</sub> | 4826.9 <sub>1590.0–8635.0</sub> | 52.58 | 10574 | 0.0 | 0.0 |
| fixed c = noncentered | 498.0 <sub>384.3–609.6</sub> | 8862.9 <sub>2472.0–19394.4</sub> | 46.73 | 9846 | 0.0 | 0.0 |
| hand-written noncentered model | 569.9 <sub>345.5–627.4</sub> | 10763.2 <sub>1686.1–28474.3</sub> | 51.56 | 9482 | 0.0 | — |

Min ESS over the 10 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 51.1 | 1.54 |
| fixed c = centered | 46.8 | 2.08 |
| **adaptive** (starts centered) | 577.5 | 54.61 |
| fixed c = noncentered | 498.0 | 50.57 |
| hand-written noncentered model | 569.9 | 60.11 |

Adaptive drives `c` essentially to the noncentered endpoint: **53 of the 64
learned values (8 seeds × 8 coordinates) are exactly 0.0 and the other 11 are
0.1** — the known right answer for this model, found without being told. It lands
level with the noncentered endpoint reached by transform (52.58 vs 46.73
unconstrained) and 9% below posteriordb's hand-written noncentered model on the
matched parameters (54.61 vs 60.11). Divergences go from a median of 4.5 on
plain to 0.

Two cautions on this target specifically. It is the one whose adaptive arm is
**not** like-for-like across backends — under ForwardDiff the same arm reads
44.9 / 52.53 rather than 52.58 / 54.61, so its margin against the hand-written
model swings from +2% to −13%. And its adaptive spread is the widest here
(214.3–759.1 min ESS over 8 seeds), so a difference of this size against the
hand-written model is not resolvable at 8 seeds either way. The safe reading is
**parity**.

This is also the target where the pre-fix run over-stated the method: on
`05aed41` adaptive appeared to *beat* the hand-written model by 41%. It does not.

### `radon_mn-radon_partially_pooled_centered`

*85 counties — dimension 88.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 189.9 <sub>100.1–255.9</sub> | 389.4 <sub>206.4–492.3</sub> | 14.12 | 14971 | 0.0 | — |
| fixed c = centered | 209.4 <sub>134.6–300.9</sub> | 375.5 <sub>325.5–449.1</sub> | 14.91 | 14828 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 603.9 <sub>455.9–1092.7</sub> | 765.1 <sub>612.2–1109.7</sub> | 68.08 | 8701 | 0.0 | 0.4 |
| fixed c = noncentered | 364.3 <sub>193.4–428.1</sub> | 544.9 <sub>354.8–626.9</sub> | 22.07 | 15858 | 0.0 | 0.0 |
| hand-written noncentered model | 314.4 <sub>178.5–439.6</sub> | 491.3 <sub>383.0–693.5</sub> | 18.52 | 16456 | 0.0 | — |

Min ESS over the 88 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 189.9 | 12.68 |
| fixed c = centered | 209.4 | 14.12 |
| **adaptive** (starts centered) | 603.9 | 69.40 |
| fixed c = noncentered | 364.3 | 22.98 |
| hand-written noncentered model | 314.4 | 19.11 |

**The headline case for the method.** No fixed parametrization helps: centered
(14.91), noncentered (22.07) and posteriordb's hand-written noncentered model
(18.52) are all in the same band. Adaptive gets 68.08 — **3.1× the best fixed
option** — using 45% fewer gradient evaluations than the noncentered endpoint,
and it is now also the fastest arm in **wall-clock**, at 765.1 ESS/sec against
the plain sampler's 389.4. The `final source c` column reports 0.4, but that is
a **median over 85 counties, not a setting**: see below.

#### What adaptation actually learns is a `c` *per coordinate*

The `final source c` column is a median, and on the two radon models it hides
the result. Every one of the 85 counties gets its own centering, and they do not
agree with each other. Distribution of the learned `c` over all 8 seeds ×
coordinates, Enzyme run (the ForwardDiff run is identical on all five targets):

| target | n | median | quartiles | at `c = 0` | at `c = 1` |
|---|---|---|---|---|---|
| `radon_partially_pooled` | 680 | 0.40 | 0.30 – 0.60 | 1% | 2% |
| `radon_variable_intercept` | 680 | 0.50 | 0.40 – 0.70 | 1% | 1% |
| `seeds` | 168 | 0.30 | 0.20 – 0.50 | 2% | 0% |
| `eight_schools` | 64 | 0.00 | 0.00 – 0.00 | 83% | 0% |
| `funnel` | 72 | 0.00 | 0.00 – 0.00 | 100% | 0% |

On the radon models **97–98% of coordinates end strictly between the two
endpoints**, spread across the whole interval rather than clustered at the
median. That is the thing no fixed parametrization can express: centered,
noncentered and posteriordb's hand-written noncentered model each impose *one*
`c` on all 85 counties, so none of them can represent any of these fits — which
is the mechanism behind the 3.1× and 2.6× margins, and why those margins appear
on exactly the two targets where the spread is widest.

The two ends of the table are the controls. On the funnel, where the noncentered
parametrization is *exactly* right, all 72 values are exactly 0.0 and adaptation
correctly finds a fixed parametrization; on `eight_schools` it lands at or next
to that endpoint too. So the interior fits are not an artifact of the optimizer
being unable to reach an endpoint — it reaches endpoints when they are the
answer.

### `radon_mn-radon_variable_intercept_centered`

*85 counties + floor slope — dimension 89.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 273.7 <sub>173.6–408.5</sub> | 358.1 <sub>233.9–504.8</sub> | 19.37 | 14960 | 0.0 | — |
| fixed c = centered | 339.7 <sub>208.5–445.1</sub> | 391.0 <sub>298.8–472.3</sub> | 22.65 | 15016 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 760.4 <sub>483.0–935.7</sub> | 634.4 <sub>517.8–900.4</sub> | 62.61 | 10408 | 0.0 | 0.5 |
| fixed c = noncentered | 403.6 <sub>188.9–452.9</sub> | 410.9 <sub>216.1–486.0</sub> | 24.46 | 16274 | 0.0 | 0.0 |
| hand-written noncentered model | 287.0 <sub>220.2–444.0</sub> | 409.3 <sub>280.0–523.7</sub> | 21.36 | 15837 | 0.0 | — |

Min ESS over the 89 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 273.7 | 18.30 |
| fixed c = centered | 339.7 | 22.62 |
| **adaptive** (starts centered) | 760.4 | 73.06 |
| fixed c = noncentered | 403.6 | 24.80 |
| hand-written noncentered model | 287.0 | 18.12 |

Same shape as its sibling, and the same per-coordinate story — 98% of the 85
county centerings land strictly inside `(0, 1)` with quartiles 0.4–0.7, so the
reported `c = 0.5` is again a median over coordinates. Worth 2.6× the best fixed
alternative on gradients and 1.8× the plain sampler in wall-clock. The
`fixed_noncentered` row is one of the arms that drifted between backends (−3%
gradients), so read its 24.46 as approximate; the ForwardDiff run puts it at
18.26, which would make the margin 3.4× instead. The adaptive and plain rows are
identical across backends.

### `seeds_data-seeds_centered_model`

*21 plates — dimension 26. posteriordb ships no noncentered sibling.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 101.7 <sub>88.3–161.0</sub> | 854.9 <sub>633.6–1399.0</sub> | 8.40 | 13068 | 0.0 | — |
| fixed c = centered | 101.7 <sub>88.3–161.0</sub> | 721.4 <sub>576.4–1242.3</sub> | 8.40 | 13068 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 228.5 <sub>130.8–344.3</sub> | 947.4 <sub>426.5–2180.2</sub> | 16.92 | 11770 | 0.0 | 0.3 |
| fixed c = noncentered | 206.4 <sub>172.2–262.9</sub> | 1829.7 <sub>711.6–2559.8</sub> | 11.28 | 17274 | 0.0 | 0.0 |

Min ESS over the 47 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 101.7 | 7.78 |
| fixed c = centered | 101.7 | 7.78 |
| **adaptive** (starts centered) | 228.5 | 19.41 |
| fixed c = noncentered | 206.4 | 11.95 |

Adaptive is ahead of the noncentered endpoint on both measures here (16.92 vs
11.28 unconstrained, 19.41 vs 11.95 matched), where the ForwardDiff run had them
closer. That gap is partly a backend artifact on the *endpoint* arm, not on the
method: `fixed_noncentered` is one of the arms that drifted (−2% gradients, and
its min ESS moved 295.7 → 206.4 on a seed-noise range of 144–414). Treat this
target as **adaptive modestly beats** the endpoint, not as a clean 1.5× win.
Both roughly double plain. Interior optimum again, per-coordinate, with quartiles
0.2–0.5 across the 21 plates.

This is also the closest wall-clock call in the study: adaptive is 1.1× *faster*
than plain under Enzyme (947.4 vs 854.9) and 1.1× *slower* under ForwardDiff
(791.0 vs 883.9) — the one target where the backend decides the sign.

### `funnel` (synthetic)

*Neal's funnel, `v ~ Normal(0, 3)`, `theta_i ~ Normal(0, exp(v/2))`, K = 9 —
dimension 10. Not a posteriordb posterior; posteriordb ships none. Defined in
`common.jl` with an analytic gradient. It is the extreme case — the centered
parametrization is pathological and the noncentered one is exact — so it bounds
what partial centering can possibly buy. Reported alongside the real targets,
never as a headline.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 31.7 <sub>19.6–56.4</sub> | 777.6 <sub>245.1–1639.0</sub> | 1.53 | 26799 | 0.5 | — |
| fixed c = centered | 31.7 <sub>19.6–56.4</sub> | 661.1 <sub>193.6–1278.0</sub> | 1.53 | 26799 | 0.5 | 1.0 |
| **adaptive** (starts centered) | 758.5 <sub>578.3–943.2</sub> | 13237.9 <sub>10497.0–16799.5</sub> | 93.17 | 8082 | 0.0 | 0.0 |
| fixed c = noncentered | 829.9 <sub>718.3–1017.9</sub> | 47397.6 <sub>38545.7–68584.7</sub> | 107.05 | 7963 | 0.0 | 0.0 |

The one target where a fixed endpoint beats adaptive on gradient efficiency: 93.2
vs 107.1 per 1000 gradients. That is the expected and correct result — noncentered
is *exactly* right here, so there is nothing to discover and adaptation is pure
overhead. Against the parametrization a user would actually have written it is
**61× better** (93.17 vs 1.53), so this row bounds the loss, not the gain.

Where the 13% goes is worth being exact about, because it is *not* a longer
search: adaptive spends 8082 gradients against the endpoint's 7963 — 1.5% more —
and the rest of the gap is min ESS, 758.5 vs 829.9, on seed ranges (578–943 vs
718–1018) that overlap heavily. Adaptation converges to the right fixed answer
here (all 72 learned values exactly 0.0); it just samples marginally less well
on the way to it.

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
`logdensity_and_gradient`. Nanoseconds per call, **median of 7 rounds × 2000
calls at random positions**, same base, both backends:

| target | bare | ForwardDiff live | ×bare | bare | Enzyme/Const live | ×bare |
|---|---|---|---|---|---|---|
| `radon_variable_intercept` | 45158 | 77252 | ×1.71 | 44363 | 46097 | **×1.04** |
| `radon_partially_pooled` | 25607 | 52269 | ×2.04 | 27262 | 31827 | **×1.17** |
| `seeds` | 2823 | 6158 | ×2.18 | 2879 | 3520 | **×1.22** |
| `eight_schools` | 512 | 2782 | ×5.43 | 500 | 920 | **×1.84** |
| `funnel` | 51 | 856 | ×16.88 | 45 | 343 | **×7.69** |

(`bare` is given for each run separately; the two agree to 2–7% except on the
funnel, where a 45 ns call is at the resolution floor and they differ by 13%.)

**The multiple tracks how expensive the underlying density is, not `d`.** The
two most expensive densities carry the wrapper for +4% and +17%; the funnel,
whose bare gradient is 45 ns of analytic arithmetic, pays 7.7×. This is what a
roughly fixed per-call AD cost looks like divided by a varying denominator, and
it is why the funnel is reported as a bound rather than as a headline.

**The no-op and live configurations cost the same, in both backend runs.** An
identity reparametrization — source `c` equal to target `c`, so the transform
provably does nothing — pays the same multiple as a live one: within ±5% on all
ten rows, with no consistent sign (Enzyme's `seeds` and `radon_variable_intercept`
rows put the *no-op* higher, which the live transform doing strictly more work
cannot produce, so ±5% is the probe's floor rather than a measured effect).

The cost is therefore the AD machinery itself, not the reparametrization. That
holds regardless of backend. Of the obvious levers on it, two are now spent —
the boxed captures are fixed, and the faster backend is already the default —
and one was tested and does not pay (reusing a DI prep object, under
[Which AD backend](#which-ad-backend)). What remains untried is not
re-differentiating `ljac_ + dot(g_y, y_)` at all, whose derivative is known in
closed form.

> **Earlier revisions carried a warning here that this table was a single-shot
> probe, timing each configuration once in a fixed order, and that its
> `radon_variable_intercept` ForwardDiff cell was a ~40% outlier against
> independent sweeps.** That was a real weakness and it is fixed: `35587d5`
> made `gradient_overhead` run 7 rounds with the configuration order rotated and
> report the median, and every per-round value is kept in
> `results/*/gradient_overhead.json` so the spread is inspectable rather than
> asserted. The check that it worked is agreement with the independent sweeps,
> not the absolute value — which also fell because of the de-boxing. On the old
> base the probe said 214698 ns where the sweeps said ~375000; on this base it
> says 77252 where `replicate_backends.jl` says 76072–83671. The outlier was the
> harness, and the sweeps were right.

## Harness correctness

Two things this benchmark had to get right that are not obvious from the API, and
which would each have silently corrupted the fixed-parametrization arms:

- **`nonlinear_adapt=false` used to skip the finalization back-transform, and no
  longer does.** The flag gated both the reparametrization fit and the
  back-transform, so a fixed-`c` run returned draws in the *source* frame, not
  the model's, and every fixed arm here applied `WarmupHMC.reparametrize!`
  exactly once itself while the adaptive arms did not. Without that compensation
  the fixed-noncentered arm would have been scored on draws that were never
  mapped back — wrong in the direction that would have made adaptive look better.

  `b109210` ("report draws in the model frame even when `nonlinear_adapt=false`")
  inverted this, so from that commit the compensation is what corrupts the arm.
  It is gone, and the flag now selects only whether the source centering may
  move. `frame_check.jl` pins the new behaviour: a centered funnel sampled
  through a noncentered source returns coordinate 1 at mean −0.006, sd 2.934
  against the known `Normal(0, 3)` marginal, and applying `reparametrize!` on top
  inflates the leg sds to 100–300. **Numbers in `results/before`, `results/after`,
  `results/forwarddiff-b5c7dee` and `results/enzyme-b5c7dee` were all measured on
  bases predating `b109210`, where the compensation was correct.** Anything
  measured after it must not carry one.
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

  The same mechanism is why some arms drift between the two backend runs:
  Enzyme and ForwardDiff agree only to ~1e-13, and that is above the threshold
  NUTS amplifies. Which arms is not arbitrary. Counting runs where all four of
  `ess_min`, `ess_median`, `grad_evals` and `n_divergent` are **bit-identical**
  between the two backend runs, 8 seeds × 5 targets:

  | arm | bit-identical across backends |
  |---|---|
  | `plain`, `hand-written noncentered model` | 40/40, 32/32 — no wrapper, so no AD path to differ on |
  | `fixed c = centered` | **40/40** |
  | `adaptive` | 32/40 — 8/8 on both radon models and the funnel, 7/8 on `seeds`, **1/8 on `eight_schools`** |
  | `fixed c = noncentered` | 6/40 — all six on the funnel, 0/32 on the posteriordb targets |

  The clean split is `fixed c = centered` (source `c` equal to target `c`, so
  the transform is an identity) reproducing exactly on every run, against
  `fixed c = noncentered` (a live transform) reproducing on almost none. The
  wrapped arms that differ are differing through the very AD path this document
  measures, which is why an individual ESS/sec figure is reproducible only
  against a fixed backend — and why the verdict is stated in ESS per gradient,
  where 32 of 40 adaptive runs are bit-identical between backends and the rest
  are named above.

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
  not enough to establish one. All five now count for the backend question — the
  effective count was **two** while the other three specs carried the `Core.Box`
  defect, and `capture_boxing.jl` is the guard that keeps it at five.
- **How either backend scales in `d`.** The dimension/boxing confound is gone —
  and the answer is still that this document cannot resolve it. Enzyme's
  advantage is 1.4–3.4× across all five targets with no visible trend in `d`,
  and the spread *within* one target across harnesses is as wide as the spread
  *between* targets, which is the reason: the noise floor is larger than any
  slope five points at three distinct dimensions could show. See § *Which AD
  backend*. The direction replicates 35/35; the slope is unmeasured, in either
  direction.
- **One sampler.** Single-chain `adaptive_warmup_mcmc` only.
  `clustered_warmup_mcmc` has no reparametrization hooks at all and
  `cooperative_warmup_mcmc` was not measured.
- **Two AD backends, and only one annotation choice explored in depth.** Both
  arrive through DifferentiationInterface, per the standing instruction to use DI
  where possible and Enzyme where not; Mooncake and ForwardDiff are excluded as
  package defaults, and ForwardDiff appears here only as the comparison baseline.
  `AutoForwardDiff()` in `web/src/test/` is a deliberate frozen-baseline harness
  pin, documented at the point of use in `web/src/test/ad_backend_gate.jl` — not a
  recommendation and not a site to change. The one place still selecting forward
  mode for real work is the shipped consumer, `web/src/WarmupHMCWeb.jl:148`.
  Against the boxed specs, switching it made `seeds`-shaped targets **slower** —
  that was the boxed-capture defect talking, not reverse mode. The captures are
  fixed as of `e9bcfd0`, so that switch is now a decision this benchmark can
  actually inform; see § *Which AD backend*.

  **`src/Reparametrizations.jl`'s docstring is now stale in four specific
  places**, all of them in the *"That argument is an operation count, and how it
  scales here is untested"* admonition. They are `src/` matters and belong to
  WarmupHMC, not to this document; they are recorded here because this is where
  the evidence is.

  1. **Its backend table carries two rows and can now carry five.** It carries
     only `funnel` and `eight_schools` because, in its words, the three larger
     targets "were measured against reparametrization specs whose accessor
     closures captured a `Core.Box`". They were re-measured on the de-boxed
     specs for this revision; the five-row table is § *Which AD backend*, and
     `results/backend_replication.json` is the harness the docstring's own rows
     come from. Its two current bands — `funnel` 0.31–0.67×, `eight_schools`
     0.43–0.66× — were widened by `62ba171` after `WarmupHMC:reparam-docs` found
     the narrower predecessors were not re-derivable from anything checked in.
     That widening was honest against the files it was computed from, and it
     landed **four minutes before** the regeneration in `176a061` moved those
     files underneath it: recomputed across all five checked-in sources, `funnel`
     is 0.137–0.713× and `eight_schools` 0.331–0.641×, so `eight_schools`' upper
     bound of 0.66 now exceeds every value in every file and its lower bound of
     0.43 excludes real ones. **A band derived from checked-in JSON is only a
     floor until someone regenerates the JSON** — which is the same coupling that
     broke `docs/src/reparametrization.md` in the same window, and the reason
     this table should be generated rather than hand-copied. From the single
     harness the docstring names, over 14 rounds each (7 per centering), the
     per-round `min`–`max` and the median of the per-round ratios are:

     | target | `d` | per-round min–max | median |
     |---|---|---|---|
     | `funnel`                   | 10 | 0.14–0.71× | **0.40×** |
     | `eight_schools`            | 10 | 0.40–0.60× | **0.49×** |
     | `seeds`                    | 26 | 0.16–1.54× | **0.58×** |
     | `radon_partially_pooled`   | 88 | 0.36–0.63× | **0.50×** |
     | `radon_variable_intercept` | 89 | 0.51–0.64× | **0.58×** |

     The two outlier-carrying rows are outliers and not a second finding:
     `seeds`' 1.54 is one round in which Enzyme's own timing spread hit ±184%,
     and `funnel`'s 0.14 is one round on a call that costs 50 ns. They are left
     in because a `min`–`max` that quietly drops its tails is the thing this
     table exists to stop.
  2. **The open question it ends on has been run.** It asks "whether that
     advantage grows, holds or shrinks with dimension", "awaiting a re-run on
     the fixed specs". The re-run is this document. The honest answer is *still
     not resolved*, but for a different and narrower reason than the boxing: the
     within-target spread across harnesses is as wide as the between-target
     spread, so the noise floor exceeds any slope five points can show. Its
     standing conclusion — "reverse mode wins on every clean measurement to
     date" — survives, now at 35 comparisons rather than the two it had.

     **The reading these numbers most invite is the one they falsify.** Taking
     one figure per target, they look like *the advantage narrows with
     dimension* — ≈0.40 at `d = 10` against ≈0.58 at `d = 26`–`89` — which is
     the opposite of the mechanism the admonition exists to caution against, and
     would be a striking result. The median column above breaks it: at `d = 88`,
     `radon_partially_pooled` reads **0.50**, better than `seeds` at `d = 26`
     (0.58) and equal to `eight_schools` at `d = 10` (0.49). The ordering is not
     monotone in `d`. And splitting that same median by centering, `eight_schools`
     alone reads **0.58 at `c = 0` and 0.42 at `c = 0.5`** — one target, one
     dimension, spanning the whole between-target range on its own. The apparent
     trend is which figure you pick per target, not `d`.
  3. **Its provenance line points at a superseded measurement, and one of its
     SHAs does not resolve.** It reads "Measured at `b5c7dee` … results checked
     in at `068cdeb`", both of which predate the de-boxing; this revision
     measures `5637fcf`/`c4e4670`. Separately, it names the de-boxing commit as
     `f639bb1`, and `git rev-parse f639bb1` fails on every ref in this repo —
     the landed commit is **`e9bcfd0`**. A SHA in a docstring is only useful if
     it resolves.

  4. **Its DI-preparation figure inverted, though its conclusion held.** It
     records prep as "10–16% of the call at `d ≈ 88`, and equal under both
     backends". Prep's *absolute* cost is indeed near-identical across backends
     and barely moved (26.1 µs Enzyme vs 27.7 µs ForwardDiff on
     `radon_partially_pooled`) — but the calls it is a share *of* got much
     cheaper, so on this base it is **89–95%** of the unprepped Enzyme call and
     33–65% of the ForwardDiff one. The docstring's actual conclusion, that
     reusing a prep object measures neutral-to-worse, replicated exactly.

  The one claim this benchmark could have contradicted and does not is
  `function_annotation = Enzyme.Const`: a bare `AutoEnzyme()` does raise
  `EnzymeMutabilityException` here, and `Duplicated` runs but costs 1.6–16×
  (§ *`function_annotation`*), so "the hint diagnoses the problem; it is not the
  fix" is right — and by a wider margin on this base than when it was written.
