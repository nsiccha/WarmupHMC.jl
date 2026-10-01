# Measured: adaptive reparametrization vs the fixed centering endpoints

**Measured on WarmupHMC `7b1757a`, with `src/` clean.** 8 pinned seeds per arm,
`n_draws` floor 1000, Julia 1.10.11, single-threaded BLAS, on `strato2`. 184 runs
per backend, 0 failed. The worktree outside `src/` was not clean — this file's
in-progress revision was uncommitted — so both runs record `worktree_dirty: true`
beside `src_dirty: false`, the combination `artifact_currency.jl` accepts.

That base is the one the artifacts **record**, not the one this sentence claims.
Each `runs.json` carries `warmuphmc_sha` and `worktree_dirty`, and the result
directories are named for the recorded SHA so that the path and the provenance
cannot drift apart. They had: an earlier revision of this pair was named for
canonical `c7c6d7c` while recording `7556a15` with `worktree_dirty: true`.
`artifact_currency.jl` was never fooled, because it reads the field rather than
the path — but the author was, by his own directory name. **A directory name is
a label; only the recorded field is provenance.**

It happened again on 2026-09-30, on the other half of the label. A run filed as
`forwarddiff-be6fb23/` was an **Enzyme** run — `WHMC_BENCH_OUT` named ForwardDiff
while `WHMC_BENCH_AD` was unset — and recorded `"ad_backend": "enzyme"`
correctly. The readers pick the pair by directory name, so for one revision the
"backend comparison" compared Enzyme with itself (0 of 184 runs differed in any
non-clock field). Code-currency could not see it: the base was fine.
`artifact_currency.jl` now also checks a `<backend>-<sha>/` directory's name
against its artifact's recorded `ad_backend` and base, and reports a mismatch as
MISLABELED.

`worktree_dirty` is the field with no recourse. A recorded SHA can be checked
against any later tip; an uncommitted tree cannot be checked against anything,
because there is no revision to name. It fails in the reassuring direction — the
SHA beside it looks like full provenance — so it is re-measured rather than
explained. When `d68d680` was this page's base, both arms were re-run from a clean
tree for exactly that reason, and the discarded dirty pair turned out to agree
with them **bit-for-bit on all 184 runs** in every field except wall-clock. That
is the evidence the flag itself could not supply. Harnesses have since recorded
the narrower `src_dirty`, which names the one part of the tree that can change
what runs; the pair below is `src_dirty: false`, and the same bit-for-bit
agreement holds between its Enzyme run and an earlier code-identical one
(`be6fb23`): all 184 runs, every field except wall-clock.

`7b1757a` will fall behind the tip, so the gap is **checked** rather than stated.
`docs/benchmark/code_identical.jl` parses every file under `src/` at two
revisions, strips docstrings and line numbers, and compares the resulting ASTs:

    julia docs/benchmark/code_identical.jl 7b1757a <tip>

The worked example is from the previous base. Against `3597dbc`, canonical when
`d68d680` was measured, it reported **CODE-IDENTICAL across all 12 files**.
The two commits that touched `src/` in between (`1d1bd41`, `879b92c`) add 68 lines and delete 8 across four files, and
every one of those lines is a comment or a docstring — including one that
*corrects a stated behaviour* ("`synchronize!` runs after construction" →
"during construction, ahead of `new`"). The sampler defines the same methods with
the same bodies, so the gradient path cannot have moved, which is why the tables
were not re-run for the tip. That SHA will be stale by the time you read it:
**re-run the script against the tip you have.** What justifies these tables is
the script's verdict, not this sentence.

Run it *before* deciding a matrix is stale, not after. A `git diff --stat`
showing `+68 −8` under `src/` is indistinguishable from a real change to the
sampler, and re-measuring on that basis costs a full matrix to learn that nothing
moved.

The same question asked of every artifact at once is
`docs/benchmark/artifact_currency.jl`, which reads each checked-in result's own
recorded `warmuphmc_sha` and holds it to the tip:

    julia docs/benchmark/artifact_currency.jl

Deliberately not a tally of how many pass. The set of artifacts changes as runs
are added and retired, by authors who never read this paragraph, so any count
written here is a census that goes stale silently — while the script's own output
is correct by construction. The statement worth having is not "this document's
base is fine" but "nothing checked in has quietly gone stale", and the script is
the only thing that can say it.

Reading `git diff` was the old check and it is the weak one here: those two
commits produce a 138-line diff under `src/` — the 76 changed lines above plus
context — in files whose docstrings are long enough to bury a one-line code
change, and an all-prose diff looks exactly like
a mostly-prose diff. One of them even edits a sentence *describing when a
function runs* — the kind of hunk that reads as behaviour at a glance. The
script exits 1 when code really does differ — verified on `b0a1c4f~1..b0a1c4f`,
a real change to the same file — so it is a check that has been observed to
fail, not just to pass.

Outside `src/`, the same span adds 293 lines and deletes 12, all of it docs
infrastructure: `docs/src/.vitepress/` gains a Vega figure component and
`docs/tables.jl` gains rendering helpers. Nothing under
`web/src/posteriordb_reparametrizations.jl` — the spec table these numbers
differentiate — moved at all.

This base was measured **twice, under both AD backends**, changing nothing else:
`results/enzyme-7b1757a/` (the default, `AutoEnzyme(; function_annotation =
Enzyme.Const)`) and `results/forwarddiff-7b1757a/` (`AutoForwardDiff()`). The
runs were sequential, back to back, never concurrent with each other — but
`strato2` was carrying other agents' jobs throughout (load 11–30 on 8 cores), and
[the wall-clock section](#it-converts-into-wall-clock-on-four-targets) measures
what that does to ESS/sec. `summarize.jl` regenerates
every table below from those records; `compare.jl` regenerates the backend diff.

Superseded runs are kept: `results/enzyme-d68d680/` and
`results/forwarddiff-d68d680/` (the previous live pair, replaced by the one
above), `results/enzyme-5637fcf/` and `results/forwarddiff-5637fcf/` (the pair
before that), `results/enzyme-7556a15/` and `results/forwarddiff-7556a15/` (the
dirty-worktree run this pair replaced, kept as the evidence that the dirty tree
changed nothing), `results/enzyme-b5c7dee/` and `results/forwarddiff-b5c7dee/` (the
**boxed-spec** base — see below), `results/after/` (base `c8fed88`, ForwardDiff)
and `results/before/` (base `05aed41`, which carried the halo-recording
regression `34ce034`). The before/after comparison is a measured result of its
own and is reported in
[Effect of the halo-recording regression](#effect-of-the-halo-recording-regression).

Two of those bases — `5637fcf` and `b5c7dee` — are **present in this repository
as objects and contained by zero refs.** They resolve here and in no fresh clone,
and `git gc` may drop them at any time. That is why each of those four
directories carries a `SUPERSEDED` file recording the unreachability alongside
the supersession: the numbers are kept, but their provenance is not
independently checkable and the marker says so rather than implying otherwise.
It is also why `fetch-depth: 0` is necessary but not sufficient for any CI job
that runs `artifact_currency.jl` — no fetch depth recovers a commit that is on
no ref, and the script distinguishes the two cases in its own output.

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
> sampler's ESS/sec). On the de-boxed base `d68d680` it became a **win on all
> five** under Enzyme (1.6–19.3×). The two targets whose specs were never boxed — `funnel` and
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
the plain sampler on every target measured** — 1.4× to 127× per gradient
evaluation. It gets there without being told which parametrization to use.

**Against the best available *fixed* parametrization the picture splits three
ways:**

| target | adaptive | best fixed alternative | |
|---|---|---|---|
| radon partially_pooled | **82.53** | 20.33 (hand-written noncentered) | **4.1× win** |
| radon variable_intercept | **79.19** | 21.00 (centered endpoint) | **3.8× win** |
| funnel (synthetic) | 111.96 | 111.24 (noncentered endpoint) | ties, +1% |
| eight_schools centered | 48.29 | 60.11 (hand-written noncentered) | loses, −20% |
| seeds centered | 7.61 | 12.06 (noncentered endpoint) | **loses, −37%** ‡ |

(min ESS per 1000 gradient evaluations, median over 8 seeds, Enzyme run; the
ForwardDiff run is within 0.01 of it on every adaptive row.)

‡ **`seeds` changed, and not by noise.** At `d68d680` all eight seeds adapted to
an interior `c` (0.1–0.6) and adaptive *won* 1.5× (16.92 against 11.28). On this
base six of the eight never leave the centered start — every learned `c` ends at
exactly 1.0 — and the arm's median gradient count equals `plain`'s. Per-gradient
columns are counts, bit-reproducible on a given `src/`, so this is a behaviour
change somewhere in the 31 `src/` commits between the two bases, not a
measurement artefact. It is being bisected (WarmupHMC todo
`2026-09-30T23-53-01-378-185dkdv`); until that settles, read this row as the
sampler's current behaviour, not as the method's ceiling.

Read as one sentence: **where none of the fixed options is good, adaptive beats
all of them; where a good noncentered parametrization exists, adaptive reaches
it on the funnel and falls short on `eight_schools` and `seeds`.** Both radon
models are the first case — centered, noncentered and posteriordb's own
hand-written noncentered model all land within seed noise of each other around
15–21, and adaptive gets 79–83 by settling at an *interior* `c ≈ 0.4–0.5` that no
hand-written model offers. That interior optimum is the strongest result here,
and it is stronger on this base than on `d68d680` (62.6–68.1 then).

**The verdict is backend-independent, and that is measured rather than
assumed.** Every `plain`, `fixed_centered` and hand-written arm is identical
across the two backend runs — same gradient counts on every seed: `plain` and
the hand-written model never touch the wrapper, and `fixed_centered`'s exact
identity transform differentiates to bit-identical gradients under both
backends. Of the five `adaptive`
arms, four (both radon models, `seeds`, the funnel) are identical on all eight
seeds; `eight_schools` is bit-identical on four seeds and moves −0.7% in median
gradient count. Final `c` vectors agree between backends on every seed of every target
except two `eight_schools` seeds. All 368 runs completed. So the backend can
change *which trajectory* a seed happens to take; it does not change what the
method learns.

### It converts into wall-clock on four targets

Under Enzyme, adaptive reparametrization is faster **in seconds**, not just per
gradient, on four of the five targets, and break-even on `seeds`:

| target | adaptive vs plain, Enzyme/Const | | | ForwardDiff |
|---|---|---|---|---|
| | boxed base `b5c7dee` | `d68d680` | **this base** | this base |
| funnel (synthetic) | 17.9× faster | 19.3× faster | **22.2× faster** | 34.5× faster |
| eight_schools centered | 21.0× faster | 18.9× faster | **10.5× faster** | 6.3× faster |
| radon partially_pooled | 3.5× **slower** | 2.1× faster | **3.2× faster** | 2.4× faster |
| radon variable_intercept | 2.3× **slower** | 2.0× faster | **2.7× faster** | 2.1× faster |
| seeds centered | 6.1× **slower** | 1.6× faster | **0.95× — break-even** | 2.0× **slower** |

(min ESS/sec, median over 8 seeds, sequential runs.)

Between the boxed base and `d68d680`, three losses became wins, and the three
that moved are precisely the three whose specs were boxed; `funnel` and
`eight_schools` were never boxed and stayed in the high band. Between `d68d680`
and this base the moves have sampling causes, not cost causes: `seeds` fell back
to break-even because its adaptive arm mostly stopped adapting (the Verdict's ‡),
and `eight_schools` roughly halved because its *plain* arm doubled per gradient
(1.55 → 3.01) while adaptive held, so the denominator moved, not the method.

**How much of that last digit is real: measured, not estimated.** When
`d68d680` was the base, the same matrix was run on two bases, `5637fcf` (run A)
and `d68d680` (run B), whose `src/` is code-identical. The two runs are
**bit-identical on all 184 runs in every field except wall-clock** — same trajectories, same gradient counts, same learned `c`,
under both backends. So ratios computed from them differ only by machine
conditions, which makes the spread between them a direct read of the noise floor
on this column:

| target | Enzyme | | ForwardDiff | |
|---|---|---|---|---|
| | run A | run B | run A | run B |
| funnel | 17.0× | 19.3× | 16.5× | 19.1× |
| eight_schools centered | 16.2× | 18.9× | 15.6× | 15.0× |
| radon partially_pooled | 2.0× | 2.1× | 1.7× | 1.6× |
| radon variable_intercept | 1.8× | 2.0× | 1.6× | 1.5× |
| seeds centered | 1.1× | 1.6× | 0.9× | 1.1× |

Ten cells, identical trajectories, **1.7% to 47% apart** — and `seeds` under
ForwardDiff **changed sign**, from a 11% loss to a 10% win. Nothing about the
method changed between those two columns; only the machine did.

**This base carries its own control, and the floor is higher.** `plain` and the
hand-written model never touch the wrapper, so their
trajectories are bit-identical between this base's Enzyme and ForwardDiff runs —
same gradient counts on every seed — and only the clock differs. Their median
ESS/sec still moved **3% to 58%** between two runs made back to back (`plain`: 4%
`eight_schools`, 3% and 18% on the two radon models, 15% `seeds`, 58% `funnel`;
hand-written: 16–35%), on a host carrying other agents' jobs. That is also why
the ForwardDiff column reads the funnel *higher* than Enzyme's (34.5× against
22.2×): the adaptive arms took the same trajectories, and the plain denominator
it is divided by moved 58%.

Read the consequences in the strong direction, not the flattering one:

- **The large ratios are robust.** `funnel` and `eight_schools` are 6–35× on
  every run and every backend. A 58% wobble on a 20× win changes nothing.
- **`seeds` is not a win.** 0.95× under Enzyme and 0.51× under ForwardDiff on
  this base, 0.9–1.6× across the four July measurements. The honest entry is "no
  measured gain under Enzyme"; under ForwardDiff it is a loss, because the
  adaptive arm pays the wrapper's +148% gradient overhead (below) without, on six
  of eight seeds, adapting at all.
- **Both radon models survive.** 2.1–3.2× across both backends on this base,
  1.5–2.1× across July's four, never crossing parity. A real win, whose *size* is
  not resolved beyond "two to three times".

The per-gradient columns carry none of this uncertainty — they are bit-identical
across all four runs, because they are counts rather than clocks. **Where a
per-gradient number and a wall-clock number disagree here, trust the
per-gradient one**, and treat ESS/sec as the coarse confirmation that the
per-gradient win is not being eaten by transform overhead.

So the claim the package can now support:

> Starting from a centered parametrization, adaptive partial centering beats
> every fixed parametrization per gradient evaluation on the two targets where no
> fixed option is good (both radon models, ~4×), matches the best one on the
> funnel, and trails the noncentered parametrizations on `eight_schools` (−20%)
> and `seeds` (−37%, where it mostly does not adapt on this base — under
> investigation). Under Enzyme that converts into wall-clock wins of 2.7× to 22×
> on four targets and break-even on `seeds`.

**The overhead is still not the transform's arithmetic.** A wrapper configured
as an exact identity costs the same as a live one, under both backends and on
every target — on this base, Enzyme ×1.07 no-op vs ×1.08 live on
`radon_partially_pooled`, ×1.06 vs ×1.07 on `radon_variable_intercept`. What
remains is the cost of carrying the AD re-differentiation of
`ljac_(x_) + dot(g_y, y_)` at all. Under Enzyme that is now **+7% of a gradient
on `radon_variable_intercept`, +8% on `radon_partially_pooled` and +25% on
`seeds`** — small enough to be paid out of the sampling gain wherever there is
one, which is what the wall-clock table above shows. It is +70% on
`eight_schools` and 6.8× (+578%) on the funnel, but those are the two targets
whose *bare* gradient costs 479 ns and 46 ns, where a roughly fixed per-call AD
cost has to dominate; both still come out 10–22× ahead overall because the
sampling gain there is enormous. Under ForwardDiff the same three overheads are
+67%, +68% and +148%, and the two small targets go to +243% and 18.5× (+1749%) —
the backend choice is most of what makes this affordable.

Every one of those figures is a ratio of two medians, which is exactly the shape
that produced a reversed verdict elsewhere on this page, so the direction is
checked against the individual rounds rather than assumed from the summary. The
rounds pair: `gradient_overhead` times all three variants on the *same* `xs`
within each round, rotating the order, so round `r`'s live and bare timings are
adjacent in time on identical inputs. Pairing them that way, **68 of the 70
paired rounds** across both backends and all five targets put the wrapper above
the bare gradient. The two exceptions are one `radon_partially_pooled` round
under Enzyme at `0.92×`, against six others from `1.04×` to `1.13×`, and one
`seeds` round under ForwardDiff at `0.40×`, against six others from `1.60×` to
`4.93×` — single anomalous rounds on a contended host, not evidence the wrapper
is ever free.
So the *sizes* above are medians and carry the usual round-to-round spread, but
the *sign* does not depend on them.

## Which AD backend

The differentiated objective `x -> ljac(x) + dot(g_y, y(x))` is scalar in the
full parameter vector with the inner gradient `g_y` frozen, so it is the
textbook reverse-mode shape: forward mode costs `ceil(d / chunksize)` tangent
sweeps of the transform per gradient where reverse mode costs one. The
prediction that follows is that reverse mode should win, and win *hardest* at
large `d`.

**Measured on this base, the direction holds on every target, in all but one
comparison.** Per wrapped `logdensity_and_gradient` call, ratio of
Enzyme/`Const` to ForwardDiff — below 1.0 means Enzyme is faster:

| target | `d` | Enzyme ÷ ForwardDiff | reading |
|---|---|---|---|
| `radon_partially_pooled` | 88 | **0.46–0.63×** | Enzyme 1.6–2.2× faster |
| `radon_variable_intercept` | 89 | **0.43–1.08×** | Enzyme 0.9–2.3× faster |
| `seeds` | 26 | **0.12–0.61×** | Enzyme 1.6–8.2× faster |
| `eight_schools` | 10 | **0.34–0.59×** | Enzyme 1.7–3.0× faster |
| `funnel` | 10 | **0.13–0.37×** | Enzyme 2.7–8.0× faster |

**This table is not hand-copied — run `docs/benchmark/backend_bands.jl` and
paste what it prints.** It reads the same four harnesses, takes the live set
from `artifact_currency.jl`'s own supersession list (so the boxed-spec run dirs,
where Enzyme measured *slower*, cannot silently widen a band), and errors rather
than narrowing if a source is missing. The reason it exists is two paragraphs of
this document's own history: a band was widened four minutes before the files it
was computed from were regenerated, and the result excluded values that were in
the files while including values that were in none of them. That is not a
mistake anyone can see in a diff — the band is prose and the evidence is JSON.

Every range spans **four independent harnesses**, each writing its own JSON:
`replicate_backends.jl` (7 rounds × 1000 calls, backend order rotated per round,
both a centering endpoint and an interior `c = 0.5`), `typical_positions.jl`
(5 rounds, at `randn` *and* at positions the sampler actually visited),
`annotation_sweep.jl` (2000 calls, both endpoints), and the driver's own
`gradient_overhead` block (7 rounds × 2000 calls, median). **Enzyme is faster in
34 of the 35 comparisons those four harnesses produce.** The exception is
`radon_variable_intercept` in `annotation_sweep` at `c = 0`, 1.077× — one
2000-call median on a contended host, in the harness whose repeat run flipped
direction in 8 of 10 cells (§ *`function_annotation`*), against 0.43–0.93× from
the other three harnesses on the same target. An earlier draft of this revision
read "32 of 35, three reversals, all in the driver" — those three came from a
`forwarddiff-be6fb23/` directory that held an Enzyme run (see the top of this
page); with the genuine ForwardDiff run the driver reverses nothing.

This reverses what every earlier revision of this document reported, where
Enzyme measured 1.2–5.4× *slower* on the three larger targets. Nothing about the
backends changed; the specs they were differentiating did. The next section is
that story, and it is kept because it is also the reason to distrust a backend
number measured through a spec you have not inspected.

**What is still not established is the second half of the prediction — that the
gap should widen with `d`.** It does not, visibly: the widest margins are on
`funnel` at `d = 10` and `seeds` at `d = 26`, the narrowest on
`radon_variable_intercept` at `d = 89`. But the spread *within* a single target
across harnesses is as large as the spread *between* targets — `seeds` alone spans
0.12 to 0.61, which covers nearly the entire between-target range on its own — so
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
the order rotated, against a bare gradient of 72382 ns:

| spec | ForwardDiff | Enzyme/`Const` | Enzyme ÷ ForwardDiff |
|---|---|---|---|
| boxed | 779220 ns | 1338543 ns | 1.72× — Enzyme *slower* |
| unboxed (**as shipped today**) | 156878 ns | **63381 ns** | **0.4× — Enzyme faster** |
| de-boxing speedup | 4.97× | **21.12×** | |

Both backends are hurt by the box; Enzyme is hurt ~4.2× harder, and that alone
**inverts which backend looks faster**. Two numbers in one table that disagree in
direction, from one process, minutes apart, on specs that produce identical
gradients: no backend verdict measured through an uninspected spec means
anything.

Note what the unboxed Enzyme figure implies for the wrapper as a whole: 63381 ns
against a bare gradient of 72382 ns is **-12% overhead** — the wrapped call timed
12% *faster* than bare, which is repeat noise around zero rather than a speedup
(every absolute timing in this run is ~2–3× its checked-in predecessor while the
ratios hold; the annotation rerun below shows what this box does to medians).
Measured the same way — against that same bare gradient — the boxed spec was a
**18× tax** (1338543 ns).
So on this target the transform went from dominating the gradient to nearly free
under reverse mode, and de-boxing is the whole of that change.

Both figures in that sentence are ratios to the *bare gradient*, and they have to
be: the 21.12× in the table above is boxed-against-unboxed, a different
denominator, and quoting it here instead would understate the tax by a factor
that happens to look plausible. This paragraph previously did exactly that.

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
  `fixed_noncentered` arm itself — **changes no verdict: Enzyme is faster at
  both position sets on all five targets** (`randn` → typical:
  `radon_partially_pooled` 0.61× → 0.51×; `radon_variable_intercept` 0.93× →
  0.91×; `seeds` 0.47× → 0.56×; `eight_schools` 0.57× → 0.42×; `funnel` 0.25× →
  0.20×). Recorded in `results/typical_positions.json`.

  **No single target's shift is *stably* separable from repeat-to-repeat noise,
  and the file now carries what proves it.** Each median is over `ROUNDS` timing
  rounds and those rounds are persisted raw, so three independent statistics can be
  computed by a reader rather than taken on trust: the per-round shift ranges
  straddle zero on four targets out of five — the exception is `eight_schools`,
  whose five rounds read −29.4% to −5.0% with `randn` and typical ranges disjoint
  (last run no target separated, so separation itself flips run to run); the
  ratio spread runs 15.5% to 856.4% of its own median; and median-to-median
  across two independent runs of *identical code* moves as much as **71.1%**
  (`funnel` 0.379 → 0.649), against a smallest-distance-from-1.0× of 7% for any
  median in the run. A 71% mover cannot resolve a 12-point shift. An earlier
  revision of this bullet read the funnel's swing in the *opposite direction*
  (0.70× at `randn` vs 0.38× typical, where this run has 0.25× vs 0.20×) and
  explained it with a story about a noisy
  `bare` column — a sign flip is what a noise column looks like when each cell is
  quoted once and there is nothing checked in to contradict it.
- **DI preparation.** The hot path calls `value_and_gradient` with **no prep
  object** (`src/Reparametrizations.jl:148`), so DifferentiationInterface
  re-prepares on every gradient evaluation. Now that the call itself is cheap,
  `prepare_gradient` alone accounts for **55–206% of the whole unprepped Enzyme
  call** on the three larger targets (28.3 of 51.6 µs on `radon_partially_pooled`,
  165.1 of 80.1 µs on `radon_variable_intercept`, 7.5 of 3.8 µs on `seeds`) —
  against 12–93% under ForwardDiff on the same three. That reads like an obvious
  speedup and **it is not one**: reusing a prep object is worse or flat on
  **every** Enzyme target — +15% on `radon_partially_pooled`, +6% on
  `radon_variable_intercept`, +286% on `seeds`, +21% on `funnel`, −6% on
  `eight_schools`. (A repeat run of the identical script read +10% on `seeds`
  and −10% on `radon_variable_intercept` — these reuse deltas move run to run
  on shared hardware, so only the direction, never a point value, is the claim.)
  A share above 100% cannot be a decomposition at all — `prepare_gradient` alone
  cannot cost more than the call that contains it — so on two of the three
  targets these shares measure contention between separately timed blocks, not
  prep's weight; only the `radon_partially_pooled` figure (55%) is even a
  candidate reading. Either way the costs do not decompose additively under
  Enzyme, so no share here is time that can be removed. Under ForwardDiff reuse *is* worth
  something — 56% and 36% on the two radon targets, 49% and 85% on the two small
  ones — but ForwardDiff is the slower backend to begin with, so the
  prepped-ForwardDiff figure (146.5 µs on `radon_partially_pooled`) is still 2.8×
  the unprepped-Enzyme one (51.6 µs). Reuse also returned gradients identical to
  the unprepped path (`|Δg| = 0.0`) in all ten configurations, which is worth
  knowing but is **not** a licence to reuse in the sampler: the objective closes
  over `g_y`, which changes every call, and this probe holds it fixed. Recorded
  in `results/prep_cost.json`.

  **These are medians over five rounds, which is a change: each cell used to be
  one unreplicated timed loop, and the shares above were quoted from it to three
  significant figures.** Two independent runs of the identical script disagreed
  by more than a factor of thirty on one cell — `eight_schools`/ForwardDiff read
  113% of the call, then 3629% — and half the ten cells moved by more than 2×.
  With rounds, the outlier is visible for what it is: exactly one round per cell
  spikes (up to 4105% of the call), the other four agree closely, and the medians
  above are stable. Note that a share *above* 100% is not automatically an error
  — `value_and_gradient` with no prep object may take a lighter path than
  `prepare_gradient` builds, so prep genuinely can cost more than the call it is
  nominally part of. The raw rounds are checked in, which is what lets a reader
  tell that case from a timer artefact.

### `function_annotation` is required, and how much it costs is target-dependent

A bare `AutoEnzyme()` **used to fail outright** on this objective with
`EnzymeMutabilityException` — the objective is a closure capturing the
reparametrizer and the frozen `g_y`. `1a395ce` ("make a bare `AutoEnzyme()` work
at both AD sites") removed that failure, so the live question is only what the
wrong annotation costs. Enzyme's own error text suggests `Duplicated`; `Const`
is the correct annotation here, since the closure is not something we
differentiate with respect to.

The cost of getting that wrong **has collapsed to noise around unity** — the
old "never free / fixed per-call cost" model does not reproduce on this tree.
Both centerings, `annotation_sweep.jl`:

| target | `d` | `Duplicated` ÷ `Const` | absolute penalty |
|---|---|---|---|
| `funnel` | 10 | 1.2–6.2× | ~0–2 µs |
| `eight_schools` | 10 | 1.2–2.7× | ~0–2 µs |
| `seeds` | 26 | 1.4–1.7× | ~3–7 µs |
| `radon_partially_pooled` | 88 | 1.1–1.3× | ~7–16 µs |
| `radon_variable_intercept` | 89 | 0.7–1.0× | −74 to −14 µs (`Duplicated` faster) |

`radon_variable_intercept` reads `Duplicated` *faster* than `Const`, and a repeat
run of the identical script flips the direction of 8 of the 10 cells — the gap
the old table measured (1.7–27.4×, "never free") is gone, whether by Enzyme
improvements or the reparametrizer rewrites since is not separated here. Both
gradients are equally correct (they agree to ≤ 6.8e-13, unchanged).

**The previous revision of this table said `Duplicated` was free on the three
larger targets (0.96–1.06×), and it flagged the reason to distrust that: those
were the three boxed specs, where a ~15× defect swallowed a 40 µs shadow copy.**
That caveat was right. Unboxed, the same three targets show a 1.7–5.0× penalty.
It is a useful calibration on how much a confounded measurement can hide: not a
few percent, but the entire effect.

So the statement that now survives, in two parts. *Some* annotation was
mandatory when a bare `AutoEnzyme()` did not run at all; since `1a395ce` that
failure is gone. And of the two candidates, `Const` is the semantically correct
one — but the performance gap that used to separate it from `Duplicated`
(1.7–27.4×, every target) has collapsed to run-to-run noise around unity.
Pick `Const` for correctness, not speed.

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
> ≥2% in median gradient count between backends (ForwardDiff relative to
> Enzyme) — `seeds`/fixed_noncentered −5.7%, `eight_schools`/fixed_noncentered
> +5.0%, `radon_variable_intercept`/fixed_noncentered −2.9%,
> `radon_partially_pooled`/fixed_noncentered +2.0% — and `eight_schools`/adaptive
> differs on four of eight seeds (−0.7% median gradient count). Every other arm is identical on
> every seed. `plain` and the hand-written model never touch the wrapper;
> `fixed_centered` does run its AD, but an exact identity transform differentiates
> to bit-identical gradients under both backends. The cause of the drift is the
> mechanism described under [Harness correctness](#harness-correctness): the
> backends' gradients through a non-trivial transform differ at the 1e-13 level
> and NUTS amplifies that into a different trajectory.
>
> **One verdict row depends on it, through its comparator.** Every adaptive
> median in the verdict is within 0.01 between backends, but `seeds` is compared
> against its `fixed_noncentered` row, which reads 15.83 under ForwardDiff
> against Enzyme's 12.06 — so `seeds`' gap is −37% on one trajectory set and −52%
> on the other; its sign does not move. The `fixed_noncentered` rows are never
> used as a cross-backend wall-clock comparison.

<!-- The TABLES below were generated by docs/benchmark/summarize.jl from
     results/enzyme-d68d680/runs.json. The PROSE between them was not, and no
     generator can reproduce it: hand-written analysis, cautions, corrected
     target descriptions and integer formatting.

     So do NOT regenerate this section by replacing it with summarize.jl output.
     Measured 2026-07-28: the generator emits 113 lines, this block is ~494 --
     a wholesale paste silently deletes ~380 lines of interpretation and looks
     in the diff like an ordinary re-measurement.

     To update after a re-run, diff generator output against generator output,
     never against this file:

         summarize.jl <old>/runs.json > /tmp/old.md
         summarize.jl <new>/runs.json > /tmp/new.md
         diff /tmp/old.md /tmp/new.md

     then apply those cells here by hand. On the d68d680 re-measurement that
     diff was ESS/sec and gradient-overhead only -- every other column is a
     count, not a clock, and was bit-identical. -->


### `eight_schools-eight_schools_centered`

*8 group effects — dimension 10. The textbook hierarchical funnel.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 58.6 <sub>5.7–143.7</sub> | 830.3 <sub>116.1–2522.6</sub> | 3.01 | 18592 | 6.0 | — |
| fixed c = centered | 63.8 <sub>4.8–87.2</sub> | 1001.8 <sub>102.0–1713.1</sub> | 2.91 | 20214 | 7.5 | 1.0 |
| **adaptive** (starts centered) | 517.5 <sub>394.4–654.7</sub> | 8693.9 <sub>4898.6–11895.0</sub> | 48.29 | 10304 | 0.0 | 0.0 |
| fixed c = noncentered | 462.3 <sub>341.2–662.9</sub> | 17120.9 <sub>512.4–26399.1</sub> | 55.14 | 8426 | 0.0 | 0.0 |
| hand-written noncentered model | 536.9 <sub>301.3–620.9</sub> | 17162.1 <sub>4952.5–29790.3</sub> | 60.11 | 8607 | 0.0 | — |

Min ESS over the 10 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 58.6 | 3.15 |
| fixed c = centered | 63.8 | 3.15 |
| **adaptive** (starts centered) | 517.5 | 50.22 |
| fixed c = noncentered | 462.3 | 54.86 |
| hand-written noncentered model | 536.9 | 62.38 |

Adaptive drives `c` to the noncentered endpoint: **59 of the 64 learned values
(8 seeds × 8 coordinates) are exactly 0.0 and the other 5 are 0.1** — the known
right answer for this model, found without being told (53 and 11 at `d68d680`).
Finding the endpoint no longer means matching it, though: adaptive reads 48.29
per 1k gradients against 55.14 for the noncentered endpoint reached by transform
and 60.11 for posteriordb's hand-written model — 12% and 20% behind, and 20%
behind on the matched parameters too (50.22 vs 62.38). It spends 10304 gradient
evaluations against the endpoint's 8426 (more on 6 of 8 seeds) — plausibly the
warm-up spent at `c = 1` before switching, though nothing here isolates that.
Divergences go
from a median of 6 on plain to 0.

Two cautions on this target specifically. Its adaptive spread is wide
(394.4–654.7 min ESS over 8 seeds), so the 12% gap to the noncentered endpoint
is not resolvable at 8 seeds; the 20% gap to the hand-written model is larger
than at `d68d680` (+2% to −13% then), and the per-gradient trajectories behind
it are bit-reproducible. The ForwardDiff run reads 48.28 / 57.10 / 60.11 for the
same three arms, so the ordering does not depend on the backend. The safe
reading is **finds the right endpoint, trails the fixed parametrizations that
start there by 10–20%**.

This is also the target where the pre-fix run over-stated the method: on
`05aed41` adaptive appeared to *beat* the hand-written model by 41%. It does not.

### `radon_mn-radon_partially_pooled_centered`

*85 counties — dimension 88.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 226.1 <sub>175.8–288.7</sub> | 388.3 <sub>271.4–616.0</sub> | 15.16 | 16050 | 0.0 | — |
| fixed c = centered | 257.5 <sub>140.6–335.1</sub> | 379.8 <sub>156.1–421.1</sub> | 17.58 | 16106 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 868.1 <sub>587.4–1388.8</sub> | 1253.1 <sub>588.4–1620.7</sub> | 82.53 | 11542 | 0.0 | 0.4 |
| fixed c = noncentered | 256.0 <sub>171.0–315.6</sub> | 437.6 <sub>367.8–616.7</sub> | 17.52 | 16066 | 0.0 | 0.0 |
| hand-written noncentered model | 297.5 <sub>164.4–473.6</sub> | 442.5 <sub>216.3–747.9</sub> | 20.33 | 15928 | 0.0 | — |

Min ESS over the 88 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 226.1 | 14.08 |
| fixed c = centered | 257.5 | 15.99 |
| **adaptive** (starts centered) | 868.1 | 75.21 |
| fixed c = noncentered | 256.0 | 15.94 |
| hand-written noncentered model | 297.5 | 18.68 |

**The headline case for the method.** No fixed parametrization helps: centered
(17.58), noncentered (17.52) and posteriordb's hand-written noncentered model
(20.33) are all in the same band. Adaptive gets 82.53 — **4.1× the best fixed
option** — using 28% fewer gradient evaluations than the noncentered endpoint,
and it is also the fastest arm in **wall-clock**, at 1253.1 ESS/sec against the
plain sampler's 388.3. The `final source c` column reports 0.4, but that is
a **median over 85 counties, not a setting**: see below.

#### What adaptation actually learns is a `c` *per coordinate*

The `final source c` column is a median, and on the two radon models it hides
the result. Every one of the 85 counties gets its own centering, and they do not
agree with each other. Distribution of the learned `c` over all 8 seeds ×
coordinates, Enzyme run (the ForwardDiff run is identical except on two
`eight_schools` seeds):

| target | n | median | quartiles | at `c = 0` | at `c = 1` |
|---|---|---|---|---|---|
| `radon_partially_pooled` | 680 | 0.40 | 0.30 – 0.60 | 1% | 1% |
| `radon_variable_intercept` | 680 | 0.50 | 0.40 – 0.70 | 1% | 1% |
| `seeds` | 168 | 1.00 | 0.93 – 1.00 | 1% | 75% |
| `eight_schools` | 64 | 0.00 | 0.00 – 0.00 | 92% | 0% |
| `funnel` | 72 | 0.00 | 0.00 – 0.00 | 100% | 0% |

On the radon models **98% of coordinates end strictly between the two
endpoints**, spread across the whole interval rather than clustered at the
median. That is the thing no fixed parametrization can express: centered,
noncentered and posteriordb's hand-written noncentered model each impose *one*
`c` on all 85 counties, so none of them can represent any of these fits — which
is the mechanism behind the 4.1× and 3.8× margins, and why those margins appear
on exactly the two targets where the spread is widest. The `seeds` row is the
exception this base introduced: at `d68d680` it read median 0.30, quartiles
0.20–0.50, 0% at `c = 1`; now three quarters of its coordinates never move (see
its section).

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
| plain (no wrapper) | 274.6 <sub>243.0–498.9</sub> | 367.7 <sub>238.2–562.7</sub> | 19.14 | 14989 | 0.0 | — |
| fixed c = centered | 283.0 <sub>209.0–366.4</sub> | 381.5 <sub>297.9–438.5</sub> | 21.00 | 14366 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 745.1 <sub>631.5–882.4</sub> | 983.0 <sub>799.8–1159.9</sub> | 79.19 | 9002 | 0.0 | 0.5 |
| fixed c = noncentered | 331.4 <sub>249.3–456.3</sub> | 318.2 <sub>257.0–453.9</sub> | 20.40 | 16434 | 0.0 | 0.0 |
| hand-written noncentered model | 284.1 <sub>175.0–465.2</sub> | 314.3 <sub>188.4–480.6</sub> | 17.62 | 16916 | 0.0 | — |

Min ESS over the 89 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 274.6 | 18.32 |
| fixed c = centered | 283.0 | 19.70 |
| **adaptive** (starts centered) | 745.1 | 82.77 |
| fixed c = noncentered | 331.4 | 20.17 |
| hand-written noncentered model | 284.1 | 16.79 |

Same shape as its sibling, and the same per-coordinate story — 98% of the 85
county centerings land strictly inside `(0, 1)` with quartiles 0.4–0.7, so the
reported `c = 0.5` is again a median over coordinates. Worth 3.8× the best fixed
alternative on gradients (here the centered endpoint, 21.00) and 2.7× the plain
sampler in wall-clock. The `fixed_noncentered` row is one of the arms that
drifted between backends (−2.9% gradients); the ForwardDiff run puts it at 24.90,
which would make it the best fixed option and the margin 3.2× instead. The
adaptive and plain rows are identical across backends.

### `seeds_data-seeds_centered_model`

*21 plates — dimension 26. posteriordb ships no noncentered sibling.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 80.1 <sub>41.2–176.7</sub> | 848.2 <sub>358.2–1947.7</sub> | 5.58 | 15973 | 0.0 | — |
| fixed c = centered | 80.1 <sub>41.2–176.7</sub> | 704.0 <sub>330.8–1332.4</sub> | 5.58 | 15973 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 119.2 <sub>41.2–215.9</sub> | 806.1 <sub>112.1–1493.3</sub> | 7.61 | 15973 | 0.0 | 1.0 |
| fixed c = noncentered | 214.3 <sub>94.1–338.1</sub> | 1837.9 <sub>915.2–3103.2</sub> | 12.06 | 17537 | 0.0 | 0.0 |

Min ESS over the 47 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 80.1 | 5.01 |
| fixed c = centered | 80.1 | 5.01 |
| **adaptive** (starts centered) | 119.2 | 7.46 |
| fixed c = noncentered | 214.3 | 12.22 |

**The target this base changed.** At `d68d680` adaptive beat the noncentered
endpoint here (16.92 vs 11.28 per 1k gradients) and every seed settled at an
interior per-plate optimum, quartiles 0.2–0.5. On this base **six of the eight
seeds never leave the start**: every one of their 21 learned `c` values is
exactly 1.0, and the arm's median gradient count is `plain`'s, 15973. Only seeds
7 and 8 adapt, to the same interior band as before. The arm still edges plain
(7.61 vs 5.58 per 1k gradients) but trails the noncentered endpoint by 37%
(12.06; 52% against ForwardDiff's 15.83 for that drifting arm), and the
`final source c` column now reads 1.0.

These columns are counts, bit-reproducible on a given `src/`, so this is a
behaviour change, not seed noise. It is under bisection over the 31 `src/`
commits between the two bases (WarmupHMC todo
`2026-09-30T23-53-01-378-185dkdv`); the numbers above describe the sampler as it
ships today.

Wall-clock follows: adaptive is break-even with plain under Enzyme (806.1 vs
848.2 ESS/sec, 0.95×) and 2.0× *slower* under ForwardDiff (374.2 vs 736.8), where
the non-adapting seeds pay the wrapper's +148% gradient overhead for nothing.

### `funnel` (synthetic)

*Neal's funnel, `v ~ Normal(0, 3)`, `theta_i ~ Normal(0, exp(v/2))`, K = 9 —
dimension 10. Not a posteriordb posterior; posteriordb ships none. Defined in
`common.jl` with an analytic gradient. It is the extreme case — the centered
parametrization is pathological and the noncentered one is exact — so it bounds
what partial centering can possibly buy. Reported alongside the real targets,
never as a headline.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 28.8 <sub>7.6–42.9</sub> | 583.3 <sub>242.6–1646.3</sub> | 0.88 | 26630 | 1.5 | — |
| fixed c = centered | 28.8 <sub>7.6–42.9</sub> | 453.2 <sub>195.2–1095.0</sub> | 0.88 | 26630 | 1.5 | 1.0 |
| **adaptive** (starts centered) | 886.2 <sub>780.5–929.1</sub> | 12965.8 <sub>5216.2–17776.7</sub> | 111.96 | 7930 | 0.0 | 0.0 |
| fixed c = noncentered | 893.0 <sub>805.1–964.8</sub> | 33175.9 <sub>20413.3–39947.1</sub> | 111.24 | 7900 | 0.0 | 0.0 |

Adaptive now **ties** the exact answer on gradient efficiency: 111.96 vs 111.24
per 1000 gradients. Noncentered is *exactly* right here, so there is nothing to
discover and the best adaptation can do is reach it; at `d68d680` it fell 13%
short (93.17 vs 107.05). Against the parametrization a user would actually have
written it is **127× better** (111.96 vs 0.88), so this row bounds the loss, not
the gain.

The search is nearly free in gradients — 7930 against the endpoint's 7900, 0.4%
more — and min ESS is level too, 886.2 vs 893.0 on overlapping seed ranges.
Adaptation converges to the right fixed answer (all 72 learned values exactly
0.0). What it does not recover is wall-clock against the endpoint (12965.8 vs
33175.9 ESS/sec): on a 46 ns gradient the wrapper's per-call AD cost is 6.8× the
gradient itself (§ *Cost on the gradient hot path*), and that is paid on every
call whether or not `c` is still moving.

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
| `radon_variable_intercept` | 49538 | 82518 | ×1.67 | 43460 | 46718 | **×1.07** |
| `radon_partially_pooled` | 27235 | 45780 | ×1.68 | 26652 | 28735 | **×1.08** |
| `seeds` | 3464 | 8598 | ×2.48 | 2853 | 3567 | **×1.25** |
| `eight_schools` | 476 | 1634 | ×3.43 | 479 | 811 | **×1.70** |
| `funnel` | 47 | 877 | ×18.49 | 46 | 313 | **×6.78** |

(`bare` is given for each run separately; the two agree to 1–3% on three targets
and differ by 14% on `radon_variable_intercept` and 21% on `seeds` — the same
call, the same inputs, two runs minutes apart on a contended host. The multiples
are taken within one run, round by round, so they carry less of that.)

**The multiple tracks how expensive the underlying density is, not `d`.** The
two most expensive densities carry the wrapper for +7% and +8% under Enzyme; the
funnel, whose bare gradient is 46 ns of analytic arithmetic, pays 6.8×. This is what a
roughly fixed per-call AD cost looks like divided by a varying denominator, and
it is why the funnel is reported as a bound rather than as a headline.

**The no-op and live configurations cost the same, in both backend runs.** An
identity reparametrization — source `c` equal to target `c`, so the transform
provably does nothing — pays the same multiple as a live one: within ±6% on all
ten rows, with no consistent sign (four of the ten rows put the *no-op* higher,
which the live transform doing strictly more work cannot produce, so ±6% is the
probe's floor rather than a measured effect).

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
> not the absolute value — which also fell because of the de-boxing. On the
> boxed base the probe said 214698 ns where the sweeps said ~375000; on `d68d680`
> it said 77252 where `replicate_backends.jl` said 76072–83671. The outlier was
> the harness, and the sweeps were right. **On this base the absolute check no
> longer holds, and the reason is the host, not the harness:** the driver says
> 82518 ns, while `replicate_backends.jl` — re-measured the same day during
> heavier contention — has medians of 171194 and 223959 at its two centerings.
> Its Enzyme÷ForwardDiff ratio on this target, 0.43–0.44, is still in line with
> the driver's 0.57 and the other harnesses' 0.43–0.93. Read absolute
> nanoseconds on this page as same-run comparisons only.

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
  inflates the leg sds to 66–301. **Numbers in `results/before`, `results/after`,
  `results/forwarddiff-b5c7dee` and `results/enzyme-b5c7dee` were all measured on
  bases predating `b109210`, where the compensation was correct.** Anything
  measured after it must not carry one.
- **The no-op arm really is a no-op.** `plain` and `fixed c = centered` are
  mathematically the same sampler, and agree to the last gradient evaluation on
  every seed for `seeds_data` (80.1 min ESS, 15973 grads, both arms) and the
  funnel (28.8, 26630) — while differing on eight_schools and both radon models.
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
  | `plain`, `hand-written noncentered model` | 40/40, 24/24 — no wrapper, so no AD path to differ on |
  | `fixed c = centered` | **40/40** |
  | `adaptive` | 36/40 — 8/8 on both radon models, `seeds` and the funnel, **4/8 on `eight_schools`** |
  | `fixed c = noncentered` | 8/40 — all eight on the funnel, 0/32 on the posteriordb targets |

  The clean split is `fixed c = centered` (source `c` equal to target `c`, so
  the transform is an identity) reproducing exactly on every run, against
  `fixed c = noncentered` (a live transform) reproducing on almost none. The
  wrapped arms that differ are differing through the very AD path this document
  measures, which is why an individual ESS/sec figure is reproducible only
  against a fixed backend — and why the verdict is stated in ESS per gradient,
  where 36 of 40 adaptive runs are bit-identical between backends and the rest
  are named above. (`seeds`' 8/8 is partly the regression the verdict flags: six
  of its adaptive runs never leave the identity transform.)

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
  advantage is 0.9–8.2× across all five targets with no visible trend in `d`,
  and the spread *within* one target across harnesses is as wide as the spread
  *between* targets, which is the reason: the noise floor is larger than any
  slope five points at three distinct dimensions could show. See § *Which AD
  backend*. The direction replicates 34/35; the slope is unmeasured, in either
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

  **`src/Reparametrizations.jl`'s docstring backend table is gone — retracted in
  `a03a271`, with the guidance inverted in `acaca4b` — so the four staleness
  notes below are kept as the evidence record, re-measured, not as fix requests.**
  At the time they indicted the *"That argument is an operation count, and how it
  scales here is untested"* admonition; they are recorded here because this is where
  the evidence is.

  1. **Its backend table carried two rows and could have carried five.** It carried
     only `funnel` and `eight_schools` because, in its words, the three larger
     targets "were measured against reparametrization specs whose accessor
     closures captured a `Core.Box`". They were re-measured on the de-boxed
     specs then; the five-row table is § *Which AD backend*, and
     `results/backend_replication.json` is the harness the docstring's own rows
     came from. Its two bands at the time — `funnel` 0.31–0.67×, `eight_schools`
     0.43–0.66× — had been widened by `62ba171` after `WarmupHMC:reparam-docs` found
     the narrower predecessors were not re-derivable from anything checked in.
     That widening was honest against the files it was computed from, and it
     landed **four minutes before** the regeneration in `176a061` moved those
     files underneath it, leaving a band that excluded values present in the
     files and included values present in none of them. **A band derived from
     checked-in JSON is only a floor until someone regenerates the JSON** — the
     same coupling that broke `docs/src/reparametrization.md` in the same window,
     and the reason this table is now generated rather than hand-copied.
     `docs/benchmark/backend_bands.jl` is that generator; across the four live
     harnesses it currently gives `funnel` 0.13–0.37× and `eight_schools`
     0.34–0.59×.

     That is a band over per-harness medians. From the single harness the
     docstring named, over 14 rounds each (7 per centering), the per-round
     `min`–`max` and the median of the per-round ratios are:

     | target | `d` | per-round min–max | median |
     |---|---|---|---|
     | `funnel`                   | 10 | 0.02–0.65× | **0.37×** |
     | `eight_schools`            | 10 | 0.30–10.56× | **0.58×** |
     | `seeds`                    | 26 | 0.09–1.12× | **0.51×** |
     | `radon_partially_pooled`   | 88 | 0.38–0.92× | **0.58×** |
     | `radon_variable_intercept` | 89 | 0.26–1.06× | **0.50×** |

     The per-round spans are wide — `eight_schools` covers 0.30× to 10.56× — and
     that is the point of showing them next to the medians. A band quoted from
     medians is a statement about the typical round, not a bound on any round.

     The two outlier-carrying rows are outliers and not a second finding:
     `eight_schools`' 10.56 is one round in which Enzyme's own timing spiked 18×
     (14828 ns against ~830), and `funnel`'s 0.02 is one round in which
     ForwardDiff spiked 20× (16980 ns against ~830). They are left
     in because a `min`–`max` that quietly drops its tails is the thing this
     table exists to stop.
  2. **The open question it ended on has been run.** It asked "whether that
     advantage grows, holds or shrinks with dimension", "awaiting a re-run on
     the fixed specs". The re-run is this document. The honest answer is *still
     not resolved*, but for a different and narrower reason than the boxing: the
     within-target spread across harnesses is as wide as the between-target
     spread, so the noise floor exceeds any slope five points can show. Its
     standing conclusion — "reverse mode wins on every clean measurement to
     date" — survives with one exception: 34 of the 35 comparisons favor Enzyme,
     and the one reversal is a single `annotation_sweep` median
     (`radon_variable_intercept`, `c = 0`, 1.077×; § *Which AD backend*).

     **The reading these numbers most invite is the one they falsify.** Taking
     one figure per target, they no longer even suggest a trend with
     dimension — ≈0.37–0.58 at `d = 10` against ≈0.50–0.58 at `d = 26`–`89`,
     overlapping. The median column above breaks any residue of it: at `d = 88`,
     `radon_partially_pooled` reads **0.58**, equal to `eight_schools` at `d = 10`
     (0.58) and worse than `seeds` at `d = 26` (0.51). The ordering is not
     monotone in `d`. And splitting that same median by centering, `eight_schools`
     alone reads **0.45 at `c = 0` and 0.59 at `c = 0.5`** — one target, one
     dimension, spanning nearly the whole between-target range (0.37–0.58) on its
     own. Any apparent trend is which figure you pick per target, not `d`.
  3. **Its provenance line pointed at a superseded measurement, and one of its
     SHAs was unreachable.** It read "Measured at `b5c7dee` … results checked in
     at `068cdeb`", both of which predate the de-boxing; this revision measures
     `7b1757a`. Separately, it names the de-boxing commit as
     `f639bb1` — the orphaned **pre-rebase** copy of the same change that landed
     as **`e9bcfd0`**. The distinction is the useful part: `f639bb1` is
     reachable from **no ref**, so it resolves in the worktree that did the
     rebase and dies in a fresh clone. It was written down *because* it resolved
     when it was checked. `git rev-parse <sha>` in your own worktree is
     therefore not evidence a SHA is citable; `git merge-base --is-ancestor
     <sha> <ref>`, or a resolve in a fresh clone, is. Same failure shape as a
     `git fetch` with no remote configured exiting 0.

     **This document then did the same thing with its own base.** `5637fcf`, the
     SHA the header stood on for every revision until this one, is in exactly the
     state described above: present here, contained by zero refs, unresolvable in
     any fresh clone. It was recorded because it resolved when it was checked —
     the identical mistake, made while this bullet sat further down the same
     page explaining it. Diagnosing a failure mode in someone else's citation
     does not inoculate you against it in your own; only running the check does,
     which is why `artifact_currency.jl` now asks `git for-each-ref --contains`
     rather than trusting `rev-parse`.

  4. **Its DI-preparation figure inverted, though its conclusion held.** It
     recorded prep as "10–16% of the call at `d ≈ 88`, and equal under both
     backends". Prep's *absolute* cost was near-identical across backends then
     (26.1 µs Enzyme vs 27.7 µs ForwardDiff on `radon_partially_pooled`); on this
     base it reads 28.3 µs vs 40.0 µs — no longer near-identical, and the shares
     moved with the calls: **55–206%** of the unprepped Enzyme call and 12–51% of
     the ForwardDiff one across the two radon targets. The docstring's actual
     conclusion, that reusing a prep object measures neutral-to-worse, still
     replicated (+15% on `radon_partially_pooled`).

  The two claims this benchmark could have contradicted now both fail, and that
  is the retraction's content: since `1a395ce` a bare `AutoEnzyme()` runs (it no
  longer raises `EnzymeMutabilityException` here), and `Duplicated` no longer
  costs 1.6–16× (§ *`function_annotation`*) — the gap collapsed to run-to-run
  noise around unity. "The hint diagnoses the problem; it is not the fix" was
  right while there was a measurable problem to diagnose; what survives is the
  semantic half: `Const` is correct because the closure is not differentiated
  with respect to.
