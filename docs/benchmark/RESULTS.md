# Measured: adaptive reparametrization vs the fixed centering endpoints

**Measured on WarmupHMC `b5c0b95`, with `src/` clean.** 8 pinned seeds per arm,
`n_draws` floor 1000, Julia 1.10.11, single-threaded BLAS, on `strato2`. 184 runs
per backend, 0 failed. The worktree outside `src/` was not clean — the same
campaign's other harnesses were writing their own artifacts beside these — so
both runs record `worktree_dirty: true` beside `src_dirty: false`, the
combination `artifact_currency.jl` accepts.

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
what runs; the pair below is `src_dirty: false`. The previous pair (`7b1757a`)
showed the same bit-for-bit agreement between its Enzyme run and an earlier
code-identical one (`be6fb23`): all 184 runs, every field except wall-clock.

`b5c0b95` will fall behind the tip, so the gap is **checked** rather than stated.
`docs/benchmark/code_identical.jl` parses every file under `src/` at two
revisions, strips docstrings and line numbers, and compares the resulting ASTs:

    julia docs/benchmark/code_identical.jl b5c0b95 <tip>

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

"Stale" means "some file under `src/` changed" unless the artifact says
otherwise. A gradient-only probe can say so with a `<stem>.SCOPE` file beside
its JSON. Line 1 is the reason; each further line is one `src/*.jl` file the
measurement exercises: the files line coverage of that harness executes, plus
`src/WarmupHMC.jl` for the imports and include order they compile under. The
script then compares only those files. It prints the scope
and the out-of-scope files it ignored on that artifact's line, and it reports a
scope naming a path outside `src/` or absent at the tip as red. Anything that
runs the sampler carries no scope and keeps the whole-`src/` rule. Without
this, a commit that only relabels progress text re-stales every gradient
timing, and re-measuring those on a contended host flips their direction from
run to run.

Code that differs is not always code that a measurement runs. When a commit
changes only code no benchmark executes, such as resume plumbing or run-directory
guards, `results/equivalence/<from>-<to>/` can record that the two revisions
produce the same output. Artifacts measured at a base code-identical to
`<from>` then stay current at a tip code-identical to `<to>`. The record has to
carry its evidence, and the script re-checks that evidence on every run:
- a `REASON` file;
- the benchmark driver's `runs.json` from each revision, run with the same
  settings, each recording its own SHA and a clean `src/`, and neither
  recording a failed run;
- the two files identical line for line, except the host-timing fields and the
  provenance header.

A record that fails any of this is red. It covers one hop, so the next code
change re-stales everything as usual. The driver's evidence covers its own
targets and arms. For artifacts from other harnesses, the claim rests on the
stated reason, which is printed on every line that relies on it.

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
`results/enzyme-b5c0b95/` (the default, `AutoEnzyme(; function_annotation =
Enzyme.Const)`) and `results/forwarddiff-b5c0b95/` (`AutoForwardDiff()`). The
runs were sequential and never concurrent with each other — both queued on the
same compute token, with one other harness between them — but `strato2` was
carrying two BRM inventory sweeps from the same campaign and other agents' jobs
throughout, and [the wall-clock section](#it-converts-into-wall-clock-on-all-five-targets)
measures what that does to ESS/sec. `summarize.jl` regenerates
every table below from those records; `compare.jl` regenerates the backend diff.

Superseded runs are kept: `results/enzyme-7b1757a/` and
`results/forwarddiff-7b1757a/` (the previous live pair, replaced by the one
above because `7b1d694` restored Pathfinder's low-rank factor and so changed
every Pathfinder-initialised trajectory), `results/enzyme-d68d680/` and
`results/forwarddiff-d68d680/` (the pair before that),
`results/enzyme-5637fcf/` and `results/forwarddiff-5637fcf/` (the pair
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
the plain sampler on every target measured** — 3.3× to 137× per gradient
evaluation. It gets there without being told which parametrization to use.

**Against the best available *fixed* parametrization the picture splits two
ways:**

| target | adaptive | best fixed alternative | |
|---|---|---|---|
| radon partially_pooled | **77.44** | 19.09 (hand-written noncentered) | **4.1× win** |
| radon variable_intercept | **75.25** | 22.72 (plain centered model) | **3.3× win** |
| seeds centered | **21.75** | 14.42 (noncentered endpoint) | **1.5× win** ‡ |
| funnel (synthetic) | 96.43 | 108.57 (noncentered endpoint) | loses, −11% |
| eight_schools centered | 46.47 | 57.32 (noncentered endpoint) | loses, −19% |

(min ESS per 1000 gradient evaluations, median over 8 seeds, Enzyme run. The
ForwardDiff run is identical on every adaptive row except `eight_schools`,
45.64; two of the comparators drift between backends — see
[What each arm is](#what-each-arm-is).)

‡ **`seeds` adapts again.** At `7b1757a`, the previous base, six of the eight
seeds never left the centered start — every learned `c` ended at exactly 1.0 —
and adaptive trailed the noncentered endpoint by 37% (7.61 against 12.06). On
this base all eight seeds settle at an interior per-plate `c` (per-seed medians
0.2–0.4, no coordinate left at 1.0) and adaptive wins 1.5× (21.75 against
14.42), as it did at `d68d680` (16.92 against 11.28). Per-gradient columns are
counts, bit-reproducible on a given `src/`, so the reversal is the code change,
not noise.

**Why it stopped, and why it is back:**
- The centering is refit only at a window that restarts.
- Whether a window restarts is decided per coordinate, in the frame of
  whichever square root of the metric is active. The test is
  `variance_cond_target`, default 2.
- `f210206` replaced the initial Pathfinder scale's low-rank factor with a
  dense Cholesky factor. In that frame `seeds`' restart statistic sat at
  1.46–1.96, so most runs never restarted and never evaluated the centering.
- `7b1d694` restored the low-rank factor (decision
  `2026-10-01T10-27-50-121-1o12joq`), and with it a frame in which `seeds`
  restarts.
- Across four square roots of the same covariance, 12 seeds adapt 5, 12, 0 and
  1 times. This is a scratch experiment with the square root swapped in, not a
  checked-in artifact; it is recorded in the WarmupHMC primer.

The frame dependence itself is unchanged. Keeping it, and documenting it, was a
deliberate choice (decision `2026-10-01T06-43-03-508-1312sap`), and the
frame-invariant alternatives measured on `seeds` are recorded in the WarmupHMC
primer. The restored factor is a frame in which `seeds` restarts; the next
change to the initial scale can move that again, so read this row as the
sampler's current behaviour, not as a property of the method.

Read as one sentence: **where the best centering is interior, adaptive beats
every fixed option; where an endpoint is exactly right, adaptive finds it but
trails the fixed endpoint by 11–19%.** Both radon models and `seeds` are the
first case. On the radon models centered, noncentered and posteriordb's own
hand-written noncentered model all land within seed noise of each other around
16–23, and adaptive gets 75–77 by settling at an *interior* `c ≈ 0.4–0.5` that no
hand-written model offers. That interior optimum is the strongest result here;
on this base it sits between `d68d680`'s 62.6–68.1 and `7b1757a`'s 79.2–82.5.
The funnel and `eight_schools` are the second case.

**The verdict is backend-independent, and that is measured rather than
assumed.** Every `plain`, `fixed_centered` and hand-written arm is identical
across the two backend runs — same gradient counts on every seed: `plain` and
the hand-written model never touch the wrapper, and `fixed_centered`'s exact
identity transform differentiates to bit-identical gradients under both
backends. Of the five `adaptive`
arms, four (both radon models, `seeds`, the funnel) are identical on all eight
seeds; `eight_schools` differs in gradient count on six seeds and moves +3.3% in
median gradient count (ForwardDiff relative to Enzyme). Final `c` vectors agree
between backends on every seed of every target except two `eight_schools`
seeds. All 368 runs completed. So the backend can change *which trajectory* a
seed happens to take; it does not change what the method learns, and no row of
the verdict changes sign under it.

### It converts into wall-clock on all five targets

Under Enzyme, adaptive reparametrization is faster **in seconds**, not just per
gradient, on all five targets, and under ForwardDiff too:

| target | adaptive vs plain, Enzyme/Const | | | | ForwardDiff |
|---|---|---|---|---|---|
| | boxed base `b5c7dee` | `d68d680` | `7b1757a` | **this base** | this base |
| funnel (synthetic) | 17.9× faster | 19.3× faster | 22.2× faster | **38.6× faster** | 33.2× faster |
| eight_schools centered | 21.0× faster | 18.9× faster | 10.5× faster | **25.3× faster** | 16.2× faster |
| radon partially_pooled | 3.5× **slower** | 2.1× faster | 3.2× faster | **2.4× faster** | 2.0× faster |
| radon variable_intercept | 2.3× **slower** | 2.0× faster | 2.7× faster | **2.1× faster** | 1.6× faster |
| seeds centered | 6.1× **slower** | 1.6× faster | 0.95× — break-even | **4.9× faster** | 3.5× faster |

(min ESS/sec, median over 8 seeds, sequential runs.)

Between the boxed base and `d68d680`, three losses became wins, and the three
that moved are precisely the three whose specs were boxed; `funnel` and
`eight_schools` were never boxed and stayed in the high band. The moves since
have sampling causes, not cost causes. At `7b1757a` `seeds` fell back to
break-even because its adaptive arm mostly stopped adapting, and
`eight_schools` roughly halved because its *plain* arm doubled per gradient
(1.55 → 3.01) while adaptive held. On this base both moved back: `seeds` adapts
again (the Verdict's ‡), and `eight_schools`' plain arm is back to 1.60 per 1k
gradients. In both cases the denominator or the adaptation moved, not the
transform's cost.

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
ESS/sec still moved **1% to 36%** between two runs made a few minutes apart
(`plain`: 8% `eight_schools`, 13% and 7% on the two radon models, 9% `seeds`, 9%
`funnel`; hand-written: 1–36%), on a host carrying other agents' jobs. Within
one run the floor is wider still: on `seeds`, `plain` and `fixed c = centered`
take bit-identical trajectories on every seed — and the wrapped arm does
strictly more work per gradient — yet the wrapped arm's median ESS/sec reads 47%
*higher* (512.5 against 349.6).

Read the consequences in the strong direction, not the flattering one:

- **The large ratios are robust.** `funnel` and `eight_schools` are 16–39× on
  both backends of this base and 6–35× on every earlier run. A 36% wobble on a
  20× win changes nothing.
- **`seeds` is a win again, of unresolved size.** 4.9× under Enzyme and 3.5×
  under ForwardDiff on this base, against 0.95× and 0.51× at `7b1757a`, where it
  mostly did not adapt, and 0.9–1.6× across the four July measurements. Its plain
  denominator is the noisiest on the page (the 47% above), so the size is not
  resolved; the direction rests on the per-gradient gain, 3.3× over plain, which
  the wrapper's +23% (Enzyme) or +181% (ForwardDiff) per-gradient overhead
  (below) cannot cancel.
- **Both radon models survive.** 1.6–2.4× across both backends on this base,
  2.1–3.2× at `7b1757a`, 1.5–2.1× across July's four, never crossing parity. A
  real win, whose *size* is not resolved beyond "one and a half to three times".

The per-gradient columns carry none of this uncertainty — they are bit-identical
across all four runs, because they are counts rather than clocks. **Where a
per-gradient number and a wall-clock number disagree here, trust the
per-gradient one**, and treat ESS/sec as the coarse confirmation that the
per-gradient win is not being eaten by transform overhead.

So the claim the package can now support:

> Starting from a centered parametrization, adaptive partial centering beats
> every fixed parametrization per gradient evaluation on the three targets whose
> best centering is interior (both radon models, 3.3–4.1×; `seeds`, 1.5×), and
> finds the exactly-right noncentered endpoint on the funnel and
> `eight_schools` but trails it there (−11% and −19%). Under Enzyme that
> converts into wall-clock wins over the plain sampler on all five targets, from
> 2.1× to 39×.

**The overhead is still not the transform's arithmetic.** A wrapper configured
as an exact identity costs the same as a live one, under both backends and on
every target — on this base, Enzyme ×1.07 no-op vs ×1.07 live on
`radon_partially_pooled`, ×1.05 vs ×1.09 on `radon_variable_intercept`. What
remains is the cost of carrying the AD re-differentiation of
`ljac_(x_) + dot(g_y, y_)` at all. Under Enzyme that is now **+9% of a gradient
on `radon_variable_intercept`, +7% on `radon_partially_pooled` and +23% on
`seeds`** — small enough to be paid out of the sampling gain wherever there is
one, which is what the wall-clock table above shows. It is +72% on
`eight_schools` and 6.5× (+552%) on the funnel, but those are the two targets
whose *bare* gradient costs 456 ns and 46 ns, where a roughly fixed per-call AD
cost has to dominate; both still come out 25–39× ahead overall because the
sampling gain there is enormous. Under ForwardDiff the same three overheads are
+49%, +86% and +181%, and the two small targets go to +382% and 16.0× (+1505%) —
the backend choice is most of what makes this affordable.

Every one of those figures is a ratio of two medians, which is exactly the shape
that produced a reversed verdict elsewhere on this page, so the direction is
checked against the individual rounds rather than assumed from the summary. The
rounds pair: `gradient_overhead` times all three variants on the *same* `xs`
within each round, rotating the order, so round `r`'s live and bare timings are
adjacent in time on identical inputs. Pairing them that way, **67 of the 70
paired rounds** across both backends and all five targets put the wrapper above
the bare gradient. The three exceptions are one `eight_schools` round under
Enzyme at `0.12×` — its *bare* timing spiked to 6088 ns against ~450 — against
six others from `1.63×` to `1.95×`; one `seeds` round under Enzyme at `0.83×`,
against six others from `1.17×` to `1.60×`; and one `radon_variable_intercept`
round under ForwardDiff at `0.91×`, against six others from `1.22×` to `1.79×`
— single anomalous rounds on a contended host, not evidence the wrapper is ever
free.
So the *sizes* above are medians and carry the usual round-to-round spread, but
the *sign* does not depend on them.

## Which AD backend

The differentiated objective `x -> ljac(x) + dot(g_y, y(x))` is scalar in the
full parameter vector with the inner gradient `g_y` frozen, so it is the
textbook reverse-mode shape: forward mode costs `ceil(d / chunksize)` tangent
sweeps of the transform per gradient where reverse mode costs one. The
prediction that follows is that reverse mode should win, and win *hardest* at
large `d`.

**Measured on this base, the direction holds on every target, in every
comparison.** Per wrapped `logdensity_and_gradient` call, ratio of
Enzyme/`Const` to ForwardDiff — below 1.0 means Enzyme is faster:

| target | `d` | Enzyme ÷ ForwardDiff | reading |
|---|---|---|---|
| `radon_partially_pooled` | 88 | **0.48–0.56×** | Enzyme 1.8–2.1× faster |
| `radon_variable_intercept` | 89 | **0.43–0.66×** | Enzyme 1.5–2.3× faster |
| `seeds` | 26 | **0.34–0.61×** | Enzyme 1.6–2.9× faster |
| `eight_schools` | 10 | **0.23–0.68×** | Enzyme 1.5–4.4× faster |
| `funnel` | 10 | **0.20–0.37×** | Enzyme 2.7–5.1× faster |

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
`annotation_sweep.jl` (5 rounds × 2000 calls, backend order rotated per round,
both endpoints), and the driver's own `gradient_overhead` block (7 rounds × 2000
calls, median). **Enzyme is faster in all 35 comparisons those four harnesses
produce.** Until `b35a718` there was one exception, `radon_variable_intercept`
in `annotation_sweep` at `c = 0`, 1.077×. It was a single 2000-call median from
a sweep that timed each backend once, in a fixed order. The rotated
re-measurement reads 0.612×, inside the 0.43–0.66× the other three harnesses
give on that target. The two lowest band edges went with it: `seeds` 0.12× and
`funnel` 0.13× were also fixed-order sweep cells. An earlier draft of this revision
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
`funnel` and `eight_schools`, both at `d = 10`, and the two radon targets at
`d = 88`–`89` have the highest lower edges. But the spread *within* a single
target across harnesses is as large as the spread *between* targets —
`eight_schools` alone spans 0.23 to 0.68, which covers nearly the entire
between-target range (0.20–0.68) on its own — so these five points cannot
resolve a scaling law either way. The direction replicates; the slope does not.

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
  `radon_partially_pooled` 0.49× → 0.51×; `radon_variable_intercept` 0.61× →
  0.66×; `seeds` 0.55× → 0.34×; `eight_schools` 0.23× → 0.56×; `funnel` 0.35× →
  0.36×). Recorded in `results/typical_positions.json`.

  **No single target's shift is *stably* separable from repeat-to-repeat noise,
  and the file now carries what proves it.** Each median is over `ROUNDS` timing
  rounds and those rounds are persisted raw, so three independent statistics can be
  computed by a reader rather than taken on trust: the `randn` and typical round
  ranges overlap on every target (the previous run separated `eight_schools`, so
  separation itself flips run to run); the ratio spread runs 30.0% to 185.4% of
  its own median; and median-to-median across two earlier runs of *identical
  code* moved as much as **71.1%** (`funnel` 0.379 → 0.649). This run's largest
  median shift, `eight_schools`' +146%, sits on the widest `randn` column of the
  four posteriordb targets, whose five rounds span 0.18× to 1.07×. The two
  columns are separate timing loops over different positions, so they are
  compared by range, never round by round; an earlier revision of this bullet
  divided round *i* of one by round *i* of the other, which pairs nothing. An
  earlier revision still read the funnel's swing in the
  *opposite direction* (0.70× at `randn` vs 0.38× typical, where this run has
  0.35× vs 0.36×) and explained it with a story about a noisy
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

### `function_annotation` is no longer required, and what it costs is now noise

A bare `AutoEnzyme()` **used to fail outright** on this objective with
`EnzymeMutabilityException` — the objective is a closure capturing the
reparametrizer and the frozen `g_y`. `1a395ce` ("make a bare `AutoEnzyme()` work
at both AD sites") removed that failure. Two cost questions remain:
- what an annotation costs against the bare backend;
- what the wrong one costs against `Const`. The wrong one is `Duplicated`,
  which Enzyme's own error text suggests.

`annotation_sweep.jl`, both centerings, 5 rounds × 2000 calls with the backend
order rotated per round:

| target | `d` | `Const` ÷ bare | `Duplicated` ÷ `Const` | `Duplicated` − `Const` |
|---|---|---|---|---|
| `funnel` | 10 | 1.01–1.48× | 1.02–1.06× | +0.01 to +0.02 µs |
| `eight_schools` | 10 | 0.86–1.40× | 0.83–0.92× | −0.19 to −0.07 µs |
| `seeds` | 26 | 1.03× | 1.00–1.31× | −0.01 to +1.17 µs |
| `radon_partially_pooled` | 88 | 0.98–1.03× | 1.00× | −0.02 to +0.10 µs |
| `radon_variable_intercept` | 89 | 0.97–1.00× | 1.01–1.03× | +0.60 to +1.28 µs |

**`Const` against bare:** 0.86–1.48×, median 1.02×. The two cells above
1.4× are `funnel` and `eight_schools` at `c = 0.5`; at `c = 0` the same targets
read 1.01× and 0.86×. So the "~1.7× pessimization" that
`docs/src/reparametrization.md` used to quote from one funnel measurement is
above the top of this range, not its typical value. That page now generates its
figure from this artifact.

**`Duplicated` against `Const`:** 0.83–1.31×. The 1.7–27.4× gap of earlier
revisions ("never free") is gone. A repeat run of the old fixed-order sweep
flipped the direction of 8 of its 10 cells, which is why the sweep now rotates
the order and takes medians over rounds. Whether Enzyme improvements or the
reparametrizer rewrites closed the gap is not separated here. Every arm's
gradient agrees with ForwardDiff's to ≤ 6.8e-13.

**The previous revision of this table said `Duplicated` was free on the three
larger targets (0.96–1.06×), and it flagged the reason to distrust that: those
were the three boxed specs, where a ~15× defect swallowed a 40 µs shadow copy.**
That caveat was right. Unboxed, the same three targets show a 1.7–5.0× penalty.
It is a useful calibration on how much a confounded measurement can hide: not a
few percent, but the entire effect.

So the statement that now survives, in two parts:
- *Some* annotation was mandatory while a bare `AutoEnzyme()` did not run at
  all. Since `1a395ce` none is: the bare backend is what
  `docs/src/reparametrization.md` recommends.
- On this sweep, the three spellings cost the same to within run-to-run noise.

The Enzyme numbers elsewhere in this document were measured with `Const`, and
this table is the evidence that they describe the bare backend too, to within
that noise.

### The two methods used to disagree. They now reconcile to within a few microseconds

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
| `radon_partially_pooled` | fixed c = centered | 8 | 16.6 µs | 23.5 µs |
| `radon_partially_pooled` | adaptive | 8 | 27.0 µs | 23.5 µs |
| `radon_variable_intercept` | fixed c = centered | 8 | 26.0 µs | 31.8 µs |
| `radon_variable_intercept` | adaptive | 8 | 37.8 µs | 31.8 µs |
| `seeds` | fixed c = centered | 8 | 3.3 µs | 4.6 µs |
| `eight_schools` | fixed c = centered | 8 | 3.2 µs | 2.0 µs |
| `funnel` | fixed c = centered | 8 | −0.3 µs | 0.5 µs |

(ForwardDiff minus Enzyme, per gradient evaluation; median over seeds on the
left, the `gradient_overhead` live-wrapper medians on the right.)

The three small targets agree to within 1.3 µs. On the two radon targets the
sampler's difference brackets the microbenchmark's rather than matching it: the
`fixed c = centered` arms realise 16.6 of a predicted 23.5 µs and 26.0 of
31.8 µs, the adaptive arms 27.0 and 37.8 — 71% to 119% of the prediction, every
row within 7 µs of it. On the previous base `radon_partially_pooled` matched to
within 4 µs and `radon_variable_intercept` was the loose one, so which row is
loose moves between runs. That points at per-gradient clock noise on a
contended host rather than a stable second cost, though nothing here isolates
it. The sign agrees on every row except the funnel's, where both differences
are under a microsecond.

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
ForwardDiff run except on the five arms noted below.

`min ESS` is the minimum over coordinates — the number that governs how long you
must run. Sub-figures are the min–max across the 8 seeds. `final source c` is
read back off the mutated `IndexedReparametrization` after the run, which is
valid because every run here is single-chain `adaptive_warmup_mcmc`; the
cooperative and clustered samplers `deepcopy` the problem per chain and would
silently return the *initial* `c` instead.

> **Where the two backend runs are not sampling-identical.** All four
> posteriordb `fixed_noncentered` arms differ on every seed, and drift in median
> gradient count between backends (ForwardDiff relative to Enzyme) by
> `seeds` +6.3%, `radon_variable_intercept` −2.6%, `eight_schools` +2.3%,
> `radon_partially_pooled` −1.9%; `eight_schools`/adaptive differs on six of
> eight seeds (+3.3% median gradient count). The funnel's `fixed_noncentered`
> arm differs on one seed, in the fifth significant figure of its ESS and in no
> gradient count. Every other arm is identical on
> every seed. `plain` and the hand-written model never touch the wrapper;
> `fixed_centered` does run its AD, but an exact identity transform differentiates
> to bit-identical gradients under both backends. The cause of the drift is the
> mechanism described under [Harness correctness](#harness-correctness): the
> backends' gradients through a non-trivial transform differ at the 1e-13 level
> and NUTS amplifies that into a different trajectory.
>
> **Three verdict rows depend on it, and none changes sign.** `eight_schools`
> drifts on both sides: adaptive reads 45.64 under ForwardDiff against
> Enzyme's 46.47, and its `fixed_noncentered` comparator 51.85 against 57.32,
> so its gap is −19% on one trajectory set and −12% on the other. `seeds` is
> compared against its `fixed_noncentered` row, 13.39 under ForwardDiff
> against 14.42, so its win is 1.5× or 1.6×. And on `radon_partially_pooled`
> the drifting `fixed_noncentered` arm reads 22.74 under ForwardDiff — above
> the hand-written model's 19.09 — which would make it the best fixed option
> and the margin 3.4× instead of 4.1×. The `fixed_noncentered` rows are never
> used as a cross-backend wall-clock comparison.

<!-- The TABLES below were generated by docs/benchmark/summarize.jl from
     results/enzyme-b5c0b95/runs.json. The PROSE between them was not, and no
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
| plain (no wrapper) | 39.5 <sub>19.9–64.2</sub> | 254.4 <sub>55.7–1086.8</sub> | 1.60 | 28560 | 5.0 | — |
| fixed c = centered | 57.7 <sub>21.0–93.5</sub> | 387.8 <sub>211.5–911.3</sub> | 1.93 | 27879 | 4.0 | 1.0 |
| **adaptive** (starts centered) | 540.6 <sub>465.7–639.6</sub> | 6438.4 <sub>5081.3–8860.1</sub> | 46.47 | 11304 | 0.0 | 0.0 |
| fixed c = noncentered | 504.1 <sub>432.2–717.1</sub> | 21055.1 <sub>3248.9–25425.9</sub> | 57.32 | 8423 | 0.0 | 0.0 |
| hand-written noncentered model | 477.5 <sub>322.9–602.6</sub> | 12599.0 <sub>3807.0–23771.1</sub> | 42.60 | 9548 | 0.0 | — |

Min ESS over the 10 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 39.5 | 1.38 |
| fixed c = centered | 57.7 | 2.07 |
| **adaptive** (starts centered) | 540.6 | 47.82 |
| fixed c = noncentered | 504.1 | 59.85 |
| hand-written noncentered model | 483.9 | 50.69 |

Adaptive drives `c` to the noncentered endpoint: **59 of the 64 learned values
(8 seeds × 8 coordinates) are exactly 0.0 and the other 5 are 0.1** — the known
right answer for this model, found without being told (the same split as at
`7b1757a`; 53 and 11 at `d68d680`). Finding the endpoint does not mean matching
it, though: adaptive reads 46.47 per 1k gradients against 57.32 for the
noncentered endpoint reached by transform — 19% behind, and 20% behind on the
matched parameters too (47.82 vs 59.85). Against posteriordb's hand-written
model it is level: 9% ahead on all parameters (46.47 vs 42.60), 6% behind on the
matched ones (47.82 vs 50.69). The gap to the endpoint is in gradients, not in
ESS — adaptive's median min ESS is the higher of the two (540.6 vs 504.1), but
it spends 11304 gradient evaluations against the endpoint's 8423 (more on 6 of
8 seeds) — plausibly the warm-up spent at `c = 1` before switching, though
nothing here isolates that. Divergences go from a median of 5 on plain to 0.

Two cautions on this target specifically. Its min-ESS spreads overlap
(465.7–639.6 adaptive, 432.2–717.1 at the endpoint), so only the gradient count
separates the two arms. And it is the one target whose adaptive arm drifts
between backends: the ForwardDiff run reads 45.64 / 51.85 / 42.60 for the same
three arms, so the gap to the endpoint is 12% there, and the ordering does not
depend on the backend. The safe reading is **finds the right endpoint, trails
the fixed endpoint it converges to by 12–19%, level with the hand-written
model**.

This is also the target where the pre-fix run over-stated the method: on
`05aed41` adaptive appeared to *beat* the hand-written model by 41%. It is level
with it.

### `radon_mn-radon_partially_pooled_centered`

*85 counties — dimension 88.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 193.5 <sub>146.0–307.5</sub> | 473.4 <sub>356.8–596.3</sub> | 16.11 | 13470 | 0.0 | — |
| fixed c = centered | 216.2 <sub>125.9–311.0</sub> | 429.2 <sub>276.6–542.0</sub> | 16.46 | 15122 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 700.7 <sub>525.5–1148.6</sub> | 1117.3 <sub>657.2–1494.2</sub> | 77.44 | 8586 | 0.0 | 0.4 |
| fixed c = noncentered | 340.2 <sub>221.3–473.3</sub> | 508.4 <sub>254.4–803.7</sub> | 18.05 | 16678 | 0.0 | 0.0 |
| hand-written noncentered model | 294.3 <sub>173.8–421.5</sub> | 527.9 <sub>384.7–709.6</sub> | 19.09 | 16333 | 0.0 | — |

Min ESS over the 88 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 193.5 | 14.37 |
| fixed c = centered | 216.2 | 14.30 |
| **adaptive** (starts centered) | 700.7 | 81.61 |
| fixed c = noncentered | 340.2 | 20.40 |
| hand-written noncentered model | 294.3 | 18.02 |

**The headline case for the method.** No fixed parametrization helps: centered
(16.46), noncentered (18.05) and posteriordb's hand-written noncentered model
(19.09) are all in the same band. Adaptive gets 77.44 — **4.1× the best fixed
option** — using 49% fewer gradient evaluations than the noncentered endpoint,
and it is also the fastest arm in **wall-clock**, at 1117.3 ESS/sec against the
plain sampler's 473.4. The `final source c` column reports 0.4, but that is
a **median over 85 counties, not a setting**: see below.

#### What adaptation actually learns is a `c` *per coordinate*

The `final source c` column is a median, and on the two radon models it hides
the result. Every one of the 85 counties gets its own centering, and they do not
agree with each other. Distribution of the learned `c` over all 8 seeds ×
coordinates, Enzyme run (the ForwardDiff run is identical except on two
`eight_schools` seeds):

| target | n | median | quartiles | at `c = 0` | at `c = 1` |
|---|---|---|---|---|---|
| `radon_partially_pooled` | 680 | 0.40 | 0.30 – 0.60 | 1% | 2% |
| `radon_variable_intercept` | 680 | 0.50 | 0.40 – 0.60 | 1% | 1% |
| `seeds` | 168 | 0.30 | 0.20 – 0.40 | 2% | 0% |
| `eight_schools` | 64 | 0.00 | 0.00 – 0.00 | 92% | 0% |
| `funnel` | 72 | 0.00 | 0.00 – 0.00 | 100% | 0% |

On the radon models and on `seeds` **98% of coordinates end strictly between the
two endpoints**, spread across the interval rather than clustered at the
median. That is the thing no fixed parametrization can express: centered,
noncentered and posteriordb's hand-written noncentered model each impose *one*
`c` on all 85 counties (or 21 plates), so none of them can represent any of
these fits — which is the mechanism behind the 4.1×, 3.3× and 1.5× margins, and
why those margins appear on exactly the three targets whose fits are interior.
The `seeds` row is back to its `d68d680` shape (median 0.30, quartiles 0.20–0.50,
0% at `c = 1` then); at `7b1757a` three quarters of its coordinates never left
`c = 1` (see its section).

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
| plain (no wrapper) | 275.7 <sub>212.1–402.2</sub> | 430.3 <sub>195.6–492.2</sub> | 22.72 | 15258 | 0.0 | — |
| fixed c = centered | 278.1 <sub>180.8–402.2</sub> | 413.2 <sub>291.1–516.5</sub> | 21.10 | 14912 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 734.6 <sub>666.6–1160.2</sub> | 918.3 <sub>716.8–1038.7</sub> | 75.25 | 10378 | 0.0 | 0.5 |
| fixed c = noncentered | 360.4 <sub>253.8–482.2</sub> | 419.8 <sub>297.6–556.6</sub> | 21.72 | 16298 | 0.0 | 0.0 |
| hand-written noncentered model | 322.4 <sub>200.7–424.6</sub> | 296.6 <sub>204.2–439.5</sub> | 18.34 | 16749 | 0.0 | — |

Min ESS over the 89 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 275.7 | 18.07 |
| fixed c = centered | 278.1 | 18.65 |
| **adaptive** (starts centered) | 734.6 | 70.79 |
| fixed c = noncentered | 360.4 | 22.11 |
| hand-written noncentered model | 322.4 | 19.25 |

Same shape as its sibling, and the same per-coordinate story — 98% of the 85
county centerings land strictly inside `(0, 1)` with quartiles 0.4–0.6, so the
reported `c = 0.5` is again a median over coordinates. Worth 3.3× the best fixed
alternative on gradients (here the plain centered model, 22.72) and 2.1× the
plain sampler in wall-clock. The `fixed_noncentered` row is one of the arms that
drifted between backends (−2.6% gradients); the ForwardDiff run puts it at
14.66, below plain, so the best fixed option and the 3.3× margin are the same
under both backends. The adaptive and plain rows are identical across backends.

### `seeds_data-seeds_centered_model`

*21 plates — dimension 26. posteriordb ships no noncentered sibling.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 94.4 <sub>5.6–141.7</sub> | 349.6 <sub>45.0–1675.8</sub> | 6.50 | 14755 | 0.0 | — |
| fixed c = centered | 94.4 <sub>5.6–141.7</sub> | 512.5 <sub>36.3–1395.5</sub> | 6.50 | 14755 | 0.0 | 1.0 |
| **adaptive** (starts centered) | 309.7 <sub>212.5–385.2</sub> | 1700.9 <sub>1301.4–2193.0</sub> | 21.75 | 13270 | 0.0 | 0.3 |
| fixed c = noncentered | 235.7 <sub>199.4–322.2</sub> | 2208.6 <sub>703.0–2921.6</sub> | 14.42 | 16820 | 0.0 | 0.0 |

Min ESS over the 47 constrained parameters all arms share:

| arm | matched min ESS | per 1k grad |
|---|---|---|
| plain (no wrapper) | 94.4 | 6.40 |
| fixed c = centered | 94.4 | 6.40 |
| **adaptive** (starts centered) | 309.7 | 23.34 |
| fixed c = noncentered | 235.7 | 14.01 |

**The target the restored factor gave back.** At `7b1757a` six of the eight
seeds never left the start: every one of their 21 learned `c` values was exactly
1.0, the arm's median gradient count equalled `plain`'s, and adaptive trailed
the noncentered endpoint by 37% (7.61 vs 12.06 per 1k gradients). On this base
**every seed adapts** — per-seed medians 0.2–0.4, quartiles 0.2–0.4 over all
168 learned values, none left at 1.0. Adaptive beats the noncentered endpoint
1.5× (21.75 vs 14.42; 1.6× against ForwardDiff's 13.39 for that drifting arm)
and the plain sampler 3.3×, on fewer gradients than either on every seed (median
13270 against 16820 and 14755). The matched parameters agree (23.34 vs 14.01).
That is the `d68d680` picture (16.92 vs 11.28) again, at a higher level.

These columns are counts, bit-reproducible on a given `src/`, so the reversal is
a behaviour change, not seed noise. Its cause is the frame dependence described
under the Verdict's ‡: whether a window restarts, and so whether the centering
is refit at all, depends on which square root of the metric is active, and
`7b1d694` restored the one in which `seeds` restarts. The numbers above describe
the sampler as it ships today.

Wall-clock follows against plain: 4.9× under Enzyme (1700.9 vs 349.6 ESS/sec)
and 3.5× under ForwardDiff (1127.7 vs 319.2), on the noisiest denominator on the
page (`fixed c = centered`, on bit-identical trajectories, reads 512.5). Against
the noncentered endpoint the per-gradient win does not show in seconds — 1700.9
vs 2208.6 under Enzyme, 1127.7 vs 1170.1 under ForwardDiff. Both arms carry the
same wrapper, and the endpoint's own seeds span 703.0–2921.6 ESS/sec, so a 1.5×
per-gradient margin is inside this column's clock spread.

### `funnel` (synthetic)

*Neal's funnel, `v ~ Normal(0, 3)`, `theta_i ~ Normal(0, exp(v/2))`, K = 9 —
dimension 10. Not a posteriordb posterior; posteriordb ships none. Defined in
`common.jl` with an analytic gradient. It is the extreme case — the centered
parametrization is pathological and the noncentered one is exact — so it bounds
what partial centering can possibly buy. Reported alongside the real targets,
never as a headline.*

| arm | min ESS | ESS/sec | ESS per 1k grad | grad evals | divergences | final source `c` |
|---|---|---|---|---|---|---|
| plain (no wrapper) | 22.6 <sub>6.3–37.9</sub> | 377.6 <sub>101.1–1461.8</sub> | 0.70 | 20425 | 1.5 | — |
| fixed c = centered | 22.6 <sub>6.3–37.9</sub> | 210.6 <sub>100.6–784.9</sub> | 0.70 | 20425 | 1.5 | 1.0 |
| **adaptive** (starts centered) | 817.7 <sub>733.5–924.0</sub> | 14557.6 <sub>6889.5–16906.6</sub> | 96.43 | 8002 | 0.0 | 0.0 |
| fixed c = noncentered | 833.6 <sub>721.1–912.1</sub> | 41670.1 <sub>33286.2–61454.0</sub> | 108.57 | 7710 | 0.0 | 0.0 |

Adaptive trails the exact answer by **11%** on gradient efficiency: 96.43 vs
108.57 per 1000 gradients. Noncentered is *exactly* right here, so there is
nothing to discover and the best adaptation can do is reach it. It did, to
within 1%, at `7b1757a` (111.96 vs 111.24), and fell 13% short at `d68d680`
(93.17 vs 107.05) — so this row moves between bases too, and is back near its
`d68d680` value; nothing here bisects which `src/` change moved it. Against the
parametrization a user
would actually have written it is **137× better** (96.43 vs 0.70), so this row
bounds the loss, not the gain.

The search costs gradients — 8002 against the endpoint's 7710, 3.8% more and
more on 6 of 8 seeds — and min ESS is level, 817.7 vs 833.6 on overlapping seed
ranges. Adaptation converges to the right fixed answer (all 72 learned values
exactly 0.0). What it does not recover is wall-clock against the endpoint
(14557.6 vs 41670.1 ESS/sec): on a 46 ns gradient the wrapper's per-call AD cost
is 6.5× the gradient itself (§ *Cost on the gradient hot path*), and that is
paid on every call whether or not `c` is still moving.

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
| `radon_variable_intercept` | 54719 | 81712 | ×1.49 | 45919 | 49955 | **×1.09** |
| `radon_partially_pooled` | 27410 | 50987 | ×1.86 | 25780 | 27508 | **×1.07** |
| `seeds` | 2908 | 8177 | ×2.81 | 2904 | 3560 | **×1.23** |
| `eight_schools` | 569 | 2746 | ×4.82 | 456 | 782 | **×1.72** |
| `funnel` | 51 | 820 | ×16.05 | 46 | 297 | **×6.52** |

(`bare` is given for each run separately; the two agree to within 6% on `seeds`
and `radon_partially_pooled` and differ by 12–25% on the other three — the same
call, the same inputs, two runs minutes apart on a contended host. The multiples
are taken within one run, round by round, so they carry less of that.)

**The multiple tracks how expensive the underlying density is, not `d`.** The
two most expensive densities carry the wrapper for +9% and +7% under Enzyme; the
funnel, whose bare gradient is 46 ns of analytic arithmetic, pays 6.5×. This is what a
roughly fixed per-call AD cost looks like divided by a varying denominator, and
it is why the funnel is reported as a bound rather than as a headline.

**The no-op and live configurations cost the same, in both backend runs.** An
identity reparametrization — source `c` equal to target `c`, so the transform
provably does nothing — pays the same multiple as a live one: within ±5% on nine
of the ten rows, with no consistent sign (four of the ten rows put the *no-op*
higher, which the live transform doing strictly more work cannot produce, so ±5%
is the probe's floor rather than a measured effect). The tenth, `seeds` under
ForwardDiff, reads ×2.37 no-op against ×2.81 live, on per-round multiples that
span 0.78–4.16 (no-op) and 1.08–4.14 (live) — inside its own spread.

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
> the harness, and the sweeps were right. **On the last two bases the absolute
> check no longer holds, and the reason is the host, not the harness:** the
> driver says 81712 ns on this base (82518 on `7b1757a`), while
> `replicate_backends.jl` — last measured at `fbe9030`, during heavier
> contention — has medians of 171194 and 223959 at its two centerings. Its
> Enzyme÷ForwardDiff ratio on this target, 0.43–0.44, is still in line with the
> driver's 0.61 and the four harnesses' 0.43–0.66 band. Read absolute
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
  through a noncentered source returns coordinate 1 at mean 0.047, sd 3.212
  against the known `Normal(0, 3)` marginal, and applying `reparametrize!` on top
  inflates the leg sds to 143–531 (`results/frame_check.json`). **Numbers in `results/before`, `results/after`,
  `results/forwarddiff-b5c7dee` and `results/enzyme-b5c7dee` were all measured on
  bases predating `b109210`, where the compensation was correct.** Anything
  measured after it must not carry one.
- **The no-op arm really is a no-op.** `plain` and `fixed c = centered` are
  mathematically the same sampler, and agree to the last gradient evaluation on
  every seed for `seeds_data` (94.4 min ESS, 14755 grads, both arms) and the
  funnel (22.6, 20425) — while differing on eight_schools and both radon models.
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
  | `adaptive` | 33/40 — 8/8 on both radon models, `seeds` and the funnel, **1/8 on `eight_schools`** |
  | `fixed c = noncentered` | 7/40 — seven of eight on the funnel, 0/32 on the posteriordb targets |

  The clean split is `fixed c = centered` (source `c` equal to target `c`, so
  the transform is an identity) reproducing exactly on every run, against
  `fixed c = noncentered` (a live transform) reproducing on almost none. The
  wrapped arms that differ are differing through the very AD path this document
  measures, which is why an individual ESS/sec figure is reproducible only
  against a fixed backend — and why the verdict is stated in ESS per gradient,
  where 33 of 40 adaptive runs are bit-identical between backends and the rest
  are named above. (`seeds`' 8/8 is now a live transform on every seed; at
  `7b1757a` six of its adaptive runs never left the identity transform.)

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
  advantage is 1.5–5.1× across all five targets with no visible trend in `d`,
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
     harnesses it currently gives `funnel` 0.20–0.37× and `eight_schools`
     0.23–0.68×.

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
     date" — survives: all 35 comparisons favor Enzyme. The one earlier
     reversal was a fixed-order `annotation_sweep` median
     (`radon_variable_intercept`, `c = 0`, 1.077×). It did not survive the
     rotated re-measurement (§ *Which AD backend*).

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
     `b5c0b95`. Separately, it names the de-boxing commit as
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
