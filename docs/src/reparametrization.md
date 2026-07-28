# Nonlinear reparametrization

Hierarchical models have a well-known pathology: written in their natural
("centered") form, a group-level coordinate's scale is itself a parameter, so the
geometry the sampler sees changes shape as that scale moves. The textbook fix is
to rewrite the model in a "non-centered" form. Which of the two is better is
model- *and* data-dependent, and for a model with many groups the answer need not
be the same for every group.

WarmupHMC can adapt that choice during warm-up, per coordinate and continuously
rather than as an either/or, by giving each reparametrized coordinate a centering
[`PartiallyCentered`](@ref)`(c)` with `c ∈ [0, 1]` — `1` centered, `0`
non-centered — and re-fitting `c` at warm-up window boundaries.

!!! warning "This page makes no claim about sampling efficiency"
    Nothing here says the reparametrization is faster, cheaper or better-mixing
    than not using it. What is documented below is what the machinery *does*.

    Efficiency is measured on other pages, each under limits those pages state,
    so this one links rather than summarising — a verdict copied to here would
    be a second copy that only one of the two updates.
    [Adaptive centering at fixed `c`](@ref) scores fixed centerings, including
    the fully centered and whitened non-centered endpoints, by minimum-coordinate
    ESS per thousand gradient evaluations; note its `c` is fixed rather than
    adapted, and its targets are constructed. [WarmupHMC vs other samplers](@ref)
    compares whole samplers at their defaults, where — as that page says — no
    reparametrization is fitted in any arm.

## It does nothing until you build it

This is the first thing to know, and it is easy to get wrong in a way that looks
like success:

```julia
WarmupHMC.reparametrizer(::Any) = IndexedReparametrization([])
```

Every problem that is not a [`ReparametrizedProblem`](@ref) reports an **empty**
reparametrization, an empty one is a **no-op**, and `nonlinear_adapt=true` — the
default on every sampler — then adapts nothing at all. There is no automatic
detection of hierarchical structure. A run with `nonlinear_adapt=true` on a plain
log-density problem is byte-for-byte a run without it.

To get any reparametrization you must supply, yourself:

* the **raw integer indices** into the unconstrained parameter vector of the
  coordinates to reparametrize — one entry per scalar coordinate, not per model
  parameter;
* for each, a **location** and a **log-scale**, either as constants or as
  closures over the whole parameter vector;
* an **AD backend**, used for the transform's own Jacobian. This page uses
  Enzyme, spelled as in the warning below.

On that last point: WarmupHMC differentiates a scalar objective in the *full*
parameter vector (see [`ReparametrizedProblem`](@ref)), which is the shape
reverse mode exists for — its cost is one pass regardless of dimension, while
forward mode costs one pass per input. That is an argument about **operation
counts**, not wall-clock. What the measurements say is a separate question, so it
is read off them rather than asserted here:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
b = load_results("capture_boxing.json")
boxed = Set(c["target"] for c in b["captures"] if c["boxed"])
rows = [r for r in load_results("annotation_sweep.json")["rows"]
        if r["ns_forwarddiff"] !== nothing && r["ns_const"] !== nothing]
wins = count(r -> Float64(r["ns_const"]) < Float64(r["ns_forwarddiff"]), rows)
nb = count(r -> r["target"] in boxed, rows)
Markdown.parse(
    (wins == length(rows) ?
     "Reverse mode is ahead on **every one of the $(length(rows)) swept rows**" :
     "**Reverse mode is ahead on only $(wins) of the $(length(rows)) swept " *
     "rows**") * ", and " *
    (nb == 0 ?
     "no swept spec captures a `Core.Box`, so no row is measuring that defect " *
     "in place of the backend." :
     "**$(nb) of them capture a `Core.Box`**, so those rows measure that defect " *
     "rather than the backend and the count above is not about reverse mode."))
```

How the advantage *scales* with dimension is a third question again, and it is
answered from the same rows under
[What the backend costs, measured](@ref) — which also holds both tables and the
A/B that isolates the capture defect. Reverse mode is a
reasonable default and what the examples below use; if the gradient is your
bottleneck, measure both on your own model rather than reasoning from dimension.

**The interface ships; the backend is yours to bring.**
`DifferentiationInterface` is a hard dependency of WarmupHMC, so
`ReparametrizedProblem` can always *talk* to a backend — but a backend object
only works once you have loaded the package behind it, and `AutoEnzyme` needs
`using Enzyme`. Nothing in `src/` ever constructs one: `ad_backend` is a field
you fill in.

!!! warning "A bare `AutoEnzyme()` does not work — it needs `function_annotation`"
    Pass it:

    ```julia
    AutoEnzyme(; function_annotation=Enzyme.Const)
    ```

    **`function_annotation=Enzyme.Const`** — without it, the run dies on the
    **first gradient**:

    ```
    EnzymeMutabilityException: Function argument passed to autodiff cannot be
    proven readonly.
    ```

    What gets differentiated is a closure capturing the frozen inner gradient
    `g_y` and the reparametrizer (see [`ReparametrizedProblem`](@ref)), and
    Enzyme will not assume on its own that captured state holds no derivative
    data. `Const` states what is already true here: `g_y` is frozen by
    construction — a constant of the differentiation, not a function of `x`.
    **Do not take Enzyme's own suggestion of `Enzyme.Duplicated`**; it computes
    the same answer, but it allocates a shadow copy of the closure on every
    call. That hint diagnoses the problem; it is not the fix. How much the
    shadow copy costs depends strongly on the target, and the per-target figures
    are tabulated under [What the backend costs, measured](@ref) rather than
    summarised into one multiplier here — no single number generalizes. `Const`
    is the right annotation regardless, because it is the *correct* one.

!!! note "`set_runtime_activity` used to be required as well — it no longer is"
    There is a *second* call site on your backend: the joint halo transport
    described under [What warm-up actually does](@ref). It ran a closure Enzyme
    could not statically prove, so a plain `Const` got past construction and past
    the first gradient and then died at the **first restarting window** with
    `EnzymeRuntimeActivityError` — partway into a run, rather than at setup.

    The objective was made statically provable, so the workaround is no longer
    needed and this page no longer recommends it. Passing
    `mode=Enzyme.set_runtime_activity(Enzyme.Reverse)` anyway is harmless — it
    measured free — so an existing script that carries it does not need editing.

    Mentioned because the failure was in a released state of the docs, and
    because it is the shape to expect if a future change adds a third call site:
    a backend that works for hundreds of gradients and then throws is a
    *coverage* problem, not a user error. `web/src/test/enzyme.jl` exists to catch
    exactly that and now pins this site with a plain `Const`. It is tagged
    `:enzyme` and skipped by the main matrix, so run it with `--tag=enzyme`.

!!! note "WarmupHMC does not depend on ForwardDiff, and does not want to"
    `Project.toml` has no ForwardDiff entry — not a direct dependency, and there
    is no `[weakdeps]` section for it to hide in either. ForwardDiff reaches the
    environment only transitively, through Pathfinder, and warm-up *actively
    defuses* it — the Pathfinder initialization passes `adtype=NoAD()` so
    Pathfinder's forward-mode default cannot call your `logdensity` with
    `Dual`-valued parameters, which a natively-backed target such as a BridgeStan
    model could not accept. If you find `AutoForwardDiff()` named in a test or a
    benchmark here, that is a harness pinned to its own frozen baselines, not a
    recommendation — the `ADBackend` snippet in `web/src/test/setup.jl` says so at
    the point of use.

## What the backend costs, measured

Everything above about reverse mode is an operation count. The numbers below are
the measurements, and they are **generated from the checked-in result files at
docs-build time** — `docs/benchmark/` holds the harnesses, `docs/benchmark/results/`
the JSON they write, and this page reads that JSON rather than restating it. No
figure here is typed by hand, and if a result file goes missing or changes shape
the docs build fails instead of rendering a stale table.

Per wrapped gradient, with the backend order rotated between rounds. `spec` is
whether that target's accessor closures capture a `Core.Box` — read from the
boxing probe's own output, not asserted here:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
sweep = load_results("annotation_sweep.json")
boxed = Set(c["target"] for c in load_results("capture_boxing.json")["captures"] if c["boxed"])
ratio(a, b) = (a === nothing || b === nothing) ? "—" : num(Float64(a) / Float64(b); sig = 3) * "×"
md_table(
    ["target", "`d`", "`c`", "spec", "`Const` ns/grad",
     "`Duplicated` ÷ `Const`", "`Const` ÷ ForwardDiff"],
    [["`" * r["target"] * "`", r["dim"], num(r["c_source"]),
      r["target"] in boxed ? "**boxed**" : "clean", num(r["ns_const"]),
      ratio(r["ns_duplicated"], r["ns_const"]),
      ratio(r["ns_const"], r["ns_forwarddiff"])] for r in sweep["rows"]])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
Markdown.parse("*" * provenance(load_results("annotation_sweep.json");
                                harness = "docs/benchmark/annotation_sweep.jl") * "*")
```

Two things to read off that table, and one not to. Both of the two are counted
from the rows rather than by eye, so that a re-measurement which breaks either
one says so here instead of leaving the sentence standing:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
rows = load_results("annotation_sweep.json")["rows"]
pair = [r for r in rows if r["ns_duplicated"] !== nothing && r["ns_const"] !== nothing]
fd = [r for r in rows if r["ns_forwarddiff"] !== nothing && r["ns_const"] !== nothing]
ndup = count(r -> Float64(r["ns_duplicated"]) > Float64(r["ns_const"]), pair)
nlt = count(r -> Float64(r["ns_const"]) < Float64(r["ns_forwarddiff"]), fd)
Markdown.parse(
    (ndup == length(pair) ?
     "`Duplicated` costs more than `Const` on all $(length(pair)) rows — the " *
     "shadow copy is real work." :
     "**`Duplicated` costs more than `Const` on only $(ndup) of $(length(pair)) " *
     "rows**, so the shadow copy is not uniformly the more expensive annotation " *
     "here.") * " " *
    (nlt == length(fd) ?
     "The last column is below `1×` on all $(length(fd)) of them." :
     "**The last column is below `1×` on only $(nlt) of $(length(fd)) rows.**"))
```

What must *not* be read off it is a trend in `d`. Which target a row is may
matter more than how large that target is, and separating the two takes more
than a glance down the column — so it is deferred to a measurement below rather
than settled here.

That last point is easier to see than to say. The same column, plotted against
`d` — every point below the dashed parity line, and the vertical spread at a
single `d` as large as the spread across all of them:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("plots.jl")
vega_figure(
    backend_ratio_spec(load_results("annotation_sweep.json")["rows"]);
    caption = "Below 1× is Enzyme reverse mode winning. Colour is the target, " *
              "shape is the source-scale c; hover a point for its absolute " *
              "ns/grad. Same rows as the table above — the figure is built " *
              "from that JSON at docs-build time, not stored beside it.")
```

That reading is different from the one this page used to carry, and the reason
is in the `spec` column. Earlier revisions compared a boxed *shipped* spec
against a de-boxed rebuild, and had to warn that the boxed rows measured the
defect rather than the backend, with dimension perfectly confounded because
every boxed spec was also one of the larger models. The shipped specs no longer
capture a `Core.Box`. The A/B therefore builds **both** controls locally and
compares each against the shipped spec's own gradients, so what it measures is
what de-boxing recovered rather than how the shipped spec happens to be written.

Two checks in that harness have different strengths, and the difference matters
when reading the table below. The boxing sweep is a hard guard: it exits
non-zero and names the offenders if any shipped closure captures a `Core.Box`.
The control-identity check only warns, and the harness writes its JSON and exits
zero either way — so controls could drift and leave a table that is no longer an
A/B, with nothing failing. That claim is therefore derived below from the
recorded differences rather than stated here:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Statistics
b = load_results("capture_boxing.json")
m(k) = Statistics.median(Float64.(b["timings_ns"][k]))
md_table(
    ["`" * b["ab_target"] * "`", "Enzyme `Const`", "ForwardDiff", "Enzyme ÷ ForwardDiff"],
    [["boxed control", num(m("boxed/const")), num(m("boxed/fd")),
      num(m("boxed/const") / m("boxed/fd"); sig = 3) * "×"],
     ["unboxed control", num(m("unboxed/const")), num(m("unboxed/fd")),
      num(m("unboxed/const") / m("unboxed/fd"); sig = 3) * "×"]])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
b = load_results("capture_boxing.json")
ds = Float64.([b["ab_max_grad_diff_boxed"], b["ab_max_grad_diff_unboxed"]])
agree = all(iszero, ds) ?
    "Both controls' gradients are bit-identical to the shipped spec's, so the " *
    "difference above is pure overhead" :
    "**The controls differ from the shipped spec by up to " *
    "$(num(maximum(ds); sig = 2)), so the table above is not an A/B**"
ncap = length(b["captures"])
ntgt = length(unique(c["target"] for c in b["captures"]))
guard = isempty(b["boxed_specs"]) ?
    "None of the $(ncap) closure arguments probed across $(ntgt) targets " *
    "captures a `Core.Box`" :
    "**$(length(b["boxed_specs"])) of $(ncap) probed closure arguments still " *
    "capture a `Core.Box`**"
Markdown.parse("*" * provenance(b; harness = "docs/benchmark/capture_boxing.jl") *
               " $(agree). $(guard).*")
```

The ranking inverts between the two controls. Whether reverse mode's advantage
grows, holds or shrinks with dimension was left open here until the sweep could
be re-run against the fixed specs. It has been, so the question can now be put
to the table above rather than deferred — and the answer is computed from it:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
rows = [r for r in load_results("annotation_sweep.json")["rows"]
        if r["ns_forwarddiff"] !== nothing && r["ns_const"] !== nothing]
ratio(r) = Float64(r["ns_const"]) / Float64(r["ns_forwarddiff"])
lo, hi = extrema(ratio.(rows))
byd = Dict{Any,Vector{Float64}}()
for r in rows
    push!(get!(byd, r["dim"], Float64[]), ratio(r))
end
spreads = [(d, maximum(v) - minimum(v)) for (d, v) in byd if length(v) > 1]
wd, wspan = isempty(spreads) ? (nothing, 0.0) : argmax(last, spreads)
share = (hi > lo && wd !== nothing) ? 100 * wspan / (hi - lo) : 0.0
Markdown.parse(
    "Across all $(length(rows)) rows the ratio spans `$(num(lo; sig = 3))×`–" *
    "`$(num(hi; sig = 3))×`. The widest spread at any *single* dimension is at " *
    "`d = $(wd)`, which alone accounts for $(num(share; sig = 3))% of it. " *
    (share >= 80 ?
     "Dimension is therefore not what this column varies with — which target it " *
     "is dominates how large that target is." :
     "Dimension may therefore carry part of the variation, though this sweep is " *
     "too small to separate it from target identity."))
```

### Where those gradients were taken

Both tables above evaluate at `randn(d)`. A sampler does not go there, so the
last thing to check is whether the ratio survives being measured where the
sampler actually went. Re-taking it at sampler-visited positions in the source
frame:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
t = load_results("typical_positions.json")
pct(r) = (s = (Float64(r["const_over_fd_typical"]) / Float64(r["const_over_fd_randn"]) - 1) * 100;
          string(s >= 0 ? "+" : "−", num(abs(s); sig = 2), "%"))
md_table(
    ["target", "`d`", "at `randn(d)`", "at visited positions", "shift"],
    [["`" * r["target"] * "`", r["dim"],
      num(r["const_over_fd_randn"]; sig = 4) * "×",
      num(r["const_over_fd_typical"]; sig = 4) * "×",
      pct(r)] for r in t["rows"]])
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
t = load_results("typical_positions.json")
shifts = [abs(Float64(r["const_over_fd_typical"]) / Float64(r["const_over_fd_randn"]) - 1)
          for r in t["rows"]]
np = get(t["rows"][1], "n_positions", nothing)
Markdown.parse("*" * provenance(t; harness = "docs/benchmark/typical_positions.jl") *
               (np === nothing ? "" : " $(np) positions per target.") *
               " Largest shift $(num(maximum(shifts) * 100; sig = 2))%, " *
               "smallest $(num(minimum(shifts) * 100; sig = 2))%.*")
```

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
t = load_results("typical_positions.json")
crossed = [r["target"] for r in t["rows"]
           if (Float64(r["const_over_fd_randn"]) - 1) *
              (Float64(r["const_over_fd_typical"]) - 1) < 0]
Markdown.parse(isempty(crossed) ?
    "No row crosses `1×`, so which backend is ahead is not an artefact of the " *
    "sampling distribution. Whether the *size* of the gap moves with it is a " *
    "separate question, and one the medians alone cannot answer — see below." :
    "**$(length(crossed)) row(s) cross `1×`** — " *
    join("`" .* crossed .* "`", ", ") * " — so on those targets which backend " *
    "is ahead depends on where the gradient is taken.")
```

A single cross-target number for "how much faster" would therefore be the wrong
thing to quote from this page: the per-target ratios differ far too much for one
figure to stand in for them. That a gradient benchmark's answer *can* depend on
where the gradient is taken is a real methodological hazard, and it is why the
second column exists at all — but whether it does so measurably on these targets
is settled below rather than asserted here.

!!! warning "The per-target `shift` column is not a result"
    Read the shifts as *unresolved*, not as small findings — and the harness now
    carries the evidence for that rather than leaving it as a caution. Each cell
    in the two ratio columns is a median over `rounds` repeats, and **those
    repeats are stored raw** beside it, under `samples` in
    `typical_positions.json`. Within one column those repeats are paired by
    construction — `typical_positions.jl` binds the position set once and times
    every backend on it inside each round, rotating which goes first — so a
    per-round ratio compares like with like, and the count below is a count of
    paired comparisons. Across the two columns they are not: those are separate
    timing loops over different positions, which is why only their ranges are
    compared below. What they support is computed below rather than
    summarised here. Note what five repeats can and cannot settle: whether two
    ranges overlap, yes — how *likely* a difference is, no. Nothing here should
    be read as a significance claim.

    One thing the repeats cannot check at all is drift of the medians between
    runs. Re-running the identical harness against identical code moved
    `funnel`'s typical-position ratio from `0.379×` to `0.649×`, about 71% — a
    comparison **between two runs**, so it cannot be derived from the single file
    checked in here and is recorded as prose on purpose.

    That is not a hypothetical worry about clocks in general either. A companion
    end-to-end measurement timed two **code-identical** revisions with
    **bit-identical** trajectories — no ESS, gradient count or final centering
    moved — and its wall-clock ratios still differed by up to tens of percent,
    with one changing sign. That was a different quantity from this page's
    per-gradient microbenchmark and its band does not transfer numerically, so
    treat it as a reason for caution about *small* clock differences here, not
    as a measured error bar for this table.

**Two different questions hang on this table** — which backend is ahead, and
whether the per-target shifts mean anything — and they have different answers.
An earlier version of this page decided both with one margin heuristic, which
could only ever move them together; the stored repeats answer each on its own
terms, so they are now asked separately:

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
t = load_results("typical_positions.json")
ratios(r, k) = Float64.(r[k]["samples"]["const"]) ./ Float64.(r[k]["samples"]["fd"])
both(r) = vcat(ratios(r, "typical"), ratios(r, "randn"))
all_r = reduce(vcat, both(r) for r in t["rows"])
n, below = length(all_r), count(<(1), all_r)
overlaps(r) = (a = extrema(ratios(r, "typical")); b = extrema(ratios(r, "randn"));
               a[1] <= b[2] && b[1] <= a[2])
sep = [String(r["target"]) for r in t["rows"] if !overlaps(r)]
Markdown.parse("*" *
    (below == n ?
     "Every one of the $(n) individual per-round ratios is below `1×`, the " *
     "closest to parity being $(num(maximum(all_r); sig = 3))× — so which " *
     "backend is ahead does not rest on the medians at all, since no single " *
     "repeat crosses over." :
     below == 0 ?
     "**Every one of the $(n) per-round ratios is above `1×`**, so the direction " *
     "stated above is backwards." :
     "**$(n - below) of the $(n) per-round ratios fall" *
     (n - below == 1 ? "s" : "") * " on the other side of `1×`**, so which " *
     "backend is ahead is not settled by these repeats.") * " " *
    (isempty(sep) ?
     "The two columns' round ranges overlap on every target, so **not one of the " *
     "shifts is separable** from repeat-to-repeat variation — the large ones " *
     "included." :
     "The round ranges fail to overlap on " * join("`" .* sep .* "`", ", ") *
     ", so $(length(sep)) of $(length(t["rows"])) shifts are separable from " *
     "repeat-to-repeat variation; the rest are not.") * "*")
```

A gradient *count* elsewhere in this manual carries weight these timings do not:
a count is provenance, a clock is a measurement of the machine that took it.

## A complete worked example

Neal's funnel — `v ~ Normal(0, 3)`, `xᵢ ~ Normal(0, exp(v/2))` — is the smallest
model where this matters. Coordinate `1` is `v`; coordinates `2:6` are the `xᵢ`,
each with location `0` and log-scale `v/2`.

```julia
using WarmupHMC, LogDensityProblems, Random
using Enzyme                                       # backs AutoEnzyme
using DifferentiationInterface: AutoEnzyme

struct Funnel
    k::Int
end
LogDensityProblems.dimension(f::Funnel) = f.k + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(f::Funnel, x) = begin
    v = x[1]; xs = @view x[2:end]
    -0.5 * v^2 / 9 - sum(xi -> 0.5 * xi^2 * exp(-v) + v / 2, xs)
end
LogDensityProblems.logdensity_and_gradient(f::Funnel, x) = begin
    v = x[1]; xs = @view x[2:end]
    g = similar(x)
    g[1] = -v / 9 + 0.5 * exp(-v) * sum(abs2, xs) - f.k / 2
    g[2:end] .= .-xs .* exp(-v)
    (LogDensityProblems.logdensity(f, x), g)
end

k = 5
funnel = Funnel(k)

# One `Reparametrization` per reparametrized COORDINATE. `target` is the
# parametrization the model is written in and never moves; `source` is what the
# sampler works in and is what warm-up re-fits. Starting them equal means
# "start where the model is written, and adapt from there".
ir = IndexedReparametrization([
    i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
                           0., x -> x[1] / 2)
    for i in 2:(k + 1)
])

rp = ReparametrizedProblem(ir, funnel,
    AutoEnzyme(; function_annotation=Enzyme.Const))
result = adaptive_warmup_mcmc(Xoshiro(20260728), rp; n_draws=1000, progress=nothing)
```

The fitted centerings live on the `IndexedReparametrization` you passed in — the
single-chain method mutates it in place — so read them back off `ir`:

```julia
julia> [v.source.c for (_, v) in ir.pairs]
5-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
```

All five coordinates were driven from `1.0` to `0.0`: warm-up found the
non-centered parametrization of the funnel on its own, without being told that
the funnel is the model it was looking at.

Reading the result back off your own object is a *single-chain* affordance. The
multi-chain method gives each chain its own `deepcopy`, so the object you built
stays exactly as you built it and each chain's fitted centerings live on its own
copy — see "What warm-up actually does" below.

`result.posterior_position` is `6 × 1000` and is in the **model's own**
parametrization — warm-up applies the fitted transform to the draws before
returning them, so nothing downstream has to know a reparametrization happened.

!!! note "Provenance of the figures above"
    They were produced by the example exactly as written, first at `ba6b4f0`,
    re-verified unchanged at `b2ff221`, and re-verified unchanged again at
    `2760400`. Each re-run earns its cost only because `src/` genuinely moved in
    between: `aac6489` made the joint halo transport exact, and `4eb08c7` added
    online candidate scoring, rewriting both `Reparametrizations.jl` and
    `adaptive_warmup_mcmc.jl` — the two files this example calls. Those are the
    kind of change that *can* move adaptation behaviour, and these figures are
    only worth printing if somebody checked rather than assumed.

    The same run was also executed under
    `AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)`
    — the spelling this page used to require — and under
    `AutoEnzyme(; function_annotation=Enzyme.Duplicated)`, as controls rather
    than recommendations. All three produce a **byte-identical** `6 × 1000`
    draw matrix and the same `[0.0, 0.0, 0.0, 0.0, 0.0]` — re-checked at
    `2760400` by comparing the draw matrices directly, not just their shapes.
    What a gradient *costs* is what the backend choice decides; what it
    *returns* here is backend-independent.

    Adaptation behaviour is *not* fixed across commits, however: these same
    figures were materially different a few commits earlier. So read the chain
    above as a record of revisions somebody checked, **not as a currency
    claim** — a plain `julia` fence is not executed by the docs build, so these
    numbers ship unchanged whether or not they still reproduce. ([Runnable
    examples](@ref) is where which examples run and which do not is tracked;
    this one is listed there as not executed.)

    You do not have to re-run blindly to find out. Ask whether the code this
    example calls has moved since the last revision named above:

        julia --project=docs docs/benchmark/code_identical.jl <that revision> HEAD

    It names the files that differ and exits non-zero when any do. A clean
    answer is not proof the figures still reproduce — a dependency can move too
    — but a dirty one tells you *which* part of the sampler changed, and that is
    the cheap question to ask first. `docs/benchmark/artifact_currency.jl` asks
    it for every checked-in measurement behind the other pages at once.

The reparametrization is re-fitted only at warm-up windows that **restart**, and
a window restarts only while the marginal-scale condition number is at or above
`variance_cond_target` (default `2.0`). In the run above, windows 1 and 2 restart
(condition number `2.08`, then `3.44`) and windows 3 and 4 do not (`1.0`), so all
five centerings are already at `0.0` by the end of the *first* window and the
remaining windows sample at the parametrization that was found.

### How long a run does adaptation need?

**On this model, not a long one.** Re-running the example unchanged at
`n_draws=200`, and again at `n_draws=100`, gives the same two restarting windows
and the same final `[0.0, 0.0, 0.0, 0.0, 0.0]`. So the centerings here are settled
well inside a run short enough to use as a smoke test — you do not have to budget
a long run just to find out whether your spec does anything.

Do not read that as a guarantee about the method. It is a property of this
funnel: on a model whose condition number starts below `variance_cond_target`, no
window restarts and the centerings never move at all, however long you sample.
What generalizes is the *mechanism*, not the window count — so if you are
checking that your spec is wired up correctly, watch the boundaries rather than
assuming a re-fit happened:

```julia
adaptive_warmup_mcmc(rng, rp; n_draws=1000, progress=nothing,
    callback=(state, stage) -> begin
        @info "boundary" stage state.outer_counter state.restart state.variance_cond
        false
    end)
```

## Building a spec for a real model

For anything bigger than the funnel, the work is not the `Reparametrization`
objects — it is mapping model structure onto raw integer positions in the
unconstrained vector. `web/src/posteriordb_reparametrizations.jl` in this
repository does exactly that for a set of PosteriorDB posteriors, and is the
most honest picture of the current UX. Eight schools, where the offsets are
constants:

```julia
c = endswith(posterior_name, "noncentered") ? 0. : 1.
1:8 .=> Ref(Reparametrization(
    PartiallyCentered(c),
    PartiallyCentered(c),
    x->x[9],
    x->x[10]
))
```

and a partially-pooled radon model, where they are computed from the Stan data:

```julia
J = stan_jdata["J"]
c = endswith(posterior_name, "noncentered") ? 0. : 1.
(l, s, o) = (J+1, J+2, 0)
map(1:J) do i
    idx = o + i
    idx=>Reparametrization(
        PartiallyCentered(c),
        PartiallyCentered(c),
        x->x[l],
        x->x[s]
    )
end
```

Note what that file has to encode by hand for every posterior: the block of
coordinates, the offset it starts at, and the two positions holding its location
and log-scale. Get one offset wrong and you will reparametrize the wrong
coordinates against the wrong scale — quietly, because nothing validates that a
"log-scale" index really holds a log-scale. The whole file is worth reading
before writing a spec of your own; every branch returns something that
`IndexedReparametrization` accepts directly, and the `else` branch returns
`Nothing[]`, i.e. the no-op.

`target` and `source` are set to the same `c` in every branch there, and `c` is
picked to match how the Stan model is written (`0.` for the `_noncentered`
variants, `1.` otherwise). That is the general rule: **`target` must describe the
parametrization the wrapped log density actually expects.** It is never adapted,
and getting it wrong does not error — it silently samples a different model.

## What warm-up actually does

At the end of a warm-up window that restarts, and immediately before the linear
metric is re-selected:

1. Each reparametrized coordinate is scored against a fixed grid of **11**
   candidate centerings, `range(0, 1, 11)`.
2. Scoring runs over the recorded **halo** — the intermediate NUTS states kept
   during the window (one per trajectory, drawn from the exact marginal proposal
   probabilities over that trajectory's leaves, up to `recording_target` of
   them), not the accepted draws.
3. Each candidate accumulates, online, the covariance of (its transformed
   position, its transformed gradient); the candidate minimizing their
   correlation wins. A coordinate whose marginal is standard-normal has position
   and gradient exactly anti-correlated, so the minimum is the most
   standard-normal-looking candidate.
4. The winner replaces that coordinate's `source`, and the halo states for that
   coordinate are transported into the new parametrization before the next
   coordinate is scored.
5. A coordinate with 2 or fewer halo states keeps the centering it had.
6. Once every coordinate has settled, the **whole** halo is rebuilt in a single
   final pass: positions and gradients are recomputed from the *originals*
   through the composed old-to-new map, one AD pass per recorded state. The
   scoring step above mutates a working pool as the search walks the
   coordinates; this final pass is what the next window actually sees. The
   wrapped model is never re-evaluated — only the transform is differentiated.

The grid is a design choice, not an approximation of a continuous search.
Centering is a bounded, one-dimensional quantity, so the candidate set can be
*enumerated* — and because it is enumerable, every candidate can be scored in a
**single pass** over the halo with online accumulators. Each halo state is
visited once per coordinate no matter how many candidates there are, nothing has
to be stored for a second look, and there is no inner optimization loop with its
own convergence behaviour to reason about. Resolution finer than `0.1` is not
what decides how the sampler behaves. The grid size is fixed and no sampler
keyword exposes it.

All of that rewrites the reparametrization **in place**, so a multi-chain run
must not let two chains share one problem. It does not: every sampler gives chain
`i` its own `deepcopy` of the `lpdf` you pass. The two that adapt a
reparametrization at all are [`adaptive_warmup_mcmc`](@ref) and
[`cooperative_warmup_mcmc`](@ref) — [`clustered_warmup_mcmc`](@ref) has no
reparametrization hooks and does not accept `nonlinear_adapt` — and for those two
the chains adapt independently, chain `i` is the same run as that chain on its
own, and the object you constructed is never mutated. Pass
`lpdfs::AbstractArray` instead to control the per-chain problems yourself; that
method is used exactly as given, so

```julia
# independently built problems — same effect as the default, spelled out
adaptive_warmup_mcmc(rngs, [ReparametrizedProblem(build_ir(), problem, backend) for _ in rngs])

# one object every chain shares — deliberate opt-in, not the default
adaptive_warmup_mcmc(rngs, fill(lpdf, length(rngs)))
```

both do what they say.

## Constraints that bite

**Coordinate order is load-bearing across a checkpoint/resume.** A checkpoint
does not store your reparametrization; it stores the fitted `source` centerings
as a bare positional list. On resume they are zipped back onto the freshly
supplied problem's `pairs` **by position**, and the stored indices are not
consulted. So a rebuild that enumerates coordinates in a different order — a
`Dict`-driven build, a data-dependent sort — silently puts every centering on the
wrong coordinate. A different *length* is not caught gracefully either: it throws
`DimensionMismatch`, or, when the overlap collapses to a single entry, silently
overwrites every pair with that one. Build `pairs` deterministically, in the same
order and with the same length, on both sides of a resume.

**Only the scalar centerings survive a checkpoint.** Not the `target`s, not the
location/log-scale closures, not the wrapped problem. That is deliberate — a
BridgeStan model and a closure are not things you want in a `.jls` file — but it
means the `lpdf` you hand to a resume has to be rebuilt by you, exactly as you
built it the first time. Nothing checks that you did; the only validation on
resume is that the dimension matches.

**The transform is on the gradient hot path.** Every `logdensity_and_gradient`
call costs one inner gradient evaluation, one extra forward pass through the
transform, and one AD pass over it. Your location and log-scale accessors run
under AD on every one of those, so keep them cheap and type-generic — indexing,
arithmetic, `exp`/`log`; not `Float64`-annotated code, not anything that mutates.

**Three construction mistakes fail late rather than at construction.** Each builds
a perfectly valid-looking object and blows up further in:

* **An under-specified Enzyme backend.** A bare `AutoEnzyme()` constructs,
  `logdensity` works on the result, and the first `logdensity_and_gradient`
  throws `EnzymeMutabilityException`. Pass `function_annotation=Enzyme.Const`;
  see the warning at the top of this page. Until `aac6489` this one failed late
  *twice* — `Const` alone then died at the first restarting window with
  `EnzymeRuntimeActivityError` — which is why an older script may carry a
  `mode=` argument it no longer needs.
* **Omitting the AD backend.** `ReparametrizedProblem(r, p)` — the two-argument
  form — stores `ad_backend === nothing`. `logdensity` works fine on that object,
  so nothing looks wrong until the first `logdensity_and_gradient`, which hands
  `nothing` to `value_and_gradient` and `MethodError`s. Always pass a backend.
* **Integer centerings.** `PartiallyCentered(1)` type-parameterizes the pair
  vector on `Int`, and the fitted `Float64` centering cannot be written back into
  it: `MethodError: Cannot convert`. This does not fire at construction or on the
  first gradient — it fires at the **end of the first restarting warm-up window**,
  the first time adaptation writes a centering back. Write
  `PartiallyCentered(1.0)`.

**The per-chain `deepcopy` descends into the wrapped problem.** It has to — the
`ReparametrizedProblem` owns the `IndexedReparametrization` that warm-up rewrites,
and nothing can copy that without copying the struct holding it. What that costs
depends on what your inner problem is made of, and the two cases differ sharply:

* **Native handles are aliased, not duplicated.** A `StanProblem` holds raw
  pointers to one `bs_model_construct`ed model; `deepcopy` copies the pointers
  verbatim, so every chain's copy addresses that same native model and no chain
  reconstructs or re-`dlopen`s anything. Only the object you built carries the
  destructor, so the copies being collected does not invalidate it. It also means
  the chains still share the model's *native* state: running them with
  `parallel=true` needs a model compiled `make_args=["STAN_THREADS=true"]`, the
  same as it always did.
* **Julia-side data is genuinely copied.** An inner problem holding a large
  read-only array pays for that array once per chain, and one that cannot be
  `deepcopy`ed at all fails here rather than in your own code. Either is a reason
  to build the `lpdfs` vector yourself and share the parts you know are safe to
  share.

**`cooperative_warmup_mcmc` accepts `progress=` and drops it.** The keyword is on
the signature and passes keyword validation, but the top-level function never
forwards it to the chains it builds, so passing it has no effect. (The per-chain
constructor `WarmupHMC.cooperative_chain` does honour it.)

## See also

* [`ReparametrizedProblem`](@ref) — the wrapper, and the gradient contract.
* [`IndexedReparametrization`](@ref) — the container, and what mutates in place.
* [`Reparametrization`](@ref) — one coordinate's rule; `target` vs `source`.
* [`PartiallyCentered`](@ref) — the centering itself, and how it is fitted.
