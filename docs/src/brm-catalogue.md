# Nonlinear reparametrization on real catalogue posteriors

Every other benchmark in this documentation runs on a synthetic target or a
posteriordb model. This one runs on real regression posteriors taken from the
published
[BayesianRegressionModels catalogue](https://juliabayes.github.io/BayesianRegressionModels.jl/),
fitted to real data, and asks one question: **does turning on nonlinear
reparametrization buy anything?**

Every figure below is generated at docs-build time from
`docs/benchmark/results/brm_catalogue/rows.json`, which holds one object per run
plus the configuration that produced them and no stored summaries. The
summaries and the verdict are computed by `docs/benchmark/brm_catalogue.jl` from
those rows in the same build.

## The knob measures nothing unless the target carries a reparametrizer

This is the trap the benchmark is built around, and it is worth stating before
any number.

`nonlinear_adapt = true` **changes nothing at all** on a plain log-density
problem. `WarmupHMC.reparametrizer(::Any)` returns an empty
`IndexedReparametrization`, and `find_reparametrization!` short-circuits when
there are no pairs. There is no automatic detection of hierarchical structure —
a target has to be wrapped in something that knows its own blocks before the
flag reaches anything.

So a benchmark that varies this flag over a bare BridgeStan model produces a
complete, well-formed table of numbers that measures nothing whatsoever. That
configuration is not avoided here; it is run deliberately, as two of the three
arms, and those two are **controls**. Their flag-on and flag-off runs must come
back identical, seed for seed. If they ever diverge, the flag is reaching
something it should not be, and the third arm's result stops being attributable
to it.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_catalogue/rows.json")
c = brmc_controls(d)
bad = [x for x in c if x.identical != x.n]
# A failed control is a BUILD failure, not a rendered warning. If this page
# could render "controls failed" and still go green, the check would be
# decorative — the reader would be looking at a table whose headline result is
# no longer attributable to the knob it names, on a page that built fine. That
# is the same anti-check shape the section above is about.
isempty(bad) || error("""
    BRM catalogue controls FAILED on $(length(bad)) (model, arm) pair(s):
    $(join(["  $(x.spec)/$(x.arm): only $(x.identical) of $(x.n) seeds identical" for x in bad], "\n"))

    The noncentered and centered arms carry no reparametrizer, so nonlinear_adapt
    must not change their draws. That it did means either the arm acquired a
    reparametrizer or the flag now reaches something else — and until that is
    explained, the adaptive_centering result cannot be attributed to the flag.
    Re-measure; do not relax this check.
    """)
Markdown.parse(
    "**Controls hold.** All $(length(c)) (model, control-arm) pairs returned " *
    "identical gradient counts and identical min-ESS with the flag on and off, " *
    "across every seed. The flag is inert exactly where it should be, which is " *
    "what makes the adaptive-centering result below attributable to it.")
```

## What is being compared

Three arms per model, each sampled with the flag off and on:

  * **non-centered** — what `SBBRMI` emits by default; the `c = 0` endpoint.
  * **static centered** — `SBBRMI(...; centered_groups = ...)`; the `c = 1` endpoint.
  * **adaptive centering** — the non-centered model wrapped by
    `BayesianRegressionModels.adaptive_centering_problem`, which builds a
    `PartiallyCentered` reparametrization with per-`(block, term, group)`
    centeredness. This is the only arm with pairs, so it is the only arm the
    flag can reach.

### The models

Transcribed by hand from catalogue cards. Every card in that catalogue carries
`parseable = false`, so nothing here is machine-derived from it — the
`catalogue_formula` column is the card's own text, kept verbatim so the
transcription can be checked rather than trusted.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
d = load_results("brm_catalogue/rows.json")
ms = brmc_models(d)
# All three arms sample the SAME parameter space, and the table below would be
# misleading if that ever stopped being true — the reported-name counts differ
# between the two programs, so a reader could easily take those for parameter
# counts. Assert the invariant rather than inviting the inference.
bad = [m["spec"] for m in ms if m["dim_noncentered"] != m["dim_centered"]]
isempty(bad) || error("centered and non-centered dimensions differ on: " * join(bad, ", "))
md_table(["model", "dataset", "source", "catalogue formula", "n obs",
          "parameters (both)", "reported names (nc / c / shared)"],
         [[m["spec"], m["dataset"], m["catalogue_source"],
           "`" * m["catalogue_formula"] * "`", m["n_obs"], m["dim_noncentered"],
           string(m["n_names_noncentered"], " / ", m["n_names_centered"],
                  " / ", m["n_names_shared"])] for m in ms])
```

### Why ESS is reduced over shared names only

**All three arms sample the same number of parameters.** That is the
`parameters (both)` column, it is asserted at build time, and it is worth
stating plainly because the next column looks like it says otherwise. On
sleepstudy every arm has 42 unconstrained parameters (45 constrained); the
adaptive arm is a reparametrization of the non-centered one, so it has 42 too.

What differs is which quantities each program *reports*. The two are different
Stan programs: the non-centered one samples `z_flat` innovations and derives the
random effects from them, the centered one samples the random effects directly.
Each therefore exposes intermediates the other has no name for — on sleepstudy
the non-centered program adds `r_mu_Subject_z_flat` and `r_mu_Subject_z` (36
each), the centered one adds `r_mu_Subject_bm` (36). That, and only that, is the
657 / 621 gap. A minimum taken over each model's own reported names would
compare different quantities and quietly favour whichever program reports fewer
awkward ones.

Everything below reduces over the **intersection**, the third number in that
column. The intersection is not a leftover: on sleepstudy its 585 names include
all 36 `r_mu_Subject_b` — **the random effects themselves**, the coordinates
centering is about, which both programs report under the same name even though
one calls them a parameter and the other a transformed parameter. What gets
dropped is exactly the parametrization-specific scaffolding. Where the three
numbers agree the two programs report the same set and the restriction costs
nothing, which is the case for every model here except sleepstudy.

Structurally-constant coordinates are dropped before the minimum is taken. A
correlation Cholesky has fixed entries (`L[1,1] ≡ 1`, `L[1,2] ≡ 0`);
`MCMCDiagnosticTools.ess` returns `NaN` for a constant chain and `minimum`
propagates it, so one such coordinate would turn a perfectly healthy run's
min-ESS into `NaN` — indistinguishable from a failure. The count that was
dropped is recorded per run, so "constant" cannot be read as "broken".

## Does the flag help?

Paired by seed on the one arm that carries a reparametrization: same seed, same
target, one knob moved.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_catalogue/rows.json")
p = brmc_paired(d)
md_table(["model", "seeds", "median ESS/grad ratio (on ÷ off)", "range", "seeds improved"],
         [[x.spec, x.n, Printf.@sprintf("%.2f×", x.med),
           Printf.@sprintf("%.2f–%.2f×", x.lo, x.hi),
           "$(x.improved)/$(x.n)"] for x in p])
```

## Where adaptive centering lands between the two endpoints

A ratio against the non-centered default is only half the story: the static
centered parametrisation is available for free and is often better. This is the
comparison that matters.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_catalogue/rows.json")
g = brmc_gap(d)
f(x) = Printf.@sprintf("%.4g", x)
md_table(["model", "non-centered", "static centered", "adaptive + nonlinear", "gap recovered"],
         [[x.spec, f(x.noncentered), f(x.centered), f(x.adaptive),
           x.recovered === nothing ? "n/a — centered does not lead" :
             Printf.@sprintf("%.0f%%", 100 * x.recovered)] for x in g])
```

## Verdict

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Markdown
d = load_results("brm_catalogue/rows.json")
v = brmc_verdict(d)
fails = brmc_failures(d)
Markdown.parse(
  "Across **$(v.n_specs)** catalogue posteriors, the nonlinear flag improved " *
  "median ESS-per-gradient on **$(v.helped)** and reduced it on **$(v.hurt)**. " *
  "Static centering beat the non-centered default on **$(v.centered_wins)** of " *
  "**$(v.n_gaps)**, and adaptive centering with the flag on beat static " *
  "centering on **$(v.beat_centered)** of **$(v.n_gaps)**. " *
  (v.controls_clean ? "The controls held throughout. " :
                      "**The controls did not hold — treat the above as unattributed.** ") *
  (isempty(fails) ? "No run errored." : "**$(length(fails))** run(s) errored."))
```

## Cost is a separate result, and stays separate

The numbers above are per **gradient evaluation**. That is the portable
quantity, and on this repository's other pages it is also the one that
reproduces. It is not a wall-clock claim, and the two must not be merged into
one sentence: the de-boxing work measured elsewhere in this documentation is a
cost result that lives on the gradient path, while this is a sampling result.

The wrapped arm pays a real and large per-gradient cost, which the table below
reports rather than buries. A reader deciding whether to use adaptive centering
needs both halves, and they point in different directions on targets this small.

That cost is **not** WarmupHMC's transform machinery. Swapping BRM's
location/log-scale accessors for trivial `x -> x[i]` ones — same
`IndexedReparametrization`, same coordinates, same pair count, same Enzyme
backend, same [`WarmupHMC._logdensity_and_gradient_reparam`](@ref) — puts the
wrapped gradient back in the same order of magnitude as the non-centered arm's.
The per-pair loop, the accessor dispatch, the scatter-write and the frozen-`g_y`
objective are all cheap. What is expensive is reverse-mode AD over the accessor
closures BRM supplies, and that is tracked as a snag on
BayesianRegressionModels, not as work here. Two things it is *not*: the
accessors are type-stable, and preparing the DifferentiationInterface gradient
changes neither the time nor the allocations materially.

So the cost column below is a property of the accessors a caller plugs in, and
should not be read as the price of nonlinear reparametrization in general.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
import Printf
d = load_results("brm_catalogue/rows.json")
s = brmc_summary(d)
rows = []
for m in brmc_models(d)
    for arm in brmc_arm_order()
        e = [x for x in s if x.spec == m["spec"] && x.arm == arm && x.adapt]
        isempty(e) && continue
        push!(rows, [m["spec"], brmc_arm_label(arm),
                     Printf.@sprintf("%.2f s", e[1].wall),
                     string(round(Int, e[1].grad))])
    end
end
md_table(["model", "arm", "median wall-clock", "median gradient evaluations"], rows)
```

## What could not be reached, and why

A benchmark that reports only the models it managed to run reads as a survey of
the corpus when it is a survey of the easy part of it. The catalogue holds 359
cards over 178 datasets; what follows is the set that was considered for this
page and excluded, with the reason.

Note that three of these are tagged `gaussian` **by the catalogue and are not
gaussian**. Selecting models by the family tag rather than by reading the
formula would have pulled a binomial, a count and a censored-survival likelihood
into a gaussian benchmark.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
load_harness("brm_catalogue.jl")
d = load_results("brm_catalogue/rows.json")
md_table(["dataset", "catalogue formula", "why not reached"],
         [[u["dataset"], "`" * u["formula"] * "`", u["reason"]]
          for u in d["config"]["unreached"]])
```

Beyond these, the binding constraint on the corpus is not data availability but
**parseability**: all 359 cards carry `parseable = false` and `sampleable =
false`, so every model on this page had to be transcribed by hand. That is the
reason this page covers eight posteriors and not eighty.

## Reproducing

`BayesianRegressionModels` and `StanBlocks` are **not registered**. The
configuration block records the exact commits both were at, which is the only
pin available — a reader cannot resolve them from a registry, and this page does
not pretend a `Pkg.instantiate` would reproduce the run.

The documentation build needs none of that. It reads the checked-in JSON, like
every other page here; `docs/Project.toml` carries no BRM, no StanBlocks and no
BridgeStan.

**The BLAS thread count below is part of the pin, not incidental.** It changes
the sampled chain: the `centered` sleepstudy arm run at this machine's default
of 8 threads returns different per-seed gradient counts and a different min-ESS
than the same seed at 1 thread, and pinned to 1 it reproduces exactly. A rerun
at a different setting is not comparable to this one. The asymmetry — the other
two arms are bit-identical across both settings — is recorded but not explained.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
import Markdown
d = load_results("brm_catalogue/rows.json")
c = d["config"]
Markdown.parse("```\n" * c["reproduction"] * "\n```\n\n" *
    "* WarmupHMC `" * c["warmuphmc_sha"][1:10] * "`\n" *
    "* BayesianRegressionModels `" * c["brm_sha"][1:10] * "` (unregistered)\n" *
    "* StanBlocks `" * c["stanblocks_sha"][1:10] * "` (unregistered)\n" *
    "* host `" * c["host"] * "`, Julia " * c["julia"] * ", BLAS threads " *
    string(c["blas_threads"]) * "\n")
```
