# Harness for the nonlinear-reparametrization benchmark.
#
# The question this exists to answer: on a hierarchical target written in its
# CENTERED parametrization, does WarmupHMC's adaptive partial centering reach
# noncentered-like sampling performance without being told to?
#
# Every arm below samples the SAME Stan model, so every arm's draws live in the
# same coordinates and their ESS is directly comparable. The arms differ only in
# what the sampler is allowed to do with the partial-centering parameter `c`.
#
#   plain              bare StanProblem, no ReparametrizedProblem at all.
#                      This is what a user gets today.
#   fixed_centered     wrapped, source c pinned to the model's own value,
#                      nonlinear_adapt=false. Exact no-op transform, so it
#                      isolates the cost of merely carrying the wrapper.
#   fixed_noncentered  wrapped, source c pinned to the opposite endpoint,
#                      nonlinear_adapt=false. The hand-written noncentered
#                      parametrization, reached by transform.
#   adaptive           wrapped, source c starts at the model's own value,
#                      nonlinear_adapt=true. The method under test.
#
# `sibling` rows additionally sample posteriordb's separately hand-written
# noncentered model. Those draws are in a DIFFERENT unconstrained frame, so only
# their constrained-space ESS is comparable with the rest.
#
# Facts about the sampler that this harness depends on, and where they come from:
#
#   - `Reparametrization(target, source, args...)` — field 1 is the model's own
#     parametrization, field 2 is the sampler's. Verified by `fieldnames`; the
#     direction is pinned by web/src/test/reparametrize_direction.jl.
#   - `PartiallyCentered(1.0)` is CENTERED, `PartiallyCentered(0.0)` is
#     NONCENTERED. Same test, plus web/src/posteriordb_reparametrizations.jl
#     assigning c=1 to every `*_centered` posteriordb model and c=0 to every
#     `*_noncentered` one.
#   - `nonlinear_adapt=false` freezes the source centering but still
#     back-transforms at finalization, so EVERY arm's draws come back in the
#     model's own frame and this harness transforms nothing itself. That is true
#     from `b109210` ("report draws in the model frame even when
#     nonlinear_adapt=false"); before it, the fixed arms came back in the source
#     frame and this file compensated with one `reparametrize!`. Re-measured on
#     this base — a centered funnel sampled through a noncentered source returns
#     coordinate 1 at mean -0.006, sd 2.934 against the known `Normal(0, 3)`
#     marginal, and applying `reparametrize!` on top inflates the leg sds to
#     100-300. If you are pinning an older WarmupHMC, put the compensation back.
#   - `result.ess` is all zeros unless `monitor_ess=true`; this harness never
#     reads it and computes ESS from the draws instead.
#   - `n_draws` is a floor, so every rate is normalized by the actual draw count.
#
# None of the above is documented in `warmuphmc-use`; the gap is filed as snag
# `warmuphmc-use-ha-f74c9962` on WarmupHMC.

using LinearAlgebra
BLAS.set_num_threads(1)   # required for run-to-run reproducibility at a fixed seed

using WarmupHMC, PosteriorDB, LogDensityProblems, StanLogDensityProblems, BridgeStan
using DifferentiationInterface, ForwardDiff, Enzyme

"""
    bench_ad_backend()

The DifferentiationInterface backend the `ReparametrizedProblem` wrapper
differentiates the transform with, from `WHMC_BENCH_AD`.

Reverse mode is the rule. The objective is `x -> ljac(x) + dot(g_y, y(x))` —
scalar in the *full* parameter vector, with the inner gradient `g_y` frozen — so
forward mode costs `ceil(d / chunksize)` sweeps of the transform per gradient
where reverse mode costs one, and the gap opens exactly on the high-dimensional
hierarchical models this benchmark targets.

`forwarddiff` is kept ONLY so the two can be measured against each other on one
base. It is not a supported configuration; it is the control that shows how much
of the wrapper's cost was the backend rather than the method. Note the backend
package must be loaded for the ADTypes object to work at all, which is why both
are `using`ed above.

# Why `function_annotation = Enzyme.Const`, and not the bare `AutoEnzyme()`

**A bare `AutoEnzyme()` does not work here.** It raises

    EnzymeMutabilityException: Function argument passed to autodiff cannot be
    proven readonly

because the differentiated objective is a *closure* over the
`ReparametrizedProblem` — capturing the reparametrizer and the frozen inner
gradient `g_y` — and Enzyme cannot prove that captured state is only read.

`Const` says the closure carries no derivative information, which is exactly
true: `g_y` is held fixed by construction and the reparametrizer's parameters
are not what we are differentiating with respect to. Measured on the funnel,
2000 gradients:

| backend | ns/gradient |
|---|---|
| `AutoForwardDiff()` | 931 |
| `AutoEnzyme(; function_annotation = Enzyme.Const)` | **545** |
| `AutoEnzyme(; function_annotation = Enzyme.Duplicated)` | 4786 |

All three agree to 3.3e-16, so this is purely a cost choice — but note that
**Enzyme's own error message suggests `Duplicated`**, which is 5.1× slower than
the ForwardDiff it was meant to replace. `Duplicated` allocates and propagates a
shadow copy of the closure on every call; `Const` does not. Take the hint as a
diagnosis of the problem, not as the fix.
"""
function bench_ad_backend()
    name = lowercase(get(ENV, "WHMC_BENCH_AD", "enzyme"))
    name == "enzyme"      && return AutoEnzyme(; function_annotation = Enzyme.Const)
    name == "forwarddiff" && return AutoForwardDiff()
    error("WHMC_BENCH_AD must be \"enzyme\" or \"forwarddiff\", got $(repr(name))")
end

const AD_BACKEND = bench_ad_backend()
const AD_BACKEND_NAME = lowercase(get(ENV, "WHMC_BENCH_AD", "enzyme"))

using WarmupHMC: ReparametrizedProblem, IndexedReparametrization, PartiallyCentered,
                 Reparametrization
using Random, Statistics, Printf
import MCMCDiagnosticTools
import JSON

const BENCH_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCH_DIR, "..", ".."))

"""
    git_provenance() -> Dict{String,Any}

The three fields every artifact under `results/` must carry: which revision was
measured, whether the whole tree was clean, and whether **`src/`** was clean.

Merge it into a harness's provenance Dict rather than re-deriving it. Six probe
scripts used to inline `readchomp(\\`git rev-parse HEAD\\`)` with no `try` and no
dirty flag at all, which is how eight artifacts came to record a base measured
from a dirty tree with only two of them able to say so.

# Why `src_dirty` exists when `worktree_dirty` already does

They answer different questions and only one of them is the gate's subject.
`worktree_dirty` is whole-tree, which is what a human reading a provenance
header wants. But `artifact_currency.jl` compares **`src/`** at the recorded
base against `src/` at the tip, so a dirty `docs/` — which is the normal state
of a session that is writing up the measurement it just took — turns the gate
red for a reason that cannot change what sampler ran. A gate red for the wrong
reason is a gate that gets muted rather than fixed.

The two are not independent, and the dependence is what makes this cheap:
a clean whole tree IMPLIES a clean `src/`, so `worktree_dirty: false` on an
older artifact is already strictly stronger than `src_dirty: false` and needs no
re-measurement to stay green.

# Why `missing` rather than `false` on failure

`git` being unreachable is not evidence of cleanliness. It serialises to JSON
`null`, which a reader must distinguish from both `false` and absent: absent
means the harness predates the field, `null` means the harness asked and could
not find out. Neither is "it was clean", and defaulting either to `false` is the
same silent-reassurance failure as a `git fetch` that no-ops and exits 0.

# Why `--untracked-files=no`

Load-bearing, not tidiness. These harnesses WRITE their results into the repo,
so a bare `--porcelain` counts the previous run's untracked output directory and
reports every run after the first as dirty — by construction, with the tracked
source byte-identical. That fires exactly when two runs are being compared,
which is the one time the flag has to mean anything.

# CALL THIS BEFORE OPENING THE OUTPUT FILE, NEVER INSIDE THE WRITE BLOCK

The natural way to write a result is `open(path, "w") do io; JSON.print(io,
Dict(..., git_provenance()..., ...)); end`, and it is wrong. `open(path, "w")`
truncates at open, so once `path` is TRACKED the provenance call inside that
block sees its own half-written output and records `worktree_dirty = true`.
Measured directly: `git status --porcelain --untracked-files=no -- <path>`
prints nothing immediately before the `open` and ` M <path>` from inside it.

Three properties make this worth a section rather than a line:

* It fails toward a strong FALSE claim. `docs/tables.jl:provenance` renders the
  flag as "**Recorded from a worktree with uncommitted changes**, so the
  revision named above does not describe the code that ran — and no revision
  does" — printed over a measurement that was clean.
* It is invisible until the second run. The first write creates an UNTRACKED
  file, which `--untracked-files=no` deliberately ignores, so a new harness
  looks correct and starts lying only once its artifact is committed.
* `worktree_dirty` is the one flag `code_identical.jl` structurally cannot
  cross-check, which is the whole argument for rendering it — so nothing
  downstream can catch a wrong value.

Bind it to a local first (`const PROV = git_provenance()`) and splat that.
`run_reparam_benchmark.jl` and `nonlinear_weighting_run.jl` already did;
`annotation_sweep.jl`, `capture_boxing.jl` and `typical_positions.jl` did not
and were fixed on 2026-07-28. Their checked-in artifacts read `false` only
because each was generated onto a path that was still untracked at the time.
"""
function git_provenance()
    sha = try
        readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`)
    catch
        "unknown"
    end
    dirty(paths...) = try
        !isempty(readchomp(`git -C $(REPO_ROOT) status --porcelain --untracked-files=no $(paths)`))
    catch
        missing
    end
    Dict{String,Any}("warmuphmc_sha" => sha,
                     "worktree_dirty" => dirty(),
                     "src_dirty" => dirty("--", "src"))
end

"""
    env_dir(name, default) -> String

Read an output-directory override from the environment, treating a
**set-but-blank** value as an error rather than as a path.

`get(ENV, name, default)` does not do this, and the difference is not cosmetic:
an unset shell variable expands to `""`, `get` then returns `""` rather than the
default, and `joinpath("", "x")` is the RELATIVE path `"x"` — so the script
writes its JSON into whatever the current directory happens to be. That is
silent on success. It put a stray `typical_positions.json` in the repository
root once, and in CI it is worse: a guard that regenerates artifacts into a temp
directory and then validates them would write them somewhere else entirely and
still exit 0, i.e. pass while checking nothing.

Blank is therefore refused loudly. Unset still falls back to `default`.
"""
function env_dir(name::AbstractString, default::AbstractString)
    haskey(ENV, name) || return default
    value = ENV[name]
    isempty(strip(value)) && error("""
        $(name) is set but blank.

        A blank value is almost always an unset shell variable that expanded to
        the empty string. It is refused rather than defaulted, because
        `joinpath("", f)` is relative and this script would write $(name)'s
        artifacts into the current directory ($(pwd())) without saying so.

        Unset $(name) to use the default ($(default)), or give it a real path.
        """)
    return value
end

# The per-posterior reparametrization table already shipped with the package.
# Reused rather than re-derived: it is what web/ samples through, so the
# benchmark measures the specs the package actually offers.
include(joinpath(REPO_ROOT, "web", "src", "posteriordb_reparametrizations.jl"))

const PDB = PosteriorDB.database()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

"""
A model to benchmark. `sibling` names posteriordb's separately hand-written
noncentered counterpart when one exists.
"""
struct Target
    name::String
    sibling::Union{Nothing,String}
    note::String
end

const TARGETS = [
    Target("eight_schools-eight_schools_centered",
           "eight_schools-eight_schools_noncentered",
           "8 group effects, dim 10 — the textbook hierarchical funnel"),
    Target("radon_mn-radon_partially_pooled_centered",
           "radon_mn-radon_partially_pooled_noncentered",
           "85 counties, dim 88"),
    Target("radon_mn-radon_variable_intercept_centered",
           "radon_mn-radon_variable_intercept_noncentered",
           "85 counties + floor slope, dim 89"),
    Target("seeds_data-seeds_centered_model",
           nothing,
           "21 plates, dim 26 — posteriordb ships no noncentered sibling"),
]

# Neal's funnel. NOT a posteriordb posterior (posteriordb carries none), so it is
# defined here and clearly labelled as synthetic. It is the extreme case: the
# centered parametrization is pathological and the noncentered one is exact, so
# it bounds what adaptive partial centering can possibly buy. Reported alongside
# the real targets, never as a headline on its own.
#
#   v      ~ Normal(0, 3)
#   theta_i ~ Normal(0, exp(v/2)),  i = 1..K
struct Funnel
    K::Int
end
LogDensityProblems.dimension(f::Funnel) = f.K + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(f::Funnel, x)
    v = x[1]
    s2 = exp(v)                      # exp(v/2)^2
    q = sum(abs2, @view x[2:end])
    -abs2(v) / 18 - f.K * v / 2 - q / (2 * s2)
end
function LogDensityProblems.logdensity_and_gradient(f::Funnel, x)
    v = x[1]
    s2 = exp(v)
    th = @view x[2:end]
    q = sum(abs2, th)
    lp = -abs2(v) / 18 - f.K * v / 2 - q / (2 * s2)
    g = similar(x)
    g[1] = -v / 9 - f.K / 2 + q / (2 * s2)
    g[2:end] .= .-th ./ s2
    (lp, g)
end

funnel_spec(f::Funnel, c) = IndexedReparametrization([
    i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(c),
                           0.0, x -> x[1] / 2)
    for i in 2:LogDensityProblems.dimension(f)])

# ---------------------------------------------------------------------------
# Problem + spec construction
# ---------------------------------------------------------------------------

"""
    stan_problem(name) -> (problem, dimension, stan_json_data)

Compile (or reuse) the posteriordb Stan model behind `name`.
"""
function stan_problem(name::AbstractString)
    post = PosteriorDB.posterior(PDB, name)
    prob = StanLogDensityProblems.StanProblem(
        PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan")),
        PosteriorDB.load(PosteriorDB.dataset(post), String);
        nan_on_error = true, make_args = ["STAN_THREADS=TRUE"], warn = false)
    prob, LogDensityProblems.dimension(prob), PosteriorDB.load(PosteriorDB.dataset(post))
end

"""
    native_spec(name, dim, jdata)

The shipped spec for `name`: source and target both at the model's own
parametrization, i.e. an exact no-op until adaptation moves the source.
"""
native_spec(name, dim, jdata) = reparametrization(String(name), dim, jdata)

"""
    with_source(spec, c)

`spec` with every source centering replaced by `c`, keeping each pair's target
and its location/scale closures untouched.
"""
with_source(spec, c) = IndexedReparametrization([
    p.first => Reparametrization(p.second.target, PartiallyCentered(c), p.second.args...)
    for p in spec.pairs])

source_cs(spec) = [Float64(p.second.source.c) for p in spec.pairs]
target_cs(spec) = [Float64(p.second.target.c) for p in spec.pairs]

"""The centering endpoint opposite the model's own."""
opposite_c(spec) = target_cs(spec)[1] == 1.0 ? 0.0 : 1.0

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# WHY `ess_con_min` EQUALS `ess_min` IN EVERY RESULT FILE — NOT A BUG
#
# Callers record ESS twice, once on the sampler's unconstrained draws and once
# on `constrained(model, draws)`. Across the whole corpus the two MINIMA are
# byte-identical in every row (measured 2026-07-28: 152 rows per file, zero
# differences, plus 48 in `sampler_comparison.json`), which reads exactly like
# one of them was computed from the wrong matrix.
#
# It isn't. `MCMCDiagnosticTools.ess` defaults to BULK ESS, which rank-normalizes
# the draws first, so it is invariant under any strictly monotone
# reparametrization of a coordinate. Stan's constraining maps are monotone
# coordinate-wise (`log tau -> tau`, and so on), so every declared parameter has
# the same bulk ESS in both frames — verified directly on an AR(1) series:
# `ess(x)`, `ess(exp(x))` and `ess(x^3)` agree to 14 digits, `ess(-x)` to 12, and
# `ess(x^2)` — not monotone — does not agree at all. Under `kind = :basic` they
# separate (119.5 vs 296.6), so this is a property of the default, not of ESS.
#
# The MEDIANS do differ (56 of 152 rows in `results/before/runs.json`), because
# `include_tp = true` adds transformed parameters, which are functions of several
# coordinates at once and so have no unconstrained counterpart to be invariant
# to. That is the whole difference between the two columns.
#
# Keep recording both. The pair is a cheap standing check that the constraining
# path still runs, and the day it stops being an identity is the day something
# changed — a non-monotone declared transform, or a different `kind`.
"""ESS per coordinate, draws given as `dimension × ndraws`."""
function ess_per_coordinate(draws::AbstractMatrix)
    d, n = size(draws)
    n < 10 && return fill(NaN, d)
    finite = all(isfinite, draws)
    finite || return fill(NaN, d)
    MCMCDiagnosticTools.ess(reshape(permutedims(draws), (n, 1, d)))
end

"""
    constrained(model, draws) -> (names, values)

Map unconstrained draws through BridgeStan into the model's constrained
parameter space, transformed parameters included. Columns that fail to
constrain are dropped and reported by the caller through the returned count.
"""
function constrained(model, draws::AbstractMatrix)
    names = BridgeStan.param_names(model; include_tp = true)
    out = Matrix{Float64}(undef, length(names), size(draws, 2))
    keep = falses(size(draws, 2))
    for j in axes(draws, 2)
        try
            out[:, j] = BridgeStan.param_constrain(model, Vector{Float64}(draws[:, j]);
                                                   include_tp = true)
            keep[j] = all(isfinite, @view out[:, j])
        catch
            keep[j] = false
        end
    end
    names, out[:, keep]
end

nanmin(v) = isempty(v) ? NaN : minimum(v)
nanmed(v) = isempty(v) ? NaN : median(v)

# ---------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------

"""
    run_arm(; arm, problem, spec, adapt, seed, n_draws, model)

Sample `problem` once and return a flat NamedTuple of measurements.

`spec === nothing` means the bare, unwrapped problem. `adapt` selects whether the
sampler may move the source centering; it does not affect the frame the draws come
back in. Every arm's draws are already in the model's own frame, so this function
applies no transform of its own — see the `nonlinear_adapt` note in the header.
"""
function run_arm(; arm::String, problem, spec, adapt::Bool, seed::Int,
                 n_draws::Int, model = nothing)
    rng = Xoshiro(seed)
    lpdf = isnothing(spec) ? problem :
           ReparametrizedProblem(spec, problem, AD_BACKEND)
    c_before = isnothing(spec) ? Float64[] : source_cs(spec)

    local res, wall
    try
        wall = @elapsed res = adaptive_warmup_mcmc(rng, lpdf; n_draws = n_draws,
                                                   nonlinear_adapt = adapt)
    catch err
        return (; arm, seed, ok = false, error = sprint(showerror, err),
                n_draws_actual = 0, wall_s = NaN, grad_evals = 0, n_divergent = 0,
                ess_min = NaN, ess_median = NaN, ess_con_min = NaN, ess_con_median = NaN,
                ess_min_per_s = NaN, ess_min_per_grad = NaN,
                c_before, c_after = Float64[], con_names = String[],
                con_mean = Float64[], con_sd = Float64[], ess_con = Float64[],
                n_con_kept = 0)
    end

    draws = Matrix{Float64}(res.posterior_position)

    n = size(draws, 2)
    ess = ess_per_coordinate(draws)
    ess_min, ess_med = nanmin(ess), nanmed(ess)

    con_names, con_mean, con_sd, ess_con = String[], Float64[], Float64[], Float64[]
    ess_con_min, ess_con_med, n_con = NaN, NaN, 0
    if !isnothing(model) && n > 0
        nms, cvals = constrained(model, draws)
        n_con = size(cvals, 2)
        if n_con > 10
            con_names = nms
            con_mean = vec(mean(cvals; dims = 2))
            con_sd = vec(std(cvals; dims = 2))
            # Kept per NAME, not just reduced: the hand-written noncentered
            # sibling declares more constrained parameters than the centered
            # model (its `theta_trans`), so a min over each model's own full set
            # compares different quantities. Keeping the vector lets the write-up
            # take the min over the names the two models share.
            ess_con = collect(ess_per_coordinate(cvals))
            ess_con_min, ess_con_med = nanmin(ess_con), nanmed(ess_con)
        end
    end

    (; arm, seed, ok = true, error = "",
     n_draws_actual = n, wall_s = wall,
     grad_evals = Int(res.total_evaluation_counter),
     n_divergent = Int(res.n_divergent_samples),
     ess_min, ess_median = ess_med, ess_con_min, ess_con_median = ess_con_med,
     ess_min_per_s = ess_min / wall,
     ess_min_per_grad = ess_min / max(res.total_evaluation_counter, 1),
     c_before, c_after = isnothing(spec) ? Float64[] : source_cs(spec),
     con_names, con_mean, con_sd, ess_con, n_con_kept = n_con)
end

# ---------------------------------------------------------------------------
# Gradient-path overhead
# ---------------------------------------------------------------------------

"""
    gradient_overhead(problem, spec; n, rounds, seed)

Per-call cost of `logdensity_and_gradient` on the bare problem, on the wrapper
with an exact no-op transform (source == target), and on the wrapper with a live
transform (source at the opposite endpoint). Isolates the transform's cost on
the gradient hot path from the sampling dynamics that would otherwise mask it.

Reported as the **median over `rounds` rounds**, with the variant order rotated
each round, and the per-round values kept so the spread is visible rather than
inferred. One shot per variant in a fixed order is not enough on a shared host:
this benchmark runs on a machine other agents also use, and a single pass
produced `seeds` no-op at 2.60x bare against live at 1.27x — the *identity*
transform apparently costing twice what the real one does. That ordering is
impossible, so it was measuring CPU contention, and nothing in a single-shot
number says which of the three variants absorbed it.
"""
function gradient_overhead(problem, spec; n::Int = 2000, rounds::Int = 7, seed::Int = 1)
    dim = LogDensityProblems.dimension(problem)

    # Built once: construction is not what is being timed, and rebuilding per
    # round would charge DI preparation to whichever round it landed in.
    bare = problem
    noop = ReparametrizedProblem(with_source(spec, target_cs(spec)[1]), problem, AD_BACKEND)
    live = ReparametrizedProblem(with_source(spec, opposite_c(spec)), problem, AD_BACKEND)
    variants = ("bare" => bare, "noop" => noop, "live" => live)

    time_it(p, xs) = begin
        LogDensityProblems.logdensity_and_gradient(p, xs[1])          # warm the JIT
        acc = 0.0
        t = @elapsed for x in xs
            v, g = LogDensityProblems.logdensity_and_gradient(p, x)
            acc += isfinite(v) ? 0.0 : 1.0
        end
        (ns_per_call = 1e9 * t / length(xs), n_nonfinite = acc)
    end

    samples = Dict(k => Float64[] for (k, _) in variants)
    nonfinite = Dict(k => 0.0 for (k, _) in variants)
    for r in 1:rounds
        xs = [randn(Xoshiro(1000seed + 17r + i), dim) for i in 1:n]
        for (k, p) in circshift(collect(variants), r)   # rotate: no fixed warm slot
            got = time_it(p, xs)
            push!(samples[k], got.ns_per_call)
            nonfinite[k] += got.n_nonfinite
        end
    end

    med(k) = median(samples[k])
    (; n_calls = n, rounds, dim,
     bare_ns = med("bare"), noop_ns = med("noop"), live_ns = med("live"),
     noop_ratio = med("noop") / med("bare"),
     live_ratio = med("live") / med("bare"),
     bare_ns_rounds = samples["bare"], noop_ns_rounds = samples["noop"],
     live_ns_rounds = samples["live"],
     nonfinite = (bare = nonfinite["bare"], noop = nonfinite["noop"],
                  live = nonfinite["live"]))
end
