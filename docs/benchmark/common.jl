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
#   - `nonlinear_adapt=false` freezes the source centering AND skips the
#     finalization back-transform, so its draws come back in the SOURCE frame.
#     Measured (docs/benchmark/README.md records the measurement). The fixed
#     arms therefore apply `WarmupHMC.reparametrize!` exactly once themselves.
#     With nonlinear_adapt=true the sampler has already done it and a second
#     application would corrupt the draws.
#   - `result.ess` is all zeros unless `monitor_ess=true`; this harness never
#     reads it and computes ESS from the draws instead.
#   - `n_draws` is a floor, so every rate is normalized by the actual draw count.
#
# None of the above is documented in `warmuphmc-use`; the gap is filed as snag
# `warmuphmc-use-ha-f74c9962` on WarmupHMC.

using LinearAlgebra
BLAS.set_num_threads(1)   # required for run-to-run reproducibility at a fixed seed

using WarmupHMC, PosteriorDB, LogDensityProblems, StanLogDensityProblems, BridgeStan
using DifferentiationInterface, ForwardDiff
using WarmupHMC: ReparametrizedProblem, IndexedReparametrization, PartiallyCentered,
                 Reparametrization, reparametrize!
using Random, Statistics, Printf
import MCMCDiagnosticTools
import JSON

const BENCH_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCH_DIR, "..", ".."))

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

`spec === nothing` means the bare, unwrapped problem. When `adapt` is false the
sampler neither moves the source centering nor back-transforms at finalization,
so this function applies `reparametrize!` itself — exactly once — to bring the
draws into the model's own frame. When `adapt` is true the sampler has already
done that and a second application would corrupt the draws.
"""
function run_arm(; arm::String, problem, spec, adapt::Bool, seed::Int,
                 n_draws::Int, model = nothing)
    rng = Xoshiro(seed)
    lpdf = isnothing(spec) ? problem :
           ReparametrizedProblem(spec, problem, AutoForwardDiff())
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
                con_mean = Float64[], con_sd = Float64[], n_con_kept = 0)
    end

    draws = Matrix{Float64}(res.posterior_position)
    # `nonlinear_adapt=false` leaves the draws in the sampler's frame.
    if !isnothing(spec) && !adapt && size(draws, 2) > 0
        reparametrize!(lpdf, draws)
    end

    n = size(draws, 2)
    ess = ess_per_coordinate(draws)
    ess_min, ess_med = nanmin(ess), nanmed(ess)

    con_names, con_mean, con_sd = String[], Float64[], Float64[]
    ess_con_min, ess_con_med, n_con = NaN, NaN, 0
    if !isnothing(model) && n > 0
        nms, cvals = constrained(model, draws)
        n_con = size(cvals, 2)
        if n_con > 10
            con_names = nms
            con_mean = vec(mean(cvals; dims = 2))
            con_sd = vec(std(cvals; dims = 2))
            ce = ess_per_coordinate(cvals)
            ess_con_min, ess_con_med = nanmin(ce), nanmed(ce)
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
     con_names, con_mean, con_sd, n_con_kept = n_con)
end

# ---------------------------------------------------------------------------
# Gradient-path overhead
# ---------------------------------------------------------------------------

"""
    gradient_overhead(problem, spec; n, seed)

Per-call cost of `logdensity_and_gradient` on the bare problem, on the wrapper
with an exact no-op transform (source == target), and on the wrapper with a live
transform (source at the opposite endpoint). Isolates the transform's cost on
the gradient hot path from the sampling dynamics that would otherwise mask it.
"""
function gradient_overhead(problem, spec; n::Int = 2000, seed::Int = 1)
    dim = LogDensityProblems.dimension(problem)
    rng = Xoshiro(seed)
    xs = [randn(rng, dim) for _ in 1:n]

    bare = problem
    noop = ReparametrizedProblem(with_source(spec, target_cs(spec)[1]), problem, AutoForwardDiff())
    live = ReparametrizedProblem(with_source(spec, opposite_c(spec)), problem, AutoForwardDiff())

    time_it(p) = begin
        LogDensityProblems.logdensity_and_gradient(p, xs[1])          # warm the JIT
        acc = 0.0
        t = @elapsed for x in xs
            v, g = LogDensityProblems.logdensity_and_gradient(p, x)
            acc += isfinite(v) ? 0.0 : 1.0
        end
        (ns_per_call = 1e9 * t / n, n_nonfinite = acc)
    end

    b, nn, lv = time_it(bare), time_it(noop), time_it(live)
    (; n_calls = n, dim,
     bare_ns = b.ns_per_call, noop_ns = nn.ns_per_call, live_ns = lv.ns_per_call,
     noop_ratio = nn.ns_per_call / b.ns_per_call,
     live_ratio = lv.ns_per_call / b.ns_per_call,
     nonfinite = (bare = b.n_nonfinite, noop = nn.n_nonfinite, live = lv.n_nonfinite))
end
