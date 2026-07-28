# Cross-sampler comparison: WarmupHMC vs DynamicHMC vs AdvancedHMC.
#
#   julia --project=bench bench/sampler_comparison.jl
#
# THE CLAIM THIS EXISTS TO BACK
#
# The README says "Results should come in faster than with 'standard' methods,
# and should often be better." Until this file, nothing in the repository
# measured that. The benchmark corpus under `docs/benchmark/` is thorough and
# says so itself, in its own § *What this does not measure*: "One sampler.
# Single-chain `adaptive_warmup_mcmc` only." It compares parametrizations and AD
# backends — never WarmupHMC against another sampler.
#
# So the headline claim of a package heading for 1.0 was prose with nothing
# behind it. That is the sharpest shape in the anti-check family: a plausible
# statement invites no check, because it reads as something someone already
# established.
#
# WHAT IS MEASURED
#
# Three samplers, the SAME target, the same number of requested draws:
#
#   warmuphmc     `adaptive_warmup_mcmc(rng, cp; n_draws)` — defaults.
#   dynamichmc    `DynamicHMC.mcmc_with_warmup(rng, cp, n_draws)` — defaults.
#   advancedhmc   NUTS (multinomial, generalised no-U-turn, max depth 10) with
#                 Stan's adaptor: diagonal mass matrix + dual averaging to 0.8,
#                 1000 adaptation iterations, warmup dropped.
#
# The three arms are deliberately the SAME ones the live dashboard runs
# (`web/src/WarmupHMCWeb.jl`), so a number here and a cell there mean the same
# thing. The difference is that these are written down, checked in, and
# regenerable, whereas the dashboard's are a recording of a run nobody can
# reproduce from the repository.
#
# Every arm is wrapped in `WarmupHMC.count_and_time`, so `grad_evals` is counted
# at the `logdensity_and_gradient` boundary by the same counter for all three —
# not read from each sampler's own bookkeeping, which counts different things.
#
# TWO RATES, AND WHY BOTH
#
#   ess_min_per_grad   min ESS per gradient evaluation. PORTABLE — it is the
#                      number to quote, and the one the verdict is stated in.
#   ess_min_per_s      min ESS per wall-clock second. Machine-specific, reported
#                      because "faster" is a wall-clock word and dropping it
#                      would answer a question nobody asked.
#
# They can disagree, and when they do that is a finding, not an error: WarmupHMC
# does strictly more work per gradient (Pathfinder init, several linear
# transformations fitted in parallel), so a win in evaluations is not
# automatically a win in seconds.
#
# WHAT THIS DOES NOT MEASURE
#
#   * CORRECTNESS. Nothing here checks the draws are from the right
#     distribution. An efficiency win on wrong draws is worse than no claim.
#     Divergence counts are recorded per arm as the one cheap smell test, and a
#     nonzero count is reported rather than hidden.
#   * More than one chain. min ESS is within-chain; there is no R-hat here.
#   * A tuned AdvancedHMC. Its arm is a reasonable default configuration, not an
#     expert's. `initial_stepsize=0.1` and `theta_init=randn(...)` are the
#     dashboard's choices, kept for comparability. A reader should read its
#     numbers as "AdvancedHMC out of the box", not "AdvancedHMC at its best".
#   * Warmup budget parity. The three samplers decide for themselves how much
#     warmup to spend; only AdvancedHMC's is stated explicitly (it has no
#     default). This is the honest comparison for a user choosing a sampler and
#     the WRONG comparison for attributing a win to one mechanism.
#   * Wall-clock portability. See above; `ess_min_per_grad` is the portable one.

using LinearAlgebra
BLAS.set_num_threads(1)   # required for run-to-run reproducibility at a fixed seed

# The shared harness: `TARGETS`, `Funnel`, `stan_problem`, `ess_per_coordinate`,
# `constrained`, `git_provenance`, `env_dir`, `nanmin`/`nanmed`. Reused rather
# than reimplemented so ESS means the same thing in both benchmarks — see the
# note in `bench/Project.toml` about the coupling this creates.
include(joinpath(@__DIR__, "..", "docs", "benchmark", "common.jl"))

import AdvancedHMC
using JSON, Dates

const N_SEEDS  = parse(Int, get(ENV, "WHMC_CMP_SEEDS",  "4"))
const N_DRAWS  = parse(Int, get(ENV, "WHMC_CMP_DRAWS",  "1000"))
# The SCRIPT lives in `bench/` (it needs its own environment, see
# `bench/Project.toml`) but the RESULTS go where every other measured artifact
# goes. `docs/src/evidence.md` states, as a property of this repository, that
# "every measurement cited anywhere in this documentation comes from a JSON file
# under docs/benchmark/results/", and `docs/evidence.jl` renders that directory
# at build time. Writing somewhere else would make that sentence false the moment
# a docs page cited this benchmark — and it would do so silently, because the
# appendix walks a directory and cannot miss a file it was never shown.
# Landing here instead means this artifact gets an appendix section, and the
# currency gate in `docs/benchmark/artifact_currency.jl` picks it up, for free:
# both work off `git ls-files` over that directory, neither has a list to update.
const OUT_DIR  = env_dir("WHMC_CMP_OUT", normpath(joinpath(@__DIR__, "..", "docs", "benchmark", "results")))
const FUNNEL_K = parse(Int, get(ENV, "WHMC_CMP_FUNNEL_K", "9"))

const SELECTED = let raw = get(ENV, "WHMC_CMP_TARGETS", "")
    isempty(raw) ? nothing : Set(strip.(split(raw, ",")))
end

mkpath(OUT_DIR)

# ---------------------------------------------------------------------------
# The three arms
# ---------------------------------------------------------------------------
#
# Each returns `(draws, n_divergent)` given an already-counting problem `cp`.
# Keeping them as three small functions rather than one branchy one is what lets
# the shared measurement code below be genuinely shared: every arm goes through
# the same `count_and_time`, the same ESS call and the same record.

arm_warmuphmc(rng, cp, dim) = begin
    res = adaptive_warmup_mcmc(rng, cp; n_draws = N_DRAWS, progress = nothing)
    (Matrix{Float64}(res.posterior_position), Int(res.n_divergent_samples))
end

arm_dynamichmc(rng, cp, dim) = begin
    res = WarmupHMC.DynamicHMC.mcmc_with_warmup(rng, cp, N_DRAWS;
              reporter = WarmupHMC.DynamicHMC.NoProgressReport())
    ndiv = count(s -> WarmupHMC.DynamicHMC.is_divergent(s.termination),
                 res.tree_statistics)
    (Matrix{Float64}(res.posterior_matrix), ndiv)
end

# 1000, matching DynamicHMC's default warmup length. AdvancedHMC has no default
# — the count is a required positional — so leaving it implicit is not an option
# and picking it IS a fairness decision. Stated here rather than buried.
const AHMC_N_ADAPTS = 1000

arm_advancedhmc(rng, cp, dim) = begin
    metric      = AdvancedHMC.DiagEuclideanMetric(Float64, dim)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, cp)
    integrator  = AdvancedHMC.Leapfrog(0.1)
    kernel      = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
                      integrator, AdvancedHMC.GeneralisedNoUTurn(10, 1000.0)))
    adaptor     = AdvancedHMC.StanHMCAdaptor(
                      AdvancedHMC.MassMatrixAdaptor(metric),
                      AdvancedHMC.StepSizeAdaptor(0.8, integrator))
    theta_init  = randn(rng, dim)
    thetas, stats = AdvancedHMC.sample(rng, hamiltonian, kernel, theta_init,
        N_DRAWS + AHMC_N_ADAPTS, adaptor, AHMC_N_ADAPTS;
        drop_warmup = true, verbose = false, progress = false)
    (Matrix{Float64}(reduce(hcat, thetas)), sum(s.numerical_error for s in stats))
end

const ARMS = [
    ("warmuphmc",   arm_warmuphmc),
    ("dynamichmc",  arm_dynamichmc),
    ("advancedhmc", arm_advancedhmc),
]

# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

"""
    run_sampler(; arm, run, problem, seed, model)

Sample once and return a flat record. A throwing arm is recorded as `ok=false`
WITH its error rather than dropped — a sampler that cannot run on a target is a
result about that sampler, and silently omitting it would quietly restrict the
comparison to the targets where everything happened to work.
"""
function run_sampler(; arm::String, run, problem, seed::Int, model = nothing)
    rng = Xoshiro(seed)
    dim = LogDensityProblems.dimension(problem)
    local timed
    try
        timed = WarmupHMC.count_and_time(problem) do cp
            run(rng, cp, dim)
        end
    catch err
        return (; arm, seed, ok = false, error = sprint(showerror, err),
                n_draws_actual = 0, wall_s = NaN, grad_evals = 0, n_divergent = 0,
                ess_min = NaN, ess_median = NaN, ess_con_min = NaN,
                ess_con_median = NaN, ess_min_per_s = NaN, ess_min_per_grad = NaN,
                n_con_kept = 0)
    end

    draws, ndiv = timed.result
    wall, grads = timed.elapsed, timed.n_evaluations
    n = size(draws, 2)

    ess = ess_per_coordinate(draws)
    ess_min, ess_med = nanmin(ess), nanmed(ess)

    # Constrained-space ESS as well, for the same reason the reparametrization
    # benchmark computes it: the unconstrained frame is the sampler's, and a
    # reader cares about the parameters they declared.
    ess_con_min, ess_con_med, n_con = NaN, NaN, 0
    if !isnothing(model) && n > 0
        _, cvals = constrained(model, draws)
        n_con = size(cvals, 2)
        if n_con > 10
            ec = ess_per_coordinate(cvals)
            ess_con_min, ess_con_med = nanmin(ec), nanmed(ec)
        end
    end

    (; arm, seed, ok = true, error = "",
     n_draws_actual = n, wall_s = wall, grad_evals = grads, n_divergent = ndiv,
     ess_min, ess_median = ess_med, ess_con_min, ess_con_median = ess_con_med,
     # `n_draws` is a floor for WarmupHMC and exact for the others, so every rate
     # is normalized by what actually came back, never by the request.
     ess_min_per_s = ess_min / wall,
     ess_min_per_grad = ess_min / max(grads, 1),
     n_con_kept = n_con)
end

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

rows = Any[]
record!(target, r) = push!(rows, merge(Dict("target" => target), Dict(string(k) => v for (k, v) in pairs(r))))

# A short discarded run per (target, arm) so the timed runs are not JIT-bound.
# Without it the first seed of each arm pays compilation and reads as several
# times slower than the rest, which would land in the medians.
function warm_up_jit(problem, run)
    try
        WarmupHMC.count_and_time(problem) do cp
            run(Xoshiro(0), cp, LogDensityProblems.dimension(problem))
        end
    catch
        # A failing warm-up is not a result; the real run below records the error.
    end
    nothing
end

# --- synthetic funnel: no Stan compilation, so it always runs -----------------
if isnothing(SELECTED) || "funnel" in SELECTED
    @info "target" name = "funnel" K = FUNNEL_K
    f = Funnel(FUNNEL_K)
    for (arm, run) in ARMS
        warm_up_jit(f, run)
        for seed in 1:N_SEEDS
            r = run_sampler(; arm, run, problem = f, seed)
            r.ok || @warn "arm failed" target = "funnel" arm seed error = r.error
            record!("funnel", r)
        end
    end
end

# --- posteriordb targets ------------------------------------------------------
for tgt in TARGETS
    isnothing(SELECTED) || tgt.name in SELECTED || continue
    @info "target" name = tgt.name
    prob, dim, _ = stan_problem(tgt.name)
    for (arm, run) in ARMS
        warm_up_jit(prob, run)
        for seed in 1:N_SEEDS
            r = run_sampler(; arm, run, problem = prob, seed, model = prob.model)
            r.ok || @warn "arm failed" target = tgt.name arm seed error = r.error
            record!(tgt.name, r)
        end
    end
end

# ---------------------------------------------------------------------------
# Write + summarize
# ---------------------------------------------------------------------------

# NaN is not valid JSON, and a failed arm produces plenty of it. Same convention
# and same four lines as `docs/benchmark/run_reparam_benchmark.jl:133` — copied
# rather than shared because that file is a driver, so including it would run the
# reparametrization benchmark as a side effect of writing a file.
jsonsafe(x::AbstractFloat) = isfinite(x) ? x : nothing
jsonsafe(x::AbstractDict) = Dict(string(k) => jsonsafe(v) for (k, v) in x)
jsonsafe(x::AbstractVector) = [jsonsafe(v) for v in x]
jsonsafe(x::NamedTuple) = Dict(string(k) => jsonsafe(v) for (k, v) in pairs(x))
jsonsafe(x) = x

out = joinpath(OUT_DIR, "sampler_comparison.json")
# Provenance goes INSIDE `config`, merged, not beside it under its own key.
# `docs/tables.jl:provenance` reads `config` first and falls back to the top
# level; a `"provenance"` key is neither, so the caption rendered "an
# **unrecorded** WarmupHMC revision" while the SHA sat in the file two lines
# away. The one shape that cannot go wrong is a single flat object — there is
# then no second copy of `warmuphmc_sha` for the first to disagree with, which
# is exactly the reason `config` is the preferred shape there.
#
# `blas_threads` is READ BACK, not asserted. Line 70 sets it to 1, so writing a
# literal `1` here would agree with reality today and keep agreeing after
# someone deletes that line — an assertion that cannot go red is the thing the
# rest of this corpus is built to avoid. Every harness under `docs/benchmark/`
# calls `BLAS.get_num_threads()` for the same reason.
open(out, "w") do io
    JSON.print(io, jsonsafe(Dict(
        "config" => merge(git_provenance(),
                          Dict{String,Any}("n_seeds" => N_SEEDS, "n_draws" => N_DRAWS,
                                           "funnel_K" => FUNNEL_K,
                                           "ahmc_n_adapts" => AHMC_N_ADAPTS,
                                           "julia" => string(VERSION),
                                           "blas_threads" => BLAS.get_num_threads())),
        "runs" => rows,
    )), 2)
end
@info "wrote" out n_rows = length(rows)

using Printf
targets = unique(r["target"] for r in rows)
println()
@printf("%-46s %-12s %10s %10s %10s %6s\n",
        "target", "arm", "minESS", "ESS/grad", "ESS/s", "div")
for t in targets, (arm, _) in ARMS
    rs = [r for r in rows if r["target"] == t && r["arm"] == arm && r["ok"]]
    isempty(rs) && (@printf("%-46s %-12s %10s %10s %10s %6s\n", t, arm, "FAILED", "-", "-", "-"); continue)
    @printf("%-46s %-12s %10.1f %10.2e %10.2f %6d\n", t, arm,
            median(r["ess_min"] for r in rs),
            median(r["ess_min_per_grad"] for r in rs),
            median(r["ess_min_per_s"] for r in rs),
            sum(r["n_divergent"] for r in rs))
end
