# Driver for the nonlinear-reparametrization benchmark.
#
#   julia --project=docs/benchmark docs/benchmark/run_reparam_benchmark.jl
#
# Knobs (all optional, all read from the environment so a rerun is one line):
#   WHMC_BENCH_SEEDS=4        number of pinned seeds per (target, arm)
#   WHMC_BENCH_DRAWS=1000     n_draws floor per run
#   WHMC_BENCH_TARGETS=...    comma-separated posteriordb names, or "funnel"
#   WHMC_BENCH_OUT=...        output directory (default docs/benchmark/results)
#
# Writes `results/runs.json` (one object per run) and `results/gradient_overhead.json`,
# and prints the summary table. Everything it writes is checked in.

include(joinpath(@__DIR__, "common.jl"))

# 8 seeds, not 4: on a centered hierarchical target the run-to-run spread in min
# ESS is larger than most of the effects being measured (a smoke run put two
# essentially identical configurations at 2.5 and 20.9), so a thin seed set would
# let noise pass for a result.
const N_SEEDS = parse(Int, get(ENV, "WHMC_BENCH_SEEDS", "8"))
const N_DRAWS = parse(Int, get(ENV, "WHMC_BENCH_DRAWS", "1000"))
const OUT_DIR = env_dir("WHMC_BENCH_OUT", joinpath(BENCH_DIR, "results"))
const WARMUP_DRAWS = 50   # discarded; only there so the timed runs are not JIT-bound

const SELECTED = let raw = get(ENV, "WHMC_BENCH_TARGETS", "")
    isempty(raw) ? nothing : Set(strip.(split(raw, ",")))
end

mkpath(OUT_DIR)

rows = Any[]
overheads = Any[]

record!(r; kwargs...) = push!(rows, merge(Dict(pairs(r)), Dict(pairs((; kwargs...)))))

# ---------------------------------------------------------------------------
# posteriordb targets
# ---------------------------------------------------------------------------

for tgt in TARGETS
    isnothing(SELECTED) || tgt.name in SELECTED || continue
    @info "target" tgt.name
    prob, dim, jdata = stan_problem(tgt.name)
    model = prob.model
    spec0 = native_spec(tgt.name, dim, jdata)
    if isempty(spec0.pairs)
        @warn "empty reparametrization spec — skipping, this target would measure nothing" tgt.name
        continue
    end
    c_native = target_cs(spec0)[1]
    c_other = opposite_c(spec0)

    arms = [
        ("plain",             () -> nothing,                       true),
        ("fixed_centered",    () -> with_source(spec0, c_native),  false),
        ("fixed_noncentered", () -> with_source(spec0, c_other),   false),
        ("adaptive",          () -> with_source(spec0, c_native),  true),
    ]

    # Per-target JIT warm-up: the location/scale closures are distinct types per
    # posterior family, so each target specializes the wrapped gradient path
    # afresh. Without this the first timed run of each arm is compile-bound.
    for (arm, mkspec, adapt) in arms
        run_arm(; arm, problem = prob, spec = mkspec(), adapt,
                seed = 999, n_draws = WARMUP_DRAWS, model = nothing)
    end

    for (arm, mkspec, adapt) in arms, seed in 1:N_SEEDS
        r = run_arm(; arm, problem = prob, spec = mkspec(), adapt, seed,
                    n_draws = N_DRAWS, model)
        record!(r; target = tgt.name, model_of_record = tgt.name,
                dim, c_native, c_other, note = tgt.note, synthetic = false)
        @info "  " arm seed r.ess_min r.grad_evals round(r.wall_s; digits=2) r.n_divergent
    end

    @info "  gradient-path overhead" tgt.name
    push!(overheads, merge(Dict(pairs(gradient_overhead(prob, spec0))),
                           Dict(:target => tgt.name)))

    # posteriordb's separately hand-written noncentered model, sampled plain.
    # Different unconstrained frame, so only its constrained-space ESS is
    # comparable with the rows above.
    if !isnothing(tgt.sibling)
        sprob, _, _ = stan_problem(tgt.sibling)
        run_arm(; arm = "sibling_plain", problem = sprob, spec = nothing, adapt = true,
                seed = 999, n_draws = WARMUP_DRAWS, model = nothing)
        for seed in 1:N_SEEDS
            r = run_arm(; arm = "sibling_plain", problem = sprob, spec = nothing,
                        adapt = true, seed, n_draws = N_DRAWS, model = sprob.model)
            record!(r; target = tgt.name, model_of_record = tgt.sibling,
                    dim, c_native, c_other, note = tgt.note, synthetic = false)
            @info "  " "sibling_plain" seed r.ess_min r.grad_evals round(r.wall_s; digits=2) r.n_divergent
        end
    end
end

# ---------------------------------------------------------------------------
# Neal's funnel (synthetic — posteriordb ships none)
# ---------------------------------------------------------------------------

if isnothing(SELECTED) || "funnel" in SELECTED
    f = Funnel(9)
    @info "target" name="funnel(K=9)"
    fspec(c) = with_source(funnel_spec(f, 1.0), c)
    farms = [
        ("plain",             () -> nothing,  true),
        ("fixed_centered",    () -> fspec(1.0), false),
        ("fixed_noncentered", () -> fspec(0.0), false),
        ("adaptive",          () -> fspec(1.0), true),
    ]
    for (arm, mkspec, adapt) in farms
        run_arm(; arm, problem = f, spec = mkspec(), adapt, seed = 999,
                n_draws = WARMUP_DRAWS, model = nothing)
    end
    for (arm, mkspec, adapt) in farms, seed in 1:N_SEEDS
        r = run_arm(; arm, problem = f, spec = mkspec(), adapt, seed, n_draws = N_DRAWS)
        record!(r; target = "funnel", model_of_record = "funnel(K=9)",
                dim = LogDensityProblems.dimension(f), c_native = 1.0, c_other = 0.0,
                note = "synthetic Neal funnel, v~N(0,3), theta_i~N(0,exp(v/2))",
                synthetic = true)
        @info "  " arm seed r.ess_min r.grad_evals round(r.wall_s; digits=2) r.n_divergent
    end
    push!(overheads, merge(Dict(pairs(gradient_overhead(f, funnel_spec(f, 1.0)))),
                           Dict(:target => "funnel")))
end

# ---------------------------------------------------------------------------
# Persist + summarize
# ---------------------------------------------------------------------------

# JSON has no NaN/Inf, and a failed or too-short run legitimately produces them.
# Write those as null rather than dropping the row or lying with a number.
jsonsafe(x::AbstractFloat) = isfinite(x) ? x : nothing
jsonsafe(x::AbstractDict) = Dict(string(k) => jsonsafe(v) for (k, v) in x)
jsonsafe(x::AbstractVector) = [jsonsafe(v) for v in x]
jsonsafe(x::NamedTuple) = Dict(string(k) => jsonsafe(v) for (k, v) in pairs(x))
jsonsafe(x) = x

# Provenance travels WITH the numbers, in the file, not in prose that can drift
# away from them. `warmuphmc_sha` because a benchmark measured against a
# regression is indistinguishable from a good one by inspection; `ad_backend`
# because forward vs reverse mode moved the wrapper-overhead figures enough to
# invert a wall-clock verdict. Both are provenance, and neither is recoverable
# after the fact.
whmc_sha() = try
    readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`)
catch
    "unknown"
end
# `--untracked-files=no` is load-bearing, not tidiness. This benchmark WRITES
# its results into the repo, so a bare `--porcelain` counts the previous run's
# untracked output directory and reports every run after the first as DIRTY —
# by construction, with the tracked source byte-identical. That fires exactly
# when two runs are being compared, which is the one time the flag has to mean
# something. Only tracked modifications can change what code ran.
whmc_dirty() = try
    !isempty(readchomp(`git -C $(REPO_ROOT) status --porcelain --untracked-files=no`))
catch
    missing
end

const PROVENANCE = Dict("n_seeds" => N_SEEDS, "n_draws_floor" => N_DRAWS,
                        "julia" => string(VERSION),
                        "blas_threads" => BLAS.get_num_threads(),
                        "ad_backend" => AD_BACKEND_NAME,
                        "warmuphmc_sha" => whmc_sha(),
                        "worktree_dirty" => whmc_dirty())

open(joinpath(OUT_DIR, "runs.json"), "w") do io
    JSON.print(io, jsonsafe(merge(PROVENANCE, Dict("runs" => rows))), 2)
end
open(joinpath(OUT_DIR, "gradient_overhead.json"), "w") do io
    JSON.print(io, jsonsafe(merge(PROVENANCE, Dict("overheads" => overheads))), 2)
end

const ARM_ORDER = ["plain", "fixed_centered", "adaptive", "fixed_noncentered", "sibling_plain"]

agg(rs, k) = begin
    v = [r[k] for r in rs if r[:ok] && isfinite(r[k])]
    isempty(v) ? NaN : median(v)
end

println("\n", "="^108)
@printf("%-42s %-18s %8s %9s %10s %11s %7s %6s\n",
        "target", "arm", "minESS", "ESS/s", "ESS/kgrad", "grads", "div", "final c")
println("="^108)
for tname in unique([r[:target] for r in rows])
    trows = [r for r in rows if r[:target] == tname]
    for arm in ARM_ORDER
        rs = [r for r in trows if r[:arm] == arm]
        isempty(rs) && continue
        cs = [r[:c_after] for r in rs if r[:ok] && !isempty(r[:c_after])]
        cstr = isempty(cs) ? "-" : @sprintf("%.2f", median(reduce(vcat, cs)))
        @printf("%-42s %-18s %8.1f %9.2f %10.2f %11.0f %7.1f %6s\n",
                tname, arm, agg(rs, :ess_min), agg(rs, :ess_min_per_s),
                1000 * agg(rs, :ess_min_per_grad), agg(rs, :grad_evals),
                agg(rs, :n_divergent), cstr)
    end
end
println("="^108)

println("\ngradient-path overhead (ns per logdensity_and_gradient call)")
@printf("%-42s %9s %9s %9s %9s %9s\n", "target", "bare", "noop", "live", "noop/bare", "live/bare")
for o in overheads
    @printf("%-42s %9.0f %9.0f %9.0f %9.2f %9.2f\n",
            o[:target], o[:bare_ns], o[:noop_ns], o[:live_ns], o[:noop_ratio], o[:live_ratio])
end

nfail = count(r -> !r[:ok], rows)
println("\n", length(rows), " runs, ", nfail, " failed")
nfail == 0 || for r in rows
    r[:ok] || println("FAILED ", r[:target], "/", r[:arm], "/seed=", r[:seed], ": ", r[:error])
end
