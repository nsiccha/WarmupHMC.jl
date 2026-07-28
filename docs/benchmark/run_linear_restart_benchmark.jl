# Driver for the linear restart-evidence benchmark.
#
#   julia --project=docs/benchmark docs/benchmark/run_linear_restart_benchmark.jl
#
# Optional environment knobs:
#   WHMC_LINEAR_BENCH_SEEDS=32
#   WHMC_LINEAR_BENCH_DRAWS=1000
#   WHMC_LINEAR_BENCH_EVALS=1000
#   WHMC_LINEAR_BENCH_TARGETS=diag_gaussian,diag_fallback_probe,correlated_gaussian,kilpisjarvi_mod-kilpisjarvi,diamonds-diamonds
#   WHMC_LINEAR_BENCH_OUT=docs/benchmark/results
#
# This is deliberately a linear-only experiment. Every run fixes
# `nonlinear_adapt=false`; the only differences between arms are the source of
# the linear restart statistic and the optional whole-trajectory step-size
# multiplier. The legacy `:halo` arm is the control.

using LinearAlgebra
BLAS.set_num_threads(1)

using WarmupHMC, LogDensityProblems
using Random, Statistics, Printf, Dates, Sockets
import MCMCDiagnosticTools
import JSON

include("linear_restart_summary.jl")

const BENCH_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCH_DIR, "..", ".."))
const N_SEEDS = parse(Int, get(ENV, "WHMC_LINEAR_BENCH_SEEDS", "32"))
const N_DRAWS = parse(Int, get(ENV, "WHMC_LINEAR_BENCH_DRAWS", "1000"))
const N_EVALUATIONS = parse(Int, get(ENV, "WHMC_LINEAR_BENCH_EVALS", "1000"))
# Blank is refused, not defaulted -- see `env_dir` in common.jl for the full
# reason. Short version: an unset shell variable expands to "", `get` returns it
# rather than the default, and `joinpath("", f)` is relative, so the run writes
# its JSON into the current directory without saying so.
if haskey(ENV, "WHMC_LINEAR_BENCH_OUT") && isempty(strip(ENV["WHMC_LINEAR_BENCH_OUT"]))
    error("WHMC_LINEAR_BENCH_OUT is set but blank; unset it or give it a real path.")
end
const OUT_DIR = get(
    ENV, "WHMC_LINEAR_BENCH_OUT",
    joinpath(BENCH_DIR, "results"),
)
const WARMUP_DRAWS = 50
const WARMUP_EVALUATIONS = min(200, N_EVALUATIONS)

const DEFAULT_TARGETS = [
    "diag_gaussian",
    "diag_fallback_probe",
    "correlated_gaussian",
    "kilpisjarvi_mod-kilpisjarvi",
    "diamonds-diamonds",
]
const SELECTED_TARGETS = let raw = get(ENV, "WHMC_LINEAR_BENCH_TARGETS", "")
    isempty(raw) ? DEFAULT_TARGETS : strip.(split(raw, ","))
end
const SYNTHETIC_TARGETS = Set([
    "diag_gaussian", "diag_fallback_probe", "correlated_gaussian",
])
const NEEDS_POSTERIORDB = any(name -> name ∉ SYNTHETIC_TARGETS, SELECTED_TARGETS)

# Keep the deterministic Gaussian controls runnable in the ordinary test
# environment. PosteriorDB and the Stan toolchain are loaded only for the full
# evidence matrix, which lets CI exercise the exact same driver cheaply when a
# fast synthetic-only pass is useful.
if NEEDS_POSTERIORDB
    @eval using PosteriorDB, StanLogDensityProblems, BridgeStan
end
const PDB = NEEDS_POSTERIORDB ? PosteriorDB.database() : nothing

const ARMS = [
    (name="halo_unit", source=:halo, weighting=:unit),
    (name="all_good_unit", source=:all_good_leaves, weighting=:unit),
    (name="all_good_stepsize", source=:all_good_leaves, weighting=:stepsize),
    (name="nuts_unit", source=:nuts_weighted, weighting=:unit),
    (name="nuts_stepsize", source=:nuts_weighted, weighting=:stepsize),
]

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

struct GaussianTarget{M}
    covariance::M
    precision::M
end
GaussianTarget(covariance::AbstractMatrix) = begin
    covariance = Matrix(Symmetric(covariance))
    GaussianTarget(covariance, inv(covariance))
end
LogDensityProblems.dimension(g::GaussianTarget) = size(g.covariance, 1)
LogDensityProblems.capabilities(::Type{<:GaussianTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(g::GaussianTarget, x) = -dot(x, g.precision, x) / 2
LogDensityProblems.logdensity_and_gradient(g::GaussianTarget, x) =
    (LogDensityProblems.logdensity(g, x), -(g.precision * x))

struct DiagonalStudentT{V,T}
    scales::V
    degrees_of_freedom::T
end
LogDensityProblems.dimension(g::DiagonalStudentT) = length(g.scales)
LogDensityProblems.capabilities(::Type{<:DiagonalStudentT}) =
    LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(g::DiagonalStudentT, x)
    nu = g.degrees_of_freedom
    -(nu + 1) / 2 * sum(log1p((xi / si)^2 / nu) for (xi, si) in zip(x, g.scales))
end
function LogDensityProblems.logdensity_and_gradient(g::DiagonalStudentT, x)
    nu = g.degrees_of_freedom
    gradient = @. -(nu + 1) * x / (nu * g.scales^2 + x^2)
    LogDensityProblems.logdensity(g, x), gradient
end

function synthetic_target(name)
    d = 8
    if name in ("diag_gaussian", "diag_fallback_probe")
        scales = exp.(range(-2.0, 2.0, d))
        if name == "diag_fallback_probe"
            nu = 5.0
            covariance = Matrix(Diagonal((nu / (nu - 2)) .* scales .^ 2))
            problem = DiagonalStudentT(scales, nu)
            note = "8D independent Student-t(5); forced diagonal start and 100-slot halo isolate fallback"
            force_initial_diagonal = true
            recording_target = 100
        else
            covariance = Matrix(Diagonal(scales .^ 2))
            problem = GaussianTarget(covariance)
            note = "8D diagonal Gaussian with four orders of marginal variance"
            force_initial_diagonal = false
            recording_target = nothing
        end
    elseif name == "correlated_gaussian"
        # Four strongly correlated 2D blocks. This is cheap but discriminating:
        # SuccessiveReflections must fit at least one non-diagonal reflection.
        block = [1.0 0.98; 0.98 1.0]
        correlation = kron(Matrix{Float64}(I, d ÷ 2, d ÷ 2), block)
        scales = exp.(range(-1.0, 1.0, d))
        covariance = Matrix(Diagonal(scales) * correlation * Diagonal(scales))
        problem = GaussianTarget(covariance)
        note = "8D Gaussian with four rho=0.98 correlated blocks"
        force_initial_diagonal = false
        recording_target = nothing
    else
        error("unknown synthetic target $(repr(name))")
    end
    init = (; position=zeros(d), squared_scale=Matrix{Float64}(I, d, d))
    (; name, problem, init, model=nothing, truth_covariance=covariance,
       reference=nothing, dimension=d, note, synthetic=true,
       force_initial_diagonal, recording_target)
end

function stan_problem(name::AbstractString)
    posterior = PosteriorDB.posterior(PDB, String(name))
    problem = StanLogDensityProblems.StanProblem(
        PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan")),
        PosteriorDB.load(PosteriorDB.dataset(posterior), String);
        nan_on_error=true, make_args=["STAN_THREADS=TRUE"], warn=false,
    )
    reference = PosteriorDB.load(PosteriorDB.reference_posterior(posterior))
    (; name=String(name), problem, init=missing, model=problem.model,
       truth_covariance=nothing, reference,
       dimension=LogDensityProblems.dimension(problem),
       note="PosteriorDB posterior with genuine linear correlation", synthetic=false,
       force_initial_diagonal=false, recording_target=nothing)
end

make_target(name) = name in SYNTHETIC_TARGETS ?
                    synthetic_target(name) : stan_problem(name)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

function ess_per_coordinate(draws::AbstractMatrix)
    d, n = size(draws)
    n < 10 && return fill(NaN, d)
    all(isfinite, draws) || return fill(NaN, d)
    MCMCDiagnosticTools.ess(reshape(permutedims(draws), (n, 1, d)))
end

function constrained_draws(model, draws::AbstractMatrix)
    names = BridgeStan.param_names(model; include_tp=false)
    values = Matrix{Float64}(undef, length(names), size(draws, 2))
    keep = falses(size(draws, 2))
    for j in axes(draws, 2)
        try
            values[:, j] = BridgeStan.param_constrain(
                model, Vector{Float64}(draws[:, j]); include_tp=false,
            )
            keep[j] = all(isfinite, @view values[:, j])
        catch
            keep[j] = false
        end
    end
    names, values[:, keep]
end

reference_name(name) = replace(name, r"\.(\d+)" => s"[\1]")

function reference_stats(reference)
    keys0 = collect(keys(first(reference)))
    Dict(String(key) => begin
        values = reduce(vcat, [chain[key] for chain in reference])
        (; mean=mean(values), sd=std(values))
    end for key in keys0)
end

function reference_errors(model, draws, reference)
    names, values = constrained_draws(model, draws)
    stats = reference_stats(reference)
    matched = [i for i in eachindex(names) if haskey(stats, reference_name(names[i]))]
    isempty(matched) && return (;
        constrained_names=String[], constrained_mean=Float64[], constrained_sd=Float64[],
        reference_mean_z_rmse=NaN, reference_mean_z_max=NaN,
        reference_sd_log_rmse=NaN, n_constrained_kept=size(values, 2),
    )
    sample_mean = vec(mean(values[matched, :]; dims=2))
    sample_sd = vec(std(values[matched, :]; dims=2))
    ref_mean = [stats[reference_name(names[i])].mean for i in matched]
    ref_sd = [stats[reference_name(names[i])].sd for i in matched]
    mean_z = (sample_mean .- ref_mean) ./ ref_sd
    sd_log_error = log.(sample_sd ./ ref_sd)
    (;
        constrained_names=names[matched],
        constrained_mean=sample_mean,
        constrained_sd=sample_sd,
        reference_mean_z_rmse=sqrt(mean(abs2, mean_z)),
        reference_mean_z_max=maximum(abs, mean_z),
        reference_sd_log_rmse=sqrt(mean(abs2, sd_log_error)),
        n_constrained_kept=size(values, 2),
    )
end

metric_reflections(result) =
    length(result.scale_options.adaptive.m1.reflections)

function run_arm(target, arm, seed; n_draws=N_DRAWS, n_evaluations=N_EVALUATIONS)
    windows = NamedTuple[]
    callback = (state, stage) -> begin
        # This one labelled mechanistic control deliberately begins in the
        # diagonal family. It is not presented as a public initialization API;
        # it makes the same-family fallback occur so that the benchmark does not
        # mistake an unexercised branch for evidence about its quality.
        if stage === :init && target.force_initial_diagonal
            state.active_transformation = :diagonal
            state.kinetic_energy = state.energy_options.diagonal
        end
        stage === :window && push!(windows, (;
            outer=state.outer_counter,
            restart=state.restart,
            n_samples=state.n_samples,
            variance_condition=state.variance_cond,
            active_transformation=state.active_transformation,
            fallback_count=state.linear_metric_fallbacks,
        ))
        false
    end
    init = ismissing(target.init) ? missing : deepcopy(target.init)
    local result, wall
    try
        wall = @elapsed result = adaptive_warmup_mcmc(
            Xoshiro(seed), target.problem;
            init,
            n_draws,
            n_evaluations,
            recording_target=something(target.recording_target, n_evaluations),
            nonlinear_adapt=false,
            monitor_ess=false,
            progress=nothing,
            linear_restart_source=arm.source,
            linear_trajectory_weighting=arm.weighting,
            callback,
        )
    catch err
        return (;
            target=target.name, arm=arm.name, source=arm.source,
            weighting=arm.weighting, seed, ok=false,
            error=sprint(showerror, err, catch_backtrace()),
        )
    end

    draws = Matrix{Float64}(result.posterior_position)
    ess = ess_per_coordinate(draws)
    actual_restarts = count(w -> w.restart && w.n_samples < n_draws, windows)
    truth = if isnothing(target.truth_covariance)
        reference_errors(target.model, draws, target.reference)
    else
        sample_covariance = cov(draws; dims=2)
        (;
            mean_scaled_error=norm(vec(mean(draws; dims=2))) /
                              sqrt(tr(target.truth_covariance)),
            covariance_relative_error=norm(sample_covariance - target.truth_covariance) /
                                      norm(target.truth_covariance),
        )
    end

    merge((;
        target=target.name,
        target_note=target.note,
        target_dimension=target.dimension,
        synthetic=target.synthetic,
        arm=arm.name,
        source=arm.source,
        weighting=arm.weighting,
        seed,
        ok=true,
        error="",
        n_draws_actual=size(draws, 2),
        wall_s=wall,
        grad_evals=Int(result.total_evaluation_counter),
        divergences=Int(result.n_divergent_samples),
        n_windows=length(windows),
        n_restarts=actual_restarts,
        restart_conditions=copy(result.scale_changes),
        active_transformation=result.active_transformation,
        adaptive_reflections=metric_reflections(result),
        linear_metric_fallbacks=result.linear_metric_fallbacks,
        ess_min=minimum(ess),
        ess_median=median(ess),
        ess_min_per_s=minimum(ess) / wall,
        ess_min_per_kgrad=1000 * minimum(ess) / max(result.total_evaluation_counter, 1),
        window_history=windows,
    ), truth)
end

# ---------------------------------------------------------------------------
# Run and persist
# ---------------------------------------------------------------------------

repo_sha() = try
    readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`)
catch
    "unknown"
end
repo_dirty() = try
    !isempty(readchomp(`git -C $(REPO_ROOT) status --porcelain --untracked-files=no`))
catch
    missing
end

jsonsafe(x::AbstractFloat) = isfinite(x) ? x : nothing
jsonsafe(x::Symbol) = String(x)
jsonsafe(x::AbstractDict) = Dict(string(k) => jsonsafe(v) for (k, v) in x)
jsonsafe(x::NamedTuple) = Dict(string(k) => jsonsafe(v) for (k, v) in pairs(x))
jsonsafe(x::AbstractVector) = [jsonsafe(v) for v in x]
jsonsafe(x) = x

mkpath(OUT_DIR)
rows = Any[]

for target_name in SELECTED_TARGETS
    @info "target" target_name
    target = make_target(target_name)
    # One discarded run specializes the target's log-density and Stan model
    # before any wall-clock observation is retained.
    warmup = run_arm(
        target, first(ARMS), 999;
        n_draws=WARMUP_DRAWS, n_evaluations=WARMUP_EVALUATIONS,
    )
    warmup.ok || error("JIT warm-up failed for $target_name: $(warmup.error)")

    # Rotate the arm order by seed. Every arm occupies every ordinal position
    # across a sufficiently large seed set, so thermal drift cannot always
    # favor the same policy.
    for seed in 1:N_SEEDS
        for order in eachindex(ARMS)
            arm_index = mod1(order + seed - 1, length(ARMS))
            arm = ARMS[arm_index]
            row = merge(run_arm(target, arm, seed), (; run_order=order))
            push!(rows, row)
            if row.ok
                @info "  run" target=target_name arm=arm.name seed=seed grad_evals=row.grad_evals n_restarts=row.n_restarts active=row.active_transformation reflections=row.adaptive_reflections ess_min=row.ess_min wall_s=round(row.wall_s; digits=3)
            else
                @error "  failed" target=target_name arm=arm.name seed row.error
            end
        end
    end
end

provenance = Dict(
    "timestamp_utc" => string(now(UTC)),
    "warmuphmc_sha" => repo_sha(),
    "worktree_dirty" => repo_dirty(),
    "host" => get(ENV, "KB_HOST", gethostname()),
    "native_hostname" => gethostname(),
    "julia" => string(VERSION),
    "blas_threads" => BLAS.get_num_threads(),
    "julia_threads" => Threads.nthreads(),
    "n_seeds" => N_SEEDS,
    "n_draws_floor" => N_DRAWS,
    "first_window_gradient_budget" => N_EVALUATIONS,
    "nonlinear_adapt" => false,
    "targets" => SELECTED_TARGETS,
    "arms" => [Dict("name" => a.name, "source" => String(a.source),
                    "weighting" => String(a.weighting)) for a in ARMS],
)

provenance["note"] = "Linear-only restart/fallback evidence; nonlinear_adapt=false in every run. Wall time is host-specific; compare ESS per gradient across hosts."
payload = jsonsafe(merge(provenance, Dict("runs" => rows)))
result_path = joinpath(OUT_DIR, "linear_restart.json")
open(result_path, "w") do io
    JSON.print(io, payload, 2)
end

summary_rows = linear_restart_summary(payload)
println("\nWrote ", result_path)
println("\n| target | arm | grads | restarts | min ESS | min ESS/kgrad | div | active | refl | fallback |")
println("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
for row in summary_rows
    @printf("| %s | %s | %.0f | %.1f | %.1f | %.2f | %.0f | %s | %.1f | %.1f |\n",
            row["target"], row["arm"], row["grad_evals_median"], row["restarts_median"],
            row["min_ess_median"], row["min_ess_per_kgrad_median"],
            row["divergences_total"], row["active_transformations"],
            row["adaptive_reflections_median"], row["linear_metric_fallbacks_median"])
end

n_failed = count(r -> !r.ok, rows)
println(length(rows), " measured runs, ", n_failed, " failed")
n_failed == 0 || exit(1)
