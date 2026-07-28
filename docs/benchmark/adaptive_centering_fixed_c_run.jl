# End-to-end fixed-c benchmark for the strict-online adaptive-centering study.
#
# Run from the repository root after the canonical docs/benchmark resolve:
#
#   julia --startup-file=no --project=docs/benchmark \
#     docs/benchmark/adaptive_centering_fixed_c_run.jl
#
# Optional environment knobs:
#   ACE_SEEDS=8             common chain seeds per arm and family
#   ACE_DRAWS=1000          retained-draw floor per chain
#   ACE_EVALUATIONS=1000    first warmup window's gradient-evaluation target
#   ACE_FAMILIES=gaussian,student
#   ACE_OUT=docs/benchmark/results/adaptive_centering_fixed_c/rows.json
#
# Every arm represents the same density in a different fixed triangular
# coordinate chart. The target supplies its own analytic gradient; there is no
# reparametrization AD backend and no candidate adaptation in this benchmark.

using LinearAlgebra
BLAS.set_num_threads(1)

using JSON, LogDensityProblems, MCMCDiagnosticTools, Random, Statistics, WarmupHMC

const ACE_K = 3
const ACE_RCORR = [1.00 0.65 -0.25; 0.65 1.00 0.35; -0.25 0.35 1.00]
const ACE_L = Matrix(cholesky(Symmetric(ACE_RCORR)).L)
const ACE_ALPHA = [0.55, 0.85, 1.15]
const ACE_CSTAR = [0.2, 0.6, 0.9]
const ACE_NU = 5.0
const ACE_PARAMETER_NAMES = ["h", "q1", "q2", "q3"]
const ACE_ARMS = [
    (name = "exact_score_reference", c = [0.2, 0.6, 0.9]),
    (name = "invariant_proxy", c = [0.5, 0.7, 0.9]),
    (name = "whitened_noncentered", c = [0.0, 0.0, 0.0]),
    (name = "fully_centered", c = [1.0, 1.0, 1.0]),
]

struct ACEFixedTarget
    c::Vector{Float64}
    family::Symbol
end

LogDensityProblems.dimension(::ACEFixedTarget) = ACE_K + 1
LogDensityProblems.capabilities(::Type{ACEFixedTarget}) =
    LogDensityProblems.LogDensityOrder{1}()

ace_scales(h) = begin
    tau = exp.(h .* ACE_ALPHA)
    tau, tau .* diag(ACE_L)
end

"Map one arm's `(h,u)` coordinates to the common `(h,q*)` frame."
function ace_to_common(target::ACEFixedTarget, x::AbstractVector)
    h = x[1]
    tau, s = ace_scales(h)
    z = zeros(Float64, ACE_K)
    m = zeros(Float64, ACE_K)
    q = zeros(Float64, ACE_K)
    for k in 1:ACE_K
        m[k] = k == 1 ? 0.0 : tau[k] * dot(@view(ACE_L[k, 1:k-1]), @view(z[1:k-1]))
        z[k] = (x[k + 1] - target.c[k] * m[k]) / s[k]^target.c[k]
        q[k] = ACE_CSTAR[k] * m[k] + s[k]^ACE_CSTAR[k] * z[k]
    end
    [h; q]
end

"Map a common `(h,q*)` point into one arm's fixed `(h,u)` coordinates."
function ace_from_common(target::ACEFixedTarget, common::AbstractVector)
    h = common[1]
    q = @view common[2:end]
    tau, s = ace_scales(h)
    z = zeros(Float64, ACE_K)
    m = zeros(Float64, ACE_K)
    u = zeros(Float64, ACE_K)
    for k in 1:ACE_K
        m[k] = k == 1 ? 0.0 : tau[k] * dot(@view(ACE_L[k, 1:k-1]), @view(z[1:k-1]))
        z[k] = (q[k] - ACE_CSTAR[k] * m[k]) / s[k]^ACE_CSTAR[k]
        u[k] = target.c[k] * m[k] + s[k]^target.c[k] * z[k]
    end
    [h; u]
end

"`log|d q* / d u|` at fixed `h` for one arm."
function ace_common_logjac(target::ACEFixedTarget, h::Real)
    _, s = ace_scales(h)
    sum((ACE_CSTAR .- target.c) .* log.(s))
end

"Analytic log density and gradient in one arm's fixed coordinates."
function ace_value_gradient(target::ACEFixedTarget, x::AbstractVector)
    h = x[1]
    tau, s = ace_scales(h)
    c = target.c
    D = ACE_K + 1

    z = zeros(Float64, ACE_K)
    m = zeros(Float64, ACE_K)
    q = zeros(Float64, ACE_K)
    dm = zeros(Float64, ACE_K, D)
    dz = zeros(Float64, ACE_K, D)
    dq = zeros(Float64, ACE_K, D)

    for k in 1:ACE_K
        if k > 1
            lrow = @view ACE_L[k, 1:k-1]
            m[k] = tau[k] * dot(lrow, @view z[1:k-1])
            for p in 1:D
                dm[k, p] = tau[k] * dot(lrow, @view dz[1:k-1, p])
            end
            dm[k, 1] += ACE_ALPHA[k] * m[k]
        end

        d = s[k]^c[k]
        z[k] = (x[k + 1] - c[k] * m[k]) / d
        for p in 1:D
            dz[k, p] = ((p == k + 1 ? 1.0 : 0.0) - c[k] * dm[k, p]) / d
        end
        dz[k, 1] -= c[k] * ACE_ALPHA[k] * z[k]

        dstar = s[k]^ACE_CSTAR[k]
        q[k] = ACE_CSTAR[k] * m[k] + dstar * z[k]
        for p in 1:D
            dq[k, p] = ACE_CSTAR[k] * dm[k, p] + dstar * dz[k, p]
        end
        dq[k, 1] += ACE_CSTAR[k] * ACE_ALPHA[k] * dstar * z[k]
    end

    lp = -0.5 * (h / 1.2)^2
    gq = similar(q)
    if target.family === :gaussian
        lp -= 0.5 * sum(abs2, q)
        gq .= .-q
    elseif target.family === :student
        lp -= ((ACE_NU + 1) / 2) * sum(log1p.(q .^ 2 ./ ACE_NU))
        gq .= .-(ACE_NU + 1) .* q ./ (ACE_NU .+ q .^ 2)
    else
        error("unknown family $(target.family)")
    end
    lp += ace_common_logjac(target, h)

    gradient = vec(dq' * gq)
    gradient[1] += -h / 1.2^2 + sum((ACE_CSTAR .- c) .* ACE_ALPHA)
    lp, gradient
end

LogDensityProblems.logdensity(target::ACEFixedTarget, x) = first(ace_value_gradient(target, x))
LogDensityProblems.logdensity_and_gradient(target::ACEFixedTarget, x) =
    ace_value_gradient(target, x)

ace_fd_gradient(target, x) = begin
    g = similar(x, Float64)
    for p in eachindex(x)
        step = cbrt(eps(Float64)) * max(1.0, abs(x[p]))
        xp = copy(x); xp[p] += step
        xm = copy(x); xm[p] -= step
        g[p] = (LogDensityProblems.logdensity(target, xp) -
                LogDensityProblems.logdensity(target, xm)) / (2step)
    end
    g
end

function ace_validate(family::Symbol; n = 32, seed = 20260728)
    rng = Xoshiro(seed)
    reference = ACEFixedTarget(copy(ACE_CSTAR), family)
    max_frame = 0.0
    max_density = 0.0
    max_gradient = 0.0
    max_roundtrip = 0.0
    for _ in 1:n
        common = [1.2randn(rng); randn(rng, ACE_K)]
        xref = ace_from_common(reference, common)
        lpref = LogDensityProblems.logdensity(reference, xref)
        for arm in ACE_ARMS
            target = ACEFixedTarget(copy(arm.c), family)
            x = ace_from_common(target, common)
            recovered = ace_to_common(target, x)
            max_frame = max(max_frame, maximum(abs, recovered .- common))
            max_roundtrip = max(max_roundtrip,
                                maximum(abs, ace_from_common(target, recovered) .- x))
            residual = LogDensityProblems.logdensity(target, x) - lpref -
                       ace_common_logjac(target, x[1])
            max_density = max(max_density, abs(residual))
            _, g = LogDensityProblems.logdensity_and_gradient(target, x)
            max_gradient = max(max_gradient, maximum(abs, g .- ace_fd_gradient(target, x)))
        end
    end
    Dict(
        "n_points" => n,
        "max_common_frame_abs" => max_frame,
        "max_transport_logdensity_abs" => max_density,
        "max_gradient_fd_abs" => max_gradient,
        "max_coordinate_roundtrip_abs" => max_roundtrip,
    )
end

ace_samples_array(draws::AbstractMatrix) = reshape(permutedims(draws), size(draws, 2), 1, size(draws, 1))

function ace_run_chain(family::Symbol, arm, seed::Int, n_draws::Int, n_evaluations::Int)
    target = ACEFixedTarget(copy(arm.c), family)
    common_init = [0.0, 0.15, -0.20, 0.25]
    # An AbstractVector asks the shipping initializer to run Pathfinder from
    # this point. Every arm therefore begins at the same common model state,
    # while still exercising WarmupHMC's ordinary initialization path.
    init = ace_from_common(target, common_init)
    total_transitions = Ref(0)
    callback = (state, stage) -> begin
        total_transitions[] = state.total_transition_counter
        false
    end

    result = nothing
    wall_s = @elapsed result = adaptive_warmup_mcmc(
        Xoshiro(seed), target;
        init,
        n_draws,
        n_evaluations,
        nonlinear_adapt = false,
        monitor_ess = false,
        progress = nothing,
        callback,
    )

    source_draws = Matrix(result.posterior_position)
    common_draws = similar(source_draws)
    roundtrip_error = 0.0
    for j in axes(source_draws, 2)
        common_draws[:, j] .= ace_to_common(target, @view source_draws[:, j])
        roundtrip_error = max(roundtrip_error,
            maximum(abs, ace_from_common(target, @view common_draws[:, j]) .-
                         @view(source_draws[:, j])))
    end

    diag_array = ace_samples_array(common_draws)
    ess_bulk = vec(MCMCDiagnosticTools.ess(diag_array; kind = :bulk))
    ess_tail = vec(MCMCDiagnosticTools.ess(diag_array; kind = :tail))
    n = size(common_draws, 2)
    evals = result.total_evaluation_counter
    row = Dict(
        "row_type" => "chain",
        "family" => string(family),
        "arm" => arm.name,
        "c" => arm.c,
        "seed" => seed,
        "n_draws" => n,
        "n_divergent" => result.n_divergent_samples,
        "gradient_evaluations" => evals,
        "total_transitions" => total_transitions[],
        "gradients_per_draw" => evals / n,
        "gradients_per_transition" => evals / total_transitions[],
        "wall_seconds" => wall_s,
        "ess_bulk" => ess_bulk,
        "ess_tail" => ess_tail,
        "min_ess_bulk" => minimum(ess_bulk),
        "min_ess_tail" => minimum(ess_tail),
        "min_bulk_ess_per_1000_grad" => 1000minimum(ess_bulk) / evals,
        "min_tail_ess_per_1000_grad" => 1000minimum(ess_tail) / evals,
        "common_mean" => vec(mean(common_draws; dims = 2)),
        "common_sd" => vec(std(common_draws; dims = 2)),
        "returned_draw_roundtrip_max_abs" => roundtrip_error,
    )
    row, common_draws
end

function ace_pooled_row(family::Symbol, arm, chain_rows, chain_draws)
    ncommon = minimum(size(x, 2) for x in chain_draws)
    samples = Array{Float64}(undef, ncommon, length(chain_draws), ACE_K + 1)
    for (chain, draws) in enumerate(chain_draws)
        samples[:, chain, :] .= permutedims(@view(draws[:, 1:ncommon]))
    end
    ess_bulk = vec(MCMCDiagnosticTools.ess(samples; kind = :bulk))
    ess_tail = vec(MCMCDiagnosticTools.ess(samples; kind = :tail))
    rhat = vec(MCMCDiagnosticTools.rhat(samples; kind = :rank))
    Dict(
        "row_type" => "pooled",
        "family" => string(family),
        "arm" => arm.name,
        "c" => arm.c,
        "n_chains" => length(chain_draws),
        "n_common_draws_per_chain" => ncommon,
        "total_gradient_evaluations" => sum(r["gradient_evaluations"] for r in chain_rows),
        "total_divergent" => sum(r["n_divergent"] for r in chain_rows),
        "ess_bulk" => ess_bulk,
        "ess_tail" => ess_tail,
        "rhat_rank" => rhat,
        "min_ess_bulk" => minimum(ess_bulk),
        "min_ess_tail" => minimum(ess_tail),
        "max_rhat" => maximum(rhat),
    )
end

ace_jsonsafe(x::AbstractFloat) = isfinite(x) ? x : nothing
ace_jsonsafe(x::AbstractDict) = Dict(string(k) => ace_jsonsafe(v) for (k, v) in x)
ace_jsonsafe(x::AbstractVector) = [ace_jsonsafe(v) for v in x]
ace_jsonsafe(x) = x

function ace_git_sha()
    try
        readchomp(`git -C $(normpath(joinpath(@__DIR__, "..", ".."))) rev-parse HEAD`)
    catch
        "unknown"
    end
end

const ACE_N_SEEDS = parse(Int, get(ENV, "ACE_SEEDS", "8"))
const ACE_N_DRAWS = parse(Int, get(ENV, "ACE_DRAWS", "1000"))
const ACE_N_EVALUATIONS = parse(Int, get(ENV, "ACE_EVALUATIONS", "1000"))
const ACE_FAMILIES = Symbol.(split(get(ENV, "ACE_FAMILIES", "gaussian,student"), ","))
const ACE_OUT = get(ENV, "ACE_OUT",
    joinpath(@__DIR__, "results", "adaptive_centering_fixed_c", "rows.json"))
const ACE_SEEDS = collect(2026072801:(2026072800 + ACE_N_SEEDS))

validation = Dict(string(family) => ace_validate(family) for family in ACE_FAMILIES)
rows = Any[]

# Compile each target/arm specialization before timing. These smoke draws are
# intentionally not retained in the artifact.
for family in ACE_FAMILIES, arm in ACE_ARMS
    ace_run_chain(family, arm, 2026072799, 30, min(ACE_N_EVALUATIONS, 100))
end

for family in ACE_FAMILIES
    stored_rows = Dict(arm.name => Any[] for arm in ACE_ARMS)
    stored_draws = Dict(arm.name => Matrix{Float64}[] for arm in ACE_ARMS)
    for (seed_index, seed) in enumerate(ACE_SEEDS)
        # Rotate sequential arm order so machine drift cannot systematically
        # favour one chart. Chains are never run concurrently.
        arm_order = circshift(ACE_ARMS, -(seed_index - 1) % length(ACE_ARMS))
        for arm in arm_order
            @info "adaptive-centering fixed-c" family arm=arm.name seed
            row, draws = ace_run_chain(family, arm, seed, ACE_N_DRAWS, ACE_N_EVALUATIONS)
            push!(stored_rows[arm.name], row)
            push!(stored_draws[arm.name], draws)
            push!(rows, row)
            ndraws = row["n_draws"]
            ndivergent = row["n_divergent"]
            gradients = row["gradient_evaluations"]
            bulk_per_kgrad = row["min_bulk_ess_per_1000_grad"]
            tail_per_kgrad = row["min_tail_ess_per_1000_grad"]
            @info "  result" ndraws ndivergent gradients bulk_per_kgrad tail_per_kgrad
        end
    end
    for arm in ACE_ARMS
        push!(rows, ace_pooled_row(family, arm,
            stored_rows[arm.name], stored_draws[arm.name]))
    end
end

config = Dict(
    "warmuphmc_sha" => ace_git_sha(),
    "julia" => string(VERSION),
    "blas_threads" => BLAS.get_num_threads(),
    "host" => get(ENV, "KB_HOST", "unknown"),
    "runner" => "docs/benchmark/adaptive_centering_fixed_c_run.jl",
    "reproduction" => "ACE_SEEDS=16 ACE_DRAWS=5000 ACE_EVALUATIONS=1000 ACE_FAMILIES=gaussian,student julia --startup-file=no --history-file=no --project=docs/benchmark docs/benchmark/adaptive_centering_fixed_c_run.jl",
    "sampler" => "adaptive_warmup_mcmc",
    "target_gradient" => "analytic",
    "ad_backend" => "not applicable (target supplies logdensity_and_gradient)",
    "n_seeds" => ACE_N_SEEDS,
    "seeds" => ACE_SEEDS,
    "n_draws_floor" => ACE_N_DRAWS,
    "n_evaluations" => ACE_N_EVALUATIONS,
    "families" => string.(ACE_FAMILIES),
    "parameter_names" => ACE_PARAMETER_NAMES,
    "cstar" => ACE_CSTAR,
    "cstar_role" => "exact-score choice and fully factorized reference geometry by target construction; not an ESS optimum found by search",
    "arms" => Dict(arm.name => arm.c for arm in ACE_ARMS),
    "alpha" => ACE_ALPHA,
    "correlation" => ACE_RCORR,
    "cholesky" => ACE_L,
    "student_df" => ACE_NU,
    "common_initial_position" => [0.0, 0.15, -0.20, 0.25],
    "init_strategy" => "Pathfinder from an arm-transported common model-frame point",
    "nonlinear_adapt" => false,
    "arm_execution" => "sequential, rotated by common seed",
    "efficiency_denominator" => "full-run gradient evaluations, including adaptation",
    "materiality_rule" => Dict(
        "metric" => "paired per-chain minimum bulk ESS per 1000 gradients",
        "materially_worse" => "median proxy/optimum < 0.8 and at least 75% of seeds < 0.8",
        "materially_better" => "median proxy/optimum > 1.25 and at least 75% of seeds > 1.25",
        "otherwise" => "inconclusive at the predeclared 20% threshold",
    ),
    "validation" => validation,
    "representativeness" => "synthetic K=3 stress/control with BRM-like draw-varying scales and dense correlation; not a fitted BRM posterior",
)

mkpath(dirname(ACE_OUT))
open(ACE_OUT, "w") do io
    JSON.print(io, ace_jsonsafe(Dict("config" => config, "rows" => rows)), 2)
end
println("wrote ", ACE_OUT, " (", length(rows), " rows)")
