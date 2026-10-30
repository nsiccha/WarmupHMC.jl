# Continuously adapting NUTS against the frozen kernel on SAEM-shaped changing
# targets. Run after resolving the test environment (see .github/workflows/test.yml):
#   julia --project=test bench/continuous_nuts_saem.jl
#
# One "subject" holds J occasion effects a_j and J·K unit effects b_jk nested in
# them, with n_rep observations per unit. The target at SAEM iteration t is the
# conditional posterior p(η | y, θ_t, σ_t, ω_t): the hyperparameters start off
# (right, 4× too wide, 4× too narrow in every scale, and θ one unit off), move
# geometrically toward the truth over the first 150 iterations (the SA phase) and
# stay there for 150 more (the averaging phase). The :jump start instead holds
# the wide hyperparameters for 50 iterations and then switches to the truth in
# one step, the move an SAEM M-step with unit gain makes; its readout is how fast
# each arm tracks again. After the path every arm takes 1000 more transitions on
# the final, stationary target, which measures what continuous adaptation costs
# or buys once nothing moves. Two likelihoods: Gaussian, whose conditional is an
# exact Gaussian with a block-arrow precision, and Poisson, whose conditional is
# not Gaussian and is compared with a long reference run.
#
# Arms:
#   frozen        NLMEEstimation's shape at snag `continuous-nuts-15212ad7`: one
#                 `adaptive_warmup_mcmc` at the start target (n_draws=20,
#                 n_evaluations=100, stepsize_adaptation_limit=20, max_tree_depth=8,
#                 init from the conditional Hessian diagonal), then
#                 `fixed_kernel_nuts_step` for every iteration.
#   continuous    `ContinuousNUTSState` from Pathfinder at the start target, one
#                 `continuous_nuts_step` per iteration; variants sweep the
#                 forgetting half-life and the start-scale pseudo-count.
#   rewarmed      the frozen arm's warm-up repeated at the FINAL target: the per-
#                 transition cost a kernel tuned to the final target pays there.
#
# Reported per (likelihood, size, start, arm): gradient evaluations of the start
# (Pathfinder, or the warm-up), per transition in each phase, divergences, the
# Mahalanobis distance² / d of the position under the current exact conditional
# (Gaussian only; 1 when the chain tracks; for :jump also over the 5, 10 and 20
# iterations after the jump), over the averaging phase the ESS of the 150 draws
# and the error of their mean against the exact / reference posterior in
# posterior-SD units, and over the stationary tail the minimum and median
# coordinate ESS per 1000 gradient evaluations. Every count is deterministic per
# seed; only wall-clock would vary, and none is reported.

using WarmupHMC, LogDensityProblems, Random, LinearAlgebra, Statistics
const MCMCDiagnosticTools = WarmupHMC.MCMCDiagnosticTools

BLAS.set_num_threads(1)

# ---------------------------------------------------------------- the subject

struct Subject
    J::Int
    K::Int
    n_rep::Int
    y::Matrix{Float64}     # (J*K) × n_rep observations (counts for Poisson)
    occasion::Vector{Int}  # unit index -> occasion index
end
dim(s::Subject) = s.J + s.J * s.K
unit_range(s::Subject) = (s.J + 1):dim(s)

function simulate(J, K, n_rep; θ=1.0, σ=0.3, ωa=0.5, ωb=0.3, poisson=false, rng)
    a = ωa .* randn(rng, J)
    b = ωb .* randn(rng, J * K)
    occasion = repeat(1:J; inner=K)
    η = θ .+ a[occasion] .+ b
    y = if poisson
        [Float64(rand(rng, WarmupHMC.Distributions.Poisson(exp(η[u])))) for u in eachindex(η), _ in 1:n_rep]
    else
        η * ones(n_rep)' .+ σ .* randn(rng, J * K, n_rep)
    end
    Subject(J, K, n_rep, y, occasion)
end

# Hyperparameters (θ, σ, ωa, ωb); σ is ignored by the Poisson target.
struct Hyper
    θ::Float64
    σ::Float64
    ωa::Float64
    ωb::Float64
end

abstract type Conditional end
struct GaussianConditional <: Conditional
    s::Subject
    h::Hyper
end
struct PoissonConditional <: Conditional
    s::Subject
    h::Hyper
end
LogDensityProblems.capabilities(::Type{<:Conditional}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::Conditional) = dim(p.s)
LogDensityProblems.logdensity(p::Conditional, η) = first(LogDensityProblems.logdensity_and_gradient(p, η))

function LogDensityProblems.logdensity_and_gradient(p::GaussianConditional, η)
    (; s, h) = p
    J = s.J
    g = zeros(length(η))
    lp = 0.0
    @inbounds for j in 1:J
        lp -= abs2(η[j]) / (2 * abs2(h.ωa))
        g[j] -= η[j] / abs2(h.ωa)
    end
    @inbounds for (u, idx) in enumerate(unit_range(s))
        j = s.occasion[u]
        b = η[idx]
        lp -= abs2(b) / (2 * abs2(h.ωb))
        g[idx] -= b / abs2(h.ωb)
        m = h.θ + η[j] + b
        for r in 1:s.n_rep
            e = (s.y[u, r] - m) / h.σ
            lp -= abs2(e) / 2
            d = e / h.σ
            g[idx] += d
            g[j] += d
        end
    end
    lp, g
end

function LogDensityProblems.logdensity_and_gradient(p::PoissonConditional, η)
    (; s, h) = p
    J = s.J
    g = zeros(length(η))
    lp = 0.0
    @inbounds for j in 1:J
        lp -= abs2(η[j]) / (2 * abs2(h.ωa))
        g[j] -= η[j] / abs2(h.ωa)
    end
    @inbounds for (u, idx) in enumerate(unit_range(s))
        j = s.occasion[u]
        b = η[idx]
        lp -= abs2(b) / (2 * abs2(h.ωb))
        g[idx] -= b / abs2(h.ωb)
        m = h.θ + η[j] + b
        λ = exp(m)
        for r in 1:s.n_rep
            lp += s.y[u, r] * m - λ
            d = s.y[u, r] - λ
            g[idx] += d
            g[j] += d
        end
    end
    lp, g
end

conditional(poisson::Bool, s, h) = poisson ? PoissonConditional(s, h) : GaussianConditional(s, h)

# Diagonal of the negative Hessian at the mode (Gaussian: exact and constant).
function hessian_diag(p::GaussianConditional)
    (; s, h) = p
    d = dim(s)
    hd = zeros(d)
    hd[1:s.J] .= s.K * s.n_rep / abs2(h.σ) + 1 / abs2(h.ωa)
    hd[unit_range(s)] .= s.n_rep / abs2(h.σ) + 1 / abs2(h.ωb)
    hd
end
# Poisson: the Hessian at η = 0 (what a Laplace-style start would take).
function hessian_diag(p::PoissonConditional)
    (; s, h) = p
    d = dim(s)
    hd = zeros(d)
    λ = exp(h.θ)
    hd[1:s.J] .= s.K * s.n_rep * λ + 1 / abs2(h.ωa)
    hd[unit_range(s)] .= s.n_rep * λ + 1 / abs2(h.ωb)
    hd
end

# Exact conditional of the Gaussian target, block by block: mean, and the
# Cholesky factor of each occasion block's precision for Mahalanobis distances.
struct ExactGaussian
    mean::Vector{Float64}
    sd::Vector{Float64}
    blocks::Vector{Tuple{Vector{Int},Cholesky{Float64,Matrix{Float64}}}}
end
function exact(p::GaussianConditional)
    (; s, h) = p
    d = dim(s)
    μ = zeros(d)
    sd = zeros(d)
    blocks = Tuple{Vector{Int},Cholesky{Float64,Matrix{Float64}}}[]
    for j in 1:s.J
        units = findall(==(j), s.occasion)
        idx = vcat(j, s.J .+ units)
        n = length(idx)
        P = zeros(n, n)
        rhs = zeros(n)
        P[1, 1] = s.K * s.n_rep / abs2(h.σ) + 1 / abs2(h.ωa)
        for (k, u) in enumerate(units)
            P[1 + k, 1 + k] = s.n_rep / abs2(h.σ) + 1 / abs2(h.ωb)
            P[1, 1 + k] = P[1 + k, 1] = s.n_rep / abs2(h.σ)
            r = sum(s.y[u, :] .- h.θ) / abs2(h.σ)
            rhs[1 + k] = r
            rhs[1] += r
        end
        C = cholesky(Symmetric(P))
        μ[idx] = C \ rhs
        Σ = inv(C)
        sd[idx] = sqrt.(diag(Σ))
        push!(blocks, (idx, C))
    end
    ExactGaussian(μ, sd, blocks)
end
function mahalanobis2(e::ExactGaussian, η)
    total = 0.0
    for (idx, C) in e.blocks
        r = η[idx] .- e.mean[idx]
        total += sum(abs2, C.U * r)   # rᵀ P r with P = Uᵀ U
    end
    total
end

# ------------------------------------------------------------- the SAEM path

const N_SA = 150
const N_AVG = 150
const N_ITER = N_SA + N_AVG
const N_TAIL = 1000
const N_JUMP = 50   # :jump switches from the wide start to the truth after this iteration
const RHO = 0.97   # 0.97^150 ≈ 0.01: converged by the end of the SA phase

truth(poisson) = Hyper(poisson ? 0.5 : 1.0, 0.3, 0.5, 0.3)
function start_hyper(poisson, start)
    t = truth(poisson)
    f = start === :right ? 1.0 : start === :narrow ? 0.25 : 4.0   # :wide and :jump
    Hyper(t.θ + 1.0, t.σ * f, t.ωa * f, t.ωb * f)
end
function hyper_at(poisson, start, t)
    h0, h1 = start_hyper(poisson, start), truth(poisson)
    start === :jump && return t <= N_JUMP ? h0 : h1
    w = t <= N_SA ? RHO^t : 0.0
    interp(a, b) = b + (a - b) * w
    # Scales interpolate on the log scale, as SAEM's variance updates move them.
    Hyper(interp(h0.θ, h1.θ), exp(interp(log(h0.σ), log(h1.σ))),
        exp(interp(log(h0.ωa), log(h1.ωa))), exp(interp(log(h0.ωb), log(h1.ωb))))
end

# ------------------------------------------------------------ counting target

mutable struct Counting{P}
    inner::P
    n::Int
end
LogDensityProblems.capabilities(::Type{<:Counting{P}}) where {P} = LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(c::Counting) = LogDensityProblems.dimension(c.inner)
LogDensityProblems.logdensity(c::Counting, η) = (c.n += 1; LogDensityProblems.logdensity(c.inner, η))
LogDensityProblems.logdensity_and_gradient(c::Counting, η) = (c.n += 1; LogDensityProblems.logdensity_and_gradient(c.inner, η))

# ------------------------------------------------------------------- the arms

const FROZEN_WARMUP = (; n_draws=20, n_evaluations=100, stepsize_adaptation_limit=20, max_tree_depth=8)

# Warm up at `target` from the Hessian start, as NLMEEstimation does; return the
# fixed-kernel state and the gradient evaluations the warm-up took.
function frozen_start(rng, target)
    c = Counting(target, 0)
    hd = hessian_diag(target)
    dir = mktempdir()
    adaptive_warmup_mcmc(rng, c; init=(; position=zeros(dim(target.s)), squared_scale=1 ./ hd),
        checkpoint_dir=dir, progress=nothing, FROZEN_WARMUP...)
    state = FixedKernelNUTSState(joinpath(dir, "cp_latest.jls"); max_tree_depth=FROZEN_WARMUP.max_tree_depth)
    rm(dir; recursive=true)
    state, c.n
end

step_arm!(state::FixedKernelNUTSState, target) = begin
    r = fixed_kernel_nuts_step(state, target)
    r.state, r.stats.steps, r.divergent
end
step_arm!(state::ContinuousNUTSState, target) = begin
    r = continuous_nuts_step(state, target)
    r.state, r.stats.steps, r.divergent
end

ess_all(draws::AbstractMatrix) = begin
    d, n = size(draws)
    MCMCDiagnosticTools.ess(reshape(permutedims(draws), n, 1, d))
end
ess_min(draws::AbstractMatrix) = minimum(ess_all(draws))

# Run one arm along the path. `make_state(rng, target1)` returns (state, start_evals);
# `exacts[t]` is the exact conditional at iteration t (Gaussian), or `nothing`.
function run_path(make_state, seed, poisson, start, subject, reference, exacts)
    rng = Xoshiro(seed)
    target1 = conditional(poisson, subject, hyper_at(poisson, start, 0))
    state, start_evals = make_state(rng, target1)
    d = dim(subject)
    evals = zeros(Int, N_ITER)
    divergent = zeros(Bool, N_ITER)
    mahal = fill(NaN, N_ITER)
    draws = zeros(d, N_AVG)
    for t in 1:N_ITER
        target = conditional(poisson, subject, hyper_at(poisson, start, t))
        state, evals[t], divergent[t] = step_arm!(state, target)
        isnothing(exacts) || (mahal[t] = mahalanobis2(exacts[t], state.position) / d)
        t > N_SA && (draws[:, t - N_SA] = state.position)
    end
    refμ, refsd = reference.mean, reference.sd
    err = sqrt(mean(abs2, (vec(mean(draws; dims=2)) .- refμ) ./ refsd))
    sd_ratio = median(vec(std(draws; dims=2)) ./ refsd)
    # The stationary tail on the final target.
    final = conditional(poisson, subject, hyper_at(poisson, start, N_ITER))
    tail = zeros(d, N_TAIL)
    tail_evals = 0
    for t in 1:N_TAIL
        state, steps, _ = step_arm!(state, final)
        tail_evals += steps
        tail[:, t] = state.position
    end
    tail_ess = ess_all(tail)
    (; start_evals, sa_evals=mean(evals[1:N_SA]), avg_evals=mean(evals[N_SA+1:end]),
       early_evals=mean(evals[1:10]), n_divergent=sum(divergent),
       mahal_sa=mean(mahal[1:N_SA]), mahal_avg=mean(mahal[N_SA+1:end]),
       mahal_jump=(mean(mahal[N_JUMP+1:N_JUMP+5]), mean(mahal[N_JUMP+6:N_JUMP+10]), mean(mahal[N_JUMP+11:N_JUMP+20])),
       ess=ess_min(draws), mean_err=err, sd_ratio,
       tail_ess_min=minimum(tail_ess) / tail_evals * 1000, tail_ess_med=median(tail_ess) / tail_evals * 1000)
end

# The posterior the averaging-phase draws are compared with: exact for the
# Gaussian target, a long adaptive run for the Poisson one.
function reference_posterior(poisson, subject)
    target = conditional(poisson, subject, truth(poisson))
    if !poisson
        e = exact(target)
        return (; mean=e.mean, sd=e.sd)
    end
    fit = adaptive_warmup_mcmc(Xoshiro(2026), target; n_draws=4000, progress=nothing)
    draws = Matrix(fit.posterior_position)
    (; mean=vec(mean(draws; dims=2)), sd=vec(std(draws; dims=2)))
end

# Per-transition cost of the frozen arm's warm-up repeated at the final target.
function rewarmed_cost(seed, poisson, subject)
    target = conditional(poisson, subject, truth(poisson))
    state, start_evals = frozen_start(Xoshiro(seed), target)
    evals = 0
    for _ in 1:50
        state, steps, _ = step_arm!(state, target)
        evals += steps
    end
    (; start_evals, evals=evals / 50)
end

# ------------------------------------------------------------------ the sweep

fmt(x; digits=2) = x isa Integer ? string(x) : isnan(x) ? "-" : string(round(x; digits))

const SIZES = [(10, 24), (20, 49), (40, 99)]   # d = 250, 1000, 4000
const STARTS = (:right, :wide, :narrow, :jump)
const SWEEP_SIZES = SIZES[1:2]   # the half-life / prior-weight sweeps run here, seed 1
const ARMS = [
    "frozen" => (rng, t) -> frozen_start(rng, t),
    "continuous" => (rng, t) -> begin
        c = Counting(t, 0)
        s = ContinuousNUTSState(rng, c; init=zeros(dim(t.s)))
        s, c.n
    end,
    "continuous hl=10" => (rng, t) -> (ContinuousNUTSState(rng, t; init=zeros(dim(t.s)), half_life=10), 0),
    "continuous hl=30" => (rng, t) -> (ContinuousNUTSState(rng, t; init=zeros(dim(t.s)), half_life=30), 0),
    "continuous hl=50" => (rng, t) -> (ContinuousNUTSState(rng, t; init=zeros(dim(t.s)), half_life=50), 0),
    "continuous hl=Inf" => (rng, t) -> (ContinuousNUTSState(rng, t; init=zeros(dim(t.s)), half_life=Inf), 0),
    "continuous pw=0" => (rng, t) -> (ContinuousNUTSState(rng, t; init=zeros(dim(t.s)), prior_weight=0), 0),
    "continuous pw=100" => (rng, t) -> (ContinuousNUTSState(rng, t; init=zeros(dim(t.s)), prior_weight=100), 0),
    "continuous hessian start" => (rng, t) -> (ContinuousNUTSState(rng, t;
        init=(; position=zeros(dim(t.s)), squared_scale=1 ./ hessian_diag(t))), 0),
]

println("WarmupHMC continuous NUTS on SAEM-shaped changing targets; Julia $VERSION; ",
    "$N_SA SA + $N_AVG averaging iterations + $N_TAIL stationary, scales 4x off at :wide/:narrow/:jump, θ one unit off at every start; ",
    ":jump switches to the truth after iteration $N_JUMP")
println(join(["likelihood", "d", "start", "arm", "seed", "start_evals", "evals/transition 1-10", "evals/transition SA",
    "evals/transition avg", "divergent", "mahal²/d SA", "mahal²/d avg", "mahal²/d jump+1-5", "jump+6-10", "jump+11-20",
    "min ESS (150 draws)", "mean err (SD units)", "median sd ratio",
    "tail min ESS / 1000 evals", "tail median ESS / 1000 evals"], "\t"))
for poisson in (false, true), (J, K) in SIZES
    tr = truth(poisson)
    subject = simulate(J, K, 3; θ=tr.θ, σ=tr.σ, ωa=tr.ωa, ωb=tr.ωb, poisson, rng=Xoshiro(J * 1000 + K))
    d = dim(subject)
    reference = reference_posterior(poisson, subject)
    for start in STARTS
        exacts = poisson ? nothing :
            [exact(conditional(poisson, subject, hyper_at(poisson, start, t))) for t in 1:N_ITER]
        for seed in 1:2
        for (name, make) in ARMS
            # The sweeps over half-life and prior weight run on the first seed and the two smaller sizes only.
            startswith(name, "continuous ") && (seed > 1 || !((J, K) in SWEEP_SIZES)) && continue
            r = run_path(make, seed, poisson, start, subject, reference, exacts)
            jump = start === :jump ? map(fmt, r.mahal_jump) : ("-", "-", "-")
            println(join([poisson ? "poisson" : "gaussian", d, start, name, seed, r.start_evals,
                fmt(r.early_evals; digits=1), fmt(r.sa_evals; digits=1), fmt(r.avg_evals; digits=1), r.n_divergent,
                fmt(r.mahal_sa), fmt(r.mahal_avg), jump..., fmt(r.ess; digits=0), fmt(r.mean_err), fmt(r.sd_ratio),
                fmt(r.tail_ess_min), fmt(r.tail_ess_med)], "\t"))
        end
        if seed == 1 && start === :right
            rw = rewarmed_cost(seed, poisson, subject)
            println(join([poisson ? "poisson" : "gaussian", d, start, "rewarmed at final target", seed, rw.start_evals,
                "-", "-", fmt(rw.evals; digits=1), "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"], "\t"))
        end
        end
    end
end
