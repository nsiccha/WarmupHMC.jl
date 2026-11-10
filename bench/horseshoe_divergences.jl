# Divergences of the regularised horseshoe under `completion_warmup_mcmc`
# (snag regularised-hors-bcdcdd84), and what opt-in step-size halving
# (`divergence_tolerance`), a higher `target_acceptance_rate` and adaptive
# partial centering of the non-centred coefficients do about them.
#
#   julia --project=bench bench/horseshoe_divergences.jl [arms] [seeds] [out.tsv]
#
# Any environment with WarmupHMC, Enzyme, DifferentiationInterface,
# LogDensityProblems and MCMCDiagnosticTools works. Defaults: every arm below,
# dataset seeds 1-6, rows to stdout only. Each fit is 4 chains, quorum 3,
# 300 draws, as the reporter ran it. Arms are `_`-joined tokens:
#
#   default        WarmupHMC defaults
#   halve          divergence_tolerance = 0 (halve on any divergent final draw)
#   halve01        divergence_tolerance = 0.01
#   tar95, tar99   target_acceptance_rate = 0.95 / 0.99
#   center         ReparametrizedProblem with one adaptive PartiallyCentered pair
#                  per z_beta[j], log scale log(tau * lambda_tilde_j)
#
# e.g. `center_halve`. Columns: arm, seed, admitted chains, divergent draws, draws,
# % divergent, per-chain divergences, max R-hat, min bulk ESS, min tail ESS,
# sampling and total gradient evaluations, per-chain step sizes, per-chain
# step-size halvings, per-chain effective target acceptance, restarts that came
# after a halving (when the recorded effective target was used), seconds.
using LinearAlgebra, Random, Statistics, Printf
using LogDensityProblems, DifferentiationInterface, Enzyme, MCMCDiagnosticTools
using WarmupHMC

# The model: regularised-horseshoe linear regression with group effects
# (Piironen and Vehtari 2017), written as a plain LogDensityProblems target.
# Unconstrained coordinates (D predictors, G groups):
#   1            alpha                       ~ Normal(0, 2)
#   2            log sigma,  sigma            ~ Exponential(1)
#   3            log tau,    tau              ~ C+(0, tau0)
#   4:3+D        log lambda, lambda[j]        ~ C+(0, 1)
#   4+D          log c2,     c2               ~ InvGamma(df/2, df*s^2/2)
#   5+D:4+2D     z_beta[j]                    ~ Normal(0, 1)   (non-centred)
#   5+2D         log sigma_g, sigma_g         ~ Normal+(0, 1)
#   6+2D:5+2D+G  z_g[k]                       ~ Normal(0, 1)   (non-centred)
# beta_j = tau * lt_j * z_j with lt_j = sqrt(c2 lambda_j^2 / (c2 + tau^2 lambda_j^2)),
# u_k = sigma_g * z_g[k], y_i ~ Normal(alpha + X_i beta + u[g_i], sigma).
# Log densities are up to additive constants.
struct HSData
    X::Matrix{Float64}
    y::Vector{Float64}
    g::Vector{Int}
    G::Int
    tau0::Float64
    df::Float64
    s::Float64
end

idx_tau(d) = 3
idx_lambda(d, j) = 3 + j
idx_c2(d) = 4 + size(d.X, 2)
idx_z(d, j) = 4 + size(d.X, 2) + j
idx_sg(d) = 5 + 2size(d.X, 2)
idx_zg(d, k) = 5 + 2size(d.X, 2) + k

halfcauchy_logkernel(x, s) = -log1p((x / s)^2)
invgamma_logkernel(x, a, b) = -(a + 1) * log(x) - b / x

# log(tau * lambda_tilde_j) = lt + ll + lc/2 - log(c2 + tau^2 lambda^2)/2, in log space.
function log_hs_scale(x, d, j)
    lt, ll, lc = x[idx_tau(d)], x[idx_lambda(d, j)], x[idx_c2(d)]
    a, b = lc, 2lt + 2ll
    m = max(a, b)
    lt + ll + lc / 2 - (m + log(exp(a - m) + exp(b - m))) / 2
end

function hs_logdensity(x, d::HSData)
    n, D = size(d.X)
    alpha, lsig, lt = x[1], x[2], x[3]
    sigma, tau = exp(lsig), exp(lt)
    lc = x[idx_c2(d)]
    lsg = x[idx_sg(d)]
    lp = -alpha^2 / 8
    lp += -sigma + lsig
    lp += halfcauchy_logkernel(tau, d.tau0) + lt
    lp += invgamma_logkernel(exp(lc), d.df / 2, d.df * d.s^2 / 2) + lc
    lp += -exp(lsg)^2 / 2 + lsg
    beta = zeros(eltype(x), D)
    for j in 1:D
        ll = x[idx_lambda(d, j)]
        lp += halfcauchy_logkernel(exp(ll), 1.0) + ll
        z = x[idx_z(d, j)]
        lp += -z^2 / 2
        beta[j] = exp(log_hs_scale(x, d, j)) * z
    end
    for k in 1:d.G
        lp += -x[idx_zg(d, k)]^2 / 2
    end
    for i in 1:n
        mu = alpha + exp(lsg) * x[idx_zg(d, d.g[i])]
        for j in 1:D
            mu += d.X[i, j] * beta[j]
        end
        lp += -((d.y[i] - mu) / sigma)^2 / 2 - lsig
    end
    lp
end

struct HSProblem
    d::HSData
end
LogDensityProblems.dimension(p::HSProblem) = 5 + 2size(p.d.X, 2) + p.d.G
LogDensityProblems.capabilities(::Type{HSProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::HSProblem, x::AbstractVector) = hs_logdensity(x, p.d)
LogDensityProblems.logdensity_and_gradient(p::HSProblem, x::AbstractVector) =
    value_and_gradient(hs_logdensity, AutoEnzyme(), collect(float.(x)), Constant(p.d))

# Synthetic data: 60 observations, 8 AR(0.5)-correlated standardised predictors,
# two nonzero at ±0.25, 6 groups of 10 with sd 0.5, residual sd 1; tau0 from a
# prior guess of 2 relevant predictors, slab df = 4, s = 2.
function simulate(seed; n=60, D=8, G=6, rho=0.5, beta_nz=0.25, sigma=1.0, sg=0.5,
                  p0=2, df=4.0, s=2.0)
    rng = Xoshiro(seed)
    L = cholesky([rho^abs(i - j) for i in 1:D, j in 1:D]).L
    X = permutedims(L * randn(rng, D, n))
    X = (X .- mean(X; dims=1)) ./ std(X; dims=1)
    beta = zeros(D); beta[1] = beta_nz; beta[2] = -beta_nz
    g = repeat(1:G, inner=cld(n, G))[1:n]
    u = sg .* randn(rng, G)
    y = 0.3 .+ X * beta .+ u[g] .+ sigma .* randn(rng, n)
    HSData(X, y, g, G, p0 / (D - p0) / sqrt(n), df, s)
end

# Adaptive partial centering of z_beta[j] against its regularised scale, which is
# computed from three coordinates. A callable struct as the log-scale accessor.
struct HSLogScale
    d::HSData
    j::Int
end
(f::HSLogScale)(x) = log_hs_scale(x, f.d, f.j)

hs_centering(d::HSData) = IndexedReparametrization([
    idx_z(d, j) => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0),
                                     0.0, HSLogScale(d, j))
    for j in 1:size(d.X, 2)
])

function arm_settings(arm)
    tokens = split(arm, "_")
    known = ("default", "halve", "halve01", "tar95", "tar99", "center")
    for t in tokens
        t in known || error("unknown arm token $(repr(t)); known: $(join(known, ", "))")
    end
    target_acceptance_rate = "tar99" in tokens ? 0.99 : "tar95" in tokens ? 0.95 : 0.8
    divergence_tolerance = "halve" in tokens ? 0.0 : "halve01" in tokens ? 0.01 : 1.0
    (; center="center" in tokens, kw=(; target_acceptance_rate, divergence_tolerance))
end

function summarize(res)
    m = minimum(r.n_samples for r in res)
    # draws × chains × parameters
    arr = cat([permutedims(Matrix(r.posterior_position)[:, end-m+1:end]) for r in res]...; dims=3)
    arr = permutedims(arr, (1, 3, 2))
    (; rhat=maximum(MCMCDiagnosticTools.rhat(arr)),
       bulk=minimum(MCMCDiagnosticTools.ess(arr; kind=:bulk)),
       tail=minimum(MCMCDiagnosticTools.ess(arr; kind=:tail)))
end

function fit_row(arm, seed)
    (; center, kw) = arm_settings(arm)
    d = simulate(seed)
    lpdf = center ? ReparametrizedProblem(hs_centering(d), HSProblem(d), AutoEnzyme()) : HSProblem(d)
    rngs = [Xoshiro(1000seed + c) for c in 1:4]
    # A window that ends with `current_transition_counter == 0` restarted.
    restarts_after_halving = Threads.Atomic{Int}(0)
    count_restarts(state, stage) = begin
        stage === :window && state.stepsize_halvings > 0 && state.current_transition_counter == 0 &&
            Threads.atomic_add!(restarts_after_halving, 1)
        false
    end
    t = @elapsed out = completion_warmup_mcmc(rngs, lpdf; min_completed=3, n_draws=300,
                                               progress=nothing, callback=count_restarts, kw...)
    res = out.results
    s = summarize(res)
    ndiv = sum(r.n_divergent_samples for r in res)
    nret = sum(r.n_samples for r in res)
    join(Any[arm, seed, length(res), ndiv, nret, @sprintf("%.2f", 100ndiv / nret),
             join((r.n_divergent_samples for r in res), "/"),
             @sprintf("%.3f", s.rhat), round(Int, s.bulk), round(Int, s.tail),
             sum(r.sampling_evaluation_counter for r in res),
             sum(r.total_evaluation_counter for r in res),
             join((@sprintf("%.3g", r.stepsize) for r in res), "/"),
             join((r.stepsize_halvings for r in res), "/"),
             join((@sprintf("%.3f", r.effective_target_acceptance_rate) for r in res), "/"),
             restarts_after_halving[], @sprintf("%.1f", t)], '\t')
end

const ALL_ARMS = "default,halve,halve01,tar95,tar99,center,center_halve"
arms = split(get(ARGS, 1, ALL_ARMS), ",")
seeds = parse.(Int, split(get(ARGS, 2, "1,2,3,4,5,6"), ","))
out = get(ARGS, 3, nothing)
foreach(arm_settings, arms)  # refuse an unknown arm before the first fit
for seed in seeds, arm in arms
    line = fit_row(arm, seed)
    println(line); flush(stdout)
    isnothing(out) || open(io -> println(io, line), out, "a")
end
