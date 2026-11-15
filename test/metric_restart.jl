@testmodule MetricRestartFixtures begin
using LogDensityProblems, LinearAlgebra, Random
export CorrGaussian, corr_gaussian, gaussian_halo

# A Gaussian with a dense precision `Λ`: a mechanistic probe of the restart test
# (snag adaptive-metric-ec8b47d3), not a verdict target.
struct CorrGaussian
    Λ::Matrix{Float64}
end
LogDensityProblems.capabilities(::Type{CorrGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::CorrGaussian) = size(p.Λ, 1)
LogDensityProblems.logdensity(p::CorrGaussian, x) = -dot(x, p.Λ * x) / 2
LogDensityProblems.logdensity_and_gradient(p::CorrGaussian, x) = (g = -(p.Λ * x); (dot(x, g) / 2, g))

# Marginal scales spanning e^±2 and two directions `narrow` times narrower than
# the narrowest coordinate, across coordinates 1:5 and 6:9. Also returns the
# diagonal fit's own fixed point, D² = sqrt(Σᵢᵢ / Λᵢᵢ): a start with that diagonal
# makes every marginal-scale correction ≈ 1 from the first window.
function corr_gaussian(seed; d = 12, narrow = 30.0)
    rng = Xoshiro(seed)
    s = exp.(randn(rng, d))
    u1 = zeros(d); u1[1:5] .= randn(rng, 5); normalize!(u1)
    u2 = zeros(d); u2[6:9] .= randn(rng, 4); normalize!(u2)
    Λ = Matrix(Diagonal(1 ./ s .^ 2)) + narrow^2 * maximum(1 ./ s .^ 2) * (u1 * u1' + u2 * u2')
    Λ = (Λ + Λ') / 2
    Σ = inv(Λ)
    CorrGaussian(Λ), sqrt.(diag(Σ) ./ diag(Λ))
end

# Exact draws and their gradients, the evidence a well-mixed window's halo holds.
function gaussian_halo(p::CorrGaussian, n; rng = Xoshiro(1))
    P = cholesky(Symmetric(inv(p.Λ))).L * randn(rng, size(p.Λ, 1), n)
    P, -(p.Λ * P)
end
end

@testitem "metric-change restart test: what the marginal-scale test cannot see" setup=[MetricRestartFixtures] begin
    using Test, WarmupHMC, LinearAlgebra, Random, Statistics
    using WarmupHMC: _metric_change_condition, _initial_diagonal_scale, _initial_pathfinder_scale,
        MatrixFactorization, SuccessiveReflections, update_loss!
    nrefl(L) = L isa MatrixFactorization && L.m1 isa SuccessiveReflections ? length(L.m1.reflections) : 0
    candidates(sq) = (; diagonal = _initial_diagonal_scale(Diagonal(sq)),
        pathfinder = _initial_pathfinder_scale(Diagonal(sq), length(sq)),
        adaptive = MatrixFactorization(SuccessiveReflections(length(sq)), Diagonal(ones(length(sq)))))

    # A correlated halo with the active diagonal at its own fixed point: the
    # marginal scales fit, the shape does not. The restart would install the
    # reflections, which fit the halo, and differ from the diagonal, far beyond
    # the default `variance_cond_target` of 2.
    p, sq = corr_gaussian(1)
    P, G = gaussian_halo(p, 1000)
    so = candidates(sq)
    P0, G0, so0 = copy(P), copy(G), deepcopy(so)
    κ = _metric_change_condition(so, :diagonal, P, G)
    @test κ > 4
    # Nothing it reads is touched: the halo, and every candidate of the run.
    @test P == P0 && G == G0
    @test so.diagonal == so0.diagonal && nrefl(so.adaptive) == 0
    # The same from the `:pathfinder` entry a diagonal start makes active.
    @test _metric_change_condition(so, :pathfinder, P, G) > 4

    # Once the active `:adaptive` metric holds the reflections, refitting it on
    # the same evidence fits no better: no restart for it.
    update_loss!(so.adaptive, copy(P), copy(G))
    @test nrefl(so.adaptive) >= 2
    @test _metric_change_condition(so, :adaptive, P, G) ≈ 1 atol = 1e-8

    # The active frame's loss as it stands, streamed column by column: the dense
    # computation, and for a refit candidate the loss its refit reports (the
    # diagonal part does not change it).
    L = so.adaptive
    @test WarmupHMC._frame_loss(L, P, G) ≈ mean(std.(eachrow(L \ P)) .* std.(eachrow(L' * G))) rtol = 1e-10
    D = deepcopy(so.diagonal)
    @test WarmupHMC._frame_loss(D, P, G) ≈ update_loss!(D, copy(P), copy(G)) rtol = 1e-10

    # An uncorrelated posterior: the restart would refit only the diagonal,
    # which is the marginal-scale test's to judge.
    q = CorrGaussian(Matrix(Diagonal(exp.(2 .* randn(Xoshiro(2), 12)))))
    Pq, Gq = gaussian_halo(q, 1000)
    @test _metric_change_condition(candidates(ones(12)), :diagonal, Pq, Gq) == 1
end

@testitem "a window whose marginal scales fit restarts to install the reflections" setup=[MetricRestartFixtures] begin
    using Test, WarmupHMC, LinearAlgebra, Random
    using WarmupHMC: MatrixFactorization, SuccessiveReflections
    nrefl(L) = L isa MatrixFactorization && L.m1 isa SuccessiveReflections ? length(L.m1.reflections) : 0
    # Started at the diagonal's fixed point, every marginal-scale correction is
    # within `variance_cond_target`: before this test existed the run never
    # restarted and sampled the whole run on the start's diagonal (569 gradient
    # evaluations per draw on this target and seed, against 9 with reflections).
    p, sq = corr_gaussian(1)
    windows = []
    cb = (state, stage) -> (stage === :window &&
        push!(windows, (; marginal = first(state.scale_changes)^2, restart = state.restart)); false)
    r = adaptive_warmup_mcmc(Xoshiro(1), p; init = (; position = zeros(12), squared_scale = sq),
        n_draws = 300, callback = cb, monitor_ess = false)
    @test windows[1].restart && windows[1].marginal < 2
    @test r.active_transformation === :adaptive
    @test nrefl(r.scale_options.adaptive) >= 2
    @test r.sampling_evaluation_counter / size(r.posterior_position, 2) < 50
end
