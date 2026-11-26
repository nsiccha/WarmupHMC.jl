@testitem "degenerate metric evidence" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "degenerate_metric.jl"))'

# Snag `singular-diagona-b1788ed3`: a coordinate whose recorded positions never
# moved made `update_loss!(::Diagonal)` write a ZERO scale, and score that frame
# as the best possible fit, so the next momentum draw threw `SingularException`.

# x1 ~ N(0, 1), x2 | x1 ~ N(x1, 1): the gradient in x2 varies with x1 even
# while x2 itself does not move.
struct ChainGaussian end
LogDensityProblems.capabilities(::Type{ChainGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::ChainGaussian) = 2
LogDensityProblems.logdensity(::ChainGaussian, x) = -x[1]^2 / 2 - (x[2] - x[1])^2 / 2
LogDensityProblems.logdensity_and_gradient(p::ChainGaussian, x) =
    (LogDensityProblems.logdensity(p, x), [-x[1] + (x[2] - x[1]), -(x[2] - x[1])])

# Coordinate 2 starts with a scale far too small to move it from 1.0 in Float64,
# so the first window records it with zero position spread. A power of two, so
# the Pathfinder frame's `L \ p` is exact too and EVERY frame sees exactly zero
# spread (with 1e-40, summation rounding in `std` leaked a spurious nonzero spread
# into that frame and unfroze the coordinate by accident).
const FROZEN_INIT = (; position=[0.0, 1.0], squared_scale=[1.0, 2.0^-140])

@testset "update_loss!(::Diagonal) never writes an unusable scale" begin
    rng = Xoshiro(1)
    p = randn(rng, 3, 200); g = -p          # Gaussian evidence
    p[2, :] .= 1.0                          # coordinate 2: positions never moved
    g[3, 5] = Inf                           # coordinate 3: a non-finite gradient
    D = Diagonal([0.5, 0.25, 0.75])
    @test WarmupHMC.update_loss!(D, p, g) == Inf
    @test D.diag[1] ≈ sqrt(std(p[1, :]) / std(g[1, :]))   # a healthy coordinate still updates
    @test D.diag[2] ≈ 1 / std(g[2, :])                     # frozen: sized from the gradient alone
    @test D.diag[3] == 0.75                                # non-finite: keeps the previous scale
    # A frozen coordinate's scale never shrinks: it was already too small.
    D5 = Diagonal([5.0])
    @test WarmupHMC.update_loss!(D5, fill(1.0, 1, 50), randn(rng, 1, 50)) == Inf
    @test D5.diag == [5.0]

    # Laplace branch with a zero mean gradient used to write `Inf`.
    D0 = Diagonal([2.0])
    @test WarmupHMC.update_loss!(D0, randn(rng, 1, 50), zeros(1, 50)) == Inf
    @test D0.diag == [2.0]

    # Fewer than two recorded states: every spread is `NaN`, which used to be
    # written into the metric.
    D1 = Diagonal([3.0, 4.0])
    @test WarmupHMC.update_loss!(D1, randn(rng, 2, 1), randn(rng, 2, 1)) == Inf
    @test D1.diag == [3.0, 4.0]

    # Healthy evidence is scored and estimated exactly as before.
    hp = randn(rng, 2, 200); hg = -hp
    H = Diagonal(ones(2))
    s1, s2 = std.(eachrow(hp)), std.(eachrow(hg))
    @test WarmupHMC.update_loss!(H, hp, hg) ≈ mean(s1 .* s2)
    @test H.diag ≈ sqrt.(s1 ./ s2)
end

@testset "diagonal scale in a one-sided tail" begin
    function fit(x, g)
        D = Diagonal([3.0])
        loss = WarmupHMC.update_loss!(D, reshape(x, 1, :), reshape(g, 1, :))
        D.diag[1], loss
    end

    # Gradient of a log-Gamma density on its unconstrained log coordinate.
    x = collect(range(-26.0, -24.0; length=100))
    g = 2 .- exp.(x) ./ 0.15
    limits = falses(2)
    D = Diagonal([3.0, 3.0])
    WarmupHMC.update_loss!(D, permutedims(hcat(x, x)), permutedims(hcat(g, x .- mean(x)));
                           tail_limits=limits)
    @test limits == [true, false]
    scale, loss = fit(x, g)
    @test 0 < std(g) < 1e-8
    @test scale ≈ sqrt(2) / mean(g)
    @test loss ≈ std(x) * std(g)
    for a in (0.125, -8.0)
        scaled, scaled_loss = fit(a .* x, g ./ a)
        @test scaled ≈ abs(a) * scale
        @test scaled_loss ≈ loss
    end
    constant_scale, constant_loss = fit(x, fill(-2.0, length(x)))
    @test constant_scale ≈ scale
    @test constant_loss == Inf

    # The same window with a single observation at zero still lies on one
    # side. A gradient across zero instead calls for the spread estimate.
    g[1] = 0.0
    @test first(fit(x, g)) ≈ sqrt(2) / mean(g)
    g[1] = -0.1
    @test sqrt(2) / mean(g) < sqrt(std(x) / std(g))
    @test first(fit(x, g)) ≈ sqrt(std(x) / std(g))

    # A one-sided window can also have enough curvature to set a smaller scale.
    curved_g = exp.(x .+ 25)
    @test sqrt(std(x) / std(curved_g)) < sqrt(2) / mean(curved_g)
    @test first(fit(x, curved_g)) ≈ sqrt(std(x) / std(curved_g))

    @test fit(fill(1.0, length(x)), curved_g) == (max(3.0, inv(std(curved_g))), Inf)
    x[1] = Inf
    @test fit(x, curved_g) == (3.0, Inf)
end

@testset "transformation selection" begin
    rng = Xoshiro(2)
    # `pathfinder` mixes the coordinates, so a coordinate frozen in the raw frame
    # still moves in it.
    L = LowerTriangular([1.0 0.0; 1.0 1.0])
    options() = (; diagonal=Diagonal(ones(2)),
                   pathfinder=WarmupHMC.MatrixFactorization(L, Diagonal(ones(2))))
    p = randn(rng, 2, 100); g = randn(rng, 2, 100)
    p[2, :] .= 0.5
    # A finite loss beats a degenerate frame, whatever is active.
    @test WarmupHMC._select_transformation!(options(), :diagonal, p, g) === :pathfinder
    # No frame has evidence: keep the active one, not the first key.
    p[1, :] .= 0.5
    @test WarmupHMC._select_transformation!(options(), :pathfinder, p, g) === :pathfinder
    @test WarmupHMC._select_transformation!(options(), :diagonal, p, g) === :diagonal
end

@testset "a frozen coordinate recovers instead of crashing warm-up" begin
    r = adaptive_warmup_mcmc(Xoshiro(2), ChainGaussian(); init=FROZEN_INIT, n_draws=300)
    @test size(r.posterior_position, 2) >= 300
    @test all(isfinite, r.posterior_position)
    # x2's marginal standard deviation is sqrt(2); a frozen coordinate reads 0.
    @test 0.8 < std(r.posterior_position[2, :]) < 2.0

    out = cooperative_warmup_mcmc(Xoshiro(1), ChainGaussian();
        n_chains=2, init=FROZEN_INIT, n_cores=1, n_evaluations_budget=20_000, nonlinear_adapt=false)
    @test length(out.results) == 2
    @test all(r -> all(isfinite, r.posterior_position), out.results)
end

@testset "init rejects a degenerate diagonal squared scale" begin
    rejection(squared_scale) = try
        adaptive_warmup_mcmc(Xoshiro(3), ChainGaussian();
            init=(; position=[0.0, 0.0], squared_scale), n_draws=10)
        nothing
    catch e
        e
    end
    for bad in (0.0, -1.0, NaN, Inf), wrap in (identity, Diagonal)
        e = rejection(wrap([1.0, bad]))
        @test e isa ArgumentError && occursin("entry 2", e.msg)
    end
end
end
