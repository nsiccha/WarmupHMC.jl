@testitem "linear scaling in the dimension" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, LinearAlgebra, Random
    using LogDensityProblems
    const Pathfinder = WarmupHMC.Pathfinder

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "linear_scaling.jl"))'

# User rule (decision `1o12joq`): nothing WarmupHMC builds may scale worse than
# linearly in the posterior dimension. `f210206` broke it silently by turning
# Pathfinder's low-rank factor into a dense Cholesky factor; nothing tested the
# structure, so nothing failed. These tests make the structure and its cost a
# checked property.

d = 2000
target = DiagGaussian(zeros(d), exp.(range(-1, 1, length = d)))
fit = Pathfinder.pathfinder(target; rng = Xoshiro(1), ndraws = 10, progress = false)
Σ = fit.fit_distribution.Σ
dense_bytes = 8 * d^2

@testset "a Pathfinder init keeps the low-rank factor" begin
    @test Σ isa Pathfinder.WoodburyPDMat
    L = WarmupHMC._initial_pathfinder_scale(Σ, d)   # also compiles
    bytes = @allocated WarmupHMC._initial_pathfinder_scale(Σ, d)
    @test !(L.m1 isa Union{LowerTriangular, UpperTriangular, Matrix})
    # O(d k): far below one dense d × d matrix (32 MB here).
    @test bytes < dense_bytes / 20
end

@testset "the metric's mat-vec allocates O(1)-to-O(d), never O(d²)" begin
    L = WarmupHMC._initial_pathfinder_scale(Σ, d)
    M = WarmupHMC.MatrixFactorization(L, L')
    x, y = randn(Xoshiro(2), d), zeros(d)
    mul!(y, M, x)
    @test @allocated(mul!(y, M, x)) < 8d
    # and it is the same operator as the dense reference
    @test y ≈ Matrix(Σ) * x rtol = 1e-8
end

@testset "the :adaptive reflection metric is O(d · max_reflections)" begin
    # Forty strongly correlated pairs among otherwise independent coordinates:
    # the reflection fit wants one reflection per pair, each stored over the
    # coordinates still badly fitted. Built pair by pair, no d × d matrix.
    # (Unbounded, a target correlated everywhere reached ~0.47·d² stored entries
    # at d = 400, go/no-go `04q64y5`.)
    rng = Xoshiro(5)
    n, m = 1000, 40
    s = exp.(range(-2, 2, d))
    ρ = range(0.95, 0.99, m)
    p = s .* randn(rng, d, n)
    g = -(p ./ s .^ 2)
    z1, z2 = randn(rng, m, n), randn(rng, m, n)
    v = ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2
    i, j = 1:2:2m, 2:2:2m
    p[i, :] .= s[i] .* z1; p[j, :] .= s[j] .* v
    g[i, :] .= -(z1 .- ρ .* v) ./ ((1 .- ρ .^ 2) .* s[i])
    g[j, :] .= -(v .- ρ .* z1) ./ ((1 .- ρ .^ 2) .* s[j])
    # The `:adaptive` candidate both samplers fit each window.
    L = WarmupHMC.MatrixFactorization(WarmupHMC.SuccessiveReflections(d), Diagonal(ones(d)))
    WarmupHMC.update_loss!(L, p, g)
    R = L.m1
    k = WarmupHMC._DEFAULT_MAX_REFLECTIONS
    @test 0 < length(R.reflections) <= k
    @test sum(length, R.reflections) <= k * d
    # The kinetic energy's M⁻¹ = L L' applies the reflections twice, each time
    # through one O(d) gather buffer, never one copy per reflection.
    M = WarmupHMC.MatrixFactorization(L, L')
    x, y = randn(rng, d), zeros(d)
    mul!(y, M, x)
    @test @allocated(mul!(y, M, x)) <= 2 * (8d + 64)
end

@testset "a caller-supplied dense matrix still factorizes (f210206's case)" begin
    small = 30
    A = randn(Xoshiro(3), small, small)
    S = A * A' + small * I
    L = WarmupHMC._initial_pathfinder_scale(S, small)
    @test L.m1 isa LowerTriangular
    @test Matrix(L.m1) * Matrix(L.m1)' ≈ S
end

@testset "grad_cov_ev's fallback is linear and matches tsvd" begin
    g = randn(Xoshiro(4), 300, 40)
    g[1, :] .*= 5   # a clear top direction
    v_tsvd = WarmupHMC.grad_cov_ev(nothing, g)
    v_pow = WarmupHMC._top_left_singular_vector(g)
    @test abs(dot(v_tsvd, v_pow)) ≈ 1 atol = 1e-8
    @test norm(v_pow) ≈ 1
end
end
