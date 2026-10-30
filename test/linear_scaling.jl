@testitem "linear scaling in the dimension" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, LinearAlgebra, Random
    using LogDensityProblems, Profile
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

@testset "the continuous sampler's state and transition are O(d)" begin
    # Seven length-d vectors and a Diagonal: no d × d object anywhere, and a
    # transition allocates the trajectory's leapfrog states plus O(d), as the
    # fixed kernel does.
    rng = Xoshiro(6)
    state = ContinuousNUTSState(rng, target; init=(; position=zeros(d), squared_scale=ones(d)))
    @test state.scale isa Diagonal
    @test Base.summarysize(state) < 16 * 8d
    continuous_nuts_step(state, target)   # compiles
    step = continuous_nuts_step(state, target)
    bytes = @allocated continuous_nuts_step(state, target)
    @test bytes < dense_bytes / 4
    # Per leapfrog step the allocation is linear: twice the dimension costs
    # about twice the bytes per step, never four times.
    d2 = 2d
    target2 = DiagGaussian(zeros(d2), exp.(range(-1, 1, length = d2)))
    state2 = ContinuousNUTSState(Xoshiro(6), target2; init=(; position=zeros(d2), squared_scale=ones(d2)))
    continuous_nuts_step(state2, target2)
    per_step(s, t) = begin
        total, steps = 0, 0
        for _ in 1:5
            r = continuous_nuts_step(s, t)
            total += @allocated(continuous_nuts_step(s, t))
            steps += r.stats.steps
        end
        total / steps
    end
    @test per_step(state2, target2) < 3 * per_step(state, target)
end

@testset "a Pathfinder start allocates no d × d object" begin
    # OptimizationOptimJL wraps L-BFGS in an `Optim.TwiceDifferentiable` whose
    # Hessian buffer defaults to dense d × d and is copied once: 2 × d² × 8 bytes
    # a first-order method never reads (SciML/Optimization.jl#893; 20 GB at
    # d = 35,784, snag `pathfinder-start-ff294a50`). WarmupHMC's objective shapes
    # that buffer `Diagonal` (`_pathfinder_objective`). Record EVERY allocation of
    # a whole start — Pathfinder, the diagonal scale, the kinetic energy and the
    # step-size search — and refuse any object of d × d size: the largest O(d · k)
    # object is Pathfinder's d × 2k history (192 kB here), the dense buffer 32 MB.
    init = fill(2.0, d)
    ContinuousNUTSState(Xoshiro(7), target; init)   # compiles
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1 ContinuousNUTSState(Xoshiro(7), target; init)
    allocs = Profile.Allocs.fetch().allocs
    Profile.Allocs.clear()
    @test !isempty(allocs)
    @test maximum(a -> a.size, allocs) < 64 * 8d
    @test sum(a -> a.size, allocs) < dense_bytes * 4
    # The same objective, with Pathfinder's own default-built one as the
    # control: the prototype is the ONLY difference, and the trajectory, the
    # fit, the ELBO and the draw are bit-identical.
    ours = WarmupHMC.mypathfinder(target; rng=Xoshiro(8), init, maxiters=100)
    theirs = Pathfinder.pathfinder(target; rng=Xoshiro(8), init, maxiters=100,
        ndraws=1, ndraws_elbo=1, ntries=1, adtype=Pathfinder.SciMLBase.NoAD(),
        optimizer=WarmupHMC._pathfinder_optimizer(6))
    @test ours.optim_solution.cache.f.hess_prototype isa Diagonal
    @test theirs.optim_solution.cache.f.hess_prototype === nothing
    @test ours.optim_trace.points == theirs.optim_trace.points
    @test ours.optim_trace.gradients == theirs.optim_trace.gradients
    @test ours.fit_iteration == theirs.fit_iteration
    @test [e.value for e in ours.elbo_estimates] == [e.value for e in theirs.elbo_estimates]
    @test ours.draws == theirs.draws
    @test ours.input === target
    @test @allocated(WarmupHMC.mypathfinder(target; rng=Xoshiro(8), init, maxiters=100)) <
        @allocated(Pathfinder.pathfinder(target; rng=Xoshiro(8), init, maxiters=100,
            ndraws=1, ndraws_elbo=1, ntries=1, adtype=Pathfinder.SciMLBase.NoAD(),
            optimizer=WarmupHMC._pathfinder_optimizer(6))) - dense_bytes
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
