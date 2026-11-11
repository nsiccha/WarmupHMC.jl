@testitem "an unusable Pathfinder fit is neither the start nor the metric" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems

# Focused invocation from the package root (see test/continuous_nuts.jl):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "pathfinder_unusable_fit.jl"))'

# The contract (snag `warmuphmc-contin-51744bcf`, reporter NLMEEstimation): a
# finite vector `init` on a target with a collapsed scale (a random-effect
# variance of 1e-12 beside a normal one, as SAEM conditionals have when Ω
# collapses) gives a start no worse than a draw of the target, with a metric of
# the target's scales. Pathfinder drops every L-BFGS curvature pair above 1e12,
# so its fit there is the identity covariance, 1e12 times too wide: its draw is
# finite but ~1e23 nats below the init, and the chain used to start there with a
# step size of 1e-9 and never return. The initializer now recognises the
# unusable fit and runs Pathfinder again on the target rescaled by a diagonal
# from the L-BFGS trajectory. The `@info` that path logs carries `maxlog=1`, so
# it is not asserted here.

quietly(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# N(0, diag(sd²)); a model that overflows beyond |x₁| > cap answers NaN there,
# as one whose likelihood exponentiates x₁ does.
struct Collapsed
    sd::Vector{Float64}
    cap::Float64
end
LogDensityProblems.capabilities(::Type{Collapsed}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::Collapsed) = length(p.sd)
LogDensityProblems.logdensity_and_gradient(p::Collapsed, x) = abs(x[1]) > p.cap ?
    (NaN, fill(NaN, length(x))) : (-sum(abs2, x ./ p.sd) / 2, -x ./ p.sd .^ 2)
LogDensityProblems.logdensity(p::Collapsed, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))
ℓ(target, x) = LogDensityProblems.logdensity(target, x)

stiff = Collapsed([1e-12, 0.5], Inf)            # the reporter's target
# The same, overflowing beyond |x₁| > 0.01: past L-BFGS's first trial step from
# `init` (about 3e-3; a cap inside it fails that line search, and Pathfinder
# returns no fit at all), but inside the unusable fit's draws, which are O(1).
overflowing = Collapsed([1e-12, 0.5], 0.01)
init = [0.7e-12, 0.3]                           # log density -0.425
pathfinder(target, seed) = WarmupHMC.mypathfinder(target; rng=Xoshiro(seed), init=copy(init), maxiters=100)

blas_threads = BLAS.get_num_threads()
BLAS.set_num_threads(1)
try
    @testset "the mechanism: Pathfinder's fit drops the collapsed curvature" begin
        # Positive controls: without these the checks below would pass vacuously.
        result = quietly(() -> pathfinder(stiff, 1))
        @test result.num_bfgs_updates_rejected >= 1
        @test diag(result.fit_distribution.Σ)[1] > 1e-6        # the target's variance is 1e-24
        draw = result.draws[:, 1]
        @test isfinite(ℓ(stiff, draw))
        @test ℓ(stiff, draw) < -1e10
        @test WarmupHMC._fit_drops_curvature(result)
        @test WarmupHMC._fit_log_weight_change(result, draw, ℓ(stiff, draw)) < -WarmupHMC._implausible_drop(2)
        # Overflowing: no draw is finite at all, which the curvature check still sees.
        result = quietly(() -> pathfinder(overflowing, 1))
        @test !isfinite(ℓ(overflowing, result.draws[:, 1]))
        @test WarmupHMC._fit_drops_curvature(result)
    end

    @testset "the start is a draw of the target, the metric its scales" begin
        for target in (stiff, overflowing), seed in 1:5
            start = quietly(() -> WarmupHMC.initialize_mcmc(target, copy(init); rng=Xoshiro(seed), progress=nothing))
            @test ℓ(target, start.position) > -10                   # a draw is ~1 nat below the mode
            @test start.position_and_gradient.q == start.position
            variance = diag(start.squared_scale)
            @test 1e-26 < variance[1] < 1e-22
            @test 0.01 < variance[2] < 10
        end
    end

    @testset "every sampler then moves" begin
        for seed in 1:3
            state = quietly(() -> ContinuousNUTSState(Xoshiro(seed), stiff; init=copy(init)))
            @test ℓ(stiff, state.position) > -10
            divergent = sum(_ -> continuous_nuts_step(state, stiff).divergent, 1:20)
            @test divergent <= 5                                    # 9–16 of 20 before
            @test state.stepsize > 0.1                              # 1e-9 before
            @test state.scale.diag[1] < 1e-10
            result = quietly(() -> adaptive_warmup_mcmc(Xoshiro(seed), stiff; init=copy(init), n_draws=100))
            @test result.sampling_evaluation_counter < 20_000      # ~4e5 before
            @test 0.3 < std(result.posterior_position[1, :]) / 1e-12 < 3
            @test 0.15 < std(result.posterior_position[2, :]) < 1.5
        end
        out = quietly(() -> cooperative_warmup_mcmc(Xoshiro(1), stiff;
            n_chains=2, n_draws=50, n_evaluations=200, init=copy(init)))
        @test all(r -> all(c -> ℓ(stiff, c) > -20, eachcol(r.posterior_position)), out.results)
    end

    @testset "a usable fit is untouched" begin
        # Healthy targets: the start is Pathfinder's own first draw with its fit,
        # exactly as before the checks existed, and no further random number is drawn.
        for target in (DiagGaussian(zeros(5), exp.(range(-2, 2, length=5))), Collapsed([0.1, 0.5], Inf)),
                seed in 1:5
            start_point = rand(Xoshiro(100 + seed), LogDensityProblems.dimension(target)) .* 4 .- 2
            reference = Xoshiro(seed)
            result = WarmupHMC.mypathfinder(target; rng=reference, init=copy(start_point), maxiters=100)
            @test !WarmupHMC._fit_drops_curvature(result)
            rng = Xoshiro(seed)
            start = WarmupHMC.initialize_mcmc(target, copy(start_point); rng, progress=nothing)
            @test start.position == result.draws[:, 1]
            @test start.squared_scale isa WarmupHMC.Pathfinder.WoodburyPDMat
            @test WarmupHMC._xoshiro_words(rng) == WarmupHMC._xoshiro_words(reference)
        end
    end

    @testset "when the rescaled run gives no fit, the trajectory's diagonal stands in" begin
        # The branch needs a rescaled run with no fit; a target stationary at the
        # start gives one (L-BFGS converges before its first step). The failed fit
        # is the stiff target's, so the diagonal is its trajectory's.
        result = quietly(() -> pathfinder(stiff, 1))
        stationary = DiagGaussian(copy(init), [1.0, 1.0])
        start = @test_logs (:warn, r"gave no usable fit either") match_mode=:any WarmupHMC._rescaled_pathfinder_start(
            stationary, result; rng=Xoshiro(2), progress=nothing, maxiters=100)
        @test start.position == result.optim_trace.points[result.fit_iteration + 1]
        @test start.squared_scale isa Diagonal
        @test 1e-26 < start.squared_scale[1, 1] < 1e-22         # the trajectory measured x₁'s scale
        @test start.squared_scale[2, 2] ≈ 1 / 1.2               # never moved: 1/|∂ℓ/∂x₂| at the init
    end
finally
    BLAS.set_num_threads(blas_threads)
end
end
