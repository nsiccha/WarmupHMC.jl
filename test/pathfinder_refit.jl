@testitem "Pathfinder refit on a stalled trajectory" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra
    using LogDensityProblems
    using Distributions

# On a log density with a kink at its mode, Pathfinder's inverse-Hessian fit can
# throw a `PosDefException` (snag `adaptive-warmup-9e5a75a0`). The initializer then
# refits on the longest prefix of the L-BFGS trajectory that factorizes, instead of
# letting the exception end the sampler call. See `_pathfinder_or_refit`.

const Pathfinder = WarmupHMC.Pathfinder
problem = KinkedGaussian(
    [2.149, 0.445, 0.697, 1.526, 2.531],
    [-0.147 -0.541 -0.134 0.484 -0.455; 1.904 -0.339 -0.231 -0.112 0.318; -0.873 0.087 1.118 0.455 -1.448],
    [-0.224, 0.921, -0.907],
    1.0,
)
d = LogDensityProblems.dimension(problem)
starts = [rand(Xoshiro(seed), Uniform(-2, 2), d) for seed in 1:20]
quietly(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
throws_posdef(f) = try
    f(); false
catch err
    err isa PosDefException || rethrow(); true
end
plain(x0, rng) = WarmupHMC.mypathfinder(problem; rng, init=x0, maxiters=100)
no_progress = (state, args...) -> false
refit(x0, rng) = WarmupHMC._pathfinder_or_refit(problem; rng, init=x0, callback=no_progress, maxiters=100)

failing = filter(x0 -> quietly(() -> throws_posdef(() -> plain(x0, Xoshiro(1)))), starts)
fitting = filter(x0 -> !(x0 in failing), starts)

@testset "the shape fails without the refit" begin
    # Positive control: plain Pathfinder throws from some of these starts and
    # succeeds from others, so both branches below are exercised.
    @test !isempty(failing)
    @test !isempty(fitting)
end

@testset "a failing start is refit on the trajectory prefix" begin
    for x0 in failing
        f = WarmupHMC._first_unfactorizable_iterate(problem;
            rng=Xoshiro(2), init=x0, callback=no_progress, maxiters=100)
        @test f isa Int && f >= 2
        result = @test_logs (:warn, r"not positive definite at iteration") match_mode=:any refit(x0, Xoshiro(2))
        # Exactly Pathfinder with fewer iterations: the same trajectory and draws.
        reference = quietly(() -> WarmupHMC.mypathfinder(problem; rng=Xoshiro(2), init=x0, maxiters=f - 1))
        @test length(result.optim_trace) == f
        @test result.optim_trace.points == reference.optim_trace.points
        @test result.draws == reference.draws
        @test !isempty(result.elbo_estimates)
        # The longest such prefix: one more iteration no longer factorizes.
        @test quietly(() -> throws_posdef(() -> WarmupHMC.mypathfinder(problem; rng=Xoshiro(2), init=x0, maxiters=f)))
    end
end

@testset "a start that fits is untouched" begin
    for x0 in fitting
        result = quietly(() -> refit(x0, Xoshiro(3)))
        reference = quietly(() -> plain(x0, Xoshiro(3)))
        @test result.optim_trace.points == reference.optim_trace.points
        @test result.draws == reference.draws
    end
end

@testset "the initializer and the sampler start from a failing vector init" begin
    x0 = first(failing)
    init = quietly(() -> WarmupHMC.initialize_mcmc(problem, x0; rng=Xoshiro(4), progress=nothing))
    @test all(isfinite, init.position)
    @test init.squared_scale isa Pathfinder.WoodburyPDMat
    # A vector init has no retry loop, so before the refit this call threw.
    result = quietly(() -> adaptive_warmup_mcmc(Xoshiro(5), problem; init=x0, n_draws=100))
    @test size(result.posterior_position, 2) >= 100
    @test all(isfinite, result.posterior_position)
end

# A target whose OWN `PosDefException` ends the L-BFGS trajectory: the recorded
# trajectory fits, so the exception is the target's, and it propagates unchanged.
struct ThrowsPastOne end
LogDensityProblems.capabilities(::Type{ThrowsPastOne}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::ThrowsPastOne) = 2
function LogDensityProblems.logdensity_and_gradient(::ThrowsPastOne, x)
    x[1] > 1 && throw(PosDefException(-42))
    (-sum(abs2, x .- [3.0, 0.0]) / 2, -(x .- [3.0, 0.0]))
end
LogDensityProblems.logdensity(p::ThrowsPastOne, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

@testset "a PosDefException from the target is rethrown" begin
    # The target throws on the way to its mode at x[1] = 3, and the trajectory
    # recorded up to that throw fits, so nothing is refit.
    @test throws_posdef(() -> LogDensityProblems.logdensity(ThrowsPastOne(), [2.0, 0.0]))
    @test isnothing(quietly(() -> WarmupHMC._first_unfactorizable_iterate(ThrowsPastOne();
        rng=Xoshiro(6), init=[0.0, 0.5], callback=no_progress, maxiters=100)))
    err = try
        quietly(() -> WarmupHMC.initialize_mcmc(ThrowsPastOne(), [0.0, 0.5]; rng=Xoshiro(6), progress=nothing))
        nothing
    catch err
        err
    end
    @test err isa PosDefException
    @test err.info == -42
end
end
