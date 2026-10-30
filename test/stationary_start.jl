@testitem "a stationary vector start (a mode)" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Distributions

# Focused invocation from the package root (see test/continuous_nuts.jl):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "stationary_start.jl"))'

# A caller-supplied vector `init` at a stationary point of the log density — a
# MAP estimate, an empirical-Bayes mode (snag `pathfinder-start-890fde27`) — used
# to end the sampler call: L-BFGS converges before its first step, Pathfinder
# returns no fit, and `initialize_mcmc` raised the empty-ELBO error. Pathfinder
# now runs from a jittered start instead, and the initializer falls back to a
# unit diagonal metric at the start when that fails too (`_pathfinder_from_start`).
# The `@info` the jittered path logs carries `maxlog=1`, so it is not asserted
# here: another item of the suite may already have logged it in this process.

const Pathfinder = WarmupHMC.Pathfinder
quietly(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

d = 20
sigma = exp.(range(-1.5, 1.5, length=d))
target = DiagGaussian(zeros(d), sigma)
mode = zeros(d)

@testset "Pathfinder itself returns no fit from the mode" begin
    # Positive control of the mechanism: the trajectory has length zero and
    # nothing is scored, so without the recovery the start cannot initialize.
    result = quietly(() -> WarmupHMC.mypathfinder(target; rng=Xoshiro(1), init=mode, maxiters=100))
    @test isempty(result.elbo_estimates)
    @test length(result.optim_trace) == 1
    @test WarmupHMC._stationary_start(zeros(d))
    @test !WarmupHMC._stationary_start(fill(1e-7, d))
    err = quietly(() -> try
        WarmupHMC._pathfinder_from_start(target, mode; rng=Xoshiro(1), progress=nothing, recover=false)
        nothing
    catch e
        e
    end)
    @test err isa ErrorException
    @test occursin("no ELBO estimates", sprint(showerror, err))
end

@testset "the initializer recovers it with a jittered Pathfinder run" begin
    init = quietly(() -> WarmupHMC.initialize_mcmc(target, mode; rng=Xoshiro(1), progress=nothing))
    @test init isa NamedTuple
    @test all(isfinite, init.position)
    # Pathfinder ran: the start is its draw, the metric its low-rank fit.
    @test init.squared_scale isa Pathfinder.WoodburyPDMat
    @test maximum(abs, init.position ./ sigma) < 4
    # Its diagonal is curvature-informed, far closer to the truth than a unit metric.
    s = sqrt.(diag(init.squared_scale))
    @test mean(abs, log.(s ./ sigma)) < 0.5 * mean(abs, log.(1 ./ sigma))
    # A seed reproduces it.
    again = quietly(() -> WarmupHMC.initialize_mcmc(target, mode; rng=Xoshiro(1), progress=nothing))
    @test again.position == init.position
    @test diag(again.squared_scale) == diag(init.squared_scale)
end

@testset "every sampler starts at a mode" begin
    state = quietly(() -> ContinuousNUTSState(Xoshiro(2), target; init=mode))
    @test all(isfinite, state.position)
    @test all(>(0), state.scale.diag)
    for _ in 1:20
        continuous_nuts_step(state, target)
    end
    @test state.n_transitions == 20
    result = quietly(() -> adaptive_warmup_mcmc(Xoshiro(3), target; init=mode, n_draws=50))
    @test size(result.posterior_position, 2) >= 50
    @test all(isfinite, result.posterior_position)
    results = quietly(() -> adaptive_warmup_mcmc([Xoshiro(4), Xoshiro(5)], target; init=[mode, mode], n_draws=50))
    @test length(results) == 2
    @test all(r -> size(r.posterior_position, 2) >= 50, results)
end

@testset "a start that is not stationary is untouched" begin
    x0 = fill(3.0, d)
    init = quietly(() -> WarmupHMC.initialize_mcmc(target, x0; rng=Xoshiro(6), progress=nothing))
    reference = quietly(() -> WarmupHMC.mypathfinder(target; rng=Xoshiro(6), init=x0, maxiters=100))
    @test init.position == reference.draws[:, 1]
    @test diag(init.squared_scale) == diag(reference.fit_distribution.Σ)
end

# Two targets Pathfinder can fit from NO start. `FlatProblem`: a constant log
# density, so every point is stationary and every try converges before its
# first step. `IslandProblem`: finite (with a nonzero gradient) at one point and
# `-Inf` everywhere else, so the first attempt is not stationary but its first
# line search leaves the support, and every jittered start is outside it.
struct FlatProblem
    dimension::Int
end
LogDensityProblems.capabilities(::Type{FlatProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::FlatProblem) = p.dimension
LogDensityProblems.logdensity(::FlatProblem, x) = 0.0
LogDensityProblems.logdensity_and_gradient(p::FlatProblem, x) = (0.0, zeros(p.dimension))
struct IslandProblem{V}
    x0::V
end
LogDensityProblems.capabilities(::Type{<:IslandProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::IslandProblem) = length(p.x0)
LogDensityProblems.logdensity(p::IslandProblem, x) = x == p.x0 ? 0.0 : -Inf
LogDensityProblems.logdensity_and_gradient(p::IslandProblem, x) =
    (LogDensityProblems.logdensity(p, x), ones(length(p.x0)))

@testset "no fit from any jittered start falls back to the position with a unit metric" begin
    x0 = fill(3.0, d)
    # The stationary shape: the first attempt is skipped, both jittered runs
    # return no fit.
    init = @test_logs (:warn, r"stationary point.*unit diagonal metric") match_mode=:any WarmupHMC.initialize_mcmc(
        FlatProblem(d), x0; rng=Xoshiro(7), progress=nothing)
    @test init.position == x0
    @test init.position !== x0
    @test init.squared_scale == Diagonal(ones(d))
    # The line-search shape: the first attempt runs and returns no fit, and the
    # jittered starts are skipped as non-finite.
    init = @test_logs (:warn, r"returned no fit from the start.*unit diagonal metric") match_mode=:any WarmupHMC.initialize_mcmc(
        IslandProblem(x0), x0; rng=Xoshiro(7), progress=nothing)
    @test init.position == x0
    @test init.squared_scale == Diagonal(ones(d))
    # refused: the fallback still needs a finite start (the §1a contract)
    @test_throws ErrorException WarmupHMC.initialize_mcmc(NaNProblem(d), x0;
        rng=Xoshiro(7), progress=nothing)
end

@testset "WarmupHMC's own random starts redraw rather than jitter" begin
    # A random draw is never stationary (here it is: the target is flat), and a
    # Pathfinder failure from it is retried from a FRESH draw, as before: the
    # loop sees the error, not a unit-metric fallback at the failed draw.
    err = quietly(() -> try
        WarmupHMC.initialize_mcmc(FlatProblem(d), Uniform(-2, 2); rng=Xoshiro(8), progress=nothing,
            ntries=2)
        nothing
    catch e
        e
    end)
    @test err isa ErrorException
    @test occursin("no ELBO estimates", sprint(showerror, err))
end
end
