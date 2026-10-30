@testitem "continuously adapting NUTS on changing targets" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "continuous_nuts.jl"))'

# The contract of `ContinuousNUTSState` / `continuous_nuts_step` (snag
# `continuous-nuts-15212ad7`): a changing-target NUTS chain whose diagonal metric
# and step size adapt on every transition, with forgetting, from Pathfinder or a
# caller-supplied start, and whose `adapt=false` step is the fixed kernel.

d = 20
sigma = exp.(range(-1.5, 1.5, length=d))
target = DiagGaussian(zeros(d), sigma)
unit_start = (; position=zeros(d), squared_scale=ones(d))
exact_start = (; position=zeros(d), squared_scale=sigma .^ 2)

blas_threads = BLAS.get_num_threads()
BLAS.set_num_threads(1)
try
    @testset "construction" begin
        # A NamedTuple init starts exactly there, runs no Pathfinder, and does not
        # alias the caller's arrays.
        q0, ss = zeros(d), ones(d)
        state = ContinuousNUTSState(Xoshiro(1), target; init=(; position=q0, squared_scale=ss))
        @test state.position == zeros(d)
        @test state.scale == Diagonal(ones(d))
        q0[1] = 99.0; ss[1] = 99.0
        @test state.position[1] == 0 && state.scale.diag[1] == 1
        @test isfinite(state.stepsize) && state.stepsize > 0
        @test state.n_transitions == 0 == state.total_evaluation_counter == state.n_divergent
        # The caller's RNG is copied, not consumed.
        rng = Xoshiro(3)
        words = WarmupHMC._xoshiro_words(rng)
        ContinuousNUTSState(rng, target; init=unit_start)
        @test WarmupHMC._xoshiro_words(rng) == words
        # A vector init runs Pathfinder from it: the chain starts at Pathfinder's
        # draw with its diagonal scale, not at the vector.
        pathfinder = ContinuousNUTSState(Xoshiro(2), target; init=fill(3.0, d))
        @test all(isfinite, pathfinder.position) && pathfinder.position != fill(3.0, d)
        @test all(>(0), pathfinder.scale.diag)
        @test maximum(abs, pathfinder.position ./ sigma) < 4
        # refused: a start outside the support cannot be stepped from (§1a's contract)
        err = try
            ContinuousNUTSState(Xoshiro(1), NaNProblem(d); init=unit_start)
        catch e
            e
        end
        @test err isa WarmupHMC.DynamicHMC.DynamicHMCError
        # refused: keywords are validated before any work
        @test_throws ArgumentError ContinuousNUTSState(Xoshiro(1), target; init=unit_start, target_acceptance_rate=1.0)
        @test_throws ArgumentError ContinuousNUTSState(Xoshiro(1), target; init=unit_start, max_tree_depth=0)
        @test_throws ArgumentError ContinuousNUTSState(Xoshiro(1), target; init=unit_start, half_life=0)
        @test_throws ArgumentError ContinuousNUTSState(Xoshiro(1), target; init=unit_start, prior_weight=-1)
        @test_throws ArgumentError ContinuousNUTSState(Xoshiro(1), target; init=unit_start, blas_threads=0)
        # refused: no silent keyword forwarding (there is no `kwargs...`)
        @test_throws MethodError ContinuousNUTSState(Xoshiro(1), target; init=unit_start, n_draws=10)
    end

    @testset "a step advances the state in place and adapts it" begin
        state = ContinuousNUTSState(Xoshiro(1), target; init=unit_start)
        before = copy(state.position)
        step = continuous_nuts_step(state, target)
        @test step.state === state
        @test step.divergent isa Bool
        @test step.stats.steps == state.total_evaluation_counter
        @test state.n_transitions == 1
        @test state.position != before
        @test state.scale != Diagonal(ones(d))
        @test_throws DimensionMismatch continuous_nuts_step(state,
            DiagGaussian(zeros(d + 1), ones(d + 1)))
        @test_throws ArgumentError continuous_nuts_step(state, target; blas_threads=0)
    end

    @testset "the metric and the step size converge on a fixed target" begin
        state = ContinuousNUTSState(Xoshiro(1), target; init=unit_start)
        acceptances = Float64[]
        for t in 1:400
            step = continuous_nuts_step(state, target)
            t > 300 && push!(acceptances, step.stats.acceptance_rate)
        end
        # The Nutpie estimate is exact on a Gaussian: within rounding of the
        # posterior SDs after the start scale has been forgotten.
        @test all(0.9 .< state.scale.diag ./ sigma .< 1.1)
        @test 0.65 < mean(acceptances) < 0.95
        @test state.n_transitions == 400
    end

    @testset "the metric follows a changed target within a few half-lives" begin
        state = ContinuousNUTSState(Xoshiro(4), target; init=exact_start, half_life=10)
        for _ in 1:100
            continuous_nuts_step(state, target)
        end
        shifted = DiagGaussian(fill(0.5, d), 4 .* sigma)
        for _ in 1:60   # six half-lives
            continuous_nuts_step(state, shifted)
        end
        @test all(0.75 .< state.scale.diag ./ (4 .* sigma) .< 1.35)
        # and the position is where the new target lives
        @test maximum(abs, (state.position .- 0.5) ./ (4 .* sigma)) < 4
    end

    @testset "half_life=Inf never forgets, and the pseudo-count forgets with the evidence" begin
        remembering = ContinuousNUTSState(Xoshiro(4), target; init=exact_start, half_life=Inf)
        for _ in 1:50
            continuous_nuts_step(remembering, target)
        end
        w = remembering.moments.weight
        @test w > 0
        continuous_nuts_step(remembering, target)
        @test remembering.moments.weight > w          # nothing was forgotten
        @test remembering.prior_weight == WarmupHMC._DEFAULT_CONTINUOUS_PRIOR_WEIGHT
        forgetting = ContinuousNUTSState(Xoshiro(4), target; init=exact_start, half_life=20)
        for _ in 1:20
            continuous_nuts_step(forgetting, target)
        end
        @test forgetting.prior_weight ≈ WarmupHMC._DEFAULT_CONTINUOUS_PRIOR_WEIGHT / 2
    end

    @testset "adapt=false is the fixed kernel, draw for draw" begin
        state = ContinuousNUTSState(Xoshiro(5), target; init=exact_start)
        for _ in 1:20
            continuous_nuts_step(state, target)
        end
        fixed = FixedKernelNUTSState(copy(state.rng), copy(state.position);
            metric=copy(state.scale.diag), stepsize=state.stepsize)
        scale, stepsize = copy(state.scale.diag), state.stepsize
        shifted = DiagGaussian(fill(0.5, d), 2 .* sigma)
        for lpdf in (target, shifted, target)
            frozen = continuous_nuts_step(state, lpdf; adapt=false)
            reference = fixed_kernel_nuts_step(fixed, lpdf)
            fixed = reference.state
            @test state.position == fixed.position
            @test frozen.divergent == reference.divergent
            @test WarmupHMC._xoshiro_words(state.rng) == WarmupHMC._xoshiro_words(fixed.rng)
        end
        @test state.scale.diag == scale
        @test state.stepsize == stepsize
        @test state.n_transitions == 23
    end

    @testset "a start too far off is corrected within tens of transitions" begin
        # A unit start on scales from e^-1.5 to e^1.5 and a wrong location: the
        # step size and metric both have to move, in both directions.
        state = ContinuousNUTSState(Xoshiro(6), target; init=(; position=fill(2.0, d), squared_scale=fill(100.0, d)))
        for _ in 1:60
            continuous_nuts_step(state, target)
        end
        @test all(0.5 .< state.scale.diag ./ sigma .< 2)
        @test state.n_divergent < 10
    end
finally
    BLAS.set_num_threads(blas_threads)
end
end
