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

# `target` restricted to the half-space x[1] > 0: outside it the log density is
# -Inf, so a position there is one a transition cannot start from (§1a's
# out-of-support contract).
struct HalfSpace{P}
    inner::P
end
LogDensityProblems.capabilities(::Type{<:HalfSpace{P}}) where {P} = LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(p::HalfSpace) = LogDensityProblems.dimension(p.inner)
LogDensityProblems.logdensity(p::HalfSpace, x) =
    x[1] > 0 ? LogDensityProblems.logdensity(p.inner, x) : -Inf
LogDensityProblems.logdensity_and_gradient(p::HalfSpace, x) =
    x[1] > 0 ? LogDensityProblems.logdensity_and_gradient(p.inner, x) : (-Inf, zero(x))

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

    # Everything a transition reads besides the position and the mean of the
    # position evidence (which `reposition!` translates with it): the kernel,
    # the evidence's spread and weight, the counters and the random stream.
    kernel(s) = (; scale=copy(s.scale.diag), stepsize=s.stepsize,
        moments=(copy(s.moments.position_m2), copy(s.moments.gradient_mean),
            copy(s.moments.gradient_m2), s.moments.weight, s.moments.weight2),
        prior_weight=s.prior_weight, log_kappa=s.log_kappa, backoff=s.backoff,
        stepsizes=copy(s.stepsizes), acceptances=copy(s.acceptances),
        n_transitions=s.n_transitions, evaluations=s.total_evaluation_counter,
        n_divergent=s.n_divergent, rng=WarmupHMC._xoshiro_words(s.rng))

    @testset "reposition! translates the chain and its evidence, and changes nothing else" begin
        state = ContinuousNUTSState(Xoshiro(8), target; init=exact_start)
        for _ in 1:30
            continuous_nuts_step(state, target)
        end
        before = kernel(state)
        mean_before = copy(state.moments.position_mean)
        old = state.position
        old_copy = copy(old)
        moved = old .+ 0.5
        @test reposition!(state, moved) === state
        @test state.position == moved
        @test state.position !== moved          # copied: the caller's vector is not aliased
        moved[1] = 99.0
        @test state.position[1] != 99.0
        @test old == old_copy                   # a vector read earlier keeps its value
        @test state.moments.position_mean ≈ mean_before .+ 0.5
        @test kernel(state) == before
        # Any real vector, stored as Float64.
        rounded = Float64.(Float32.(old_copy))
        reposition!(state, Float32.(old_copy))
        @test state.position isa Vector{Float64} && state.position == rounded
        reposition!(state, 1:d)
        @test state.position == 1:d
        reposition!(state, old_copy)
        @test state.moments.position_mean ≈ mean_before   # the moves telescope
        @test kernel(state) == before
        # refused: the wrong length, or a position no transition can start from;
        # a refused move changes nothing.
        @test_throws DimensionMismatch reposition!(state, zeros(d + 1))
        @test_throws ArgumentError reposition!(state, [NaN; zeros(d - 1)])
        @test_throws ArgumentError reposition!(state, [Inf; zeros(d - 1)])
        @test state.position == old_copy
        @test state.moments.position_mean ≈ mean_before
        @test kernel(state) == before
    end

    @testset "a translated target sampled by the translated chain is the same chain" begin
        still = ContinuousNUTSState(Xoshiro(9), target; init=exact_start)
        for _ in 1:30
            continuous_nuts_step(still, target)
        end
        moved = deepcopy(still)
        δ = 3 .* sigma
        shifted = DiagGaussian(δ, sigma)
        reposition!(moved, moved.position .+ δ)
        for _ in 1:10
            a = continuous_nuts_step(still, target)
            b = continuous_nuts_step(moved, shifted)
            @test b.stats.steps == a.stats.steps
            @test b.divergent == a.divergent
            @test moved.position .- δ ≈ still.position rtol=1e-6
            @test moved.scale.diag ≈ still.scale.diag rtol=1e-6
            @test moved.stepsize ≈ still.stepsize rtol=1e-6
            @test moved.moments.position_mean .- δ ≈ still.moments.position_mean rtol=1e-6
            @test moved.moments.position_m2 ≈ still.moments.position_m2 rtol=1e-6
        end
    end

    @testset "the next step starts from the moved position under its target" begin
        state = ContinuousNUTSState(Xoshiro(10), target; init=exact_start)
        for _ in 1:20
            continuous_nuts_step(state, target)
        end
        # adapt=false from the moved position is the fixed kernel's draw from there.
        q = 0.5 .* sigma
        reposition!(state, q)
        fixed = FixedKernelNUTSState(copy(state.rng), q; metric=copy(state.scale.diag), stepsize=state.stepsize)
        frozen = continuous_nuts_step(state, target; adapt=false)
        reference = fixed_kernel_nuts_step(fixed, target)
        @test state.position == reference.state.position
        @test frozen.divergent == reference.divergent
        # The move evaluates nothing: a position the next target excludes is
        # that step's error (§1a's contract), and the refused step changes nothing.
        outside = [-1.0; zeros(d - 1)]
        reposition!(state, outside)
        before = kernel(state)
        err = try
            continuous_nuts_step(state, HalfSpace(target))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("at the state's position, so it lies outside the target's support", err.msg)
        @test state.position == outside
        @test kernel(state) == before
        # Positive control: one step inside the half-space samples.
        reposition!(state, [1.0; zeros(d - 1)])
        @test continuous_nuts_step(state, HalfSpace(target)).state.position[1] > 0
    end

    @testset "a re-centred chain keeps tracking: a translation leaves the kernel valid" begin
        # SAEM's mu-referenced re-centring: after every M-step the target moves
        # by δ and the chain is moved with it. Three chains from one adapted
        # state and random stream: `still` samples the fixed target; `moved` is
        # re-centred by δ_t and samples the target shifted by the running sum;
        # `left` samples the shifted target from where it was. The first two
        # must start every transition on the target (Mahalanobis²/d ≈ 1), cost
        # the same and keep the same kernel; `left` starts every transition
        # δ_t off, the drift `reposition!` removes.
        still = ContinuousNUTSState(Xoshiro(11), target; init=exact_start)
        for _ in 1:50
            continuous_nuts_step(still, target)
        end
        moved = deepcopy(still)
        left = deepcopy(still)
        mahalanobis(q, mu) = sum(abs2, (q .- mu) ./ sigma) / d
        mu = zeros(d)
        pre = (; still=Float64[], moved=Float64[], left=Float64[])
        shifts = Xoshiro(12)
        for t in 1:150
            δ = sigma .* randn(shifts, d)
            mu .+= δ
            shifted = DiagGaussian(copy(mu), sigma)
            reposition!(moved, moved.position .+ δ)
            push!(pre.still, mahalanobis(still.position, zeros(d)))
            push!(pre.moved, mahalanobis(moved.position, mu))
            push!(pre.left, mahalanobis(left.position, mu))
            continuous_nuts_step(still, target)
            continuous_nuts_step(moved, shifted)
            continuous_nuts_step(left, shifted)
        end
        @test 0.8 < mean(pre.still) < 1.2
        @test 0.8 < mean(pre.moved) < 1.2
        @test mean(pre.left) > 1.5                 # E = 1 + E‖δ/σ‖²/d = 2 when it tracked the last target
        @test all(0.8 .< moved.scale.diag ./ sigma .< 1.25)
        @test 0.8 < moved.stepsize / still.stepsize < 1.25
        @test 0.8 < moved.total_evaluation_counter / still.total_evaluation_counter < 1.25
        @test moved.n_divergent <= still.n_divergent + 2
    end
finally
    BLAS.set_num_threads(blas_threads)
end
end
