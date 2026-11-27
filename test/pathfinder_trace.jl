@testitem "a Pathfinder start without its trace" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems

# Focused invocation from the package root (see test/continuous_nuts.jl):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "pathfinder_trace.jl"))'

# With our Pathfinder fork's `save_trace=false` (snag `pathfinder-start-30deca51`)
# Pathfinder keeps neither its L-BFGS trajectory nor its fits, so the checks of the
# start read what `_TrajectoryRecorder` kept of the run and replay L-BFGS to the
# fit's iterate when the fit's history stretch lies before that
# (`_pathfinder_path`). These tests pin that the replay is the trajectory
# Pathfinder took, and that the checks read exactly what they read from the
# whole trajectory before (WarmupHMC 0.3.9, written out below).

quietly(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# Neal's funnel: Pathfinder's fit is selected early in the run, so the end of
# the trajectory the recorder keeps does not hold its history stretch.
struct Funnel
    d::Int
end
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::Funnel) = p.d
function LogDensityProblems.logdensity_and_gradient(p::Funnel, x)
    v, z = x[1], x[2:end]
    g = [-v / 9 - (p.d - 1) / 2 + sum(abs2, z) * exp(-v) / 2; -z .* exp(-v)]
    (-v^2 / 18 - (p.d - 1) * v / 2 - sum(abs2, z) * exp(-v) / 2, g)
end
LogDensityProblems.logdensity(p::Funnel, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

# The checks of WarmupHMC 0.3.9 on the whole trajectory (points P, gradients G).
function checks_on_trajectory(P, G, k, Σ)
    history_length = size(Σ.B, 2) ÷ 2
    first_pair, accepted = k + 1, 0
    for l in k:-1:1
        first_pair = l
        s, y = P[l + 1] - P[l], G[l] - G[l + 1]
        dot(s, y) > WarmupHMC._PATHFINDER_CURVATURE_EPS * sum(abs2, y) && (accepted += 1)
        accepted == history_length && break
    end
    drops = any(first_pair:k) do l
        s, y = P[l + 1] - P[l], G[l] - G[l + 1]
        dot(s, y) > 0 && dot(y, Σ * y) > WarmupHMC._DROPPED_CURVATURE_RATIO * dot(s, y)
    end
    points, gradients = P[first_pair:k + 1], G[first_pair:k + 1]
    scale = map(eachindex(first(gradients))) do i
        s = (var(p[i] for p in points) / var(g[i] for g in gradients))^(1 / 4)
        isfinite(s) && s > 0 && return s
        g = abs(first(gradients)[i])
        isfinite(g) && g > 0 ? 1 / sqrt(g) : 1.0
    end
    (; lo=first_pair - 1, drops, scale, fit_point=P[k + 1])
end

function replayed(target, result, stop=length(result.optim_trace) - 1)
    P, G = Vector{Float64}[], Vector{Float64}[]
    WarmupHMC._replay_trajectory((_, x, g) -> (push!(P, copy(x)); push!(G, copy(g))), target, result, stop)
    P, G
end
no_progress = (state, args...) -> false

targets = (DiagGaussian(zeros(20), exp.(range(-1.5, 1.5, length=20))), Funnel(10))
starts(target, seed) = rand(Xoshiro(seed), LogDensityProblems.dimension(target)) .* 4 .- 2

@testset "the loaded Pathfinder drops its trace" begin
    # Project.toml's [sources] pins the fork; a Pathfinder release falls back to
    # the trace and every check below reads it.
    @test WarmupHMC._PATHFINDER_SAVE_TRACE_OPTIONAL
    result = WarmupHMC.mypathfinder(first(targets); rng=Xoshiro(1), init=starts(first(targets), 1), maxiters=100)
    @test isempty(result.optim_trace.points)
    @test isempty(result.fit_distributions)
    @test length(result.optim_trace) > 1
end

@testset "the replay is Pathfinder's trajectory, bit for bit" begin
    for target in targets, seed in 1:5
        traced = WarmupHMC.mypathfinder(target; rng=Xoshiro(seed), init=starts(target, seed),
            maxiters=100, save_trace=true)
        P, G = replayed(target, traced)
        @test P == traced.optim_trace.points
        @test G == traced.optim_trace.gradients
        # And without the trace, the trajectory is the same one.
        recorder = WarmupHMC._TrajectoryRecorder(target)
        result = WarmupHMC._pathfinder_or_refit(target; rng=Xoshiro(seed), init=starts(target, seed),
            callback=no_progress, maxiters=100, recorder)
        @test isempty(result.optim_trace.log_densities)
        @test result.fit_log_density == traced.optim_trace.log_densities[result.fit_iteration + 1]
        @test recorder.n == length(P)
        @test recorder.start == first(P)
        @test [(i, x) for (i, x, _) in recorder.iterates] ==
            [(i - 1, P[i]) for i in max(1, length(P) - WarmupHMC._RECORDED_ITERATES + 1):length(P)]
    end
end

@testset "the recorder has bounded retention even on long rejected histories" begin
    target = first(targets)
    recorder = WarmupHMC._TrajectoryRecorder(target)
    x, g = zeros(2), ones(2)
    for i in 0:99
        x .= i
        WarmupHMC._record!(recorder, x, g)
    end
    bytes = Base.summarysize(recorder)
    for i in 100:100_000
        x .= i
        WarmupHMC._record!(recorder, x, g)
    end
    @test recorder.n == 100_001
    @test length(recorder.iterates) == WarmupHMC._RECORDED_ITERATES
    @test Base.summarysize(recorder) == bytes
    @test WarmupHMC._recorded_history_start(recorder, 100_000, 1) === nothing
    @test WarmupHMC._recorded_history_start(recorder, 100_000, 0) == 0
    @test WarmupHMC._recorded_history_start(recorder, 10, 1) === nothing
end

@testset "the checks read what they read from the whole trajectory" begin
    from_recorder = from_replay = 0
    for target in targets, seed in 1:10
        recorder = WarmupHMC._TrajectoryRecorder(target)
        result = quietly(() -> WarmupHMC._pathfinder_or_refit(target; rng=Xoshiro(seed),
            init=starts(target, seed), callback=no_progress, maxiters=100, recorder))
        P, G = replayed(target, result)
        expected = checks_on_trajectory(P, G, result.fit_iteration, result.fit_distribution.Σ)
        kept = first(first(recorder.iterates)) <= expected.lo
        from_recorder += kept
        from_replay += !kept
        # With the run's recorder, and without it (a caller's PathfinderResult).
        for path in (WarmupHMC._pathfinder_path(target, result, recorder), WarmupHMC._pathfinder_path(target, result))
            @test path.span.lo == expected.lo
            @test path.span.drops_curvature == expected.drops
            @test path.span.fit_point == expected.fit_point
            @test WarmupHMC._trajectory_scale(path.span) ≈ expected.scale rtol=1e-12
            @test path.start == first(P)
            @test path.n_iterates == length(P)
            @test last(path.iterates) == (0, first(P))
            @test all(x == P[i + 1] for (i, x) in path.iterates)
        end
    end
    # Both sources of the history stretch are exercised: some Gaussian fits are
    # made near the end of their run, the funnel's early (measured 2 and 18).
    @test from_recorder >= 1
    @test from_replay >= 1
end

@testset "a caller's PathfinderResult without its trace starts the chain" begin
    for target in targets, seed in 1:3
        reference = Xoshiro(seed)
        result = quietly(() -> WarmupHMC.mypathfinder(target; rng=reference, init=starts(target, seed), maxiters=100))
        rng = Xoshiro(seed)
        own = quietly(() -> WarmupHMC.initialize_mcmc(target, starts(target, seed); rng, progress=nothing))
        given = quietly(() -> WarmupHMC.initialize_mcmc(target, result; progress=nothing))
        @test given.position == own.position
        @test given.squared_scale == own.squared_scale
        # The replay draws no random number.
        @test WarmupHMC._xoshiro_words(rng) == WarmupHMC._xoshiro_words(reference)
    end
end

@testset "with the trace, the checks read it" begin
    for target in targets, seed in 1:3
        traced = quietly(() -> WarmupHMC.mypathfinder(target; rng=Xoshiro(seed), init=starts(target, seed),
            maxiters=100, save_trace=true))
        path = WarmupHMC._pathfinder_path(target, traced)
        P, G = traced.optim_trace.points, traced.optim_trace.gradients
        expected = checks_on_trajectory(P, G, traced.fit_iteration, traced.fit_distribution.Σ)
        @test path.span.lo == expected.lo
        @test path.span.drops_curvature == expected.drops
        @test path.span.fit_point == expected.fit_point
        @test WarmupHMC._trajectory_scale(path.span) ≈ expected.scale rtol=1e-12
        @test [x for (_, x) in path.iterates] == reverse(P)
        start = quietly(() -> WarmupHMC.initialize_mcmc(target, starts(target, seed);
            rng=Xoshiro(seed), progress=nothing, save_trace=true))
        @test all(isfinite, start.position)
    end
end
end
