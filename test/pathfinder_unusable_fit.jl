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

# Near 0 about a standard normal; the curvature along each coordinate is cosh(a x),
# 5e12 at |x| = 2 for a = 15, so from a random start in ±2 the first L-BFGS steps
# measure curvatures up to 1e12 times the bulk's.
struct SteepTails
    d::Int
    a::Float64
end
LogDensityProblems.capabilities(::Type{SteepTails}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::SteepTails) = p.d
LogDensityProblems.logdensity_and_gradient(p::SteepTails, x) =
    (-sum(xi -> (cosh(p.a * xi) - 1) / p.a^2, x), -sinh.(p.a .* x) ./ p.a)
LogDensityProblems.logdensity(p::SteepTails, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))
# The whole L-BFGS trajectory of a run: WarmupHMC's Pathfinder keeps none
# (`save_trace=false`), so it is replayed, as `_pathfinder_path` replays it.
function trajectory(target, result)
    isempty(result.optim_trace.points) || return result.optim_trace.points, result.optim_trace.gradients
    P, G = Vector{Float64}[], Vector{Float64}[]
    WarmupHMC._replay_trajectory((_, x, g) -> (push!(P, copy(x)); push!(G, copy(g))),
        target, result, length(result.optim_trace) - 1)
    P, G
end
# What the checks read of the run, replayed for want of the run's recorder.
path(target, result) = WarmupHMC._pathfinder_path(target, result)
drops_curvature(target, result) = path(target, result).span.drops_curvature
# The pairs the fit is built from, `l` joining iterates `l - 1` and `l`.
history_pairs(target, result) = (path(target, result).span.lo + 1):result.fit_iteration
# The pre-fix criterion: every pair up to the fit's iterate, not only the fit's own.
function drops_curvature_any_pair(target, result)
    (P, G), Σ = trajectory(target, result), result.fit_distribution.Σ
    any(1:result.fit_iteration) do l
        s, y = P[l + 1] - P[l], G[l] - G[l + 1]
        dot(s, y) > 0 && dot(y, Σ * y) > WarmupHMC._DROPPED_CURVATURE_RATIO * dot(s, y)
    end
end
rejected_pairs(target, result) = ((P, G) = trajectory(target, result);
    [l for l in 1:length(P) - 1 if !(dot(P[l + 1] - P[l], G[l] - G[l + 1]) >
        WarmupHMC._PATHFINDER_CURVATURE_EPS * sum(abs2, G[l] - G[l + 1]))])

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
        @test drops_curvature(stiff, result)
        @test WarmupHMC._fit_log_weight_change(result, path(stiff, result), draw, ℓ(stiff, draw)) <
            -WarmupHMC._implausible_drop(2)
        # Overflowing: no draw is finite at all, which the curvature check still sees.
        result = quietly(() -> pathfinder(overflowing, 1))
        @test !isfinite(ℓ(overflowing, result.draws[:, 1]))
        @test drops_curvature(overflowing, result)
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
            @test !drops_curvature(target, result)
            rng = Xoshiro(seed)
            start = WarmupHMC.initialize_mcmc(target, copy(start_point); rng, progress=nothing)
            @test start.position == result.draws[:, 1]
            @test start.squared_scale isa WarmupHMC.Pathfinder.WoodburyPDMat
            @test WarmupHMC._xoshiro_words(rng) == WarmupHMC._xoshiro_words(reference)
        end
    end

    @testset "only the pairs the fit is built from judge it" begin
        # Snag pathfinder-start-6abaede3: the first L-BFGS steps out of steep tails
        # measure the tails' curvature, far above the bulk's. Checked against a fit
        # made in the bulk 17-25 iterates later, they declared a sound fit unusable,
        # and the recovery then started the chain in the tails. Only the fit's own
        # L-BFGS history counts.
        steep = SteepTails(10, 15.0)
        old_flags = 0
        for seed in 1:10
            start_point = rand(Xoshiro(100 + seed), 10) .* 4 .- 2
            reference = Xoshiro(seed)
            result = quietly(() -> WarmupHMC.mypathfinder(steep; rng=reference, init=copy(start_point), maxiters=100))
            pairs = history_pairs(steep, result)
            @test last(pairs) == result.fit_iteration
            rejected = rejected_pairs(steep, result)
            accepted = count(l -> l ∉ rejected, pairs)
            @test accepted == min(WarmupHMC.Pathfinder.DEFAULT_HISTORY_LENGTH,
                count(l -> l ∉ rejected, 1:result.fit_iteration))
            old_flags += drops_curvature_any_pair(steep, result)
            @test !drops_curvature(steep, result)
            # The start is Pathfinder's own draw with its fit, and no further random number is drawn.
            rng = Xoshiro(seed)
            start = quietly(() -> WarmupHMC.initialize_mcmc(steep, copy(start_point); rng, progress=nothing))
            @test start.position == result.draws[:, 1]
            @test ℓ(steep, start.position) > -10
            @test start.squared_scale isa WarmupHMC.Pathfinder.WoodburyPDMat
            @test WarmupHMC._xoshiro_words(rng) == WarmupHMC._xoshiro_words(reference)
        end
        # Positive control: every pair up to the fit's iterate, as before the fix,
        # flags most of these fits. Measured 9 of 10.
        @test old_flags >= 5

        # The recovery's diagonal is taken from the same stretch. Over the whole
        # trajectory the tails' gradients (3e8 to 3e11 at these starts) shrink it to
        # 2e-5 here; over the fit's history it stays within 0.03-1 (measured).
        result = quietly(() -> WarmupHMC.mypathfinder(steep; rng=Xoshiro(1),
            init=rand(Xoshiro(101), 10) .* 4 .- 2, maxiters=100))
        @test first(history_pairs(steep, result)) > 1
        @test drops_curvature_any_pair(steep, result)
        scale = WarmupHMC._trajectory_scale(path(steep, result).span)
        @test all(s -> 0.01 < s < 10, scale)
        P, G = trajectory(steep, result)
        whole = [(var(p[i] for p in P) / var(g[i] for g in G))^(1 / 4) for i in 1:10]
        @test minimum(whole) < 1e-4

        # A caller may supply a PathfinderResult with a nondefault fit history.
        # Its covariance records that history independently of the optimizer.
        # With its trace (Pathfinder's default) the checks read it; without it
        # (`save_trace=false`) its trajectory is replayed.
        trace_modes = WarmupHMC._PATHFINDER_SAVE_TRACE_OPTIONAL ? ((;), (; save_trace=false)) : ((;),)
        for history_length in (2, 10), kw in trace_modes
            result = quietly(() -> WarmupHMC.Pathfinder.pathfinder(
                WarmupHMC._pathfinder_objective(steep, WarmupHMC.Pathfinder.SciMLBase.NoAD());
                dim=10, input=steep, rng=Xoshiro(1),
                init=rand(Xoshiro(101), 10) .* 4 .- 2,
                history_length, maxiters=100, ndraws=1, ndraws_elbo=1, ntries=1, kw...))
            @test isempty(result.optim_trace.points) == haskey(kw, :save_trace)
            span = history_pairs(steep, result)
            @test count(l -> l ∉ rejected_pairs(steep, result), span) == history_length
            @test !drops_curvature(steep, result)
            start = quietly(() -> WarmupHMC.initialize_mcmc(steep, result; progress=nothing))
            @test start.position == result.draws[:, 1]
        end
    end

    @testset "when the rescaled run gives no fit, the trajectory's diagonal stands in" begin
        # The branch needs a rescaled run with no fit; a target stationary at the
        # start gives one (L-BFGS converges before its first step). The failed fit
        # is the stiff target's, so the diagonal is its trajectory's.
        result = quietly(() -> pathfinder(stiff, 1))
        stationary = DiagGaussian(copy(init), [1.0, 1.0])
        start = @test_logs (:warn, r"gave no usable fit either") match_mode=:any WarmupHMC._rescaled_pathfinder_start(
            stationary, result, path(stiff, result); rng=Xoshiro(2), progress=nothing, maxiters=100)
        @test start.position == first(trajectory(stiff, result))[result.fit_iteration + 1]
        @test start.squared_scale isa Diagonal
        @test 1e-26 < start.squared_scale[1, 1] < 1e-22         # the trajectory measured x₁'s scale
        @test start.squared_scale[2, 2] ≈ 1 / 1.2               # never moved: 1/|∂ℓ/∂x₂| at the init
    end
finally
    BLAS.set_num_threads(blas_threads)
end
end
