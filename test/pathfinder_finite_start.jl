@testitem "Pathfinder's start is a point at which the target is finite" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, LogDensityProblems

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "pathfinder_finite_start.jl"))'

# The contract (snag `pathfinder-draw-c1e43cb7`, reporter NLMEEstimation): a
# finite vector `init` gives a start at which the target is finite. Pathfinder's
# fit is a Gaussian, so the draw it starts the chain from can land outside a
# bounded support, where the log density is -Inf and no NUTS transition can
# start; the initializer then redraws from the fit and, when no draw lands
# inside, starts at an L-BFGS iterate. A first draw inside is untouched.

# `inner` restricted to |x[1]| <= bound; `outside(x)` is what it answers beyond
# it (-Inf with a zero gradient, as NLMEEstimation's `Bounded` model scores a
# subject whose first random effect exceeds `max_eta`; or NaN).
struct Boxed{P,F}
    inner::P
    bound::Float64
    outside::F
end
Boxed(inner, bound) = Boxed(inner, bound, x -> (-Inf, zero(x)))
LogDensityProblems.capabilities(::Type{<:Boxed{P}}) where {P} = LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(p::Boxed) = LogDensityProblems.dimension(p.inner)
LogDensityProblems.logdensity_and_gradient(p::Boxed, x) =
    abs(x[1]) <= p.bound ? LogDensityProblems.logdensity_and_gradient(p.inner, x) : p.outside(x)
LogDensityProblems.logdensity(p::Boxed, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))
inside(target, q) = isfinite(LogDensityProblems.logdensity(target, q))

# NLMEEstimation's shape: the conditional's mode sits 0.2 SD inside the bound,
# so about 40 % of the fit's mass lies beyond it.
straddling = Boxed(DiagGaussian([0.5, 0.0, 0.0], [0.5, 1.0, 1.0]), 0.6)
# The fit's mass lies entirely beyond the bound: L-BFGS stops at the bound and
# the fit's mean is the unconstrained mode at 3, with SD 0.2.
beyond = Boxed(DiagGaussian([3.0, 0.0, 0.0], [0.2, 1.0, 1.0]), 0.6)
# The same box, answering NaN beyond the bound as an unguarded model does.
nan_outside = Boxed(DiagGaussian([0.5, 0.0, 0.0], [0.5, 1.0, 1.0]), 0.6, x -> (NaN, fill(NaN, length(x))))
init = zeros(3)
@test inside(straddling, init)

# Pathfinder's own run from `init`, as `initialize_mcmc` makes it: the first
# draw is the start the chain used to be refused at.
pathfinder(target, seed) = WarmupHMC.mypathfinder(target; rng=Xoshiro(seed), init=copy(init), maxiters=100)
first_draw(target, seed) = pathfinder(target, seed).draws[:, 1]
outside_seeds = filter(seed -> !inside(straddling, first_draw(straddling, seed)), 1:40)
inside_seeds = setdiff(1:40, outside_seeds)

blas_threads = BLAS.get_num_threads()
BLAS.set_num_threads(1)
try
    @testset "the mechanism: Pathfinder's first draw can land outside the support" begin
        # 12 of 40 when this was written; both kinds of seed must exist for the
        # checks below to mean anything.
        @test length(outside_seeds) >= 4
        @test length(inside_seeds) >= 4
    end

    @testset "a first draw outside the support is replaced by a redraw inside it" begin
        for seed in outside_seeds
            start = WarmupHMC.initialize_mcmc(straddling, copy(init); rng=Xoshiro(seed), progress=nothing)
            @test inside(straddling, start.position)
            @test start.position_and_gradient.q == start.position
            @test isfinite(start.position_and_gradient.ℓq)
            @test start.position != first_draw(straddling, seed)
            # Pathfinder's fit still gives the scale.
            @test start.squared_scale isa WarmupHMC.Pathfinder.WoodburyPDMat
        end
    end

    @testset "a first draw inside the support is untouched" begin
        for seed in inside_seeds[1:4]
            rng = Xoshiro(seed)
            start = WarmupHMC.initialize_mcmc(straddling, copy(init); rng, progress=nothing)
            @test start.position == first_draw(straddling, seed)
            # and no further random number is drawn: the stream is where
            # Pathfinder itself left it.
            reference = Xoshiro(seed)
            WarmupHMC.mypathfinder(straddling; rng=reference, init=copy(init), maxiters=100)
            @test WarmupHMC._xoshiro_words(rng) == WarmupHMC._xoshiro_words(reference)
        end
    end

    @testset "when no draw from the fit is inside, an L-BFGS iterate stands in" begin
        result = pathfinder(beyond, 1)
        # Positive control: the draw and the fit really lie beyond the bound,
        # and the fit's own iterate inside it.
        @test !inside(beyond, result.draws[:, 1])
        @test result.fit_distribution.μ[1] > 0.6
        points = result.optim_trace.points
        @test inside(beyond, points[result.fit_iteration + 1])
        # Pathfinder itself warns here too: its ELBO draw lands outside the
        # support as well, so its "failed after 1 tries" precedes our warning.
        start = @test_logs (:warn, r"L-BFGS iterate 1 of 1") match_mode=:any WarmupHMC.initialize_mcmc(
            beyond, copy(init); rng=Xoshiro(1), progress=nothing)
        @test start.position == points[result.fit_iteration + 1]
        @test inside(beyond, start.position)
    end

    @testset "a draw at which the model answers NaN is redrawn too" begin
        nan_seeds = filter(seed -> isnan(LogDensityProblems.logdensity(nan_outside, first_draw(nan_outside, seed))), 1:40)
        @test length(nan_seeds) >= 4
        for seed in nan_seeds[1:4]
            start = WarmupHMC.initialize_mcmc(nan_outside, copy(init); rng=Xoshiro(seed), progress=nothing)
            @test inside(nan_outside, start.position)
        end
    end

    @testset "every sampler starts from such an init" begin
        for seed in outside_seeds[1:3]
            state = ContinuousNUTSState(Xoshiro(seed), straddling; init=copy(init))
            @test inside(straddling, state.position)
            @test all(>(0), state.scale.diag)
            for _ in 1:5
                continuous_nuts_step(state, straddling)
            end
            @test inside(straddling, state.position)
            result = adaptive_warmup_mcmc(Xoshiro(seed), straddling; init=copy(init), n_draws=20)
            @test size(result.posterior_position, 2) >= 20
            @test all(c -> inside(straddling, c), eachcol(result.posterior_position))
        end
        out = cooperative_warmup_mcmc(Xoshiro(first(outside_seeds)), straddling;
            n_chains=2, n_draws=10, n_evaluations=200, init=copy(init))
        @test all(r -> all(c -> inside(straddling, c), eachcol(r.posterior_position)), out.results)
    end
finally
    BLAS.set_num_threads(blas_threads)
end
end
