@testitem "returned ess is per coordinate, in coordinate order" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using DifferentiationInterface, Enzyme

# Every returned `ess` is addressable by coordinate: `ess[k]` is the bulk ESS of
# coordinate `k` of the draws returned beside it, in the frame those draws are in
# (snag `result-ess-is-so-3a7b42a3`, decision `1x29n9m`). It used to be sorted
# ascending, so `ess[k]` was the k-th smallest ESS, and a consumer naming the
# worst-mixing parameters by `sortperm(fit.ess)` named coordinates 1, 2, 3, … in
# declaration order. For a `ReparametrizedProblem` it was also the SOURCE-frame
# ESS beside MODEL-frame draws.
#
# The reference is the recipe a consumer would write, computed independently of
# `WarmupHMC._coordinate_ess`.
reference_ess(draws) = vec(WarmupHMC.MCMCDiagnosticTools.ess(
    reshape(Matrix(draws)', (:, 1, size(draws, 1)))))

# A 2-D funnel in its CENTERED (model) parametrization: y1 ~ N(0, 1) and
# y2 | y1 ~ N(0, exp(y1)).
struct _Funnel2 end
LogDensityProblems.dimension(::_Funnel2) = 2
LogDensityProblems.capabilities(::Type{_Funnel2}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_Funnel2, y) = -y[1]^2 / 2 - (y[2] * exp(-y[1]))^2 / 2 - y[1]
LogDensityProblems.logdensity_and_gradient(p::_Funnel2, y) = (
    LogDensityProblems.logdensity(p, y),
    [-y[1] + y[2]^2 * exp(-2y[1]) - 1, -y[2] * exp(-2y[1])],
)

# The sampler works non-centered: y2 = x2 * exp(x1), so the source frame is a
# standard normal. Model coordinate 2 depends on BOTH source coordinates, so its
# bulk ESS (which is rank-based, hence invariant only under a per-coordinate
# monotone map) differs between the two frames, while coordinate 1 is shared.
funnel_rp() = ReparametrizedProblem(
    IndexedReparametrization([
        2 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), 0.0, x -> x[1]),
    ]),
    _Funnel2(), AutoEnzyme(),
)
# The inverse of the map above, applied to model-frame draws.
source_draws(y) = vcat(y[1:1, :], (y[2, :] .* exp.(-y[1, :]))')

_blas_threads = LinearAlgebra.BLAS.get_num_threads()
LinearAlgebra.BLAS.set_num_threads(1)
try
    @testset "_min_ess is the ascending sort's first element" begin
        for ess in ([3.0, 1.0, 2.0], [NaN, 4.0, 2.5], [2.0, NaN, NaN], [0.0, 0.0])
            @test WarmupHMC._min_ess(ess) == sort(ess)[1]
        end
        @test isnan(WarmupHMC._min_ess([NaN, NaN]))
    end

    target = DiagGaussian([0.0, 1.0, -2.0, 0.5, 3.0, -1.0], [1.0, 0.3, 2.0, 5.0, 0.7, 1.5])
    d = LogDensityProblems.dimension(target)

    @testset "adaptive_warmup_mcmc: plain target" begin
        cap = CaptureProgress()
        fit = adaptive_warmup_mcmc(Xoshiro(11), target; n_draws=400,
            progress=_Treebars.ProgressNode(cap))
        n = size(fit.posterior_position, 2)
        @test length(fit.ess) == d
        @test fit.ess == reference_ess(fit.posterior_position)
        # Non-vacuous: a sorted vector would fail the equality above.
        @test !issorted(fit.ess)
        # The progress label still RANKS its entries (worst three first).
        vals = [x isa Pair ? x.second : x for x in cap.seen]
        strs = filter(x -> x isa AbstractString, vals)
        ess_labels = filter(s -> endswith(s, " samples.") && occursin(" from ", s), strs)
        @test !isempty(ess_labels)
        @test last(ess_labels) == WarmupHMC.short_string(sort(fit.ess)) * " from $n samples."
        @test last(ess_labels) != WarmupHMC.short_string(fit.ess) * " from $n samples."
        @test any(s -> startswith(s, "min. ESS: $(WarmupHMC.short_string(minimum(fit.ess))),"), strs)

        # Without monitoring the returned vector is unchanged: all zeros.
        unmonitored = adaptive_warmup_mcmc(Xoshiro(11), target; n_draws=100, progress=nothing)
        @test unmonitored.ess == zeros(d)
    end

    @testset "adaptive_warmup_mcmc: ReparametrizedProblem reports the model frame" begin
        fit = adaptive_warmup_mcmc(Xoshiro(3), funnel_rp(); n_draws=400,
            nonlinear_adapt=false, monitor_ess=true, progress=nothing)
        y = fit.posterior_position
        @test fit.ess == reference_ess(y)
        working = reference_ess(source_draws(y))
        @test fit.ess[1] == working[1]
        @test fit.ess[2] != working[2]
    end

    @testset "cooperative_warmup_mcmc: per-chain results" begin
        for (lpdf, kw) in ((target, (;)), (funnel_rp(), (; nonlinear_adapt=false)))
            out = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
                n_cores=1, n_draws=300, n_evaluations_budget=400_000, kw...)
            estimated = [r for r in out.results if size(r.posterior_position, 2) > 10]
            @test !isempty(estimated)
            for r in estimated
                @test r.ess == reference_ess(r.posterior_position)
            end
        end
    end

    @testset "stream_mcmc: ess of the returned source-frame draws" begin
        r = stream_mcmc(Xoshiro(5), target, target.mu; path=tempname(), n_draws=300,
            metric=target.sigma, stepsize=0.6, max_tree_depth=8)
        @test r.ess == reference_ess(r.draws)
        @test !issorted(r.ess)
        WarmupHMC.close_stream(r)
    end
finally
    LinearAlgebra.BLAS.set_num_threads(_blas_threads)
end
end
