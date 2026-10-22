@testitem "cooperative_warmup_mcmc: shared centering" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems, Serialization
    using DifferentiationInterface, Enzyme

# The merged sampler shares one nonlinear reparametrization across its chains and
# refits it from the pooled halos at batch ends where every chain restarts
# (decision 0m327rr). Neal's funnel with no data is the textbook case where the
# non-centered parametrization wins, so a run started fully centered must move.

# v ~ N(0, 3²); x_i ~ N(0, exp(v)) in the MODEL frame.
struct Funnel
    k::Int
end
LogDensityProblems.dimension(f::Funnel) = f.k + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(f::Funnel, y) = first(LogDensityProblems.logdensity_and_gradient(f, y))
function LogDensityProblems.logdensity_and_gradient(f::Funnel, y)
    v = y[1]; x = @view y[2:end]; e = exp(-v)
    lp = -v^2 / 18 - sum(abs2, x) * e / 2 - f.k * v / 2
    (lp, [-v / 9 + sum(abs2, x) * e / 2 - f.k / 2; -x .* e])
end
# Each x_i's location is 0 and its LOG-scale is v/2; target and source start centered.
funnel_rp(k) = ReparametrizedProblem(
    IndexedReparametrization([i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0), 0.0, y -> y[1] / 2)
                              for i in 2:k+1]),
    Funnel(k), AutoEnzyme())

@testset "the pooled refit moves a funnel toward non-centered" begin
    dir = mktempdir()
    out = cooperative_warmup_mcmc(Xoshiro(1), funnel_rp(4); n_chains=4, n_draws=1000, checkpoint_dir=dir)
    @test out.stop_reason === :n_draws
    sources = deserialize(joinpath(dir, "chain_1", "cp_latest.jls")).reparam_sources
    @test all(s -> last(s).c < 0.5, sources)
    # Every chain shares it.
    for i in 2:4
        @test deserialize(joinpath(dir, "chain_$i", "cp_latest.jls")).reparam_sources == sources
    end
    @test any(r -> any(b -> b.centering_refit, r.batches), out.results)
    # Model-frame draws: v keeps its N(0, 3²) marginal.
    v = reduce(vcat, [r.posterior_position[1, :] for r in out.results])
    @test abs(mean(v)) < 0.5
    @test 2.5 < std(v) < 3.5
end

@testset "online candidate scoring is not pooled yet" begin
    # A capability gap, not a wrong shape: the run refuses (ArgumentError) instead
    # of silently never adapting. Tracked as WarmupHMC todo 08l86ue.
    plan = CandidateScoringPlan((x, g) -> nothing, (pair_index, candidate, frame) -> nothing)
    rp = ReparametrizedProblem(funnel_rp(2).reparametrizer, Funnel(2), AutoEnzyme(); scoring_plan=plan)
    adapts = try
        cooperative_warmup_mcmc(Xoshiro(1), rp; n_chains=2, n_draws=10)
        true
    catch e
        e isa ArgumentError || rethrow()
        false
    end
    @test_broken adapts
end
end
