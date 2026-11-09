@testitem "Adaptive checkpoints retain nonlinear adaptation" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, Serialization, LogDensityProblems
    using DifferentiationInterface, Enzyme
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: stop_after, latest, comparable, same

    # Neal's funnel: a curved, non-Gaussian target whose centering is refitted
    # from halo evidence. Every resume reconstructs a fresh centered problem.
    struct CheckpointFunnel end
    LogDensityProblems.dimension(::CheckpointFunnel) = 3
    LogDensityProblems.capabilities(::Type{CheckpointFunnel}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(::CheckpointFunnel, y)
        v = y[1]; x = @view y[2:end]; e = exp(-v)
        (-v^2 / 18 - sum(abs2, x) * e / 2 - v,
            [-v / 9 + sum(abs2, x) * e / 2 - 1; -x .* e])
    end
    LogDensityProblems.logdensity(p::CheckpointFunnel, y) = first(LogDensityProblems.logdensity_and_gradient(p, y))
    problem() = ReparametrizedProblem(
        IndexedReparametrization([i => Reparametrization(
            PartiallyCentered(1.0), PartiallyCentered(1.0), 0.0, y -> y[1] / 2) for i in 2:3]),
        CheckpointFunnel(), AutoEnzyme())
    config = (; n_draws=60, n_evaluations=32, stepsize_adaptation_limit=8,
        recording_target=32, max_tree_depth=5, variance_cond_target=1.0,
        monitor_ess=true, init=(; position=[0.3, -0.2, 0.4], squared_scale=ones(3)))

    @test !isnothing(Base.get_extension(WarmupHMC, :WarmupHMCEnzymeExt))
    mktempdir() do root
        full, sliced = joinpath(root, "full"), joinpath(root, "sliced")
        reference = adaptive_warmup_mcmc(Xoshiro(17), problem(); config..., checkpoint_dir=full)
        expected = latest(full)
        @test any(p -> last(p).c != 1.0, expected.reparam_sources)
        result = adaptive_warmup_mcmc(Xoshiro(17), problem(); config..., checkpoint_dir=sliced,
            stop_requested=stop_after(5))
        attempts = 1
        while latest(sliced).n_samples < config.n_draws && attempts < 150
            result = adaptive_warmup_mcmc(Xoshiro(999), problem(); config..., checkpoint_dir=sliced,
                resume=true, stop_requested=stop_after(5))
            attempts += 1
        end
        @test 2 < attempts < 150
        @test result.posterior_position == reference.posterior_position
        @test result.posterior_gradient == reference.posterior_gradient
        @test same(comparable(latest(sliced)), comparable(expected))
    end
end
