@testitem "Opt-in step-size halving on the window that reaches n_draws" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Logging
    using LogDensityProblems
    using Serialization

# Focused invocation from the package root:
# julia --startup-file=no --project=test -e 'using TestItemRunner; using LinearAlgebra;
# BLAS.set_num_threads(1); TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "stepsize_halving.jl"))'

# With `divergence_tolerance` below 1, a window that reaches `n_draws` with more
# than that fraction of its retained draws divergent halves the frozen step size,
# drops the draws and samples `n_draws` new ones without re-adapting anything, at
# most three times per run (snag regularised-hors-bcdcdd84; user decision
# WarmupHMC:snag.regularised-hors-bcdcdd84/decisions/2026-11-10T10-05-15-713-0sux6ko:
# "why not just half the stepsize? ... default off, opt in.").

# A standard normal confined to the box |x_i| < 0.5 (as in divergence_gate.jl):
# NUTS trajectories run into the walls and diverge whatever the step size, so
# `divergence_tolerance = 0` halves on every final window and reaches the bound.
struct Box end
LogDensityProblems.dimension(::Box) = 2
LogDensityProblems.capabilities(::Type{Box}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity_and_gradient(::Box, x) =
    all(xi -> abs(xi) < 0.5, x) ? (-sum(abs2, x) / 2, -collect(float.(x))) : (-Inf, zero(float.(x)))
LogDensityProblems.logdensity(p::Box, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

box_init = (; position=zeros(2), squared_scale=fill(0.01, 2))
final_eps(state) = WarmupHMC.DynamicHMC.final_ϵ(state.stepsize_state)
record(log) = (state, stage) -> begin
    push!(log, (; stage, state.stepsize_halvings, state.stepsize, eps=final_eps(state),
        state.n_samples, state.current_transition_counter, state.stepsize_adaptation_limit))
    false
end

@testset "halves at most three times, without re-adapting, then warns" begin
    log = NamedTuple[]
    dir = mktempdir()
    result = @test_logs (:warn, r"step-size halvings") match_mode=:any adaptive_warmup_mcmc(
        Xoshiro(20261110), Box(); init=box_init, n_draws=100, divergence_tolerance=0.0,
        max_divergent_fraction=1.0, checkpoint_dir=dir, callback=record(log), progress=nothing)
    @test result.stepsize_halvings == 3
    @test size(result.posterior_position, 2) >= 100
    @test result.n_divergent_samples > 0
    # Whenever the step size is frozen, it is the dual-averaged one halved as
    # often as the run has halved: restarts re-adapt and freeze at that size again.
    for b in log
        b.current_transition_counter >= b.stepsize_adaptation_limit || continue
        @test b.stepsize == ldexp(b.eps, -b.stepsize_halvings)
    end
    halvings = [i for i in 2:length(log) if log[i].stepsize_halvings > log[i-1].stepsize_halvings]
    @test length(halvings) == 3
    for i in halvings
        b = log[i]
        # The draws were dropped; the kernel stays frozen (no re-adaptation), so the
        # next transition is retained under the halved step size.
        @test b.n_samples == 0
        @test b.current_transition_counter == b.stepsize_adaptation_limit
        payload = deserialize(joinpath(dir, "cp_window_$(i - 1).jls"))
        @test isempty(payload.posterior_position)
        @test size(payload.dropped_posterior_position, 2) >= 100
        @test payload.dropped_n_divergent_samples > 0
        @test payload.stepsize_halvings == b.stepsize_halvings
        @test payload.sampling_evaluation_counter == 0
    end

    @testset "the count is checkpointed, so a resume keeps the factor and the bound" begin
        payload = deserialize(joinpath(dir, "cp_latest.jls"))
        @test payload.stepsize_halvings == 3
        resumed_log = NamedTuple[]
        resumed = @test_logs (:warn, r"step-size halvings") match_mode=:any adaptive_warmup_mcmc(
            Xoshiro(1), Box(); n_draws=200, divergence_tolerance=0.0, max_divergent_fraction=1.0,
            checkpoint_dir=dir, resume=true, callback=record(resumed_log), progress=nothing)
        @test all(b -> b.stepsize_halvings == 3, resumed_log)
        @test resumed.stepsize_halvings == 3
        @test size(resumed.posterior_position, 2) >= 200
    end

    @testset "a checkpoint written before the key existed resumes with no halving" begin
        old = deserialize(joinpath(dir, "cp_latest.jls"))
        legacy = NamedTuple(k => v for (k, v) in pairs(old) if k != :stepsize_halvings)
        state = WarmupHMC.restore_state(legacy, Box(), nothing; n_draws=400)
        @test state.stepsize_halvings == 0
    end
end

@testset "divergence_tolerance = 1 (the default) switches halving off" begin
    log = NamedTuple[]
    result = @test_logs min_level=Logging.Warn adaptive_warmup_mcmc(
        Xoshiro(20261110), Box(); init=box_init, n_draws=100, max_divergent_fraction=1.0,
        callback=record(log), progress=nothing)
    @test result.stepsize_halvings == 0
    @test all(b -> b.stepsize_halvings == 0, log)
end

@testset "a run without divergences is unchanged by opting in" begin
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    default = adaptive_warmup_mcmc(Xoshiro(20261024), target; n_draws=300, progress=nothing)
    opted = adaptive_warmup_mcmc(Xoshiro(20261024), target; n_draws=300,
        divergence_tolerance=0.0, progress=nothing)
    @test default.n_divergent_samples == 0
    @test opted.stepsize_halvings == 0
    @test default.posterior_position == opted.posterior_position
    @test default.stepsize == opted.stepsize
    @test default.total_evaluation_counter == opted.total_evaluation_counter
end

@testset "out-of-range tolerances are refused" begin
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    for bad in (-0.1, 1.5, NaN, :off)
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), target; n_draws=10,
            divergence_tolerance=bad, progress=nothing)
    end
end

@testset "the completion stuck check compares adapted step sizes" begin
    # A halving answers divergences; it must not make a chain look stuck next to
    # siblings that did not halve.
    @test WarmupHMC._adapted_stepsize((; stepsize=0.1, stepsize_halvings=2)) == 0.4
    @test WarmupHMC._adapted_stepsize((; stepsize=0.1)) == 0.1
end
end
