@testitem "Divergence gate on the window that reaches n_draws" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Logging
    using LogDensityProblems
    using Serialization

# Focused invocation from the package root:
# julia --startup-file=no --project=test -e 'using TestItemRunner; using LinearAlgebra;
# BLAS.set_num_threads(1); TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "divergence_gate.jl"))'

# A run ends as soon as its retained draws reach `n_draws`, without the restart
# test on the window that got there. When more than `max_divergent_fraction` of
# those draws diverged, `adaptive_warmup_mcmc` restarts warm-up instead, at most
# twice per run, and then returns the draws with a warning (snag
# final-window-mos-3252ed56; user decision
# WarmupHMC:snag.final-window-mos-3252ed56/decisions/2026-10-23T09-54-10-257-1b2iq06).

# A standard normal confined to the box |x_i| < 0.5: nearly flat inside, so NUTS
# trajectories run into the walls and many transitions diverge whatever the step
# size. `max_divergent_fraction = 0` then makes the gate fire on every final
# window, which exercises its bound deterministically.
struct Box end
LogDensityProblems.dimension(::Box) = 2
LogDensityProblems.capabilities(::Type{Box}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity_and_gradient(::Box, x) =
    all(xi -> abs(xi) < 0.5, x) ? (-sum(abs2, x) / 2, -collect(float.(x))) : (-Inf, zero(float.(x)))
LogDensityProblems.logdensity(p::Box, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

box_init = (; position=zeros(2), squared_scale=fill(0.01, 2))
gate_counts(counts) = (state, stage) -> (push!(counts, state.divergence_gate_restarts); false)

@testset "the gate restarts at most twice, then warns and returns the draws" begin
    counts = Int[]
    dir = mktempdir()
    result = @test_logs (:warn, r"divergence-gate restarts") match_mode=:any adaptive_warmup_mcmc(
        Xoshiro(20261023), Box(); init=box_init, n_draws=100, max_divergent_fraction=0.0,
        checkpoint_dir=dir, callback=gate_counts(counts), progress=nothing)
    @test maximum(counts) == 2
    @test last(counts) == 2
    @test size(result.posterior_position, 2) >= 100
    @test result.n_divergent_samples > 0
    # Every gate restart dropped a full draw set, kept under `dropped_*` as at any restart.
    gate_windows = [i for i in 2:length(counts) if counts[i] > counts[i-1]]
    @test length(gate_windows) == 2
    for w in gate_windows
        payload = deserialize(joinpath(dir, "cp_window_$(w - 1).jls"))
        @test isempty(payload.posterior_position)
        @test size(payload.dropped_posterior_position, 2) >= 100
    end

    @testset "the count is checkpointed, so a resume keeps the bound" begin
        payload = deserialize(joinpath(dir, "cp_latest.jls"))
        @test payload.divergence_gate_restarts == 2
        resumed_counts = Int[]
        resumed = @test_logs (:warn, r"divergence-gate restarts") match_mode=:any adaptive_warmup_mcmc(
            Xoshiro(1), Box(); n_draws=200, max_divergent_fraction=0.0, checkpoint_dir=dir,
            resume=true, callback=gate_counts(resumed_counts), progress=nothing)
        @test all(==(2), resumed_counts)
        @test size(resumed.posterior_position, 2) >= 200
    end

    @testset "a checkpoint written before the key existed resumes with the count at zero" begin
        old = deserialize(joinpath(dir, "cp_latest.jls"))
        legacy = NamedTuple(k => v for (k, v) in pairs(old) if k != :divergence_gate_restarts)
        state = WarmupHMC.restore_state(legacy, Box(), nothing; n_draws=400)
        @test state.divergence_gate_restarts == 0
    end
end

@testset "max_divergent_fraction = 1 switches the gate off" begin
    counts = Int[]
    @test_logs min_level=Logging.Warn adaptive_warmup_mcmc(
        Xoshiro(20261023), Box(); init=box_init, n_draws=100, max_divergent_fraction=1.0,
        callback=gate_counts(counts), progress=nothing)
    @test all(iszero, counts)
end

@testset "a run the gate does not fire on is unchanged" begin
    # Default fraction against the gate switched off, same seed: draw for draw equal.
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    counts = Int[]
    default = adaptive_warmup_mcmc(Xoshiro(20261024), target; n_draws=300,
        callback=gate_counts(counts), progress=nothing)
    off = adaptive_warmup_mcmc(Xoshiro(20261024), target; n_draws=300,
        max_divergent_fraction=1.0, progress=nothing)
    @test all(iszero, counts)
    @test default.posterior_position == off.posterior_position
    @test default.total_evaluation_counter == off.total_evaluation_counter
end

@testset "out-of-range fractions are refused" begin
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    for bad in (-0.1, 1.5, NaN)
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), target; n_draws=10,
            max_divergent_fraction=bad, progress=nothing)
    end
end
end
