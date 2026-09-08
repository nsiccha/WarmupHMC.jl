# Focused invocation (root environment; no private data or AD backend needed):
# julia --startup-file=no --project=. -t4 -e 'using Test, WarmupHMC, Random,
# LinearAlgebra, Statistics, LogDensityProblems, Distributions;
# BLAS.set_num_threads(1); include("test/test_problems.jl");
# include("test/completion_policy.jl")'

_completion_fixture(i; n=5) = (; posterior_position=fill(Float64(i), 2, n),
    n_divergent_samples=i % 2)

@testset "controlled completion times and original identities" begin
    now = Ref(0.0)
    s = WarmupHMC._CompletionState(ReentrantLock(), () -> now[], 0.0, 2, 3.0,
        nothing, nothing, fill(:pending, 5), Any[nothing for _ in 1:5],
        Any[nothing for _ in 1:5], Union{Nothing,Float64}[nothing for _ in 1:5])
    # Finishing out of order preserves IDs; quorum is based on full results.
    complete = (i, stop) -> (_completion_fixture(i), true)
    now[] = 1.0
    WarmupHMC._completion_worker!(complete, s, 3)
    @test isnothing(s.quorum_at)
    now[] = 4.0
    WarmupHMC._completion_worker!(complete, s, 1)
    @test s.quorum_at == 4.0
    @test s.cutoff_at == 7.0
    now[] = 6.0
    WarmupHMC._completion_worker!(complete, s, 4)
    @test s.statuses[4] === :completed
    # A worker already running across the deadline finishes but is omitted.
    WarmupHMC._completion_worker!(s, 2) do i, stop
        now[] = 7.0
        @test stop()
        _completion_fixture(i), true
    end
    @test s.statuses[2] === :completed_after_cutoff
    @test s.results[2].posterior_position == fill(2.0, 2, 5)
    WarmupHMC._completion_worker!(complete, s, 5)
    @test s.statuses[5] === :not_started
    @test findall(==(:completed), s.statuses) == [1, 3, 4]
end

@testset "heterogeneous workers settle before return" begin
    entered = Channel{Int}(3)
    release = [Channel{Nothing}(1) for _ in 1:3]
    cleaned = Threads.Atomic{Int}(0)
    slow_finished = Threads.Atomic{Bool}(false)
    batch = Threads.@spawn WarmupHMC._completion_batch(3; min_completed=1, grace_seconds=0.0) do i, stop
        put!(entered, i)
        take!(release[i])
        try
            if i == 2
                return _completion_fixture(i), true
            end
            # Emulates a long active window; observe cancellation only at its
            # checkpoint. Successful return requires both slow workers to exit.
            while !stop()
                yield()
            end
            i == 3 && (slow_finished[] = true)
            _completion_fixture(i; n=2), false
        finally
            Threads.atomic_add!(cleaned, 1)
        end
    end
    @test sort([take!(entered) for _ in 1:3]) == [1, 2, 3]
    foreach(c -> put!(c, nothing), release)
    out = fetch(batch)
    @test cleaned[] == 3
    @test slow_finished[]
    @test out.completion.completed_chain_indices == [2]
    @test out.completion.omitted_chain_indices == [1, 3]
    @test out.completion.n_started == 3
    @test only(out.results).chain_index == 2
    @test out.completion.n_samples == 5
    @test out.completion.n_divergent_samples == 0
    @test out.completion.stop_reason === :grace_expired
    @test all(c -> c.n_retained_samples == (c.chain_index == 2 ? 5 : 0), out.completion.chains)
end

@testset "grace admits extra workers, failures stay explicit" begin
    # All clocks are zero: an arbitrarily slow worker remains inside grace.
    out = WarmupHMC._completion_batch(4; min_completed=2, grace_seconds=1.0, clock=() -> 0.0) do i, stop
        i == 2 && error("controlled chain 2 failure")
        _completion_fixture(i), true
    end
    @test out.completion.completed_chain_indices == [1, 3, 4]
    @test out.completion.failed_chain_indices == [2]
    @test out.completion.omitted_chain_indices == [2]
    @test out.completion.n_completed == 3
    @test out.completion.n_failed == 1
    @test out.completion.stop_reason === :all_settled
    @test occursin("controlled chain 2 failure", sprint(showerror, out.completion.chains[2].error))
    @test out.completion.n_samples == sum(r -> size(r.posterior_position, 2), out.results)
    @test out.completion.n_divergent_samples == sum(r -> r.n_divergent_samples, out.results)

    unmet = WarmupHMC._completion_batch(2; min_completed=2, grace_seconds=0.0) do i, stop
        i == 1 && error("failed initialization")
        _completion_fixture(i; n=0), false
    end
    @test isempty(unmet.results)
    @test !unmet.completion.quorum_met
    @test unmet.completion.stop_reason === :quorum_unmet
    @test occursin("0 of 2", sprint(showerror, WarmupHMC.CompletionQuorumError(unmet)))
end

@testset "already-complete admission precedes scheduling" begin
    initial = Dict(3 => _completion_fixture(3), 1 => _completion_fixture(1))
    out = WarmupHMC._completion_batch(4; min_completed=1, grace_seconds=0.0,
        initial_results=initial) do i, stop
        error("No incomplete worker should start after zero grace.")
    end
    @test out.completion.completed_chain_indices == [1, 3]
    @test out.completion.started_chain_indices == [1, 3]
    @test out.completion.failed_chain_indices == Int[]
    @test out.completion.omitted_chain_indices == [2, 4]
end

@testset "numerical equivalence and recovery" begin
    BLAS.set_num_threads(1)
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    cfg = (; n_draws=600, progress=nothing)
    baseline = adaptive_warmup_mcmc([Xoshiro(i) for i in 1:3], p; cfg..., parallel=false)
    out = completion_warmup_mcmc([Xoshiro(i) for i in 1:3], p; cfg...)
    @test out.completion.stop_reason === :all_completed
    @test out.completion.completed_chain_indices == collect(1:3)
    @test out.completion.n_samples == 1800
    for (i, r) in enumerate(out.results)
        @test r.chain_index == i
        @test r.posterior_position == baseline[i].posterior_position
        @test r.posterior_gradient == baseline[i].posterior_gradient
        @test r.n_divergent_samples == baseline[i].n_divergent_samples
        @test r.total_evaluation_counter == baseline[i].total_evaluation_counter
        @test Set(keys(r)) == union(Set(keys(baseline[i])), Set((:chain_index, :n_samples)))
    end
    draws = hcat(getproperty.(out.results, :posterior_position)...)
    @test all(abs.(vec(mean(draws; dims=2)) .- p.mu) .< 0.2 .* p.sigma)
    @test all(abs.(vec(std(draws; dims=2)) ./ p.sigma .- 1) .< 0.2)
    @test out.completion.n_divergent_samples <= 3
end

@testset "safe checkpoints and immutable terminal selection" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    mktempdir() do dir
        # A stopped callback is an incomplete chain, not a completed window.
        err = try
            completion_warmup_mcmc([Xoshiro(7), Xoshiro(8)], p;
                n_draws=40, checkpoint_dir=dir, callback=(s, stage) -> true)
        catch e
            e
        end
        @test err isa WarmupHMC.CompletionQuorumError
        @test err.outcome.completion.n_completed == 0
        @test err.outcome.completion.n_samples == 0
        @test all(c -> c.status === :stopped, err.outcome.completion.chains)
        @test isfile(joinpath(err.outcome.completion.attempt_directory, "run_summary.json"))
        @test !isfile(joinpath(dir, "completion_terminal.jls"))
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(7), Xoshiro(8)], p;
            n_draws=40, checkpoint_dir=dir)
        # Checkpoints are ordinary adaptive payloads and contain source state.
        for i in 1:2
            cp = WarmupHMC.deserialize(joinpath(dir, "chain_$i", "cp_latest.jls"))
            @test cp.sampler === :adaptive
            @test size(cp.posterior_position, 2) == 0
            @test !haskey(cp, :min_completed)
            @test !haskey(cp, :grace_seconds)
        end
        # Complete chain 2 separately. Resume must admit it before any worker
        # has a chance to initialize chain 1, even with zero grace.
        ordinary = adaptive_warmup_mcmc(Xoshiro(999), p;
            n_draws=40, checkpoint_dir=joinpath(dir, "chain_2"), resume=true)
        out = completion_warmup_mcmc([Xoshiro(999), Xoshiro(999)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true)
        @test out.completion.completed_chain_indices == [2]
        @test out.completion.omitted_chain_indices == [1]
        @test out.results[1].posterior_position == ordinary.posterior_position
        @test out.completion.chains[1].status === :not_started
        cp1 = read(joinpath(dir, "chain_1", "cp_latest.jls"))
        @test isfile(joinpath(dir, "completion_terminal.jls"))
        @test isfile(joinpath(out.completion.attempt_directory, "run_summary.json"))
        @test occursin("\"completed_chain_indices\":[2]",
            read(joinpath(out.completion.attempt_directory, "run_summary.json"), String))
        # Advance the omitted checkpoint: reopening must still preserve the
        # recorded one-chain selection, draws, times, counts and original IDs.
        adaptive_warmup_mcmc(Xoshiro(999), p;
            n_draws=40, checkpoint_dir=joinpath(dir, "chain_1"), resume=true)
        @test read(joinpath(dir, "chain_1", "cp_latest.jls")) != cp1
        again = completion_warmup_mcmc([Xoshiro(11), Xoshiro(12)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true,
            callback=(s, stage) -> error("Terminal reopen must never run a callback"))
        @test again.completion == out.completion
        @test again.results[1].posterior_position == out.results[1].posterior_position
        @test again.results[1].chain_index == 2
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p;
            n_draws=40, checkpoint_dir=dir, resume=true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=41, min_completed=1, checkpoint_dir=dir, resume=true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, grace_seconds=1, checkpoint_dir=dir, resume=true)
    end
end

@testset "public policy validation precedes work" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    for value in (0, 3, 1.5, true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p; min_completed=value)
    end
    for value in (-1.0, Inf, NaN)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; grace_seconds=value)
    end
    for value in (0, -1, typemax(Int), true, 1.5)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; n_draws=value)
    end
    @test_throws ArgumentError completion_warmup_mcmc(Xoshiro[], p)
    @test_throws DimensionMismatch completion_warmup_mcmc([Xoshiro(1)], [p, p])
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; resume=true)
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; n_draw=5)
end
