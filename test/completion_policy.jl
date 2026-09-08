# Focused invocation (root environment; no private data or AD backend needed):
# julia --startup-file=no --project=. -t4 -e 'using Test, WarmupHMC, Random,
# LinearAlgebra, Statistics, LogDensityProblems, Distributions;
# BLAS.set_num_threads(1); include("test/test_problems.jl");
# include("test/completion_policy.jl")'

_completion_fixture(i; n=5) = (; posterior_position=fill(Float64(i), 2, n),
    n_divergent_samples=i % 2)

struct CompletionBimodal end
LogDensityProblems.dimension(::CompletionBimodal) = 1
LogDensityProblems.capabilities(::Type{CompletionBimodal}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(::CompletionBimodal, x)
    a, b = -0.5 * (x[1] + 8)^2, -0.5 * (x[1] - 8)^2
    l = WarmupHMC.logaddexp(a, b)
    l - log(2) - log(2π) / 2, [exp(a-l) * (-x[1]-8) + exp(b-l) * (-x[1]+8)]
end
LogDensityProblems.logdensity(p::CompletionBimodal, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

@testset "controlled completion times and original identities" begin
    now = Ref(0.0)
    s = WarmupHMC._CompletionState(ReentrantLock(), () -> now[], 0.0, 2,
        nothing, fill(:pending, 5), Any[nothing for _ in 1:5],
        Any[nothing for _ in 1:5], Union{Nothing,Float64}[nothing for _ in 1:5])
    # Finishing out of order preserves IDs; quorum is based on full results.
    complete = (i, stop) -> (_completion_fixture(i), true)
    now[] = 1.0
    WarmupHMC._completion_worker!(complete, s, 3)
    @test isnothing(s.quorum_at)
    # Chain 1 is running when chain 4 reaches quorum. Its final round can
    # finish arbitrarily later: completion is still admitted, without a timer.
    WarmupHMC._completion_worker!(s, 1) do i, stop
        @test !stop()
        now[] = 4.0
        WarmupHMC._completion_worker!(complete, s, 4)
        @test s.quorum_at == 4.0
        now[] = 600.0
        @test stop()
        _completion_fixture(i), true
    end
    @test s.statuses[1] === :completed
    @test s.finished_at[1] == 600.0
    @test s.quorum_at == 4.0
    WarmupHMC._completion_worker!(complete, s, 2)
    @test s.statuses[2] === :not_started
    WarmupHMC._completion_worker!(complete, s, 5)
    @test s.statuses[5] === :not_started
    @test findall(==(:completed), s.statuses) == [1, 3, 4]
end

@testset "heterogeneous workers settle before return" begin
    entered = Channel{Int}(3)
    release = [Channel{Nothing}(1) for _ in 1:3]
    cleaned = Threads.Atomic{Int}(0)
    slow_finished = Threads.Atomic{Bool}(false)
    batch = Threads.@spawn WarmupHMC._completion_batch(3; min_completed=1) do i, stop
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
            if i == 3
                slow_finished[] = true
                return _completion_fixture(i), true
            end
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
    @test out.completion.completed_chain_indices == [2, 3]
    @test out.completion.omitted_chain_indices == [1]
    @test out.completion.n_started == 3
    @test getproperty.(out.results, :chain_index) == [2, 3]
    @test out.completion.n_samples == 10
    @test out.completion.n_divergent_samples == 1
    @test out.completion.stop_reason === :quorum_reached
    @test out.completion.stop_policy === :finish_current_round
    @test out.completion.stop_requested_at_seconds == out.completion.quorum_at_seconds
    @test !haskey(out.completion, :grace_seconds)
    @test !haskey(out.completion, :cutoff_at_seconds)
    @test all(c -> c.n_retained_samples == (c.chain_index in (2, 3) ? 5 : 0), out.completion.chains)
end

@testset "final rounds admit extra workers, failures stay explicit" begin
    entered = Channel{Int}(4)
    release = Channel{Nothing}(4)
    batch = Threads.@spawn WarmupHMC._completion_batch(4; min_completed=2) do i, stop
        put!(entered, i)
        take!(release)
        i == 2 && error("controlled chain 2 failure")
        _completion_fixture(i), true
    end
    @test sort([take!(entered) for _ in 1:4]) == [1, 2, 3, 4]
    foreach(_ -> put!(release, nothing), 1:4)
    out = fetch(batch)
    @test out.completion.completed_chain_indices == [1, 3, 4]
    @test out.completion.failed_chain_indices == [2]
    @test out.completion.omitted_chain_indices == [2]
    @test out.completion.n_completed == 3
    @test out.completion.n_failed == 1
    @test isnothing(out.completion.chains[2].n_samples)
    @test out.completion.chains[2].n_retained_samples == 0
    @test out.completion.stop_reason === :quorum_reached
    @test occursin("controlled chain 2 failure", sprint(showerror, out.completion.chains[2].error))
    @test out.completion.n_samples == sum(r -> size(r.posterior_position, 2), out.results)
    @test out.completion.n_divergent_samples == sum(r -> r.n_divergent_samples, out.results)

    unmet = WarmupHMC._completion_batch(2; min_completed=2) do i, stop
        i == 1 && error("failed initialization")
        _completion_fixture(i; n=0), false
    end
    @test isempty(unmet.results)
    @test !unmet.completion.quorum_met
    @test unmet.completion.stop_reason === :quorum_unmet
    @test occursin("0 of 2", sprint(showerror, WarmupHMC.CompletionQuorumError(unmet)))
end

@testset "control characters in failure JSON" begin
    @test WarmupHMC._json_val("tab\tnewline\nreturn\rnull\0quote\"slash\\") ==
        "\"tab\\u0009newline\\nreturn\\u000dnull\\u0000quote\\\"slash\\\\\""
end

@testset "already-complete admission precedes scheduling" begin
    initial = Dict(3 => _completion_fixture(3), 1 => _completion_fixture(1))
    out = WarmupHMC._completion_batch(4; min_completed=1,
        initial_results=initial) do i, stop
        error("No incomplete worker should start after quorum.")
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

@testset "mode-dependent runtime selection is not inferential validation" begin
    # A symmetric two-mode density has half its mass on each side. These real
    # adaptive draws stay in their well-separated initialized modes. Artificial
    # round delays below are conditioned on that mode: this is a controlled
    # selection-bias counterexample, not a timing benchmark or a claim of mixing.
    p = CompletionBimodal()
    inits = [(; position=[x], squared_scale=Diagonal(ones(1))) for x in (-7.5, 7.5, -8.5, 8.5)]
    full = adaptive_warmup_mcmc([Xoshiro(i+100) for i in 1:4], p;
        init=inits, n_draws=100, n_evaluations=100, parallel=false)
    @test all(r -> size(r.posterior_position, 2) == 100, full)
    @test [mean(r.posterior_position) > 0 for r in full] == [false, true, false, true]
    @test mean(hcat(getproperty.(full, :posterior_position)...) .> 0) == 0.5
    selected = WarmupHMC._completion_batch(4; min_completed=2) do i, stop
        if mean(full[i].posterior_position) < 0
            while !stop()
                yield()
            end
            # This mode's current round reaches only half the draw target;
            # a full result would have to be included even after quorum.
            return merge(full[i], (; posterior_position=full[i].posterior_position[:, 1:50],
                posterior_gradient=full[i].posterior_gradient[:, 1:50])), false
        end
        full[i], true
    end
    @test selected.completion.completed_chain_indices == [2, 4]
    @test selected.completion.omitted_chain_indices == [1, 3]
    @test mean(hcat(getproperty.(selected.results, :posterior_position)...) .> 0) == 1.0
    @test selected.completion.n_samples == 200
    @test occursin("bias inference", selected.completion.selection_warning)
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
        # Legacy run manifests wrap the same adaptive checkpoint payloads.
        # An unfinished legacy run switches to the current round-stop policy.
        WarmupHMC._atomic_serialize_manifest(joinpath(dir, "completion_manifest.jls"),
            (; schema_version=1, sampler=:completion, n_requested=2))
        # Complete chain 2 separately. Resume must admit it before any worker
        # has a chance to initialize chain 1.
        ordinary = adaptive_warmup_mcmc(Xoshiro(999), p;
            n_draws=40, checkpoint_dir=joinpath(dir, "chain_2"), resume=true)
        out = completion_warmup_mcmc([Xoshiro(999), Xoshiro(999)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true)
        @test out.completion.completed_chain_indices == [2]
        @test out.completion.omitted_chain_indices == [1]
        @test out.results[1].posterior_position == ordinary.posterior_position
        @test out.completion.chains[1].status === :not_started
        @test out.completion.schema_version == 2
        @test out.completion.stop_policy === :finish_current_round
        cp1 = read(joinpath(dir, "chain_1", "cp_latest.jls"))
        @test isfile(joinpath(dir, "completion_terminal.jls"))
        @test isfile(joinpath(out.completion.attempt_directory, "run_summary.json"))
        @test occursin("\"completed_chain_indices\":[2]",
            read(joinpath(out.completion.attempt_directory, "run_summary.json"), String))
        @test occursin("\"stop_policy\":\"finish_current_round\"",
            read(joinpath(out.completion.attempt_directory, "run_summary.json"), String))
        @test !occursin("grace_seconds",
            read(joinpath(out.completion.attempt_directory, "run_manifest.json"), String))
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
        # A crash between terminal serialization and JSON publication is
        # recoverable from the immutable record, without sampling again.
        summary = joinpath(out.completion.attempt_directory, "run_summary.json")
        original_summary = read(summary)
        rm(summary)
        recovered = completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true)
        @test recovered.completion == out.completion
        @test read(summary) == original_summary
        # A schema-1 terminal fixture retains the OLD selection semantics,
        # including a full chain omitted after cutoff. It must not be silently
        # reinterpreted under the new policy when reopened.
        legacy_fields = (; (k => v for (k, v) in pairs(out.completion)
            if k ∉ (:schema_version, :stop_policy, :stop_requested_at_seconds))...)
        legacy_chains = copy(out.completion.chains)
        legacy_chains[1] = merge(legacy_chains[1], (; status=:completed_after_cutoff,
            finished_at_seconds=4.0, n_samples=40))
        legacy = (; out.results, completion=merge(legacy_fields, (; grace_seconds=3.0,
            cutoff_at_seconds=3.0, quorum_at_seconds=0.0, elapsed_seconds=4.0,
            started_chain_indices=[1, 2], n_started=2, stop_reason=:grace_expired,
            chains=legacy_chains)))
        WarmupHMC._atomic_serialize_manifest(joinpath(dir, "completion_terminal.jls"),
            (; schema_version=1, n_draws=40, outcome=legacy))
        rm(summary)
        old = completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true,
            callback=(s, stage) -> error("Legacy terminal reopen must not sample"))
        @test old.completion == legacy.completion
        @test getproperty.(old.results, :chain_index) == [2]
        @test old.results[1].posterior_position == out.results[1].posterior_position
        @test occursin("\"grace_seconds\":3.0", read(summary, String))
        # Explicit overwrite starts a new run and clears the old selection.
        fresh = completion_warmup_mcmc([Xoshiro(21), Xoshiro(22)], p;
            n_draws=40, checkpoint_dir=dir, overwrite=true)
        @test fresh.completion.completed_chain_indices == [1, 2]
        @test fresh.completion.n_samples == 80
    end
end

@testset "real chain failures are terminal and do not satisfy quorum" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    init = (; position=[0.1, 0.2], squared_scale=Matrix{Float64}(I, 2, 2))
    mktempdir() do dir
        # NamedTuple initialization avoids the initializer's intentional retry
        # loop: this density throws at the actual strict gradient evaluation.
        err = try
            completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], [NaNProblem(2), p];
                n_draws=40, checkpoint_dir=dir, init)
        catch e
            e
        end
        @test err isa WarmupHMC.CompletionQuorumError
        @test err.outcome.completion.completed_chain_indices == [2]
        @test err.outcome.completion.failed_chain_indices == [1]
        @test err.outcome.completion.n_samples == 40
        @test err.outcome.completion.n_completed == 1
        @test err.outcome.completion.n_failed == 1
        @test !isnothing(err.outcome.completion.chains[1].error)
        @test isfile(joinpath(err.outcome.completion.attempt_directory, "run_summary.json"))
        @test !isfile(joinpath(dir, "completion_terminal.jls"))
        # The slot that failed before any checkpoint can initialize on resume;
        # the already-complete second slot is restored without new transitions.
        recovered = completion_warmup_mcmc([Xoshiro(1), Xoshiro(999)], p;
            n_draws=40, checkpoint_dir=dir, init, resume=true)
        @test recovered.completion.completed_chain_indices == [1, 2]
        @test recovered.results[2].posterior_position == err.outcome.results[1].posterior_position
    end
end

@testset "public policy validation precedes work" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    for value in (0, 3, 1.5, true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p; min_completed=value)
    end
    for value in (0.0, 30.0)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; grace_seconds=value)
    end
    @test :grace_seconds ∉ WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
    for value in (0, -1, typemax(Int), true, 1.5)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; n_draws=value)
    end
    @test_throws ArgumentError completion_warmup_mcmc(Xoshiro[], p)
    @test_throws DimensionMismatch completion_warmup_mcmc([Xoshiro(1)], [p, p])
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; resume=true)
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; n_draw=5)
end
