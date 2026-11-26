@testitem "quorum records account for every worker's model-gradient calls" begin
    using WarmupHMC, LogDensityProblems, Random, Serialization, JSON

    struct CountedSlot
        index::Int
        calls::Base.RefValue{Int}
        fail_at::Base.RefValue{Int}
    end
    CountedSlot(i) = CountedSlot(i, Ref(0), Ref(typemax(Int)))
    # Deepcopy this stateless model with the same external observation handles.
    # Each original slot is still the sole writer to its own counters.
    Base.deepcopy_internal(p::CountedSlot, ::IdDict) = p
    LogDensityProblems.dimension(::CountedSlot) = 2
    LogDensityProblems.capabilities(::Type{CountedSlot}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::CountedSlot, x) = -sum(abs2, x) / 2
    function LogDensityProblems.logdensity_and_gradient(p::CountedSlot, x)
        p.calls[] += 1
        p.calls[] == p.fail_at[] && error("gradient failure in slot $(p.index)")
        LogDensityProblems.logdensity(p, x), -x
    end
    init = (; position=[0.5, 0.0], squared_scale=ones(2))
    settings = (; n_draws=8, stepsize_adaptation_limit=2, n_evaluations=1000,
        nonlinear_adapt=false, variance_cond_target=Inf, init)
    function unmet(f)
        try
            f()
        catch err
            err isa WarmupHMC.CompletionQuorumError || rethrow()
            return err.outcome
        end
        error("expected unmet quorum")
    end

    mktempdir() do dir
        targets = CountedSlot.(1:3)
        targets[2].fail_at[] = 1
        init_calls = zeros(Int, 3)
        function on_boundary(state, stage)
            i = state.lpdf.index
            if stage === :init
                init_calls[i] = targets[i].calls[]
                i == 3 && (targets[i].fail_at[] = targets[i].calls[] + 1)
            end
            false
        end
        out = unmet() do
            completion_warmup_mcmc(Xoshiro.(1:3), targets;
                settings..., checkpoint_dir=dir, callback=on_boundary)
        end
        c = out.completion
        calls = [p.calls[] for p in targets]
        @test c.completed_chain_indices == [1] && c.failed_chain_indices == [2, 3]
        @test getproperty.(c.chains, :gradient_calls) == calls
        @test c.gradient_calls == c.attempt_gradient_calls == sum(calls)
        @test c.initialization_gradient_calls == init_calls[1] + 1 + init_calls[3]
        @test c.chains[2].gradient_calls == c.chains[2].initialization_gradient_calls == 1
        @test c.chains[2].n_warmup_windows == 0
        @test c.chains[3].gradient_calls == init_calls[3] + 1
        @test c.chains[3].n_warmup_windows == 1
        @test c.chains[3].total_evaluation_counter == c.chains[3].sampling_gradient_calls == 0
        @test 0 < c.sampling_gradient_calls < c.gradient_calls
        @test c.sampling_evaluation_counter == only(out.results).sampling_evaluation_counter
        @test c.total_evaluation_counter == only(out.results).total_evaluation_counter
        summary = JSON.parsefile(joinpath(c.attempt_directory, "run_summary.json"))
        @test summary["gradient_calls"] == sum(calls)
        @test summary["chain_gradient_calls"] == calls
        @test summary["chain_n_warmup_windows"] == getproperty.(c.chains, :n_warmup_windows)
    end

    mktempdir() do dir
        # Stop on a transition boundary inside the first window. Its calls
        # and window count carry into the resumed invocation exactly once.
        p = CountedSlot(1)
        stop_at = Ref(typemax(Int))
        partial = unmet() do
            completion_warmup_mcmc([Xoshiro(17)], p; settings..., checkpoint_dir=dir,
                callback=(s, stage) -> (stage === :init && (stop_at[] = p.calls[]+1); false),
                stop_requested=() -> p.calls[] >= stop_at[])
        end
        prefix = partial.completion
        @test only(prefix.chains).status === :stopped
        @test prefix.gradient_calls == p.calls[] > prefix.initialization_gradient_calls > 0
        @test prefix.n_warmup_windows == 1
        cp = deserialize(joinpath(dir, "chain_1", "cp_latest.jls"))
        @test cp.window_in_progress && cp.gradient_calls == prefix.gradient_calls
        out = completion_warmup_mcmc([Xoshiro(99)], p; settings..., checkpoint_dir=dir, resume=true)
        c = out.completion
        @test c.gradient_calls == p.calls[] == prefix.gradient_calls + c.attempt_gradient_calls
        @test c.initialization_gradient_calls == prefix.initialization_gradient_calls
        @test c.n_warmup_windows == deserialize(joinpath(dir, "chain_1", "cp_latest.jls")).outer_counter
        @test c.sampling_gradient_calls == c.sampling_evaluation_counter > 0
        reopened = completion_warmup_mcmc([Xoshiro(1)], p; settings..., checkpoint_dir=dir, resume=true)
        @test reopened.completion == c
        @test p.calls[] == c.gradient_calls
    end
end
