@testitem "Streamed centering matches buffered NUTS proposal groups" begin
    using WarmupHMC, LogDensityProblems
    const W = WarmupHMC
    struct Uncalled end
    LogDensityProblems.dimension(::Uncalled) = 3
    ir = IndexedReparametrization([i => Reparametrization(
        PartiallyCentered(1.0), PartiallyCentered(0.4), 0.0, x -> x[1]/2) for i in 2:3])
    prepares = Ref(0)
    plan = CandidateScoringPlan((ir, q, g) -> (prepares[] += 1; (q, g)),
        (frame, k, idx, rule, c) -> k == 1 ? nothing :
            (0.1c.c, frame[1][idx] + c.c, frame[2][idx] - c.c))
    p = ReparametrizedProblem(ir, Uncalled(); scoring_plan=plan)
    for scorer in (p, ReparametrizedProblem(ir, Uncalled())), weighting in (:unit, :stepsize)
        recorder() = W.NonlinearRecorder(scorer; mode=:nuts_weighted, trajectory_weighting=weighting)
        online, oracle = recorder(), recorder()
        stream = W._centering_leaves(scorer, online, :halo, :online, 3, nothing, true)
        @test isnothing(stream.linear)
        raw = W.NUTSLeaves(3)
        # Complete and partial invalid doublings, a root-only transition, zero
        # proposal mass, overflow-prone log energies and a strongly biased switch.
        for (depth, dh) in ((0, [0., 2.]), (1, [0., -3., 8.]),
                            (2, [0., -Inf, -2., -1., 50., 50., 50., 50.]),
                            (3, [0., 2., -Inf, -Inf, 710., 709., 706., 700.]))
            W.reset!(stream)
            W.reset!(raw)
            q, g = zeros(3), zeros(3)
            for (j, delta) in enumerate(dh)
                q .= (sin(j), j/4, j^2/8)
                g .= (cos(j), 1-j, -sqrt(j))
                W.record_leaf!(raw, q, g, delta)
                W._observe_trajectory_leaf!(stream, scorer, (; Q=(; q, ∇ℓq=g)), delta)
                fill!(q, NaN)
                fill!(g, NaN)
            end
            W.finalize_leaf_weights!(raw, depth)
            W.finalize_leaf_weights!(stream, depth)
            W.record_nonlinear!(online, scorer, stream, 0.37)
            W.record_nonlinear!(oracle, scorer, raw, 0.37)
            for ((_, a), (_, b)) in zip(online.online.pairs, oracle.online.pairs)
                for ((_, x), (_, y)) in zip(a.pairs, b.pairs)
                    @test x.groups == y.groups
                    @test all(isapprox(getfield(x,f), getfield(y,f); rtol=1e-11, atol=1e-11)
                              for f in fieldnames(typeof(x)) if f != :groups)
                end
            end
        end
    end
    @test prepares[] > 0
end

@testitem "Online centering installs between trees and resumes through completion" begin
    using WarmupHMC, Random, Serialization, Enzyme, DifferentiationInterface, LogDensityProblems
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: stop_after, latest, same, comparable
    const W = WarmupHMC
    struct CountingFunnel
        calls::Base.RefValue{Int}
        reject::Base.RefValue{Bool}
    end
    Base.deepcopy_internal(p::CountingFunnel, ::IdDict) = p
    LogDensityProblems.dimension(::CountingFunnel) = 3
    LogDensityProblems.capabilities(::Type{CountingFunnel}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(p::CountingFunnel, y)
        p.calls[] += 1
        p.reject[] && return (-Inf, fill(NaN, 3))
        v, a, b = y
        z2 = (a^2+b^2)*exp(-v)
        (-v^2/2-z2/2-v, [-v+z2/2-1, -a*exp(-v), -b*exp(-v)])
    end
    function problem(; reject_switch=false)
        model = CountingFunnel(Ref(0), Ref(false))
        ir = IndexedReparametrization([i => Reparametrization(
            PartiallyCentered(1.0), PartiallyCentered(1.0), 0.0, x -> x[1]/2) for i in 2:3])
        # The child's preferred candidate changes when its parent's source moves.
        plan = CandidateScoringPlan(
            (ir, q, g) -> (last(ir.pairs[1]).source.c, q),
            (frame, k, idx, rule, c) -> begin
                x = frame[2][idx]
                winner = k == 1 ? 0.0 : frame[1]
                (0., x, c.c == winner ? -x : x)
            end;
            synchronize! = ir -> (model.reject[] = reject_switch && last(ir.pairs[1]).source.c == 0))
        ReparametrizedProblem(ir, model, AutoEnzyme(); scoring_plan=plan)
    end
    cfg = (; n_draws=20, n_evaluations=10000, stepsize_adaptation_limit=24, max_tree_depth=5,
        recording_target=30, monitor_ess=false, max_divergent_fraction=1.,
        init=(; position=[0.1, -0.3, 0.5], squared_scale=ones(3)))
    mktempdir() do root
        full, sliced = joinpath(root, "full"), joinpath(root, "sliced")
        p = problem()
        r = completion_warmup_mcmc([Xoshiro(124)], p; cfg..., centering_schedule=:online,
                                   checkpoint_dir=full)
        result, c = only(r.results), only(r.completion.chains)
        @test length(result.centering_installations) == 2
        first_change, second_change = result.centering_installations
        @test first_change >= 5 && second_change >= first_change + 5
        @test second_change <= result.centering_frozen_after == cfg.stepsize_adaptation_limit - 2
        @test c.gradient_calls == p.problem.calls[]
        @test c.sampling_gradient_calls == c.sampling_evaluation_counter > 0
        @test c.gradient_calls == c.initialization_gradient_calls + c.total_evaluation_counter + length(result.centering_installations) + c.n_warmup_windows
        final_cp = latest(joinpath(full, "chain_1"))
        @test last.(final_cp.reparam_sources) == fill(PartiallyCentered(0.), 2)
        @test Base.get_extension(W, :WarmupHMCEnzymeExt) !== nothing
        p2 = problem()
        @test_throws W.CompletionQuorumError completion_warmup_mcmc([Xoshiro(124)], p2;
            cfg..., centering_schedule=:online, checkpoint_dir=sliced, stop_requested=stop_after(4))
        partial = latest(joinpath(sliced, "chain_1"))
        @test partial.window_in_progress && partial.schema_version == 4
        @test maximum(partial.online_centering.streaks) > 0
        @test isempty(partial.online_centering.installations)
        resumed = completion_warmup_mcmc([Xoshiro(999)], p2; cfg..., checkpoint_dir=sliced, resume=true)
        rerun = only(resumed.results)
        @test rerun.posterior_position == result.posterior_position
        @test rerun.posterior_gradient == result.posterior_gradient
        @test rerun.centering_installations == result.centering_installations
        @test same(comparable(latest(joinpath(sliced, "chain_1"))), comparable(final_cp))
        @test only(resumed.completion.chains).gradient_calls == c.gradient_calls
    end

    mktempdir() do dir
        # Retain the full previous frame and its history on validation failure.
        p = problem(; reject_switch=true)
        adaptive_warmup_mcmc(Xoshiro(124), p; cfg..., centering_schedule=:online,
                             checkpoint_dir=dir, stop_requested=stop_after(7))
        cp = latest(dir)
        @test isempty(cp.online_centering.installations) && maximum(cp.online_centering.streaks) >= 3
        @test all(p -> last(p).c == 1, cp.reparam_sources) && !p.problem.reject[]
        @test W.OnlineStatsBase.nobs(cp.nonlinear_recorder.online) == cp.total_transition_counter
        @test cp.position_and_gradient.∇ℓq ≈ last(LogDensityProblems.logdensity_and_gradient(p, cp.position_and_gradient.q))
    end
    r = adaptive_warmup_mcmc(Xoshiro(124), problem(); cfg..., centering_schedule=:online,
        centering_persistence=1, centering_max_updates=1, linear_restart_source=:nuts_weighted)
    @test only(r.centering_installations) == r.centering_frozen_after
    @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), problem(); cfg...,
        centering_schedule=:online, nonlinear_evidence=:all_good_leaves)
end
