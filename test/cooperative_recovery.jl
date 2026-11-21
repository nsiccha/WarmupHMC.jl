@testitem "cooperative chain rescue, replacement and ESS membership" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics, Serialization, LogDensityProblems
    using MCMCDiagnosticTools, JSON, DifferentiationInterface, Enzyme
    const W = WarmupHMC

    struct SlotTargets
        targets::Vector{DiagGaussian}
    end
    W._coop_target(f::SlotTargets, i) = f.targets[i]
    struct RandomStarts end
    W._coop_cold_init(::RandomStarts, i, rng) = (; position=[10i + rand(rng)], squared_scale=[1.0])

    function fixture(; targets=DiagGaussian([0.0], [2.0]), init=nothing, master=nothing)
        starts = isnothing(init) ? [(; position=[x], squared_scale=[1.0]) for x in (0.0, 0.0, 20.0)] : init
        cfg = (; family=false, max_reflections=0, nonlinear_settings=(;), n_chains=3, n_cores=1, n_evaluations=8, recording_target=100,
            initializer_kw=(;), target_acceptance_rate=0.8, monitor_ess=true, n_draws=80,
            target_ess=Inf, rhat_threshold=1.01, n_evaluations_budget=typemax(Int), time_budget=Inf,
            max_clusters=8, bandwidth=1.0, restart_threshold=Inf, kappa_decay=0.9,
            kappa_prior_weight=2.0, stepsize_adaptation_limit=4, algorithm=W.DynamicHMC.NUTS(; max_depth=6))
        run = W.CoopRun(Xoshiro(1234), targets, cfg, starts, nothing, "recovery test",
            time_ns(), 0.0, master, W.CoopChain[], W.CoopClusters(), Threads.Atomic{Int}(0),
            1, 1, nothing, NaN)
        W._create_chains!(run, 1:3)
        for c in run.chains
            install_evidence!(c)
        end
        W._update_clusters!(run.clusters, run.chains, cfg)
        run
    end

    # Controlled observations at the batch boundary. All decisions, state
    # changes, serialization, subsequent transitions and reported results are
    # made by the production sampler.
    function install_evidence!(c; divergent=0, transitions=32)
        c.current_transition_counter = transitions
        c.total_transition_counter += transitions
        c.total_evaluation_counter += transitions
        c.n_divergent = divergent
        c.n_divergent_samples = min(divergent, 20)
        c.status = :sampling
        c.stepsize = 0.5
        c.log_kappa = W._kappa_log(c.stepsize, 0.8)
        c.n_samples = 20
        c.ess .= 16
        for dx in repeat([-1.0, 1.0], 10)
            x = c.position_and_gradient.q .+ dx
            g = -x ./ 4
            W._fit_moments!(c.moments, x, g)
            append!(c.recording_lpdf.posterior_position, x)
            append!(c.recording_lpdf.posterior_gradient, g)
        end
    end

    function first_rescue!(run, divergent)
        c = last(run.chains)
        c.n_divergent = c.n_divergent_samples = divergent
        point, rng = c.position_and_gradient, copy(c.rng)
        W._batch_end!(run)
        @test last(c.batches).recovery === :rescue
        @test c.position_and_gradient === point && c.rng == rng
        @test c.generation == 1 && isempty(c.abandoned)
        @test diag(c.scale) ≈ [2.0]
        @test c.n_samples == W.n_chain_draws(c) == 0
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        W._update_kappa_priors!(run.clusters, run.chains)
        @test c.kappa_prior == c.rescue_prior ≈ W._kappa_log(0.5, 0.8)
        @test W._awaiting_recovery_draws(c)
        c
    end

    @testset "a recoverable singleton keeps its position and its generation" begin
        run = fixture()
        c = first_rescue!(run, 2)
        install_evidence!(c)
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        W._batch_end!(run)
        @test last(c.batches).recovery === :none
        @test c === last(run.chains) && c.n_samples == 20
        # A poorly sampled singleton can also qualify through per-draw ESS.
        run = fixture()
        last(run.chains).ess .= 0.5
        W._batch_end!(run)
        @test last(last(run.chains).batches).recovery_reason === :low_ess_per_draw
    end

    @testset "step-size evidence and locally efficient narrow modes" begin
        run = fixture()
        c = last(run.chains)
        c.stepsize = 0.001
        context = W._coop_recovery_context(run, run.lpdf)
        @test W._coop_recovery_action(context, c, 1).kind === :none
        c.ess .= 0.5
        W._batch_end!(run)
        @test last(c.batches).recovery_reason === :stepsize
        @test c === last(run.chains)
        install_evidence!(c)
        c.stepsize = 0.001
        c.ess .= 0.5
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        W._batch_end!(run)
        @test last(run.chains).generation == 2
        @test only(last(run.chains).abandoned).reason === :stepsize
    end

    @testset "a failed gate retries its own target once using the seed RNG" begin
        run = fixture(; targets=SlotTargets([DiagGaussian([10.0i], [2.0]) for i in 1:3]),
            init=RandomStarts())
        old = run.chains[2]
        old_rng = copy(old.rng)
        seed_rng = copy(run.rng)
        stream = Xoshiro(rand(seed_rng, UInt64))
        init = W._coop_cold_init(run.init, 2, stream)
        expected = W._coop_chain(2, stream, run.lpdf.targets[2], init, run.cfg; progress=nothing)
        old.status = :needs_replacement
        @test !W._running(old)
        W._replace_gate_failures!(run)
        c = run.chains[2]
        @test c.chain_index == 2 && c.generation == 2 && c.lpdf.mu == [20.0]
        @test c.position_and_gradient.q == expected.position_and_gradient.q != old.position_and_gradient.q
        @test c.rng == expected.rng && c.rng != old_rng && run.rng == seed_rng
        @test c.n_samples == 0 && c.moments.weight == 0 && isempty(W.chain_draws(c))
        @test c.total_evaluation_counter == old.total_evaluation_counter
        @test only(c.abandoned).n_samples == 20
        c.status = :needs_replacement
        W._replace_gate_failures!(run)
        @test W._check_stop!(run)
        out = W._finalize(run)
        @test run.rng == seed_rng && out.stop_reason === :recovery_failed
        @test out.results[2].status === :recovery_failed && isempty(out.results[2].posterior_position)
        @test [a.generation for a in out.results[2].abandoned] == [1, 2]
        @test all(a -> a.reason === :first_draw_gate, out.results[2].abandoned)
        @test 2 ∉ out.pooled_chains
    end

    @testset "an exhausted first-draw gate terminates a draw-count-only run" begin
        run = fixture()
        old = run.chains[1]
        old.status = :needs_replacement
        W._replace_gate_failures!(run)
        empty_slot = run.chains[1]
        empty_slot.status = :needs_replacement
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        dir = mktempdir()
        W._write_checkpoints!(run, dir; batch_end=false)
        state = deserialize(joinpath(dir, "run_latest.jls"))
        @test state.schema_version == state.chains[1].schema_version == 3
        @test state.chains[2].schema_version == 2
        out = cooperative_warmup_mcmc(Xoshiro(7), run.lpdf; n_chains=3, n_draws=1,
            checkpoint_dir=dir, resume=true)
        @test out.stop_reason === :recovery_failed
        @test out.results[1].n_samples == 0 && length(out.results[1].abandoned) == 2
        @test JSON.parsefile(joinpath(dir, "run_summary.json"))["failed_chains"] == [1]
        failed = deserialize(joinpath(dir, "chain_1", "cp_latest.jls"))
        @test failed.is_final && failed.stuck_reason === :first_draw_gate
        @test failed.status === failed.stop_reason === :recovery_failed
        @test failed.schema_version == 3 < W.checkpoint_schema_version(Val(:cooperative))
        @test isnothing(W._check_schema_version(failed))
        @test_throws ArgumentError W._check_schema_version(merge(failed, (; schema_version=5)))
        @test isempty(failed.posterior_position)
    end

    @testset "a pre-recovery chain checkpoint begins in generation one" begin
        run = fixture()
        p = W._coop_payload(first(run.chains), run)
        @test p.schema_version == 2
        old = Base.structdiff(p, NamedTuple{(:abandoned, :generation, :rescue_prior)})
        c = W._restore_coop_chain(old, deepcopy(run.lpdf), run.cfg; progress=nothing, adapt_centering=false, geometry_current=true)
        @test c.generation == 1 && isempty(c.abandoned) && isnothing(c.rescue_prior)
        @test W.chain_draws(c) == p.posterior_position
        opaque = W._checkpoint_cold_init(Any[RandomStarts(), [1.0]])
        @test_throws ArgumentError W._coop_cold_init(opaque, 1, Xoshiro(1))
        @test W._coop_cold_init(opaque, 2, Xoshiro(1)) == [1.0]
    end

    @testset "a replacement's cold start enters the current centering frame" begin
        target = ReparametrizedProblem(IndexedReparametrization([
            1 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0), 10.0, 0.0)
        ]), DiagGaussian([0.0], [2.0]), AutoEnzyme())
        master = deepcopy(target)
        run = fixture(; targets=target, master)
        W.restore_reparam_sources!(master, [1 => PartiallyCentered(0.0)])
        foreach(c -> (c.geometry_current = false), run.chains)
        # None of the previous-frame moments are eligible donor evidence.
        @test isnothing(W._coop_recovery_context(run, run.lpdf))
        run.chains[3].status = :needs_replacement
        W._replace_gate_failures!(run)
        c = run.chains[3]
        @test W.reparam_sources(c.lpdf) == W.reparam_sources(master) != W.reparam_sources(target)
        q = c.position_and_gradient.q
        @test q != last(run.init).position
        @test last(W.reparametrizer(c.lpdf)(q)) ≈ last(run.init).position
        @test c.moments.weight == 0 && c.n_samples == 0

        # A caller can resume with adaptation disabled while the checkpoint
        # still contains a different frame for an individual slot.
        frozen = fixture(; targets=target)
        old = frozen.chains[3]
        W.restore_reparam_sources!(old.lpdf, W.reparam_sources(master))
        old.geometry_current = false
        old.n_divergent = old.n_divergent_samples = 18
        context = W._coop_recovery_context(frozen, target)
        @test W._coop_recovery_action(context, old, 1).kind === :none
        old.status = :needs_replacement
        W._replace_gate_failures!(frozen)
        @test W.reparam_sources(frozen.chains[3].lpdf) == W.reparam_sources(master)
        @test !W._coop_geometry_current(frozen.chains[3])
    end

    @testset "public resume on either side of an extreme replacement" begin
        run = fixture()
        c = first_rescue!(run, 16)
        install_evidence!(c; divergent=96, transitions=128)
        run.round = 2
        run.total_evaluations[] = sum(c -> c.total_evaluation_counter, run.chains)
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        before, after = mktempdir(), mktempdir()
        W._write_checkpoints!(run, before; batch_end=false)
        # Execute barrier three once and checkpoint immediately after the swap.
        run.round += 1
        W._run_round!(run)
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        @test !W._check_stop!(run)
        W._batch_end!(run)
        fresh = last(run.chains)
        @test fresh.generation == 2 && fresh.chain_index == 3
        @test only(fresh.abandoned).reason === :divergences
        @test fresh.total_evaluation_counter == c.total_evaluation_counter
        @test isempty(W.chain_draws(fresh)) && fresh.moments.weight == 0
        W._update_clusters!(run.clusters, run.chains, run.cfg)
        W._update_kappa_priors!(run.clusters, run.chains)
        W._write_checkpoints!(run, after; batch_end=true)
        cp = deserialize(joinpath(after, "chain_3", "cp_window_2.jls"))
        @test cp.generation == 2 && cp.n_samples == 0 && isempty(cp.dropped_posterior_position)
        @test cp.rng == fresh.rng && only(cp.abandoned).generation == 1
        @test only(cp.abandoned).generation_divergences == 16 + only(cp.abandoned).n_divergent
        @test only(cp.abandoned).generation_evaluations == c.total_evaluation_counter
        resume(dir, cores) = cooperative_warmup_mcmc(Xoshiro(999), run.lpdf; resume=true,
            checkpoint_dir=dir, n_chains=3, n_cores=cores, n_draws=80, n_evaluations=8,
            stepsize_adaptation_limit=4, max_tree_depth=6, restart_threshold=Inf,
            recording_target=100)
        a, b = resume(before, 1), resume(after, 3)
        @test a.stop_reason === b.stop_reason === :n_draws
        @test a.total_evaluation_counter == b.total_evaluation_counter == sum(r -> r.total_evaluation_counter, b.results)
        @test a.joint_ess == b.joint_ess && a.pooled_chains == b.pooled_chains
        for (x, y) in zip(a.results, b.results)
            @test x.posterior_position == y.posterior_position
            @test x.posterior_gradient == y.posterior_gradient
            @test x.generation == y.generation && isequal(x.abandoned, y.abandoned)
        end
        @test [r.generation for r in b.results] == [1, 1, 2]
        summary = JSON.parsefile(joinpath(after, "run_summary.json"))
        @test summary["pooled_chains"] == b.pooled_chains
        @test summary["chain_generations"] == [1, 1, 2]
    end

    @testset "ESS-ranked prefixes retain identities and the agreement veto" begin
        rng = Xoshiro(42)
        slow = repeat(randn(rng, 1, 30); inner=(1, 200)) .+ 30
        draws = [slow, randn(rng, 1, 1500), randn(rng, 1, 1200)]
        e, which = W._best_pooled_ess(draws, 2)
        @test sort(which) == [2, 3]
        @test e ≈ only(ess(W._stack_chain_tails(draws[which]; min_chain_draws=11)))
        @test W._retained_rhat(draws) > 1.1
        @test W._best_pooled_ess([zeros(1, 5)], 1) == (0.0, Int[])
    end

    struct DoubleWell end
    LogDensityProblems.dimension(::DoubleWell) = 1
    LogDensityProblems.capabilities(::Type{DoubleWell}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(::DoubleWell, q)
        z = (only(q)^2 - 25^2) / 50
        (-z^2 / 2, -z .* q ./ 25)
    end
    LogDensityProblems.logdensity(p::DoubleWell, q) = first(LogDensityProblems.logdensity_and_gradient(p, q))

    @testset "a genuine minority mode remains visible to R-hat" begin
        init = [(; position=[x], squared_scale=[1.0]) for x in (-25.0, -25.0, 25.0)]
        out = cooperative_warmup_mcmc(Xoshiro(27), DoubleWell(); init, n_chains=3,
            target_ess=50, n_draws=300, n_evaluations=100, n_evaluations_budget=100_000)
        @test out.stop_reason === :n_draws && out.n_used == 3
        draws = [r.posterior_position for r in out.results]
        @test all(<(0), draws[1]) && all(<(0), draws[2]) && all(>(0), draws[3])
        @test W._retained_rhat(draws) > 1.1
        @test all(r -> r.generation == 1 && isempty(r.abandoned), out.results)
    end
end

@testitem "cooperative recovery from a narrow support pocket" begin
    using Test, WarmupHMC, LogDensityProblems, Random, Serialization

    # A broad Student mode and a tiny separate interval. The interval has a
    # finite density and gradient, but most trajectories hit a support boundary.
    struct Pocket
        log_height::Float64
    end
    LogDensityProblems.dimension(::Pocket) = 1
    LogDensityProblems.capabilities(::Type{Pocket}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(p::Pocket, q)
        x = only(q)
        if x < 0
            z = x + 20
            return -3.5log1p(z^2 / 6), [-7z / (6 + z^2)]
        elseif abs(x - 20) <= 0.005
            return p.log_height - (x - 20)^2 / 2, [20 - x]
        end
        -Inf, [NaN]
    end
    LogDensityProblems.logdensity(p::Pocket, q) = first(LogDensityProblems.logdensity_and_gradient(p, q))
    struct PocketStart end
    WarmupHMC.initialize_mcmc(::Pocket, ::PocketStart; rng, kwargs...) =
        (; position=[rand(rng, Bool) ? 20.0 : -20.0], squared_scale=ones(1, 1))
    broad = (; position=[-20.0], squared_scale=[1.0])
    sample(third; log_height=-1.0, kw...) = cooperative_warmup_mcmc(Xoshiro(27), Pocket(log_height); n_chains=3, n_cores=1,
        init=Any[broad, broad, third], n_evaluations=64, stepsize_adaptation_limit=12,
        max_tree_depth=5, n_draws=1000, n_evaluations_budget=50_000, restart_threshold=Inf, kw...)

    # A comparable density isolates the batch divergence rescue. A lower
    # pocket can now be caught earlier by the first-draw qualification.
    low = sample((; position=[20.0], squared_scale=[1.0]); log_height=-10.)
    @test low.stop_reason === :recovery_failed
    @test low.results[3].n_samples == 0 && length(low.results[3].abandoned) == 2
    @test low.results[3].accounting.metric_reviews > 0
    healthy = sample(broad)
    @test healthy.stop_reason === :n_draws
    @test all(c -> c.generation == 1 && isempty(c.abandoned), healthy.results)
    failed = sample((; position=[20.0], squared_scale=[1.0]))
    @test failed.stop_reason === :recovery_failed
    @test failed.results[3].n_samples == 0 && 3 ∉ failed.pooled_chains
    @test Set(a.reason for a in failed.results[3].abandoned) == Set((:divergences, :first_draw_gate))
    @test failed.results[3].accounting.freeze_extensions >= 30

    # For this seed, the third slot starts in the pocket; its fresh stream
    # chooses the broad region. The replay uses real NUTS transitions throughout.
    full = sample(PocketStart())
    third = full.results[3]
    @test full.stop_reason === :n_draws && third.generation == 2
    @test third.n_samples == 1000 && all(<(0), third.posterior_position)
    abandoned = only(third.abandoned)
    @test abandoned.reason === :divergences
    @test abandoned.batches[end-1].recovery === :rescue && last(abandoned.batches).extreme

    for (k, generation) in ((abandoned.batch - 1, 1), (abandoned.batch, 2))
        work = sum(full.results) do c
            history = c.chain_index == 3 ? abandoned.batches : c.batches
            only(b.total_evaluation_counter for b in history if b.batch == k)
        end
        dir = mktempdir()
        partial = sample(PocketStart(); checkpoint_dir=dir, n_evaluations_budget=work + 1)
        @test partial.stop_reason === :eval_budget
        state = deserialize(joinpath(dir, "run_latest.jls"))
        @test state.chains[3].generation == generation && state.total_evaluations == work
        rm(joinpath(dir, "run_summary.json")) # simulate a crash after the saved complete barrier
        resumed = sample(PocketStart(); resume=true, checkpoint_dir=dir, n_cores=3)
        @test resumed.stop_reason == full.stop_reason
        @test resumed.total_evaluation_counter == full.total_evaluation_counter
        @test resumed.pooled_chains == full.pooled_chains && resumed.joint_ess == full.joint_ess
        for (x, y) in zip(resumed.results, full.results)
            @test x.posterior_position == y.posterior_position && x.posterior_gradient == y.posterior_gradient
            @test x.generation == y.generation && isequal(x.abandoned, y.abandoned)
        end
    end
end
