@testitem "Cooperative families of conditional targets" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, LogDensityProblems, Random, Statistics, Serialization, JSON
    using DifferentiationInterface, Enzyme
    const W = WarmupHMC

    @testset "conditional coordinates and within-target moments" begin
        p = DiagGaussian([2., 3., 4.], [1., 2., 3.])
        base = [0., 7., 0.]
        f = FixedCoordinates(p, [3, 1], base)
        x = [8., -1.]
        value, g = LogDensityProblems.logdensity_and_gradient(p, [-1., 7., 8.])
        @test LogDensityProblems.dimension(f) == 2
        @test LogDensityProblems.logdensity(f, x) == value
        @test LogDensityProblems.logdensity_and_gradient(f, x) == (value, g[[3,1]])
        @test base == [0., 7., 0.]
        @test_throws ArgumentError FixedCoordinates(p, [3, 3], base)
        @test_throws ArgumentError FixedCoordinates(p, [0], base)
        @test_throws DimensionMismatch FixedCoordinates(p, 1:2, base[1:2])
        @test_throws DimensionMismatch LogDensityProblems.logdensity(f, [8.])
        parts = [W.CoopMoments(1) for _ in 1:2]
        for (part, center) in zip(parts, (-10, 10)), z in (-1., 1.)
            W._fit_moments!(part, [center + z], [-z])
        end
        pool(within_target) = W._combine_moments!(W.CoopMoments(1), parts, (1.,1.); within_target)
        @test only(W._moment_scales(pool(true))) == 1.0
        @test only(W._moment_scales(pool(false))) ≈ 101^0.25
    end

    targets = [DiagGaussian([2.5i, -i/2], [1., 1.2]) for i in 1:9]
    init(i) = (; position=copy(targets[i].mu), squared_scale=targets[i].sigma.^2)
    features = [[Float64(i), 1.] for i in eachindex(targets)]
    kwargs = (; features, n_wave1=3, n_draws=1, n_evaluations=40, stepsize_adaptation_limit=8)
    function sample(seed=11; extra...)
        cooperative_warmup_mcmc(Xoshiro(seed), targets; init, kwargs..., extra...)
    end
    @testset "two waves, one receiving-target draw, original order" begin
        calls = Threads.Atomic{Int}(0)
        a = sample(; n_cores=2, init=(rng,i) -> (Threads.atomic_add!(calls,1); init(i)))
        b = sample(; n_cores=1, init=init.(eachindex(targets)), features=stack(features))
        @test calls[] == kwargs.n_wave1
        @test a.stop_reason === :n_draws
        @test a.joint_ess === nothing && a.n_pooled == 0
        @test getproperty.(a.results, :chain_index) == eachindex(targets)
        @test getproperty.(a.results, :posterior_position) == getproperty.(b.results, :posterior_position)
        @test a.total_evaluation_counter == sum(r.total_evaluation_counter for r in a.results)
        medoids = findall(r -> r.accounting.wave == 1, a.results)
        @test length(medoids) == 3
        for (i, r) in enumerate(a.results)
            @test r.n_samples == 1 && r.status === :done
            @test vec(r.posterior_gradient) ≈ last(LogDensityProblems.logdensity_and_gradient(targets[i], vec(r.posterior_position)))
            if r.accounting.wave == 2
                @test r.accounting.source in medoids
                @test abs(r.accounting.source-i) == minimum(abs(j-i) for j in medoids)
                @test r.accounting.warmup_transitions >= 2
                @test r.accounting.pilot_draws == 0
            else
                @test r.accounting.source == 0 && r.accounting.pilot_draws == 30
            end
        end
        allcold = sample(; features=nothing, n_draws=15)
        @test all(r -> r.n_samples == 15 && r.accounting.source == r.accounting.pilot_draws == 0, allcold.results)
        @test allcold.joint_ess === nothing && allcold.n_pooled == 0
        short = sample(; n_evaluations_budget=1)
        @test short.stop_reason === :eval_budget
        @test count(r -> r.status === :waiting && r.n_samples == 0, short.results) == 6
    end

    @testset "inherit only position and kernel across distinct conditionals" begin
        two = targets[[1,9]]
        family = W.CoopFamily(two, ["a","b"], [0.,1.], 1)
        cfg = (; two_waves=true, n_evaluations=40, n_cores=1, recording_target=100,
            target_acceptance_rate=0.8, monitor_ess=false, initializer_kw=(;),
            n_evaluations_budget=typemax(Int), time_budget=Inf)
        run = W.CoopRun(Xoshiro(16), family, cfg, W.CoopFamilyInit([init(1),init(9)]),
            nothing, "family transfer", time_ns(), 0.0, nothing, W.CoopChain[],
            W.CoopClusters(), Threads.Atomic{Int}(0), 1, 1, nothing, NaN)
        W._create_chains!(run, 1:2)
        donor = only(filter(c -> c.freeze.wave == 1, run.chains))
        receiver = only(filter(c -> c.freeze.wave == 2, run.chains))
        donor.status = :done
        donor.freeze.pilot = [copy(donor.position_and_gradient.q) for _ in 1:30]
        W._start_second_wave!(run)
        pg = receiver.position_and_gradient
        @test pg.q == donor.position_and_gradient.q
        @test pg.q !== donor.position_and_gradient.q
        @test (pg.ℓq, pg.∇ℓq) == LogDensityProblems.logdensity_and_gradient(receiver.lpdf, pg.q)
        @test pg.ℓq < donor.position_and_gradient.ℓq - 10
        @test receiver.status === :warming && receiver.n_samples == 0
        @test receiver.freeze.warmup == 50
        @test receiver.stepsize == donor.stepsize && receiver.log_kappa == donor.log_kappa
        @test receiver.moments.weight == 0
        receiver.freeze.scores = [4., 20.]
        receiver.freeze.total_extensions = receiver.freeze.extensions = 1
        push!(receiver.batches, (; recovery=:rescue))
        replacement = W._coop_chain(receiver.chain_index, Xoshiro(17), receiver.lpdf,
            init(9), cfg; progress=nothing)
        W._coop_replacement_slot!(replacement, receiver, family)
        @test isempty(replacement.freeze.scores)
        @test W.chain_result(replacement).accounting.abandoned_freeze_scores == receiver.freeze.scores
        @test replacement.freeze.extensions == 0 && replacement.freeze.total_extensions == 1
        @test W.chain_result(replacement).accounting.rescues == 1
    end

    @testset "crash resume preserves unfinished waves and rejects a different target order" begin
        mktempdir() do dir
            ids = string.(eachindex(targets))
            partial = sample(; n_cores=1, target_ids=ids, checkpoint_dir=dir, n_evaluations_budget=350)
            @test partial.stop_reason === :eval_budget
            @test JSON.parsefile(joinpath(dir, "run_summary.json"))["joint_ess"] === nothing
            state = deserialize(joinpath(dir, "run_latest.jls"))
            waiting = first(c for c in state.chains if c.status === :waiting)
            @test waiting.target_id == ids[waiting.chain_index]
            @test_throws ArgumentError FixedKernelNUTSState(waiting)
            @test_throws ArgumentError FixedKernelNUTSState(waiting; position=zeros(waiting.dimension))
            @test_throws ArgumentError stream_mcmc(waiting, targets[waiting.chain_index];
                path=joinpath(dir, "waiting.bin"), n_draws=1)
            rm(joinpath(dir, "run_summary.json"))
            @test_throws ArgumentError sample(; target_ids=reverse(ids), checkpoint_dir=dir, resume=true)
            resumed = sample(657; target_ids=ids, checkpoint_dir=dir, resume=true)
            full = sample()
            @test getproperty.(resumed.results, :target_id) == ids
            @test full.total_evaluation_counter == resumed.total_evaluation_counter
            @test getproperty.(full.results, :posterior_position) == getproperty.(resumed.results, :posterior_position)
            @test getproperty.(full.results, :accounting) == getproperty.(resumed.results, :accounting)
            terminal = deserialize(joinpath(dir, "run_latest.jls"))
            @test all(c -> c.status === :done && c.n_samples == 1, terminal.chains)
            @test [c.posterior_position for c in terminal.chains] == [r.posterior_position for r in resumed.results]
        end
    end

    @testset "each conditional keeps its controls and returns model-frame draws" begin
        framed = [ReparametrizedProblem(
            IndexedReparametrization([2 => Reparametrization(PartiallyCentered(1.),
                PartiallyCentered(i/8), 2., 0.5)]), targets[i], AutoEnzyme()) for i in 1:6]
        mktempdir() do dir
            out = cooperative_warmup_mcmc(Xoshiro(41), framed; features=1.:6., n_wave1=2,
                init, n_draws=2, target_ids=1:6, checkpoint_dir=dir)
            @test out.stop_reason === :n_draws
            for (i, t) in enumerate(framed)
                cp = deserialize(joinpath(dir, "chain_$i", "cp_latest.jls"))
                @test cp.status === :done && cp.n_samples == 2
                @test cp.reparam_sources == W.reparam_sources(t)
                expected = stack(last(t.reparametrizer(v)) for v in eachcol(cp.posterior_position))
                @test out.results[i].posterior_position ≈ expected
            end
        end
    end

    @testset "the same freeze gate runs before any shared-target draw" begin
        cfg = (; n_evaluations=64, recording_target=100, initializer_kw=(;),
            target_acceptance_rate=0.8, monitor_ess=false, stepsize_adaptation_limit=1)
        chain = W._coop_chain(1, Xoshiro(13), targets[1], init(1), cfg; progress=nothing)
        push!(chain.stepsizes, 0.1)
        for _ in 1:31
            chain.freeze.pending = true
            chain.freeze.clean = false
            push!(chain.freeze.scores, 1.)
            W._qualify_freezes!([chain])
        end
        @test chain.status === :needs_replacement && chain.freeze.extensions == 30
        @test chain.n_samples == 0
        observer = W._coop_chain(2, Xoshiro(14), targets[2], init(2), cfg; progress=nothing)
        append!(observer.freeze.scores, fill(1., 9))
        chain.status = :warming
        chain.freeze.extensions = 0
        empty!(chain.freeze.scores)
        for score in (12., 2.)
            push!(chain.freeze.scores, score)
            chain.freeze.pending = chain.freeze.clean = true
            W._qualify_freezes!([observer, chain])
            @test chain.status === (score > 10 ? :warming : :piloting)
        end
        @test chain.freeze.extensions == 1
        out = cooperative_warmup_mcmc(Xoshiro(18), targets[1]; init=init(1), n_chains=1, n_draws=1)
        @test only(out.results).accounting.warmup_transitions >= 50
        @test only(out.results).n_samples == 1
    end

    @test_throws ArgumentError sample(; target_ess=5)
    @test_throws DimensionMismatch sample(; init=[init(1)])
    @test_throws DimensionMismatch sample(; n_chains=2)
    @test_throws DimensionMismatch sample(; features=[[0.]])
end
