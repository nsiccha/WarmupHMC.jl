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

const W = WarmupHMC
test_problem(plan) = ReparametrizedProblem(
    IndexedReparametrization([2 => Reparametrization(PartiallyCentered(0.0),
        PartiallyCentered(0.0), 0.0, x -> x[1])]),
    DiagGaussian(zeros(2), ones(2)), AutoEnzyme(); scoring_plan=plan)

@testset "pooling streaming moments agrees with serial weighted observations" begin
    serial = W.WeightedReparametrizationLoss()
    pooled = W.WeightedReparametrizationLoss()
    rng = Xoshiro(14)
    for groups in (0, 5, 0, 7, 19, 0)
        part = W.WeightedReparametrizationLoss()
        for j in 1:groups
            for _ in 1:mod1(j, 4)
                obs = Tuple(randn(rng, 3))
                weight = exp(randn(rng))
                for loss in (part, serial)
                    W.OnlineStatsBase.fit!(loss, obs; weight, count=false)
                end
            end
            W._mark_group!(part)
            W._mark_group!(serial)
        end
        W._merge_nonlinear!(pooled, part, nothing)
    end
    for field in fieldnames(typeof(serial))
        @test getfield(pooled, field) ≈ getfield(serial, field)
    end
    @test pooled.groups == 31
    @test W.effective_n(pooled) ≈ W.effective_n(serial)
    @test W.reparametrization_loss(pooled) ≈ W.reparametrization_loss(serial)
end

@testset "pooling leaf samples preserves paired columns and stream mass" begin
    capacity = 10_000
    reservoir() = W.WeightedLeafSample(1, capacity)
    # ChainEvidence combines both representations for an indexed MarkovChain.
    combined = W.ChainEvidence(W.OnlineReparametrizer([]), reservoir())
    rng = Xoshiro(19)
    for (x, w) in ((1, 1), (2, 3), (3, 6), (4, 0))
        part = reservoir()
        w > 0 && W._observe_sample!(part, [x], [-x], w)
        W._merge_nonlinear!(combined, W.ChainEvidence(W.OnlineReparametrizer([]), part), rng)
    end
    @test combined.sample.total == 10
    @test combined.sample.gradient == -combined.sample.position
    @test [mean(==(x), combined.sample.position) for x in 1:4] ≈ [0.1, 0.3, 0.6, 0.0] atol=0.02
end

# Only the candidate c=0.3 has correlation -1. Every other candidate has +1.
scorer() = CandidateScoringPlan((ir, q, g) -> q[2],
    (x, number, index, pair, c) -> (0.0, x, c.c ≈ 0.3 ? -x : x))
function select_from_sample!(ir, q, g)
    isempty(q) && return false
    c = clamp(0.5 + mean(view(q, 1, :)) / 10, 0, 1)
    W.restore_reparam_controls!(ir, [2 => PartiallyCentered(c)])
    true
end

@testset "pooled refits replay from a round of streaming evidence" begin
    for (plan, mode) in ((scorer(), :linear_pool), (WindowSelectionPlan(select_from_sample!), :nuts_weighted))
        config = (; n_chains=2, n_cores=1, n_evaluations=32, n_draws=55,
            recording_target=19, max_tree_depth=3, restart_threshold=0.0,
            stepsize_adaptation_limit=2, init=(; position=[0.1,-0.1], squared_scale=ones(2)))
        evidence = (; nonlinear_evidence=mode, nonlinear_trajectory_weighting=:stepsize,
            nonlinear_good_leaf_threshold=-2.0)
        full, sliced = mktempdir(), mktempdir()
        sample_dir(dir; kwargs...) = cooperative_warmup_mcmc(Xoshiro(21), test_problem(plan);
            config..., checkpoint_dir=dir, kwargs...)
        reference = sample_dir(full; evidence...)
        p2(i) = deserialize(joinpath(full, "chain_$i", "cp_window_2.jls"))
        @test p2(1).reparam_sources == p2(2).reparam_sources
        @test !iszero(last(only(p2(1).reparam_sources)).c)
        if plan isa CandidateScoringPlan
            @test last(only(p2(1).reparam_sources)).c ≈ 0.3
        end
        @test any(b -> b.centering_refit, first(reference.results).batches)
        # Stopping in round 3 leaves its previous complete round (2) on disk.
        stop = sample_dir(sliced; evidence..., n_evaluations_budget=sum(p2(i).total_evaluation_counter for i in 1:2))
        @test stop.stop_reason === :eval_budget
        saved = deserialize(joinpath(sliced, "run_latest.jls"))
        @test saved.round == 2
        for p in saved.chains
            r = p.nonlinear_recorder
            @test (r.mode, r.trajectory_weighting, r.good_leaf_threshold) == (mode, :stepsize, -2.0)
            if plan isa CandidateScoringPlan
                @test all(row -> row.groups > 2, candidate_scoring_losses(p))
            else
                @test r.online.total > 0 && size(r.online.position, 2) == 19
                changed = W._coop_nonlinear_recorder(p, test_problem(plan), 19,
                    (; nonlinear_trajectory_weighting=:unit), Xoshiro(702))
                @test changed.mode === :nuts_weighted && changed.trajectory_weighting === :unit
                @test changed.online.total == 0 && r.online.total > 0
            end
        end
        rm(joinpath(sliced, "run_summary.json"))
        resumed = sample_dir(sliced; resume=true, n_cores=2)
        for (a, b) in zip(reference.results, resumed.results)
            @test a.posterior_position == b.posterior_position
            @test a.posterior_gradient == b.posterior_gradient
            @test isequal(a.batches, b.batches)
            @test a.total_evaluation_counter == b.total_evaluation_counter
        end
    end
end

function barrier_run(plan; mode, n_chains)
    rp = test_problem(plan)
    cfg = (; family=false, max_reflections=0, recording_target=11, n_evaluations=32, n_cores=1, target_acceptance_rate=0.8,
        initializer_kw=(;), monitor_ess=false, kappa_prior_weight=2.0, max_clusters=2,
        bandwidth=1.0, restart_threshold=1.5, nonlinear_settings=(; nonlinear_evidence=mode))
    init = (; position=[0.2,0.1], squared_scale=ones(2))
    run = W.CoopRun(Xoshiro(81), rp, cfg, init, nothing, "centering barrier", time_ns(),
        0.0, deepcopy(rp), W.CoopChain[], W.CoopClusters(), Threads.Atomic{Int}(0), 1, 1, nothing, NaN)
    W._create_chains!(run, 1:n_chains)
    run
end

@testset "candidate scoring needs both chains to cross the selection threshold" begin
    run = barrier_run(scorer(); mode=:linear_pool, n_chains=2)
    for c in run.chains
        for t in 1:2
            q = [0.0 0.1; 0.0 c.chain_index + t]
            leaves = (; position=q, gradient=-q, dH=[0.0, 0.0])
            W.record_nonlinear!(c.nonlinear_recorder, c.lpdf, leaves, 1.0)
        end
        @test all(r -> r.groups == 2 && ismissing(r.loss), candidate_scoring_losses(c.nonlinear_recorder))
        @test isempty(c.recording_lpdf.halo_position)
    end
    controls = W._refit_reparametrization!(run)
    @test last(only(controls)).c ≈ 0.3
end

@testset "one shared refit validates every running point before committing" begin
    function select!(ir, q, g)
        @test size(q) == size(g) == (2, 11)
        W.restore_reparam_controls!(ir, [2 => PartiallyCentered(1.0)])
        true
    end
    run = barrier_run(WindowSelectionPlan(select!); mode=:all_good_leaves, n_chains=3)
    cfg, init = run.cfg, run.init
    a, b, done = run.chains
    function fill_evidence!(c)
        for t in (-0.5, 0.1, 0.7)
            q = [t, t+0.25]
            W._fit_moments!(c.moments, q, -q)
            append!(c.recording_lpdf.halo_position, q)
            append!(c.recording_lpdf.halo_gradient, -q)
            W._observe_sample!(c.nonlinear_recorder.online, q, -q, 1.0)
        end
    end
    for c in run.chains
        fill_evidence!(c)
        append!(c.recording_lpdf.posterior_position, init.position)
        append!(c.recording_lpdf.posterior_gradient, -init.position)
        c.n_samples = 1
        c.status = :sampling
    end
    done.status = :done
    b.scale = Diagonal([1.0, 10.0])
    W._update_clusters!(run.clusters, run.chains, cfg)
    W._batch_end!(run)
    @test !only(a.batches).restart && only(b.batches).restart
    @test !only(a.batches).centering_refit && !only(b.batches).centering_refit
    @test a.n_samples == done.n_samples == 1 && b.n_samples == 0
    @test a.nonlinear_recorder.online.total == 3 && b.nonlinear_recorder.online.total == 0
    fill_evidence!(b)

    # The pool and first point transport, but the SECOND active point overflows.
    old_point = b.position_and_gradient
    b.position_and_gradient = W.DynamicHMC.evaluate_ℓ(b.lpdf, [1000.0, 0.5])
    positions = [copy(c.recording_lpdf.halo_position) for c in run.chains]
    points = [c.position_and_gradient for c in run.chains]
    logs, change = Test.collect_test_logs() do
        W._refit_reparametrization!(run)
    end
    @test isnothing(change)
    @test occursin("current point has nonfinite coordinates", only(logs).message)
    @test Dict(only(logs).kwargs)[:chain] == 2
    @test last(only(W.reparam_sources(run.reparam_lpdf))).c == 0.0
    for (c, q, point) in zip(run.chains, positions, points)
        @test last(only(W.reparam_sources(c.lpdf))).c == 0.0
        @test c.recording_lpdf.halo_position == q && c.position_and_gradient === point
        @test c.nonlinear_recorder.online.total == 3
    end

    b.position_and_gradient = old_point
    @test !isnothing(W._refit_reparametrization!(run))
    for (c, q) in zip((a,b), positions)
        @test last(only(W.reparam_sources(c.lpdf))).c == 1.0
        @test c.recording_lpdf.halo_position != q
        @test hcat((last(W.reparametrizer(c.lpdf)(col)) for col in eachcol(c.recording_lpdf.halo_position))...) ≈ q
        @test c.moments.position_mean ≈ vec(mean(c.recording_lpdf.halo_position; dims=2))
    end
    @test last(only(W.reparam_sources(done.lpdf))).c == 0.0
    @test W.chain_draws(done)[:,1] == init.position
    @test !W._coop_geometry_current(done)
    run.clusters = W.CoopClusters()
    W._update_clusters!(run.clusters, run.chains, cfg)
    @test all(iszero, run.clusters.responsibilities[3, :])
    @test sum(s.weight for s in run.clusters.stats) ≈ a.moments.weight + b.moments.weight
end

@testset "a cooperative payload predating streaming resumes its halo fit" begin
    rp = test_problem(WindowSelectionPlan(select_from_sample!))
    config = (; n_chains=2, n_cores=1, n_draws=55, n_evaluations=16, max_tree_depth=3,
        stepsize_adaptation_limit=2, restart_threshold=0.0,
        init=(; position=zeros(2), squared_scale=ones(2)))
    full, dir = mktempdir(), mktempdir()
    reference = cooperative_warmup_mcmc(Xoshiro(22), rp; config..., checkpoint_dir=full)
    budget = sum(deserialize(joinpath(full, "chain_$i", "cp_window_2.jls")).total_evaluation_counter for i in 1:2)
    cooperative_warmup_mcmc(Xoshiro(22), rp; config..., checkpoint_dir=dir, n_evaluations_budget=budget)
    path = joinpath(dir, "run_latest.jls")
    checkpoint = deserialize(path)
    @test checkpoint.round == 2
    chains = [Base.structdiff(p, (; nonlinear_recorder=nothing, geometry_current=true)) for p in checkpoint.chains]
    serialize(path, merge(checkpoint, (; chains)))
    rm(joinpath(dir, "run_summary.json"))
    resumed = cooperative_warmup_mcmc(Xoshiro(999), rp; config..., checkpoint_dir=dir, resume=true)
    @test all(a.posterior_position == b.posterior_position for (a,b) in zip(reference.results, resumed.results))
end
end
