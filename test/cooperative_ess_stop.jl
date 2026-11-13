@testitem "cooperative target_ess diagnoses every returned history" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, Statistics, Distributions, LogDensityProblems
    using MCMCDiagnosticTools, DifferentiationInterface, Enzyme, JSON
    const W = WarmupHMC

    struct StudentTarget
        dimension::Int
    end
    LogDensityProblems.dimension(p::StudentTarget) = p.dimension
    LogDensityProblems.capabilities(::Type{StudentTarget}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity_and_gradient(::StudentTarget, x) =
        (-3.5sum(log1p, x.^2 ./ 6), -7x ./ (6 .+ x.^2))
    LogDensityProblems.logdensity(p::StudentTarget, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

    # Install retained draws at a round barrier, leaving production stop and
    # finalization code in charge. Their lengths/statuses model unequal warm-up.
    function retained_run(draws; target_ess=100.0, rhat_threshold=1.01,
            n_evaluations_budget=typemax(Int), time_budget=Inf,
            targets=fill(StudentTarget(size(first(draws), 1)), length(draws)))
        d = size(first(draws), 1)
        cfg = (; n_chains=length(draws), n_cores=1, n_evaluations=1000,
            recording_target=1000, initializer_kw=(;), target_acceptance_rate=0.8,
            monitor_ess=true, target_ess, rhat_threshold, n_evaluations_budget, time_budget)
        init = (; position=zeros(d), squared_scale=ones(d))
        run = W.CoopRun(Xoshiro(1), first(targets), cfg, init, nothing, "retained regression",
            time_ns(), 0.0, nothing, W.CoopChain[], W.CoopClusters(),
            Threads.Atomic{Int}(0), 0, 1, nothing, NaN)
        for (i, (target, x)) in enumerate(zip(targets, draws))
            chain = W._coop_chain(i, Xoshiro(i), target, init, cfg; progress=nothing)
            append!(chain.recording_lpdf.posterior_position, x)
            chain.n_samples = size(x, 2)
            chain.status = isempty(x) ? :warming : :sampling
            push!(run.chains, chain)
        end
        run
    end

    @testset "ragged rank-normalized split R-hat" begin
        rng = Xoshiro(101)
        # The rectangular case is independently checked against the maintained
        # diagnostic library, including ranks with ties and heavy tails.
        for distribution in (Normal(), TDist(1), Poisson(2))
            draws = [rand(rng, distribution, 3, 2000) for _ in 1:4]
            rectangular = W._stack_chain_tails(draws; min_chain_draws=11)
            @test W._retained_rhat(draws) ≈ maximum(rhat(rectangular)) atol=1e-12
        end
        draws = [rand(rng, TDist(6), 2, n) for n in (4001, 3000, 2001)]
        before = deepcopy(draws)
        @test W._retained_rhat(draws) < 1.01
        @test W._retained_rhat(reverse(draws)) ≈ W._retained_rhat(draws)
        @test draws == before
        # Folding catches a scale mismatch even when chain means agree.
        scaled = [rand(rng, TDist(6), 1, 2000) for _ in 1:3]
        scaled[3] .*= 100
        @test W._retained_rhat(scaled) > 1.1
        # Equal split-chain weights keep a short discrepant chain visible.
        @test W._retained_rhat([randn(rng, 1, 100_000), randn(rng, 1, 100) .+ 20]) > 1.1
        @test W._retained_rhat([zeros(1, 100), ones(1, 100)]) == Inf
        @test W._retained_rhat([ones(1, 100), ones(1, 100)]) == Inf
        @test W._retained_rhat([zeros(1, 0)]) == Inf
        @test W._retained_rhat([draws[1], zeros(2, 10)]) == Inf
        nonfinite = deepcopy(draws)
        nonfinite[1][1, 1] = NaN
        @test W._retained_rhat(nonfinite) == Inf
        nonfinite[1][1, 1] = Inf
        @test W._retained_rhat(nonfinite) == Inf
        @test_throws DimensionMismatch W._retained_rhat([zeros(1, 100), zeros(2, 100)])
    end

    @testset "a large best-subset ESS cannot hide retained disagreement" begin
        rng = Xoshiro(20261113)
        td(n) = rand(rng, TDist(6), 1, n)
        cases = [
            [td(4000), td(3000), td(200) .+ 100],
            [hcat(td(4000) .+ 100, td(1000)), td(1000), td(1000)],
            [td(4000), td(3000), td(5) .+ 100],
        ]
        # In the second case even ALL chains' common tails agree.
        @test maximum(rhat(W._stack_chain_tails(cases[2]; min_chain_draws=11))) < 1.01
        for draws in cases
            @test first(W._best_pooled_ess(draws, 2)) > 100
            @test W._retained_rhat(draws) > 1.01
            run = retained_run(draws)
            @test !W._check_stop!(run)
            @test isnothing(run.stop_reason)
            @test run.last_pooled_ess > 100
            out = W._finalize(run)
            @test out.stop_reason !== :target_ess
            @test [r.posterior_position for r in out.results] == draws
        end
        # A disagreeing history still counts when its chain has finished.
        run = retained_run(first(cases))
        last(run.chains).status = :done
        @test !W._check_stop!(run)
        foreach(c -> c.status = :done, run.chains)
        @test W._check_stop!(run)
        @test run.stop_reason === :n_draws

        for draws in ([td(4000), td(3000), td(2001)], [td(4000), td(3000), zeros(1, 0)], [td(4000)])
            run = retained_run(draws)
            @test W._check_stop!(run)
            @test run.stop_reason === :target_ess
            out = W._finalize(run)
            @test out.joint_ess >= 100
            @test W._retained_rhat([r.posterior_position for r in out.results]) < 1.01
            @test [r.posterior_position for r in out.results] == draws
        end
        # A single eligible chain cannot satisfy a multi-chain run's ESS bound.
        @test !W._check_stop!(retained_run([td(4000), zeros(1, 0)]))
        for (kwargs, reason) in (( (; n_evaluations_budget=0), :eval_budget),
                ((; time_budget=0.0), :time_budget))
            run = retained_run(first(cases); kwargs...)
            @test W._check_stop!(run)
            @test run.stop_reason === reason
        end
        run = retained_run(first(cases); target_ess=Inf)
        @test !W._check_stop!(run)
        foreach(c -> c.status = :done, run.chains)
        @test W._check_stop!(run)
        @test run.stop_reason === :n_draws
    end

    @testset "stop and returned results use the same model frame" begin
        translated(shift) = ReparametrizedProblem(IndexedReparametrization([
            1 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), shift, 0.0)
        ]), StudentTarget(1), AutoEnzyme())
        rng = Xoshiro(20261114)
        draws = [rand(rng, TDist(6), 1, 3000) for _ in 1:3]
        run = retained_run(draws; targets=[translated(0.0), translated(0.0), translated(100.0)])
        @test !W._check_stop!(run)
        @test run.last_pooled_ess > 100
        @test [copy(W.chain_draws(c)) for c in run.chains] == draws
        expected = [copy(W._model_chain_draws(c)) for c in run.chains]
        @test W._retained_rhat(expected) > 1.1
        out = W._finalize(run)
        @test [r.posterior_position for r in out.results] == expected

        # Conversely, different source frames can yield agreeing model draws.
        shifts = [0.0, 100.0]
        sources = [draws[i] .- shifts[i] for i in 1:2]
        run = retained_run(sources; targets=translated.(shifts))
        @test first(W._best_pooled_ess(sources, 2)) < 100
        @test W._check_stop!(run)
        @test run.stop_reason === :target_ess
        @test [copy(W.chain_draws(c)) for c in run.chains] == sources
        out = W._finalize(run)
        @test out.joint_ess >= 100
        @test W._retained_rhat([r.posterior_position for r in out.results]) < 1.01
    end

    @testset "non-Gaussian sampling and threshold contract" begin
        target = StudentTarget(4)
        for threshold in (1.0, 0.99, Inf, NaN)
            @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(1), target;
                n_draws=10, rhat_threshold=threshold)
        end
        dir = mktempdir()
        out = cooperative_warmup_mcmc(Xoshiro(1), target; n_chains=4, target_ess=1000,
            init=(; position=zeros(4), squared_scale=ones(4)),
            n_evaluations_budget=100_000, checkpoint_dir=dir, rhat_threshold=1.02)
        @test out.stop_reason === :target_ess
        @test out.joint_ess >= 1000
        @test W._retained_rhat([r.posterior_position for r in out.results]) < 1.02
        @test JSON.parsefile(joinpath(dir, "run_manifest.json"))["rhat_threshold"] == 1.02

        # A crash/restart does not bypass the agreement guard or change the
        # retained history at success. The resume call supplies the thresholds.
        resumed_dir = mktempdir()
        partial = cooperative_warmup_mcmc(Xoshiro(1), target; n_chains=4, target_ess=Inf,
            init=(; position=zeros(4), squared_scale=ones(4)),
            n_evaluations_budget=8000, checkpoint_dir=resumed_dir, rhat_threshold=1.05)
        @test partial.stop_reason === :eval_budget
        rm(joinpath(resumed_dir, "run_summary.json")) # simulate a crash before finalization
        resumed = cooperative_warmup_mcmc(Xoshiro(99), target; n_chains=4, target_ess=1000,
            n_evaluations_budget=100_000, checkpoint_dir=resumed_dir, resume=true,
            rhat_threshold=1.02)
        @test resumed.stop_reason === :target_ess
        @test [r.posterior_position for r in resumed.results] == [r.posterior_position for r in out.results]
        @test resumed.total_evaluation_counter == out.total_evaluation_counter
    end
end
