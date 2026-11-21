@testitem "A family slot that exhausts its recovery fails alone" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, LogDensityProblems, Random, Serialization, JSON
    const W = WarmupHMC

    # Flat inside a small box and outside the support everywhere else: NUTS
    # never turns around inside it, so every transition ends on a divergent
    # leaf and no kernel passes the first-draw check. It stands in for a
    # conditional that no chain can sample (snag cooperative-fami-a8bdb100).
    struct Box
        center::Vector{Float64}
    end
    LogDensityProblems.dimension(p::Box) = length(p.center)
    LogDensityProblems.capabilities(::Type{Box}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity_and_gradient(p::Box, x) =
        (maximum(abs, x .- p.center) <= 0.05 ? 0.0 : -Inf, zeros(length(x)))
    LogDensityProblems.logdensity(p::Box, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

    center(i) = [2.5i, -i / 2]
    features = [[Float64(i), 1.0] for i in 1:9]
    medoids = findall(iszero, W.CoopFamily([DiagGaussian(center(i), [1.0, 1.2]) for i in 1:9],
        nothing, features, 3).source)
    family(bad) = Union{DiagGaussian{Vector{Float64}},Box}[i in bad ? Box(center(i)) :
        DiagGaussian(center(i), [1.0, 1.2]) for i in 1:9]
    init(i) = (; position=center(i), squared_scale=[1.0, 1.44])
    kwargs = (; init, n_draws=1, n_evaluations=40, stepsize_adaptation_limit=8, n_cores=1)

    function check_others(out, targets, failed; n_draws=1)
        for (i, r) in enumerate(out.results)
            i in failed && continue
            @test r.status === :done && r.n_samples == n_draws
            lp, g = LogDensityProblems.logdensity_and_gradient(targets[i], r.posterior_position[:, end])
            @test isfinite(lp) && r.posterior_gradient[:, end] ≈ g
        end
        for i in failed
            r = out.results[i]
            @test r.status === :recovery_failed && r.n_samples == 0 && isempty(r.posterior_position)
            @test [a.generation for a in r.abandoned] == [1, 2]
            @test r.accounting.replacements == 1 && !isempty(r.accounting.abandoned_freeze_scores)
        end
    end

    @testset "cold slots: the other targets still draw" begin
        targets = family([4])
        # Enough draws that the slot fails before the others hold theirs.
        out = cooperative_warmup_mcmc(Xoshiro(3), targets; kwargs..., n_draws=200)
        @test out.stop_reason === :recovery_failed
        check_others(out, targets, [4]; n_draws=200)
    end

    @testset "a failed first-wave slot is no source" begin
        bad = first(medoids)
        targets = family([bad])
        out = cooperative_warmup_mcmc(Xoshiro(5), targets; features, n_wave1=3, kwargs...)
        @test out.stop_reason === :recovery_failed
        check_others(out, targets, [bad])
        @test sort(findall(r -> r.accounting.wave == 1, out.results)) == medoids
        for r in out.results
            r.accounting.wave == 2 || continue
            others = setdiff(medoids, [bad])
            @test r.accounting.source in others
            @test abs(r.accounting.source - r.chain_index) == minimum(abs(j - r.chain_index) for j in others)
        end
        # The released sampler ended here with every receiver still waiting.
        @test count(r -> r.accounting.wave == 2, out.results) == 6
    end

    @testset "with no qualified first-wave slot every receiver starts cold" begin
        targets = family(medoids)
        out = cooperative_warmup_mcmc(Xoshiro(7), targets; features, n_wave1=3, kwargs...)
        @test out.stop_reason === :recovery_failed
        check_others(out, targets, medoids)
        for r in out.results
            r.accounting.wave == 2 && @test r.accounting.source == 0
        end
    end

    @testset "crash resume continues past a failed slot" begin
        bad = first(medoids)
        targets = family([bad])
        ids = string.(1:9)
        sample(; extra...) = cooperative_warmup_mcmc(Xoshiro(5), targets; features, n_wave1=3,
            target_ids=ids, kwargs..., extra...)
        full = sample()
        # The smallest budget step after which the slot has failed: its
        # receivers are still waiting then, so the run is unfinished.
        dir, partial = "", nothing
        for budget in 50:50:full.total_evaluation_counter
            dir = mktempdir()
            partial = sample(; checkpoint_dir=dir, n_evaluations_budget=budget)
            partial.results[bad].status === :recovery_failed && break
        end
        begin
            @test partial.stop_reason === :eval_budget
            @test partial.results[bad].status === :recovery_failed
            @test any(r -> r.n_samples == 0 && r.status !== :recovery_failed, partial.results)
            rm(joinpath(dir, "run_summary.json"))
            resumed = sample(; checkpoint_dir=dir, resume=true)
            @test resumed.stop_reason === :recovery_failed
            @test getproperty.(resumed.results, :posterior_position) == getproperty.(full.results, :posterior_position)
            @test getproperty.(resumed.results, :status) == getproperty.(full.results, :status)
            @test JSON.parsefile(joinpath(dir, "run_summary.json"))["failed_chains"] == [bad]
            failed = deserialize(joinpath(dir, "chain_$bad", "cp_latest.jls"))
            @test failed.is_final && failed.status === failed.stop_reason === :recovery_failed
        end
    end
end
