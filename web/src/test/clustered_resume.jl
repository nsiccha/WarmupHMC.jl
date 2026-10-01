# Crash-resume for the clustered sampler: `restore_clustered_chain` round-trips
# a chain through its on-disk payload, and `resume=true` continues a crashed run
# as a whole ensemble (a clustered chain's scale is pooled across its
# cluster-mates, so per-chain resume is only meaningful ensemble-wide).
#
# Same guarantee as the cooperative side (decision `y72yij` option A): restored
# chains stepped with their restored RNGs reproduce the uninterrupted run
# exactly; pool-level assertions are statistical, never whole-run byte-identity.
#
# Run this item alone with
#   -- --file=clustered_resume.jl

@testitem "clustered crash-resume" setup=[Determinism] tags=[:checkpoint] begin
    using WarmupHMC, Serialization, Random, LinearAlgebra, LogDensityProblems
    using OnlineStatsBase
    using WarmupHMC: ClusteredChain, clustered_chain, advance_chain!,
                     cluster_and_adapt!, clustered_step!,
                     clustered_checkpoint_payload, restore_clustered_chain

    struct _Gauss3 end
    LogDensityProblems.dimension(::_Gauss3) = 3
    LogDensityProblems.capabilities(::Type{_Gauss3}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::_Gauss3, x) = -sum(abs2, x) / 2
    LogDensityProblems.logdensity_and_gradient(::_Gauss3, x) = (-sum(abs2, x) / 2, -x)

    struct _Gauss2 end
    LogDensityProblems.dimension(::_Gauss2) = 2

    # Explicit-`init` trick (see cooperative_resume.jl): no Pathfinder, no AD.
    _init() = (; position=zeros(3), squared_scale=Matrix{Float64}(I, 3, 3) * 2.0)
    _chain(seed=1; kwargs...) = clustered_chain(Xoshiro(seed), _Gauss3(); init=_init(), kwargs...)

    _roundtrip(p) = mktempdir() do dir
        path = joinpath(dir, "cp_latest.jls")
        open(io -> serialize(io, p), path, "w")
        deserialize(path)
    end

    @testset "restore round-trips an ensemble member" begin
        chains = [_chain(1), _chain(2)]
        foreach(advance_chain!, chains)
        cluster_and_adapt!(chains)

        restored = map(enumerate(chains)) do (i, chain)
            p = _roundtrip(clustered_checkpoint_payload(chain, i, 1))
            restore_clustered_chain(p, _Gauss3())
        end

        for (chain, r) in zip(chains, restored)
            @test (r.rng.s0, r.rng.s1, r.rng.s2, r.rng.s3) ==
                  (chain.rng.s0, chain.rng.s1, chain.rng.s2, chain.rng.s3)
            @test r.position_and_gradient.q == chain.position_and_gradient.q
            @test r.scale == chain.scale
            @test r.stepsize == chain.stepsize
            @test r.n_evaluations == chain.n_evaluations
            @test r.cluster_id == chain.cluster_id
            @test r.total_evaluation_counter == chain.total_evaluation_counter
            @test r.current_transition_counter == chain.current_transition_counter
            @test r.n_divergent_samples == chain.n_divergent_samples
            @test r.n_samples == chain.n_samples
            @test r.status == chain.status
            @test r.checkpoints == chain.checkpoints
            # The accumulating pooled-estimate input survives: same weight,
            # same prior, same variances.
            @test OnlineStatsBase.nobs(r.adaptation) == OnlineStatsBase.nobs(chain.adaptation)
            @test r.adaptation.position_variances.regularizing_n ==
                  chain.adaptation.position_variances.regularizing_n
            @test Matrix(r.recording_lpdf.posterior_position) ==
                  Matrix(chain.recording_lpdf.posterior_position)
        end

        # Ensemble continuity: one joint step on each ensemble is identical —
        # same re-clustering, same adoptions, same draws.
        clusters_live = clustered_step!(chains; parallel=false)
        clusters_restored = clustered_step!(restored; parallel=false)
        @test clusters_restored == clusters_live
        for (chain, r) in zip(chains, restored)
            @test Matrix(r.recording_lpdf.posterior_position) ==
                  Matrix(chain.recording_lpdf.posterior_position)
            @test r.scale == chain.scale
            @test r.status == chain.status
            @test r.cluster_id == chain.cluster_id
            @test r.total_evaluation_counter == chain.total_evaluation_counter
        end
    end

    @testset "resume-and-extend flips :done back to :sampling" begin
        chain = _chain(; n_draws=5)
        # Draws accumulate across windows (no restart without clustering), so
        # this reaches the cap deterministically.
        while size(chain.recording_lpdf.posterior_position, 2) < 5
            advance_chain!(chain)
        end
        chain.status = :done   # as `cluster_and_adapt!` would leave a capped chain
        p = _roundtrip(clustered_checkpoint_payload(chain, 1, 1))
        same = restore_clustered_chain(p, _Gauss3(); n_draws=5)
        @test same.status === :done
        extended = restore_clustered_chain(p, _Gauss3(); n_draws=50)
        @test extended.status === :sampling
    end

    @testset "config rules: inherit, refuse, re-pass" begin
        chain = _chain(; recording_target=64, regularizing_n=7.0)
        advance_chain!(chain)
        p = _roundtrip(clustered_checkpoint_payload(chain, 1, 1))

        inherited = restore_clustered_chain(p, _Gauss3())
        @test inherited.recording_lpdf.recorder.target == 64
        @test inherited.adaptation.position_variances.regularizing_n == 7.0

        @test_throws ArgumentError restore_clustered_chain(p, _Gauss3(); recording_target=128)
        @test_throws ArgumentError restore_clustered_chain(p, _Gauss3(); regularizing_n=5.0)
        @test_throws ArgumentError restore_clustered_chain(p, _Gauss3(); regularizing_var=1.0)

        # `weighting` always comes from the call — re-passed, never inherited.
        custom(hp, hg) = ones(size(hp, 2))
        @test restore_clustered_chain(p, _Gauss3(); weighting=custom).weighting === custom
        # Init-only knobs are accepted and ignored for a verbatim resuming call.
        verbatim = restore_clustered_chain(p, _Gauss3(); n_evaluations=1, init=_init())
        @test verbatim.n_evaluations == chain.n_evaluations
    end

    @testset "masquerade, schema, and dimension guards" begin
        chain = _chain()
        advance_chain!(chain)
        p = clustered_checkpoint_payload(chain, 1, 1)
        @test_throws ArgumentError restore_clustered_chain(merge(p, (; sampler=:cooperative)), _Gauss3())
        @test_throws ArgumentError restore_clustered_chain(merge(p, (; schema_version=999)), _Gauss3())
        @test_throws DimensionMismatch restore_clustered_chain(p, _Gauss2())
    end

    @testset "crash-resume end to end, windows as totals" begin
        mktempdir() do d
            part1 = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, init=_init())
            @test part1.n_windows == 1
            @test isfile(joinpath(d, "run_summary.json"))
            rm(joinpath(d, "run_summary.json"))

            # Same `max_windows`: nothing left to run — output only, no new windows.
            same = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, resume=true, init=_init())
            @test same.n_windows == 1
            @test length(same.results) == 2
            rm(joinpath(d, "run_summary.json"))

            # Larger `max_windows`: the run continues to the new total.
            resumed = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=3, checkpoint_dir=d, resume=true, init=_init())
            @test resumed.n_windows == 3
            @test resumed.total_evaluation_counter > part1.total_evaluation_counter
            @test isfile(joinpath(d, "run_summary.json"))
            # The output partition covers the whole ensemble exactly once.
            @test sort!(vcat([c.chain_indices for c in resumed.clusters]...)) == [1, 2]
            @test sum(r -> size(r.posterior_position, 2), resumed.results) > 0
            # The manifest is write-once: resume kept the original's criteria.
            manifest = read(joinpath(d, "run_manifest.json"), String)
            @test occursin("\"max_windows\":1", manifest)
        end
    end

    @testset "eval budget is cumulative across the crash" begin
        mktempdir() do d
            part1 = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, init=_init())
            rm(joinpath(d, "run_summary.json"))
            # The restored counters already hold part 1's evals, so a budget at
            # exactly that level is already spent: the resumed run stops BEFORE
            # stepping, where the uninterrupted run stopped — no extra window.
            spent = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=5, n_evaluations_budget=part1.total_evaluation_counter,
                checkpoint_dir=d, resume=true, init=_init())
            @test spent.n_windows == 1
            @test spent.total_evaluation_counter == part1.total_evaluation_counter
            @test occursin("\"stop_reason\":\"eval_budget\"",
                read(joinpath(d, "run_summary.json"), String))
            rm(joinpath(d, "run_summary.json"))
            # One eval above it runs exactly one more window rather than five.
            resumed = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=5, n_evaluations_budget=part1.total_evaluation_counter + 1,
                checkpoint_dir=d, resume=true, init=_init())
            @test resumed.n_windows == 2
            @test resumed.total_evaluation_counter > part1.total_evaluation_counter
        end
    end

    @testset "an ensemble already done stops before stepping" begin
        mktempdir() do d
            part1 = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                n_draws=5, max_windows=20, checkpoint_dir=d, init=_init())
            @test occursin("\"stop_reason\":\"n_draws\"",
                read(joinpath(d, "run_summary.json"), String))
            rm(joinpath(d, "run_summary.json"))
            # Same `n_draws`: every restored chain is `:done`, so no window runs.
            again = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                n_draws=5, max_windows=20, checkpoint_dir=d, resume=true, init=_init())
            @test again.n_windows == part1.n_windows
            @test again.total_evaluation_counter == part1.total_evaluation_counter
            @test occursin("\"stop_reason\":\"n_draws\"",
                read(joinpath(d, "run_summary.json"), String))
        end
    end

    @testset "a slot with no checkpoint starts fresh in place" begin
        mktempdir() do d
            clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, init=_init())
            rm(joinpath(d, "run_summary.json"))
            rm(joinpath(d, "chain_2"); recursive=true)
            resumed = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=2, checkpoint_dir=d, resume=true, init=_init())
            @test resumed.n_windows == 2
            @test length(resumed.results) == 2
        end
    end

    @testset "no-windows-left partition agrees with the chains' cluster ids" begin
        # The output names cluster `k` by its position, so the partition must be
        # ordered by id for `clusters[k]` to hold the chains with `cluster_id == k`.
        chains = [_chain(s) for s in 1:3]
        foreach(((c, id),) -> c.cluster_id = id, zip(chains, (2, 1, 2)))
        @test WarmupHMC._clusters_from_ids(chains) == [[2], [1, 3]]
        # A fresh slot (`cluster_id == 0`, never pooled) stands alone, after the pools.
        foreach(((c, id),) -> c.cluster_id = id, zip(chains, (1, 0, 1)))
        @test WarmupHMC._clusters_from_ids(chains) == [[1, 3], [2]]
    end

    @testset "run-dir guard" begin
        mktempdir() do d
            clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, init=_init())
            # A second run into the same dir would interleave — refuse.
            @test_throws ArgumentError clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, init=_init())
            # A FINALIZED run is refused too: run_summary.json is write-once and
            # its existence is the run-completed signal.
            summary = read(joinpath(d, "run_summary.json"), String)
            @test_throws ArgumentError clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=2, checkpoint_dir=d, resume=true, init=_init())
            @test read(joinpath(d, "run_summary.json"), String) == summary
            # ... while an unfinished (crashed) run resumes and overwrite clears.
            rm(joinpath(d, "run_summary.json"))
            resumed = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=2, checkpoint_dir=d, resume=true, init=_init())
            @test resumed.n_windows == 2
            cleared = clustered_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                max_windows=1, checkpoint_dir=d, overwrite=true, init=_init())
            @test cleared.n_windows == 1
        end
        mktempdir() do d
            @test_throws ArgumentError clustered_warmup_mcmc([Xoshiro(1)], _Gauss3();
                max_windows=1, checkpoint_dir=d, resume=true, init=_init())
        end
    end
end
