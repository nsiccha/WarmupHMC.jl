using WarmupHMC, Test, Serialization, Random, LinearAlgebra, LogDensityProblems
using WarmupHMC: ClusteredChain, clustered_checkpoint_payload, NutpieScaleAdaptation,
                 _write_clustered_checkpoints, write_clustered_run_manifest,
                 write_clustered_run_summary, checkpoint_schema_version,
                 default_weighting

# Analytic-gradient Gaussian: lets us build a real ClusteredChain without
# Pathfinder (and so without an AD backend), exactly as cooperative_checkpoint.jl
# does. The payload contract is about which FIELDS reach disk, so a struct with
# real containers and placeholder scalars exercises it precisely.
struct _CGauss end
LogDensityProblems.dimension(::_CGauss) = 2
LogDensityProblems.capabilities(::Type{_CGauss}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_CGauss, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(::_CGauss, x) = (-sum(abs2, x) / 2, -x)

_cchain(; dimension=2) = begin
    rng = Random.Xoshiro(1)
    lpdf = _CGauss()
    recording_lpdf = WarmupHMC.RecordingPosterior2(
        lpdf; recorder=WarmupHMC.LimitedRecorder2(10), rng,
    )
    ClusteredChain(
        rng, lpdf, recording_lpdf,
        nothing, nothing, dimension,        # algorithm, stepsize_adaptation, dimension
        10, 50, 1000, typemax(Int),         # recording_target, sa_limit, n_draws, max_window_evals
        default_weighting,
        nothing,                            # position_and_gradient
        Diagonal(ones(dimension)),          # scale
        nothing,                            # kinetic_energy
        1.0, nothing, 1000,                 # stepsize, stepsize_state, n_evaluations
        NutpieScaleAdaptation(dimension),
        0, 0, 0, 0, 0,                      # cluster_id + the four counters
        :warming, NamedTuple[],
    )
end

@testset "clustered checkpoint payload" begin
    chain = _cchain()
    chain.cluster_id = 4

    @testset "excludes lpdf, carries the shared contract" begin
        p = clustered_checkpoint_payload(chain, 3, 7)
        # The whole reason resume re-supplies lpdf: ClusteredChain holds it as a
        # `const` field and it may be a non-serializable BridgeStan handle. A
        # field-selecting payload must never carry it.
        @test !hasproperty(p, :lpdf)
        @test p.schema_version == checkpoint_schema_version()
        @test p.sampler === :clustered
        @test p.chain_index == 3
        @test p.window == 7
        @test p.cluster_id == 4
        @test p.status === :warming
        @test p.is_final == false
        @test p.stop_reason === :running
        # Present-but-inapplicable, so a generic reader needs no per-sampler branch.
        @test p.stuck_reason === nothing
    end

    @testset "is_final/stop_reason follow status" begin
        chain.status = :done
        p = clustered_checkpoint_payload(chain, 1, 1)
        @test p.is_final
        @test p.stop_reason === :n_draws
    end

    @testset "round-trips through serialize" begin
        p = clustered_checkpoint_payload(chain, 1, 2)
        path, io = mktemp()
        serialize(io, p); close(io)
        q = deserialize(path)
        @test q.sampler === :clustered
        @test q.chain_index == 1
        @test q.cluster_id == 4
        rm(path; force=true)
    end
end

@testset "clustered checkpoint layout matches the cooperative one" begin
    dir = mktempdir()
    chains = [_cchain(), _cchain()]
    chains[1].cluster_id = 1
    chains[2].cluster_id = 2

    _write_clustered_checkpoints(dir, chains, 1)
    _write_clustered_checkpoints(dir, chains, 2)

    @testset "per-chain dirs, numbered windows, and cp_latest" begin
        @test isdir(joinpath(dir, "chain_1"))
        @test isdir(joinpath(dir, "chain_2"))
        @test isfile(joinpath(dir, "chain_1", "cp_window_1.jls"))
        @test isfile(joinpath(dir, "chain_1", "cp_window_2.jls"))
        @test isfile(joinpath(dir, "chain_1", "cp_latest.jls"))
        # Same as cooperative: there is deliberately no cp_init.jls, so a reader
        # must not require one.
        @test !isfile(joinpath(dir, "chain_1", "cp_init.jls"))
    end

    @testset "chain_<i> matches the payload's chain_index" begin
        for i in 1:2
            p = deserialize(joinpath(dir, "chain_$i", "cp_latest.jls"))
            @test p.chain_index == i
            @test p.cluster_id == i
        end
    end

    @testset "cp_latest tracks the newest window" begin
        @test deserialize(joinpath(dir, "chain_1", "cp_latest.jls")).window == 2
        @test deserialize(joinpath(dir, "chain_1", "cp_window_1.jls")).window == 1
    end

    @testset "written checkpoints are immutable" begin
        # cluster_id is as-of-that-checkpoint and never backfilled — the property
        # a consumer's (config, checkpoint#) pinning relies on.
        before = read(joinpath(dir, "chain_1", "cp_window_1.jls"))
        chains[1].cluster_id = 99
        _write_clustered_checkpoints(dir, chains, 3)
        @test read(joinpath(dir, "chain_1", "cp_window_1.jls")) == before
        @test deserialize(joinpath(dir, "chain_1", "cp_window_1.jls")).cluster_id == 1
        @test deserialize(joinpath(dir, "chain_1", "cp_latest.jls")).cluster_id == 99
    end
end

@testset "clustered run manifest / summary" begin
    dir = mktempdir()
    rngs = [Random.Xoshiro(11), Random.Xoshiro(22)]

    write_clustered_run_manifest(dir, rngs,
        (; n_draws=1000, max_windows=16, n_evaluations_budget=typemax(Int), threshold=sqrt(2.0)))
    m = read(joinpath(dir, "run_manifest.json"), String)

    @test occursin("\"sampler\":\"clustered\"", m)
    @test occursin("\"schema_version\":1", m)   # Int, not 1.0
    @test occursin("\"n_chains_requested\":2", m)
    # JSON has no Inf: an unset bound reads as null rather than a lie.
    @test occursin("\"n_evaluations_budget\":null", m)
    @test occursin("chain_rngs", m)

    out = (; n_windows=3, results=[(; cluster_id=1), (; cluster_id=2)],
             clusters=[(;), (;)], total_evaluation_counter=1234)
    write_clustered_run_summary(dir, out, :n_draws)
    s = read(joinpath(dir, "run_summary.json"), String)

    @test occursin("\"sampler\":\"clustered\"", s)
    @test occursin("\"is_final\":true", s)
    @test occursin("\"stop_reason\":\"n_draws\"", s)
    # Genuinely run-level state: a per-chain checkpoint records the cluster_id a
    # chain believed it was in, but only the run knows the partition.
    @test occursin("\"cluster_assignments\":[1,2]", s)
    @test occursin("\"n_clusters\":2", s)

    @testset "both are write-once" begin
        # EXISTENCE of run_summary.json is the run-completed signal, so neither
        # file may be rewritten (that would reintroduce the torn-read hazard).
        @test isfile(joinpath(dir, "run_manifest.json"))
        @test isfile(joinpath(dir, "run_summary.json"))
    end
end

@testset "nothing-dir writes nothing" begin
    @test _write_clustered_checkpoints(nothing, [_cchain()], 1) === nothing
    @test write_clustered_run_manifest(nothing, [Random.Xoshiro(1)], (;)) === nothing
    @test write_clustered_run_summary(nothing, (;), :n_draws) === nothing
end
