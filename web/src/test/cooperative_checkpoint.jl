using WarmupHMC, Test, Serialization, Random, LinearAlgebra, LogDensityProblems
using OnlineStatsBase
using WarmupHMC: CooperativeChain, cooperative_checkpoint_payload,
                 _chain_checkpoint_paths, _flush_checkpoint!,
                 checkpoint_schema_version, stuck_reason, is_stuck

# Analytic-gradient Gaussian: lets us build a real CooperativeChain without
# Pathfinder (and so without the AD backend), by passing `init` as a NamedTuple.
struct _Gauss end
LogDensityProblems.dimension(::_Gauss) = 2
LogDensityProblems.capabilities(::Type{_Gauss}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_Gauss, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(::_Gauss, x) = (-sum(abs2, x) / 2, -x)

# Build a CooperativeChain DIRECTLY rather than through `cooperative_chain`.
# The constructor runs the full init path (Pathfinder / `factorize(squared_scale).L`
# / matrix-free kinetic-energy operators), which is not what this file tests and
# which drags in an AD backend. The payload contract is about which FIELDS reach
# disk, so a struct with real containers and placeholder scalars exercises it
# exactly.
_chain(; chain_index=0, dimension=2) = begin
    rng = Random.Xoshiro(1)
    lpdf = _Gauss()
    recording_lpdf = WarmupHMC.RecordingPosterior2(
        lpdf; recorder=WarmupHMC.LimitedRecorder2(10), rng,
    )
    CooperativeChain(
        rng, chain_index, lpdf, recording_lpdf,
        nothing, nothing, dimension,        # algorithm, stepsize_adaptation, dimension
        10, 50, 2.0, false, false, 10, typemax(Int),
        (;), (;), (;), UInt64(0),           # scale/energy options, kwargs, start_time
        nothing, :pathfinder, nothing,      # position_and_gradient, active_transformation, kinetic_energy
        1.0, nothing, 1000,                 # stepsize, stepsize_state, n_evaluations
        zeros(dimension), nothing, nothing, Inf, Float64[],
        0, 0, 0, 0, 0, 0, OnlineStatsBase.Mean(), zeros(dimension),
        true, 0, :warming, nothing, NamedTuple[],
    )
end

@testset "cooperative checkpoint payload" begin
    chain = _chain(chain_index=3)

    @testset "excludes lpdf, carries the contract fields" begin
        p = cooperative_checkpoint_payload(chain)
        # The whole reason resume re-supplies lpdf: it may be a non-serializable
        # BridgeStan handle. It must never reach the payload.
        @test !haskey(p, :lpdf)
        @test p.schema_version == checkpoint_schema_version()
        @test p.sampler === :cooperative
        @test p.chain_index == 3
        @test p.status === :warming
        @test p.stuck_reason === nothing
        @test p.is_final == false          # alive chain is not final
        @test p.stop_reason === :running
        # Payload must survive a real round-trip, not just be constructible.
        mktempdir() do dir
            path = joinpath(dir, "p.jls")
            serialize(path, p)
            @test deserialize(path).chain_index == 3
        end
    end

    @testset "is_final / stop_reason track terminal status" begin
        chain.status = :done
        @test cooperative_checkpoint_payload(chain).is_final
        @test cooperative_checkpoint_payload(chain).stop_reason === :n_draws

        chain.status = :stuck
        chain.stuck_reason = :geometry_stall
        p = cooperative_checkpoint_payload(chain)
        @test p.is_final
        @test p.stop_reason === :stuck
        @test p.stuck_reason === :geometry_stall
        chain.status = :warming; chain.stuck_reason = nothing
    end

    @testset "paths use the stable chain index; nothing disables" begin
        @test _chain_checkpoint_paths(nothing, chain) === nothing
        d, w, l = _chain_checkpoint_paths("/tmp/run", chain)
        @test d == "/tmp/run/chain_3"
        @test basename(w) == "cp_window_$(chain.outer_counter).jls"
        @test basename(l) == "cp_latest.jls"
    end

    @testset "flush writes both files and they deserialize" begin
        mktempdir() do dir
            paths = _chain_checkpoint_paths(dir, chain)
            _flush_checkpoint!((paths, cooperative_checkpoint_payload(chain)))
            (d, w, l) = paths
            @test isfile(w) && isfile(l)
            @test deserialize(w).chain_index == 3
            @test deserialize(l).chain_index == 3
            # Atomic writer must leave no temp litter behind.
            @test sort(readdir(d)) == sort([basename(w), basename(l)])
        end
        @test _flush_checkpoint!(nothing) === nothing   # checkpointing off
    end
end

@testset "stuck_reason is the single source of truth for is_stuck" begin
    chain = _chain()
    # Fewer than min_windows checkpoints: not stuck, no reason.
    @test stuck_reason(chain) === nothing
    @test is_stuck(chain) == false

    # Divergence blow-up: the reason must name the pathology that fired, and
    # is_stuck must agree with it by construction rather than by duplication.
    chain.n_samples = 100
    chain.n_divergent_samples = 90
    push!(chain.checkpoints, (; status=:sampling, variance_cond=1.0))
    @test stuck_reason(chain; min_windows=1) === :divergence_blowup
    @test is_stuck(chain; min_windows=1)
end

@testset "run manifest / summary JSON" begin
    using WarmupHMC: _json_val, _json_object, _write_json,
                     write_run_manifest, write_run_summary

    @testset "JSON has no Inf/NaN — unset bounds must read as null, not a lie" begin
        # `target_ess` and `time_budget` default to Inf. Emitting `Inf` produces
        # invalid JSON that a strict parser rejects; emitting a big number would
        # claim a bound that was never set.
        @test _json_val(Inf) == "null"
        @test _json_val(-Inf) == "null"
        @test _json_val(NaN) == "null"
        @test _json_val(1000) == "1000"
        @test _json_val(nothing) == "null"
        @test _json_val(true) == "true"
        @test _json_val(:cooperative) == "\"cooperative\""
        @test _json_val(["a", "b"]) == "[\"a\",\"b\"]"
        # Xoshiro reprs contain no quotes, but escaping must hold regardless.
        @test _json_val("a\"b") == "\"a\\\"b\""
    end

    @testset "object shape" begin
        @test _json_object(["a" => 1, "b" => nothing]) == "{\"a\":1,\"b\":null}"
    end

    @testset "write is atomic and leaves no litter" begin
        mktempdir() do dir
            path = joinpath(dir, "run_manifest.json")
            _write_json(path, Pair{String,Any}["schema_version" => 1, "target_ess" => Inf])
            @test isfile(path)
            @test read(path, String) == "{\"schema_version\":1,\"target_ess\":null}"
            @test readdir(dir) == ["run_manifest.json"]
        # Integers must not be promoted to floats by a mixed literal — a consumer
        # parsing schema_version as an Int would break on `1.0`.
        @test occursin("\"schema_version\":1,", read(path, String))
        end
    end

    @testset "nothing disables both writers" begin
        @test write_run_manifest(nothing, nothing) === nothing
        @test write_run_summary(nothing, nothing, nothing) === nothing
    end
end
