@testitem "checkpoints keep the draws a restart discards" setup=[Determinism] tags=[:checkpoint] begin
    using WarmupHMC, Serialization, Random, LinearAlgebra, LogDensityProblems
    using WarmupHMC: checkpoint_payload, cooperative_checkpoint_payload,
                     clustered_checkpoint_payload, checkpoint_schema_version

    # A restarting warm-up window empties the recorder, and the checkpoint is written
    # AFTER that reset — so before the `dropped_*` keys, such a checkpoint paired
    # complete resume state with ZERO retained draws, and a consumer materializing a
    # long run's partial results got nothing. These tests pin the preservation.

    # Neal's funnel with an analytic gradient: no AD backend, and its marginal-scale
    # condition number stays above `variance_cond_target` for several windows, so
    # restarts keep happening AFTER draws have accumulated (a well-scaled Gaussian
    # restarts once, before its first draw, and never exercises this at all).
    struct _DropFunnel; d::Int; end
    LogDensityProblems.dimension(f::_DropFunnel) = f.d
    LogDensityProblems.capabilities(::Type{<:_DropFunnel}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(f::_DropFunnel, x) = begin
        v = x[1]; z = @view x[2:end]
        -abs2(v / 3) / 2 - sum(abs2, z) * exp(-v) / 2 - (f.d - 1) * v / 2
    end
    LogDensityProblems.logdensity_and_gradient(f::_DropFunnel, x) = begin
        v = x[1]; z = @view x[2:end]
        g = similar(x)
        g[1] = -v / 9 + sum(abs2, z) * exp(-v) / 2 - (f.d - 1) / 2
        g[2:end] .= -z .* exp(-v)
        (LogDensityProblems.logdensity(f, x), g)
    end

    @testset "checkpoints keep the draws a restart discards" begin
        d = 10
        lpdf = _DropFunnel(d)
        dir = mktempdir()
        adaptive_warmup_mcmc(Random.Xoshiro(11), lpdf;
            n_draws=600, n_evaluations=4000, stepsize_adaptation_limit=20,
            checkpoint_dir=dir, init=(; position=zeros(d), squared_scale=ones(d)))

        windows = sort(filter(f -> startswith(f, "cp_window_"), readdir(dir)),
                       by = f -> parse(Int, match(r"cp_window_(\d+)", f).captures[1]))
        @test !isempty(windows)
        payloads = [deserialize(joinpath(dir, f)) for f in windows]

        @testset "a restarting window checkpoints its draws instead of nothing" begin
            # The regression: at least one checkpoint has an empty live recorder
            # (its window restarted) yet still carries the draws that restart threw
            # away. Before the fix this file's run produced a checkpoint with both
            # empty, and that was the newest file on disk for the whole first epoch.
            restarted = filter(p -> size(p.posterior_position, 2) == 0, payloads)
            @test !isempty(restarted)
            @test any(p -> size(p.dropped_posterior_position, 2) > 0, restarted)
            @test all(p -> size(p.dropped_posterior_position, 2) > 0, restarted)
        end

        @testset "positions and gradients stay paired and finite" begin
            for p in payloads
                @test size(p.dropped_posterior_gradient) == size(p.dropped_posterior_position)
                @test size(p.dropped_posterior_position, 1) == d
                @test all(isfinite, p.dropped_posterior_position)
                @test all(isfinite, p.dropped_posterior_gradient)
                # Dropped divergences count divergent DRAWS, so they can never exceed
                # the dropped draws themselves.
                @test p.dropped_n_divergent_samples <= size(p.dropped_posterior_position, 2)
            end
        end

        @testset "the keys are ADDITIVE — schema version does not move" begin
            # Bruno (and any other consumer) errors hard when a payload's
            # `schema_version` exceeds its own constant, so an additive key must not
            # bump it. Consumers read the new keys with `get(payload, key, default)`.
            @test all(p -> p.schema_version == checkpoint_schema_version(), payloads)
            @test checkpoint_schema_version() == 2
        end

        @testset "inert on resume, and old payloads still resume" begin
            r = adaptive_warmup_mcmc(Random.Xoshiro(11), lpdf;
                n_draws=800, n_evaluations=4000, stepsize_adaptation_limit=20,
                checkpoint_dir=dir, resume=true)
            @test size(r.posterior_position, 2) >= 800

            # A checkpoint written BEFORE these keys existed has none of them; resume
            # must not reach for a field that is not there.
            old = deserialize(joinpath(dir, "cp_latest.jls"))
            legacy = NamedTuple(k => v for (k, v) in pairs(old)
                                if !startswith(String(k), "dropped_"))
            @test !haskey(legacy, :dropped_posterior_position)
            path = joinpath(dir, "legacy_schema.jls")
            serialize(path, legacy)
            r2 = resume_warmup_mcmc(lpdf, path; n_draws=900, stepsize_adaptation_limit=20)
            @test size(r2.posterior_position, 2) >= 900
        end
    end

    @testset "the cooperative and clustered payloads carry the same keys" begin
        # The shared checkpoint contract is what lets a consumer read any sampler's
        # payload without a per-sampler branch; these keys are part of it.
        for k in (:dropped_posterior_position, :dropped_posterior_gradient,
                  :dropped_n_divergent_samples)
            @test hasfield(WarmupHMC.CooperativeChain, k)
            @test hasfield(WarmupHMC.ClusteredChain, k)
        end
    end
end
