@testitem "back_transform maps checkpoint draws into the model frame" setup=[ADBackend, Determinism] tags=[:checkpoint, :reparametrization] begin
    using WarmupHMC, Serialization, Random, LinearAlgebra, LogDensityProblems
    using WarmupHMC: back_transform, reparametrize!, restore_reparam_sources!,
                     reparam_sources, reparametrizer

    # A checkpoint's draws are in the sampler's SOURCE frame — `finalize_warmup!`
    # holds the back-transform, and a checkpoint never reaches it. `back_transform`
    # is the entry point that applies it to a payload a consumer deserialized
    # itself, so a running fit's partial results are available for a reparametrized
    # run and not only for a plain one.
    #
    # WHAT MAKES THE ABSENCE OF THIS DANGEROUS, and so what this file has to pin:
    # feeding source-frame draws to a constraining transform does not error and does
    # not produce NaN. It returns plausible, wrong numbers. So every assertion below
    # is about VALUES, and the mismatch guards are tested as hard as the happy path.

    # y[1] ~ N(0, 1) and y[2] ~ N(1000, 1) in MODEL coordinates — the same probe
    # `back_transform_frame.jl` uses, and for the same reason: the two frames are a
    # full 1000 apart by construction, so "which frame is this" is answered by
    # inspection with no threshold to pick.
    struct _CpProbe end
    LogDensityProblems.dimension(::_CpProbe) = 2
    LogDensityProblems.capabilities(::Type{_CpProbe}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::_CpProbe, y) = -(y[1]^2 + (y[2] - 1000.0)^2) / 2
    LogDensityProblems.logdensity_and_gradient(p::_CpProbe, y) =
        (LogDensityProblems.logdensity(p, y), [-y[1], -(y[2] - 1000.0)])

    # y[2] = 1000 + x[2] at c = 0: a constant location, unit scale, so the source
    # frame is standard normal in both coordinates and nothing here depends on the
    # sampler coping with awkward geometry.
    probe_rp() = ReparametrizedProblem(
        IndexedReparametrization([
            2 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), 1000.0, 0.0),
        ]),
        _CpProbe(), AutoForwardDiff(),
    )

    cfg = (; n_draws=200, monitor_ess=false, progress=nothing,
             init=(; position=zeros(2), squared_scale=ones(2)))

    # The documented rule for which draws a partial-results consumer takes.
    payload_draws(p) = isempty(p.posterior_position) ?
        get(p, :dropped_posterior_position, p.posterior_position) : p.posterior_position

    @testset "back_transform maps checkpoint draws into the model frame" begin

        @testset "the probe's two frames really are 1000 apart" begin
            # If this fails every assertion below is vacuous — it would be comparing
            # a frame against itself.
            _, y = reparametrizer(probe_rp())([0.25, -0.5])
            @test y[1] == 0.25
            @test y[2] == 999.5
        end

        # --- the snag, end to end ------------------------------------------------
        #
        # `nonlinear_adapt=false` pins the centering at the constructed c = 0, so
        # the source frame is known exactly: raw coordinate 2 is O(1), model
        # coordinate 2 is 1000 ± O(1). No adaptation path to reason about.
        @testset "a checkpoint's raw draws are source-frame; back_transform fixes that" begin
            dir = mktempdir()
            adaptive_warmup_mcmc(Xoshiro(7), probe_rp();
                                 nonlinear_adapt=false, checkpoint_dir=dir, cfg...)
            payload = deserialize(joinpath(dir, "cp_latest.jls"))
            raw = payload_draws(payload)
            @test size(raw, 2) > 0

            # The hazard itself: these look like ordinary draws and are not in the
            # model's frame. A consumer constraining them gets numbers ~1000 off.
            @test all(<(100), abs.(raw[2, :]))

            draws = back_transform(payload, probe_rp(), raw)
            @test size(draws) == size(raw)
            @test all(>(900), draws[2, :])
            @test draws[1, :] == raw[1, :]          # coordinate 1 is not reparametrized
            @test draws[2, :] ≈ raw[2, :] .+ 1000.0
        end

        @testset "it agrees with restore + reparametrize!, adaptation and all" begin
            # The pin that keeps the two readers of `reparam_sources` from drifting:
            # `back_transform` must equal what resume's restore path followed by the
            # finalization transform produces on the same payload. Run WITH
            # adaptation, so the payload's centering is whatever the sampler learned
            # rather than the one that was constructed.
            dir = mktempdir()
            adaptive_warmup_mcmc(Xoshiro(11), probe_rp();
                                 nonlinear_adapt=true, checkpoint_dir=dir, cfg...)
            payload = deserialize(joinpath(dir, "cp_latest.jls"))
            raw = payload_draws(payload)
            @test size(raw, 2) > 0

            expected = Matrix{Float64}(undef, size(raw))
            copyto!(expected, raw)
            reparametrize!(restore_reparam_sources!(probe_rp(), payload.reparam_sources),
                           expected)

            @test back_transform(payload, probe_rp(), raw) == expected
            # And it is the model frame, whatever the learned centering was.
            @test all(>(900), expected[2, :])
        end

        @testset "the current position transforms too, by the same rule" begin
            # `position_and_gradient.q` is the only thing a checkpoint carries when a
            # window restarted or recording has not started — and it is in the same
            # source frame, so it needs the same treatment.
            dir = mktempdir()
            adaptive_warmup_mcmc(Xoshiro(13), probe_rp();
                                 nonlinear_adapt=false, checkpoint_dir=dir, cfg...)
            payload = deserialize(joinpath(dir, "cp_init.jls"))
            q = payload.position_and_gradient.q

            mapped = back_transform(payload, probe_rp(), q)
            @test mapped isa AbstractVector
            @test length(mapped) == length(q)
            @test mapped == back_transform(payload, probe_rp(), reshape(collect(q), :, 1))[:, 1]
            @test mapped[2] ≈ q[2] + 1000.0
        end

        # --- it mutates nothing --------------------------------------------------
        @testset "neither the positions nor the log density are touched" begin
            dir = mktempdir()
            adaptive_warmup_mcmc(Xoshiro(7), probe_rp();
                                 nonlinear_adapt=false, checkpoint_dir=dir, cfg...)
            payload = deserialize(joinpath(dir, "cp_latest.jls"))
            raw = payload_draws(payload)
            before = copy(raw)

            rp = probe_rp()
            centerings_before = [s.c for (_, s) in reparam_sources(rp)]
            out = back_transform(payload, rp, raw)

            @test raw == before                     # the payload's own array survives
            @test out !== raw
            @test [s.c for (_, s) in reparam_sources(rp)] == centerings_before
        end

        # --- the empty case ------------------------------------------------------
        @testset "a plain log density gets a copy, so it is safe to call blind" begin
            dir = mktempdir()
            adaptive_warmup_mcmc(Xoshiro(3), _CpProbe(); checkpoint_dir=dir, cfg...)
            payload = deserialize(joinpath(dir, "cp_latest.jls"))
            raw = payload_draws(payload)
            @test isempty(payload.reparam_sources)

            out = back_transform(payload, _CpProbe(), raw)
            @test out == raw
            @test out !== raw                       # a copy, not the payload's array
        end

        # --- the mismatch guards -------------------------------------------------
        #
        # Each of these would otherwise be a silently different posterior rather
        # than an error, which is the failure mode this whole entry point exists to
        # close. They are the reason `back_transform` takes the payload rather than
        # just the centerings.
        @testset "a spec that cannot be the one that wrote the payload is refused" begin
            dir = mktempdir()
            adaptive_warmup_mcmc(Xoshiro(7), probe_rp();
                                 nonlinear_adapt=false, checkpoint_dir=dir, cfg...)
            payload = deserialize(joinpath(dir, "cp_latest.jls"))
            raw = payload_draws(payload)

            # Forgot the reparametrization entirely — exactly the shape of the bug
            # that is correct today only because `reparam_sources == []`.
            @test_throws ArgumentError back_transform(payload, _CpProbe(), raw)

            # Right count, wrong coordinates.
            other_rp = ReparametrizedProblem(
                IndexedReparametrization([
                    1 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0),
                                           1000.0, 0.0),
                ]),
                _CpProbe(), AutoForwardDiff(),
            )
            @test_throws ArgumentError back_transform(payload, other_rp, raw)

            # Positions that are not this payload's — including the classic
            # transposed matrix, where draws are rows instead of columns.
            @test_throws DimensionMismatch back_transform(payload, probe_rp(), permutedims(raw))
            @test_throws DimensionMismatch back_transform(payload, probe_rp(), raw[1:1, :])

            # Not a payload at all.
            @test_throws ArgumentError back_transform((; dimension=2), probe_rp(), raw)
        end

        # --- it is a payload contract, not an adaptive-sampler one ---------------
        @testset "cooperative payloads back-transform the same way" begin
            # `back_transform` reads exactly two keys — `dimension` and
            # `reparam_sources` — and all three samplers write both, so the entry
            # point is not adaptive-only. Cooperative is the cheap end-to-end
            # witness for that.
            dir = mktempdir()
            cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], probe_rp();
                n_cores=1, n_evaluations_budget=6000, nonlinear_adapt=false,
                checkpoint_dir=dir, init=(; position=zeros(2), squared_scale=ones(2)))
            payload = deserialize(joinpath(dir, "chain_1", "cp_latest.jls"))
            @test payload.sampler === :cooperative

            raw = payload_draws(payload)
            if size(raw, 2) > 0
                draws = back_transform(payload, probe_rp(), raw)
                @test all(<(100), abs.(raw[2, :]))
                @test all(>(900), draws[2, :])
            end
            # The current position is live at every boundary either way.
            @test back_transform(payload, probe_rp(), payload.position_and_gradient.q)[2] ≈
                  payload.position_and_gradient.q[2] + 1000.0
        end
    end
end
