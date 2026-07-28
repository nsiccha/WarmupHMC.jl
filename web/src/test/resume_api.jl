# Tests for the unified resume API: `checkpoint_dir` + `resume=` / `overwrite=` on
# the samplers themselves, and the checkpoint payload's reduction to PURE SAMPLER
# STATE (config now comes from the resuming call, which is what makes
# resume-and-extend possible).
#
# Run this item alone with
#   -- --file=resume_api.jl

@testitem "the unified resume API" setup=[Determinism] tags=[:sampler] begin
    using WarmupHMC, LogDensityProblems, LinearAlgebra, Random
    using Serialization: deserialize

    struct MvNormalLP; P::Matrix{Float64}; end
    LogDensityProblems.dimension(p::MvNormalLP) = size(p.P, 1)
    LogDensityProblems.capabilities(::Type{<:MvNormalLP}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(p::MvNormalLP, x) = -0.5 * dot(x, p.P, x)
    LogDensityProblems.logdensity_and_gradient(p::MvNormalLP, x) = (-0.5*dot(x,p.P,x), -(p.P*x))
    function build_target(dim, seed=123)
        rng = Xoshiro(seed); L = LowerTriangular(randn(rng, dim, dim)) + Diagonal(fill(2.0, dim))
        MvNormalLP(inv(Symmetric(L*L')))
    end

    lpdf = build_target(3)
    # An explicit `init` NamedTuple short-circuits `initialize_mcmc`, so these tests
    # never touch Pathfinder and so never need an AD backend — which is why this
    # item lists no `ADBackend` setup. Same trick the cooperative tests use.
    _init(l) = (; position=zeros(LogDensityProblems.dimension(l)),
                 squared_scale=Matrix(inv(Symmetric(l.P))))
    cfg = (; n_draws=200, nonlinear_adapt=false, monitor_ess=false, init=_init(lpdf))
    mktempdir() do d
        @testset "resume=/overwrite= on the sampler" begin
            # 1. fresh run into an empty dir needs no flag
            a = adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, cfg...)
            @test size(a.posterior_position, 2) >= 200

            # 2. payload is pure state: no config, no dead `stage`
            p = deserialize(joinpath(d, "cp_latest.jls"))
            for k in (:n_draws, :stepsize_adaptation_limit, :variance_cond_target,
                      :linear_restart_source, :linear_trajectory_weighting,
                      :linear_metric_fallback,
                      :nonlinear_adapt, :monitor_ess, :recording_target, :kwargs,
                      :algorithm, :stepsize_adaptation, :stage)
                @test !hasproperty(p, k)
            end
            @test p.sampler === :adaptive
            @test p.schema_version == 2
            @test hasproperty(p, :position_and_gradient)   # state still there
            @test hasproperty(p, :linear_recorder)

            # 3. re-running into a non-empty dir refuses to guess
            @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, cfg...)
            @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf;
                checkpoint_dir=d, resume=true, overwrite=true, cfg...)

            # 4. RESUME-AND-EXTEND: config comes from the call, so a larger n_draws keeps going
            b = adaptive_warmup_mcmc(Xoshiro(1), lpdf;
                checkpoint_dir=d, resume=true, n_draws=500, nonlinear_adapt=false,
                monitor_ess=false, init=_init(lpdf))
            @test size(b.posterior_position, 2) >= 500

            # 5. overwrite starts fresh
            c = adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, overwrite=true, cfg...)
            @test size(c.posterior_position, 2) >= 200
            # ...and reproduces the original run exactly
            @test c.posterior_position == a.posterior_position

            # 6. flags without a dir are meaningless
            @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; resume=true, cfg...)
        end
    end

    mktempdir() do d
        @testset "masquerade guard + validation" begin
            adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, cfg...)
            path = joinpath(d, "cp_latest.jls")
            # dimension mismatch is caught
            @test_throws DimensionMismatch WarmupHMC.restore_state(
                deserialize(path), build_target(5), nothing)
            # a cooperative-tagged payload is refused by the adaptive reader
            p = deserialize(path)
            coop = merge(NamedTuple(pairs(p)), (; sampler=:cooperative))
            @test_throws ArgumentError WarmupHMC.restore_state(coop, lpdf, nothing)
            # pre-tag checkpoints (no `sampler` field) still read as adaptive
            @test WarmupHMC.checkpoint_sampler((; a=1)) === :adaptive
            # recording_target cannot change on resume
            @test_throws ArgumentError WarmupHMC.restore_state(p, lpdf, nothing; recording_target=7)
        end
    end

    mktempdir() do d
        @testset "opt-in transformed estimator resumes exactly" begin
            weighted_cfg = (;
                n_draws=160, n_evaluations=80, stepsize_adaptation_limit=12,
                nonlinear_adapt=false, monitor_ess=false, init=_init(lpdf),
                linear_restart_source=:nuts_weighted,
                linear_trajectory_weighting=:stepsize,
            )
            straight = adaptive_warmup_mcmc(Xoshiro(17), lpdf; weighted_cfg...)
            stop_after_first_window = (state, stage) ->
                stage === :window && state.outer_counter == 1
            adaptive_warmup_mcmc(
                Xoshiro(17), lpdf; checkpoint_dir=d,
                callback=stop_after_first_window, weighted_cfg...,
            )
            resumed = adaptive_warmup_mcmc(
                Xoshiro(999), lpdf; checkpoint_dir=d, resume=true, weighted_cfg...,
            )
            @test resumed.posterior_position == straight.posterior_position
            @test resumed.posterior_gradient == straight.posterior_gradient
            @test resumed.total_evaluation_counter == straight.total_evaluation_counter
            @test resumed.scale_changes == straight.scale_changes
            @test resumed.linear_metric_fallbacks == straight.linear_metric_fallbacks

            payload = deserialize(joinpath(d, "cp_latest.jls"))
            @test payload.linear_recorder.source === :nuts_weighted
            @test payload.linear_recorder.trajectory_weighting === :stepsize
            inherited = WarmupHMC.restore_state(payload, lpdf, nothing;
                linear_restart_source=nothing, linear_trajectory_weighting=nothing,
                nonlinear_adapt=false)
            @test inherited.linear_restart_source === :nuts_weighted
            @test inherited.linear_trajectory_weighting === :stepsize
            @test inherited.transformed_adaptation.weight ==
                payload.linear_recorder.adaptation.weight

            changed = WarmupHMC.restore_state(payload, lpdf, nothing;
                linear_restart_source=:nuts_weighted,
                linear_trajectory_weighting=:unit, nonlinear_adapt=false)
            @test changed.transformed_adaptation.weight == 0
        end
    end

    mktempdir() do d
        @testset "cooperative run-dir guard" begin
            cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
                n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, init=_init(lpdf))
            @test !isempty(filter(f -> startswith(f, "chain_"), readdir(d)))
            # a second run into the same dir would interleave — refuse
            @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
                n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, init=_init(lpdf))
            # resume is honestly reported as unsupported, not silently ignored
            @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
                n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, resume=true, init=_init(lpdf))
            # overwrite clears it
            cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
                n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, overwrite=true, init=_init(lpdf))
        end
    end
end
