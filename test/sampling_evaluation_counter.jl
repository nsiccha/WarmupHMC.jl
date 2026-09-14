# `sampling_evaluation_counter`: the exact retained-epoch sampling cost.
#
# `total_evaluation_counter` is the RUN total over every MCMC transition — the
# retained epoch's own step-size-adaptation transitions and every discarded
# restart epoch included — so `min ESS / total` understates sampling
# efficiency, and no exact sampling-only denominator could be reconstructed
# from the saved state (per-transition costs are gone; `steps_per_draw` mixes
# adaptation in). This file pins the dedicated counter that closes that gap:
# accumulated only for transitions actually appended to `posterior_position`,
# zeroed with the draws on a restart, returned and checkpointed alongside the
# run total.
using Serialization

@testset "sampling-only counter equals the run total without adaptation" begin
    # With `stepsize_adaptation_limit=0` there are no adaptation transitions:
    # every transition appends a draw, so the two counters must agree exactly.
    # `variance_cond_target=Inf` holds restarts off (a Gaussian's condition
    # number is finite), keeping this a single-epoch run.
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    result = adaptive_warmup_mcmc(Xoshiro(20260914), target;
        n_draws=200, stepsize_adaptation_limit=0, variance_cond_target=Inf,
        progress=nothing)
    @test size(result.posterior_position, 2) >= 200
    @test result.total_evaluation_counter > 0
    @test result.sampling_evaluation_counter == result.total_evaluation_counter
end

@testset "default adaptation transitions are excluded from the sampling counter" begin
    # Same single-epoch shape, but the first `stepsize_adaptation_limit`
    # transitions adapt (no draw appended): sampling is positive yet strictly
    # below the run total.
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    result = adaptive_warmup_mcmc(Xoshiro(20260915), target;
        n_draws=200, stepsize_adaptation_limit=50, variance_cond_target=Inf,
        progress=nothing)
    @test size(result.posterior_position, 2) >= 200
    @test 0 < result.sampling_evaluation_counter < result.total_evaluation_counter
end

@testset "run total excludes pre-transition setup evaluations" begin
    # `total_evaluation_counter` starts at zero in `init_state`, AFTER
    # Pathfinder, the initial step-size search, and the initial evaluation. A
    # `CountingPosterior` around the problem sees all of those, so it must
    # count strictly more than the run total reports.
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    (; n_evaluations, result) = WarmupHMC.count_and_time(target) do cp
        adaptive_warmup_mcmc(Xoshiro(20260916), cp;
            n_draws=200, variance_cond_target=Inf, progress=nothing)
    end
    @test 0 < result.sampling_evaluation_counter <= result.total_evaluation_counter
    @test result.total_evaluation_counter < n_evaluations
end

@testset "restart and resume accounting" begin
    # `variance_cond_target=1.0` restarts EVERY budget-exhausting window (the
    # condition number is always >= 1), so restarts, discarded draws, and
    # checkpoint boundaries are all exercised on a cheap Gaussian — no funnel
    # needed. (A well-scaled Gaussian at the default target would restart at
    # most once, before its first draw.)
    target = DiagGaussian([0.0, 0.5], [1.0, 1.5])
    dir = mktempdir()
    captured = NamedTuple[]
    result = adaptive_warmup_mcmc(Xoshiro(7), target;
        n_draws=300, n_evaluations=300, stepsize_adaptation_limit=10,
        variance_cond_target=1.0, checkpoint_dir=dir, progress=nothing,
        callback=(state, stage) -> begin
            if stage === :window
                push!(captured, (;
                    outer=state.outer_counter,
                    sampling=state.sampling_evaluation_counter,
                    total=state.total_evaluation_counter,
                    n=state.n_samples,
                ))
            end
            false
        end)

    # Several windows ran, and at least one of them restarted (empty recorder).
    @test length(captured) >= 2
    @test any(c -> c.n == 0, captured)

    # A restarted window's sampling cost was discarded with its draws ...
    @test all(c -> c.n == 0 ? c.sampling == 0 : c.sampling > 0, captured)

    # ... and every checkpoint carries exactly the live state's counter: the
    # write precedes the callback, so the captured value IS the payload's.
    for c in captured
        payload = Serialization.deserialize(joinpath(dir, "cp_window_$(c.outer).jls"))
        @test haskey(payload, :sampling_evaluation_counter)
        @test payload.sampling_evaluation_counter == c.sampling
        @test payload.total_evaluation_counter == c.total
    end

    # The final epoch always opens with `stepsize_adaptation_limit` adaptation
    # transitions, so the retained sampling cost is strictly below the total
    # even before counting the discarded epochs.
    @test size(result.posterior_position, 2) >= 300
    @test 0 < result.sampling_evaluation_counter < result.total_evaluation_counter

    @testset "resume continues the retained counter" begin
        resumed = adaptive_warmup_mcmc(Xoshiro(7), target;
            n_draws=500, n_evaluations=300, stepsize_adaptation_limit=10,
            variance_cond_target=1.0, checkpoint_dir=dir, resume=true,
            progress=nothing)
        @test size(resumed.posterior_position, 2) >= 500
        @test 0 < resumed.sampling_evaluation_counter <= resumed.total_evaluation_counter
    end

    @testset "legacy checkpoints without the key still resume" begin
        old = Serialization.deserialize(joinpath(dir, "cp_latest.jls"))
        legacy = NamedTuple(k => v for (k, v) in pairs(old)
                            if k != :sampling_evaluation_counter)
        @test !haskey(legacy, :sampling_evaluation_counter)
        path = joinpath(dir, "legacy_schema.jls")
        Serialization.serialize(path, legacy)
        # The counter restarts at zero at resume time (the pre-resume sampling
        # cost of the live epoch is unrecoverable and never estimated); the
        # resumed run's own appended transitions accumulate exactly from there.
        # (`cp_latest` at this point holds the 500-draw resume above, so 700
        # forces real continuation.)
        r = resume_warmup_mcmc(target, path;
            n_draws=700, stepsize_adaptation_limit=10)
        @test size(r.posterior_position, 2) >= 700
        @test 0 < r.sampling_evaluation_counter <= r.total_evaluation_counter
    end
end
