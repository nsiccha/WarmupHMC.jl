using WarmupHMC, Test, Random, LinearAlgebra, LogDensityProblems
using WarmupHMC: RecordingPosterior2, LimitedRecorder2, record_leaf!,
                 finalize_leaf_recording!, reset!

include(joinpath(@__DIR__, "targets.jl"))

# Guards for the HALO RECORDING RATE.
#
# `halo_position` / `halo_gradient` are a single shared pool, written only by
# `_store_leaf!` (`src/WrappedLogDensityProblems.jl`) and read by BOTH adaptation
# consumers at a window restart (`src/adaptive_warmup_mcmc.jl`):
#
#   * `find_reparametrization!`      — the nonlinear centering `argmin`
#   * `argmin(map(update_loss!, …))` — the linear metric selection
#
# So the number of columns in that pool is the sample size of every adaptation
# decision the sampler makes. It is a behavioural contract, and byte-identity
# against a frozen baseline provably cannot see it: a baseline regenerated after
# a rate change simply pins the new rate.
#
# HISTORY. `34ce034` (2026-07-19) replaced the recorder's `thin` mechanism —
# `thin = n_evaluations ÷ recording_target`, recomputed per window so that ONE
# window filled the entire `recording_target`-slot ring — with one
# proposal-weighted leaf per transition. The exact-proposal-weight change was
# right; the rate collapse rode along uncaught because no test looked at the
# pool size. `todo 1fqvovq` on the parent queue carries the source fix
# (retain all leaves with their exact weights).
#
# The two `@test_broken` assertions below are the regression, stated as a
# contract. They are deliberately NOT pinning current behaviour: when the fix
# lands, `@test_broken` reports `Unbroken` — a FAILURE — which forces whoever
# lands it to promote them to `@test`. That handshake is the point.

@testset "halo recording rate" begin

    # --- Unit level: what ONE trajectory contributes -------------------------
    # Drives the recorder directly with a synthetic NUTS leaf set, so the
    # measurement is exact and free of sampler nondeterminism.
    _recorder_for(dim, target) = RecordingPosterior2(
        DiagGaussian(ones(dim)); rng=Xoshiro(1), recorder=LimitedRecorder2(target),
    )

    # Record one full depth-`depth` trajectory and return how many halo columns
    # it contributed. `dH[1]` is the initial state; a valid tree of depth
    # `depth` supplies `2^depth` leaves in total.
    function states_from_one_trajectory(depth; dim=3, target=100_000)
        p = _recorder_for(dim, target)
        rng = Xoshiro(20260728 + depth)
        before = size(p.halo_position, 2)
        for i in 1:(1 << depth)
            record_leaf!(p.leaves, randn(rng, dim), randn(rng, dim),
                         i == 1 ? 0.0 : -abs(randn(rng)))
        end
        finalize_leaf_recording!(p, depth)
        size(p.halo_position, 2) - before
    end

    depths = [1, 3, 5, 7]
    contributed = map(states_from_one_trajectory, depths)
    println("  halo states recorded per trajectory:")
    for (d, n) in zip(depths, contributed)
        println("    depth=$d  leaves=$(lpad(1 << d, 4))  recorded=$(lpad(n, 4))")
    end

    @testset "a trajectory never contributes more states than it has leaves" begin
        # Holds before and after the fix — an upper bound, not a rate pin.
        for (d, n) in zip(depths, contributed)
            @test 1 <= n <= (1 << d)
        end
    end

    @testset "deeper trajectories contribute more states" begin
        # THE CONTRACT. A depth-7 trajectory visits 128 leaves and costs 127
        # gradient evaluations; a depth-1 trajectory visits 2 and costs 1. A
        # recorder that keeps one leaf per TRANSITION returns 1 for both, so the
        # halo's sample size is set by the transition count and is independent of
        # the work actually done — which is the regression.
        #
        # Stated as strict growth rather than `== 2^depth` so it stays correct
        # under either fix semantics (all leaves, or only positive-weight ones).
        @test_broken last(contributed) > first(contributed)
    end

    # --- Integration level: what ONE WINDOW contributes ----------------------
    # `recording_target` sizes the ring; `n_evaluations` is the window's
    # gradient-evaluation budget. The original design sized the two against each
    # other so a window that spends its budget fills the ring exactly once.
    RECORDING_TARGET = 100
    N_EVALUATIONS = 400          # 4x the ring, so the budget cannot be the binder

    windows = NamedTuple[]
    callback = (state, stage) -> begin
        push!(windows, (; stage,
            outer = state.outer_counter,
            halo = size(state.recording_lpdf.halo_position, 2),
            target = state.recording_lpdf.recorder.target,
            evals = state.total_evaluation_counter,
            transitions = state.total_transition_counter,
            restart = state.restart))
        nothing
    end
    BLAS.set_num_threads(1)
    adaptive_warmup_mcmc(
        Xoshiro(20260718), DiagGaussian(exp.(range(-1.5, 1.5, 6)));
        n_draws=200, n_evaluations=N_EVALUATIONS, recording_target=RECORDING_TARGET,
        progress=nothing, callback=callback,
    )

    # Per-window deltas. A window that RESTARTED has already had its pool emptied
    # by `reset!(recording_lpdf)` before the callback fires, so only a
    # non-restarting window exposes the pool at the size the consumers saw.
    prev_evals = 0
    observable = NamedTuple[]
    for w in windows
        window_evals = w.evals - prev_evals
        prev_evals = w.evals
        w.stage === :window || continue
        push!(observable, (; w.outer, window_evals, w.halo, w.target, w.restart))
    end
    println("  halo pool per window (recording_target=$RECORDING_TARGET, n_evaluations=$N_EVALUATIONS):")
    for w in observable
        println("    window $(w.outer): gradient evals=$(lpad(w.window_evals, 5))  " *
                "halo=$(lpad(w.halo, 4))/$(w.target)  restart=$(w.restart)")
    end

    @testset "a window that can fill the ring, does" begin
        # Pick the first window that (a) spent at least `recording_target`
        # gradient evaluations — so the budget is not the binding constraint —
        # and (b) did not restart, so its pool survived to be observed.
        candidates = filter(w -> !w.restart && w.window_evals >= RECORDING_TARGET, observable)
        @test !isempty(candidates)        # otherwise nothing below is verified
        if !isempty(candidates)
            w = first(candidates)
            println("    first fillable window: $(w.outer) — spent $(w.window_evals) gradient " *
                    "evaluations for a $(w.target)-slot ring, filled $(w.halo)")
            # The ring has `target` slots and the window spent `window_evals >=
            # target` gradient evaluations. Under the original `thin`-based
            # recorder the pool would hold exactly `target` states. Under
            # one-leaf-per-transition it holds one per transition, which is
            # `window_evals / (states per trajectory)` — a fraction of the ring.
            @test_broken w.halo == w.target
        end
    end

    @testset "the ring never overfills, and a restart empties it" begin
        for w in observable
            @test w.halo <= w.target
        end
        p = _recorder_for(3, 8)
        rng = Xoshiro(5)
        for _ in 1:20
            reset!(p.leaves)
            for i in 1:4
                record_leaf!(p.leaves, randn(rng, 3), randn(rng, 3), i == 1 ? 0.0 : -abs(randn(rng)))
            end
            finalize_leaf_recording!(p, 2)
        end
        @test size(p.halo_position, 2) <= 8
        @test size(p.halo_position, 2) == size(p.halo_gradient, 2)
        reset!(p)
        @test size(p.halo_position, 2) == 0
        @test size(p.halo_gradient, 2) == 0
    end
end
