@testitem "halo recording rate" setup=[Targets, Determinism] tags=[:recording] begin
    using WarmupHMC, Random, LinearAlgebra, LogDensityProblems
    using WarmupHMC: RecordingPosterior2, LimitedRecorder2, record_leaf!,
                     finalize_leaf_recording!, record!, reset!

    # Guards for the HALO RECORDING RATE.
    #
    # `halo_position` / `halo_gradient` are a single shared pool, written by
    # `record!` during the NUTS traversal (`src/WrappedLogDensityProblems.jl`) and
    # read by BOTH adaptation consumers at a window restart
    # (`src/adaptive_warmup_mcmc.jl`):
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
    # proposal-weighted leaf per transition, collapsing the pool to ~1 state per
    # TRANSITION regardless of tree depth. The exact-proposal-weight change was
    # right; the rate collapse rode along uncaught because no test looked at the
    # pool size. Restored by `095efb0`, landed as `c8fed88` (2026-07-28); the two
    # assertions this file was written to fail now pass and are plain `@test`.
    #
    # NOTE ON THE WRITE PATH. `finalize_leaf_recording!` computes the exact marginal
    # proposal weights but deliberately does NOT write the halo — `record!` does,
    # once per leaf, reservoir-sampled one state per `thin` leaf evaluations. A
    # harness that drives `record_leaf!` + `finalize_leaf_recording!` and then reads
    # `halo_position` measures nothing; that is what this file did before the fix
    # landed, and it read as 0 rather than as an error.

    @testset "halo recording rate" begin

        # --- Unit level: what ONE trajectory contributes -------------------------
        # Drives the recorder directly with a synthetic NUTS leaf set, so the
        # measurement is exact and free of sampler nondeterminism.
        _recorder_for(dim, target, thin=1) = RecordingPosterior2(
            DiagGaussian(ones(dim)); rng=Xoshiro(1), recorder=LimitedRecorder2(target, thin),
        )

        # `record!` reads only `z.Q.q` / `z.Q.∇ℓq`, so a plain NamedTuple stands in
        # for the DynamicHMC phase point and this stays independent of its internals.
        _leaf(q, g) = (; Q = (; q, ∇ℓq = g))

        # Feed `n` leaves of one trajectory through both the weight bookkeeping and
        # the halo writer, exactly as `DynamicHMC.leaf` does, and return how many
        # halo columns they contributed. `dH = -0.5` for every non-initial leaf, so
        # all of them clear the `dH > log(1e-2)` acceptance filter and the count is
        # deterministic rather than a draw.
        function feed_leaves!(p, n; dim=3, seed=20260728)
            rng = Xoshiro(seed)
            before = size(p.halo_position, 2)
            for i in 1:n
                q, g = randn(rng, dim), randn(rng, dim)
                dH = i == 1 ? 0.0 : -0.5
                record_leaf!(p.leaves, q, g, dH)
                record!(p, _leaf(q, g); is_initial = i == 1, dH)
            end
            size(p.halo_position, 2) - before
        end

        function states_from_one_trajectory(depth; dim=3, target=100_000, thin=1)
            p = _recorder_for(dim, target, thin)
            n = feed_leaves!(p, 1 << depth; dim, seed=20260728 + depth)
            finalize_leaf_recording!(p, depth)
            n
        end

        depths = [1, 3, 5, 7]
        contributed = map(states_from_one_trajectory, depths)
        println("  halo states recorded per trajectory:")
        for (d, n) in zip(depths, contributed)
            println("    depth=$d  leaves=$(lpad(1 << d, 4))  recorded=$(lpad(n, 4))")
        end

        @testset "a trajectory never contributes more states than it has leaves" begin
            # An upper bound, not a rate pin — held before the fix too.
            for (d, n) in zip(depths, contributed)
                @test 1 <= n <= (1 << d)
            end
        end

        @testset "deeper trajectories contribute more states" begin
            # THE CONTRACT. A depth-7 trajectory visits 128 leaves and costs 127
            # gradient evaluations; a depth-1 trajectory visits 2 and costs 1. The
            # recorder this file was written against kept one leaf per TRANSITION and
            # returned 1 for both, so the halo's sample size was set by the
            # transition count and was independent of the work actually done.
            #
            # Stated as strict growth rather than `== 2^depth` so it stays correct
            # under either fix semantics (all leaves, or only positive-weight ones).
            @test last(contributed) > first(contributed)
            # At `thin=1` every accepted leaf is retained, so the count is exactly
            # the leaf count less the initial state. This is the rate itself, pinned.
            for (d, n) in zip(depths, contributed)
                @test n == (1 << d) - 1
            end
        end

        @testset "`thin` sets the rate: one state per `thin` leaf evaluations" begin
            # The mechanism `34ce034` deleted and `095efb0` restored. Warm-up sets
            # `thin = n_evaluations ÷ recording_target` and recomputes it when the
            # window budget doubles, which is what makes one window fill the ring.
            # Within each block of `thin` leaves the recorder reservoir-samples, so
            # WHICH state lands is random but HOW MANY is not: exactly one per
            # complete block.
            for thin in (1, 2, 4, 8)
                p = _recorder_for(3, 100_000, thin)
                n = feed_leaves!(p, 128; seed=99)
                # One state per complete block — except the block holding the initial
                # state, which has no acceptance statistic and so is never written.
                # At `thin >= 2` that block still contains `thin-1` writable leaves;
                # at `thin == 1` the initial state has the block to itself and the
                # count is one short.
                expected = 128 ÷ thin - (thin == 1)
                println("    thin=$thin over 128 leaves → $n states (expected $expected)")
                @test n == expected
            end
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
                # target` gradient evaluations, so the `thin`-based recorder fills it
                # exactly. Under one-leaf-per-transition it held one state per
                # transition — `window_evals / (states per trajectory)`, a fraction
                # of the ring — which is the regression this pins against.
                @test w.halo == w.target
            end
        end

        @testset "the ring never overfills, and a restart empties it" begin
            for w in observable
                @test w.halo <= w.target
            end
            # 20 trajectories × 4 leaves = 80 writes into an 8-slot ring: the ring
            # must wrap by overwriting, never grow past `target`.
            p = _recorder_for(3, 8)
            for t in 1:20
                reset!(p.leaves)
                feed_leaves!(p, 4; seed=5 + t)
                finalize_leaf_recording!(p, 2)
            end
            @test size(p.halo_position, 2) == 8
            @test size(p.halo_position, 2) == size(p.halo_gradient, 2)
            reset!(p)
            @test size(p.halo_position, 2) == 0
            @test size(p.halo_gradient, 2) == 0
        end
    end
end
