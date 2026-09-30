# Crash-resume for the cooperative sampler: `restore_cooperative_chain` round-trips
# a chain through its on-disk payload, and `resume=true` continues a crashed run.
#
# The guarantee under test is decision `y72yij` option A — per-chain
# determinism, nondeterministic pool. So these tests assert per-chain trajectory
# identity (a restored chain stepped with its restored RNG reproduces the
# uninterrupted run exactly) plus pool-level statistical validity — never
# whole-run byte-identity, which the scheduler's thread timing cannot promise.
#
# Run this item alone with
#   -- --file=cooperative_resume.jl

@testitem "cooperative crash-resume" setup=[Determinism] tags=[:checkpoint] begin
    using WarmupHMC, Serialization, Random, LinearAlgebra, LogDensityProblems
    using WarmupHMC: CooperativeChain, cooperative_chain, advance_window!,
                     cooperative_checkpoint_payload, restore_cooperative_chain,
                     _read_chain_payload, _chain_dir_indices

    struct _Gauss3 end
    LogDensityProblems.dimension(::_Gauss3) = 3
    LogDensityProblems.capabilities(::Type{_Gauss3}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::_Gauss3, x) = -sum(abs2, x) / 2
    LogDensityProblems.logdensity_and_gradient(::_Gauss3, x) = (-sum(abs2, x) / 2, -x)

    struct _Gauss2 end
    LogDensityProblems.dimension(::_Gauss2) = 2

    # An explicit `init` NamedTuple short-circuits `initialize_mcmc`, so chain
    # construction never touches Pathfinder and needs no AD backend (same trick
    # as resume_api.jl). Full-matrix `squared_scale`: the diagonal/small form
    # fails this fixture (todo 1g03pcl).
    _init() = (; position=zeros(3), squared_scale=Matrix{Float64}(I, 3, 3) * 2.0)
    _chain(seed=1; kwargs...) = cooperative_chain(Xoshiro(seed), _Gauss3();
        chain_index=1, nonlinear_adapt=false, init=_init(), kwargs...)

    _roundtrip(p) = mktempdir() do dir
        path = joinpath(dir, "cp_latest.jls")
        open(io -> serialize(io, p), path, "w")
        deserialize(path)
    end

    @testset "restore round-trips a live chain" begin
        chain = _chain()
        advance_window!(chain)
        @test chain.outer_counter == 1
        n_before = size(chain.recording_lpdf.posterior_position, 2)

        p = _roundtrip(cooperative_checkpoint_payload(chain))
        restored = restore_cooperative_chain(p, _Gauss3(); nonlinear_adapt=false)

        # Learned state comes back exactly; the RNG bit-for-bit (its four words).
        @test restored.chain_index == 1
        @test (restored.rng.s0, restored.rng.s1, restored.rng.s2, restored.rng.s3) ==
              (chain.rng.s0, chain.rng.s1, chain.rng.s2, chain.rng.s3)
        @test restored.position_and_gradient.q == chain.position_and_gradient.q
        @test restored.position_and_gradient.∇ℓq == chain.position_and_gradient.∇ℓq
        @test restored.active_transformation == chain.active_transformation
        @test restored.stepsize == chain.stepsize
        @test restored.n_evaluations == chain.n_evaluations
        @test restored.variance_cond == chain.variance_cond
        @test restored.scale_changes == chain.scale_changes
        @test restored.total_evaluation_counter == chain.total_evaluation_counter
        @test restored.outer_counter == chain.outer_counter
        @test restored.current_transition_counter == chain.current_transition_counter
        @test restored.total_transition_counter == chain.total_transition_counter
        @test restored.n_divergent == chain.n_divergent
        @test restored.n_divergent_samples == chain.n_divergent_samples
        @test restored.ess == chain.ess
        @test restored.restart == chain.restart
        @test restored.n_samples == n_before == chain.n_samples
        @test restored.status == chain.status
        # `isequal`, not `==`: the log holds `min_ess = NaN` (no ESS monitoring here).
        @test isequal(restored.checkpoints, chain.checkpoints)
        @test restored.recording_lpdf.recorder.target == chain.recording_lpdf.recorder.target
        @test Matrix(restored.recording_lpdf.posterior_position) ==
              Matrix(chain.recording_lpdf.posterior_position)
        @test restored.nonlinear_recorder.mode == chain.nonlinear_recorder.mode

        # The per-chain guarantee: one more window on each is bit-identical.
        advance_window!(chain)
        advance_window!(restored)
        @test Matrix(restored.recording_lpdf.posterior_position) ==
              Matrix(chain.recording_lpdf.posterior_position)
        @test Matrix(restored.recording_lpdf.posterior_gradient) ==
              Matrix(chain.recording_lpdf.posterior_gradient)
        @test restored.n_samples == chain.n_samples
        @test restored.status == chain.status
        @test restored.stepsize == chain.stepsize
        @test restored.total_evaluation_counter == chain.total_evaluation_counter
        @test restored.outer_counter == chain.outer_counter
    end

    @testset "resume-and-extend flips :done back to :sampling" begin
        chain = _chain(; n_draws=5)
        while advance_window!(chain) != :done
        end
        @test chain.status === :done
        p = _roundtrip(cooperative_checkpoint_payload(chain))

        # Same `n_draws`: stays done — the scheduler will skip it.
        same = restore_cooperative_chain(p, _Gauss3(); n_draws=5, nonlinear_adapt=false)
        @test same.status === :done
        # Larger `n_draws`: back to sampling, and it collects past the old cap.
        extended = restore_cooperative_chain(p, _Gauss3(); n_draws=50, nonlinear_adapt=false)
        @test extended.status === :sampling
        advance_window!(extended)
        @test size(extended.recording_lpdf.posterior_position, 2) > 5
    end

    @testset "config rules: inherit, refuse, rebuild" begin
        chain = _chain(; recording_target=64)
        advance_window!(chain)
        p = _roundtrip(cooperative_checkpoint_payload(chain))

        # Omitted `recording_target` inherits the checkpoint's ring size.
        @test restore_cooperative_chain(p, _Gauss3(); nonlinear_adapt=false
            ).recording_lpdf.recorder.target == 64
        # An explicit different one is refused — the ring is persisted state.
        @test_throws ArgumentError restore_cooperative_chain(p, _Gauss3();
            nonlinear_adapt=false, recording_target=128)
        # An explicit different evidence mode rebuilds the accumulator.
        rebuilt = restore_cooperative_chain(p, _Gauss3();
            nonlinear_adapt=false, nonlinear_evidence=:all_good_leaves)
        @test rebuilt.nonlinear_recorder.mode === :all_good_leaves
        # Init-only knobs are accepted and ignored, so the resuming call can
        # repeat the original verbatim.
        verbatim = restore_cooperative_chain(p, _Gauss3(); nonlinear_adapt=false,
            n_evaluations=1, init=_init(), pathfinder_kw=(; ndraws=5))
        @test verbatim.n_evaluations == chain.n_evaluations
    end

    @testset "masquerade, schema, and dimension guards" begin
        chain = _chain()
        advance_window!(chain)
        p = cooperative_checkpoint_payload(chain)

        # A foreign sampler's tag is refused loudly, not misread.
        @test_throws ArgumentError restore_cooperative_chain(
            merge(p, (; sampler=:adaptive)), _Gauss3(); nonlinear_adapt=false)
        # A newer schema means changed semantics — refuse, don't misread.
        @test_throws ArgumentError restore_cooperative_chain(
            merge(p, (; schema_version=999)), _Gauss3(); nonlinear_adapt=false)
        # Wrong dimension is refused.
        @test_throws DimensionMismatch restore_cooperative_chain(
            p, _Gauss2(); nonlinear_adapt=false)
    end

    @testset "chain-dir readers" begin
        mktempdir() do d
            mkpath(joinpath(d, "chain_2"))
            mkpath(joinpath(d, "chain_10"))
            touch(joinpath(d, "chain_foo"))   # junk never kills a resume
            touch(joinpath(d, "run_manifest.json"))
            @test _chain_dir_indices(d) == [2, 10]
            @test isnothing(_read_chain_payload(d, 2))  # empty dir: no checkpoint yet
            @test isnothing(_read_chain_payload(d, 7))  # missing dir likewise
        end
        mktempdir() do d
            # Crash between the window write and the latest-pointer overwrite:
            # the newest window file restores with zero windows lost.
            chain = _chain()
            advance_window!(chain)
            p = cooperative_checkpoint_payload(chain)
            cdir = joinpath(d, "chain_1")
            mkpath(cdir)
            open(io -> serialize(io, p), joinpath(cdir, "cp_window_1.jls"), "w")
            @test _read_chain_payload(d, 1).window == 1
        end
    end

    @testset "crash-resume end to end" begin
        mktempdir() do d
            # A short run that stops on the eval budget with live chains, then
            # the summary is deleted: exactly the on-disk state of a crash
            # before finalize (checkpoints written, run not finalized).
            # `n_cores=1`: one worker, so scheduling is deterministic.
            part1 = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                n_cores=1, n_evaluations_budget=2500, nonlinear_adapt=false,
                checkpoint_dir=d, init=_init())
            @test isfile(joinpath(d, "run_summary.json"))
            @test part1.total_evaluation_counter >= 2500
            @test any(r -> r.status === :sampling || r.status === :warming, part1.results)
            rm(joinpath(d, "run_summary.json"))

            resumed = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                n_cores=1, n_evaluations_budget=6000, nonlinear_adapt=false,
                checkpoint_dir=d, resume=true, init=_init())
            # The run continued rather than restarted: membership preserved,
            # eval budget cumulative across the crash, summary rewritten.
            @test resumed.n_started == part1.n_started
            @test resumed.total_evaluation_counter >= 6000
            @test resumed.total_evaluation_counter > part1.total_evaluation_counter
            @test isfile(joinpath(d, "run_summary.json"))
            # Pool-level validity: draws exist, statuses are terminal-or-live,
            # every chain kept its identity.
            @test sum(r -> size(r.posterior_position, 2), resumed.results) > 0
            @test all(r -> r.status in (:warming, :sampling, :done, :stuck), resumed.results)
            @test sort!([r.chain_index for r in resumed.results]) ==
                  collect(1:resumed.n_started)
            # The manifest is write-once: resume kept the original's criteria.
            manifest = read(joinpath(d, "run_manifest.json"), String)
            @test occursin("\"n_evaluations_budget\":2500", manifest)
        end
    end

    @testset "time budget counts live time only" begin
        mktempdir() do d
            cooperative_warmup_mcmc([Xoshiro(1)], _Gauss3();
                n_cores=1, n_evaluations_budget=1500, nonlinear_adapt=false,
                checkpoint_dir=d, init=_init())
            rm(joinpath(d, "run_summary.json"))
            sleep(2)   # the outage: downtime must not eat the resumed budget
            resumed = cooperative_warmup_mcmc([Xoshiro(1)], _Gauss3();
                n_cores=1, n_evaluations_budget=4000, time_budget=30.0,
                nonlinear_adapt=false, checkpoint_dir=d, resume=true, init=_init())
            # Without a re-based clock the 2 s outage would already show up here.
            @test resumed.elapsed < 2.0
            @test resumed.total_evaluation_counter >= 4000
        end
    end

    @testset "a slot with no checkpoint starts fresh in place" begin
        mktempdir() do d
            # Small `n_draws`, generous budget: chain 1 finishes, chain 2 starts
            # and finishes — a deterministic two-chain run on one worker.
            part1 = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                n_cores=1, n_draws=20, n_evaluations_budget=20000, nonlinear_adapt=false,
                checkpoint_dir=d, init=_init())
            @test part1.n_started == 2
            rm(joinpath(d, "run_summary.json"))
            rm(joinpath(d, "chain_2"); recursive=true)   # chain 2 never wrote
            resumed = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], _Gauss3();
                n_cores=1, n_draws=20, n_evaluations_budget=20000, nonlinear_adapt=false,
                checkpoint_dir=d, resume=true, init=_init())
            # Chain 1 restored, chain 2 restarted fresh in its slot: membership
            # preserved, run completed.
            @test resumed.n_started == 2
            @test sort!([r.chain_index for r in resumed.results]) == [1, 2]
            @test all(r -> r.status === :done, resumed.results)
        end
    end

    @testset "resume guard edges" begin
        mktempdir() do d
            # `resume=true` with no run present is refused, not silently fresh.
            @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(1)], _Gauss3();
                n_cores=1, n_evaluations_budget=1000, nonlinear_adapt=false,
                checkpoint_dir=d, resume=true, init=_init())
            # ... and a foreign sampler's directory is refused at restore.
            adaptive_payload = (;
                schema_version=WarmupHMC.checkpoint_schema_version(),
                sampler=:adaptive, dimension=3)
            cdir = joinpath(d, "chain_1")
            mkpath(cdir)
            open(io -> serialize(io, adaptive_payload), joinpath(cdir, "cp_latest.jls"), "w")
            @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(1)], _Gauss3();
                n_cores=1, n_evaluations_budget=1000, nonlinear_adapt=false,
                checkpoint_dir=d, resume=true, init=_init())
        end
    end
end
