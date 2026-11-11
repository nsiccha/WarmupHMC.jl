@testitem "Adaptive checkpoint re-scoring" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, Serialization, LogDensityProblems
    import DynamicHMC
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: Banana, CONFIG, run_case, latest, same, comparable
    const WH = WarmupHMC

    # A deterministic stand-in for a changed CPU's small arithmetic differences.
    struct ShiftedBanana
        value_shift::Float64
        gradient_shift::Float64
    end
    LogDensityProblems.dimension(::ShiftedBanana) = 2
    LogDensityProblems.capabilities(::Type{ShiftedBanana}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(p::ShiftedBanana, x)
        value, gradient = LogDensityProblems.logdensity_and_gradient(Banana(), x)
        (value + p.value_shift, gradient .+ p.gradient_shift)
    end
    LogDensityProblems.logdensity(p::ShiftedBanana, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))
    struct NoncopyableBanana end
    LogDensityProblems.dimension(::NoncopyableBanana) = 2
    LogDensityProblems.logdensity_and_gradient(::NoncopyableBanana, x) =
        LogDensityProblems.logdensity_and_gradient(Banana(), x)
    Base.deepcopy_internal(::NoncopyableBanana, ::IdDict) = error("native target must not be copied")
    struct BadGradient end
    LogDensityProblems.dimension(::BadGradient) = 2
    LogDensityProblems.logdensity_and_gradient(::BadGradient, x) = (0.0, [1.0])

    without_audit(p) = Base.structdiff(comparable(p), (; rescored=nothing))
    mktempdir() do root
        original_dir = joinpath(root, "original")
        run_case(original_dir; stop_at=15)
        p = latest(original_dir)
        before = deepcopy(p)
        @test p.window_in_progress
        @test size(p.posterior_position, 2) > 0
        @test size(p.halo_position, 2) > 0

        @testset "identity, independent ownership and audit" begin
            refreshed = WH.rescore_warmup_checkpoint(Banana(), p; rtol=0)
            @test same(without_audit(refreshed), without_audit(p))
            @test same(comparable(p), comparable(before))
            @test refreshed.rng !== p.rng
            @test same(refreshed.rng, p.rng)
            @test refreshed.recorder !== p.recorder
            @test refreshed.position_and_gradient.q !== p.position_and_gradient.q
            @test refreshed.scale_options !== p.scale_options
            audit = only(refreshed.rescored)
            @test audit.n_evaluations == 1 + size(p.posterior_position, 2) + size(p.halo_position, 2)
            @test audit.max_abs == audit.max_scaled == 0
            @test audit.changed_values == audit.changed_gradient_components == 0
            @test audit.policy === :retained_pointwise
            @test audit.historical_state === :preserved
            counted = WH.CountingPosterior(NoncopyableBanana())
            @test only(WH.rescore_warmup_checkpoint(counted, p; rtol=0).rescored).max_scaled == 0
            @test counted.count[] == 0
            refreshed.posterior_position[1, 1] += 10
            refreshed.halo_gradient[1, 1] += 10
            @test same(comparable(p), comparable(before))
        end

        @testset "toleranced values and every retained gradient" begin
            target = ShiftedBanana(2e-12, 3e-12)
            refreshed = WH.rescore_warmup_checkpoint(target, p)
            audit = only(refreshed.rescored)
            @test audit.changed_values == 1
            @test audit.changed_gradient_components == 2audit.n_evaluations
            @test 0 < audit.max_scaled <= 1e-10
            @test refreshed.position_and_gradient.ℓq == p.position_and_gradient.ℓq + 2e-12
            for (positions, gradients) in ((refreshed.posterior_position, refreshed.posterior_gradient),
                                           (refreshed.halo_position, refreshed.halo_gradient))
                for column in axes(positions, 2)
                    @test gradients[:, column] == last(LogDensityProblems.logdensity_and_gradient(target, positions[:, column]))
                end
            end
            for key in keys(p)
                key in (:position_and_gradient, :posterior_gradient, :halo_gradient, :rescored, :scale_options) && continue
                @test same(p[key], refreshed[key])
            end
            @test same(comparable(p), comparable(before))
            @test_throws ArgumentError WH.rescore_warmup_checkpoint(target, p; rtol=0)
            twice = WH.rescore_warmup_checkpoint(target, refreshed; rtol=0)
            @test length(twice.rescored) == 2
            @test twice.rescored[2].max_scaled == 0
            continued_dir = joinpath(root, "small-drift")
            mkpath(continued_dir)
            serialize(joinpath(continued_dir, "cp_latest.jls"), refreshed)
            result = resume_warmup_mcmc(target, joinpath(continued_dir, "cp_latest.jls");
                CONFIG..., checkpoint_dir=continued_dir)
            @test size(result.posterior_position, 2) >= CONFIG.n_draws
            @test all(isfinite, result.posterior_position)
            @test latest(continued_dir).rescored == refreshed.rescored
        end

        @testset "reject drift and malformed inputs without modifying payload" begin
            for target in (ShiftedBanana(1.0, 0.0), ShiftedBanana(0.0, 1.0),
                           ShiftedBanana(NaN, 0.0), ShiftedBanana(0.0, Inf))
                @test_throws ArgumentError WH.rescore_warmup_checkpoint(target, p)
                @test same(comparable(p), comparable(before))
            end
            bad = deepcopy(p)
            bad.halo_gradient[2, 1] += 100
            saved = deepcopy(bad)
            err = try WH.rescore_warmup_checkpoint(Banana(), bad) catch e; e end
            @test err isa ArgumentError
            @test occursin("halo_gradient", sprint(showerror, err))
            @test occursin("coordinate = 2", sprint(showerror, err))
            @test same(comparable(bad), comparable(saved))
            bad.posterior_gradient[1, 1] = NaN
            @test_throws ArgumentError WH.rescore_warmup_checkpoint(Banana(), bad)
            @test_throws DimensionMismatch WH.rescore_warmup_checkpoint(BadGradient(), p)
            @test_throws DimensionMismatch WH.rescore_warmup_checkpoint(Banana(), merge(p, (; dimension=3)))
            @test_throws DimensionMismatch WH.rescore_warmup_checkpoint(Banana(), merge(p, (; halo_gradient=zeros(2, 0))))
            @test_throws ArgumentError WH.rescore_warmup_checkpoint(Banana(), merge(p, (; schema_version=999)))
            @test_throws ArgumentError WH.rescore_warmup_checkpoint(Banana(), merge(p, (; sampler=:cooperative)))
            for rtol in (-1, NaN, Inf, true, "0")
                @test_throws ArgumentError WH.rescore_warmup_checkpoint(Banana(), p; rtol)
            end
        end

        @testset "legacy boundary and dropped snapshots" begin
            boundary = deserialize(joinpath(original_dir, "cp_init.jls"))
            legacy = Base.structdiff(boundary, (; schema_version=nothing, sampler=nothing,
                rescored=nothing, nonlinear_recorder=nothing, linear_recorder=nothing))
            restored = WH.rescore_warmup_checkpoint(Banana(), legacy; rtol=0)
            @test only(restored.rescored).n_evaluations == 1
            # Dropped epochs can use a different working frame; they are inert.
            dropped = merge(p, (; dropped_posterior_position=fill(NaN, 2, 1),
                dropped_posterior_gradient=fill(NaN, 2, 1)))
            refreshed = WH.rescore_warmup_checkpoint(Banana(), dropped; rtol=0)
            @test isequal(refreshed.dropped_posterior_gradient, dropped.dropped_posterior_gradient)
            @test refreshed.dropped_posterior_gradient !== dropped.dropped_posterior_gradient
        end

        @testset "ordinary and explicit resume agree; audits survive serialization" begin
            ordinary_dir, refreshed_dir = joinpath(root, "ordinary"), joinpath(root, "refreshed")
            mkpath(ordinary_dir); mkpath(refreshed_dir)
            serialize(joinpath(ordinary_dir, "cp_latest.jls"), p)
            refreshed = WH.rescore_warmup_checkpoint(Banana(), p; rtol=0)
            serialize(joinpath(refreshed_dir, "cp_latest.jls"), refreshed)
            reference = run_case(ordinary_dir; resume=true)
            result = run_case(refreshed_dir; resume=true)
            @test result.posterior_position == reference.posterior_position
            @test result.posterior_gradient == reference.posterior_gradient
            @test same(without_audit(latest(refreshed_dir)), without_audit(latest(ordinary_dir)))
            @test latest(refreshed_dir).rescored == refreshed.rescored
            @test isempty(latest(ordinary_dir).rescored)
        end
    end
end

@testitem "Nonlinear checkpoint re-scoring" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, Serialization, LogDensityProblems
    using DifferentiationInterface, Enzyme
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: stop_after, latest, comparable, same
    const WH = WarmupHMC

    struct RescoreFunnel end
    LogDensityProblems.dimension(::RescoreFunnel) = 3
    LogDensityProblems.capabilities(::Type{RescoreFunnel}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(::RescoreFunnel, y)
        v = y[1]; x = @view y[2:end]; e = exp(-v)
        (-v^2 / 18 - sum(abs2, x) * e / 2 - v,
            [-v / 9 + sum(abs2, x) * e / 2 - 1; -x .* e])
    end
    LogDensityProblems.logdensity(p::RescoreFunnel, y) = first(LogDensityProblems.logdensity_and_gradient(p, y))
    problem(c=0.3; plan=nothing) = ReparametrizedProblem(
        IndexedReparametrization([i => Reparametrization(
            PartiallyCentered(1.0), PartiallyCentered(c), 0.0, y -> y[1] / 2) for i in 2:3]),
        RescoreFunnel(), AutoEnzyme(); scoring_plan=plan)
    config = (; n_draws=30, n_evaluations=32, stepsize_adaptation_limit=4,
        recording_target=12, max_tree_depth=4, variance_cond_target=1.0,
        monitor_ess=true, linear_restart_source=:nuts_weighted,
        nonlinear_evidence=:nuts_weighted,
        init=(; position=[0.3, -0.2, 0.4], squared_scale=ones(3)))
    @test !isnothing(Base.get_extension(WarmupHMC, :WarmupHMCEnzymeExt))
    mktempdir() do root
        for sampled in (false, true)
            # Selection is deliberately a no-op; its occupied reservoir still
            # feeds a real WindowSelectionPlan on subsequent restarts.
            plan = sampled ? WindowSelectionPlan((ir, x, g) -> false) : nothing
            dir = joinpath(root, string(sampled))
            adaptive_warmup_mcmc(Xoshiro(17), problem(; plan); config...,
                checkpoint_dir=dir, stop_requested=stop_after(4))
            p = latest(dir)
            before = deepcopy(p)
            target = problem(0.8; plan)
            caller_controls = WH.reparam_sources(target)
            refreshed = WH.rescore_warmup_checkpoint(target, p; rtol=0)
            @test WH.reparam_sources(target) == caller_controls
            @test same(comparable(p), comparable(before))
            @test same(refreshed.linear_recorder, p.linear_recorder)
            @test same(refreshed.variance_gradient, p.variance_gradient)
            @test same(refreshed.nonlinear_recorder, p.nonlinear_recorder)
            count = 1 + size(p.halo_position, 2) + size(p.posterior_position, 2)
            if sampled
                sample = p.nonlinear_recorder.online
                @test sample.total > 0
                @test only(refreshed.rescored).n_evaluations == count + size(sample.position, 2)
                @test same(sample.rng, refreshed.nonlinear_recorder.online.rng)
                rounded = deepcopy(p)
                rounded.nonlinear_recorder.online.gradient[1, 1] += 1e-12
                repaired = WH.rescore_warmup_checkpoint(target, rounded)
                @test same(repaired.nonlinear_recorder.online, sample)
                @test only(repaired.rescored).changed_gradient_components == 1
                corrupt = deepcopy(p)
                corrupt.nonlinear_recorder.online.gradient[1, 1] += 100
                @test_throws ArgumentError WH.rescore_warmup_checkpoint(target, corrupt)
                @test WH.reparam_sources(target) == caller_controls
                # A reset reservoir has stale, unoccupied entries: never score them.
                WH.reset!(corrupt.nonlinear_recorder.online)
                corrupt.nonlinear_recorder.online.position .= NaN
                empty = WH.rescore_warmup_checkpoint(target, corrupt; rtol=0)
                @test only(empty.rescored).n_evaluations == count
                @test all(isnan, empty.nonlinear_recorder.online.position)
            else
                @test only(refreshed.rescored).n_evaluations == count
            end
            refreshed_dir = dir * "-refreshed"
            mkpath(refreshed_dir)
            serialize(joinpath(refreshed_dir, "cp_latest.jls"), refreshed)
            expected = adaptive_warmup_mcmc(Xoshiro(99), problem(; plan); config...,
                checkpoint_dir=dir, resume=true)
            actual = adaptive_warmup_mcmc(Xoshiro(99), problem(; plan); config...,
                checkpoint_dir=refreshed_dir, resume=true)
            @test actual.posterior_position == expected.posterior_position
            @test actual.posterior_gradient == expected.posterior_gradient
            @test latest(refreshed_dir).rescored == refreshed.rescored
        end
    end
end
