@testitem "Adaptive mid-window checkpoints and exact resume" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, Serialization, LogDensityProblems
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: Banana, WatchingBanana, CONFIG, RESTART_CONFIG, stop_after, latest, comparable, same, run_case
    const WH = WarmupHMC

    @testset "stop during adaptation or retained sampling, unchanged boundary callbacks" begin
        mktempdir() do root
            full = joinpath(root, "full")
            reference = run_case(full)
            p = latest(full)
            for stop_at in (3, 20)
                dir = joinpath(root, "stop_$stop_at")
                stages = Symbol[]
                partial = run_case(dir; stop_at,
                    callback=(s, stage) -> (push!(stages, stage); nothing))
                saved = latest(dir)
                @test stages == [:init]
                @test saved.window_in_progress
                @test saved.schema_version == 3
                @test 0 < saved.current_evaluation_counter < saved.n_evaluations
                @test saved.outer_counter == 1
                @test readdir(dir) == ["cp_init.jls", "cp_latest.jls"]
                @test size(partial.posterior_position, 2) == saved.n_samples
                @test (saved.n_samples == 0) == (stop_at == 3)
                @test_throws ArgumentError WH._check_schema_version(merge(saved, (; schema_version=4)))
                resumed = run_case(dir; resume=true)
                @test resumed.posterior_position == reference.posterior_position
                @test resumed.posterior_gradient == reference.posterior_gradient
                @test same(comparable(latest(dir)), comparable(p))
                @test rand(deepcopy(latest(dir).rng), 8) == rand(deepcopy(p.rng), 8)
                @test latest(dir).schema_version == 2
                @test !latest(dir).window_in_progress
            end
        end
    end

    @testset "repeated suspensions through metric restarts" begin
        mktempdir() do root
            full = joinpath(root, "full")
            run_case(full; config=RESTART_CONFIG)
            p = latest(full)
            @test p.outer_counter > 1
            @test !isempty(p.scale_changes)
            dir = joinpath(root, "sliced")
            run_case(dir; config=RESTART_CONFIG, stop_at=4)
            attempts = 1
            while latest(dir).n_samples < CONFIG.n_draws && attempts < 150
                run_case(dir; config=RESTART_CONFIG, resume=true, stop_at=4)
                attempts += 1
            end
            @test 2 < attempts < 150
            @test same(comparable(latest(dir)), comparable(p))
        end
    end

    @testset "old boundary payloads, draw floor, and legacy resume wrapper" begin
        mktempdir() do root
            full = joinpath(root, "full")
            reference = run_case(full)
            dir = joinpath(root, "legacy")
            run_case(dir; callback=(s, stage) -> stage === :init)
            # The v2 writer had neither in-window field.
            old = Base.structdiff(latest(dir), (; window_in_progress=nothing, current_evaluation_counter=nothing))
            serialize(joinpath(dir, "cp_latest.jls"), old)
            resumed = run_case(dir; resume=true)
            @test resumed.posterior_position == reference.posterior_position
            @test same(comparable(latest(dir)), comparable(latest(full)))

            partial_dir = joinpath(root, "partial")
            partial = run_case(partial_dir; stop_at=20)
            payload = latest(partial_dir)
            bytes = read(joinpath(partial_dir, "cp_latest.jls"))
            short = run_case(partial_dir; resume=true, config=merge(CONFIG, (; n_draws=1)))
            @test short.posterior_position == partial.posterior_position
            @test read(joinpath(partial_dir, "cp_latest.jls")) == bytes
            legacy = WH.resume_warmup_mcmc(Banana(), joinpath(partial_dir, "cp_latest.jls");
                n_draws=CONFIG.n_draws, stepsize_adaptation_limit=CONFIG.stepsize_adaptation_limit,
                max_tree_depth=CONFIG.max_tree_depth, checkpoint_dir=partial_dir)
            @test legacy.posterior_position == reference.posterior_position
            @test legacy.total_evaluation_counter == reference.total_evaluation_counter
            @test payload.n_samples < CONFIG.n_draws
        end
    end

    @testset "periodic saves do not affect draws or add callback stages" begin
        mktempdir() do root
            full = joinpath(root, "full")
            reference = run_case(full)
            timed = joinpath(root, "timed")
            saw_in_window = Ref(false)
            stages = Symbol[]
            # A cheap slow target makes elapsed-time saves observable. Read the
            # previous save at the next safe poll; no sampler state is mutated.
            observer = () -> begin
                isfile(joinpath(timed, "cp_latest.jls")) &&
                    (saw_in_window[] |= latest(timed).window_in_progress)
                false
            end
            result = adaptive_warmup_mcmc(Xoshiro(123), Banana(0.001); CONFIG...,
                checkpoint_dir=timed, checkpoint_interval=0.001, stop_requested=observer,
                callback=(s, stage) -> (push!(stages, stage); nothing))
            @test saw_in_window[]
            @test stages == [:init, :window]
            @test result.posterior_position == reference.posterior_position
            @test same(comparable(latest(timed)), comparable(latest(full)))
            @test readdir(timed) == ["cp_init.jls", "cp_latest.jls", "cp_window_1.jls"]

            # Timer without any predicate also saves inside the window. The
            # target observes an already-written file on its next evaluation.
            timer = joinpath(root, "timer")
            fail_after_save = WatchingBanana(() -> begin
                if isfile(joinpath(timer, "cp_latest.jls")) && latest(timer).window_in_progress
                    error("simulated loss after periodic checkpoint")
                end
            end)
            @test_throws "simulated loss after periodic checkpoint" adaptive_warmup_mcmc(
                Xoshiro(123), fail_after_save; CONFIG...,
                checkpoint_dir=timer, checkpoint_interval=eps(Float64))
            @test latest(timer).window_in_progress
            @test latest(timer).total_transition_counter == 1
            timer_result = run_case(timer; resume=true)
            @test timer_result.posterior_position == reference.posterior_position
            @test same(comparable(latest(timer)), comparable(latest(full)))
        end
    end

    @testset "literal true, no disk, and input validation before overwriting" begin
        plain = adaptive_warmup_mcmc(Xoshiro(123), Banana(); CONFIG...)
        observational = adaptive_warmup_mcmc(Xoshiro(123), Banana(); CONFIG..., stop_requested=() -> 1)
        @test plain.posterior_position == observational.posterior_position
        stopped = adaptive_warmup_mcmc(Xoshiro(123), Banana(); CONFIG..., stop_requested=stop_after(3))
        @test size(stopped.posterior_position, 2) == 0
        mktempdir() do dir
            run_case(dir; stop_at=3)
            original = read(joinpath(dir, "cp_latest.jls"))
            for bad in (0, -1, Inf, NaN, true, "60")
                @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), Banana(); CONFIG...,
                    checkpoint_dir=dir, overwrite=true, checkpoint_interval=bad)
                @test read(joinpath(dir, "cp_latest.jls")) == original
            end
            @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), Banana(); CONFIG...,
                checkpoint_dir=dir, overwrite=true, stop_requested=true)
            @test read(joinpath(dir, "cp_latest.jls")) == original
        end
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), Banana(); checkpoint_interval=1)
    end

    @testset "independent chains resume with distinct random streams" begin
        mktempdir() do root
            targets = [Banana(), Banana()]
            rngs() = [Xoshiro(123), Xoshiro(456)]
            full = joinpath(root, "full")
            reference = adaptive_warmup_mcmc(rngs(), targets; CONFIG..., parallel=true,
                checkpoint_dir=full)
            sliced = joinpath(root, "sliced")
            # Independent thread-safe budgets: the target need not be shared and
            # neither do its random streams. One predicate instance is shared.
            checks = Threads.Atomic{Int}(0)
            adaptive_warmup_mcmc(rngs(), targets; CONFIG..., parallel=true,
                checkpoint_dir=sliced, stop_requested=() -> Threads.atomic_add!(checks, 1) >= 8)
            @test any(i -> latest(joinpath(sliced, "chain_$i")).n_samples < CONFIG.n_draws, 1:2)
            resumed = adaptive_warmup_mcmc(rngs(), [Banana(), Banana()]; CONFIG..., parallel=true,
                checkpoint_dir=sliced, resume=true)
            for i in 1:2
                @test resumed[i].posterior_position == reference[i].posterior_position
                @test same(comparable(latest(joinpath(sliced, "chain_$i"))),
                    comparable(latest(joinpath(full, "chain_$i"))))
            end
        end
    end

    @testset "Treebars interrupts at the opted-in transition points" begin
        mktempdir() do dir
            stages = Symbol[]
            calls = Ref(0)
            _Treebars.with_progress(:state; description="mid-window stop") do parent
                adaptive_warmup_mcmc(Xoshiro(123), Banana(); CONFIG...,
                    progress=parent, checkpoint_dir=dir, stop_requested=() -> begin
                        calls[] += 1
                        calls[] == 3 && _Treebars.request_interrupt!(parent)
                        false
                    end,
                    callback=(s, stage) -> (push!(stages, stage); false))
            end
            @test stages == [:init]
            @test latest(dir).window_in_progress
            @test latest(dir).total_transition_counter == 2
        end
    end

    @testset "completion forwards controls and retains its quorum contract" begin
        mktempdir() do dir
            err = try
                completion_warmup_mcmc([Xoshiro(123)], [Banana()]; CONFIG...,
                    checkpoint_dir=dir, stop_requested=stop_after(3),
                    stuck_stepsize_ratio=Inf, max_admitted_divergent_fraction=1)
            catch e
                e
            end
            @test err isa WH.CompletionQuorumError
            @test latest(joinpath(dir, "chain_1")).window_in_progress
            result = completion_warmup_mcmc([Xoshiro(999)], [Banana()]; CONFIG...,
                checkpoint_dir=dir, resume=true, checkpoint_interval=1,
                stuck_stepsize_ratio=Inf, max_admitted_divergent_fraction=1)
            @test result.completion.n_completed == 1
            @test latest(joinpath(dir, "chain_1")).n_samples == CONFIG.n_draws
        end
    end

    @testset "fresh-process continuation" begin
        mktempdir() do root
            cases = joinpath(@__DIR__, "checkpoint_cases.jl")
            full, sliced = joinpath(root, "full"), joinpath(root, "sliced")
            run_case(full; config=RESTART_CONFIG)
            run_case(sliced; config=RESTART_CONFIG, stop_at=4)
            # Base.julia_cmd preserves the active Julia executable and flags.
            program = "include(" * repr(cases) * "); using .AdaptiveCheckpointCases; " *
                "AdaptiveCheckpointCases.run_case(" * repr(sliced) *
                "; resume=true, config=AdaptiveCheckpointCases.RESTART_CONFIG)"
            run(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $program`)
            @test same(comparable(latest(sliced)), comparable(latest(full)))
        end
    end
end
