@testitem "completion_warmup_mcmc" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Pkg, TOML
    using Distributions
    using DifferentiationInterface, Enzyme

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner; using LinearAlgebra;
# BLAS.set_num_threads(1); TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "completion_policy.jl"))'

_completion_fixture(i; n=5) = (; posterior_position=fill(Float64(i), 2, n),
    n_divergent_samples=i % 2, stepsize=1.0)

struct CompletionBimodal end
LogDensityProblems.dimension(::CompletionBimodal) = 1
LogDensityProblems.capabilities(::Type{CompletionBimodal}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(::CompletionBimodal, x)
    a, b = -0.5 * (x[1] + 8)^2, -0.5 * (x[1] - 8)^2
    l = WarmupHMC.logaddexp(a, b)
    l - log(2) - log(2π) / 2, [exp(a-l) * (-x[1]-8) + exp(b-l) * (-x[1]+8)]
end
LogDensityProblems.logdensity(p::CompletionBimodal, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

@testset "controlled completion times and original identities" begin
    now = Ref(0.0)
    s = WarmupHMC._CompletionState(5; clock=() -> now[], min_completed=2)
    # Finishing out of order preserves IDs; quorum is based on full results.
    complete = (i, stop) -> (_completion_fixture(i), true)
    now[] = 1.0
    WarmupHMC._completion_worker!(complete, s, 3)
    @test isnothing(s.quorum_at)
    # Chain 1 is running when chain 4 reaches quorum. Its final round can
    # finish arbitrarily later: completion is still admitted, without a timer.
    WarmupHMC._completion_worker!(s, 1) do i, stop
        @test !stop()
        now[] = 4.0
        WarmupHMC._completion_worker!(complete, s, 4)
        @test s.quorum_at == 4.0
        now[] = 600.0
        @test stop()
        _completion_fixture(i), true
    end
    @test s.statuses[1] === :completed
    @test s.finished_at[1] == 600.0
    @test s.quorum_at == 4.0
    WarmupHMC._completion_worker!(complete, s, 2)
    @test s.statuses[2] === :not_started
    WarmupHMC._completion_worker!(complete, s, 5)
    @test s.statuses[5] === :not_started
    @test findall(==(:completed), s.statuses) == [1, 3, 4]
end

@testset "heterogeneous workers settle before return" begin
    entered = Channel{Int}(3)
    release = [Channel{Nothing}(1) for _ in 1:3]
    cleaned = Threads.Atomic{Int}(0)
    slow_finished = Threads.Atomic{Bool}(false)
    batch = Threads.@spawn WarmupHMC._completion_batch(3; min_completed=1) do i, stop
        put!(entered, i)
        take!(release[i])
        try
            if i == 2
                return _completion_fixture(i), true
            end
            # Emulates a long active window; observe cancellation only at its
            # checkpoint. Successful return requires both slow workers to exit.
            while !stop()
                yield()
            end
            if i == 3
                slow_finished[] = true
                return _completion_fixture(i), true
            end
            _completion_fixture(i; n=2), false
        finally
            Threads.atomic_add!(cleaned, 1)
        end
    end
    @test sort([take!(entered) for _ in 1:3]) == [1, 2, 3]
    foreach(c -> put!(c, nothing), release)
    out = fetch(batch)
    @test cleaned[] == 3
    @test slow_finished[]
    @test out.completion.completed_chain_indices == [2, 3]
    @test out.completion.omitted_chain_indices == [1]
    @test out.completion.n_started == 3
    @test getproperty.(out.results, :chain_index) == [2, 3]
    @test out.completion.n_samples == 10
    @test out.completion.n_divergent_samples == 1
    @test out.completion.stop_reason === :quorum_reached
    @test out.completion.stop_policy === :finish_current_round
    @test out.completion.stop_requested_at_seconds == out.completion.quorum_at_seconds
    @test !haskey(out.completion, :grace_seconds)
    @test !haskey(out.completion, :cutoff_at_seconds)
    @test all(c -> c.n_retained_samples == (c.chain_index in (2, 3) ? 5 : 0), out.completion.chains)
end

@testset "final rounds admit extra workers, failures stay explicit" begin
    entered = Channel{Int}(4)
    release = Channel{Nothing}(4)
    batch = Threads.@spawn WarmupHMC._completion_batch(4; min_completed=2) do i, stop
        put!(entered, i)
        take!(release)
        i == 2 && error("controlled chain 2 failure")
        _completion_fixture(i), true
    end
    @test sort([take!(entered) for _ in 1:4]) == [1, 2, 3, 4]
    foreach(_ -> put!(release, nothing), 1:4)
    out = fetch(batch)
    @test out.completion.completed_chain_indices == [1, 3, 4]
    @test out.completion.failed_chain_indices == [2]
    @test out.completion.omitted_chain_indices == [2]
    @test out.completion.n_completed == 3
    @test out.completion.n_failed == 1
    @test isnothing(out.completion.chains[2].n_samples)
    @test out.completion.chains[2].n_retained_samples == 0
    @test out.completion.stop_reason === :quorum_reached
    @test occursin("controlled chain 2 failure", sprint(showerror, out.completion.chains[2].error))
    @test out.completion.n_samples == sum(r -> size(r.posterior_position, 2), out.results)
    @test out.completion.n_divergent_samples == sum(r -> r.n_divergent_samples, out.results)

    unmet = WarmupHMC._completion_batch(2; min_completed=2) do i, stop
        i == 1 && error("failed initialization")
        _completion_fixture(i; n=0), false
    end
    @test isempty(unmet.results)
    @test !unmet.completion.quorum_met
    @test unmet.completion.stop_reason === :quorum_unmet
    @test occursin("0 of 2", sprint(showerror, WarmupHMC.CompletionQuorumError(unmet)))
end

@testset "control characters in failure JSON" begin
    @test WarmupHMC._json_val("tab\tnewline\nreturn\rnull\0quote\"slash\\") ==
        "\"tab\\u0009newline\\nreturn\\u000dnull\\u0000quote\\\"slash\\\\\""
end

@testset "already-complete admission precedes scheduling" begin
    initial = Dict(3 => _completion_fixture(3), 1 => _completion_fixture(1))
    out = WarmupHMC._completion_batch(4; min_completed=1,
        initial_results=initial) do i, stop
        error("No incomplete worker should start after quorum.")
    end
    @test out.completion.completed_chain_indices == [1, 3]
    @test out.completion.started_chain_indices == [1, 3]
    @test out.completion.failed_chain_indices == Int[]
    @test out.completion.omitted_chain_indices == [2, 4]
end

# Snag quorum-admits-a-7aa7f768, decision 1dl030w: reaching n_draws first is no
# evidence of health. A chain whose step size collapsed, or whose transitions
# diverge, takes short cheap trajectories and can finish before every sibling.
_admission_fixture(stepsize; n=20, n_divergent=0) = (; posterior_position=zeros(2, n),
    n_divergent_samples=n_divergent, stepsize)
const _ADMISSION_CHECKS = WarmupHMC._CompletionAdmission(10.0, 0.1)

@testset "a collapsed chain that finishes first does not count" begin
    s = WarmupHMC._CompletionState(4; clock=() -> 0.0, min_completed=2,
        admission=_ADMISSION_CHECKS)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._report_boundary!(stop1, 0.17, 30)
        WarmupHMC._completion_worker!(s, 2) do _, stop2
            WarmupHMC._report_boundary!(stop2, 0.16, 30)
            # Chains 1 and 2 keep draws at a healthy step size when chain 4,
            # whose step size collapsed, finishes first.
            WarmupHMC._completion_worker!((_, _) -> (_admission_fixture(0.004), true), s, 4)
            @test s.statuses[4] === :stuck
            @test s.admissions[4].stuck_reason === :stepsize
            @test s.admissions[4].stepsize == 0.004
            @test s.admissions[4].reference_stepsize ≈ 0.165
            _admission_fixture(0.16), true
        end
        # Counted, chain 4 would have met the quorum of two with chain 2 and
        # stopped this chain. It does not count.
        @test s.statuses[2] === :completed
        @test !stop1()
        _admission_fixture(0.17), true
    end
    @test s.statuses[1] === :completed
    @test !isnothing(s.quorum_at)
    WarmupHMC._completion_worker!((_, _) -> error("the quorum is met"), s, 3)
    @test s.statuses == [:completed, :completed, :not_started, :stuck]
end

@testset "a finished chain waits, uncounted, for a sibling's step size" begin
    s = WarmupHMC._CompletionState(3; clock=() -> 0.0, min_completed=1,
        admission=_ADMISSION_CHECKS)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        # Chain 2 finishes while no sibling keeps draws yet.
        WarmupHMC._completion_worker!((_, _) -> (_admission_fixture(0.17), true), s, 2)
        @test s.statuses[2] === :awaiting_reference
        @test !stop1()
        # A boundary that keeps no draws (initialization, or a window that
        # restarted) is no reference.
        WarmupHMC._report_boundary!(stop1, 0.9, 0)
        @test s.statuses[2] === :awaiting_reference
        # The first boundary that keeps draws settles it, and with it the quorum.
        WarmupHMC._report_boundary!(stop1, 0.15, 30)
        @test s.statuses[2] === :completed
        @test s.admissions[2].reference_stepsize == 0.15
        @test stop1()
        _admission_fixture(0.15; n=5), false
    end
    WarmupHMC._completion_worker!((_, _) -> error("the quorum is met"), s, 3)
    @test s.statuses == [:stopped, :completed, :not_started]

    # A chain judged after waiting can be left out too.
    s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1,
        admission=_ADMISSION_CHECKS)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._completion_worker!((_, _) -> (_admission_fixture(0.004), true), s, 2)
        @test s.statuses[2] === :awaiting_reference
        WarmupHMC._report_boundary!(stop1, 0.17, 30)
        @test s.statuses[2] === :stuck
        @test !stop1()
        _admission_fixture(0.17), true
    end
    @test s.statuses == [:completed, :stuck]

    # Once no sibling can supply a step size, the waiting chain is admitted.
    s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1,
        admission=_ADMISSION_CHECKS)
    WarmupHMC._completion_worker!(s, 1) do _, _
        WarmupHMC._completion_worker!((_, _) -> (_admission_fixture(0.17), true), s, 2)
        @test s.statuses[2] === :awaiting_reference
        error("chain 1 fails before it keeps draws")
    end
    @test s.statuses == [:failed, :completed]
    @test isnothing(s.admissions[2].reference_stepsize)
    @test !isnothing(s.quorum_at)
    single = WarmupHMC._completion_batch((_, _) -> (_admission_fixture(1e-6), true), 1;
        min_completed=1, admission=_ADMISSION_CHECKS)
    @test single.completion.completed_chain_indices == [1]
end

@testset "a stuck sibling is no reference" begin
    s = WarmupHMC._CompletionState(3; clock=() -> 0.0, min_completed=1,
        admission=_ADMISSION_CHECKS)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        # Chain 3 is left out for its divergences. Its large step size must not
        # make chain 2's healthy one look collapsed.
        WarmupHMC._completion_worker!((_, _) ->
            (_admission_fixture(5.0; n_divergent=10), true), s, 3)
        @test s.statuses[3] === :stuck
        @test s.admissions[3].stuck_reason === :divergence
        WarmupHMC._completion_worker!((_, _) -> (_admission_fixture(0.17), true), s, 2)
        @test s.statuses[2] === :awaiting_reference
        WarmupHMC._report_boundary!(stop1, 0.18, 30)
        @test s.statuses[2] === :completed
        _admission_fixture(0.18), true
    end
    @test s.statuses == [:completed, :completed, :stuck]
end

@testset "stuck chains are omitted and named in the outcome" begin
    # Chain 3 diverges on 15% of its draws. That needs no sibling to judge.
    out = WarmupHMC._completion_batch(4; min_completed=3, admission=_ADMISSION_CHECKS) do i, stop
        _admission_fixture(0.17; n_divergent=i == 3 ? 3 : 0), true
    end
    c = out.completion
    @test c.completed_chain_indices == [1, 2, 4]
    @test c.stuck_chain_indices == [3]
    @test c.omitted_chain_indices == [3]
    @test c.n_stuck == 1
    @test c.quorum_met && c.stop_reason === :quorum_reached
    @test getproperty.(out.results, :chain_index) == [1, 2, 4]
    @test c.n_samples == 60
    @test c.chains[3].status === :stuck
    @test c.chains[3].stuck_reason === :divergence
    @test c.chains[3].divergent_fraction == 0.15
    @test c.chains[3].n_samples == 20
    @test c.chains[3].n_retained_samples == 0
    @test all(i -> isnothing(c.chains[i].stuck_reason), (1, 2, 4))
    @test all(i -> c.chains[i].stepsize == 0.17, 1:4)
    @test c.stuck_stepsize_ratio == 10.0
    @test c.max_admitted_divergent_fraction == 0.1
    # Fewer passing chains than the quorum: the existing error, naming them.
    unmet = WarmupHMC._completion_batch(2; min_completed=2, admission=_ADMISSION_CHECKS) do i, stop
        _admission_fixture(0.17; n_divergent=i == 1 ? 10 : 0), true
    end
    @test !unmet.completion.quorum_met
    @test unmet.completion.stuck_chain_indices == [1]
    message = sprint(showerror, WarmupHMC.CompletionQuorumError(unmet))
    @test occursin("1 of 2", message)
    @test occursin("left out as stuck after reaching `n_draws`: [1]", message)
    # Switched off, both checks admit every full chain.
    off = WarmupHMC._completion_batch(2; min_completed=2,
            admission=WarmupHMC._CompletionAdmission(Inf, 1.0)) do i, stop
        i == 1 ? (_admission_fixture(1e-6; n_divergent=20), true) : (_admission_fixture(0.17), true)
    end
    @test off.completion.completed_chain_indices == [1, 2]
    @test off.completion.chains[1].divergent_fraction == 1.0
    @test isnothing(off.completion.chains[1].reference_stepsize)
end

@testset "resume judges already-complete checkpoints before scheduling" begin
    # Slot 3's checkpoint keeps draws at 0.18 but is not complete yet.
    initial = Dict(1 => _admission_fixture(0.004), 2 => _admission_fixture(0.17))
    out = WarmupHMC._completion_batch(3; min_completed=1, admission=_ADMISSION_CHECKS,
            initial_results=initial, initial_stepsizes=Dict(3 => 0.18)) do i, stop
        error("No incomplete worker should start after quorum.")
    end
    c = out.completion
    @test c.stuck_chain_indices == [1]
    @test c.chains[1].reference_stepsize ≈ 0.175
    @test c.completed_chain_indices == [2]
    @test c.chains[3].status === :not_started
    @test c.quorum_at_seconds == 0.0
end

# A standard normal cut by a hard wall through its mode: every trajectory that
# crosses x[1] = 0 meets -Inf, which the sampler counts as a divergence.
struct CompletionWall end
LogDensityProblems.dimension(::CompletionWall) = 2
LogDensityProblems.capabilities(::Type{CompletionWall}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity_and_gradient(::CompletionWall, x) =
    x[1] < 0 ? (-Inf, zero(x)) : (-sum(abs2, x) / 2, -x)
LogDensityProblems.logdensity(p::CompletionWall, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

@testset "a real chain that reaches n_draws mostly divergent is left out" begin
    BLAS.set_num_threads(1)
    healthy = DiagGaussian([0.0, 0.0], [1.0, 1.0])
    init = (; position=[0.5, 0.0], squared_scale=Diagonal(ones(2)))
    # The healthy chains hold their final boundary until the walled chain has
    # reached n_draws, so it is judged however the threads are scheduled.
    walled_finished = Threads.Atomic{Bool}(false)
    hold = (state, stage) -> begin
        if size(state.recording_lpdf.posterior_position, 2) >= state.n_draws
            state.lpdf isa CompletionWall ? (walled_finished[] = true) :
                timedwait(() -> walled_finished[], 120)
        end
        false
    end
    # `max_divergent_fraction=1` keeps the adaptive sampler's own restart gate
    # out of the way: the walled chain returns its draws as they are.
    out = completion_warmup_mcmc([Xoshiro(i) for i in 1:4],
        [healthy, healthy, CompletionWall(), healthy];
        min_completed=3, n_draws=200, init, callback=hold, max_divergent_fraction=1.0)
    c = out.completion
    @test walled_finished[]
    @test c.completed_chain_indices == [1, 2, 4]
    @test c.stuck_chain_indices == [3]
    @test c.omitted_chain_indices == [3]
    @test c.stop_reason === :quorum_reached
    @test getproperty.(out.results, :chain_index) == [1, 2, 4]
    @test c.chains[3].stuck_reason === :divergence
    @test c.chains[3].n_samples == 200
    @test c.chains[3].divergent_fraction > 0.5
    @test all(i -> c.chains[i].divergent_fraction == 0, (1, 2, 4))
    @test all(i -> !isnothing(c.chains[i].reference_stepsize), (1, 2, 4))
    @test c.n_samples == 600
end

@testset "numerical equivalence and recovery" begin
    BLAS.set_num_threads(1)
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    cfg = (; n_draws=600, progress=nothing)
    baseline = adaptive_warmup_mcmc([Xoshiro(i) for i in 1:3], p; cfg..., parallel=false)
    out = completion_warmup_mcmc([Xoshiro(i) for i in 1:3], p; cfg...)
    @test out.completion.stop_reason === :all_completed
    @test out.completion.completed_chain_indices == collect(1:3)
    @test out.completion.n_samples == 1800
    for (i, r) in enumerate(out.results)
        @test r.chain_index == i
        @test r.posterior_position == baseline[i].posterior_position
        @test r.posterior_gradient == baseline[i].posterior_gradient
        @test r.n_divergent_samples == baseline[i].n_divergent_samples
        @test r.total_evaluation_counter == baseline[i].total_evaluation_counter
        @test Set(keys(r)) == union(Set(keys(baseline[i])), Set((:chain_index, :n_samples)))
    end
    draws = hcat(getproperty.(out.results, :posterior_position)...)
    @test all(abs.(vec(mean(draws; dims=2)) .- p.mu) .< 0.2 .* p.sigma)
    @test all(abs.(vec(std(draws; dims=2)) ./ p.sigma .- 1) .< 0.2)
    @test out.completion.n_divergent_samples <= 3
end

@testset "mode-dependent runtime selection is not inferential validation" begin
    # A symmetric two-mode density has half its mass on each side. These real
    # adaptive draws stay in their well-separated initialized modes. Artificial
    # round delays below are conditioned on that mode: this is a controlled
    # selection-bias counterexample, not a timing benchmark or a claim of mixing.
    p = CompletionBimodal()
    inits = [(; position=[x], squared_scale=Diagonal(ones(1))) for x in (-7.5, 7.5, -8.5, 8.5)]
    full = adaptive_warmup_mcmc([Xoshiro(i+100) for i in 1:4], p;
        init=inits, n_draws=100, n_evaluations=100, parallel=false)
    @test all(r -> size(r.posterior_position, 2) == 100, full)
    @test [mean(r.posterior_position) > 0 for r in full] == [false, true, false, true]
    @test mean(hcat(getproperty.(full, :posterior_position)...) .> 0) == 0.5
    selected = WarmupHMC._completion_batch(4; min_completed=2) do i, stop
        if mean(full[i].posterior_position) < 0
            while !stop()
                yield()
            end
            # This mode's current round reaches only half the draw target;
            # a full result would have to be included even after quorum.
            return merge(full[i], (; posterior_position=full[i].posterior_position[:, 1:50],
                posterior_gradient=full[i].posterior_gradient[:, 1:50])), false
        end
        full[i], true
    end
    @test selected.completion.completed_chain_indices == [2, 4]
    @test selected.completion.omitted_chain_indices == [1, 3]
    @test mean(hcat(getproperty.(selected.results, :posterior_position)...) .> 0) == 1.0
    @test selected.completion.n_samples == 200
    @test occursin("bias inference", selected.completion.selection_warning)
end

@testset "safe checkpoints and immutable terminal selection" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    mktempdir() do dir
        # A stopped callback is an incomplete chain, not a completed window.
        err = try
            completion_warmup_mcmc([Xoshiro(7), Xoshiro(8)], p;
                n_draws=40, checkpoint_dir=dir, callback=(s, stage) -> true)
        catch e
            e
        end
        @test err isa WarmupHMC.CompletionQuorumError
        @test err.outcome.completion.n_completed == 0
        @test err.outcome.completion.n_samples == 0
        @test all(c -> c.status === :stopped, err.outcome.completion.chains)
        @test isfile(joinpath(err.outcome.completion.attempt_directory, "run_summary.json"))
        @test !isfile(joinpath(dir, "completion_terminal.jls"))
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(7), Xoshiro(8)], p;
            n_draws=40, checkpoint_dir=dir)
        # Checkpoints are ordinary adaptive payloads and contain source state.
        for i in 1:2
            cp = WarmupHMC.deserialize(joinpath(dir, "chain_$i", "cp_latest.jls"))
            @test cp.sampler === :adaptive
            @test size(cp.posterior_position, 2) == 0
            @test !haskey(cp, :min_completed)
            @test !haskey(cp, :grace_seconds)
        end
        # Legacy run manifests wrap the same adaptive checkpoint payloads.
        # An unfinished legacy run switches to the current round-stop policy.
        WarmupHMC._atomic_serialize_manifest(joinpath(dir, "completion_manifest.jls"),
            (; schema_version=1, sampler=:completion, n_requested=2))
        # Complete chain 2 separately. Resume must admit it before any worker
        # has a chance to initialize chain 1. Chain 1 keeps no draws, so the
        # step-size check would make chain 2 wait for it to sample ("a finished
        # chain waits, uncounted"); this run pins admission without that check.
        ordinary = adaptive_warmup_mcmc(Xoshiro(999), p;
            n_draws=40, checkpoint_dir=joinpath(dir, "chain_2"), resume=true)
        unchecked = (; stuck_stepsize_ratio=Inf)
        out = completion_warmup_mcmc([Xoshiro(999), Xoshiro(999)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true, unchecked...)
        @test out.completion.completed_chain_indices == [2]
        @test out.completion.omitted_chain_indices == [1]
        @test out.results[1].posterior_position == ordinary.posterior_position
        @test out.completion.chains[1].status === :not_started
        @test out.completion.schema_version == 3
        @test out.completion.stop_policy === :finish_current_round
        cp1 = read(joinpath(dir, "chain_1", "cp_latest.jls"))
        @test isfile(joinpath(dir, "completion_terminal.jls"))
        @test isfile(joinpath(out.completion.attempt_directory, "run_summary.json"))
        @test occursin("\"completed_chain_indices\":[2]",
            read(joinpath(out.completion.attempt_directory, "run_summary.json"), String))
        @test occursin("\"stop_policy\":\"finish_current_round\"",
            read(joinpath(out.completion.attempt_directory, "run_summary.json"), String))
        @test !occursin("grace_seconds",
            read(joinpath(out.completion.attempt_directory, "run_manifest.json"), String))
        # Advance the omitted checkpoint: reopening must still preserve the
        # recorded one-chain selection, draws, times, counts and original IDs.
        adaptive_warmup_mcmc(Xoshiro(999), p;
            n_draws=40, checkpoint_dir=joinpath(dir, "chain_1"), resume=true)
        @test read(joinpath(dir, "chain_1", "cp_latest.jls")) != cp1
        again = completion_warmup_mcmc([Xoshiro(11), Xoshiro(12)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true, unchecked...,
            callback=(s, stage) -> error("Terminal reopen must never run a callback"))
        @test again.completion == out.completion
        @test again.results[1].posterior_position == out.results[1].posterior_position
        @test again.results[1].chain_index == 2
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p;
            n_draws=40, checkpoint_dir=dir, resume=true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=41, min_completed=1, checkpoint_dir=dir, resume=true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, grace_seconds=1, checkpoint_dir=dir, resume=true)
        # refused: the admission thresholds are part of a finished run's
        # immutable policy, like `min_completed` (docstring, decision 1dl030w).
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true)
        # A crash between terminal serialization and JSON publication is
        # recoverable from the immutable record, without sampling again.
        summary = joinpath(out.completion.attempt_directory, "run_summary.json")
        original_summary = read(summary)
        rm(summary)
        recovered = completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true, unchecked...)
        @test recovered.completion == out.completion
        @test read(summary) == original_summary
        # A schema-1 terminal fixture retains the OLD selection semantics,
        # including a full chain omitted after cutoff. It must not be silently
        # reinterpreted under the new policy when reopened.
        legacy_fields = (; (k => v for (k, v) in pairs(out.completion)
            if k ∉ (:schema_version, :stop_policy, :stop_requested_at_seconds,
                :stuck_stepsize_ratio, :max_admitted_divergent_fraction, :n_stuck,
                :stuck_chain_indices))...)
        legacy_chains = Any[Base.structdiff(c, WarmupHMC._UNJUDGED) for c in out.completion.chains]
        legacy_chains[1] = merge(legacy_chains[1], (; status=:completed_after_cutoff,
            finished_at_seconds=4.0, n_samples=40))
        legacy = (; out.results, completion=merge(legacy_fields, (; grace_seconds=3.0,
            cutoff_at_seconds=3.0, quorum_at_seconds=0.0, elapsed_seconds=4.0,
            started_chain_indices=[1, 2], n_started=2, stop_reason=:grace_expired,
            chains=legacy_chains)))
        WarmupHMC._atomic_serialize_manifest(joinpath(dir, "completion_terminal.jls"),
            (; schema_version=1, n_draws=40, outcome=legacy))
        rm(summary)
        # It predates the admission checks, so it reopens whatever thresholds
        # are passed: its recorded selection was made without them.
        old = completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true,
            callback=(s, stage) -> error("Legacy terminal reopen must not sample"))
        @test old.completion == legacy.completion
        @test getproperty.(old.results, :chain_index) == [2]
        @test old.results[1].posterior_position == out.results[1].posterior_position
        @test occursin("\"grace_seconds\":3.0", read(summary, String))
        @test !occursin("chain_stuck_reasons", read(summary, String))
        # Explicit overwrite starts a new run and clears the old selection.
        fresh = completion_warmup_mcmc([Xoshiro(21), Xoshiro(22)], p;
            n_draws=40, checkpoint_dir=dir, overwrite=true)
        @test fresh.completion.completed_chain_indices == [1, 2]
        @test fresh.completion.n_samples == 80
    end
end

@testset "real chain failures are terminal and do not satisfy quorum" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    init = (; position=[0.1, 0.2], squared_scale=Matrix{Float64}(I, 2, 2))
    mktempdir() do dir
        # NamedTuple initialization avoids the initializer's intentional retry
        # loop: this density throws at the actual strict gradient evaluation.
        err = try
            completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], [NaNProblem(2), p];
                n_draws=40, checkpoint_dir=dir, init)
        catch e
            e
        end
        @test err isa WarmupHMC.CompletionQuorumError
        @test err.outcome.completion.completed_chain_indices == [2]
        @test err.outcome.completion.failed_chain_indices == [1]
        @test err.outcome.completion.n_samples == 40
        @test err.outcome.completion.n_completed == 1
        @test err.outcome.completion.n_failed == 1
        @test !isnothing(err.outcome.completion.chains[1].error)
        @test isfile(joinpath(err.outcome.completion.attempt_directory, "run_summary.json"))
        @test !isfile(joinpath(dir, "completion_terminal.jls"))
        # The slot that failed before any checkpoint can initialize on resume;
        # the already-complete second slot is restored without new transitions.
        recovered = completion_warmup_mcmc([Xoshiro(1), Xoshiro(999)], p;
            n_draws=40, checkpoint_dir=dir, init, resume=true)
        @test recovered.completion.completed_chain_indices == [1, 2]
        @test recovered.results[2].posterior_position == err.outcome.results[1].posterior_position
    end
end

@testset "public policy validation precedes work" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    for value in (0, 3, 1.5, true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p; min_completed=value)
    end
    for value in (0.0, 30.0)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; grace_seconds=value)
    end
    @test :grace_seconds ∉ WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
    # refused: a step-size ratio below 1 would leave out chains above their
    # siblings' median, and a fraction outside [0, 1] bounds nothing (decision 1dl030w).
    for value in (0.5, NaN, true, "10")
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; stuck_stepsize_ratio=value)
    end
    for value in (-0.1, 1.5, NaN, true)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p;
            max_admitted_divergent_fraction=value)
    end
    @test :stuck_stepsize_ratio in WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
    @test :max_admitted_divergent_fraction in WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
    for value in (0, -1, typemax(Int), true, 1.5)
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; n_draws=value)
    end
    @test_throws ArgumentError completion_warmup_mcmc(Xoshiro[], p)
    @test_throws DimensionMismatch completion_warmup_mcmc([Xoshiro(1)], [p, p])
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; resume=true)
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p; n_draw=5)
end

@testset "progress tree labels chains by slot and shows quorum" begin
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    # Finished run: one quorum parent plus exactly one subtree per original slot.
    root = Ref{Any}(nothing)
    out = _Treebars.with_progress(:state; description="probe") do probe
        root[] = probe
        completion_warmup_mcmc([Xoshiro(31), Xoshiro(32)], p;
            n_draws=40, min_completed=2, progress=probe)
    end
    @test out.completion.stop_reason === :all_completed
    done = _Treebars.render_text(root[])
    @test occursin("MCMC (2/2)", done)
    @test occursin("all_completed: 2 of 2 chains completed", done)
    mcmc_lines = filter(l -> occursin("MCMC", l), split(done, '\n'))
    @test length(mcmc_lines) == 3  # parent plus MCMC.1 and MCMC.2, no same-named duplicates
    @test count("MCMC.1", done) == 1
    @test count("MCMC.2", done) == 1
    @test occursin("finished", done) && occursin("2 of 2", done)
    @test occursin("stop requested", done)
    @test occursin("stopped", done)
    @test occursin("not started", done)
    @test occursin("failed", done)
    @test occursin("stuck", done)
    @test occursin("awaiting reference", done)
    # Live run: the first callback parks one chain behind a gate while the main
    # task snapshots the mid-flight tree — labelled subtrees and quorum state
    # must already be visible, before any chain has completed.
    entered = Channel{Nothing}(1)
    release = Channel{Nothing}(1)
    gated = Ref(false)
    glock = ReentrantLock()
    gate_callback = (state, stage) -> begin
        take_gate = lock(glock) do
            if !gated[]
                gated[] = true
                true
            else
                false
            end
        end
        take_gate && (put!(entered, nothing); take!(release))
        false
    end
    live_root = Ref{Any}(nothing)
    task = Threads.@spawn _Treebars.with_progress(:state; description="live") do probe
        live_root[] = probe
        completion_warmup_mcmc([Xoshiro(41), Xoshiro(42)], p;
            n_draws=40, min_completed=1, progress=probe, callback=gate_callback)
    end
    fired = timedwait(() -> isready(entered), 120)
    @test fired === :ok
    live = ""
    if fired === :ok
        take!(entered)
        live = _Treebars.render_text(live_root[])
    end
    put!(release, nothing)
    live_out = fetch(task)
    @test live_out.completion.quorum_met
    if fired === :ok
        @test occursin(r"MCMC \([01]/1\)", live)
        @test occursin(r"MCMC\.[12]", live)
        @test occursin("stop requested", live)
        @test occursin(r"[01] of 1", live)
    end
    # Over-quorum completions stay exact in the labels: the rendered counter
    # clamps at N, so the `finished` label (not the bar) is the source of truth.
    s = WarmupHMC._CompletionState(3; clock=() -> 0.0, min_completed=2)
    s.quorum_at = 1.0
    s.statuses .= :completed
    s.results .= [_completion_fixture(i) for i in 1:3]
    s.finished_at .= 1.0
    over_root = Ref{Any}(nothing)
    _Treebars.with_progress(:state; description="over") do probe
        over_root[] = probe
        _Treebars.with_progress(probe, 2; description="MCMC") do c
            WarmupHMC._update_completion_progress!(c, s)
        end
    end
    over = _Treebars.render_text(over_root[])
    @test occursin("MCMC (2/2)", over)
    @test occursin("3 of 2", over)
end
end
