@testitem "completion_warmup_mcmc" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Pkg, TOML
    using Distributions
    using DifferentiationInterface, Enzyme
    using JSON

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner; using LinearAlgebra;
# BLAS.set_num_threads(1); TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "completion_policy.jl"))'

_completion_fixture(i; n=5) = (; posterior_position=fill(Float64(i), 2, n),
    n_divergent_samples=i % 2, stepsize=1.0)

# How often `f()` logs the runtime-selection caveat, and what `f()` returned.
# Other warnings (an initializer's retry, say) are not counted.
function _selection_warnings(f)
    logs, value = Test.collect_test_logs(f; min_level=Base.CoreLogging.Warn)
    count(l -> l.message == WarmupHMC._COMPLETION_SELECTION_WARNING, logs), value
end

_run_json(path) = JSON.parsefile(path)

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
    # A failed chain's first error line is in the quorum error itself.
    @test occursin("; chain 1 failed: failed initialization",
                   sprint(showerror, WarmupHMC.CompletionQuorumError(unmet)))
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
# These two checks are judged here without the mass check, which has its own tests.
const _ADMISSION_CHECKS = WarmupHMC._CompletionAdmission(10.0, 0.1, Inf)

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
    # Every slot enters its round before any returns: otherwise chains 1, 2 and
    # 4 can meet the quorum before slot 3's task starts, which then never runs.
    entered = Channel{Int}(4)
    release = Channel{Nothing}(4)
    batch = Threads.@spawn WarmupHMC._completion_batch(4; min_completed=3,
            admission=_ADMISSION_CHECKS) do i, stop
        put!(entered, i)
        take!(release)
        _admission_fixture(0.17; n_divergent=i == 3 ? 3 : 0), true
    end
    foreach(_ -> take!(entered), 1:4)
    foreach(_ -> put!(release, nothing), 1:4)
    out = fetch(batch)
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
            admission=WarmupHMC._CompletionAdmission(Inf, 1.0, Inf)) do i, stop
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

# Snag default-starts-t-013b07f7, decision 1csfn2g: a chain can also settle in a
# minor mode far below its siblings, at an ordinary step size and without
# diverging. A fixture's estimate is `log_mass` with a standard error at
# rounding level: its log density is constant and its one coordinate
# alternates ±1.
function _mass_fixture(log_mass; n=20, n_divergent=0, stepsize=0.17)
    log_volume = log(n / (n - 1)) / 2
    (; posterior_position=reshape(repeat([-1.0, 1.0], n ÷ 2), 1, n),
       posterior_log_density=fill(log_mass - log_volume, n), n_divergent_samples=n_divergent, stepsize)
end
_mass(log_mass; se=1.0) = (; log_mass, se)
# A 2-D funnel in its centred (model) frame: y2 | y1 ~ N(0, exp(y1)^2).
struct _MassFunnel end
LogDensityProblems.dimension(::_MassFunnel) = 2
LogDensityProblems.capabilities(::Type{_MassFunnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_MassFunnel, y) = -y[1]^2 / 2 - (y[2] * exp(-y[1]))^2 / 2 - y[1]
const _MASS_CHECK = WarmupHMC._CompletionAdmission(Inf, 0.1, 25.0)

@testset "the mass estimate does not mistake a wide mode for a minor one" begin
    # Two equal-mass Gaussian modes, of widths 1/2 and 2 in 20 dimensions: the
    # wide one's mean log density is lower by 20 log 4 ≈ 27.7, its mass is not.
    d, n = 20, 4000
    rng = Xoshiro(7)
    estimate(sigma) = begin
        x = sigma .* randn(rng, d, n)
        lp = [-sum(abs2, c) / (2sigma^2) - d * log(sigma) - d * log(2π) / 2 for c in eachcol(x)]
        WarmupHMC._log_mass_estimate(nothing, x, lp), mean(lp)
    end
    (narrow, narrow_lp), (wide, wide_lp) = estimate(0.5), estimate(2.0)
    @test narrow_lp - wide_lp > 25
    @test abs(narrow.log_mass - wide.log_mass) < 1
    @test 0 < narrow.se < 0.2 && 0 < wide.se < 0.2
    # A mode holding e^-30 of the mass sits 30 below.
    x = randn(rng, d, n)
    lp = [-sum(abs2, c) / 2 - d * log(2π) / 2 - 30 for c in eachcol(x)]
    @test WarmupHMC._log_mass_estimate(nothing, x, lp).log_mass ≈ narrow.log_mass - 30 atol=1
    # Too few draws: no estimate.
    @test isnothing(WarmupHMC._log_mass_estimate(nothing, x[:, 1:10], lp[1:10]))
    @test_throws DimensionMismatch WarmupHMC._log_mass_estimate(nothing, x, lp[1:end-1])
end

@testset "equal-weight modes with different correlations can score 20 nats apart" begin
    # Draw from each component of a 50:50 Gaussian mixture. Their centres are
    # 100 marginal SDs apart, so the other component contributes below machine
    # precision at these draws. Common random numbers isolate the volume effect.
    pairs, draws, rho = 113, 500, 0.55
    z = randn(Xoshiro(42), 2pairs, draws)
    function component_score(correlation, centre)
        positions = copy(z)
        residual_scale = sqrt(1 - correlation^2)
        @views positions[2:2:end, :] .= correlation .* z[1:2:end, :] .+ residual_scale .* z[2:2:end, :]
        positions[1, :] .+= centre
        # Both components have the same weight and unit marginal variances.
        # Constants common to both log densities cancel in the comparison.
        lp = [-sum(abs2, draw) / 2 - pairs * log(residual_scale) for draw in eachcol(z)]
        WarmupHMC._log_mass_estimate(nothing, positions, lp)
    end
    independent = component_score(0.0, -50.0)
    correlated = component_score(rho, 50.0)
    expected_gap = -pairs * log1p(-rho^2) / 2
    @test correlated.log_mass - independent.log_mass ≈ expected_gap atol=1

    comparison = (; log_mass=independent.log_mass, log_mass_se=independent.se,
        reference_log_mass=correlated.log_mass, reference_log_mass_se=correlated.se)
    default_policy = WarmupHMC._CompletionAdmission(Inf, 0.1, WarmupHMC._DEFAULT_MAX_ADMITTED_LOG_MASS_GAP)
    @test !WarmupHMC._low_mass(comparison, default_policy)
    @test WarmupHMC._low_mass(comparison, WarmupHMC._CompletionAdmission(Inf, 0.1, 15.0))
end

@testset "the mass estimate maps sampling-frame draws into the model frame" begin
    # A 2-D funnel sampled non-centred: the source frame is a standard normal and
    # the model frame is y = (x1, x2 exp(x1)). Estimating from the source draws
    # with their source log density, through the map, must equal estimating from
    # the model draws with the model's own log density.
    funnel(y) = LogDensityProblems.logdensity(_MassFunnel(), y)
    rp = ReparametrizedProblem(IndexedReparametrization([
            2 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), 0.0, x -> x[1])]),
        _MassFunnel(), AutoEnzyme())
    x = randn(Xoshiro(3), 2, 500)
    source_lp = [LogDensityProblems.logdensity(rp, c) for c in eachcol(x)]
    y = vcat(x[1:1, :], (x[2, :] .* exp.(x[1, :]))')
    model = WarmupHMC._log_mass_estimate(nothing, y, [funnel(c) for c in eachcol(y)])
    mapped = WarmupHMC._log_mass_estimate(WarmupHMC.reparametrizer(rp), x, source_lp)
    @test mapped.log_mass ≈ model.log_mass
    @test mapped.se ≈ model.se
    # Unmapped, the source frame's estimate is a different number.
    @test abs(WarmupHMC._log_mass_estimate(nothing, x, source_lp).log_mass - model.log_mass) > 0.1
end

@testset "a chain far below its siblings' mass does not count" begin
    s = WarmupHMC._CompletionState(4; clock=() -> 0.0, min_completed=2, admission=_MASS_CHECK)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._report_boundary!(stop1, 0.17, 30, _mass(100.0))
        # Chain 3 finishes 40 below chain 1's estimate, at a healthy step size
        # and without divergences.
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(60.0), true), s, 3)
        @test s.statuses[3] === :stuck
        @test s.admissions[3].stuck_reason === :low_mass
        @test s.admissions[3].log_mass ≈ 60.0
        @test s.admissions[3].log_mass_se < 1e-12
        @test s.admissions[3].reference_log_mass == 100.0
        @test s.admissions[3].reference_log_mass_se == 1.0
        # 10 below is within the 25-nat bound.
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(90.0), true), s, 2)
        @test s.statuses[2] === :completed
        @test s.admissions[2].reference_log_mass == 100.0
        # Counted, chain 3 would have met the quorum of two with chain 2 and
        # stopped this chain. It does not count.
        @test !stop1()
        _mass_fixture(100.0), true
    end
    @test s.statuses[1] === :completed
    # Chain 1 is compared with chain 2, the best sibling not left out.
    @test s.admissions[1].reference_log_mass ≈ 90.0
    WarmupHMC._completion_worker!((_, _) -> error("the quorum is met"), s, 4)
    @test s.statuses == [:completed, :completed, :stuck, :not_started]
end

@testset "the gap must exceed both bounds" begin
    # 30 below: beyond 25 nats. With the reference's standard error at 10, 30 is
    # within 4 standard errors and the chain is kept; at 5 it is left out.
    for (se, verdict) in ((10.0, :completed), (5.0, :stuck))
        s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1, admission=_MASS_CHECK)
        WarmupHMC._completion_worker!(s, 1) do _, stop1
            WarmupHMC._report_boundary!(stop1, 0.17, 30, _mass(100.0; se))
            WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(70.0), true), s, 2)
            @test s.statuses[2] === verdict
            _mass_fixture(100.0; n=5), false
        end
    end
    # A larger bound keeps a 40-nat gap.
    s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1,
        admission=WarmupHMC._CompletionAdmission(Inf, 0.1, 50.0))
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._report_boundary!(stop1, 0.17, 30, _mass(100.0))
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(60.0), true), s, 2)
        @test s.statuses[2] === :completed
        _mass_fixture(100.0; n=5), false
    end
end

@testset "a finished chain waits, uncounted, for a sibling's mass" begin
    # The step-size check is on too: its comparison is made once, at the first
    # sibling boundary that keeps draws, and is not repeated while the chain
    # waits for a mass to compare with.
    checks = WarmupHMC._CompletionAdmission(10.0, 0.1, 25.0)
    s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1, admission=checks)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(60.0), true), s, 2)
        @test s.statuses[2] === :awaiting_reference
        # Draws but too few for an estimate: the step size is compared, the mass is not.
        WarmupHMC._report_boundary!(stop1, 0.15, 8, nothing)
        @test s.statuses[2] === :awaiting_reference
        @test s.admissions[2].reference_stepsize == 0.15
        @test isnothing(s.admissions[2].reference_log_mass)
        # A boundary with an estimate settles it.
        WarmupHMC._report_boundary!(stop1, 0.9, 30, _mass(100.0))
        @test s.statuses[2] === :stuck
        @test s.admissions[2].stuck_reason === :low_mass
        @test s.admissions[2].reference_stepsize == 0.15
        @test !stop1()
        _mass_fixture(100.0), true
    end
    # Chain 1 has no sibling left to compare with, so it is admitted unchecked.
    @test s.statuses == [:completed, :stuck]

    # Once no sibling can supply a mass, the waiting chain is admitted unchecked.
    s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1, admission=_MASS_CHECK)
    WarmupHMC._completion_worker!(s, 1) do _, _
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(60.0), true), s, 2)
        @test s.statuses[2] === :awaiting_reference
        error("chain 1 fails before it keeps draws")
    end
    @test s.statuses == [:failed, :completed]
    @test isnothing(s.admissions[2].reference_log_mass)
    # A chain without an estimate of its own (at most 10 draws) is not compared.
    s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1, admission=_MASS_CHECK)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._report_boundary!(stop1, 0.17, 30, _mass(100.0))
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(-1e6; n=10), true), s, 2)
        @test s.statuses[2] === :completed
        @test isnothing(s.admissions[2].log_mass)
        _mass_fixture(100.0; n=5), false
    end
end

@testset "a stuck sibling's mass is no reference" begin
    s = WarmupHMC._CompletionState(3; clock=() -> 0.0, min_completed=1, admission=_MASS_CHECK)
    WarmupHMC._completion_worker!(s, 1) do _, stop1
        WarmupHMC._report_boundary!(stop1, 0.17, 30, _mass(100.0))
        # Chain 3 is left out for its divergences. Its large estimate must not
        # make chain 2 look minor.
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(1000.0; n_divergent=10), true), s, 3)
        @test s.admissions[3].stuck_reason === :divergence
        WarmupHMC._completion_worker!((_, _) -> (_mass_fixture(95.0), true), s, 2)
        @test s.statuses[2] === :completed
        @test s.admissions[2].reference_log_mass == 100.0
        _mass_fixture(100.0), true
    end
    @test s.statuses == [:completed, :completed, :stuck]
end

@testset "resume judges mass against every checkpoint before scheduling" begin
    # Slot 3's checkpoint keeps draws whose estimate is 101 but is not complete.
    initial = Dict(1 => _mass_fixture(60.0), 2 => _mass_fixture(100.0))
    out = WarmupHMC._completion_batch(3; min_completed=1, admission=_MASS_CHECK,
            initial_results=initial, initial_stepsizes=Dict(3 => 0.18),
            initial_log_masses=Dict(3 => _mass(101.0))) do i, stop
        error("No incomplete worker should start after quorum.")
    end
    c = out.completion
    @test c.stuck_chain_indices == [1]
    @test c.chains[1].stuck_reason === :low_mass
    @test c.chains[1].reference_log_mass == 101.0
    @test c.chains[2].reference_log_mass == 101.0
    @test c.completed_chain_indices == [2]
    @test c.chains[3].status === :not_started
    @test isnothing(c.chains[3].log_mass)
    @test c.max_admitted_log_mass_gap == 25.0
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

# Targets for the real-chain tests of decision 1csfn2g. These must not depend on
# exact trajectories, which differ across platforms and Julia versions: a
# side-lobe trap of the calibration's sinusoid regression held its chain on the
# development host but not on CI (run 43728333809).
#
# Two unit Gaussian modes 2m apart along coordinate 1, the one at +m holding
# exp(log_weight) of the mass. A chain started in the small mode cannot cross
# the barrier between them, and it samples a unit Gaussian like its siblings:
# the same step size, no divergences.
struct CompletionMinorMode
    d::Int
    m::Float64
    log_weight::Float64
end
LogDensityProblems.dimension(t::CompletionMinorMode) = t.d
LogDensityProblems.capabilities(::Type{CompletionMinorMode}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(t::CompletionMinorMode, x)
    c1, c2 = copy(x), copy(x)
    c1[1] += t.m; c2[1] -= t.m
    l1 = -0.5 * sum(abs2, c1)
    l2 = t.log_weight - 0.5 * sum(abs2, c2)
    lp = WarmupHMC.LogExpFunctions.logaddexp(l1, l2)
    lp, -(exp(l1 - lp) .* c1 .+ exp(l2 - lp) .* c2)
end
LogDensityProblems.logdensity(t::CompletionMinorMode, x) = first(LogDensityProblems.logdensity_and_gradient(t, x))

# Two equal-mass Gaussian modes: N(-m e1, s1^2 I) and N(+m e1, s2^2 I).
struct CompletionMix2
    d::Int
    m::Float64
    s1::Float64
    s2::Float64
end
LogDensityProblems.dimension(t::CompletionMix2) = t.d
LogDensityProblems.capabilities(::Type{CompletionMix2}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(t::CompletionMix2, x)
    c1, c2 = copy(x), copy(x)
    c1[1] += t.m; c2[1] -= t.m
    l1 = log(0.5) - t.d * log(t.s1) - 0.5 * sum(abs2, c1) / t.s1^2
    l2 = log(0.5) - t.d * log(t.s2) - 0.5 * sum(abs2, c2) / t.s2^2
    lp = WarmupHMC.LogExpFunctions.logaddexp(l1, l2)
    w1, w2 = exp(l1 - lp), exp(l2 - lp)
    lp, -(w1 .* c1 ./ t.s1^2 .+ w2 .* c2 ./ t.s2^2)
end
LogDensityProblems.logdensity(t::CompletionMix2, x) = first(LogDensityProblems.logdensity_and_gradient(t, x))

# Neal's centred funnel: v ~ N(0, 3^2), x_i | v ~ N(0, exp(v)), coordinates (v, x...).
struct CompletionFunnel
    d::Int
end
LogDensityProblems.dimension(t::CompletionFunnel) = t.d
LogDensityProblems.capabilities(::Type{CompletionFunnel}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(t::CompletionFunnel, x)
    v = x[1]; z = @view x[2:end]; k = length(z); q = sum(abs2, z)
    lp = -v^2 / 18 - k * v / 2 - q * exp(-v) / 2
    g = similar(x); g[1] = -v / 9 - k / 2 + q * exp(-v) / 2; g[2:end] .= -z .* exp(-v)
    lp, g
end
LogDensityProblems.logdensity(t::CompletionFunnel, x) = first(LogDensityProblems.logdensity_and_gradient(t, x))

_completion_outcome(f) = try
    f()
catch e
    e isa WarmupHMC.CompletionQuorumError || rethrow()
    e.outcome
end

@testset "a real chain in a minor mode is left out for its mass alone" begin
    BLAS.set_num_threads(1)
    # The small mode holds e^-60 of the mass; chain 4 starts in it, the others in
    # the large one, each at its mode with the unit metric (no Pathfinder run).
    p = CompletionMinorMode(10, 15.0, -60.0)
    start(x1) = (; position=[x1; zeros(9)], squared_scale=Diagonal(ones(10)))
    inits = [start(-15.0), start(-15.0), start(-15.0), start(15.0)]
    sample(; kwargs...) = _completion_outcome(() -> completion_warmup_mcmc(
        [Xoshiro(300 + i) for i in 1:4], p; init=inits, min_completed=4, n_draws=1000, kwargs...))
    # Without the mass check both other checks admit it: the snag's case.
    unchecked = sample(; max_admitted_log_mass_gap=Inf)
    @test unchecked.completion.completed_chain_indices == 1:4
    @test [mean(r.posterior_position[1, :]) > 0 for r in unchecked.results] == [false, false, false, true]
    c = sample().completion
    @test !c.quorum_met
    @test c.completed_chain_indices == [1, 2, 3]
    @test c.stuck_chain_indices == [4]
    @test c.chains[4].stuck_reason === :low_mass
    @test c.chains[4].divergent_fraction <= 0.1
    @test c.chains[4].stepsize >= c.chains[4].reference_stepsize / 10
    @test c.chains[4].reference_log_mass - c.chains[4].log_mass ≈ 60 atol=3
    @test all(i -> !isnothing(c.chains[i].reference_log_mass), 1:4)
end

@testset "real chains in equal-mass modes of unequal widths are kept" begin
    BLAS.set_num_threads(1)
    m = CompletionMix2(20, 12.0, 0.5, 2.0)
    inits = [[-12.0; zeros(19)], [-12.0; zeros(19)], [12.0; zeros(19)], [12.0; zeros(19)]]
    out = completion_warmup_mcmc([Xoshiro(100 + i) for i in 1:4], m; init=inits, n_draws=1000)
    c = out.completion
    @test c.completed_chain_indices == 1:4
    # Each chain stays in its mode, and the wide mode's mean log density is more
    # than 25 below the narrow one's: compared by density, it would be left out.
    @test [mean(r.posterior_position[1, :]) > 0 for r in out.results] == [false, false, true, true]
    @test mean(out.results[1].posterior_log_density) - mean(out.results[3].posterior_log_density) > 25
    @test maximum(ch -> ch.log_mass, c.chains) - minimum(ch -> ch.log_mass, c.chains) < 1
end

@testset "real centred-funnel chains are kept" begin
    BLAS.set_num_threads(1)
    # The other two checks are off, so only the mass check could leave one out.
    out = completion_warmup_mcmc([Xoshiro(100 + i) for i in 1:4], CompletionFunnel(50);
        n_draws=1000, stuck_stepsize_ratio=Inf, max_admitted_divergent_fraction=1.0)
    # Its estimates are noisy (calibration: gaps to 49.5 nats within 2.3
    # standard errors; on CI they spread less), and the bound on standard errors
    # keeps every chain.
    c = out.completion
    @test c.completed_chain_indices == 1:4
    @test all(ch -> ch.stuck_reason !== :low_mass, c.chains)
end

@testset "numerical equivalence and recovery" begin
    BLAS.set_num_threads(1)
    p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    cfg = (; n_draws=600, progress=nothing)
    baseline = adaptive_warmup_mcmc([Xoshiro(i) for i in 1:3], p; cfg..., parallel=false)
    # Every chain is required, so none can be selected by runtime: no caveat.
    n_warned, out = _selection_warnings(() -> completion_warmup_mcmc([Xoshiro(i) for i in 1:3], p; cfg...))
    @test n_warned == 0
    @test out.completion.selection_warning == ""
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
                posterior_gradient=full[i].posterior_gradient[:, 1:50],
                posterior_log_density=full[i].posterior_log_density[1:50])), false
        end
        full[i], true
    end
    @test selected.completion.completed_chain_indices == [2, 4]
    @test selected.completion.omitted_chain_indices == [1, 3]
    @test mean(hcat(getproperty.(selected.results, :posterior_position)...) .> 0) == 1.0
    @test selected.completion.n_samples == 200
    @test occursin("bias inference", selected.completion.selection_warning)
end

@testset "the selection caveat follows the outcome" begin
    # Snag completion-warni-282737a4: the caveat says chains WERE selected, so it
    # is recorded only when a kept chain stands beside one that was cut short or
    # never started. Every slot below enters its round before any returns, so
    # none is `:not_started` unless the test makes it so. `plan[i]` is `:full`
    # (reaches `n_draws`), `:short` (stopped at once, as by a callback), `:late`
    # (stopped at the boundary after the quorum) or `:fail`.
    function rendezvous(plan; min_completed, admission=WarmupHMC._ADMIT_EVERY_FULL_CHAIN)
        n = length(plan)
        entered = Channel{Int}(n)
        release = Channel{Nothing}(n)
        batch = Threads.@spawn WarmupHMC._completion_batch(n; min_completed, admission) do i, stop
            put!(entered, i)
            take!(release)
            plan[i] === :fail && error("controlled chain $i failure")
            plan[i] === :full && return _completion_fixture(i), true
            if plan[i] === :late
                while !stop()
                    yield()
                end
            end
            _completion_fixture(i; n=2), false
        end
        foreach(_ -> take!(entered), 1:n)
        foreach(_ -> put!(release, nothing), 1:n)
        fetch(batch)
    end
    caveat = WarmupHMC._COMPLETION_SELECTION_WARNING

    # Every chain required, every chain kept: the reporter's case.
    out = rendezvous([:full, :full, :full]; min_completed=3)
    @test out.completion.stop_reason === :all_completed
    @test out.completion.selection_warning == ""
    # A quorum below the slot count that kept every chain anyway returns exactly
    # the draws of all-chain sampling, so it claims no selection either.
    out = rendezvous([:full, :full, :full]; min_completed=2)
    @test out.completion.stop_reason === :all_completed
    @test out.completion.selection_warning == ""
    # A chain the quorum cut short: the kept ones finished first.
    out = rendezvous([:full, :late, :full]; min_completed=2)
    @test out.completion.stop_reason === :quorum_reached
    @test out.completion.chains[2].status === :stopped
    @test out.completion.selection_warning == caveat
    # A callback or an interrupt cutting a chain short selects just the same.
    out = rendezvous([:full, :short]; min_completed=1)
    @test out.completion.chains[2].status === :stopped
    @test out.completion.selection_warning == caveat
    # A slot that never started, because the resume pre-pass already met quorum.
    out = WarmupHMC._completion_batch((i, stop) -> error("slot $i must not start"), 2;
        min_completed=1, initial_results=Dict{Int,Any}(1 => _completion_fixture(1)))
    @test out.completion.chains[2].status === :not_started
    @test out.completion.selection_warning == caveat
    # Stuck and failed chains are health verdicts, named in their own fields.
    # The odd fixtures are 20% divergent, above this 10% threshold.
    out = rendezvous([:full, :full, :full, :full, :fail]; min_completed=2,
        admission=WarmupHMC._CompletionAdmission(Inf, 0.1, Inf))
    @test out.completion.stuck_chain_indices == [1, 3]
    @test out.completion.failed_chain_indices == [5]
    @test out.completion.stop_reason === :quorum_reached
    @test out.completion.selection_warning == ""
    # Nothing kept, nothing selected: the quorum error carries no caveat.
    out = rendezvous([:short, :short]; min_completed=2)
    @test !out.completion.quorum_met
    @test out.completion.selection_warning == ""
    # The warning prints iff the outcome records the caveat.
    @test_logs (:warn, caveat) WarmupHMC._warn_runtime_selection((; selection_warning=caveat))
    @test_logs WarmupHMC._warn_runtime_selection((; selection_warning=""))
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
        # Every chain was required and none was kept: no caveat in any record.
        @test err.outcome.completion.selection_warning == ""
        @test _run_json(joinpath(err.outcome.completion.attempt_directory,
            "run_manifest.json"))["selection_warning"] == ""
        @test _run_json(joinpath(err.outcome.completion.attempt_directory,
            "run_summary.json"))["selection_warning"] == ""
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
        # step-size and mass checks would make chain 2 wait for it to sample ("a
        # finished chain waits, uncounted"); this run pins admission without them.
        ordinary = adaptive_warmup_mcmc(Xoshiro(999), p;
            n_draws=40, checkpoint_dir=joinpath(dir, "chain_2"), resume=true)
        unchecked = (; stuck_stepsize_ratio=Inf, max_admitted_log_mass_gap=Inf)
        n_warned, out = _selection_warnings(() -> completion_warmup_mcmc([Xoshiro(999), Xoshiro(999)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true, unchecked...))
        @test out.completion.completed_chain_indices == [2]
        @test out.completion.omitted_chain_indices == [1]
        @test out.results[1].posterior_position == ordinary.posterior_position
        @test out.completion.chains[1].status === :not_started
        # Chain 2 is kept and chain 1 never started: a runtime selection, warned
        # once, after the chains settled, and recorded in every run record.
        caveat = WarmupHMC._COMPLETION_SELECTION_WARNING
        @test n_warned == 1
        @test out.completion.selection_warning == caveat
        @test _run_json(joinpath(out.completion.attempt_directory,
            "run_manifest.json"))["selection_warning"] == caveat
        @test _run_json(joinpath(out.completion.attempt_directory,
            "run_summary.json"))["selection_warning"] == caveat
        @test out.completion.schema_version == 4
        @test out.completion.max_admitted_log_mass_gap == Inf
        # The judged chain's own estimate is recorded even with the check off.
        @test isfinite(out.completion.chains[2].log_mass)
        @test isnothing(out.completion.chains[2].reference_log_mass)
        @test occursin("\"chain_log_masses\":[null,",
            read(joinpath(out.completion.attempt_directory, "run_summary.json"), String))
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
        n_warned, again = _selection_warnings(() -> completion_warmup_mcmc([Xoshiro(11), Xoshiro(12)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true, unchecked...,
            callback=(s, stage) -> error("Terminal reopen must never run a callback")))
        @test n_warned == 1
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
        # refused: the mass threshold is part of the recorded policy too.
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true,
            unchecked..., max_admitted_log_mass_gap=3.0)
        # A run recorded before the mass check has no mass threshold: it reopens
        # as recorded whatever `max_admitted_log_mass_gap` is passed.
        pre_mass = (; out.results, completion=(; (k => v for (k, v) in pairs(out.completion)
            if k !== :max_admitted_log_mass_gap)...))
        WarmupHMC._atomic_serialize_manifest(joinpath(dir, "completion_terminal.jls"),
            (; schema_version=2, n_draws=40, outcome=pre_mass))
        n_warned, reopened_pre_mass = _selection_warnings(() -> completion_warmup_mcmc(
            [Xoshiro(1), Xoshiro(2)], p; n_draws=40, min_completed=1, checkpoint_dir=dir,
            resume=true, stuck_stepsize_ratio=Inf, max_admitted_log_mass_gap=3.0))
        @test n_warned == 1
        @test reopened_pre_mass.completion == pre_mass.completion
        # A schema-1 terminal fixture retains the OLD selection semantics,
        # including a full chain omitted after cutoff. It must not be silently
        # reinterpreted under the new policy when reopened.
        legacy_fields = (; (k => v for (k, v) in pairs(out.completion)
            if k ∉ (:schema_version, :stop_policy, :stop_requested_at_seconds,
                :stuck_stepsize_ratio, :max_admitted_divergent_fraction,
                :max_admitted_log_mass_gap, :n_stuck, :stuck_chain_indices))...)
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
        n_warned, old = _selection_warnings(() -> completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], p;
            n_draws=40, min_completed=1, checkpoint_dir=dir, resume=true,
            callback=(s, stage) -> error("Legacy terminal reopen must not sample")))
        # Its recorded caveat is reopened, and warned, as recorded.
        @test n_warned == 1
        @test old.completion == legacy.completion
        @test getproperty.(old.results, :chain_index) == [2]
        @test old.results[1].posterior_position == out.results[1].posterior_position
        @test occursin("\"grace_seconds\":3.0", read(summary, String))
        @test !occursin("chain_stuck_reasons", read(summary, String))
        @test !occursin("chain_log_masses", read(summary, String))
        # Explicit overwrite starts a new run and clears the old selection.
        n_warned, fresh = _selection_warnings(() -> completion_warmup_mcmc([Xoshiro(21), Xoshiro(22)], p;
            n_draws=40, checkpoint_dir=dir, overwrite=true))
        @test fresh.completion.completed_chain_indices == [1, 2]
        @test fresh.completion.n_samples == 80
        # Every chain required and kept: no caveat, and none on reopening it.
        @test n_warned == 0
        @test fresh.completion.selection_warning == ""
        @test _run_json(joinpath(fresh.completion.attempt_directory,
            "run_manifest.json"))["selection_warning"] == ""
        n_warned, reopened = _selection_warnings(() -> completion_warmup_mcmc([Xoshiro(21), Xoshiro(22)], p;
            n_draws=40, checkpoint_dir=dir, resume=true))
        @test n_warned == 0
        @test reopened.completion == fresh.completion
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
    # refused: a negative gap would leave out chains above the best sibling
    # (decision 1csfn2g).
    for value in (-1.0, NaN, true, "25")
        @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1)], p;
            max_admitted_log_mass_gap=value)
    end
    @test :stuck_stepsize_ratio in WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
    @test :max_admitted_divergent_fraction in WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
    @test :max_admitted_log_mass_gap in WarmupHMC._SAMPLER_KWARGS[:completion_warmup_mcmc]
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
