@testitem "nonfinite candidate losses never win the centering selection" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, Statistics
    using LogDensityProblems
    using DifferentiationInterface, Enzyme

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "nonfinite_candidate_losses.jl"))'

# Snag `nonlinear-center-dd5bbc3d`: a scorer whose candidate observations
# underflowed (a BRM HSGP cell with log-scale -426) returned NaN losses for the
# candidates t >= 0.9, and `argmin` — which prefers `NaN` — committed t = 0.9.

# A stub accumulator with a fixed loss, to pin `minimizer` on exact values.
struct NCFixedLoss
    loss::Float64
end
WarmupHMC.reparametrization_loss(l::NCFixedLoss; kwargs...) = l.loss
nc_fixed(losses) = WarmupHMC.OnlineReparametrizer([
    WarmupHMC.PartiallyCentered(t) => NCFixedLoss(l)
    for (t, l) in zip(range(0, 1, 11), losses)
])

# One accumulator per grid candidate, fed `observe(t, i)` for draws `i in 1:n`.
function nc_reparametrizer(observe; n=50)
    or = WarmupHMC.OnlineReparametrizer(WarmupHMC.PartiallyCentered(1.0);
        accumulator=WarmupHMC.WeightedReparametrizationLoss)
    for i in 1:n, (candidate, accumulator) in or.pairs
        WarmupHMC.OnlineStatsBase.fit!(accumulator, observe(candidate.c, i))
    end
    or
end
nc_losses(or; kwargs...) = [WarmupHMC.reparametrization_loss(last(p); kwargs...) for p in or.pairs]

@testset "minimizer: only finite losses compete" begin
    rng = Xoshiro(1)
    grid = collect(range(0, 1, 11))
    # The correlation is most negative at t = 0.2; t >= 0.5 cannot be scored.
    observe(t, _) = t >= 0.5 ? (0.0, NaN, NaN) : begin
        q = randn(rng)
        (0.0, q, -q + abs(t - 0.2) * randn(rng))
    end
    or = nc_reparametrizer(observe)
    losses = nc_losses(or)
    @test all(isnan, losses[grid .>= 0.5]) && all(isfinite, losses[grid .< 0.5])
    @test grid[argmin(losses)] == 0.5             # what selection used to commit
    @test first(WarmupHMC.minimizer(or)).c == 0.2

    # No finite loss: no winner, and the coordinate keeps its centering.
    none = nc_reparametrizer((t, _) -> (0.0, NaN, NaN))
    @test isnothing(WarmupHMC.minimizer(none))
    excluded = Pair{Any,Bool}[]
    current = WarmupHMC.PartiallyCentered(0.7)
    @test WarmupHMC._minimizing_source!(excluded, none, 3, current) === current
    @test WarmupHMC._minimizing_source!(excluded, or, 4, current).c == 0.2
    @test excluded == [3 => true, 4 => false]
    @test_logs (:warn, r"nonfinite loss") WarmupHMC._warn_excluded_candidates(excluded)

    # Every nonfinite loss is excluded, -Inf included; Inf never beats a finite one.
    @test first(WarmupHMC.minimizer(nc_fixed([NaN, -Inf, Inf, 0.4, -0.3, NaN,
        -0.3, 0.1, -Inf, Inf, 0.0]))).c == 0.4
    @test first(WarmupHMC.minimizer(nc_fixed([Inf, -Inf, NaN, Inf, Inf, Inf,
        Inf, Inf, Inf, Inf, 2.0]))).c == 1.0
    @test isnothing(WarmupHMC.minimizer(nc_fixed([-Inf, Inf, NaN, -Inf, Inf, NaN,
        NaN, NaN, NaN, NaN, NaN])))
    # Ties resolve as under `argmin`, which orders `-0.0` before `0.0`.
    signed_zero = nc_fixed([0.0, -0.0, fill(1.0, 9)...])
    @test first(WarmupHMC.minimizer(signed_zero)).c == 0.1 ==
          first(argmin(p -> WarmupHMC.reparametrization_loss(last(p)), signed_zero.pairs)).c

    # All-finite losses select exactly as `argmin` did, ties included.
    for seed in 1:20
        r = Xoshiro(seed)
        finite = nc_reparametrizer((t, _) -> (0.0, randn(r), randn(r)); n=5)
        @test WarmupHMC.minimizer(finite) ===
              argmin(p -> WarmupHMC.reparametrization_loss(last(p)), finite.pairs)
    end
    # Every candidate sees the same draws, so every loss is equal: the first
    # candidate wins, as under `argmin`.
    q, g = randn(rng, 5), randn(rng, 5)
    tied = nc_reparametrizer((t, i) -> (0.0, q[i], g[i]); n=5)
    @test allequal(nc_losses(tied)) && all(isfinite, nc_losses(tied))
    @test first(WarmupHMC.minimizer(tied)).c == 0.0
end

# Coordinates 2:4 are centered on 0 with log-scale x[1]; pair 1 cannot be scored
# from t = 0.5 on, pair 2 cannot be scored at all, pair 3 is healthy.
nc_ir() = WarmupHMC.IndexedReparametrization([
    idx => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(1.0), WarmupHMC.PartiallyCentered(1.0),
        0.0, x -> x[1],
    ) for idx in 2:4
])
nc_plan() = WarmupHMC.CandidateScoringPlan(
    (ir, position, gradient) -> nothing,
    (frame, pair_number, idx, value, candidate) ->
        pair_number == 2 || (pair_number == 1 && candidate.c >= 0.5) ?
            (0.0, NaN, NaN) : nothing,
)

@testset "online selection with a scoring plan" begin
    rng = Xoshiro(2)
    ir = nc_ir()
    rp = WarmupHMC.ReparametrizedProblem(ir, DiagGaussian(zeros(4), ones(4)),
        AutoEnzyme(); scoring_plan=nc_plan())
    recorder = WarmupHMC.NonlinearRecorder(rp; mode=:all_good_leaves)
    for _ in 1:40
        position = randn(rng, 4)
        WarmupHMC.OnlineStatsBase.fit!(ir, recorder.online, position,
            -position .+ 0.3 .* randn(rng, 4);
            scoring_plan=WarmupHMC.candidate_scoring_plan(rp))
    end
    rows = WarmupHMC.candidate_scoring_losses((; nonlinear_recorder=recorder))
    finite_winner(k) = begin
        pair = filter(r -> r.pair_number == k && isfinite(r.loss), rows)
        pair[argmin([r.loss for r in pair])].candidate
    end
    @test any(r -> r.pair_number == 1 && isnan(r.loss), rows)
    @test all(r -> isnan(r.loss), filter(r -> r.pair_number == 2, rows))
    expected = [finite_winner(1), 1.0, finite_winner(3)]
    @test expected[1] < 0.5

    @test_logs (:warn, r"nonfinite loss") WarmupHMC.optimize!(ir, recorder.online)
    @test [value.source.c for (_, value) in ir.pairs] == expected
end

@testset "a restarting window never commits a nonfinite candidate" begin
    ir = nc_ir()
    rp = WarmupHMC.ReparametrizedProblem(ir, DiagGaussian(zeros(4), ones(4)),
        AutoEnzyme(); scoring_plan=nc_plan())
    # `variance_cond_target=1.0` restarts every window, so selection runs at
    # every boundary.
    result = adaptive_warmup_mcmc(Xoshiro(3), rp; n_draws=300, progress=nothing,
        variance_cond_target=1.0)
    @test all(isfinite, result.posterior_position)
    sources = [value.source.c for (_, value) in ir.pairs]
    @test sources[1] < 0.5        # used to commit 0.5, the first NaN candidate
    @test sources[2] == 1.0       # used to commit 0.0; nothing could be scored
end
end
