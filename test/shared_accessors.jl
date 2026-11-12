@testitem "Shared accessor work is evaluated once per run of pairs" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra
    using LogDensityProblems
    const OnlineStatsBase = WarmupHMC.OnlineStatsBase  # not a test dependency of its own
    using DifferentiationInterface, Enzyme

# Snag use-the-public-i-0660224b: an accessor shared by N pairs (one mean, one
# vector of locations from a model graph) was evaluated once per pair on every
# path, O(N) work each, so O(N^2) per map, gradient, transport and selection.
# A run of consecutive pairs that share it evaluates it once. The results equal
# the pair-by-pair evaluation: exactly for the maps and the selection, to
# rounding for the derivatives.

const CALLS = Ref(0)
tick!() = (CALLS[] += 1; nothing)
# Counted in the primal; Enzyme does not differentiate the counter.
Enzyme.EnzymeRules.inactive(::typeof(tick!), args...) = nothing
counted(f) = (CALLS[] = 0; f(); CALLS[])

# The reporter's map: coordinates n+1:2n are centred on the mean of 1:n, one
# accessor object shared by every pair.
struct SharedMean <: Function
    n::Int
end
(a::SharedMean)(x) = (tick!(); sum(view(x, 1:a.n)) / a.n)
# One vector of locations, element i for coordinate n + i, from one O(n) pass.
struct Locations <: Function
    n::Int
end
(a::Locations)(x) = (tick!(); sum(view(x, 1:a.n)) / a.n .+ 0.25 .* view(x, 1:a.n))
# A log-scale every pair shares, read from a coordinate outside the run.
struct LogScale <: Function end
(::LogScale)(x) = 0.3 * x[1]

# The same maps with an accessor per pair, which nothing shares: the
# pair-by-pair evaluation the runs must reproduce.
struct OwnMean <: Function
    n::Int
    pair::Int
end
(a::OwnMean)(x) = sum(view(x, 1:a.n)) / a.n
struct OwnLocation <: Function
    n::Int
    pair::Int
end
(a::OwnLocation)(x) = (sum(view(x, 1:a.n)) / a.n .+ 0.25 .* view(x, 1:a.n))[a.pair]
struct OwnLogScale <: Function
    pair::Int
end
(::OwnLogScale)(x) = 0.3 * x[1]

const PC = PartiallyCentered
rule(target, source, location, log_scale) =
    Reparametrization(PC(target), PC(source), location, log_scale)
shared_mean(n; t=0.0, s=1.0) =
    IndexedReparametrization([n + i => rule(t, s, SharedMean(n), LogScale()) for i in 1:n])
own_mean(n; t=0.0, s=1.0) =
    IndexedReparametrization([n + i => rule(t, s, OwnMean(n, i), OwnLogScale(i)) for i in 1:n])
shared_locations(n; t=0.0, s=1.0) = IndexedReparametrization([
    n + i => rule(t, s, SharedAccessor(Locations(n), i), LogScale()) for i in 1:n])
own_locations(n; t=0.0, s=1.0) = IndexedReparametrization([
    n + i => rule(t, s, OwnLocation(n, i), OwnLogScale(i)) for i in 1:n])

struct StdNormal
    d::Int
end
LogDensityProblems.dimension(p::StdNormal) = p.d
LogDensityProblems.capabilities(::Type{StdNormal}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::StdNormal, y) = -dot(y, y) / 2
LogDensityProblems.logdensity_and_gradient(::StdNormal, y) = (-dot(y, y) / 2, -y)
problem(ir, n) = ReparametrizedProblem(ir, StdNormal(2n), AutoEnzyme())

one_pass(ir, x, g) = invoke(WarmupHMC._reparam_gradient, Tuple{Any,Any,Any,Any},
                            AutoEnzyme(), ir, x, g)
agrees(a, b) = isapprox(a, b; rtol=1e-12)

function transport(ir_old, ir_new, n, P, G)
    P2, G2 = copy(P), copy(G)
    WarmupHMC._jointly_transport_halo!(problem(ir_new, n), ir_old, P, G, P2, G2)
    P2, G2
end

@testset "SharedAccessor reads element i of one evaluation" begin
    a = SharedAccessor(Locations(3), 2)
    x = [1.0, 2.0, 4.0]
    @test a(x) == Locations(3)(x)[2]
    @test WarmupHMC._share_key(a) === Locations(3)
end

@testset "run boundaries" begin
    ir = shared_locations(4)
    @test WarmupHMC._run_end(ir.pairs, 1) == 4
    @test WarmupHMC._run_start(ir.pairs, 4) == 1
    # Decreasing indices, distinct accessors, or two blocks interleaved end a run.
    @test WarmupHMC._run_end(reverse(ir.pairs), 1) == 1
    @test WarmupHMC._run_end(own_locations(4).pairs, 1) == 1
    other = rule(0.0, 1.0, SharedAccessor(Locations(3), 1), LogScale())
    mixed = [ir.pairs[1], ir.pairs[2], 7 => other, 8 => other]
    @test [WarmupHMC._run_end(mixed, i) for i in (1, 3)] == [2, 4]
end

for (label, shared, own) in (("one shared mean", shared_mean, own_mean),
                             ("a shared vector of locations", shared_locations, own_locations))
    @testset "$label" begin
        # Constant evaluations per operation, whatever the number of pairs.
        counts = map((8, 64)) do n
            rng = Xoshiro(n)
            x, g = randn(rng, 2n), randn(rng, 2n)
            P, G = randn(rng, 2n, 5), randn(rng, 2n, 5)
            ir = shared(n; s=0.7)
            ors = WarmupHMC.OnlineReparametrizer(ir; accumulator=WarmupHMC.WeightedReparametrizationLoss)
            (; forward = counted(() -> ir(x)),
               inverse = counted(() -> WarmupHMC._inverse_with_logabsdet_jacobian(ir, x)),
               gradient = counted(() -> LogDensityProblems.logdensity_and_gradient(problem(ir, n), x)),
               scoring = counted(() -> OnlineStatsBase.fit!(ir, ors, x, g)),
               pool = counted(() -> WarmupHMC.optimize!(shared(n; s=0.7), copy(P), copy(G))),
               transport = counted(() -> transport(shared(n; s=0.7), shared(n; s=0.2), n, P, G)))
        end
        @test counts[1] == counts[2]
        @test counts[1].forward == 1
        @test counts[1].inverse == 1
        @test counts[1].scoring == 1
        @test counts[1].gradient <= 2
        @test counts[1].pool <= 2 * 5
        @test counts[1].transport <= 6 * 5 + 2

        n = 12
        rng = Xoshiro(2)
        x, g = randn(rng, 2n), randn(rng, 2n)
        for (t, s) in ((0.0, 1.0), (1.0, 0.3), (0.4, 0.4))
            a, b = shared(n; t, s), own(n; t, s)
            # The maps are exactly the pair-by-pair maps.
            @test a(x) == b(x)
            @test WarmupHMC._inverse_with_logabsdet_jacobian(a, x) ==
                  WarmupHMC._inverse_with_logabsdet_jacobian(b, x)
            # The derivatives agree to rounding, under Enzyme pair by pair and in one pass.
            ga = last(LogDensityProblems.logdensity_and_gradient(problem(a, n), x))
            gb = last(LogDensityProblems.logdensity_and_gradient(problem(b, n), x))
            @test agrees(ga, gb)
            @test agrees(WarmupHMC._reparam_gradient(AutoEnzyme(), a, x, g), one_pass(b, x, g))
            @test agrees(one_pass(a, x, g), one_pass(b, x, g))
        end

        # Transport of retained states to new centerings.
        P, G = randn(rng, 2n, 6), randn(rng, 2n, 6)
        Pa, Ga = transport(shared(n; s=0.9), shared(n; s=0.1), n, P, G)
        Pb, Gb = transport(own(n; s=0.9), own(n; s=0.1), n, P, G)
        @test Pa == Pb
        @test agrees(Ga, Gb)

        # Selection: the same scores, the same centerings, the same moved pool.
        P, G = randn(rng, 2n, 40), randn(rng, 2n, 40)
        @test candidate_scoring_losses(problem(shared(n; s=0.5), n), P, G) ==
              candidate_scoring_losses(problem(own(n; s=0.5), n), P, G)
        a, b = shared(n; s=0.5), own(n; s=0.5)
        Pa, Ga, Pb, Gb = copy(P), copy(G), copy(P), copy(G)
        WarmupHMC.optimize!(a, Pa, Ga)
        WarmupHMC.optimize!(b, Pb, Gb)
        @test [last(p).source for p in a.pairs] == [last(p).source for p in b.pairs]
        @test any(p -> last(p).source != PC(0.5), a.pairs)
        @test (Pa, Ga) == (Pb, Gb)

        # A scoring plan that declines every candidate falls back to the direct
        # scorer, which evaluates the shared accessor once per run.
        declining = CandidateScoringPlan((ir, x, g) -> nothing, (frame, args...) -> nothing)
        rp = ReparametrizedProblem(shared(n; s=0.5), StdNormal(2n), AutoEnzyme();
                                   scoring_plan=declining)
        @test candidate_scoring_losses(rp, P, G) ==
              candidate_scoring_losses(problem(shared(n; s=0.5), n), P, G)
        ors = WarmupHMC.OnlineReparametrizer(rp.reparametrizer;
                                             accumulator=WarmupHMC.WeightedReparametrizationLoss)
        @test counted(() -> OnlineStatsBase.fit!(rp.reparametrizer, ors, P[:, 1], G[:, 1];
                                                 scoring_plan=declining)) == 1
    end
end

@testset "adaptive warm-up samples through a shared map" begin
    n = 6
    rp = problem(shared_locations(n; t=0.0, s=1.0), n)
    result = adaptive_warmup_mcmc(Xoshiro(3), rp; n_draws=100, n_evaluations=200,
                                  init=(; position=zeros(2n), squared_scale=ones(2n)))
    @test size(result.posterior_position) == (2n, size(result.posterior_position, 2))
    @test all(isfinite, result.posterior_position)
end
end
