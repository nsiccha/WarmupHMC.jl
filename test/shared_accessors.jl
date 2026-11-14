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

one_pass(ir, x, g) = invoke(WarmupHMC._reparam_gradient, Tuple{Any,Any,Any,Any,Any},
                            AutoEnzyme(), ir, x, nothing, g)
# The extension's gradient, given the map's image of `x` as the sampler gives it.
pairwise(ir, x, g) = WarmupHMC._reparam_gradient(AutoEnzyme(), ir, x, last(ir(x)), g)
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
    # what a run shares: the function, at its stage
    @test WarmupHMC._share_key(a) === (Locations(3), 1)
    @test WarmupHMC._share_key(SharedAccessor(Locations(3), 2, 2)) === (Locations(3), 2)
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
            @test agrees(pairwise(a, x, g), one_pass(b, x, g))
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

@testitem "Accessor work is reused across positions and runs, and skipped by identity pairs" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra
    using LogDensityProblems
    const OnlineStatsBase = WarmupHMC.OnlineStatsBase  # not a test dependency of its own
    using DifferentiationInterface, Enzyme

# Snag centering-evalua-54d5e611 (RKPPL's adaptive centring audit): each accessor
# position was evaluated on its own, so a location and log-scale computed by one
# function ran it twice per run, and a pair whose source and target centering are
# equal still evaluated both. Coordinates 1:n are the hyperparameters; block A is
# n+1:2n and block B 2n+1:3n.

# What a computation is, and how often it ran.
const KINDS = (:energy, :loc_a, :loc_b, :joint, :log_scale, :joint_ab)
const CALLS = zeros(Int, length(KINDS))
tick!(kind) = (CALLS[kind] += 1; nothing)
Enzyme.EnzymeRules.inactive(::typeof(tick!), args...) = nothing
function counted(f)
    fill!(CALLS, 0)
    f()
    copy(CALLS)
end
calls(c, tag) = c[findfirst(==(tag), KINDS)]
none(c) = all(iszero, c)

# One function type for every location and log-scale computation, so the pairs of
# both blocks share one concrete type. `kind` picks what it computes:
#   :energy    [sum(abs2, x[1:n]) / n]
#   :loc_a     block A's locations, energy .+ offsets (a length-n vector)
#   :loc_b     block B's locations, read from block A's coordinates
#   :joint     [energy .+ offsets; energy / 2], locations and log-scale at once
#   :log_scale [energy / 2]
#   :joint_ab  [energy .+ offsets; 2energy .+ offsets; energy / 2], both blocks'
#              locations and their log-scale at once
struct Graph <: Function
    kind::Int
    n::Int
end
Graph(kind::Symbol, n) = Graph(findfirst(==(kind), KINDS), n)
_energy(n, x) = sum(abs2, view(x, 1:n)) / n
_offsets(n) = [-0.3 + 0.7 * (i - 1) / (n - 1) for i in 1:n]
function (g::Graph)(x)
    tick!(g.kind)
    n = g.n
    e = _energy(n, x)
    if g.kind == 1
        [e]
    elseif g.kind == 2
        e .+ _offsets(n)
    elseif g.kind == 3
        0.5 .* x[n+1:2n] .+ 0.1
    elseif g.kind == 4
        vcat(e .+ _offsets(n), [e / 2])
    elseif g.kind == 5
        [e / 2]
    else
        vcat(e .+ _offsets(n), 2e .+ _offsets(n), [e / 2])
    end
end
# The same quantities read by an accessor of their own per pair: the pair-by-pair
# evaluation the shared ones must reproduce.
struct Own <: Function
    kind::Int
    n::Int
    i::Int
end
Own(kind::Symbol, n, i) = Own(findfirst(==(kind), KINDS), n, i)
(a::Own)(x) = Graph(a.kind, a.n)(x)[a.i]

# A frame of endpoint arrays read with a two-part selector, the shape RKPPL's
# joint graph kernel returns (co-report on the snag): family 1 is block A's
# locations, family 2 its log-scale.
struct EndpointFrame
    values::Vector{Vector{Float64}}
end
Base.getindex(f::EndpointFrame, i::CartesianIndex{2}) = f.values[i[1]][i[2]]
struct Endpoints <: Function
    n::Int
end
function (f::Endpoints)(x)
    tick!(4)    # counted as :joint
    e = _energy(f.n, x)
    EndpointFrame([e .+ _offsets(f.n), [e / 2]])
end

const PC = PartiallyCentered
rule(t, s, location, log_scale) = Reparametrization(PC(t), PC(s), location, log_scale)
S(kind, n, i) = SharedAccessor(Graph(kind, n), i)

# A location and a log-scale computed by one function.
same_function(n; t=0.0, s=0.7) =
    IndexedReparametrization([n + i => rule(t, s, S(:energy, n, 1), S(:energy, n, 1)) for i in 1:n])
# Locations and log-scale from one joint computation.
joint(n; t=0.0, s=0.7) = IndexedReparametrization(
    [n + i => rule(t, s, S(:joint, n, i), S(:joint, n, n + 1)) for i in 1:n])
# The same from a frame of endpoint arrays and `CartesianIndex` selectors.
framed(n; t=0.0, s=0.7) = IndexedReparametrization([n + i => rule(t, s,
    SharedAccessor(Endpoints(n), CartesianIndex(1, i)),
    SharedAccessor(Endpoints(n), CartesianIndex(2, 1))) for i in 1:n])
# Two blocks, different locations, one log-scale. Block B's locations read block
# A's coordinates, the dependency the inverse allows.
two_blocks(n; t=0.0, s=0.7, tb=t, sb=s) = IndexedReparametrization([
    [n + i => rule(t, s, S(:loc_a, n, i), S(:log_scale, n, 1)) for i in 1:n];
    [2n + i => rule(tb, sb, S(:loc_b, n, i), S(:log_scale, n, 1)) for i in 1:n]])
# Both blocks' endpoints from one joint computation: one run over both blocks.
joint_blocks(n; t=0.0, s=0.7) = IndexedReparametrization([
    [n + i => rule(t, s, S(:joint_ab, n, i), S(:joint_ab, n, 2n + 1)) for i in 1:n];
    [2n + i => rule(t, s, S(:joint_ab, n, n + i), S(:joint_ab, n, 2n + 1)) for i in 1:n]])
own_blocks(n; t=0.0, s=0.7, tb=t, sb=s) = IndexedReparametrization([
    [n + i => rule(t, s, Own(:loc_a, n, i), Own(:log_scale, n, 1)) for i in 1:n];
    [2n + i => rule(tb, sb, Own(:loc_b, n, i), Own(:log_scale, n, 1)) for i in 1:n]])

struct StdNormal
    d::Int
end
LogDensityProblems.dimension(p::StdNormal) = p.d
LogDensityProblems.capabilities(::Type{StdNormal}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::StdNormal, y) = -dot(y, y) / 2
LogDensityProblems.logdensity_and_gradient(::StdNormal, y) = (-dot(y, y) / 2, -y)
problem(ir, d) = ReparametrizedProblem(ir, StdNormal(d), AutoEnzyme())
one_pass(ir, x, g) = invoke(WarmupHMC._reparam_gradient, Tuple{Any,Any,Any,Any,Any},
                            AutoEnzyme(), ir, x, nothing, g)
# The extension's gradient, given the map's image of `x` as the sampler gives it.
pairwise(ir, x, g) = WarmupHMC._reparam_gradient(AutoEnzyme(), ir, x, last(ir(x)), g)
agrees(a, b) = isapprox(a, b; rtol=1e-12)
function transport(ir_old, ir_new, d, P, G)
    P2, G2 = copy(P), copy(G)
    WarmupHMC._jointly_transport_halo!(problem(ir_new, d), ir_old, P, G, P2, G2)
    P2, G2
end

function operation_counts(build, d, n; kwargs...)
    rng = Xoshiro(n)
    x, g = randn(rng, d), randn(rng, d)
    P, G = randn(rng, d, 5), randn(rng, d, 5)
    ir = build(n; kwargs...)
    ors = WarmupHMC.OnlineReparametrizer(ir; accumulator=WarmupHMC.WeightedReparametrizationLoss)
    (; forward = counted(() -> ir(x)),
       inverse = counted(() -> WarmupHMC._inverse_with_logabsdet_jacobian(ir, x)),
       gradient = counted(() -> LogDensityProblems.logdensity_and_gradient(problem(ir, d), x)),
       scoring = counted(() -> OnlineStatsBase.fit!(ir, ors, x, g)),
       transport = counted(() -> transport(build(n; kwargs...), build(n; s=0.2), d, P, G)))
end

@testset "a location and a log-scale computed by one function evaluate it once" begin
    for (build, tag) in ((same_function, :energy), (joint, :joint), (framed, :joint))
        counts = [operation_counts(build, 2n, n) for n in (8, 64)]
        @test counts[1] == counts[2]
        c = counts[1]
        @test calls(c.forward, tag) == 1
        @test calls(c.inverse, tag) == 1
        @test calls(c.scoring, tag) == 1
        @test calls(c.gradient, tag) <= 2
        @test calls(c.transport, tag) <= 6 * 5 + 2
    end
    # The joint frame gives the same map as separate location and log-scale functions.
    n = 9
    x = randn(Xoshiro(4), 2n)
    separate = IndexedReparametrization(
        [n + i => rule(0.0, 0.7, S(:loc_a, n, i), S(:log_scale, n, 1)) for i in 1:n])
    @test joint(n)(x) == separate(x)
    @test WarmupHMC._inverse_with_logabsdet_jacobian(joint(n), x) ==
          WarmupHMC._inverse_with_logabsdet_jacobian(separate, x)
    @test agrees(pairwise(joint(n), x, x),
                 one_pass(separate, x, x))
    @test framed(n)(x) == separate(x)
    @test WarmupHMC._inverse_with_logabsdet_jacobian(framed(n), x) ==
          WarmupHMC._inverse_with_logabsdet_jacobian(separate, x)
    @test agrees(pairwise(framed(n), x, x),
                 one_pass(separate, x, x))
    P, G = randn(Xoshiro(8), 2n, 6), randn(Xoshiro(9), 2n, 6)
    Pa, Ga = transport(framed(n; s=0.9), framed(n; s=0.2), 2n, P, G)
    Pb, Gb = transport(IndexedReparametrization([n + i => rule(0.0, 0.9, S(:loc_a, n, i),
                           S(:log_scale, n, 1)) for i in 1:n]),
                       IndexedReparametrization([n + i => rule(0.0, 0.2, S(:loc_a, n, i),
                           S(:log_scale, n, 1)) for i in 1:n]), 2n, P, G)
    @test Pa == Pb
    @test agrees(Ga, Gb)
end

@testset "blocks that read their endpoints from one function evaluate it once" begin
    counts = [operation_counts(joint_blocks, 3n, n) for n in (8, 64)]
    @test counts[1] == counts[2]
    c = counts[1]
    @test WarmupHMC._run_end(joint_blocks(8).pairs, 1) == 16
    for op in (:forward, :inverse, :scoring)
        @test calls(getproperty(c, op), :joint_ab) == 1
    end
    @test calls(c.gradient, :joint_ab) <= 2
    @test calls(c.transport, :joint_ab) <= 6 * 5 + 2
    # With a function per block, a log-scale the blocks share is evaluated once
    # per run, so once per block.
    c = operation_counts(two_blocks, 3 * 8, 8)
    for op in (:forward, :inverse, :scoring)
        @test calls(getproperty(c, op), :log_scale) == 2
        @test calls(getproperty(c, op), :loc_a) == 1
        @test calls(getproperty(c, op), :loc_b) == 1
    end
    # The joint map is the map of the separate functions.
    n = 8
    x = randn(Xoshiro(7), 3n)
    separate = IndexedReparametrization([
        [n + i => rule(0.0, 0.7, Own(:loc_a, n, i), Own(:log_scale, n, 1)) for i in 1:n];
        [2n + i => rule(0.0, 0.7, Own(:joint_ab, n, n + i), Own(:log_scale, n, 1)) for i in 1:n]])
    @test joint_blocks(n)(x) == separate(x)
    @test WarmupHMC._inverse_with_logabsdet_jacobian(joint_blocks(n), x) ==
          WarmupHMC._inverse_with_logabsdet_jacobian(separate, x)
    @test agrees(pairwise(joint_blocks(n), x, x),
                 one_pass(separate, x, x))
end

@testset "blocks that share a log-scale reproduce the pair-by-pair evaluation" begin
    n = 7
    d = 3n
    rng = Xoshiro(5)
    x, g = randn(rng, d), randn(rng, d)
    for (t, s, tb, sb) in ((0.0, 1.0, 0.0, 1.0), (1.0, 0.3, 0.2, 0.9), (0.4, 0.4, 0.0, 0.6),
                           (0.0, 0.6, 0.5, 0.5))
        a, b = two_blocks(n; t, s, tb, sb), own_blocks(n; t, s, tb, sb)
        @test a(x) == b(x)
        @test WarmupHMC._inverse_with_logabsdet_jacobian(a, x) ==
              WarmupHMC._inverse_with_logabsdet_jacobian(b, x)
        _, y = a(x)
        @test last(WarmupHMC._inverse_with_logabsdet_jacobian(a, y)) ≈ x
        ga = pairwise(a, x, g)
        @test agrees(ga, pairwise(b, x, g))
        @test agrees(ga, one_pass(b, x, g))
        @test agrees(one_pass(a, x, g), one_pass(b, x, g))
    end
    P, G = randn(rng, d, 6), randn(rng, d, 6)
    for (old, new) in (((; s=0.9, sb=0.8), (; s=0.1, sb=0.3)),
                       ((; s=0.0, sb=0.0), (; s=0.1, sb=0.0)),   # old identity
                       ((; s=0.9, sb=0.0), (; s=0.0, sb=0.4)))
        Pa, Ga = transport(two_blocks(n; old...), two_blocks(n; new...), d, P, G)
        Pb, Gb = transport(own_blocks(n; old...), own_blocks(n; new...), d, P, G)
        @test Pa == Pb
        @test agrees(Ga, Gb)
    end
    # Selection: the same scores, the same centerings, the same moved pool.
    P, G = randn(rng, d, 40), randn(rng, d, 40)
    @test candidate_scoring_losses(problem(two_blocks(n; s=0.5), d), P, G) ==
          candidate_scoring_losses(problem(own_blocks(n; s=0.5), d), P, G)
    a, b = two_blocks(n; s=0.5), own_blocks(n; s=0.5)
    Pa, Ga, Pb, Gb = copy(P), copy(G), copy(P), copy(G)
    WarmupHMC.optimize!(a, Pa, Ga)
    WarmupHMC.optimize!(b, Pb, Gb)
    @test [last(p).source for p in a.pairs] == [last(p).source for p in b.pairs]
    @test (Pa, Ga) == (Pb, Gb)
end

@testset "identity pairs evaluate no accessor and map exactly to themselves" begin
    n = 8
    d = 3n
    rng = Xoshiro(6)
    x, g = randn(rng, d), randn(rng, d)
    ir = two_blocks(n; t=0.7, s=0.7)
    ors = WarmupHMC.OnlineReparametrizer(ir; accumulator=WarmupHMC.WeightedReparametrizationLoss)
    @test none(counted(() -> ir(x)))
    @test none(counted(() -> WarmupHMC._inverse_with_logabsdet_jacobian(ir, x)))
    @test none(counted(() -> LogDensityProblems.logdensity_and_gradient(problem(ir, d), x)))
    # Selection still scores candidates away from the identity, once per run.
    @test calls(counted(() -> OnlineStatsBase.fit!(ir, ors, x, g)), :log_scale) == 2
    @test ir(x) == (0.0, x)
    @test WarmupHMC._inverse_with_logabsdet_jacobian(ir, x) == (0.0, x)
    # The gradient of the identity map is the seed.
    @test pairwise(ir, x, g) == g
    @test agrees(one_pass(ir, x, g), g)
    # The formula the identity replaces agrees to rounding.
    value = last(ir.pairs[1])
    loc, log_scale = value.args[1](x), value.args[2](x)
    ljac, y = WarmupHMC.reparam(PC(0.7), PC(0.7), x[n + 1], loc, log_scale)
    @test ljac == 0 && y ≈ x[n + 1]
    # One block at the identity: only the other block evaluates its accessors.
    mixed = two_blocks(n; t=0.7, s=0.7, tb=0.0, sb=0.6)
    c = counted(() -> mixed(x))
    @test calls(c, :loc_a) == 0 && calls(c, :loc_b) == 1 && calls(c, :log_scale) == 1
    @test agrees(pairwise(mixed, x, g),
                 one_pass(own_blocks(n; t=0.7, s=0.7, tb=0.0, sb=0.6), x, g))
end

@testset "adaptive warm-up samples through blocks that share a log-scale" begin
    n = 4
    rp = problem(two_blocks(n; t=0.0, s=0.0), 3n)
    result = adaptive_warmup_mcmc(Xoshiro(3), rp; n_draws=100, n_evaluations=200,
                                  init=(; position=zeros(3n), squared_scale=ones(3n)))
    @test all(isfinite, result.posterior_position)
end
end
