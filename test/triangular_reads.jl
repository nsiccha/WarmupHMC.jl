@testitem "Accessors read earlier pairs' coordinates at their model values" begin
    using Test, WarmupHMC
    using Random, LinearAlgebra
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const OnlineStatsBase = WarmupHMC.OnlineStatsBase  # not a test dependency of its own

# Snag indexedreparamet-1e1f6ef7 (reporter ReactiveKernelsPPL): a two-level
# nested hierarchy with both levels adaptively centred, each about its parent.
#
#   block A  z_g[1:G]  location mu,                          log-scale log_tau_g
#   block B  z_u[1:U]  location mu + tau_g * z_g[group[u]],  log-scale log_tau_u
#
# Both blocks are non-centred in the model (target c = 0). B's location reads
# A's coordinate. Accessors read an earlier pair's coordinate at its MODEL value,
# so B is centred on its parent's model value whatever A's centering is: moving
# A leaves B's sampler coordinates alone, and the per-coordinate scorer's
# evidence for A is the committed parametrization's. Before, B read A's sampler
# coordinate: the map was still a bijection, but transport moved B whenever only
# A moved, and the scorer's gradient for A differed from the committed one.

const G, U = 3, 6
const GROUP = (1, 1, 2, 2, 3, 3)
const MU, LTG, LTU = 1, 2, 3
iA(g) = 3 + g
iB(u) = 3 + G + u
const D = 3 + G + U
const PC = PartiallyCentered

model_location_B(x, zA) = x[MU] + exp(x[LTG]) * zA

# One accessor type for every pair. kind: 1 A location, 2 A log-scale,
# 3 B location (reads A's coordinate), 4 B log-scale.
struct Endpoint <: Function
    kind::Int
    u::Int
end
(e::Endpoint)(x) = e.kind == 1 ? x[MU] : e.kind == 2 ? x[LTG] :
                   e.kind == 3 ? model_location_B(x, x[iA(GROUP[e.u])]) : x[LTU]

rule(t, s, location, log_scale) = Reparametrization(PC(t), PC(s), location, log_scale)
nested(cA::AbstractVector, cB::AbstractVector) = IndexedReparametrization([
    [iA(g) => rule(0.0, cA[g], Endpoint(1, 0), Endpoint(2, 0)) for g in 1:G];
    [iB(u) => rule(0.0, cB[u], Endpoint(3, u), Endpoint(4, 0)) for u in 1:U]])
nested(cA::Real, cB::Real) = nested(fill(cA, G), fill(cB, U))

# The reference: the composition written out, with t = 0. `centre` inverts
# `noncentre`; each is the scalar PartiallyCentered map.
noncentre(x, c, location, log_scale) = (x - c * location) * exp(-log_scale * c)
centre(y, c, location, log_scale) = c * location + y * exp(log_scale * c)
function reference_map(x, cA, cB)
    y = float.(copy(x))
    ljac = zero(eltype(y))
    for g in 1:G
        ljac -= x[LTG] * cA[g]
        y[iA(g)] = noncentre(x[iA(g)], cA[g], x[MU], x[LTG])
    end
    for u in 1:U
        ljac -= x[LTU] * cB[u]
        y[iB(u)] = noncentre(x[iB(u)], cB[u], model_location_B(x, y[iA(GROUP[u])]), x[LTU])
    end
    ljac, y
end
function reference_inverse(y, cA, cB)
    x = copy(y)
    for g in 1:G
        x[iA(g)] = centre(y[iA(g)], cA[g], y[MU], y[LTG])
    end
    for u in 1:U
        x[iB(u)] = centre(y[iB(u)], cB[u], model_location_B(y, y[iA(GROUP[u])]), y[LTU])
    end
    x
end

struct StdNormal end
LogDensityProblems.dimension(::StdNormal) = D
LogDensityProblems.capabilities(::Type{StdNormal}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::StdNormal, y) = -dot(y, y) / 2
LogDensityProblems.logdensity_and_gradient(::StdNormal, y) = (-dot(y, y) / 2, -y)
problem(ir) = ReparametrizedProblem(ir, StdNormal(), AutoEnzyme())
gradient_at(ir, x) = last(LogDensityProblems.logdensity_and_gradient(problem(ir), x))
transported(old, new, x) = last(WarmupHMC._inverse_with_logabsdet_jacobian(new, last(old(x))))
agrees(a, b) = isapprox(a, b; rtol=1e-12)
function fd_gradient(f, x; h=1e-6)
    map(eachindex(x)) do i
        e = zero(x); e[i] = h
        (f(x .+ e) - f(x .- e)) / 2h
    end
end

# A candidate accumulator that keeps what the scorer hands it.
struct Seen
    observations::Vector{Any}
end
Seen() = Seen(Any[])
OnlineStatsBase.fit!(s::Seen, observation; kwargs...) = (push!(s.observations, observation); s)
OnlineStatsBase.nobs(s::Seen) = length(s.observations)
# The observation the scorer recorded for pair `k`'s candidate `c`.
function observation(ors, k, c)
    candidates = last(ors.pairs[k]).pairs
    only(last(candidates[findfirst(p -> first(p).c ≈ c, candidates)]).observations)
end
function scored(ir, x, g; scoring_plan=WarmupHMC.DIRECT_CANDIDATE_SCORING)
    ors = WarmupHMC.OnlineReparametrizer(ir; accumulator=Seen)
    OnlineStatsBase.fit!(ir, ors, x, g; scoring_plan)
    ors
end

rng = Xoshiro(20261114)
centerings = ((0.7, 0.4), (0.0, 1.0), (1.0, 0.0), (0.3, 0.9), (0.5, 0.5))

@testset "map, inverse and log-Jacobian are the composition written out" begin
    for (cA, cB) in centerings, _ in 1:3
        x = randn(rng, D)
        ir = nested(cA, cB)
        ljac, y = ir(x)
        ljac_ref, y_ref = reference_map(x, fill(cA, G), fill(cB, U))
        @test y ≈ y_ref rtol = 1e-14
        @test ljac ≈ ljac_ref rtol = 1e-14
        ljac_inverse, x_back = WarmupHMC._inverse_with_logabsdet_jacobian(ir, y)
        @test x_back ≈ reference_inverse(y, fill(cA, G), fill(cB, U)) rtol = 1e-14
        @test x_back ≈ x rtol = 1e-12
        @test ljac_inverse ≈ -ljac rtol = 1e-12
        # B is centred on its parent's model value (when A is not at the identity,
        # its sampler coordinate differs from that value).
        cB == 0 || cA == 0 ||
            @test !(y[iB(1)] ≈ noncentre(x[iB(1)], cB, model_location_B(x, x[iA(1)]), x[LTU]))
        # Recovering in place gives the same point.
        z = copy(y)
        WarmupHMC._inverse_with_logabsdet_jacobian!(z, ir, z)
        @test z == x_back
    end
end

@testset "gradient: pairwise Enzyme, one pass, finite differences" begin
    for (cA, cB) in centerings
        ir = nested(cA, cB)
        x = randn(rng, D)
        ljac, y = ir(x)
        g_y = randn(rng, D)
        pairwise = WarmupHMC._reparam_gradient(AutoEnzyme(), ir, x, y, g_y)
        one_pass = invoke(WarmupHMC._reparam_gradient, Tuple{Any,Any,Any,Any,Any},
                          AutoEnzyme(), ir, x, nothing, g_y)
        @test agrees(pairwise, one_pass)
        objective(q) = (r = reference_map(q, fill(cA, G), fill(cB, U)); r[1] + dot(g_y, r[2]))
        @test pairwise ≈ fd_gradient(objective, x) rtol = 1e-7
        # And the sampler's gradient of the density.
        lp(q) = LogDensityProblems.logdensity(problem(ir), q)
        @test gradient_at(ir, x) ≈ fd_gradient(lp, x) rtol = 1e-7
    end
end

@testset "moving only A leaves B's coordinates and the model point alone" begin
    x = randn(rng, D)
    old, new = nested(0.7, 0.4), nested(0.2, 0.4)
    xt = transported(old, new, x)
    @test last(new(xt)) ≈ last(old(x)) rtol = 1e-14
    @test xt[iB.(1:U)] ≈ x[iB.(1:U)] rtol = 1e-14
    @test xt[iA.(1:G)] != x[iA.(1:G)]
    # The halo transport, positions and gradients, through both Enzyme paths.
    P, Gr = randn(rng, D, 5), zeros(D, 5)
    for j in 1:5
        Gr[:, j] = gradient_at(old, P[:, j])
    end
    P2, G2 = copy(P), copy(Gr)
    WarmupHMC._jointly_transport_halo!(problem(new), old, P, Gr, P2, G2)
    for j in 1:5
        @test P2[:, j] ≈ transported(old, new, P[:, j]) rtol = 1e-14
        @test P2[iB.(1:U), j] ≈ P[iB.(1:U), j] rtol = 1e-14
        @test G2[:, j] ≈ gradient_at(new, P2[:, j]) rtol = 1e-10
    end
    one_pass! = invoke(WarmupHMC._transport_gradient_method, Tuple{Any,Any,Any,Any,Any},
                       AutoEnzyme(), new, old, P2[:, 1], Gr[:, 1])
    g_one_pass = similar(x)
    one_pass!(g_one_pass, P2[:, 1], Gr[:, 1])
    @test agrees(G2[:, 1], g_one_pass)
end

@testset "a candidate's scored evidence is the committed parametrization's" begin
    x = randn(rng, D)
    cA, cB = 0.7, 0.4
    ir = nested(cA, cB)
    g = gradient_at(ir, x)
    for plan in (WarmupHMC.DIRECT_CANDIDATE_SCORING,
                 # a plan that declines every pair falls back to the same rule
                 CandidateScoringPlan((ir, x, g) -> nothing, (frame, args...) -> nothing))
        ors = scored(ir, x, g; scoring_plan=plan)
        # Pair A[1] at candidate 0.2, against the parametrization that commits it.
        cA1 = fill(cA, G); cA1[1] = 0.2
        committed = nested(cA1, fill(cB, U))
        xc = transported(ir, committed, x)
        gc = gradient_at(committed, xc)
        _, position, gradient = observation(ors, 1, 0.2)
        @test position ≈ xc[iA(1)] rtol = 1e-12
        @test gradient ≈ gc[iA(1)] rtol = 1e-12
        # Pair B[1] at candidate 0.9: B's evidence does not depend on A's centering.
        before = observation(ors, G + 1, 0.9)
        after = observation(scored(committed, xc, gc; scoring_plan=plan), G + 1, 0.9)
        @test all(map((a, b) -> isapprox(a, b; rtol=1e-12), before, after))
        # And B's candidate is its own committed parametrization.
        cB1 = fill(cB, U); cB1[1] = 0.9
        committed_B = nested(fill(cA, G), cB1)
        xb = transported(ir, committed_B, x)
        _, position, gradient = before
        @test position ≈ xb[iB(1)] rtol = 1e-12
        @test gradient ≈ gradient_at(committed_B, xb)[iB(1)] rtol = 1e-12
    end
end

@testset "the pool search moves each column as the transport does" begin
    ir = nested(0.5, 0.5)
    P = randn(rng, D, 40)
    Gr = reduce(hcat, (gradient_at(ir, P[:, j]) for j in 1:40))
    old = IndexedReparametrization(copy(ir.pairs))
    Pm, Gm = copy(P), copy(Gr)
    WarmupHMC.optimize!(ir, Pm, Gm)
    @test [last(p).source for p in ir.pairs] != [last(p).source for p in old.pairs]
    # The search moves A before it selects B, so A's gradient in the pool is the
    # one with A moved and B not yet; B's is the final one. Each pair's scoring
    # read exactly those.
    intermediate = IndexedReparametrization([ir.pairs[1:G]; old.pairs[G+1:end]])
    for j in 1:40
        @test Pm[:, j] ≈ transported(old, ir, P[:, j]) rtol = 1e-12
        @test Gm[iB.(1:U), j] ≈ gradient_at(ir, Pm[:, j])[iB.(1:U)] rtol = 1e-10
        xi = transported(old, intermediate, P[:, j])
        @test Gm[iA.(1:G), j] ≈ gradient_at(intermediate, xi)[iA.(1:G)] rtol = 1e-10
    end
end

# One joint endpoint function for both levels, split into stages: the parent level
# at stage 1, the level centred on it at stage 2. The function runs once per stage.
const CALLS = Ref(0)
tick!() = (CALLS[] += 1; nothing)
Enzyme.EnzymeRules.inactive(::typeof(tick!), args...) = nothing
counted(f) = (CALLS[] = 0; f(); CALLS[])
struct Endpoints <: Function end
function (::Endpoints)(x)
    tick!()
    [[x[MU] for g in 1:G]; [model_location_B(x, x[iA(GROUP[u])]) for u in 1:U];
     x[LTG]; x[LTU]]
end
staged(cA, cB; stages=(1, 2)) = IndexedReparametrization([
    [iA(g) => rule(0.0, cA, SharedAccessor(Endpoints(), g, stages[1]),
                   SharedAccessor(Endpoints(), G + U + 1, stages[1])) for g in 1:G];
    [iB(u) => rule(0.0, cB, SharedAccessor(Endpoints(), G + u, stages[2]),
                   SharedAccessor(Endpoints(), G + U + 2, stages[2])) for u in 1:U]])

@testset "stages split one shared function into ordered runs" begin
    x, g = randn(rng, D), randn(rng, D)
    for (cA, cB) in ((0.7, 0.4), (0.2, 0.9))
        a, b = staged(cA, cB), nested(cA, cB)
        @test a(x) == b(x)
        y = last(a(x))
        @test WarmupHMC._inverse_with_logabsdet_jacobian(a, y) ==
              WarmupHMC._inverse_with_logabsdet_jacobian(b, y)
        @test agrees(gradient_at(a, x), gradient_at(b, x))
        @test counted(() -> a(x)) == 2
        @test counted(() -> WarmupHMC._inverse_with_logabsdet_jacobian(a, y)) == 2
        @test counted(() -> scored(a, x, g)) == 2
        @test counted(() -> gradient_at(a, x)) <= 4
        @test all(map((p, q) -> isapprox(p, q; rtol=1e-12),
                      observation(scored(a, x, g), G + 1, 0.9),
                      observation(scored(b, x, g), G + 1, 0.9)))
    end
    # Negative control: one stage for both levels makes them one run, so the child
    # locations are computed before the parents are mapped and read their sampler
    # coordinates; the map then differs from the model-frame one.
    one_stage = staged(0.7, 0.4; stages=(1, 1))
    @test counted(() -> one_stage(x)) == 1
    @test !(last(one_stage(x)) ≈ last(nested(0.7, 0.4)(x)))
end

# A nested hierarchy with informative data: the data fix every unit's centred
# effect, so warm-up should centre both levels.
struct NestedData
    observed::Vector{Float64}
    sigma::Float64
end
LogDensityProblems.dimension(::NestedData) = D
LogDensityProblems.capabilities(::Type{NestedData}) = LogDensityProblems.LogDensityOrder{1}()
function nested_logdensity(p::NestedData, y)
    lp = -y[MU]^2 / 2 - (y[LTG] + 1)^2 / 2 - (y[LTU] + 1)^2 / 2
    for g in 1:G
        lp -= y[iA(g)]^2 / 2
    end
    for u in 1:U
        lp -= y[iB(u)]^2 / 2
        theta = model_location_B(y, y[iA(GROUP[u])]) + exp(y[LTU]) * y[iB(u)]
        lp -= (p.observed[u] - theta)^2 / (2 * p.sigma^2)
    end
    lp
end
LogDensityProblems.logdensity(p::NestedData, y) = nested_logdensity(p, y)
LogDensityProblems.logdensity_and_gradient(p::NestedData, y) =
    (nested_logdensity(p, y),
     last(Enzyme.gradient(Enzyme.Reverse, nested_logdensity, Enzyme.Const(p), y)))

@testset "adaptive warm-up adapts both levels of a nested hierarchy" begin
    data = NestedData(randn(Xoshiro(5), U) .* 2, 0.05)
    rp = ReparametrizedProblem(nested(0.0, 0.0), data, AutoEnzyme())
    result = adaptive_warmup_mcmc(Xoshiro(7), rp; n_draws=200, nonlinear_adapt=true,
                                  init=(; position=zeros(D), squared_scale=ones(D)))
    @test size(result.posterior_position, 2) >= 200
    @test all(isfinite, result.posterior_position)
    sources = [last(p).c for p in WarmupHMC.reparam_sources(rp)]
    # The data pin every unit's centred effect, so both levels move toward centred.
    @test all(>=(0.5), sources[G+1:end])
    @test any(>(0.0), sources[1:G])
end
end
