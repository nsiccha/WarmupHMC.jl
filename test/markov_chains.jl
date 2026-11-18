# Snag random-walk-bloc-fc175e21 (decision 1idvzrg): latent random walks and
# AR(1) series centred adaptively per time step by `MarkovChain`.

@testmodule MarkovChainFixtures begin
using LogDensityProblems, Random, LinearAlgebra
export WalkModel, walk_model, walk_posterior, Smooth
export AnchoredWalk, anchored_walk, anchored_posterior, condition_number

# The reporter's walk: states w_1..w_T with w_t ~ N(w_{t-1}, q) and w_0 = 0,
# observed as y_t ~ N(w_t, sigma) outside the first `head` and last `tail`
# steps. With `sampled_q`, log q is coordinate T + 1 with prior N(log 0.6, 1).
struct WalkModel
    y::Vector{Float64}
    observed::BitVector
    log_q::Float64
    sigma::Float64
    sampled_q::Bool
end
function walk_model(T; q=0.6, sigma=1.6, seed=1, head=0, tail=0, sampled_q=false)
    rng = Xoshiro(seed)
    w = q .* cumsum(randn(rng, T))
    y = w .+ sigma .* randn(rng, T)
    observed = trues(T)
    observed[1:head] .= false
    observed[T-tail+1:T] .= false
    WalkModel(y, observed, log(q), sigma, sampled_q)
end
LogDensityProblems.dimension(m::WalkModel) = length(m.y) + m.sampled_q
LogDensityProblems.capabilities(::Type{WalkModel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(m::WalkModel, x) =
    first(LogDensityProblems.logdensity_and_gradient(m, x))
function LogDensityProblems.logdensity_and_gradient(m::WalkModel, x)
    T = length(m.y)
    w = x[1:T]
    lq = m.sampled_q ? x[T+1] : m.log_q
    q = exp(lq)
    step = diff([0.0; w]) ./ q
    residual = m.observed .* (m.y .- w) ./ m.sigma
    lp = -sum(abs2, step) / 2 - T * lq - sum(abs2, residual) / 2
    g = residual ./ m.sigma .- step ./ q
    g[1:T-1] .+= step[2:T] ./ q
    m.sampled_q || return lp, g
    lp - (lq - log(0.6))^2 / 2, [g; sum(abs2, step) - T - (lq - log(0.6))]
end

# The exact Gaussian posterior of a fixed-q walk, dense (small T only).
function walk_posterior(m::WalkModel)
    T = length(m.y)
    D = Matrix(Bidiagonal(ones(T), -ones(T - 1), :L))   # (D w)_t = w_t - w_{t-1}
    Q = D' * D ./ exp(2m.log_q) + Diagonal(m.observed ./ m.sigma^2)
    S = Symmetric(inv(Symmetric(Q)))
    S * (m.observed .* m.y ./ m.sigma^2), S, Symmetric(Q)
end

# A smooth non-Gaussian target for derivative checks.
struct Smooth
    d::Int
end
LogDensityProblems.dimension(m::Smooth) = m.d
LogDensityProblems.capabilities(::Type{Smooth}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(m::Smooth, y) = first(LogDensityProblems.logdensity_and_gradient(m, y))
function LogDensityProblems.logdensity_and_gradient(m::Smooth, y)
    neighbours = [0.0; y[1:end-1]] .+ [y[2:end]; 0.0]
    (-sum(abs2, y) / 2 - 0.05sum(y .^ 4) + 0.2sum(y[1:end-1] .* y[2:end]),
     -y .- 0.2 .* y .^ 3 .+ 0.2 .* neighbours)
end

# Snag markovchain-on-a-ed98880f: the walk w_t ~ N(w_{t-1}, q), w_0 = 0, read
# only relative to its state `anchor`, as y_t ~ N(base + w_t - w_anchor, sigma)
# at the steps `observed` (the anchor's own reading carries nothing). With
# `offset`, `base` is coordinate T + 1 with prior N(0, 2); without, it is 0. With
# `sampled_q`, log q is the last coordinate, prior N(log 0.6, 1).
struct AnchoredWalk
    y::Vector{Float64}
    anchor::Int
    observed::Vector{Int}
    log_q::Float64
    sigma::Float64
    offset::Bool
    sampled_q::Bool
end
function anchored_walk(T; anchor, observed, q=0.6, sigma=0.5, seed=1, offset=true,
                       sampled_q=false)
    rng = Xoshiro(seed)
    w = q .* cumsum(randn(rng, T))
    y = (offset ? 1.0 : 0.0) .+ w .- w[anchor] .+ sigma .* randn(rng, T)
    AnchoredWalk(y, anchor, setdiff(observed, anchor), log(q), sigma, offset, sampled_q)
end
LogDensityProblems.dimension(m::AnchoredWalk) = length(m.y) + m.offset + m.sampled_q
LogDensityProblems.capabilities(::Type{AnchoredWalk}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(m::AnchoredWalk, x) =
    first(LogDensityProblems.logdensity_and_gradient(m, x))
function LogDensityProblems.logdensity_and_gradient(m::AnchoredWalk, x)
    T = length(m.y)
    w = x[1:T]
    base = m.offset ? x[T+1] : 0.0
    lq = m.sampled_q ? x[end] : m.log_q
    q = exp(lq)
    step = diff([0.0; w]) ./ q
    residual = (m.y[m.observed] .- base .- w[m.observed] .+ w[m.anchor]) ./ m.sigma
    lp = -sum(abs2, step) / 2 - T * lq - sum(abs2, residual) / 2 - base^2 / 8
    g = -step ./ q
    g[1:T-1] .+= step[2:T] ./ q
    g[m.observed] .+= residual ./ m.sigma
    g[m.anchor] -= sum(residual) / m.sigma
    m.offset && (g = [g; sum(residual) / m.sigma - base / 4])
    m.sampled_q || return lp, g
    lp - (lq - log(0.6))^2 / 2, [g; sum(abs2, step) - T - (lq - log(0.6))]
end

# The exact Gaussian posterior of a fixed-q anchored walk (dense, small T only):
# mean and precision of (w, base).
function anchored_posterior(m::AnchoredWalk)
    T = length(m.y)
    d = T + m.offset
    D = Matrix(Bidiagonal(ones(T), -ones(T - 1), :L))
    Q = zeros(d, d)
    Q[1:T, 1:T] .= D' * D ./ exp(2m.log_q)
    m.offset && (Q[d, d] += 1 / 4)
    b = zeros(d)
    for t in m.observed
        e = zeros(d)
        e[t] += 1
        e[m.anchor] -= 1
        m.offset && (e[d] += 1)
        Q .+= e * e' ./ m.sigma^2
        b .+= e .* m.y[t] ./ m.sigma^2
    end
    Symmetric(Q) \ b, Symmetric(Q)
end

# The condition number of a covariance's correlation matrix: roughly the squared
# number of leapfrog steps a draw needs under a well-fitted diagonal metric.
function condition_number(S)
    s = sqrt.(diag(S))
    ev = eigvals(Symmetric(S ./ (s * s')))
    maximum(ev) / minimum(ev)
end
end

@testitem "MarkovChain: map, inverse, gradient and transport" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const LDP = LogDensityProblems
    const W = WarmupHMC

    # Coordinates: a walk with drift (1:8), an AR(1) series around a mean with a
    # stationary first state (9:14), log q (15), atanh rho (16), mu (17),
    # log s (18), and one pair centred on mu (19).
    rho(x) = tanh(x[16])
    walk(; kw...) = MarkovChain(1:8; log_scale=x -> x[15], drift=0.3, kw...)
    ar1(; kw...) = MarkovChain(9:14; coefficient=rho, drift=x -> (1 - rho(x)) * x[17],
        initial=x -> x[17], log_scale=x -> [x[18] - log1p(-rho(x)^2) / 2; fill(x[18], 5)], kw...)
    pairs(c=0.4) = [19 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(c),
                                             x -> x[17], x -> x[18])]
    build(; c=0.4, kw...) = IndexedReparametrization(pairs(c); chains=(walk(; kw...), ar1(; kw...)))
    moved(seed) = IndexedReparametrization(pairs(); chains=(
        walk(; phi=2 .* rand(Xoshiro(seed), 8) .- 0.5, c=rand(Xoshiro(seed + 1), 8)),
        ar1(; phi=2 .* rand(Xoshiro(seed + 2), 6) .- 0.5, c=rand(Xoshiro(seed + 3), 6))))
    x = 0.5 .* randn(Xoshiro(4), 19)

    @testset "construction" begin
        @test_throws ArgumentError MarkovChain(1:3; log_scale=[1.0])
        @test_throws ArgumentError MarkovChain([1, 2, 2]; log_scale=0.0)
        @test_throws ArgumentError MarkovChain(Int[]; log_scale=0.0)
        @test_throws DimensionMismatch MarkovChain(1:3; log_scale=0.0, phi=[0.0, 1.0])
        @test_throws ArgumentError MarkovChain(1:3; log_scale=0.0, c=NaN)
        @test_throws ArgumentError IndexedReparametrization(pairs(); chains=[MarkovChain(18:19; log_scale=0.0)])
        @test_throws ArgumentError IndexedReparametrization([]; chains=[
            MarkovChain(1:3; log_scale=0.0), MarkovChain(3:4; log_scale=0.0)])
        @test_throws ArgumentError IndexedReparametrization([]; chains=Any[walk(), ar1()])
        @test_throws ArgumentError IndexedReparametrization([]; chains=[1])
        chain = MarkovChain(1:3; log_scale=0.0)
        @test chain.phi == zeros(3) && chain.c == ones(3)
        @test W.has_reparametrization(IndexedReparametrization([]; chains=[chain]))
    end

    @testset "the map and its inverse" begin
        ir = moved(10)
        ljac, y = ir(x)
        back_ljac, back = W._inverse_with_logabsdet_jacobian(ir, y)
        @test back ≈ x atol = 1e-12
        @test ljac + back_ljac ≈ 0 atol = 1e-12
        # The previous state is read in the model frame, with the per-state values.
        a = ir.chains[2]
        r = rho(x)
        @test y[11] ≈ a.phi[3] * (r * y[10] + (1 - r) * x[17]) + x[11] * exp((1 - a.c[3]) * x[18])
        @test y[9] ≈ a.phi[1] * x[17] + x[9] * exp((1 - a.c[1]) * (x[18] - log1p(-r^2) / 2))
        @test ljac ≈ sum((1 .- ir.chains[1].c) .* x[15]) +
                     (1 - a.c[1]) * (x[18] - log1p(-r^2) / 2) + sum((1 .- a.c[2:6]) .* x[18]) +
                     0.6 * x[18]
        # The innovation frame (1, 0) maps standardized innovations to the walk.
        innovations = IndexedReparametrization([]; chains=[walk(; phi=1.0, c=0.0)])
        @test last(innovations(x))[1:8] ≈ cumsum(exp(x[15]) .* x[1:8] .+ [0.0; fill(0.3, 7)])
        # At the identity a chain changes nothing, bit for bit.
        plain = IndexedReparametrization(pairs())
        @test build()(x) == plain(x)
        @test last(build()(x))[1:14] == x[1:14]
    end

    @testset "the gradient" begin
        problem = Smooth(19)
        for ir in (moved(20), moved(30), build())
            rp = ReparametrizedProblem(ir, problem, AutoEnzyme())
            lp, g = LDP.logdensity_and_gradient(rp, x)
            @test lp ≈ LDP.logdensity(rp, x)
            @test g ≈ fd_gradient(z -> LDP.logdensity(rp, z), x) rtol = 1e-6
            y = last(ir(x))
            g_y = last(LDP.logdensity_and_gradient(problem, y))
            generic = invoke(W._reparam_gradient, Tuple{Any,Any,Any,Any,Any},
                             AutoEnzyme(), ir, x, y, g_y)
            @test W._reparam_gradient(AutoEnzyme(), ir, x, y, g_y) ≈ generic atol = 1e-12
        end
        # The identity chain's gradient equals the pairs alone, bit for bit.
        y = last(build()(x))
        g_y = last(LDP.logdensity_and_gradient(problem, y))
        @test W._reparam_gradient(AutoEnzyme(), build(), x, y, g_y) ==
              W._reparam_gradient(AutoEnzyme(), IndexedReparametrization(pairs()), x, y, g_y)
    end

    @testset "the window-end transport" begin
        problem = Smooth(19)
        old, new = moved(40), moved(50)
        x_old = copy(x)
        g_old = last(LDP.logdensity_and_gradient(ReparametrizedProblem(old, problem, AutoEnzyme()), x_old))
        x_new, model = similar(x), similar(x)
        W._transport_position!(x_new, model, new, old, x_old)
        @test last(new(x_new)) ≈ last(old(x_old)) atol = 1e-12
        enzyme = W._transport_gradient_method(AutoEnzyme(), new, old, copy(x_new), copy(g_old))
        generic = invoke(W._transport_gradient_method, Tuple{Any,Any,Any,Any,Any},
                         AutoEnzyme(), new, old, copy(x_new), copy(g_old))
        g1, g2 = similar(x), similar(x)
        enzyme(g1, copy(x_new), copy(g_old))
        generic(g2, copy(x_new), copy(g_old))
        @test g1 ≈ g2 atol = 1e-10
        # The transported gradient is the new frame's own gradient.
        direct = last(LDP.logdensity_and_gradient(ReparametrizedProblem(new, problem, AutoEnzyme()), x_new))
        @test g1 ≈ direct atol = 1e-10
    end

    @testset "controls, snapshots and restore" begin
        ir = moved(60)
        controls = W.reparam_controls(ir)
        @test keys(controls) == (:pairs, :chains)
        @test controls.chains[2].phi == ir.chains[2].phi
        snap = W.snapshot_reparametrization(ir)
        snap.chains[1].phi .= 7.0
        @test ir.chains[1].phi != snap.chains[1].phi
        fresh = build()
        W.restore_reparam_controls!(fresh, controls)
        @test fresh(x) == ir(x)
        W._restore_reparametrization!(snap, ir)
        @test snap.chains[1].phi == ir.chains[1].phi
        # Controls without chains cannot belong to a map with chains, and back.
        @test_throws ArgumentError W.restore_reparam_controls!(build(), W.reparam_controls(IndexedReparametrization(pairs())))
        @test_throws ArgumentError W.restore_reparam_controls!(
            IndexedReparametrization([]; chains=[walk()]), controls)
        # Without chains the controls are the pairs' vector, as before.
        @test W.reparam_controls(IndexedReparametrization(pairs())) isa Vector
    end
end

@testitem "MarkovChain: selection on a Gaussian Markov posterior" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const W = WarmupHMC

    # The posterior of a fixed-q walk is a Gaussian Markov chain, so regressing
    # each state on the previous one recovers its exact conditional mean, and
    # the residuals are independent: the chain's frame whitens the posterior up
    # to scale.
    model = walk_model(30; head=5, tail=8)
    mu, S, Q = walk_posterior(model)
    rng = Xoshiro(2)
    n = 8000
    draws = mu .+ cholesky(Q).U \ randn(rng, 30, n)
    gradients = -(Q * (draws .- mu))
    slopes = [S[t, t-1] / S[t-1, t-1] for t in 2:30]

    ir = IndexedReparametrization([]; chains=[MarkovChain(1:30; log_scale=model.log_q)])
    W.optimize!(ir, copy(draws), copy(gradients))
    chain = ir.chains[1]
    @test maximum(abs, chain.phi[2:end] .- slopes) < 0.05
    # Unobserved steps follow the prior: the innovation.
    @test all(abs.(chain.phi[25:30] .- 1) .< 0.05)

    source = reduce(hcat, [W._inverse_with_logabsdet_jacobian(ir, d)[2] for d in eachcol(draws)])
    offdiagonal(C) = maximum(abs, C - Diagonal(C))
    @test offdiagonal(cor(draws')) > 0.5          # the states are strongly correlated
    @test offdiagonal(cor(source')) < 0.1         # the chain's coordinates are not

    # The retrospective rows score the same candidates at the same coefficient.
    rp = ReparametrizedProblem(
        IndexedReparametrization([]; chains=[MarkovChain(1:30; log_scale=model.log_q)]),
        model, AutoEnzyme())
    rows = candidate_scoring_losses(rp, draws, gradients)
    @test length(rows) == 30 * 11
    @test all(r -> r.chain_number == 1 && r.groups == n, rows)
    @test [rows[11(t-1)+1].phi for t in 1:30] == chain.phi
    for t in 1:30
        state = rows[11(t-1)+1:11t]
        @test state[argmin([r.loss for r in state])].candidate == chain.c[t]
    end
    @test_throws ArgumentError candidate_scoring_losses(rp, draws, gradients; weights=[-1.0; ones(n - 1)])
end

@testitem "MarkovChain: a location that barely varies keeps phi in [0, 1]" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random
    const W = WarmupHMC

    # Snag non-centred-ou-s-83c75167: an Ornstein-Uhlenbeck series whose
    # coefficient exp(-1 / range) underflows where a window's evidence holds a
    # tiny range. Its locations m_t = coefficient * w_{t-1} then vary by about
    # 1e-65, and the least-squares slope of w_t on m_t is noise divided by that
    # variance (phi of 1e47 to 1e67 on the reporter's model). Back at an ordinary
    # range, phi * m_t moved every state by as much, so the chain stayed where the
    # range was tiny for the rest of the run.
    T = 20
    rng = Xoshiro(5)
    n = 2000
    draws = [randn(rng, T, n); -5.0 .+ 0.01 .* randn(rng, 1, n)]
    gradients = -draws
    coefficient(x) = exp(-1 / exp(x[T + 1]))
    ir = IndexedReparametrization([]; chains=[MarkovChain(1:T; coefficient, log_scale=0.0)])
    W.optimize!(ir, copy(draws), copy(gradients))
    chain = ir.chains[1]
    @test all(p -> 0 <= p <= 1, chain.phi)

    # An ordinary range (3, coefficient 0.72) maps to coordinates of the states'
    # own size in the selected frame.
    x = [randn(rng, T); log(3.0)]
    source = W._inverse_with_logabsdet_jacobian(ir, x)[2]
    @test all(isfinite, source)
    @test maximum(abs, source[1:T]) < 10
end

@testitem "MarkovChain: adaptive warm-up on partially observed walks" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, Statistics
    using LogDensityProblems
    using MCMCDiagnosticTools
    using DifferentiationInterface, Enzyme

    # The reporter's toy, shorter: neither fixed frame suits a walk observed only
    # in its middle with a sampled step size. The chain adapts per time step.
    T = 120
    model = walk_model(T; head=15, tail=30, sampled_q=true)
    problem() = ReparametrizedProblem(
        IndexedReparametrization([]; chains=[MarkovChain(1:T; log_scale=x -> x[T+1])]),
        model, AutoEnzyme())
    lpdfs = [problem() for _ in 1:2]
    init = [(; position=zeros(T + 1), squared_scale=ones(T + 1)) for _ in 1:2]
    results = adaptive_warmup_mcmc([Xoshiro(1), Xoshiro(2)], lpdfs; n_draws=300,
                                   init, parallel=false)
    n = minimum(size(r.posterior_position, 2) for r in results)
    draws = cat([permutedims(r.posterior_position[:, end-n+1:end]) for r in results]...; dims=3)
    draws = permutedims(draws, (1, 3, 2))           # draws × chains × parameters
    @test maximum(rhat(draws)) < 1.1
    @test minimum(ess(draws)) > 50
    evaluations = sum(r.sampling_evaluation_counter for r in results)
    @test evaluations / sum(size(r.posterior_position, 2) for r in results) < 64
    # The unobserved tail moved to the innovation; the observed middle did not.
    for lpdf in lpdfs
        chain = WarmupHMC.reparametrizer(lpdf).chains[1]
        tail = T-25:T
        @test mean(chain.phi[tail]) > 0.8
        @test mean(chain.c[tail]) < 0.3
    end
    # Model-frame draws: the observed walk follows its data.
    @test cor(vec(mean(draws[:, :, 20:85]; dims=(1, 2))), model.y[20:85]) > 0.5
end

@testitem "MarkovChain: checkpoints, back_transform, resume and fixed controls" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, Statistics, Serialization, LinearAlgebra
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const W = WarmupHMC

    T = 40
    model = walk_model(T; head=5, tail=10)
    mu, S, _ = walk_posterior(model)
    problem(; kw...) = ReparametrizedProblem(
        IndexedReparametrization([]; chains=[MarkovChain(1:T; log_scale=model.log_q, kw...)]),
        model, AutoEnzyme())
    init = (; position=zeros(T), squared_scale=ones(T))

    @testset "checkpoints and back_transform" begin
        dir = mktempdir()
        lpdf = problem()
        result = adaptive_warmup_mcmc(Xoshiro(3), lpdf; n_draws=200, init, checkpoint_dir=dir)
        payload = deserialize(joinpath(dir, "cp_latest.jls"))
        controls = payload.reparam_sources
        @test controls isa NamedTuple
        @test controls.chains[1].phi == W.reparametrizer(lpdf).chains[1].phi
        @test any(!iszero, controls.chains[1].phi)
        # The payload's raw draws mapped with its controls are the returned draws.
        raw = payload.posterior_position
        @test size(raw, 2) == size(result.posterior_position, 2)
        @test W.back_transform(payload, problem(), raw) ≈ result.posterior_position
        # A map without the chain cannot have written this checkpoint.
        @test_throws ArgumentError W.back_transform(payload, model, raw)
        @test_throws ArgumentError W.back_transform(payload,
            ReparametrizedProblem(IndexedReparametrization([]; chains=[
                MarkovChain(1:T; log_scale=0.0), MarkovChain(T+1:T+1; log_scale=0.0)]),
                model, AutoEnzyme()), raw)
        # Resume restores the controls onto a fresh problem and continues.
        fresh = problem()
        resumed = adaptive_warmup_mcmc(Xoshiro(99), fresh; n_draws=300, checkpoint_dir=dir, resume=true)
        @test size(resumed.posterior_position, 2) >= 300
        @test abs(mean(resumed.posterior_position[20, :]) - mu[20]) < 4sqrt(S[20, 20] / 50)
    end

    @testset "nonlinear_adapt = false keeps the controls" begin
        lpdf = problem(phi=1.0, c=0.0)
        result = adaptive_warmup_mcmc(Xoshiro(4), lpdf; n_draws=300, init, nonlinear_adapt=false)
        chain = W.reparametrizer(lpdf).chains[1]
        @test chain.phi == ones(T) && chain.c == zeros(T)
        # Returned draws are in the model frame (the states, not innovations).
        @test maximum(abs, vec(mean(result.posterior_position; dims=2)) .- mu) < 4maximum(sqrt.(diag(S)) ./ sqrt(30))
    end
end

@testitem "MarkovChain: cooperative pooled selection" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, Statistics, Serialization
    using LogDensityProblems
    using DifferentiationInterface, Enzyme

    T = 60
    model = walk_model(T; head=5, tail=20, sampled_q=true)
    lpdf = ReparametrizedProblem(
        IndexedReparametrization([]; chains=[MarkovChain(1:T; log_scale=x -> x[T+1])]),
        model, AutoEnzyme())
    dir = mktempdir()
    out = cooperative_warmup_mcmc(Xoshiro(5), lpdf; n_chains=2, n_cores=1, n_draws=400,
        init=(; position=zeros(T + 1), squared_scale=ones(T + 1)), checkpoint_dir=dir)
    controls = deserialize(joinpath(dir, "chain_1", "cp_latest.jls")).reparam_sources
    @test controls isa NamedTuple
    @test deserialize(joinpath(dir, "chain_2", "cp_latest.jls")).reparam_sources == controls
    @test any(r -> any(b -> b.centering_refit, r.batches), out.results)
    @test mean(controls.chains[1].phi[T-15:T]) > 0.8
    # The caller's problem is copied per chain and keeps its identity controls.
    @test all(iszero, WarmupHMC.reparametrizer(lpdf).chains[1].phi)
end

@testitem "MarkovChain: streaming evidence, pairs alongside, and re-scoring" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, Statistics, Serialization
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const LDP = LogDensityProblems
    const W = WarmupHMC

    # A walk plus a centred pair block whose scale is the walk's step size.
    struct WalkAndBlock
        walk::WalkModel
    end
    T = 50
    LDP.dimension(m::WalkAndBlock) = T + 1 + 4
    LDP.capabilities(::Type{WalkAndBlock}) = LDP.LogDensityOrder{1}()
    LDP.logdensity(m::WalkAndBlock, x) = first(LDP.logdensity_and_gradient(m, x))
    function LDP.logdensity_and_gradient(m::WalkAndBlock, x)
        lp, g = LDP.logdensity_and_gradient(m.walk, x[1:T+1])
        lq = x[T+1]
        z = x[T+2:end]
        (lp - sum(abs2, z) * exp(-2lq) / 2 - 4lq,
         [g[1:T]; g[T+1] + sum(abs2, z) * exp(-2lq) - 4; -z .* exp(-2lq)])
    end
    model = WalkAndBlock(walk_model(T; head=5, tail=15, sampled_q=true))
    problem() = ReparametrizedProblem(IndexedReparametrization(
            [T + 1 + i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
                                            0.0, x -> x[T+1]) for i in 1:4];
            chains=[MarkovChain(1:T; log_scale=x -> x[T+1])]),
        model, AutoEnzyme())
    dir = mktempdir()
    lpdf = problem()
    result = adaptive_warmup_mcmc(Xoshiro(6), lpdf; n_draws=200, checkpoint_dir=dir,
        init=(; position=zeros(T + 5), squared_scale=ones(T + 5)),
        nonlinear_evidence=:all_good_leaves)
    payload = deserialize(joinpath(dir, "cp_latest.jls"))
    @test payload.nonlinear_recorder.online isa W.ChainEvidence
    ir = W.reparametrizer(lpdf)
    @test any(!iszero, ir.chains[1].phi)
    @test any(p -> last(p).source.c != 1.0, ir.pairs)
    # The payload's rows are the pairs' alone; the matrix form adds the chain's.
    @test all(r -> haskey(r, :pair_number), candidate_scoring_losses(payload))
    rows = candidate_scoring_losses(problem(), payload.halo_position, payload.halo_gradient)
    @test count(r -> haskey(r, :chain_number), rows) == 11T
    # Re-scoring covers the chain's leaf sample.
    rescored = W.rescore_warmup_checkpoint(problem(), payload)
    @test last(rescored.rescored).max_scaled <= 1e-10
    @test W.back_transform(payload, problem(), payload.posterior_position) ≈ result.posterior_position
end

@testitem "MarkovChain: a reference state — map, inverse, gradient, transport and controls" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const LDP = LogDensityProblems
    const W = WarmupHMC

    # Snag markovchain-on-a-ed98880f. Coordinates: a walk with drift and a reference
    # in its middle (1:8, reference 4), read at a level that varies per state
    # (base + a * t, base 20 and a 21), an AR(1) series without a reference
    # (9:14), log q (15), atanh rho (16), mu (17), log s (18), one pair (19).
    rho(x) = tanh(x[16])
    level(x) = [x[20] + x[21] * t for t in 1:8]
    walk(; reference=4, kw...) = MarkovChain(1:8; log_scale=x -> x[15], drift=0.3,
        coefficient=x -> fill(0.9, 8), initial=0.2, reference, level, kw...)
    ar1(; kw...) = MarkovChain(9:14; coefficient=rho, drift=x -> (1 - rho(x)) * x[17],
        initial=x -> x[17], log_scale=x -> [x[18] - log1p(-rho(x)^2) / 2; fill(x[18], 5)], kw...)
    pairs(c=0.4) = [19 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(c),
                                             x -> x[17], x -> x[18])]
    build(; kw...) = IndexedReparametrization(pairs(); chains=(walk(; kw...), ar1()))
    moved_psi(seed) = (p = rand(Xoshiro(seed), 8); p[4] = 0; p)
    moved(seed) = IndexedReparametrization(pairs(); chains=(
        walk(; phi=rand(Xoshiro(seed), 8), c=rand(Xoshiro(seed + 1), 8), psi=moved_psi(seed + 2)),
        ar1(; phi=2 .* rand(Xoshiro(seed + 3), 6) .- 0.5, c=rand(Xoshiro(seed + 4), 6))))
    x = 0.5 .* randn(Xoshiro(4), 21)

    @testset "construction" begin
        @test_throws ArgumentError MarkovChain(1:3; log_scale=0.0, level=1.0)
        @test_throws ArgumentError MarkovChain(1:3; log_scale=0.0, psi=0.5)
        for r in (0, 4, true, 2.0)
            @test_throws ArgumentError MarkovChain(1:3; log_scale=0.0, reference=r)
        end
        @test_throws ArgumentError MarkovChain(1:3; log_scale=0.0, reference=2, level=[1.0])
        @test_throws ArgumentError MarkovChain(1:3; log_scale=0.0, reference=2, psi=[0.1, 0.2, 0.3])
        @test_throws DimensionMismatch MarkovChain(1:3; log_scale=0.0, reference=2, psi=[0.0, 0.0])
        chain = MarkovChain(1:3; log_scale=0.0, reference=2)
        @test chain.reference == 2 && chain.level == 0.0 && chain.psi == zeros(3)
        @test MarkovChain(1:3; log_scale=0.0).reference == 0
    end

    @testset "the map and its inverse" begin
        ir = moved(10)
        ljac, y = ir(x)
        back_ljac, back = W._inverse_with_logabsdet_jacobian(ir, y)
        @test back ≈ x atol = 1e-12
        @test ljac + back_ljac ≈ 0 atol = 1e-12
        a = ir.chains[1]
        e(t, l) = exp((1 - a.c[t]) * l)
        R(t) = y[4] - level(x)[t]
        # The reference is located at `initial`, the states after it at the one
        # before, the states before it at the one after, through that one's
        # transition, with that one's scale.
        @test y[4] ≈ a.phi[4] * 0.2 + x[4] * e(4, x[15])
        @test y[6] ≈ a.phi[6] * (0.9 * y[5] + 0.3) + a.psi[6] * R(6) + x[6] * e(6, x[15])
        @test y[2] ≈ a.phi[2] * (0.9 * y[3] + 0.3) + a.psi[2] * R(2) + x[2] * e(2, x[15])
        @test y[1] ≈ a.phi[1] * (0.9 * y[2] + 0.3) + a.psi[1] * R(1) + x[1] * e(1, x[15])
        # A per-state log-scale makes the backward states take their next state's.
        perstate = IndexedReparametrization([]; chains=[MarkovChain(1:8; reference=4,
            log_scale=x -> x[13:20], phi=0.5, c=0.3, psi=[fill(0.2, 3); 0.0; fill(0.2, 4)])])
        ljac2, y2 = perstate(x)
        # The backward states 1:3 take the scales of 2:4, the reference and 5:8 their own.
        @test ljac2 ≈ 0.7 * (sum(x[14:16]) + sum(x[16:20]))
        @test y2[3] ≈ 0.5 * y2[4] + 0.2 * y2[4] + x[3] * exp(0.7 * x[16])
        @test last(W._inverse_with_logabsdet_jacobian(perstate, y2)) ≈ x atol = 1e-12
        # With identity controls a chain with a reference changes nothing, bit for bit.
        @test build()(x) == IndexedReparametrization(pairs())(x)
    end

    @testset "the gradient" begin
        problem = Smooth(21)
        for ir in (moved(20), moved(30), build())
            rp = ReparametrizedProblem(ir, problem, AutoEnzyme())
            lp, g = LDP.logdensity_and_gradient(rp, x)
            @test lp ≈ LDP.logdensity(rp, x)
            @test g ≈ fd_gradient(z -> LDP.logdensity(rp, z), x) rtol = 1e-6
            y = last(ir(x))
            g_y = last(LDP.logdensity_and_gradient(problem, y))
            generic = invoke(W._reparam_gradient, Tuple{Any,Any,Any,Any,Any},
                             AutoEnzyme(), ir, x, y, g_y)
            @test W._reparam_gradient(AutoEnzyme(), ir, x, y, g_y) ≈ generic atol = 1e-12
        end
    end

    @testset "the window-end transport" begin
        problem = Smooth(21)
        old, new = moved(40), moved(50)
        g_old = last(LDP.logdensity_and_gradient(ReparametrizedProblem(old, problem, AutoEnzyme()), x))
        x_new, model = similar(x), similar(x)
        W._transport_position!(x_new, model, new, old, x)
        @test last(new(x_new)) ≈ last(old(x)) atol = 1e-12
        enzyme = W._transport_gradient_method(AutoEnzyme(), new, old, copy(x_new), copy(g_old))
        generic = invoke(W._transport_gradient_method, Tuple{Any,Any,Any,Any,Any},
                         AutoEnzyme(), new, old, copy(x_new), copy(g_old))
        g1, g2 = similar(x), similar(x)
        enzyme(g1, copy(x_new), copy(g_old))
        generic(g2, copy(x_new), copy(g_old))
        @test g1 ≈ g2 atol = 1e-10
        direct = last(LDP.logdensity_and_gradient(ReparametrizedProblem(new, problem, AutoEnzyme()), x_new))
        @test g1 ≈ direct atol = 1e-10
        # The selection's own move of a column agrees with the transport.
        positions, gradients = reshape(copy(x), :, 1), reshape(copy(g_old), :, 1)
        chain0 = old.chains[1]
        W._move_chain_columns!(new.chains[1], chain0.phi, chain0.psi, chain0.c, positions, gradients)
        @test positions[1:8] ≈ x_new[1:8] atol = 1e-12
    end

    @testset "controls, snapshots and restore" begin
        ir = moved(60)
        controls = W.reparam_controls(ir)
        @test keys(controls.chains[1]) == (:reference, :phi, :c, :psi)
        @test keys(controls.chains[2]) == (:phi, :c)
        @test controls.chains[1].psi == ir.chains[1].psi && controls.chains[1].reference == 4
        snap = W.snapshot_reparametrization(ir)
        snap.chains[1].psi .= 0.5
        @test ir.chains[1].psi != snap.chains[1].psi
        fresh = build()
        W.restore_reparam_controls!(fresh, controls)
        @test fresh(x) == ir(x)
        W._restore_reparametrization!(snap, ir)
        @test snap.chains[1].psi == ir.chains[1].psi
        # Controls recorded with another reference, or without one, do not belong here.
        other = IndexedReparametrization(pairs(); chains=(walk(; reference=5), ar1()))
        @test_throws ArgumentError W.restore_reparam_controls!(other, controls)
        plain = IndexedReparametrization(pairs(); chains=(
            MarkovChain(1:8; log_scale=x -> x[15], drift=0.3, coefficient=x -> fill(0.9, 8),
                        initial=0.2), ar1()))
        @test_throws ArgumentError W.restore_reparam_controls!(plain, controls)
        @test_throws ArgumentError W.restore_reparam_controls!(fresh, W.reparam_controls(plain))
    end
end

@testitem "MarkovChain: a reference on Gaussian posteriors read relative to a state" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const W = WarmupHMC

    # The frame each chain selects on exact posterior draws, as a covariance of the
    # sampler's coordinates.
    function selected(chain, model, n=8000)
        mu, Q = anchored_posterior(model)
        draws = mu .+ cholesky(Q).U \ randn(Xoshiro(3), length(mu), n)
        gradients = -(Q * (draws .- mu))
        ir = IndexedReparametrization([]; chains=[chain])
        W.optimize!(ir, copy(draws), copy(gradients))
        source = reduce(hcat, [W._inverse_with_logabsdet_jacobian(ir, d)[2] for d in eachcol(draws)])
        ir.chains[1], cov(source')
    end
    T = 30
    walk(; kw...) = MarkovChain(1:T; log_scale=log(0.6), kw...)

    # Readings of w_t - w_10 on both sides of the anchor, without an offset: the
    # chain's frame relative to the anchor whitens the posterior, which the walk
    # from the first state cannot.
    model = anchored_walk(T; anchor=10, observed=3:24, offset=false)
    chain, S = selected(walk(; reference=10), model)
    @test condition_number(S) < 1.5
    @test all(p -> 0 <= p <= 1, chain.phi + chain.psi) && all(>=(0), chain.psi)
    @test mean(chain.psi[11:24]) > 0.2
    @test condition_number(last(selected(walk(), model))) > 20   # 69 at T = 30

    # With a sampled offset `base` (coordinate T + 1), the readings' reference is
    # w_10 - base: the chain needs it as its `level`.
    model = anchored_walk(T; anchor=10, observed=11:24)
    _, S_level = selected(walk(; reference=10, level=x -> x[T+1]), model)
    _, S_no_level = selected(walk(; reference=10), model)
    _, S_plain = selected(walk(), model)
    @test condition_number(S_level) < 10
    @test condition_number(S_no_level) > 10condition_number(S_level)
    @test condition_number(S_plain) > 10condition_number(S_level)

    # Absolute readings: a declared reference keeps the exact frame, psi fits to
    # zero away from the reference.
    absolute = walk_model(T; head=5, tail=8)
    mu, S0, Q = walk_posterior(absolute)
    draws = mu .+ cholesky(Q).U \ randn(Xoshiro(2), T, 8000)
    ir = IndexedReparametrization([]; chains=[walk(; reference=15)])
    W.optimize!(ir, copy(draws), -(Q * (draws .- mu)))
    source = reduce(hcat, [W._inverse_with_logabsdet_jacobian(ir, d)[2] for d in eachcol(draws)])
    @test condition_number(cov(source')) < 1.5
    @test maximum(abs, ir.chains[1].psi[[1:13; 17:T]]) < 0.05

    # The retrospective rows score at the fitted (phi, psi).
    rp = ReparametrizedProblem(IndexedReparametrization([]; chains=[walk(; reference=15)]),
                               absolute, AutoEnzyme())
    rows = candidate_scoring_losses(rp, draws, -(Q * (draws .- mu)))
    @test [rows[11(t-1)+1].psi for t in 1:T] == ir.chains[1].psi
    @test [rows[11(t-1)+1].phi for t in 1:T] == ir.chains[1].phi
end

@testitem "MarkovChain: adaptive warm-up of a walk read relative to a state" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, Statistics, Serialization
    using LogDensityProblems
    using MCMCDiagnosticTools
    using DifferentiationInterface, Enzyme
    const W = WarmupHMC

    # The reporter's shape, shorter: readings base + (w_t - w_12) from the anchor
    # to step 80, nothing before it or after; base and log q sampled.
    T = 100
    model = anchored_walk(T; anchor=12, observed=13:80, q=0.3, sampled_q=true)
    d = T + 2
    problem(; kw...) = ReparametrizedProblem(IndexedReparametrization([]; chains=[
            MarkovChain(1:T; log_scale=x -> x[d], reference=12, level=x -> x[T+1], kw...)]),
        model, AutoEnzyme())
    init = (; position=zeros(d), squared_scale=ones(d))

    @testset "mixing" begin
        lpdfs = [problem() for _ in 1:2]
        results = adaptive_warmup_mcmc([Xoshiro(1), Xoshiro(2)], lpdfs; n_draws=300,
                                       init=[init, init], parallel=false)
        n = minimum(size(r.posterior_position, 2) for r in results)
        draws = cat([permutedims(r.posterior_position[:, end-n+1:end]) for r in results]...; dims=3)
        draws = permutedims(draws, (1, 3, 2))
        @test maximum(rhat(draws)) < 1.1
        @test minimum(ess(draws)) > 50
        evaluations = sum(r.sampling_evaluation_counter for r in results)
        @test evaluations / sum(size(r.posterior_position, 2) for r in results) < 64
        for lpdf in lpdfs
            chain = W.reparametrizer(lpdf).chains[1]
            @test mean(chain.psi[14:80]) > 0.2
        end
    end

    @testset "checkpoints, back_transform, resume and fixed controls" begin
        dir = mktempdir()
        lpdf = problem()
        result = adaptive_warmup_mcmc(Xoshiro(3), lpdf; n_draws=200, init, checkpoint_dir=dir)
        payload = deserialize(joinpath(dir, "cp_latest.jls"))
        controls = payload.reparam_sources.chains[1]
        @test controls.reference == 12 && controls.psi == W.reparametrizer(lpdf).chains[1].psi
        @test any(!iszero, controls.psi)
        @test W.back_transform(payload, problem(), payload.posterior_position) ≈ result.posterior_position
        @test_throws ArgumentError W.back_transform(payload, ReparametrizedProblem(
            IndexedReparametrization([]; chains=[MarkovChain(1:T; log_scale=x -> x[d], reference=13,
                                                             level=x -> x[T+1])]),
            model, AutoEnzyme()), payload.posterior_position)
        resumed = adaptive_warmup_mcmc(Xoshiro(99), problem(); n_draws=300, checkpoint_dir=dir, resume=true)
        @test size(resumed.posterior_position, 2) >= 300
        rescored = W.rescore_warmup_checkpoint(problem(), payload)
        @test last(rescored.rescored).max_scaled <= 1e-10

        fixed = problem(phi=0.5, psi=[fill(0.3, 11); 0.0; fill(0.3, T - 12)], c=0.5)
        adaptive_warmup_mcmc(Xoshiro(4), fixed; n_draws=100, init, nonlinear_adapt=false)
        chain = W.reparametrizer(fixed).chains[1]
        @test chain.phi == fill(0.5, T) && chain.c == fill(0.5, T) && chain.psi[1] == 0.3
    end
end

@testitem "MarkovChain: accessors that read one function evaluate it once" setup=[WarmupHMCSharedFixtures, MarkovChainFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra
    using LogDensityProblems
    using DifferentiationInterface, Enzyme
    const LDP = LogDensityProblems
    const W = WarmupHMC

    # Snag markovchain-acce-da401817: a chain whose accessors all read one frame
    # function, as RKPPL's compiled chain kernel is read, ran it once per
    # accessor on every path: four times per map, eight per gradient. Accessors
    # that are `SharedAccessor`s over one `f` now share one evaluation per
    # chain, as a pair's location and log-scale do, and give the same results as
    # reading `f` once per accessor: exactly for the maps and the selection, to
    # rounding for the derivatives.
    const CALLS = Ref(0)
    tick!() = (CALLS[] += 1; nothing)
    # Counted in the primal; Enzyme does not differentiate the counter.
    Enzyme.EnzymeRules.inactive(::typeof(tick!), args...) = nothing
    counted(f) = (CALLS[] = 0; f(); CALLS[])

    # Coordinates: one chain (1:8), log s (9), atanh rho (10), mu (11), a reading
    # offset (12), and one pair centred on mu (13). The kernel computes every
    # state's values in one pass, as families read by number or, for a single
    # value, by `CartesianIndex(family, 1)`, RKPPL's selectors: log-scales (1),
    # coefficients (2), drifts (3), the initial location (4), the level (5).
    struct Families{T}
        values::Vector{Vector{T}}
    end
    Base.getindex(f::Families, family::Int) = f.values[family]
    Base.getindex(f::Families, i::CartesianIndex{2}) = f.values[i[1]][i[2]]
    struct Kernel <: Function end
    function (::Kernel)(x)
        tick!()
        rho = tanh(x[10])
        Families([[x[9] - log1p(-rho^2) / 2; fill(x[9], 7)], fill(rho, 8),
                  fill((1 - rho) * x[11], 8), [x[11]], [x[12] + 0.1 * t for t in 1:8]])
    end
    # The same values read by an accessor of its own, each running the kernel.
    struct Own{I} <: Function
        i::I
    end
    (a::Own)(x) = Kernel()(x)[a.i]
    shared(i) = SharedAccessor(Kernel(), i)

    pairs() = [13 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.4),
                                       x -> x[11], x -> x[9])]
    function build(read; reference=false, seed=10)
        accessors = (; log_scale=read(1), coefficient=read(2), drift=read(3),
                       initial=read(CartesianIndex(4, 1)))
        controls = (; phi=rand(Xoshiro(seed), 8), c=rand(Xoshiro(seed + 1), 8))
        chain = reference ?
            MarkovChain(1:8; accessors..., controls..., reference=4, level=read(5),
                        psi=(p = rand(Xoshiro(seed + 2), 8); p[4] = 0; p)) :
            MarkovChain(1:8; accessors..., controls...)
        IndexedReparametrization(pairs(); chains=[chain])
    end
    x = 0.5 .* randn(Xoshiro(4), 13)
    problem = Smooth(13)
    rp(ir) = ReparametrizedProblem(ir, problem, AutoEnzyme())

    for reference in (false, true)
        n = reference ? 5 : 4    # accessors that read the kernel
        ir, own = build(shared; reference), build(Own; reference)
        @testset "reference = $reference: the map and its inverse" begin
            @test counted(() -> ir(x)) == 1
            @test counted(() -> own(x)) == n
            @test ir(x) == own(x)
            y = last(ir(x))
            inverse(ir) = W._inverse_with_logabsdet_jacobian(ir, y)
            @test counted(() -> inverse(ir)) == 1
            @test inverse(ir) == inverse(own)
            @test last(inverse(ir)) ≈ x atol = 1e-12
            @test counted(() -> LDP.logdensity(rp(ir), x)) == 1
        end
        @testset "reference = $reference: the gradient" begin
            # One evaluation for the map's value, one in the Enzyme objective.
            @test counted(() -> LDP.logdensity_and_gradient(rp(ir), x)) <= 2
            @test counted(() -> LDP.logdensity_and_gradient(rp(own), x)) >= 2n
            lp, g = LDP.logdensity_and_gradient(rp(ir), x)
            lp_own, g_own = LDP.logdensity_and_gradient(rp(own), x)
            @test lp == lp_own
            @test g ≈ g_own rtol = 1e-12
            @test g ≈ fd_gradient(z -> LDP.logdensity(rp(ir), z), x) rtol = 1e-6
            y = last(ir(x))
            g_y = last(LDP.logdensity_and_gradient(problem, y))
            generic = invoke(W._reparam_gradient, Tuple{Any,Any,Any,Any,Any},
                             AutoEnzyme(), ir, x, y, g_y)
            @test W._reparam_gradient(AutoEnzyme(), ir, x, y, g_y) ≈ generic atol = 1e-12
        end
        @testset "reference = $reference: selection and transport" begin
            P = x .+ 0.3 .* randn(Xoshiro(5), 13, 6)
            G = randn(Xoshiro(6), 13, 6)
            losses(ir) = candidate_scoring_losses(rp(ir), P, G)
            # Two passes over the columns, the regression and the scoring.
            @test counted(() -> losses(ir)) == 2 * 6
            @test counted(() -> losses(own)) == 2 * 6 * n
            @test isequal(losses(ir), losses(own))
            function transport(old, new)
                P2, G2 = copy(P), copy(G)
                W._jointly_transport_halo!(rp(new), old, P, G, P2, G2)
                P2, G2
            end
            shared_moved, own_moved = transport(ir, build(shared; reference, seed=20)),
                                      transport(own, build(Own; reference, seed=20))
            @test shared_moved[1] == own_moved[1]
            @test shared_moved[2] ≈ own_moved[2] rtol = 1e-12
            @test n * counted(() -> transport(ir, build(shared; reference, seed=20))) <=
                  counted(() -> transport(own, build(Own; reference, seed=20)))
            # The selection's own move of a column agrees with the transport.
            new = build(shared; reference, seed=20)
            positions, gradients = copy(P), copy(G)
            old_chain = ir.chains[1]
            @test counted(() -> W._move_chain_columns!(new.chains[1], old_chain.phi, old_chain.psi,
                                                       old_chain.c, positions, gradients)) == 6
            @test positions[1:8, :] ≈ shared_moved[1][1:8, :] atol = 1e-12
        end
    end
end
