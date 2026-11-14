# Snag random-walk-bloc-fc175e21 (decision 1idvzrg): latent random walks and
# AR(1) series centred adaptively per time step by `MarkovChain`.

@testmodule MarkovChainFixtures begin
using LogDensityProblems, Random, LinearAlgebra
export WalkModel, walk_model, walk_posterior, Smooth

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
