@testmodule BlockArbitrationFixtures begin
# Snag warmuphmc-0-3-4-aeb3d521: after the per-pair centering choice, every block
# of pairs reading one log-scale is checked against the block-wide candidates by
# the whole-frame metric-fit loss (mean over all coordinates of
# std(position) * std(gradient)) on the pool transported into the trial frame.
using WarmupHMC, LogDensityProblems, LinearAlgebra, Random
export Level, level_ir, pool_and_gradients, Mu, LogTau

# A normal level, non-centred in the model: x = [mu; log tau; z[1:J]], value
# theta_j = mu + tau z_j, mu ~ N(0, 5), log tau ~ N(0, 1), z ~ N(0, 1), and unit j
# observed once as y_j ~ N(theta_j, s) where `observed[j]`.
struct Level
    y::Vector{Float64}
    observed::Vector{Bool}
    s::Float64
end
LogDensityProblems.dimension(t::Level) = 2 + length(t.y)
LogDensityProblems.capabilities(::Type{Level}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(t::Level, x) = first(LogDensityProblems.logdensity_and_gradient(t, x))
function LogDensityProblems.logdensity_and_gradient(t::Level, x)
    mu, ltau = x[1], x[2]
    tau = exp(ltau)
    J = length(t.y)
    z = @view x[3:end]
    g = zeros(eltype(x), J + 2)
    lp = -mu^2 / 50 - ltau^2 / 2 - dot(z, z) / 2
    g[1] = -mu / 25
    g[2] = -ltau
    g[3:end] .= -z
    for j in 1:J
        t.observed[j] || continue
        r = (t.y[j] - (mu + tau * z[j])) / t.s^2
        lp -= (t.y[j] - (mu + tau * z[j]))^2 / (2 * t.s^2)
        g[1] += r
        g[2] += r * tau * z[j]
        g[2 + j] += r * tau
    end
    lp, g
end
struct Mu <: Function end
(::Mu)(x) = x[1]
struct LogTau <: Function end
(::LogTau)(x) = x[2]
level_ir(J, c=0.0) = IndexedReparametrization(
    [2 + j => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(c), Mu(), LogTau()) for j in 1:J])

# Pool states in the model's (all non-centred) frame with their gradients there.
# Pinned units: theta_j near y_j whatever tau is, so z_j = (theta_j - mu) / tau
# follows tau. Unobserved units: exact draws of the prior, which is then the
# posterior.
function pool_and_gradients(t::Level, n; rng=Xoshiro(1))
    J = length(t.y)
    X = zeros(J + 2, n)
    for k in 1:n
        mu = sum(t.y) / J + 0.05 * randn(rng)
        ltau = randn(rng) * 0.3 + log(1.0)
        theta = [t.observed[j] ? t.y[j] + t.s * randn(rng) : mu + exp(ltau) * randn(rng) for j in 1:J]
        X[:, k] = [mu; ltau; (theta .- mu) ./ exp(ltau)]
    end
    if !any(t.observed)
        X[1, :] .= 5 .* randn(rng, n)
        X[2, :] .= randn(rng, n)
        X[3:end, :] .= randn(rng, J, n)
    end
    G = reduce(hcat, [last(LogDensityProblems.logdensity_and_gradient(t, c)) for c in eachcol(X)])
    X, G
end
end

@testitem "Blocks are the pairs that read one callable log-scale" setup=[BlockArbitrationFixtures] begin
    using Test, WarmupHMC
    struct OtherScale <: Function end
    (::OtherScale)(x) = x[3]
    pairs = [
        # a constant log-scale is in no block
        [3 => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0), Mu(), 0.0)];
        [4 + j => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0), Mu(), LogTau()) for j in 1:3];
    ]
    @test WarmupHMC._centering_blocks(pairs) == [[2, 3, 4]]
    constant = [j => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0), 0.0, 0.0) for j in 1:3]
    @test isempty(WarmupHMC._centering_blocks(constant))
    # one pair alone on its log-scale is no block
    single = [1 => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0), Mu(), OtherScale())]
    @test isempty(WarmupHMC._centering_blocks(single))
end

@testitem "The whole-frame metric-fit loss is 1 on a Gaussian it fits" setup=[BlockArbitrationFixtures] begin
    using Test, WarmupHMC, Random
    rng = Xoshiro(2)
    sd = [0.1, 1.0, 30.0]
    X = sd .* randn(rng, 3, 20_000)
    G = -X ./ sd .^ 2
    # std(x) * std(-x / σ²) is the sample variance over σ²: 1 up to sampling noise
    @test isapprox(WarmupHMC._frame_fit_loss(X, G), 1; rtol=0.01)
    # a coordinate the pool cannot measure never wins
    X[2, :] .= 0
    @test WarmupHMC._frame_fit_loss(X, G) == Inf
end

@testitem "Arbitration centres a block the data pin, whatever the per-pair choice" setup=[BlockArbitrationFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme
    J = 8
    t = Level(collect(range(-1.5, 1.5, J)), fill(true, J), 0.05)
    rp = ReparametrizedProblem(level_ir(J), t, AutoEnzyme())
    X, G = pool_and_gradients(t, 400)
    old = WarmupHMC.snapshot_reparametrization(WarmupHMC.reparametrizer(rp))
    # The per-pair choice left the block non-centred (as the selection can).
    WarmupHMC._arbitrate_blocks!(rp, old, X, G)
    @test all(p -> last(p).source.c >= 0.8, WarmupHMC.reparametrizer(rp).pairs)
    # The pool and the caller's evidence are not touched.
    X2, G2 = pool_and_gradients(t, 400)
    @test X == X2 && G == G2
end

@testitem "Arbitration keeps the per-pair choice where no block-wide candidate is clearly better" setup=[BlockArbitrationFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme
    # Unobserved units: the posterior is the prior, exactly non-centred.
    J = 8
    t = Level(zeros(J), fill(false, J), 1.0)
    rp = ReparametrizedProblem(level_ir(J), t, AutoEnzyme())
    X, G = pool_and_gradients(t, 2000)
    old = WarmupHMC.snapshot_reparametrization(WarmupHMC.reparametrizer(rp))
    WarmupHMC._arbitrate_blocks!(rp, old, X, G)
    @test all(p -> last(p).source == PartiallyCentered(0.0), WarmupHMC.reparametrizer(rp).pairs)
    # A per-pair choice that differs between pairs survives too.
    ir = WarmupHMC.reparametrizer(rp)
    ir.pairs[1] = first(ir.pairs[1]) => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.1),
                                                           Mu(), LogTau())
    chosen = copy(ir.pairs)
    WarmupHMC._arbitrate_blocks!(rp, old, X, G)
    @test ir.pairs == chosen
end

@testitem "Arbitration leaves a CandidateScoringPlan's choice alone" setup=[BlockArbitrationFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme
    J = 8
    t = Level(collect(range(-1.5, 1.5, J)), fill(true, J), 0.05)
    plan = CandidateScoringPlan((x, g) -> nothing, (pair_index, candidate, frame) -> nothing)
    rp = ReparametrizedProblem(level_ir(J), t, AutoEnzyme(); scoring_plan=plan)
    X, G = pool_and_gradients(t, 400)
    old = WarmupHMC.snapshot_reparametrization(WarmupHMC.reparametrizer(rp))
    WarmupHMC._arbitrate_blocks!(rp, old, X, G)
    @test all(p -> last(p).source == PartiallyCentered(0.0), WarmupHMC.reparametrizer(rp).pairs)
end

@testitem "A fit of a pinned level ends centred" setup=[BlockArbitrationFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme
    J = 8
    t = Level(collect(range(-1.5, 1.5, J)), fill(true, J), 0.05)
    rp = ReparametrizedProblem(level_ir(J), t, AutoEnzyme())
    r = adaptive_warmup_mcmc(Xoshiro(11), rp; n_draws=400,
                             init=(; position=zeros(J + 2), squared_scale=ones(J + 2)))
    @test size(r.posterior_position, 2) >= 400
    @test all(p -> last(p).c > 0.5, WarmupHMC.reparam_sources(rp))
end
