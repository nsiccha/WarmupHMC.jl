@testitem "Window selection plans" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Pkg, TOML
    using Distributions
    using DifferentiationInterface, Enzyme

# WindowSelectionPlan: a user rule chooses the controls at every restarting
# window, and a reparametrizer that is not an IndexedReparametrization adapts
# through the controls interface. The block below is dense (non-triangular), so
# it cannot be written as per-coordinate pairs.

const WS_Q = Matrix(qr([1.0 0.3 -0.2; 0.4 1.0 0.5; -0.3 0.2 1.0]).Q)
const WS_S = [3.0, 1.0, 0.2]

"""
    DenseGaussian(Q, s)

`y[1:3] ~ N(0, Q * Diagonal(s.^2) * Q')` and an independent `y[4] ~ N(0, 1)`.
"""
struct DenseGaussian
    precision::Matrix{Float64}
end
DenseGaussian(Q, s) = DenseGaussian(Q * Diagonal(inv.(s .^ 2)) * Q')
LogDensityProblems.capabilities(::Type{DenseGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::DenseGaussian) = 4
LogDensityProblems.logdensity(p::DenseGaussian, y) =
    -dot(y[1:3], p.precision * y[1:3]) / 2 - y[4]^2 / 2
LogDensityProblems.logdensity_and_gradient(p::DenseGaussian, y) =
    (LogDensityProblems.logdensity(p, y), vcat(-p.precision * y[1:3], -y[4]))

"""
    RotScaleBlock(idx, Q, c, direction)

`y[idx] = Q * Diagonal(exp.(c)) * Q' * x[idx]`, log-Jacobian `sum(c)`; the
controls are `c`. `direction = -1` is the inverse map.
"""
struct RotScaleBlock <: WarmupHMC.AbstractReparametrization
    idx::Vector{Int}
    Q::Matrix{Float64}
    c::Vector{Float64}
    direction::Int
end
RotScaleBlock(c=zeros(3)) = RotScaleBlock([1, 2, 3], WS_Q, collect(float.(c)), 1)
function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector, t::RotScaleBlock, x::AbstractVector)
    s = t.direction .* t.c
    y[t.idx] = t.Q * (exp.(s) .* (t.Q' * x[t.idx]))
    sum(s), y
end
WarmupHMC.InverseFunctions.inverse(t::RotScaleBlock) =
    RotScaleBlock(t.idx, t.Q, t.c, -t.direction)
WarmupHMC.reparam_controls(t::RotScaleBlock) = copy(t.c)
function WarmupHMC.restore_reparam_controls!(t::RotScaleBlock, c)
    c isa AbstractVector{<:Real} && length(c) == length(t.c) ||
        throw(ArgumentError("RotScaleBlock controls must be $(length(t.c)) reals"))
    t.c .= c
    t
end

# Target-frame second moments along Q's columns; select c = log(sd).
const WS_SELECTIONS = Ref(0)
const WS_SIZES = Int[]
function ws_select!(ir, positions, gradients)
    size(positions, 2) > 0 || return false
    WS_SELECTIONS[] += 1
    push!(WS_SIZES, size(positions, 2))
    u = reduce(hcat, [ir.Q' * last(ir(x))[ir.idx] for x in eachcol(positions)])
    ir.c .= log.(vec(mean(u .^ 2; dims=2))) ./ 2
    true
end
ws_plan() = WarmupHMC.WindowSelectionPlan(ws_select!)
ws_problem(c=zeros(3); plan=ws_plan()) = WarmupHMC.ReparametrizedProblem(
    RotScaleBlock(c), DenseGaussian(WS_Q, WS_S), AutoEnzyme(); scoring_plan=plan)

@testset "custom reparametrizer: exact density, gradient and inverse" begin
    rp = ws_problem([0.4, -0.3, 1.1])
    ir = WarmupHMC.reparametrizer(rp)
    x = [0.3, -1.2, 0.8, 0.5]
    ljac, y = ir(x)
    @test ljac ≈ sum(ir.c)
    @test y[4] == x[4]
    @test LogDensityProblems.logdensity(rp, x) ≈
        ljac + LogDensityProblems.logdensity(DenseGaussian(WS_Q, WS_S), y)
    ljac_inverse, back = WarmupHMC._inverse_with_logabsdet_jacobian(ir, y)
    @test back ≈ x
    @test ljac_inverse ≈ -ljac
    _, g = LogDensityProblems.logdensity_and_gradient(rp, x)
    @test g ≈ fd_gradient(z -> LogDensityProblems.logdensity(rp, z), x) rtol = 1e-6
end

@testset "controls interface: copies, restore, snapshot, back_transform" begin
    rp = ws_problem([0.1, 0.2, 0.3])
    controls = WarmupHMC.reparam_sources(rp)
    @test controls == [0.1, 0.2, 0.3]
    controls[1] = 99.0                      # a copy, never an alias
    @test WarmupHMC.reparametrizer(rp).c[1] == 0.1
    snapshot = WarmupHMC.snapshot_reparametrization(WarmupHMC.reparametrizer(rp))
    WarmupHMC.restore_reparam_sources!(rp, [0.5, 0.0, -0.5])
    @test WarmupHMC.reparametrizer(rp).c == [0.5, 0.0, -0.5]
    @test snapshot.c == [0.1, 0.2, 0.3]
    @test_throws ArgumentError WarmupHMC.restore_reparam_sources!(rp, [1.0])
    payload = (; dimension=4, reparam_sources=[1.0, -1.0, 0.0])
    positions = randn(Xoshiro(3), 4, 5)
    mapped = WarmupHMC.back_transform(payload, rp, positions)
    reference = RotScaleBlock([1.0, -1.0, 0.0])
    @test mapped ≈ reduce(hcat, [last(reference(col)) for col in eachcol(positions)])
    @test WarmupHMC.reparametrizer(rp).c == [0.5, 0.0, -0.5]   # lpdf untouched
end

@testset "custom reparametrizer needs a selection plan to adapt" begin
    rp = WarmupHMC.ReparametrizedProblem(RotScaleBlock(), DenseGaussian(WS_Q, WS_S), AutoEnzyme())
    @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), rp; n_draws=200,
        progress=nothing, nonlinear_adapt=true)
    fixed = adaptive_warmup_mcmc(Xoshiro(1), rp; n_draws=400, progress=nothing,
        nonlinear_adapt=false)
    @test size(fixed.posterior_position, 2) >= 400
    @test all(iszero, WarmupHMC.reparametrizer(rp).c)
end

@testset "weighted leaf sample draws proportionally to weight" begin
    sample = WarmupHMC.WeightedLeafSample(1, 20_000)
    for (i, w) in enumerate([1.0, 3.0, 0.5, 2.5, 3.0])
        WarmupHMC._observe_sample!(sample, [float(i)], [0.0], w)
    end
    positions, _ = WarmupHMC._sample_evidence(sample)
    @test [mean(==(i), positions) for i in 1:5] ≈ [1, 3, 0.5, 2.5, 3] ./ 10 atol = 0.01
    WarmupHMC.reset!(sample)
    @test size(first(WarmupHMC._sample_evidence(sample)), 2) == 0
end

@testset "selection plan adapts a dense block ($evidence)" for evidence in
        (:linear_pool, :nuts_weighted)
    WS_SELECTIONS[] = 0
    empty!(WS_SIZES)
    rp = ws_problem()
    result = adaptive_warmup_mcmc(Xoshiro(20260926), rp; n_draws=1000,
        progress=nothing, nonlinear_evidence=evidence, recording_target=500)
    @test WS_SELECTIONS[] >= 1
    # One evidence size: the pool, or a same-size sample of the leaves.
    @test all(<=(500), WS_SIZES)
    evidence === :nuts_weighted && @test all(==(500), WS_SIZES)
    # The rule whitens the block: controls near log(s).
    @test WarmupHMC.reparametrizer(rp).c ≈ log.(WS_S) atol = 0.5
    # Draws come back in the model's own frame.
    u = WS_Q' * result.posterior_position[1:3, :]
    @test vec(std(u; dims=2)) ≈ WS_S rtol = 0.25
    @test std(result.posterior_position[4, :]) ≈ 1.0 rtol = 0.25
    @test result.n_divergent_samples <= 3
end

@testset "selection plan drives an IndexedReparametrization" begin
    ir = WarmupHMC.IndexedReparametrization([
        i => WarmupHMC.Reparametrization(WarmupHMC.PartiallyCentered(0.0),
            WarmupHMC.PartiallyCentered(0.0), 0.0, log(s))
        for (i, s) in enumerate([2.0, 0.5])
    ])
    selections = Ref(0)
    plan = WarmupHMC.WindowSelectionPlan(
        (ir, positions, gradients) -> begin
            selections[] += 1
            ir.pairs .= [idx => WarmupHMC.Reparametrization(value.target,
                WarmupHMC.PartiallyCentered(1.0), value.args...) for (idx, value) in ir.pairs]
            true
        end,
    )
    rp = WarmupHMC.ReparametrizedProblem(ir, DiagGaussian([0.0, 0.0], [2.0, 0.5]),
        AutoEnzyme(); scoring_plan=plan)
    # An easy Gaussian may never restart; force restarting windows.
    result = adaptive_warmup_mcmc(Xoshiro(5), rp; n_draws=500, progress=nothing,
        variance_cond_target=1.0)
    @test selections[] >= 1
    @test all(last(p).c == 1.0 for p in WarmupHMC.reparam_sources(rp))
    @test vec(std(result.posterior_position; dims=2)) ≈ [2.0, 0.5] rtol = 0.25
end

@testset "selection plan checkpoints its controls" begin
    dir = mktempdir()
    rp = ws_problem()
    adaptive_warmup_mcmc(Xoshiro(7), rp; n_draws=300, progress=nothing,
        checkpoint_dir=dir)
    payload = WarmupHMC.deserialize(joinpath(dir, "cp_latest.jls"))
    @test payload.reparam_sources isa Vector{Float64}
    @test payload.custom_candidate_scoring
    @test payload.nonlinear_recorder.online isa WarmupHMC.WeightedLeafSample
    # Resuming without the plan is refused, as for a scoring plan.
    plain = WarmupHMC.ReparametrizedProblem(RotScaleBlock(), DenseGaussian(WS_Q, WS_S), AutoEnzyme())
    @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(7), plain; n_draws=300,
        progress=nothing, checkpoint_dir=dir, resume=true)
end
end
