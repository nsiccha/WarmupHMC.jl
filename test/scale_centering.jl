@testmodule ScaleCenteringFixtures begin
# Snag nested-centering-3858d508: a centering that follows the coordinate's own
# scale, `ScaleCentered`, selected only by the opt-in two-pass warm-up
# `adaptive_warmup_mcmc(...; scale_centering = true)`.
using WarmupHMC, LogDensityProblems, LinearAlgebra
export EightSchools, Mu, LogTau, eight_schools_ir, logistic_

# Eight schools, non-centred in the model: x = [mu; log tau; z[1:8]].
struct EightSchools end
const Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
LogDensityProblems.dimension(::EightSchools) = 10
LogDensityProblems.capabilities(::Type{EightSchools}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(t::EightSchools, x) = first(LogDensityProblems.logdensity_and_gradient(t, x))
function LogDensityProblems.logdensity_and_gradient(::EightSchools, x)
    mu, ltau = x[1], x[2]
    tau = exp(ltau)
    z = @view x[3:10]
    g = zeros(eltype(x), 10)
    # mu ~ N(0, 5), tau ~ HalfCauchy(5) with the log transform's Jacobian, z ~ N(0, 1)
    lp = -mu^2 / 50 - log1p((tau / 5)^2) + ltau - dot(z, z) / 2
    g[1] = -mu / 25
    g[2] = -2 * (tau / 5)^2 / (1 + (tau / 5)^2) + 1
    g[3:10] .= -z
    for j in 1:8
        r = (Y[j] - (mu + tau * z[j])) / SIGMA[j]^2
        lp -= (Y[j] - (mu + tau * z[j]))^2 / (2 * SIGMA[j]^2)
        g[1] += r
        g[2] += r * tau * z[j]
        g[2+j] += r * tau
    end
    lp, g
end
struct Mu <: Function end
(::Mu)(x) = x[1]
struct LogTau <: Function end
(::LogTau)(x) = x[2]
eight_schools_ir(source=PartiallyCentered(0.0)) = IndexedReparametrization(
    [2 + j => Reparametrization(PartiallyCentered(0.0), source, Mu(), LogTau()) for j in 1:8])
logistic_(u) = 1 / (1 + exp(-u))
end

@testitem "ScaleCentered maps as the scale-dependent partial centering" setup=[ScaleCenteringFixtures] begin
    using Test, WarmupHMC, LogDensityProblems, Random
    using DifferentiationInterface, Enzyme
    ir = eight_schools_ir(ScaleCentered(-0.5))
    x = randn(Xoshiro(1), 10)
    ljac, y = ir(x)
    mu, l = x[1], x[2]
    c = logistic_(2 * (l - (-0.5)))
    @test y[1:2] == x[1:2]
    @test y[3:10] ≈ (x[3:10] .- c * mu) .* exp(-c * l)
    # log|dy/dx| = l (target.c - c(l)) per pair, target.c = 0
    @test ljac ≈ 8 * l * (0 - c)
    _, back = WarmupHMC._inverse_with_logabsdet_jacobian(ir, y)
    @test back ≈ x
    # The gradient differentiates through c(l): compare with central differences.
    rp = ReparametrizedProblem(ir, EightSchools(), AutoEnzyme())
    _, grad = LogDensityProblems.logdensity_and_gradient(rp, x)
    h = 1e-6
    fd = [(LogDensityProblems.logdensity(rp, x .+ h .* (1:10 .== k)) -
           LogDensityProblems.logdensity(rp, x .- h .* (1:10 .== k))) / 2h for k in 1:10]
    @test grad ≈ fd rtol=1e-5
end

@testitem "Default fits never see a ScaleCentered candidate" setup=[ScaleCenteringFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme
    ir = eight_schools_ir()
    # Built exactly as before the opt-in existed: the source field is not widened.
    @test fieldtype(typeof(last(ir.pairs[1])), :source) === PartiallyCentered{Float64}
    rp = ReparametrizedProblem(ir, EightSchools(), AutoEnzyme())
    positions = randn(Xoshiro(2), 10, 50)
    gradients = reduce(hcat, [last(LogDensityProblems.logdensity_and_gradient(rp, p)) for p in eachcol(positions)])
    rows = candidate_scoring_losses(rp, positions, gradients)
    @test all(r -> r.candidate isa Float64, rows)
    @test length(rows) == 8 * 11
end

@testitem "scale_centering = true: two passes, the caller's problem untouched" setup=[ScaleCenteringFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme
    rp = ReparametrizedProblem(eight_schools_ir(), EightSchools(), AutoEnzyme())
    r = adaptive_warmup_mcmc(Xoshiro(3), rp; n_draws=100, scale_centering=true)
    @test size(r.posterior_position) == (10, 100)
    # Both passes' gradients are counted; the second pass's sampling is in there.
    @test r.first_pass.total_evaluation_counter > 0
    @test r.total_evaluation_counter > r.first_pass.total_evaluation_counter
    @test r.sampling_evaluation_counter <= r.total_evaluation_counter - r.first_pass.total_evaluation_counter
    @test length(r.selected_centering) == 8
    # A short run need not reach a restarting window, so whether it moved any
    # pair is not pinned here; the selection itself is, below.
    @test all(p -> last(p) isa Union{PartiallyCentered,ScaleCentered}, r.selected_centering)
    # Not mutated: the caller's centerings are those it built.
    @test all(p -> last(p) == PartiallyCentered(0.0), WarmupHMC.reparam_sources(rp))
    # Draws are in the model's coordinates, as for every fit.
    @test all(isfinite, r.posterior_position)

    # Pass 1's selection: on a pool whose log-scale varies, the widened copy
    # scores midpoints only, so every pair ends `ScaleCentered`.
    widened = WarmupHMC._scale_centering_problem(rp)
    @test fieldtype(typeof(last(WarmupHMC.reparametrizer(widened).pairs[1])), :source) ===
          Union{PartiallyCentered{Float64},ScaleCentered{Float64}}
    positions = randn(Xoshiro(9), 10, 200)
    gradients = reduce(hcat, [last(LogDensityProblems.logdensity_and_gradient(widened, p)) for p in eachcol(positions)])
    WarmupHMC.optimize!(WarmupHMC.reparametrizer(widened), positions, gradients)
    @test all(p -> last(p).source isa ScaleCentered, WarmupHMC.reparametrizer(widened).pairs)
    @test all(p -> last(p) == PartiallyCentered(0.0), WarmupHMC.reparam_sources(rp))

    # The multi-chain method runs both passes per chain.
    rs = adaptive_warmup_mcmc([Xoshiro(4), Xoshiro(5)], rp; n_draws=50, parallel=false, scale_centering=true)
    @test length(rs) == 2 && all(x -> haskey(x, :selected_centering), rs)
end

@testitem "A pair built with a ScaleCentered source adapts among midpoints" setup=[ScaleCenteringFixtures] begin
    using Test, WarmupHMC, Random
    using DifferentiationInterface, Enzyme
    # Its vector holds only `ScaleCentered` sources, so selection must never
    # write a constant into it: midpoints where the pool places them, else itself.
    rp = ReparametrizedProblem(eight_schools_ir(ScaleCentered(0.0)), EightSchools(), AutoEnzyme())
    r = adaptive_warmup_mcmc(Xoshiro(8), rp; n_draws=50)
    @test size(r.posterior_position, 2) == 50
    @test all(p -> last(p) isa ScaleCentered, WarmupHMC.reparam_sources(rp))
    @test WarmupHMC.reparametrization_candidates(ScaleCentered(0.4)) == [ScaleCentered(0.4)]
    @test WarmupHMC.reparametrization_candidates(PartiallyCentered(0.4)) == PartiallyCentered.(range(0, 1, 11))
end

@testitem "scale_centering = true: a given start keeps its model point" setup=[ScaleCenteringFixtures] begin
    using Test, WarmupHMC, Random
    initial = eight_schools_ir(PartiallyCentered(0.3))
    selected = eight_schools_ir(ScaleCentered(1.2))
    x = randn(Xoshiro(6), 10)
    moved = WarmupHMC._scale_centering_position(x, initial, selected)
    @test last(selected(moved)) ≈ last(initial(x))
    nt = WarmupHMC._scale_centering_init((; position=x, squared_scale=ones(10)), initial, selected)
    @test nt.position ≈ moved && nt.squared_scale == ones(10)
    @test ismissing(WarmupHMC._scale_centering_init(missing, initial, selected))
end

@testitem "scale_centering = true refuses what it cannot honour" setup=[ScaleCenteringFixtures] begin
    using Test, WarmupHMC, Random
    using DifferentiationInterface, Enzyme
    rp() = ReparametrizedProblem(eight_schools_ir(), EightSchools(), AutoEnzyme())
    run(; kw...) = adaptive_warmup_mcmc(Xoshiro(7), rp(); n_draws=20, scale_centering=true, kw...)
    @test_throws ArgumentError run(nonlinear_adapt=false)
    @test_throws ArgumentError run(nonlinear_evidence=:all_good_leaves)
    @test_throws ArgumentError run(callback=(state, stage) -> false)
    @test_throws ArgumentError run(checkpoint_dir=mktempdir())
    @test_throws ArgumentError run(stop_requested=() -> false)
    # Not a reparametrized problem.
    @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(7), EightSchools(); scale_centering=true)
    # Every log-scale a constant: no sampled scale to follow.
    constant = IndexedReparametrization([2 + j => Reparametrization(PartiallyCentered(0.0),
        PartiallyCentered(0.0), Mu(), 0.0) for j in 1:8])
    @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(7),
        ReparametrizedProblem(constant, EightSchools(), AutoEnzyme()); scale_centering=true)
    # The samplers built on it refuse the keyword by name.
    @test_throws ArgumentError completion_warmup_mcmc([Xoshiro(1), Xoshiro(2)], rp(); scale_centering=true)
    @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(1), rp(); scale_centering=true)
end
