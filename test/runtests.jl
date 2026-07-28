using Test
using WarmupHMC
using Random, LinearAlgebra, Statistics
using LogDensityProblems
using Pkg, TOML
using Distributions
using DifferentiationInterface, ForwardDiff   # loads WarmupHMC's DifferentiationInterfaceExt

include("test_problems.jl")

@testset "DiagGaussian fixture sanity" begin
    n = 4
    mu = randn(n); sigma = rand(n) .+ 0.5
    p = DiagGaussian(mu, sigma)
    x = randn(n)
    @test LogDensityProblems.dimension(p) == n
    @test LogDensityProblems.capabilities(typeof(p)) == LogDensityProblems.LogDensityOrder{1}()
    # matches Distributions exactly (normalizing constant included)
    @test LogDensityProblems.logdensity(p, x) ≈ logpdf(MvNormal(mu, Diagonal(sigma .^ 2)), x)
    # analytic gradient matches finite differences
    _, g = LogDensityProblems.logdensity_and_gradient(p, x)
    @test g ≈ fd_gradient(z -> LogDensityProblems.logdensity(p, z), x) rtol = 1e-5
end

@testset "WarmupHMC.jl" begin
    @testset "MatrixExpressions" begin
        include("matrix_expressions.jl")
    end
    @testset "Reparametrizations" begin
        include("reparametrizations.jl")
    end
    @testset "Invariant candidate scoring" begin
        include("invariant_scoring.jl")
    end
    @testset "WrappedLogDensityProblems + AD" begin
        include("wrapped_logdensity.jl")
    end
    @testset "adaptive_warmup_mcmc (end-to-end)" begin
        include("adaptive_warmup.jl")
    end
    include("readme.jl")
    include("compat.jl")
    include("public_api.jl")
end
