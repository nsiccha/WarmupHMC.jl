using Test
using WarmupHMC
using Random, LinearAlgebra, Statistics
using LogDensityProblems
using Pkg, TOML
using Distributions
# `Enzyme` is NOT an idle import. `AutoEnzyme()` is a BACKEND HANDLE, not a
# backend: DifferentiationInterface can only differentiate through it when Enzyme
# itself is loaded in the session. `invariant_scoring.jl` and
# `wrapped_logdensity.jl` construct one directly. Remove this and those testsets
# do not go quiet — they error.
#
# Enzyme is the only backend this suite exercises, deliberately. The initializer
# does not need one: `mypathfinder` pins `adtype = NoAD()` precisely so
# Optimization never synthesizes a gradient of its own, so no second AD package
# is reachable from the tested paths.
using DifferentiationInterface, Enzyme

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
    @testset "Initialization errors" begin
        include("initialization_errors.jl")
    end
    @testset "adaptive_warmup_mcmc (end-to-end)" begin
        include("adaptive_warmup.jl")
    end
    @testset "interruptible stream_mcmc" begin
        include("interruptible.jl")
    end
    include("readme.jl")
    include("compat.jl")
    include("public_api.jl")
end
