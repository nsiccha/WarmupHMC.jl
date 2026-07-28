using Test
using WarmupHMC
using Random, LinearAlgebra, Statistics
using LogDensityProblems
using Pkg, TOML
using Distributions
# `ForwardDiff` is NOT an idle import, and the reason it used to give was wrong:
# it said "loads WarmupHMC's DifferentiationInterfaceExt". There is no such
# extension. `53f82ea` made DifferentiationInterface a hard dependency and
# deleted it, leaving a comment that named a mechanism the package no longer has
# — an invitation to drop the import as vestigial.
#
# The real reason is that `AutoForwardDiff()` is a BACKEND HANDLE, not a
# backend: DifferentiationInterface can only differentiate through it when
# ForwardDiff is loaded in the session. `invariant_scoring.jl` and
# `wrapped_logdensity.jl` construct one directly, and Pathfinder's own default
# initialization path is `AutoForwardDiff()` too. Remove this and those testsets
# do not go quiet — they error.
#
# This is the deliberate exception to the repo's "DifferentiationInterface where
# possible, Enzyme directly where not" rule: it is confined to test code that
# names the backend explicitly, and to the frozen `golden_awm.jl` baseline pin.
using DifferentiationInterface, ForwardDiff

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
