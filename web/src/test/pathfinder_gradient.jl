@testitem "Pathfinder uses the supplied gradient" setup=[Determinism] tags=[:init] begin
    using WarmupHMC, LogDensityProblems, Random

    # Like a native-backed LogDensityProblems target, this density deliberately
    # accepts only ordinary Float64 vectors. Its supplied gradient is the only
    # valid differentiation path; ForwardDiff.Dual inputs must never reach it.
    struct GradientOnlyLP
        dimension::Int
    end
    LogDensityProblems.dimension(p::GradientOnlyLP) = p.dimension
    LogDensityProblems.capabilities(::Type{GradientOnlyLP}) =
        LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::GradientOnlyLP, x::AbstractVector{Float64}) =
        -sum(abs2, x) / 2
    LogDensityProblems.logdensity_and_gradient(::GradientOnlyLP, x::AbstractVector{Float64}) =
        (-sum(abs2, x) / 2, -x)

    @testset "Pathfinder uses the supplied LogDensityProblems gradient" begin
        init = WarmupHMC.initialize_mcmc(
            GradientOnlyLP(1), missing; rng=Xoshiro(1), progress=nothing
        )

        @test init.position isa Vector{Float64}
        @test length(init.position) == 1
        @test size(init.squared_scale) == (1, 1)
    end
end
