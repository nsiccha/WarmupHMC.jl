using TestEnv; TestEnv.activate("WarmupHMC");
using WarmupHMC, Test
using LogDensityProblems, Distributions
# using Test, Documenter, WarmupHMC
# doctest(WarmupHMC)

@testset "WarmupHMC.jl" begin
    # cldp = ConvenientLogDensityProblem(
    #     [Normal(0,1)], info->println(info)
    # )
    # println(cldp)
    # LogDensityProblems.logdensity(cldp, [1])


end
