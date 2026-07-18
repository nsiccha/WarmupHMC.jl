using TestModules
using WarmupHMC, Random, Test

@testset "module loads" begin
    @test isdefined(WarmupHMC, :WarmupHMC)
end

include("cooperative.jl")
