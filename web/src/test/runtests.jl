using TestModules
using WarmupHMC, Random

@testset "module loads" begin
    @test isdefined(WarmupHMC, :WarmupHMC)
end
