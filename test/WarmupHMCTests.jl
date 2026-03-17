@testmodule begin

@testset "module loads" begin
    @test isdefined(WarmupHMC, :WarmupHMC)
end

end # @testmodule
