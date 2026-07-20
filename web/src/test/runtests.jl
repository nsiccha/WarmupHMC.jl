using Test, WarmupHMC, Random

include("leaf_weights.jl")

@testset "module loads" begin
    @test isdefined(WarmupHMC, :WarmupHMC)
end

include("cooperative.jl")
include("clustered.jl")
include("kwarg_validation.jl")
include("clustered_checkpoint.jl")
