using Test, WarmupHMC, Random

include("leaf_weights.jl")
include("namedtuple_init.jl")
include("pathfinder_gradient.jl")

@testset "module loads" begin
    @test isdefined(WarmupHMC, :WarmupHMC)
end

include("resume_api.jl")
include("cooperative.jl")
include("clustered.jl")
include("kwarg_validation.jl")
include("clustered_checkpoint.jl")
include("checkpoint_dropped_draws.jl")
