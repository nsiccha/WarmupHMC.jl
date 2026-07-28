using Test, WarmupHMC, Random, LinearAlgebra

# Reproducibility: the adaptive transformation update runs multithreaded BLAS,
# whose reduction order is not deterministic run-to-run. Every byte-identity
# comparison in this suite is verified under this pin.
BLAS.set_num_threads(1)

# ForwardDiff / DifferentiationInterface are not direct dependencies of
# WarmupHMC — DifferentiationInterface is a `[weakdeps]` entry and ForwardDiff
# arrives transitively via Pathfinder — so a plain `using` fails under
# `--project=.`. `ad_backend.jl` loads them by UUID, and aborts loudly rather
# than letting the reparametrization testsets silently skip.
include("ad_backend.jl")

@testset "module loads" begin
    @test isdefined(WarmupHMC, :WarmupHMC)
end

# --- Leaf recording / halo pool ---------------------------------------------
include("leaf_weights.jl")
include("halo_recording.jl")

# --- Initialization ----------------------------------------------------------
include("namedtuple_init.jl")
include("pathfinder_gradient.jl")

# --- Reparametrization -------------------------------------------------------
include("reparametrization_unit.jl")
include("reparametrize_direction.jl")
include("reparametrized_e2e.jl")

# --- Samplers, resume, checkpointing -----------------------------------------
include("resume_api.jl")
include("cooperative.jl")
include("clustered.jl")
include("kwarg_validation.jl")
include("checkpoint_atomicity.jl")
include("clustered_checkpoint.jl")
include("cooperative_checkpoint.jl")
include("checkpoint_dropped_draws.jl")

# --- Golden harness (byte-identity + property assertions) --------------------
# Slowest file in the suite; last so a failure elsewhere surfaces sooner.
include("golden_awm.jl")
