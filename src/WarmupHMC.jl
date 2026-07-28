module WarmupHMC
using LogDensityProblems,
    LinearAlgebra,
    Pathfinder,
    Distributions,
    ElasticArrays,
    MCMCDiagnosticTools,
    TSVD,
    FillArrays,
    Statistics,
    LogExpFunctions,
    InverseFunctions,
    Treebars
import DynamicHMC,
    OnlineStatsBase,
    Random
using Serialization: serialize, deserialize
# Only the reparametrization gradient needs AD, and it needs exactly one entry
# point. Importing the name rather than the package keeps DifferentiationInterface's
# broad export surface (`gradient`, `jacobian`, `pullback`, …) out of WarmupHMC.
using DifferentiationInterface: value_and_gradient

export adaptive_warmup_mcmc, resume_warmup_mcmc, cooperative_warmup_mcmc, clustered_warmup_mcmc,
    ReparametrizedProblem, IndexedReparametrization, PartiallyCentered, Reparametrization

include("kwarg_validation.jl")
include("MatrixExpressions.jl")
include("NUTSLeafWeights.jl")
include("WrappedLogDensityProblems.jl")
include("Reparametrizations.jl")
include("adaptive_warmup_mcmc.jl")
include("pooled_scale.jl")
include("clustering.jl")
include("cooperative_warmup_mcmc.jl")
include("clustered_warmup_mcmc.jl")
include("progress.jl")

end
