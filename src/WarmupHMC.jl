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
    Random,
    Mmap,
    CRC32c,
    Dates
using Serialization: serialize, deserialize
# Only the reparametrization gradient and the window-end halo transport need AD.
# Importing the names rather than the package keeps DifferentiationInterface's
# broad export surface (`gradient`, `jacobian`, `pullback`, …) out of WarmupHMC.
using DifferentiationInterface: value_and_gradient, value_and_gradient!,
    prepare_gradient, Constant, Cache

export adaptive_warmup_mcmc, resume_warmup_mcmc, cooperative_warmup_mcmc,
    ReparametrizedProblem, IndexedReparametrization, PartiallyCentered, Reparametrization,
    CandidateScoringPlan, candidate_scoring_losses, WindowSelectionPlan,
    stream_mcmc, open_stream, close_stream, FixedKernelNUTSState, fixed_kernel_nuts_step,
    completion_warmup_mcmc

include("kwarg_validation.jl")
include("MatrixExpressions.jl")
include("NUTSLeafWeights.jl")
include("WrappedLogDensityProblems.jl")
include("Reparametrizations.jl")
include("run_files.jl")
include("adaptive_warmup_mcmc.jl")
include("weighted_scale.jl")
include("cooperative_warmup_mcmc.jl")
include("completion_warmup_mcmc.jl")
include("interruptible_mcmc.jl")
include("progress.jl")

end
