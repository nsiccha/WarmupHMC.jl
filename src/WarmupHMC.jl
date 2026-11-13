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
    StatsBase,
    JSON,
    OnlineStatsBase,
    Random,
    Mmap,
    CRC32c,
    Dates
using Serialization: serialize, deserialize
import ReparametrizableDistributions: PartiallyCentered, reparam
# Only the reparametrization gradient and the window-end halo transport need AD.
# Importing the names rather than the package keeps DifferentiationInterface's
# broad export surface (`gradient`, `jacobian`, `pullback`, …) out of WarmupHMC.
using DifferentiationInterface: value_and_gradient, value_and_gradient!,
    prepare_gradient, Constant, Cache

export adaptive_warmup_mcmc, resume_warmup_mcmc, cooperative_warmup_mcmc,
    ReparametrizedProblem, IndexedReparametrization, PartiallyCentered, Reparametrization,
    SharedAccessor,
    CandidateScoringPlan, candidate_scoring_losses, WindowSelectionPlan,
    stream_mcmc, open_stream, close_stream, FixedKernelNUTSState, fixed_kernel_nuts_step,
    ContinuousNUTSState, continuous_nuts_step, reposition!,
    completion_warmup_mcmc, scale_data, scale_from_data, fit_gaussian_moments

include("kwarg_validation.jl")
include("blas_threads.jl")
include("MatrixExpressions.jl")
include("moment_scale.jl")
include("NUTSLeafWeights.jl")
include("WrappedLogDensityProblems.jl")
include("Reparametrizations.jl")
include("run_files.jl")
include("adaptive_warmup_mcmc.jl")
include("weighted_scale.jl")
include("checkpoint_rescore.jl")
include("retained_rhat.jl")
include("cooperative_warmup_mcmc.jl")
include("completion_warmup_mcmc.jl")
include("interruptible_mcmc.jl")
include("continuous_nuts.jl")
include("progress.jl")

end
