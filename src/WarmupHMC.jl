module WarmupHMC
using DynamicHMC, LogDensityProblems, LinearAlgebra, Pathfinder, ProgressMeter, Distributions, ElasticArrays, MCMCDiagnosticTools


include("MatrixExpressions.jl")    
include("WrappedLogDensityProblems.jl")    
include("adaptive_warmup_mcmc.jl")

end
