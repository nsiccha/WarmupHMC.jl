module StanLogDensityProblemsExt
import WarmupHMC, StanLogDensityProblems
# WarmupHMC.adaptive_warmup_mcmc(rngs::AbstractArray, lpdf::StanLogDensityProblems.StanProblem; parallel=true, kwargs...) = begin
#     parallel && @info "Disabling parallelism due to https://github.com/roualdes/bridgestan/issues/67"
#     parallel && @info "Enabling parallelism due to https://github.com/roualdes/bridgestan/pull/273"
#     WarmupHMC.adaptive_warmup_mcmc(rngs, fill(lpdf, size(rngs)); parallel, kwargs...)
# end
end