using WarmupHMC, LogDensityProblems, Random

struct Gaussian end
LogDensityProblems.dimension(::Gaussian) = 2
LogDensityProblems.capabilities(::Type{Gaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::Gaussian, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(p::Gaussian, x) =
    (LogDensityProblems.logdensity(p, x), -x)

result = adaptive_warmup_mcmc(Xoshiro(71), Gaussian();
    n_draws=1000, monitor_ess=true, progress=nothing)
size(result.posterior_position)
