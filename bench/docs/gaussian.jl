using WarmupHMC, Random
import LogDensityProblems as LDP

struct Gaussian end
LDP.dimension(::Gaussian) = 2
LDP.capabilities(::Type{Gaussian}) = LDP.LogDensityOrder{1}()
LDP.logdensity(::Gaussian, x) = -sum(abs2, x) / 2
LDP.logdensity_and_gradient(p::Gaussian, x) = (LDP.logdensity(p, x), -x)

result = adaptive_warmup_mcmc(Xoshiro(71), Gaussian();
    n_draws=1000, monitor_ess=true, progress=nothing)
size(result.posterior_position)
