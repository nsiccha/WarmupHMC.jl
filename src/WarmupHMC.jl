module WarmupHMC

export regularize, to_x1, to_xc, klp, klps#, klps_plot!

using DynamicObjects
using Distributions, LinearAlgebra

regularize(sample_covariance, no_draws, regularization_no_draws=5, regularization_constant=1e-3) = (
  no_draws / ((no_draws + regularization_no_draws) * (no_draws - 1)) * sample_covariance + regularization_constant * (regularization_no_draws / (no_draws + regularization_no_draws)) * I
)

# to_x1(xc, log_sd, centeredness) = xc * exp(log_sd * (1 - centeredness))
# xc = mu * c + sigma^c * x0
# x0 = (x1 - mu) / sigma
# xc = mu * c + sigma^(c-1) * (x1 - mu)
to_xc(x1, mean, log_sd, centeredness) = (
    mean * centeredness + exp(log_sd * (centeredness - 1)) * (x1 - mean)
)
# to_xc(x1, log_sd, centeredness) = x1 * exp(log_sd * (centeredness - 1))
# klp(x1, log_sd, centeredness) = klp(to_xc.(x1, log_sd, centeredness), exp.(log_sd .* centeredness))
klp(x1s, means, log_sds, centeredness) = klp(
    to_xc.(x1s, means, log_sds, centeredness), 
    means .* centeredness, 
    exp.(log_sds .* centeredness)
)
klp(xc, meanc, sdc) = mean(logpdf.(Normal.(meanc, sdc), xc)) + log(std(xc))
klps(x1s, means, log_sds, cs) = [klp(x1s, means, log_sds, c) for c in cs]


end
