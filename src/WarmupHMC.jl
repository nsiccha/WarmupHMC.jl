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
to_x1(xc, mean, log_sd, centeredness) = (
    mean + (xc - mean * centeredness) * exp(log_sd * (1 - centeredness))
)
to_xc(previous_xc, mean, log_sd, target_centeredness, previous_centeredness) = (
    mean * target_centeredness 
    + exp(log_sd * (target_centeredness - previous_centeredness)) 
        * (previous_xc - mean * previous_centeredness)
)
# to_xc(
#     to_x1(xcp, mean, log_sd, previous_centeredness),
#     mean, log_sd, target_centeredness
# )
# to_xc(x1, log_sd, centeredness) = x1 * exp(log_sd * (centeredness - 1))
# klp(x1, log_sd, centeredness) = klp(to_xc.(x1, log_sd, centeredness), exp.(log_sd .* centeredness))
klp(x1s, means, log_sds, centeredness) = klp(
    to_xc.(x1s, means, log_sds, centeredness), 
    means .* centeredness, 
    exp.(log_sds .* centeredness)
)
klp(xc, meanc, sdc) = mean(logpdf.(Normal.(meanc, sdc), xc)) + log(std(xc))
klps(x1s, means, log_sds, cs) = [klp(x1s, means, log_sds, c) for c in cs]

# @dynamic_object DiagonalScaling <: Reparametrization scaling::Vector
# reparametrize(what::DiagonalScaling, parameters) = what.scaling .* parameters
# unreparametrize(what::DiagonalScaling, reparameters) = what.scaling .\ reparameters
# logjacobian(what::DiagonalScaling, ::Any, ::Any) = 0
end
