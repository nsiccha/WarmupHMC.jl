using DynamicObjects, DynamicPlots, WarmupHMC
using BridgeStan
using Distributions, ReverseDiff, ChainRulesCore 

"""
A type whose instances contain unconstrained draws.
"""
@dynamic_type UnconstrainedDraws

subset(what::UnconstrainedDraws, no_draws=size(what, 1) ÷ 2) = DynamicObjects.update(
    what,
    unconstrained_draws=what.unconstrained_draws[rand(axes(what, 1), no_draws), :]
)
Base.size(what::UnconstrainedDraws, args...) = size(what.unconstrained_draws, args...)
Base.axes(what::UnconstrainedDraws, args...) = axes(what.unconstrained_draws, args...)
sds(what::UnconstrainedDraws) = exp.(what.log_sds)
xcs(what::UnconstrainedDraws) = what.unconstrained_draws[:, what.xcs_start:what.xcs_end]
replace_xcs(what::UnconstrainedDraws, xcs) = DynamicObjects.update(what, unconstrained_draws=hcat(
    what.unconstrained_draws[:, 1:what.xcs_start-1], xcs, what.unconstrained_draws[:, what.xcs_end+1:end]
))
scatter_funnel(what::UnconstrainedDraws, i=1) = Scatter(what.x1s[:, i], what.log_sds[:, i])
kl_plot(what::UnconstrainedDraws, i=1, cs=LinRange(0, 1, 100)) = Line(
    cs, klps(
        what.x1s[:, i], 
        what.means[:, min(i, size(what.means, 2))], 
        what.log_sds[:, min(i, size(what.log_sds, 2))], 
        cs
    )
)

PairPlot(what::UnconstrainedDraws, i, j) = i < j ? Scatter(
    what.unconstrained_draws[:, i], what.unconstrained_draws[:, j]
) : EmptyPlot()

@dynamic_type UnconstrainedDraw <: AbstractVector{Any}
Base.size(what::UnconstrainedDraw, args...) = size(what.unconstrained_draw, args...)
Base.axes(what::UnconstrainedDraw, args...) = axes(what.unconstrained_draw, args...)
Base.getindex(what::UnconstrainedDraw, args...) = getindex(what.unconstrained_draw, args...)
Base.collect(what::UnconstrainedDraw) = collect(what.unconstrained_draw)
sds(what::UnconstrainedDraw) = exp.(what.log_sds)
xcs(what::UnconstrainedDraw) = what.unconstrained_draw[what.xcs_start:what.xcs_end]
replace_xcs(what::UnconstrainedDraw, xcs) = DynamicObjects.update(what, unconstrained_draw=vcat(
    what.unconstrained_draw[1:what.xcs_start-1], xcs, what.unconstrained_draw[what.xcs_end+1:end]
))

"""
A matrix of unconstrained draws from the centered eight schools model:

parameters {
  real theta[J]; // treatment effect in school j
  real mu; // hyper-parameter of mean
  real<lower=0> tau; // hyper-parameter of sdv
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/eight_schools_centered.stan
"""
@dynamic_object EightSchoolsDraws <: UnconstrainedDraws unconstrained_draws::Matrix
xcs_start(what::EightSchoolsDraws) = 1
xcs_end(what::EightSchoolsDraws) = 8
log_sds(what::EightSchoolsDraws) = what.unconstrained_draws[:, end]
means(what::EightSchoolsDraws) = what.unconstrained_draws[:, end-1]
x1s(what::EightSchoolsDraws) = what.unconstrained_draws[:, 1:end-2]

@dynamic_object EightSchoolsDraw <: UnconstrainedDraw unconstrained_draw::AbstractVector
xcs_start(what::EightSchoolsDraw) = 1
xcs_end(what::EightSchoolsDraw) = 8
means(what::EightSchoolsDraw) = what.unconstrained_draw[9]
log_sds(what::EightSchoolsDraw) = what.unconstrained_draw[10]

"""
The unconstrained parameters of an HSGP:

sds = alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
"""
@dynamic_object HSGPDraws <: UnconstrainedDraws unconstrained_draws::AbstractMatrix L=1.5
no_basis_functions(what::HSGPDraws) = size(what.unconstrained_draws, 2)-2
indices(what::HSGPDraws) = 1:what.no_basis_functions
# https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
# alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
function log_sds(what::HSGPDraws)
    log_sd = what.unconstrained_draws[:, 1]
    log_lengthscale = what.unconstrained_draws[:, 2]
    lengthscale = exp.(log_lengthscale)
    (
        log_sd
        .+ log.(sqrt.(sqrt(2*pi)))
        .+ .5 .* log_lengthscale
        .- .25 .* (lengthscale.*(pi/2/what.L)).^2 .* what.indices'.^2
    )
end
x0s(what::HSGPDraws) = what.unconstrained_draws[:, 3:end]
x1s(what::HSGPDraws) = what.x0s .* what.sds
means(what::HSGPDraws) = 0

"""
A matrix of unconstrained draws from the MotorcycleGP model:

parameters {
  // temporary intercept for centered predictors
  real Intercept;
  // GP standard deviation parameters
  real<lower=0> sdgp_1;
  // GP length-scale parameters
  real<lower=0> lscale_1;
  // latent variables of the GP
  vector[NBgp_1] zgp_1;
  // temporary intercept for centered predictors
  real Intercept_sigma;
  // GP standard deviation parameters
  real<lower=0> sdgp_sigma_1;
  // GP length-scale parameters
  real<lower=0> lscale_sigma_1;
  // latent variables of the GP
  vector[NBgp_sigma_1] zgp_sigma_1;
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/accel_gp.stan
"""
@dynamic_object MotorcycleDraws <: UnconstrainedDraws unconstrained_draws::Matrix

hsgps(what::MotorcycleDraws) = [
    HSGPDraws(what.unconstrained_draws[:, 2:43]),
    HSGPDraws(what.unconstrained_draws[:, 45:end])
]
log_sds(what::MotorcycleDraws) = hcat(log_sds.(what.hsgps)...)
means(what::MotorcycleDraws) = hcat(means.(what.hsgps)...)
x0s(what::MotorcycleDraws) = hcat(x0s.(what.hsgps)...)
x1s(what::MotorcycleDraws) = hcat(x1s.(what.hsgps)...)
# scatter_x1(what::MotorcycleDraws, i=1) = scatter(what.x1s[:, i], what.log_sds[:, i], alpha=.2)
# scatter_x0(what::MotorcycleDraws, i=1) = scatter(what.x0s[:, i], what.log_sds[:, i], alpha=.2)

"""
A wrapper around a BridgeStan model.
"""
@dynamic_object BSModel model::StanModel

logpdfbs(what::BSModel, parameters::Vector) = log_density(what.model, parameters)
# logpdfbs(what::BSModel, parameters::AbstractVector) = log_density(what.model, collect(parameters))
Distributions.logpdf(what::BSModel, parameters::Vector) = logpdfbs(what, parameters)
Distributions.logpdf(what::BSModel, parameters::AbstractVector) = logpdf(what, collect(parameters))
function ChainRulesCore.rrule(::typeof(logpdfbs), what, parameters)
    rv = log_density_gradient(what.model, parameters)
    return rv[1], rva -> (NoTangent(), NoTangent(), rva .* rv[2])
end
ReverseDiff.@grad_from_chainrules logpdfbs(what, parameters::ReverseDiff.TrackedArray)

"""
A reparametrized distribution.
"""
@dynamic_object ReparametrizedDistribution distribution reparametrization
Distributions.logpdf(what::ReparametrizedDistribution, reparameters) = (
    parameters = unreparametrize(what.reparametrization, reparameters);
    (
        logpdfbs(what.distribution, collect(parameters)) 
        + logjacobian(what.reparametrization, parameters, reparameters)
    )
)
