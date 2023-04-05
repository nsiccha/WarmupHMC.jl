using DynamicObjects, DynamicPlots, WarmupHMC
using BridgeStan
using Distributions, ReverseDiff, ChainRulesCore 
using PosteriorDB
using Statistics, LinearAlgebra, ForwardDiff, DataFrames
using Plots
using Random


"""
A posteriorDB posterior.
"""
@dynamic_object PDBPosterior name::String

database(what::PDBPosterior) = PosteriorDB.database()
posterior(what::PDBPosterior) = PosteriorDB.posterior(what.database, what.name)
model(what::PDBPosterior) = PosteriorDB.model(what.posterior)
dataset(what::PDBPosterior) = PosteriorDB.dataset(what.posterior)
datastring(what::PDBPosterior) = PosteriorDB.load(what.dataset, String)
reference_posterior(what::PDBPosterior) = PosteriorDB.reference_posterior(what.posterior)
constrained_df(what::PDBPosterior) = DataFrame(PosteriorDB.load(what.reference_posterior))
constrained_draws(what::PDBPosterior) = hcat([vcat(col...) for col in eachcol(what.constrained_df)]...)
stan_file(what::PDBPosterior) = PosteriorDB.path(PosteriorDB.implementation(what.model, "stan"))
bridgestan_model(what::PDBPosterior) = StanModel(stan_file=what.stan_file, data=what.datastring)
unconstrained_draws(what::PDBPosterior) = (
    bmod = what.bridgestan_model;
    vcat([
        param_unconstrain(bmod, collect(row))' for row in eachrow(what.constrained_draws)
    ]...) 
)

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
no_xcs(what::UnconstrainedDraws) = size(what.xcs, 2)
replace_xcs(what::UnconstrainedDraws, xcs) = DynamicObjects.update(what, unconstrained_draws=hcat(
    what.unconstrained_draws[:, 1:what.xcs_start-1], xcs, what.unconstrained_draws[:, what.xcs_end+1:end]
))
scatter_funnel(what::UnconstrainedDraws, i=1) = Scatter(
    what.xcs[:, i], what.log_sds[:, min(i, size(what.log_sds, 2))]
)
kl_plot(what::UnconstrainedDraws, i=1, cs=LinRange(0, 1, 100)) = Line(
    cs, klps(
        what.x1s[:, i], 
        what.means[:, min(i, size(what.means, 2))], 
        what.log_sds[:, min(i, size(what.log_sds, 2))], 
        cs
    )
)
best_centeredness(what::UnconstrainedDraws) = best_centeredness.([what], 1:size(what.x1s, 2))
best_centeredness(what::UnconstrainedDraws, i, cs=LinRange(0, 1, 100)) = cs[argmin(klps(
    what.x1s[:, i], 
    what.means[:, min(i, size(what.means, 2))], 
    what.log_sds[:, min(i, size(what.log_sds, 2))], 
    cs
))]

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
means(what::EightSchoolsDraws) = what.unconstrained_draws[:, 9]
log_sds(what::EightSchoolsDraws) = what.unconstrained_draws[:, 10]
x1s(what::EightSchoolsDraws) = what.unconstrained_draws[:, 1:8]

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
xcs_start(what::HSGPDraws) = size(what.unconstrained_draws, 2)
xcs_end(what::HSGPDraws) = 8
means(what::HSGPDraws) = 0
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

hsgp_draws(what::MotorcycleDraws) = [
    HSGPDraws(what.unconstrained_draws[:, 2:43]),
    # HSGPDraws(what.unconstrained_draws[:, 45:end])
]
log_sds(what::MotorcycleDraws) = hcat(log_sds.(what.hsgp_draws)...)
means(what::MotorcycleDraws) = hcat(means.(what.hsgp_draws)...)
x0s(what::MotorcycleDraws) = hcat(x0s.(what.hsgp_draws)...)
x1s(what::MotorcycleDraws) = hcat(x1s.(what.hsgp_draws)...)
xcs_start(what::MotorcycleDraws) = 4#47
xcs_end(what::MotorcycleDraws) = 43#size(what.unconstrained_draws, 2)

"""
A matrix of unconstrained draws from the centered radon_hierarchical_intercept_centered model:

parameters {
  vector[J] alpha;
  vector[2] beta;
  real mu_alpha;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_y;
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_hierarchical_intercept_centered.stan
"""
@dynamic_object RadonHierarchicalInterceptCenteredDraws <: UnconstrainedDraws unconstrained_draws::Matrix
xcs_start(what::RadonHierarchicalInterceptCenteredDraws) = 1
xcs_end(what::RadonHierarchicalInterceptCenteredDraws) = 85
means(what::RadonHierarchicalInterceptCenteredDraws) = what.unconstrained_draws[:, 88]
log_sds(what::RadonHierarchicalInterceptCenteredDraws) = what.unconstrained_draws[:, 89]
# x1s(what::RadonHierarchicalInterceptCenteredDraws) = what.unconstrained_draws[:, 1:8]


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


"""
Reparametrizations.
"""
@dynamic_type Reparametrization

@dynamic_object ContinuousNoncentering <: Reparametrization centeredness::Vector
previous_centeredness(what::ContinuousNoncentering) = 1
pad_left(what, ::AbstractVector) = what
pad_left(what, ::AbstractMatrix) = what'
reparametrize(what::ContinuousNoncentering, parameters) = replace_xcs(
    parameters,
    to_xc.(
        parameters.xcs, parameters.means, parameters.log_sds, 
        pad_left(what.centeredness, parameters.xcs), 
        pad_left(what.previous_centeredness, parameters.xcs)
    )
)
unreparametrize(what::ContinuousNoncentering, reparameters) = replace_xcs(
    reparameters,
    to_xc.(
        parameters.xcs, parameters.means, parameters.log_sds, 
        pad_left(what.previous_centeredness, parameters.xcs),
        pad_left(what.centeredness, parameters.xcs), 
    )
    # to_x1.(reparameters.xcs, reparameters.means, reparameters.log_sds, pad_left(what.centeredness, reparameters.xcs))
)
logjacobian(what::ContinuousNoncentering, parameters, reparameters) = sum(
    parameters.log_sds .* what.centeredness
)