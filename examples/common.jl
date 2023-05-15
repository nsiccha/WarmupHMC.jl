using DynamicObjects, DynamicPlots, WarmupHMC
using BridgeStan
using Distributions, ReverseDiff, ChainRulesCore 
using PosteriorDB
using Statistics, LinearAlgebra, ForwardDiff, DataFrames
using MCMCDiagnosticTools
using LogDensityProblems
using Plots
using Plots.PlotMeasures
using Random
using JSON
using AdvancedHMC
using Random
using ThreadsX

to_unit_range(x) = (
    x = x .- minimum(x);
    x ./ maximum(x)
)
function println_finite_problem(pre, x)
    # if !all(isfinite.(x))
    # idxs = (1:length(x))[(y->!isfinite(y)).(x)]
    # println(pre, isfinite.(x))
    # println(x[idxs])
    # end
end

"""
A base type for distributions.
"""
@dynamic_type DynamicDistribution
# Distributions.logpdf(what, ::DummyDistributionInfo, parameters) = logpdf(what, parameters)
LogDensityProblems.logdensity(what::DynamicDistribution, parameters) = logpdf(what, parameters)
LogDensityProblems.dimension(what::DynamicDistribution) = what.no_dimensions
LogDensityProblems.capabilities(::Type{DynamicDistribution{T}}) where T = LogDensityProblems.LogDensityOrder{0}()
logpdf_gradient(what::DynamicDistribution, parameters) = ReverseDiff.gradient(x->logpdf(what, x), parameters)
function unconstrained_draws(what::DynamicDistribution, no_draws=1000, seed=1, progress=true)
    rng = Xoshiro(seed)
    # https://github.com/TuringLang/AdvancedHMC.jl
    D = what.no_dimensions
    initial_θ = rand(rng, D)
    while !all(isfinite.(logpdf_gradient(what, initial_θ)))
        println(isfinite.(logpdf_gradient(what, initial_θ)))
        # readline()
        initial_θ = rand(rng, D)
    end
    n_samples, n_adapts = 1_000 + no_draws, 1_000
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, what, ReverseDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    samples, stats = AdvancedHMC.sample(
        rng,
        hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=progress, verbose=false
    )
    UnconstrainedDraws(
        what,
        hcat(samples[1001:end]...)',
        all_stats=stats,
        warmup_draws=samples[1:1000]
    )
end
ccat(args...) = UnconstrainedDraws(
    args[1].distribution,
    vcat(args...),
    all_stats=vcat(getproperty.(args, :all_stats)...),
    no_chains=sum(getproperty.(args, :no_chains))
)
unconstrained_chains(what::DynamicDistribution, no_draws=1000, no_chains=16) = ccat(ThreadsX.collect(
    unconstrained_draws(what, no_draws, chain, false) for chain in 1:no_chains
)...)
reference_chains(what::DynamicDistribution, no_draws=10000, no_chains=16, thin=10) = thinned(
    unconstrained_chains(what, no_draws, no_chains), thin
)

"""
A base type for ....
"""
@dynamic_type DistributionInfo
Base.getindex(what::DistributionInfo, args...) = getindex(what.data, args...)
no_xcs(what::DistributionInfo) = length(what.xcs_idxs)

# info(::DynamicDistribution) = DistributionInfo()
@dynamic_object DummyDistribution <: DynamicDistribution info::DistributionInfo
# centeredness(what::DynamicDistribution) = what.info.centeredness
# xcs_idxs(what::DynamicDistribution) = what.info.xcs_idxs
# means_idxs(what::DynamicDistribution) = what.info.means_idxs
# log_sds_idxs(what::DynamicDistribution) = what.info.log_sds_idxs


"""
A base type whose instances contain a matrix of unconstrained draws.
"""
@dynamic_type AbstractUnconstrainedDraws <: AbstractMatrix{Real}
"""
A type whose instances contain a matrix of unconstrained draws.
"""
@dynamic_object UnconstrainedDraws <: AbstractUnconstrainedDraws distribution::DynamicDistribution unconstrained_draws::AbstractMatrix
pack(distribution::DynamicDistribution, unconstrained_draws::AbstractMatrix) =  UnconstrainedDraws(
    distribution, unconstrained_draws
)

Base.size(what::UnconstrainedDraws, args...) = size(what.unconstrained_draws, args...)
Base.axes(what::UnconstrainedDraws, args...) = axes(what.unconstrained_draws, args...)
Base.getindex(what::UnconstrainedDraws, args...) = getindex(what.unconstrained_draws, args...)
Base.eachcol(what::UnconstrainedDraws) = eachcol(what.unconstrained_draws)
Base.eachrow(what::UnconstrainedDraws) = eachrow(what.unconstrained_draws)
Base.reshape(what::UnconstrainedDraws, args...) = reshape(what.unconstrained_draws, args...)
Base.reshape(what::UnconstrainedDraws, shape::Tuple{Int64, Vararg{Int64, N}})  where N = reshape(
    what.unconstrained_draws, shape
)
subset(what::UnconstrainedDraws, idxs::AbstractVector) = DynamicObjects.update(
    what,
    unconstrained_draws=what[idxs, :]
)
no_draws(what::UnconstrainedDraws) = size(what, 1)
no_parameters(what::UnconstrainedDraws) = size(what, 2)
subset(what::UnconstrainedDraws, no_draws=size(what, 1) ÷ 2) = subset(what, rand(axes(what, 1), no_draws))
thinned(what::UnconstrainedDraws, n=10) = update(what, unconstrained_draws=what[1:n:end, :])
unconstrained_draw(what::UnconstrainedDraws, idx::Integer) = UnconstrainedDraw(what.info, what[idx, :])
eachdraw(what::UnconstrainedDraws) = UnconstrainedDraw.([what.info], eachrow(what))



xcs(info::DistributionInfo, ::Any, draws::AbstractMatrix) = draws[:, info.xcs_idxs]
means(info::DistributionInfo, ::Any, draws::AbstractMatrix) = draws[:, info.means_idxs]
log_sds(info::DistributionInfo, ::Any, draws::AbstractMatrix) = draws[:, info.log_sds_idxs]
xcs(info::DistributionInfo, ::Any, draw::AbstractVector) = draw[info.xcs_idxs]
means(info::DistributionInfo, ::Any, draw::AbstractVector) = draw[info.means_idxs]
log_sds(info::DistributionInfo, ::Any, draw::AbstractVector) = draw[info.log_sds_idxs]
# xcs(info, distribution, draws) = xcs(distribution, draws)
# means(info, distribution, draws) = means(distribution, draws)
# log_sds(info, distribution, draws) = log_sds(distribution, draws)

# distribution_info(what) = what.distribution.info
xcs(what) = xcs(what.distribution.info, unpack(what)...)
means(what) = means(what.distribution.info, unpack(what)...)
log_sds(what) = log_sds(what.distribution.info, unpack(what)...)
sds(what) = exp.(what.log_sds)
# centeredness(what) = what.distribution.centeredness
x1s(what::AbstractVector) = to_x1.(what.xcs, what.means, what.log_sds, what.distribution.info.centeredness)
x1s(what::AbstractMatrix) = to_x1.(what.xcs, what.means, what.log_sds, what.distribution.info.centeredness')

function replace_xcs(distribution, unconstrained_draws::AbstractMatrix, xcs::AbstractMatrix)
    unconstrained_draws = 1 .* unconstrained_draws
    unconstrained_draws[:, distribution.info.xcs_idxs] .= xcs
    unconstrained_draws
end
replace_xcs(what::UnconstrainedDraws, xcs) = update(what, unconstrained_draws=replace_xcs(
    unpack(what)..., xcs
))
# function ChainRulesCore.rrule(::typeof(replace_xcs), unconstrained_draws::AbstractMatrix, xcs_idxs, xcs::AbstractMatrix)
#     rv = replace_xcs(unconstrained_draws, xcs_idxs, xcs)
#     mul = ones((1, size(rv, 2)))
#     mul[:, xcs_idxs] .= 0
#     function replace_xcs_pullback(rva) 
#         (NoTangent(), rva .* mul, NoTangent(), rva[:, xcs_idxs])
#     end
#     return rv, replace_xcs_pullback
# end
# ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws::AbstractMatrix, xcs_idxs, xcs::AbstractMatrix{ReverseDiff.TrackedReal{Float64, Float64, Nothing}})

"""
A base type whose instances contain a single unconstrained draw.
"""
@dynamic_type AbstractUnconstrainedDraw <: AbstractVector{Real}

"""
A type whose instances contain a single unconstrained draw.
"""
@dynamic_object UnconstrainedDraw <: AbstractUnconstrainedDraw distribution::DynamicDistribution unconstrained_draw::AbstractVector
pack(distribution::DynamicDistribution, unconstrained_draw::AbstractVector) =  UnconstrainedDraw(
    distribution, unconstrained_draw
)
Base.size(what::UnconstrainedDraw, args...) = size(what.unconstrained_draw, args...)
Base.axes(what::UnconstrainedDraw, args...) = axes(what.unconstrained_draw, args...)
Base.getindex(what::UnconstrainedDraw, args...) = getindex(what.unconstrained_draw, args...)
Base.collect(what::UnconstrainedDraw) = collect(what.unconstrained_draw)

function replace_xcs(unconstrained_draw::AbstractVector, xcs_idxs, xcs::AbstractVector)
    unconstrained_draw = 1 .* unconstrained_draw
    unconstrained_draw[xcs_idxs] .= xcs
    unconstrained_draw
end
replace_xcs(what::UnconstrainedDraw, xcs) = DynamicObjects.update(
    what, unconstrained_draw=replace_xcs(
        what.unconstrained_draw, what.distribution.info.xcs_idxs, xcs
    )
)
function ChainRulesCore.rrule(::typeof(replace_xcs), unconstrained_draw::AbstractVector, xcs_idxs, xcs::AbstractVector)
    rv = replace_xcs(unconstrained_draw, xcs_idxs, xcs)
    function replace_xcs_pullback(rva) 
        mul = ones(length(rv))
        mul[xcs_idxs] .= 0
        (NoTangent(), rva .* mul, NoTangent(), rva[xcs_idxs])
    end
    return rv, replace_xcs_pullback
end
ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws::AbstractVector, xcs_idxs, xcs::AbstractVector{ReverseDiff.TrackedReal{Float64, Float64, Nothing}})
ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws::AbstractVector, xcs_idxs, xcs::ReverseDiff.TrackedArray)
ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws::ReverseDiff.TrackedArray, xcs_idxs, xcs::ReverseDiff.TrackedArray)


"""
A wrapper around a BridgeStan model.
"""
ENV["STAN_THREADS"] = "true"
@dynamic_object BSDistribution <: DynamicDistribution model::StanModel
DynamicDistribution{:BSDistribution}(stan_file, data_string; kwargs...) = BSDistribution(
    StanModel(
        stan_file=stan_file, 
        data=data_string
    ), 
    stan_file=stan_file, data_string=data_string;
    kwargs...
)
remake(what::BSDistribution) = update(what, model=StanModel(
    stan_file=what.stan_file, data=what.data_string
))

DynamicObjects.persistent_hash(what::StanModel, h) = hash((name(what), what.data), h)
no_dimensions(what::BSDistribution) = param_unc_num(what.model)
function bslogpdf(what::BSDistribution, parameters::Vector{Float64}) 
    if !all(isfinite.(parameters))
        println_finite_problem("Parameter problem: ", parameters)
        # throw(DomainError(parameters))
    end
    try
        log_density(what.model, parameters)
    catch
        -Inf
    end
end
bslogpdf(what::BSDistribution, parameters::AbstractVector{Float64}) = bslogpdf(what, collect(parameters))
bslogpdf(what::BSDistribution, parameters::UnconstrainedDraw) = bslogpdf(what, parameters.unconstrained_draw)
# bslogpdf(what::BSDistribution, parameters::UnconstrainedDraw) = bslogpdf(what, parameters.unconstrained_draw)
Distributions.logpdf(what::BSDistribution, parameters) = bslogpdf(what, parameters)

# bslogpdf(what::BSDistribution, parameters::AbstractVector) = log_density(what.model, collect(parameters))
# Distributions.logpdf(what::BSDistribution, parameters::AbstractVector) = logpdf(what, collect(parameters))
function ChainRulesCore.rrule(::typeof(bslogpdf), what, parameters)
    if !all(isfinite.(parameters))
        println_finite_problem("Parameter problem: ", parameters)
        # throw(DomainError(parameters))
    end
    rv = try
        log_density_gradient(what.model, parameters)
    catch e
        println("Gradient error ($e): ", parameters)
        -Inf, fill(-Inf, size(parameters))
    end
    if !all(isfinite.(rv[2]))
        println_finite_problem("Gradient problem: ", rv[2])
        # throw(DomainError(parameters))
        rv = -Inf, fill(-Inf, size(parameters))
    end
    return rv[1], rva -> (NoTangent(), NoTangent(), rva .* rv[2])
end
ReverseDiff.@grad_from_chainrules bslogpdf(what, parameters::ReverseDiff.TrackedArray)
ReverseDiff.@grad_from_chainrules bslogpdf(what, parameters::Vector{ReverseDiff.TrackedReal{Float64, Float64, Nothing}})

"""
Reparametrizations.
"""
@dynamic_type Reparametrization
reparametrize(distribution, what::Reparametrization, parameters) = reparametrize(
    what, pack(distribution, parameters)
)
unreparametrize(distribution, what::Reparametrization, parameters) = reparametrize(
    distribution, what.inverse, parameters
)

"""
A reparametrized distribution.
"""
@dynamic_object ReparametrizedDistribution <: DynamicDistribution distribution::DynamicDistribution reparametrization::Reparametrization
pack(distribution::DynamicDistribution, reparametrization::Reparametrization) = ReparametrizedDistribution(
    distribution, reparametrization
)
no_dimensions(what::ReparametrizedDistribution) = what.distribution.no_dimensions
function Distributions.logpdf(what::ReparametrizedDistribution, reparameters) 
    if !all(isfinite.(collect(reparameters)))
        println_finite_problem("Reparameter problem: ", reparameters)
        # println("Reparameter problem: ", (1:length(reparameters))[(x->!isfinite(x)).(collect(reparameters))])
        # throw(DomainError(reparameters))
    end
    parameters = unreparametrize(what, reparameters)
    if !all(isfinite.(collect(parameters)))
        println_finite_problem("Reparameter problem: ", reparameters)
        println_finite_problem("Parameter problem: ", parameters)
        # println("Parameter problem: ", (1:length(parameters))[(x->!isfinite(x)).(collect(parameters))])
        # throw(DomainError(parameters))
    end
    (
        logpdf(what.distribution, parameters) 
        + log_jacobian(what, parameters, reparameters)
    )
end
unreparametrize(::DynamicDistribution, reparameters) = reparameters
unreparametrize(what::ReparametrizedDistribution, reparameters) = unreparametrize(
    unpack(what)..., reparameters,
)
log_jacobian(what::ReparametrizedDistribution, parameters, reparameters) = log_jacobian(
    unpack(what)..., parameters, reparameters
)
log_jacobian(distribution::DynamicDistribution, reparametrization::Reparametrization, parameters, reparameters) = log_jacobian(
    reparametrization, parameters, reparameters
)
# reparametrize(what::DynamicDistribution, parameters) = parameters
# reparametrize(what::ReparametrizedDistribution, parameters) = reparametrize(
#     what.reparametrization, parameters,
# )
info(what::ReparametrizedDistribution) = info(unpack(what)...)
info(distribution, ::Reparametrization) = distribution.info



# reparametrized(how::Reparametrization, what::DynamicDistribution) = ReparametrizedDistribution(
#     what, how
# )
# log_jacobian(how::Reparametrization, parameters) = log_jacobian(
#     how, parameters, reparametrize(how, parameters)
# )

"""
Chained reparametrizations
"""
@dynamic_object ChainedReparametrization <: Reparametrization left right
Base.:*(left::Reparametrization, right::Reparametrization) = ChainedReparametrization(left, right)
Base.:*(left::Reparametrization, right::ChainedReparametrization) = ChainedReparametrization(
    left*right.left, right.right
)
reparametrize(what::ChainedReparametrization, parameters) = reparametrize(
    what.left, reparametrize(what.right, parameters)
)
inverse(what::ChainedReparametrization) = ChainedReparametrization(what.right.inverse, what.left.inverse)
log_jacobian(what::ChainedReparametrization, parameters, reparameters) = (
    rv = 0;
    left_parameters = unreparametrize(what.left, reparameters);
    rv += log_jacobian(what.left, left_parameters, reparameters);
    rv + log_jacobian(what.right, parameters, left_parameters)
)

"""
Continuous Noncentering
"""
@dynamic_object ContinuousNoncentering <: Reparametrization target_centeredness previous_centeredness
info(distribution, reparametrization::ContinuousNoncentering) = distribution.info |> update(
    centeredness=reparametrization.target_centeredness
)
# pad_left(what, ::AbstractVector) = what
# pad_left(what, ::AbstractMatrix) = what'

safe_exp(x) = (ex = exp(x); isfinite(ex) ? ex : zero(ex))
to_xc2(previous_xc, mean, log_sd, target_centeredness, previous_centeredness) = (
    mean * target_centeredness 
    + safe_exp(
        log_sd * (target_centeredness - previous_centeredness)
    ) * (previous_xc - mean * previous_centeredness)
)
reparametrize(what::ContinuousNoncentering, parameters::UnconstrainedDraws) = (
    replace_xcs(
        parameters,
        to_xc2.(
            parameters.xcs, parameters.means, parameters.log_sds, 
            what.target_centeredness', 
            what.previous_centeredness'
        )
    ) |> update(distribution=ReparametrizedDistribution(parameters.distribution, what))
)
reparametrize(what::ContinuousNoncentering, parameters::UnconstrainedDraw) = (
    replace_xcs(
        parameters,
        to_xc2.(
            parameters.xcs, parameters.means, parameters.log_sds, 
            what.target_centeredness, 
            what.previous_centeredness
        )
    ) |> update(distribution=ReparametrizedDistribution(parameters.distribution, what))
)
inverse(what::ContinuousNoncentering) = ContinuousNoncentering(
    what.previous_centeredness, what.target_centeredness
)
log_jacobians(what::ContinuousNoncentering, parameters, reparameters) = (
    parameters.log_sds .* (what.previous_centeredness .- what.target_centeredness)
)
log_jacobian(what::ContinuousNoncentering, parameters, reparameters) = sum(
    log_jacobians(what, parameters, reparameters)
)
recenter(what::UnconstrainedDraws, centeredness=what.best_centeredness) = reparametrize(
    ContinuousNoncentering(centeredness, what.distribution.info.centeredness), what
)
recenter(what::DynamicDistribution, centeredness) = ReparametrizedDistribution(
    what, ContinuousNoncentering(centeredness, what.info.centeredness)
)
# recenter(what::DynamicDistribution, centeredness) = ReparametrizedDistribution(
#     what, 
#     ContinuousNoncentering(centeredness, what.wrapper, previous_centeredness=what.centeredness)
# )

"""
Parameterwise rescaling
"""
@dynamic_object Rescaling <: Reparametrization scale::AbstractVector
reparametrize(what::Rescaling, parameters) = pad_left(what.scale, parameters) .\ parameters
inverse(what::Rescaling) = Rescaling(1 ./ what.scale)
# unreparametrize(what::Rescaling, reparameters) = pad_left(what.scale, reparameters) .* reparameters
log_jacobian(what::Rescaling, ::Any, ::Any) = 0
Base.:*(lhs::Rescaling, rhs::Rescaling) = Rescaling(lhs.scale .* rhs.scale)
rescale(what::DynamicDistribution, scale) = ReparametrizedDistribution(what, Rescaling(scale))
rescale(what::ReparametrizedDistribution, scale) = ReparametrizedDistribution(
    what.distribution,
    Rescaling(scale) * what.reparametrization
)

"""
A posteriorDB posterior.
"""
@dynamic_object PDBPosterior name::String

database(::PDBPosterior) = PosteriorDB.database()
posterior(what::PDBPosterior) = PosteriorDB.posterior(what.database, what.name)
model(what::PDBPosterior) = PosteriorDB.model(what.posterior)
dataset(what::PDBPosterior) = PosteriorDB.dataset(what.posterior)
data_string(what::PDBPosterior) = PosteriorDB.load(what.dataset, String)
reference_posterior(what::PDBPosterior) = PosteriorDB.reference_posterior(what.posterior)
constrained_df(what::PDBPosterior) = DataFrame(PosteriorDB.load(what.reference_posterior))
constrained_draws(what::PDBPosterior) = hcat([vcat(col...) for col in eachcol(what.constrained_df)]...)
stan_file(what::PDBPosterior) = PosteriorDB.path(PosteriorDB.implementation(what.model, "stan"))
bridgestan_model(what::PDBPosterior) = StanModel(stan_file=what.stan_file, data=what.data_string)
data_dict(what::PDBPosterior) = Dict(PosteriorDB.load(what.dataset))
info(what::PDBPosterior) = what.info_wrapper(what.data_dict)
bridgestan_distribution(what::PDBPosterior) = BSDistribution(
    what.bridgestan_model,
    info=what.info
)
# info_wrapper(what::PDBPosterior) = DummyDistributionInfo

# unconstrained_draws(what::PDBPosterior) = (
#     info = what.info;
#     UnconstrainedDraws(
#         info,
#         vcat([
#             param_unconstrain(info.distribution.model, collect(row))' 
#             for row in eachrow(what.constrained_draws)
#         ]...),
#     )
# )
# update_data(what::PDBPosterior, update) = (
#     dd = merge(what.data_dict, update);
#     DynamicObjects.update(
#         what,
#         data_dict=dd,
#         data_string=json(dd)
#     )
# )

# no_xcs(what::DistributionInfo) = length(what.xcs_idxs)
# logpdf_gradient(what::DistributionInfo, parameters) = ReverseDiff.gradient(
#     x->logpdf(what.distribution, UnconstrainedDraw(what, x)), parameters
# )

# @dynamic_object DummyDistributionInfo <: DistributionInfo data::Dict


# ReverseDiff.value(what::UnconstrainedDraws) = DynamicObjects.update(
#     what, unconstrained_draws=ReverseDiff.value(what.unconstrained_draws)
# )
# Base.similar(what::UnconstrainedDraws, D::Type{T}) where {T} = DynamicObjects.update(
#     what, unconstrained_draws=similar(what.unconstrained_draws, D)
# )
# Base.fill!(what::UnconstrainedDraws, value) = fill!(what.unconstrained_draws, value)
# Base.eltype(what::UnconstrainedDraws) = eltype(what.unconstrained_draws)
# ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws, xcs_idxs, xcs::ReverseDiff.TrackedArray)
# ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws::ReverseDiff.TrackedArray, xcs_idxs, xcs)
# ReverseDiff.@grad_from_chainrules replace_xcs(unconstrained_draws::ReverseDiff.TrackedArray, xcs_idxs, xcs::ReverseDiff.TrackedArray)

# scatter_funnels(what::UnconstrainedDraws) = Figure([
#     scatter_funnel(what, i, j)
#     for i in I, j in J
# ], plot_width=200, extra_figure_kwargs=(
#     margin=0mm,
#     xaxis=false, yaxis=false, xticks=false, yticks=false, 
#     markerstrokewidth=0,
#     plot_title="$(I) vs $(J)"
# ))  

klp2(previous_xcs, means, log_sds, target_centeredness, previous_centeredness) = (
    -mean(log_sds .* target_centeredness)
    +log(std(to_xc2.(previous_xcs, means, log_sds, target_centeredness, previous_centeredness)))
)
best_centeredness(what::UnconstrainedDraws, loss=klp2, cs=LinRange(0, 1, 100)) = best_centeredness.(
    eachcol(what.xcs), eachcol(what.means), eachcol(what.log_sds), 
    loss, [cs], what.distribution.info.centeredness
)
# best_centeredness(what::UnconstrainedDraws, i, loss, cs) = best_centeredness(
#     what.x1s, what.means, what.log_sds, i, loss, cs
# )
best_centeredness(previous_xcs, means, log_sds, loss, cs, previous_centeredness) = cs[argmin(
    centeredness_losses(previous_xcs, means, log_sds, loss, cs, previous_centeredness)
)]
centeredness_losses(previous_xcs, means, log_sds, loss, cs, previous_centeredness) = [
    loss(
        previous_xcs, 
        means, 
        log_sds, 
        c,
        previous_centeredness
    ) for c in cs
]
# klp2(x1s, means, log_sds, centeredness) = (
#     -sum(log_sds .* centeredness) + 
#     log(std(to_xc.(x1s, means, log_sds, centeredness)))
# )
centeredness_losses(what::UnconstrainedDraws, loss=klp2, cs=LinRange(0, 1, 100)) = hcat(
    to_unit_range.(centeredness_losses.(
        eachcol(what.xcs), eachcol(what.means), eachcol(what.log_sds), 
        loss, [cs], what.distribution.info.centeredness
    ))...
)
best_centeredness_convergence(what::UnconstrainedDraws) = what.no_draws < 100 ? hcat(what.best_centeredness) : hcat(
    thinned(what, 2).best_centeredness_convergence, what.best_centeredness
)
best_centeredness_convergence_plot(what::UnconstrainedDraws) = (
    bcc = what.best_centeredness_convergence;
    Line(
        1:size(bcc, 2), bcc'# .+ reshape(1:size(bcc, 1), (1, :))
    )
)

# function naive_loss(x1s, means, log_sds, centeredness)
#     xcs = to_xc.(x1s, means, log_sds, centeredness)
#     # xc_snormal = Normal(mean(xcs), std(xcs));
#     # mean(-log_sds .* centeredness - logpdf.(xc_snormal, xcs))
#     mean(-log_sds .* centeredness) + log(std(xcs))
# end
# naive_best_centeredness(what::UnconstrainedDraws) = best_centeredness(what, naive_loss)

# scalar_loss(what::UnconstrainedDraws, centeredness, loss=klp) = sum(
#     loss.(eachcol(what.x1s), eachcol(what.means), eachcol(what.log_sds), centeredness)
# )
# scalar_losses(what::UnconstrainedDraws, centeredness, loss=klp) = loss.(
#     eachcol(what.x1s), eachcol(what.means), eachcol(what.log_sds), centeredness
# )

# parameter_mean(what::UnconstrainedDraws) = (mean(what, dims=1))
# parameter_var(what::UnconstrainedDraws) = (var(what, dims=1))
# gradient_mean(what::UnconstrainedDraws) = (mean(what.gradients, dims=1))
# gradient_var(what::UnconstrainedDraws) = (var(what.gradients, dims=1))
# fisher_var(what::UnconstrainedDraws) = sqrt.(what.parameter_var ./ what.gradient_var)
# fisher_mean(what::UnconstrainedDraws) = (
#     what.fisher_var .* what.gradient_mean .+ what.parameter_mean
# )
# parameter_normal(what::UnconstrainedDraws) = MvNormal(
#     vec(what.parameter_mean), Diagonal(vec(what.parameter_var))
# )
# fisher_normal(what::UnconstrainedDraws) = MvNormal(
#     vec(what.fisher_mean), Diagonal(vec(what.fisher_var))
# )
# gradient_update(what, centeredness) = hasproperty(what, :gradients) ? 
#     xcs_gradient_update(what, centeredness) :
#     DynamicObjects.update(recenter(what, centeredness), :gradients)
# xcs_gradient_update(what::UnconstrainedDraws, centeredness) = (
#     rewhat = DynamicObjects.update(
#         recenter(what, centeredness), gradients=1 .* what.gradients 
#     );
#     rewhat.gradients[:, what.info.xcs_idxs] .*= exp.(-what.log_sds .* centeredness');
#     rewhat
# )
# fisher_scalar_losses(what::UnconstrainedDraws, centeredness) = (
#     how = ContinuousNoncentering(centeredness, what.info.centeredness);
#     rewhat = gradient_update(what, centeredness);
#     mean(
#         +log_jacobian.([how], eachdraw(what), eachdraw(rewhat)) 
#         -logpdf.([rewhat.fisher_normal], eachrow(rewhat))
#     )
# ) 
# fisher_scalar_loss(what::UnconstrainedDraws, centeredness) = (
#     how = ContinuousNoncentering(centeredness, what.info.centeredness);
#     rewhat = gradient_update(what, centeredness);
#     mean(
#         +log_jacobian.([how], eachdraw(what), eachdraw(rewhat)) 
#         -logpdf.([rewhat.fisher_normal], eachrow(rewhat))
#     )
# ) 


# normal_gradients(what::UnconstrainedDraws) = -(what .- what.parameter_mean) ./ what.parameter_var
# fisher_normal_gradients(what::UnconstrainedDraws) = -(what .- what.fisher_mean) ./ what.fisher_var
# gradients(what::UnconstrainedDraws) = hcat(logpdf_gradient.([what.info], eachrow(what))...)'
# reparametrized(how::Reparametrization, what::DistributionInfo) = DynamicObjects.update(
#     what,
#     distribution=reparametrized(how, what.distribution)
# )
# reparametrized(how::ContinuousNoncentering, what::DistributionInfo) = DynamicObjects.update(
#     what,
#     distribution=reparametrized(how, what.distribution),
#     centeredness=how.target_centeredness
# )
# reparametrized(how::Reparametrization, what::UnconstrainedDraws) = DynamicObjects.update(
#     what, 
#     info=reparametrized(how, what.info), 
#     unconstrained_draws=reparametrize(how, what)
# )


# recenter(what::UnconstrainedDraws, centeredness=what.best_centeredness) = reparametrized(
#     ContinuousNoncentering(centeredness, what.info.centeredness), what
# )
# descale(what::UnconstrainedDraws) = reparametrized(Rescaling(sqrt.(what.parameter_var)), what)

# square(x) = x^2
# gradient_loss(what::UnconstrainedDraws) = mean(square.(what.gradients - what.normal_gradients) .* what.parameter_var)
# gradient_loss(what::UnconstrainedDraws, centeredness) = gradient_update(what, centeredness).gradient_loss
# all_gradient_losses(what::UnconstrainedDraws) = square.(what.gradients - what.normal_gradients) .* (what.parameter_var)
# gradient_losses(what::UnconstrainedDraws) = mean(what.all_gradient_losses, dims=1)[1, what.info.xcs_idxs]
# gradient_losses(what::UnconstrainedDraws, centeredness) = gradient_update(what, centeredness).gradient_losses
# # DynamicObjects.update(
# #     recenter(what, centeredness), :gradients
# # ).gradient_loss
# all_fisher_gradient_losses(what::UnconstrainedDraws) = square.(what.gradients - what.fisher_normal_gradients) .* what.fisher_var
# fisher_gradient_losses(what::UnconstrainedDraws) = mean(what.all_fisher_gradient_losses, dims=1)[1, what.info.xcs_idxs]
# fisher_gradient_losses(what::UnconstrainedDraws, centeredness) = gradient_update(what, centeredness).fisher_gradient_losses
# fisher_gradient_loss(what::UnconstrainedDraws) = mean(square.(what.gradients - what.fisher_normal_gradients) .* what.fisher_var)
# fisher_gradient_loss(what::UnconstrainedDraws, centeredness) = gradient_update(what, centeredness).fisher_gradient_loss

# wasserstein_loss(what::AbstractVector) = mean((
#     sort((what .- mean(what)) ./ std(what)) - quantile.(
#         Normal(0, 1), range(0, 1, 2+length(what))[2:end-1]
#     )
# ).^2)
# wasserstein_losses(what::UnconstrainedDraws) = wasserstein_loss.(eachcol(what))
# wasserstein_losses(what::UnconstrainedDraws, centeredness) = gradient_update(what, centeredness).wasserstein_losses
# cdf_loss(what::AbstractVector) = mean((
#     cdf.(Normal(mean(what), std(what)), sort(what)) 
#     - range(0, 1, 2+length(what))[2:end-1]
# ).^2)
# cdf_losses(what::UnconstrainedDraws) = cdf_loss.(eachcol(what))
# cdf_losses(what::UnconstrainedDraws, centeredness) = gradient_update(what, centeredness).cdf_losses


# var_sqrt(what::UnconstrainedDraws) = sqrt(Diagonal(vec(what.parameter_var)))
# cov_sqrt(what::UnconstrainedDraws) = sqrt(cov(what.unconstrained_draws))
# pca_sqrt(what::UnconstrainedDraws) = (
#     e = eigen(cov(what.unconstrained_draws));
#     e.vectors * Diagonal(sqrt.(e.values))
# )
# descaled(what::UnconstrainedDraws) = DynamicObjects.update(
#     what, unconstrained_draws=what.unconstrained_draws / what.var_sqrt 
# )
# whitened(what::UnconstrainedDraws) = DynamicObjects.update(
#     what, unconstrained_draws=what.unconstrained_draws / what.cov_sqrt' 
# )
# pcad(what::UnconstrainedDraws) = DynamicObjects.update(
#     what, unconstrained_draws=what.unconstrained_draws / what.pca_sqrt' 
# ) 



# function ChainRulesCore.rrule(::typeof(replace_xcs), what::UnconstrainedDraw, parameters)
#     rv = replace_xcs(what, parameters)
#     return rv, rva -> (NoTangent())
# end

# unconstrained_draw(what::AbstractVector) = what
# unconstrained_draw(what::UnconstrainedDraw) = what.unconstrained_draw


"""
parameters {
  real theta[J]; // treatment effect in school j
  real mu; // hyper-parameter of mean
  real<lower=0> tau; // hyper-parameter of sdv
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/eight_schools_centered.stan
"""
@dynamic_object EightSchoolsInfo <: DistributionInfo data::Dict
xcs_idxs(what::EightSchoolsInfo) = 1:what["J"]
means_idxs(what::EightSchoolsInfo) = what["J"]+1
log_sds_idxs(what::EightSchoolsInfo) = what["J"]+2
centeredness(what::EightSchoolsInfo) = 1

"""
sds = alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
"""
@dynamic_object HSGPInfo <: DistributionInfo no_basis_functions::Integer L=1.5
indices(what::HSGPInfo) = 1:what.no_basis_functions
xcs_idxs(what::HSGPInfo) = 3:(2+what.no_basis_functions)
means(what::HSGPInfo, distribution, ::AbstractMatrix) = zeros(what.no_basis_functions)
means(what::HSGPInfo, distribution, ::AbstractVector) = zeros(what.no_basis_functions)
# https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
# alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
function log_sds(what::HSGPInfo, ::Any, draws::AbstractMatrix)
    log_sd = draws[:, 1]
    log_lengthscale = draws[:, 2]
    lengthscale = exp.(log_lengthscale)
    (
        log_sd
        .+ log.(sqrt.(sqrt(2*pi)))
        .+ .5 .* log_lengthscale
        .- .25 .* (lengthscale.*(pi/2/what.L)).^2 .* what.indices'.^2
    )
end
log_sds(what::HSGPInfo, ::Any, draw::AbstractVector) = hsgp_log_sds(unpack(what)..., draw[1], draw[2])
function hsgp_log_sds(no_basis_functions, L, log_sd, log_lengthscale)
    lengthscale = exp.(log_lengthscale)
    (
        log_sd
        + log(sqrt(sqrt(2*pi)))
        + .5 * log_lengthscale
        .- .25 * (lengthscale*(pi/2/L))^2 .* (1:no_basis_functions).^2
    )
end
function ChainRulesCore.rrule(::typeof(hsgp_log_sds), no_basis_functions, L, log_sd, log_lengthscale)
    rv = hsgp_log_sds(no_basis_functions, L, log_sd, log_lengthscale)
    function hsgp_log_sds_pullback(rva) 
        log_sda = sum(rva)
        lengthscale = exp(log_lengthscale)
        log_lengthscalea = .5 * log_sda - .25 * 2 * (lengthscale*(pi/2/L))^2 * dot(
            rva, (1:no_basis_functions).^2
        )
        (NoTangent(), NoTangent(), NoTangent(), log_sda, log_lengthscalea)
    end
    return rv, hsgp_log_sds_pullback
end
ReverseDiff.@grad_from_chainrules hsgp_log_sds(
    no_basis_functions, L, log_sd::ReverseDiff.TrackedReal, log_lengthscale::ReverseDiff.TrackedReal
)
centeredness(what::HSGPInfo) = zeros(what.no_basis_functions)

"""
parameters {
  // temporary intercept for centered predictors
  real Intercept; //1
  // GP standard deviation parameters
  real<lower=0> sdgp_1;//2
  // GP length-scale parameters
  real<lower=0> lscale_1;//3
  // latent variables of the GP
  vector[NBgp_1] zgp_1;//4:3+NBgp_1
  // temporary intercept for centered predictors
  real Intercept_sigma;//4+NBgp_1
  // GP standard deviation parameters
  real<lower=0> sdgp_sigma_1;//5+NBgp_1
  // GP length-scale parameters
  real<lower=0> lscale_sigma_1;
  // latent variables of the GP
  vector[NBgp_sigma_1] zgp_sigma_1;
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/accel_gp.stan
"""
@dynamic_object MotorcycleInfo <: DistributionInfo data::Dict
hsgps_info(what::MotorcycleInfo) = [HSGPInfo(what["NBgp_1"]), HSGPInfo(what["NBgp_sigma_1"])]
hsgps_distributions(what::MotorcycleInfo) = DummyDistribution.(what.hsgps_info)
# hsgps_idx = [2:3+what["NBgp_1"], 5+what["NBgp_1"]:end]
hsgps(what::MotorcycleInfo, draws::AbstractMatrix) = [
    UnconstrainedDraws(what.hsgps_distributions[1], draws[:, 2:3+what["NBgp_1"]]),
    UnconstrainedDraws(what.hsgps_distributions[2], draws[:, 5+what["NBgp_1"]:end])
]
hsgps(what::MotorcycleInfo, draw::AbstractVector) = [
    UnconstrainedDraw(what.hsgps_distributions[1], draw[2:3+what["NBgp_1"]]),
    UnconstrainedDraw(what.hsgps_distributions[2], draw[5+what["NBgp_1"]:end])
]
means(::MotorcycleInfo, ::Any, ::AbstractMatrix) = zeros((1,1))
means(::MotorcycleInfo, ::Any, ::AbstractVector) = zeros((1,))
xcs_idxs(what::MotorcycleInfo) = vcat(4:3+what["NBgp_1"], 7+what["NBgp_1"]:6+what["NBgp_1"]+what["NBgp_sigma_1"])
log_sds(what::MotorcycleInfo, ::Any, draws::AbstractMatrix) = hcat(log_sds.(hsgps(what, draws))...)
log_sds(what::MotorcycleInfo, ::Any, draw::AbstractVector) = vcat(log_sds.(hsgps(what, draw))...)
# scalar_centeredness(what::MotorcycleInfo) = 1
centeredness(what::MotorcycleInfo) = vcat(
    ones(what["NBgp_1"]),
    zeros(what["NBgp_sigma_1"])
)

"""
parameters {
  vector[J] alpha;
  vector[2] beta;
  real mu_alpha;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_y;
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_hierarchical_intercept_centered.stan
"""
@dynamic_object RadonICInfo <: DistributionInfo data::Dict
xcs_idxs(what::RadonICInfo) = 1:what["J"]
means_idxs(what::RadonICInfo) = what["J"]+3
log_sds_idxs(what::RadonICInfo) = what["J"]+4
centeredness(what::RadonICInfo) = 1

"""
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  vector[J] alpha;
  vector[J] beta;
  real mu_alpha;
  real mu_beta;
}

https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_variable_intercept_slope_centered.stan
"""
@dynamic_object RadonISCInfo <: DistributionInfo data::Dict
xcs_idxs(what::RadonISCInfo) = 4:(3+2*what["J"])
means_idxs(what::RadonISCInfo) = vcat(fill(2*what["J"]+4, what["J"]), fill(2*what["J"]+5, what["J"]))
log_sds_idxs(what::RadonISCInfo) = vcat(fill(2, what["J"]), fill(3, what["J"]))
centeredness(what::RadonISCInfo) = 1


# """
# The unconstrained parameters of an HSGP:

# sds = alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
# https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
# """
# @dynamic_object HSGPDraws <: UnconstrainedDraws unconstrained_draws::AbstractMatrix L=1.5
# no_basis_functions(what::HSGPDraws) = size(what, 2)-2
# indices(what::HSGPDraws) = 1:what.no_basis_functions
# xcs_idxs(what::HSGPDraws) = 3:size(what, 2)
# means(what::HSGPDraws) = 0 .* what.log_sds
# # https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
# # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
# function log_sds(what::HSGPDraws)
#     log_sd = what[:, 1]
#     log_lengthscale = what[:, 2]
#     lengthscale = exp.(log_lengthscale)
#     (
#         log_sd
#         .+ log.(sqrt.(sqrt(2*pi)))
#         .+ .5 .* log_lengthscale
#         .- .25 .* (lengthscale.*(pi/2/what.L)).^2 .* what.indices'.^2
#     )
# end

# @dynamic_object HSGPDraw <: UnconstrainedDraw unconstrained_draw::AbstractVector L=1.5
# no_basis_functions(what::HSGPDraw) = size(what, 1)-2
# indices(what::HSGPDraw) = 1:what.no_basis_functions
# xcs_idxs(what::HSGPDraw) = 3:size(what, 1)
# means(what::HSGPDraw) = fill(0, size(what.xcs_idxs))
# # https://github.com/avehtari/casestudies/blob/master/Birthdays/gpbasisfun_functions.stan
# # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)
# @views function log_sds(what::HSGPDraw)
#     log_sd = what[1]
#     log_lengthscale = what[2]
#     lengthscale = exp.(log_lengthscale)
#     (
#         log_sd
#         + log(sqrt(sqrt(2*pi)))
#         + .5 * log_lengthscale
#         .- .25 * (lengthscale*(pi/2/what.L))^2 .* what.indices.^2
#     )
# end

# """
# A matrix of unconstrained draws from the MotorcycleGP model:

# parameters {
#   // temporary intercept for centered predictors
#   real Intercept;
#   // GP standard deviation parameters
#   real<lower=0> sdgp_1;
#   // GP length-scale parameters
#   real<lower=0> lscale_1;
#   // latent variables of the GP
#   vector[NBgp_1] zgp_1;
#   // temporary intercept for centered predictors
#   real Intercept_sigma;
#   // GP standard deviation parameters
#   real<lower=0> sdgp_sigma_1;
#   // GP length-scale parameters
#   real<lower=0> lscale_sigma_1;
#   // latent variables of the GP
#   vector[NBgp_sigma_1] zgp_sigma_1;
# }

# https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/accel_gp.stan
# """
# @dynamic_object MotorcycleDraws <: UnconstrainedDraws unconstrained_draws::AbstractMatrix
# hsgps_draws(what::MotorcycleDraws) = [
#     HSGPDraws(what[:, 2:43]),
#     HSGPDraws(what[:, 45:end])
# ]
# log_sds(what::MotorcycleDraws) = hcat(log_sds.(what.hsgps_draws)...)
# means(what::MotorcycleDraws) = 0 .* what.log_sds
# xcs_idxs(what::MotorcycleDraws) = (
#     hsgps_draws = what.hsgps_draws;
#     vcat(1 .+ hsgps_draws[1].xcs_idxs, 44 .+ hsgps_draws[2].xcs_idxs)
# )
# centeredness(what::MotorcycleDraws) = 0

# @dynamic_object MotorcycleDraw <: UnconstrainedDraw unconstrained_draw::AbstractVector
# @views hsgps_draw(what::MotorcycleDraw) = [
#     HSGPDraw(what[2:43]),
#     HSGPDraw(what[45:end])
# ]
# log_sds(what::MotorcycleDraw) = vcat(log_sds.(what.hsgps_draw)...)
# means(what::MotorcycleDraw) = fill(0, size(what.xcs_idxs))
# xcs_idxs(what::MotorcycleDraw) = (
#     hsgps_draw = what.hsgps_draw;
#     vcat(1 .+ hsgps_draw[1].xcs_idxs, 44 .+ hsgps_draw[2].xcs_idxs)
# )
# centeredness(what::MotorcycleDraw) = 0

# motorcycle_wrapper(what::AbstractVector) = MotorcycleDraw(what)
# motorcycle_wrapper(what::AbstractMatrix) = MotorcycleDraws(what)
# motorcycle_wrapper(what::UnconstrainedDraw) = MotorcycleDraw(what.unconstrained_draw)
# motorcycle_wrapper(what::UnconstrainedDraws) = MotorcycleDraws(what.unconstrained_draws)

# """
# A matrix of unconstrained draws from the centered radon_hierarchical_intercept_centered model:

# parameters {
#   vector[J] alpha;
#   vector[2] beta;
#   real mu_alpha;
#   real<lower=0> sigma_alpha;
#   real<lower=0> sigma_y;
# }

# https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_hierarchical_intercept_centered.stan
# """
# @dynamic_object RadonHierarchicalInterceptCenteredDraws <: UnconstrainedDraws unconstrained_draws::AbstractMatrix
# xcs_idxs(what::RadonHierarchicalInterceptCenteredDraws) = 1:size(what, 2)-5
# means(what::RadonHierarchicalInterceptCenteredDraws) = what[:, end-2]
# log_sds(what::RadonHierarchicalInterceptCenteredDraws) = what[:, end-1]

# @dynamic_object RadonHierarchicalInterceptCenteredDraw <: UnconstrainedDraw unconstrained_draw::AbstractVector
# xcs_idxs(what::RadonHierarchicalInterceptCenteredDraw) = 1:size(what, 1)-5
# means(what::RadonHierarchicalInterceptCenteredDraw) = what[end-2]
# log_sds(what::RadonHierarchicalInterceptCenteredDraw) = what[end-1]

# radon_wrapper(what::AbstractVector) = RadonHierarchicalInterceptCenteredDraw(what)
# radon_wrapper(what::AbstractMatrix) = RadonHierarchicalInterceptCenteredDraws(what)
# radon_wrapper(what::UnconstrainedDraw) = RadonHierarchicalInterceptCenteredDraw(what.unconstrained_draw)
# radon_wrapper(what::UnconstrainedDraws) = RadonHierarchicalInterceptCenteredDraws(what.unconstrained_draws)

# """
# A matrix of unconstrained draws from the centered radon_hierarchical_intercept_centered model:

# parameters {
#   real<lower=0> sigma_y;
#   real<lower=0> sigma_alpha;
#   real<lower=0> sigma_beta;
#   vector[J] alpha;
#   vector[J] beta;
#   real mu_alpha;
#   real mu_beta;
# }

# https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/radon_variable_intercept_slope_centered.stan
# """
# @dynamic_object RadonISCenteredDraws <: UnconstrainedDraws unconstrained_draws::AbstractMatrix
# xcs_idxs(what::RadonISCenteredDraws) = 4:size(what, 2)-2
# no_hps(what::RadonISCenteredDraws) = length(what.xcs_idxs) ÷ 2
# means(what::RadonISCenteredDraws) = hcat(
#     repeat(what[:, end-1], 1, what.no_hps),
#     repeat(what[:, end], 1, what.no_hps)
# )
# log_sds(what::RadonISCenteredDraws) = hcat(
#     repeat(what[:, 2], 1, what.no_hps),
#     repeat(what[:, 3], 1, what.no_hps)
# )

# @dynamic_object RadonISCenteredDraw <: UnconstrainedDraw unconstrained_draw::AbstractVector
# xcs_idxs(what::RadonISCenteredDraw) = 4:size(what, 1)-2
# no_hps(what::RadonISCenteredDraw) = length(what.xcs_idxs) ÷ 2
# means(what::RadonISCenteredDraw) = vcat(
#     fill(what[end-1], what.no_hps),
#     fill(what[end], what.no_hps)
# )
# log_sds(what::RadonISCenteredDraw) = vcat(
#     fill(what[2], what.no_hps),
#     fill(what[3], what.no_hps)
# )

# radon_is_wrapper(what::AbstractVector) = RadonISCenteredDraw(what)
# radon_is_wrapper(what::AbstractMatrix) = RadonISCenteredDraws(what)
# radon_is_wrapper(what::UnconstrainedDraw) = RadonISCenteredDraw(what.unconstrained_draw)
# radon_is_wrapper(what::UnconstrainedDraws) = RadonISCenteredDraws(what.unconstrained_draws)

function global_convergence(reference_chains::UnconstrainedDraws, draws::UnconstrainedDraws)
    reference_means = reference_chains.parameter_mean
    reference_cov = reference_chains.parameter_cov
    reference_L = cholesky(reference_cov).L
    estimates = cumsum(draws, dims=1) ./ reshape(1:size(draws, 1), (:, 1))
    deviations = estimates .- reference_means
    xi_deviations = deviations / reference_L'
    sqrt.(mean(xi_deviations .^ 2, dims=2))
end
global_convergence(distribution::DynamicDistribution, draws::UnconstrainedDraws) = global_convergence(
    distribution.cached_reference_chains, draws
)
# global_convergence(draws::UnconstrainedDraws) = global_convergence(draws.reference_distribution, draws)

cost(what::AbstractMatrix) = 1
no_chains(what::UnconstrainedDraws) = 1
chains(what::UnconstrainedDraws) = [
    update(what, unconstrained_draws=chain_draws, no_chains=1, all_stats=chain_stats) 
    for (chain_draws, chain_stats) in zip(
        eachslice(
            reshape(what.unconstrained_draws, (:, what.no_chains, size(what, 2))), 
            dims=2
        ), 
        eachcol(
            reshape(what.all_stats, (:, what.no_chains))
        )
    )
]
cost(what::UnconstrainedDraws) = hasproperty(what, :all_stats) ? sum(getproperty.(what.all_stats, :n_steps)) : 1
esss(what::UnconstrainedDraws) = ess(reshape(what.unconstrained_draws, (:, what.no_chains, size(what, 2))))
effs(what::UnconstrainedDraws) = what.esss ./ what.cost
sq_esss(what::UnconstrainedDraws) = ess(reshape(what.unconstrained_draws.^2, (:, what.no_chains, size(what, 2))))
sq_effs(what::UnconstrainedDraws) = what.sq_esss ./ what.cost
mv_effs(what::UnconstrainedDraws) = vcat(what.effs, what.sq_effs)
parameter_mean(what::UnconstrainedDraws) = mean(what, dims=1)
parameter_std(what::UnconstrainedDraws) = std(what, dims=1)
parameter_cov(what::UnconstrainedDraws) = cov(what)
sample_stats(what::UnconstrainedDraws) = what.all_stats[.!getproperty.(what.all_stats, :is_adapt)]
no_sampling_divergences(what::UnconstrainedDraws) = sum(getproperty.(
    what.sample_stats,
    :numerical_error
))
no_samples(what::UnconstrainedDraws) = sum(.!getproperty.(what.all_stats, :is_adapt))
sampling_divergence_ratio(what::UnconstrainedDraws) = what.no_sampling_divergences / what.no_samples

PairPlot(what::AbstractMatrix, i, j) = i < j ? Scatter(
    what[:, j], what[:, i], plot_kwargs=(alpha=size(what, 1)^-.125, )
) : (
    i == j ? Histogram(
        what[:, i], 
        plot_kwargs=(title=i, label=round(
            ess(reshape(what[:, i], (:, 1, 1)))[1] / cost(what), sigdigits=2
        ))
    ) : EmptyPlot())
PairPlots(what::AbstractMatrix, I, J=I) = Figure([
    PairPlot(what, i, j)
    for i in I, j in J
], plot_width=200, extra_figure_kwargs=(
    margin=0mm,
    xaxis=false, yaxis=false, xticks=false, yticks=false, 
    markerstrokewidth=0,
    plot_title="$(I) vs $(J)"
))  
PairPlots(what::AbstractMatrix, no_parameters::Integer=20) = PairPlots(
    what, sort(sample(axes(what, 2), min(no_parameters, size(what, 2)), replace=false))
)
DynamicPlots.plot_kwargs(::PlotSum) = (xaxis=false,)



SortedScatter(x,y,z; kwargs...) = (idxs=sortperm(z); Scatter(
    x[idxs], y[idxs], plot_kwargs=(
        alpha=size(x, 1)^-.125, marker_z=z[idxs], markerstrokewidth=0, 
        legend=:none, colorbar=:none, color=cgrad(:redsblues, rev=true), kwargs...
    )
))
scatter_funnel(what::UnconstrainedDraws, i=1; kwargs...) = Scatter(
    what.xcs[:, i], what.log_sds[:, min(i, size(what.log_sds, 2))],
    plot_kwargs=(
        alpha=size(what, 1)^-.125, markerstrokewidth=0, label="", 
        xaxis=false, yaxis=false, xticks=false, yticks=false, 
        kwargs...
    )
)
# scatter_funnel(what::UnconstrainedDraws, i, z; kwargs...) = (idxs=sortperm(z); Scatter(
#     what.xcs[idxs, i], what.log_sds[idxs, min(i, size(what.log_sds, 2))],
#     plot_kwargs=(
#         alpha=size(what, 1)^-.125, marker_z=z[idxs], markerstrokewidth=0, 
#         legend=:none, colorbar=:none, color=cgrad(:redsblues, rev=true), kwargs...
#     )
# ))
CenterednessComparison(drawss, idxs=1:20) = Figure([
    scatter_funnel(draws, i, title=round(draws.distribution.info.centeredness[i], digits=2))
    for i in idxs, draws in drawss
], plot_width=200)
kl_plot(what::UnconstrainedDraws, i=1, cs=LinRange(0, 1, 100)) = Line(
    cs, klps(
        what.x1s[:, i], 
        what.means[:, min(i, size(what.means, 2))], 
        what.log_sds[:, min(i, size(what.log_sds, 2))], 
        cs
    )
)



mcycle_dist(model) = (
    json_string = PDBPosterior("mcycle_gp-accel_gp").data_string;
    BSDistribution(
        "mcycle/$model.stan", json_string,
        info=MotorcycleInfo(JSON.parse(json_string)), 
    )
)