module WarmupHMC

export regularize, to_x1, to_xc, klp, klps, approximately_whitened, mcmc_with_reparametrization, ConvenientLogDensityProblem#, klps_plot!

using DynamicObjects
using Random, Distributions, LinearAlgebra
using LogDensityProblems
using NaNStatistics

include("deprecated.jl")

struct Ignore <: Real end
Base.:+(lhs::Ignore, ::Real) = lhs
Base.:-(lhs::Ignore, ::Real) = lhs

reparametrization_parameters(::Any) = Float64[]
reparametrize(source::Any, ::Any) = source
lja_reparametrize(source, target, draws::AbstractMatrix) = begin 
    rv = lja_reparametrize.([source], [target], eachcol(draws))
    first.(rv), hcat(last.(rv)...)
end

lja(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[1]
lja(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[1]
# lja(source, target, draws::AbstractMatrix) = lja.([source], [target], eachcol(draws))
reparametrize(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[2]
reparametrize(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[2]
# reparametrize(source, target, draws::AbstractMatrix) = hcat(
    # reparametrize.([source], [target], eachcol(draws))...
# )
reparametrization_loss(source, target, draws::AbstractMatrix) = begin 
    ljas, reparametrized = lja_reparametrize(source, target, draws)
    nanmean(ljas) + nansum(log.(nanstd(reparametrized, dims=2)))
end
reparametrization_loss_function(source, draws::AbstractMatrix) = begin 
    loss(v) = reparametrization_loss(source, reparametrize(source, v), draws)
end
# unconstrained_reparametrization_loss_function(source, draws::AbstractMatrix) = begin 
#     loss(v) = reparametrization_loss(source, unconstrained_reparametrize(source, v), draws)
# end
# find_reparametrization(source, ::AbstractMatrix) = source
find_reparametrization(kind::Symbol, source, draws::AbstractMatrix; kwargs...) = find_reparametrization(Val{kind}(), source, draws; kwargs...)
function mcmc_with_reparametrization end
function mcmc_keep_reparametrization end



# lpdf_and_invariants(source, draw::AbstractVector) = (
#     LogDensityProblems.logdensity(source, draw), nothing
# )
lpdf_and_invariants(source, draw::AbstractVector, lpdf=0.) = begin 
    lpdf += sum(logpdf.(source, draw))
    (;lpdf, draw)
end

lja_reparametrize(::Any, ::Any, invariants::NamedTuple, lja=0.) = begin 
    lja, invariants.draw
end

# lpdf_and_invariants(source, draws::AbstractMatrix) = begin 
#     @warn "This overload is deprecated!"
#     rv = lpdf_and_invariants.([source], eachcol(draws))
#     first.(rv), last.(rv)
# end

# struct ConvenientLogDensityProblem{P,L,I}
#     prior::P
#     likelihood::L
#     draw_boundaries::Vector{I}
#     parameter_boundaries::Vector{I}
# end
# ConvenientLogDensityProblem(prior, likelihood) = begin
#     ConvenientLogDensityProblem(
#         prior, likelihood,  
#         vcat(0, cumsum(LogDensityProblems.dimension.(prior))), 
#         vcat(0, cumsum(length.(reparametrization_parameters.(prior)))), 
#     )
# end
# LogDensityProblems.dimension(source::ConvenientLogDensityProblem) = sum(LogDensityProblems.dimension.(source.prior))
# subdraws(source::ConvenientLogDensityProblem, draw::AbstractVector) = view.([draw], range.(1 .+ source.draw_boundaries[1:end-1], source.draw_boundaries[2:end]))
# subparameters(source::ConvenientLogDensityProblem, parameters) = view.([parameters], range.(1 .+ source.parameter_boundaries[1:end-1], source.parameter_boundaries[2:end]))
# reparametrization_parameters(source::ConvenientLogDensityProblem) = vcat(
#     reparametrization_parameters.(source.prior)...
# )

# @views LogDensityProblems.logdensity(source::ConvenientLogDensityProblem, draw::AbstractVector) = begin 
#     intermediates = map(
#         lpdf_and_invariants, source.prior, subdraws(source, draw)
#     )
#     sum(first.(intermediates)) + source.likelihood(intermediates)
# end
# reparametrize(source::ConvenientLogDensityProblem, parameters) = ConvenientLogDensityProblem(reparametrize.(source.prior, subparameters(source, parameters)), source.likelihood, source.draw_boundaries, source.parameter_boundaries)
# reparametrize(source::ConvenientLogDensityProblem, target::ConvenientLogDensityProblem, draw::AbstractVector) = vcat(reparametrize.(source.prior, target.prior, subdraws(source, draw))...)
# lja(source::ConvenientLogDensityProblem, target::ConvenientLogDensityProblem, draw::AbstractVector) = sum(lja.(source.prior, target.prior, subdraws(source, draw)))
end

