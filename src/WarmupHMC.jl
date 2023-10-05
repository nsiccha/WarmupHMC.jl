module WarmupHMC

export regularize, to_x1, to_xc, klp, klps, approximately_whitened, mcmc_with_reparametrization, ConvenientLogDensityProblem, Ignore#, klps_plot!

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
lja_reparametrize(source, target, draws::AbstractMatrix, lja=0.) = begin 
    lja_reparametrize(source, target, lpdf_and_invariants(source, draws, Ignore()), lja)
    # rv = lja_reparametrize.([source], [target], eachcol(draws), lja)
    # first.(rv), hcat(last.(rv)...)
end
lja_reparametrize(source, target, draws::AbstractVector{<:NamedTuple}, lja=0.) = begin 
    rv = lja_reparametrize.([source], [target], draws, lja)
    first.(rv), hcat(last.(rv)...)
end

lja_reparametrize(source, target, draw::AbstractVector, lja=0.) = try 
    lja, tdraw = lja_reparametrize(source, target, lpdf_and_invariants(source, draw), lja)
    lja, collect(tdraw)
catch e
    @warn e
    NaN, NaN .* draw
end

lja_reparametrize(::Any, ::Any, invariants::NamedTuple, lja=0.) = begin 
    lja, invariants.draw
end

lpdf_and_invariants(source, draw::AbstractVector, lpdf=0.) = begin 
    lpdf += sum(logpdf.(source, draw))
    (;lpdf, draw)
end
lpdf_and_invariants(source, draws::AbstractMatrix, lpdf=0.) = lpdf_and_invariants.([source], eachcol(draws), [lpdf])

# lja(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[1]
# lja(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[1]
# lja(source, target, draws::AbstractMatrix) = lja.([source], [target], eachcol(draws))
reparametrize(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[2]
reparametrize(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[2]
# reparametrize(source, target, draws::AbstractMatrix) = hcat(
    # reparametrize.([source], [target], eachcol(draws))...
# )
reparametrization_loss(source, target, draws) = begin 
    ljas, reparametrized = lja_reparametrize(source, target, draws)
    nanmean(ljas) + nansum(log.(nanstd(reparametrized, dims=2)))
end
reparametrization_loss_function(source, draws::AbstractMatrix) = begin 
    reparametrization_loss_function(source, lpdf_and_invariants(source, draws, Ignore()))
end
reparametrization_loss_function(source, draws::AbstractVector{<:NamedTuple}) = begin 
    loss(v) = reparametrization_loss(source, reparametrize(source, v), draws)
end
find_reparametrization(source::UnivariateDistribution, ::AbstractMatrix; kwargs...) = source
find_reparametrization(kind::Symbol, source, draws; kwargs...) = find_reparametrization(Val{kind}(), source, draws; kwargs...)
# find_reparametrization_kwargs()
function mcmc_with_reparametrization end
function mcmc_keep_reparametrization end

end

