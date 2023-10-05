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
end
lja_reparametrize(source, target, draws::AbstractVector{<:NamedTuple}, lja=0.) = begin 
    rv = lja_reparametrize.(Ref(source), Ref(target), draws, lja)
    first.(rv), hcat(last.(rv)...)
end

lja_reparametrize(source, target, draw::AbstractVector, lja=0.) = try 
    lja_reparametrize(source, target, lpdf_and_invariants(source, draw, Ignore()), lja)
catch e
    @warn """
Failed to reparametrize: 
$source 
$target 
$draw
$(lpdf_and_invariants(source, draw, Ignore()))
$(exception_to_string(e))
    """
    NaN, NaN .* draw
end

lja_reparametrize(::Any, ::Any, invariants::NamedTuple, lja=0.) = begin 
    lja, invariants.draw
end

lpdf_and_invariants(source, draw::AbstractVector, lpdf=0.) = begin 
    lpdf += sum(logpdf.(source, draw))
    (;lpdf, draw)
end
lpdf_and_invariants(source, draws::AbstractMatrix, lpdf=0.) = lpdf_and_invariants.(Ref(source), eachcol(draws), lpdf)

# lja(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[1]
# lja(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[1]
# lja(source, target, draws::AbstractMatrix) = lja.(Ref(source), Ref(target), eachcol(draws))
reparametrize(source::Any, target::Any, draw) = lja_reparametrize(source, target, draw, Ignore())[2]
# reparametrize(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[2]
# reparametrize(source, target, draws::AbstractMatrix) = hcat(
    # reparametrize.(Ref(source), Ref(target), eachcol(draws))...
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
find_reparametrization(source::UnivariateDistribution, ::Any; kwargs...) = source
find_reparametrization(kind::Symbol, source, draws; kwargs...) = find_reparametrization(Val{kind}(), source, draws; kwargs...)
# find_reparametrization_kwargs()
function mcmc_with_reparametrization end
function mcmc_keep_reparametrization end


# Debug print exceptions:
# https://stackoverflow.com/questions/72718578/julia-how-to-get-an-error-message-and-stacktrace-as-string
function exception_to_string(e)
    error_msg = sprint(showerror, e)
    st = sprint((io,v) -> show(io, "text/plain", v), stacktrace(catch_backtrace()))
    "Trouble doing things:\n$(error_msg)\n$(st)"
end
end

