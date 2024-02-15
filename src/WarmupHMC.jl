module WarmupHMC

export mcmc_with_reparametrization, Ignore

using Distributions, LinearAlgebra, NaNStatistics

struct Ignore <: Real end
Base.:+(lhs::Ignore, ::Real) = lhs
Base.:-(lhs::Ignore, ::Real) = lhs

# Maybe replace NamedTuple by a custom type

# IMPLEMENT THIS
reparametrization_parameters(::Any) = NamedTuple()
# IMPLEMENT THIS
optimization_reparametrization_parameters(::Any) = Float64[]
# IMPLEMENT THIS
reparametrize(source, ::Any) = source
reparametrize(source, target, draw::AbstractArray) = to_array(
    source, lja_and_reparametrize(source, target, draw, Ignore())
)
# MAYBE IMPLEMENT THIS
to_array(::Any, draw::NamedTuple) = draw.draw
to_array(::Any, draw::AbstractVector) = draw
to_array(::Any, draw::AbstractMatrix) = draw
to_array(source, draw::AbstractVector{<:NamedTuple}) = hcat(to_array.(Ref(source), draw)...)
# MAYBE IMPLEMENT THIS
to_nt(::Any, draw::AbstractArray) = (;draw)

# IMPLEMENT THIS
lpdf_update(source, draw::NamedTuple, lpdf=0.) = begin
    lpdf += sum(logpdf.(source, draw.draw))
    (;lpdf)
end
lpdf_update(::Tuple{}, draw::NamedTuple, lpdf=0.) = (;lpdf)
# IMPLEMENT THIS
lja_update(::Any, ::Any, invariants::NamedTuple, lja=0.) = begin 
    (;lja)
end

lpdf_and_invariants(source, draw::NamedTuple, lpdf=0.) = merge(
    draw, lpdf_update(source, draw, lpdf)
)
lpdf_and_invariants(source, draw::AbstractVector, lpdf=0.) = lpdf_and_invariants(
    source, to_nt(source, draw), lpdf
)
lpdf_and_invariants(source, draw::AbstractMatrix, lpdf=0.) = lpdf_and_invariants.(
    Ref(source), eachcol(draw), lpdf
)
lja_and_reparametrize(source, target, draw::NamedTuple, lja=0.) = merge(
    draw, lja_update(source, target, draw, lja)
)
lja_and_reparametrize(source, target, draw::AbstractVector, lja=0.) = lja_and_reparametrize(
    source, target, lpdf_and_invariants(source, draw, Ignore()), lja
)
lja_and_reparametrize(source, target, draw::AbstractMatrix, lja=0.) = lja_and_reparametrize.(
    Ref(source), Ref(target), eachcol(draw), lja
)
lja_and_reparametrize(source, target, draw::AbstractVector{<:NamedTuple}, lja=0.) = 
    lja_and_reparametrize.(Ref(source), Ref(target), draw, lja)

reparametrization_loss_function(::Any, ::AbstractMatrix) = error("This overload should generally be inefficient!")
reparametrization_loss_function(source, draw::AbstractVector{<:NamedTuple}) = begin 
    loss(v) = reparametrization_loss(source, reparametrize(source, v), draw)
end
reparametrization_loss(source, target, draw::AbstractVector{<:NamedTuple}) = begin 
    tmp = lja_and_reparametrize(source, target, draw)
    ljas = getproperty.(tmp, :lja)
    reparametrized = to_array(source, tmp)
    nanmean(ljas) + nansum(log.(nanstd(reparametrized, dims=2)))
end

# MAY IMPLEMENT THIS
find_reparametrization(source, draw::AbstractMatrix; kwargs...) = find_reparametrization(
    source, lpdf_and_invariants(source, draw, Ignore()); kwargs...
)
# IMPLEMENT THIS
find_reparametrization(source, ::AbstractVector{<:NamedTuple}; kwargs...) = source
find_reparametrization(source, ::Any; kwargs...) = source
find_reparametrization(kind::Symbol, source, draw; kwargs...) = find_reparametrization(
    Val{kind}(), source, draw; kwargs...
)
function mcmc_with_reparametrization end
function mcmc_keep_reparametrization end
# MAY IMPLEMENT THIS
find_reparametrization_and_reparametrize(reparametrization, posterior_matrix; kwargs...) = begin 
    new_reparametrization = find_reparametrization(reparametrization, posterior_matrix; kwargs...)
    posterior_matrix = reparametrize(reparametrization, new_reparametrization, posterior_matrix)
    return new_reparametrization, posterior_matrix
end


# Debug print exceptions:
# https://stackoverflow.com/questions/72718578/julia-how-to-get-an-error-message-and-stacktrace-as-string
function exception_to_string(e)
    error_msg = sprint(showerror, e)
    st = sprint((io,v) -> show(io, "text/plain", v), stacktrace(catch_backtrace()))
    "Trouble doing things:\n$(error_msg)\n$(st)"
end

abstract type InfoStruct end
Base.getproperty(source::T, key::Symbol) where {T<:InfoStruct} = hasfield(T, key) ? getfield(source, key) : getproperty(source.info, key)
struct TuningConfig{T,I<:NamedTuple} <: InfoStruct
    info::I
    TuningConfig{T}(;kwargs...) where {T} = TuningConfig{T}((;kwargs...))
    TuningConfig{T}(info::I) where {T,I<:NamedTuple} = new{T,I}(info)
end

end

