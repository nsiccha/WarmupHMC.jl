PairTT{T} = Pair{T,T}
find_reparametrization!(lpdf, p, g, pg) = pg
find_reparametrization!(lpdf::WrappedLogDensityProblem, args...) = find_reparametrization!(parent(lpdf), args...)
reparametrize!(lpdf::WrappedLogDensityProblem, args...) = reparametrize!(parent(lpdf), args...)
reparametrize!(st::Pair, p) = WarmupHMC.reparametrize!(st, p=>p) 
reparametrize!(st::Pair, p::PairTT{<:AbstractMatrix}) = mapreduce(+, eachcol(p[1]), eachcol(p[2])) do source, target
    reparametrize!(st, source=>target)
end
reparametrize!(lpdf, args...) = nothing

abstract type AbstractReparametrizer{M} end
struct Argmin{C}
    candidates::C
end
(m::Argmin)(x::AbstractReparametrizer) = argmin(c->WarmupHMC.reparametrization_loss(x, c), m.candidates)

struct CenterednessReparametrizer{M,W} <: AbstractReparametrizer{M}
    method::M
    working_memory::W
end
CenterednessReparametrizer(method, m::Integer) = CenterednessReparametrizer(
    method,
    (loc=zeros(m), log_scale=zeros(m), xi=zeros(m), xc=zeros(m))
)
CenterednessReparametrizer(working_memory; method=Argmin((0., 1.))) = CenterednessReparametrizer(method, working_memory)
reparametrize!(x::CenterednessReparametrizer, xc::AbstractVector, c::Real) = begin 
    (;loc, log_scale, xi) = x.working_memory 
    xc .= loc .* c .+ xi .* exp.(log_scale .* c)
    -mean(Base.broadcasted(*, log_scale, c))
end
reparametrization_loss(x::CenterednessReparametrizer, c) = begin 
    (;xc) = x.working_memory
    lj = reparametrize!(x, xc, c)
    lj + log(std(xc))
end
find_reparametrization!(x::CenterednessReparametrizer, xc::AbstractVector, c::Real, loc, log_scale) = begin 
    x.working_memory.loc .= loc
    x.working_memory.log_scale .= log_scale
    x.working_memory.xi .= (xc .- loc .* c) .* exp.(.-log_scale.*c)
    cstar = x.method(x)
    reparametrize!(x, xc, cstar)
    cstar
end

maybeeachcol(n) = Base.Fix2(maybeeachcol, n)
maybeeachcol(x::AbstractMatrix, n) = eachcol(x)
maybeeachcol(x, n) = Fill(x, n)
find_reparametrization!(x::AbstractReparametrizer, draws::AbstractMatrix, params::AbstractVector, args...; progress=nothing) = with_progress(progress, length(params)) do progress
    map!(params, eachcol(draws), params, map(maybeeachcol(length(params)), args)...) do subargs...
        WarmupHMC.update_progress!(progress)
        find_reparametrization!(x, subargs...)
    end
end