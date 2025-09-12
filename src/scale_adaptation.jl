import OnlineStatsBase

abstract type AbstractScaleAdaptation{T} end

struct Variances{T}
    location::Vector{T}
    squared_scale::Vector{T}
    n::Ref{Int}
    Variances(dim::Int; T=Float64) = new{T}(zeros(T, dim), zeros(T, dim), Ref(0))
end
OnlineStatsBase.fit!(v::Variances, y::AbstractVector) = begin 
    n = v.n[] += 1
    w = 1/n
    for i in eachindex(y)
        yi = y[i]
        preloc = v.location[i]
        postloc = v.location[i] = OnlineStatsBase.smooth(preloc, yi, w)
        v.squared_scale[i] = OnlineStatsBase.smooth(
            v.squared_scale[i], (yi - preloc) * (yi - postloc), w
        )
    end
    v
end
OnlineStatsBase.nobs(v::Variances) = v.n[]
OnlineStatsBase.value(v::Variances) = var(v)
Statistics.mean(v::Variances) = v.location
Statistics.var(v::Variances) = begin
    b = OnlineStatsBase.bessel(v)
    @broadcasted(v.squared_scale * b)
end
value!(target::AbstractVector, v::Variances) = target .= var(v)

struct RegularizedVariances{T}
    regularizing_n::Int
    regularizing_var::T
    variances::Variances{T}
    RegularizedVariances(dim::Int; T=Float64, regularizing_n=5, regularizing_var=1e-3) = new{T}(
        regularizing_n, regularizing_var, Variances(dim; T)
    )
end
OnlineStatsBase.fit!(v::RegularizedVariances, y::AbstractVector) = begin
    OnlineStatsBase.fit!(v.variances, y)
    v
end
Statistics.var((;regularizing_n, regularizing_var, variances)::RegularizedVariances) = begin
    n = OnlineStatsBase.nobs(variances)
    w = n / (n + regularizing_n)
    estimated_variances = var(variances)
    @broadcasted(OnlineStatsBase.smooth(regularizing_var, estimated_variances, w))
end


struct StanScaleAdaptation{T} <: AbstractScaleAdaptation{T} 
    variances::RegularizedVariances{T}
end
StanScaleAdaptation(dim::Int; kwargs...) = StanScaleAdaptation(RegularizedVariances(dim; kwargs...))
OnlineStatsBase.fit!(a::StanScaleAdaptation, p::AbstractVector, g=nothing) = begin
    OnlineStatsBase.fit!(a.variances, p)
    a
end
marginal_variances(a::StanScaleAdaptation) = var(a.variances)

struct NutpieScaleAdaptation{T} <: AbstractScaleAdaptation{T} 
    position_variances::RegularizedVariances{T}
    gradient_variances::RegularizedVariances{T}
end
NutpieScaleAdaptation(dim::Int; kwargs...) = NutpieScaleAdaptation(
    RegularizedVariances(dim; kwargs...),
    RegularizedVariances(dim; kwargs...)
)
OnlineStatsBase.fit!(a::NutpieScaleAdaptation, p::AbstractVector, g::AbstractVector) = begin
    OnlineStatsBase.fit!(a.position_variances, p)
    OnlineStatsBase.fit!(a.gradient_variances, g)
    a
end
marginal_variances((;position_variances, gradient_variances)::NutpieScaleAdaptation) = @broadcasted(
    sqrt($var(position_variances) / $var(gradient_variances))
)


marginal_variances!(a::AbstractScaleAdaptation) = marginal_variances(a)
marginal_scales(a::AbstractScaleAdaptation) = @broadcasted(sqrt($marginal_variances(a)))
marginal_scales!(a::AbstractScaleAdaptation) = @broadcasted(sqrt($marginal_variances!(a)))
marginal_scales!(scale::Diagonal, a::AbstractScaleAdaptation) = parent(scale) .= marginal_scales!(a)

struct IntermediateScaleAdaptation{T,A<:AbstractScaleAdaptation{T}} <: AbstractScaleAdaptation{T} 
    thresholds::Vector{T}
    adaptations::Vector{A}
end
IntermediateScaleAdaptation(dim::Int; A=NutpieScaleAdaptation, n_thresholds=2, thresholds=range(0, log(100), n_thresholds), kwargs...) = IntermediateScaleAdaptation(
    collect(thresholds),
    [
        A(dim; kwargs...) for i in 1:length(thresholds)
    ]
)
OnlineStatsBase.fit!(a::IntermediateScaleAdaptation, args...; dH) = begin
    for (threshold, adaptation) in zip(a.thresholds, a.adaptations)
        abs(dH) < threshold || continue
        OnlineStatsBase.fit!(adaptation, args...)
        break
    end
    a
end
OnlineStatsBase.fit!(a::IntermediateScaleAdaptation, lpdf::NUTSPosterior) = begin
    for (dH, p, g) in zip(lpdf.dH, eachcol(lpdf.position), eachcol(lpdf.gradient))
        OnlineStatsBase.fit!(a, p, g; dH)
    end
    a
end
marginal_variances(::IntermediateScaleAdaptation) = error("Computing the marginal variances for a `IntermediateScaleAdaptation` mutates state! Use `marginal_variances!`.")
@views marginal_variances!((;adaptations)::IntermediateScaleAdaptation) = begin 
    copy!(adaptations[1], adaptations[2])
    for adaptation in adaptations[3:end]
        compatible(adaptations[1], adaptation) || break
        merge!(adaptations[1], adaptation)
    end
    marginal_variances(adaptations[1])
end

Base.copy!(dest::StanScaleAdaptation, src::StanScaleAdaptation) = begin
    map(copy!, (;dest.variances), (;src.variances)) 
    dest
end

Base.copy!(dest::NutpieScaleAdaptation, src::NutpieScaleAdaptation) = begin
    map(copy!, (;dest.position_variances, dest.gradient_variances), (;src.position_variances, src.gradient_variances)) 
    dest
end
Base.copy!(dest::RegularizedVariances, src::RegularizedVariances) = begin
    map(copy!, (;dest.variances), (;src.variances)) 
    dest
end
Base.copy!(dest::Variances, src::Variances) = begin
    map(copy!, (;dest.location, dest.squared_scale), (;src.location, src.squared_scale))
    dest.n[] = src.n[] 
    dest
end
OnlineStatsBase.nobs(a::RegularizedVariances) = OnlineStatsBase.nobs(a.variances)
OnlineStatsBase.nobs(a::StanScaleAdaptation) = OnlineStatsBase.nobs(a.variances)
OnlineStatsBase.nobs(a::NutpieScaleAdaptation) = OnlineStatsBase.nobs(a.position_variances) + OnlineStatsBase.nobs(a.gradient_variances)
OnlineStatsBase.nobs(a::IntermediateScaleAdaptation) = OnlineStatsBase.nobs(a.adaptations[1])