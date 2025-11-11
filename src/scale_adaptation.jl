import OnlineStatsBase
import OnlineStatsBase: fit!, nobs, smooth

abstract type AbstractScaleAdaptation{T} end

struct Variances{T}
    location::Vector{T}
    squared_scale::Vector{T}
    n::Ref{T}
    Variances(dim::Int; T=Float64) = new{T}(zeros(T, dim), zeros(T, dim), Ref(zero(T)))
end
OnlineStatsBase.fit!(v::Variances, y::AbstractVector; dw=1) = begin 
    @assert dw >= 0 dw
    n = v.n[] += dw
    w = dw/n
    for i in eachindex(y)
        yi = y[i]
        preloc = v.location[i]
        postloc = v.location[i] = smooth(preloc, yi, w)
        v.squared_scale[i] = smooth(
            v.squared_scale[i], (yi - preloc) * (yi - postloc), w
        )
    end
    v
end
OnlineStatsBase.nobs(v::Variances) = v.n[]
OnlineStatsBase.value(v::Variances) = var(v)
Statistics.mean(v::Variances) = v.location
positiveorNaN(x) = x > 0 ? x : zero(x)#NaN
Statistics.var(v::Variances) = begin
    b = OnlineStatsBase.bessel(v)
    @broadcasted(positiveorNaN(v.squared_scale * b))
end
value!(target::AbstractVector, v::Variances) = target .= var(v)

struct RegularizedVariances{T}
    regularizing_n::T
    regularizing_var::T
    variances::Variances{T}
    RegularizedVariances(dim::Int; T=Float64, regularizing_n=5., regularizing_var=1e-3) = new{T}(
        regularizing_n, regularizing_var, Variances(dim; T)
    )
end
OnlineStatsBase.fit!(v::RegularizedVariances, y::AbstractVector; kwargs...) = begin
    fit!(v.variances, y; kwargs...)
    v
end
Statistics.var((;regularizing_n, regularizing_var, variances)::RegularizedVariances) = begin
    n = nobs(variances)
    w = n / (n + regularizing_n)
    estimated_variances = var(variances)
    @broadcasted(smooth(regularizing_var, estimated_variances, w))
end


struct StanScaleAdaptation{T} <: AbstractScaleAdaptation{T} 
    variances::RegularizedVariances{T}
end
StanScaleAdaptation(dim::Int; kwargs...) = StanScaleAdaptation(RegularizedVariances(dim; kwargs...))
OnlineStatsBase.fit!(a::StanScaleAdaptation, p::AbstractVector, g=nothing; kwargs...) = begin
    fit!(a.variances, p; kwargs...)
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
OnlineStatsBase.fit!(a::NutpieScaleAdaptation, p::AbstractVector, g::AbstractVector; kwargs...) = begin
    fit!(a.position_variances, p; kwargs...)
    fit!(a.gradient_variances, g; kwargs...)
    a
end
marginal_variances((;position_variances, gradient_variances)::NutpieScaleAdaptation) = @broadcasted(
    sqrt($var(position_variances) / $var(gradient_variances))
)


marginal_variances!(a::AbstractScaleAdaptation) = marginal_variances(a)
marginal_scales(a::AbstractScaleAdaptation) = @broadcasted(sqrt($marginal_variances(a)))
marginal_scales!(a::AbstractScaleAdaptation) = @broadcasted(sqrt($marginal_variances!(a)))
marginal_scales!(scale::Diagonal, a::AbstractScaleAdaptation) = parent(scale) .= marginal_scales!(a)

struct IntermediateScaleAdaptation{W,T,A<:AbstractScaleAdaptation{T}} <: AbstractScaleAdaptation{T} 
    thresholds::Vector{T}
    adaptations::Vector{A}
    IntermediateScaleAdaptation(W, thresholds, adaptations) = new{W,eltype(thresholds),eltype(adaptations)}(thresholds, adaptations)
end
IntermediateScaleAdaptation(dim::Int; W=:uniform, A=NutpieScaleAdaptation, n_thresholds=2, thresholds=range(0, log(100), n_thresholds), kwargs...) = IntermediateScaleAdaptation(
    W,
    collect(thresholds),
    [
        A(dim; kwargs...) for i in 1:length(thresholds)
    ]
)
OnlineStatsBase.fit!(a::IntermediateScaleAdaptation, args...; dH, kwargs...) = begin
    for (threshold, adaptation) in zip(a.thresholds, a.adaptations)
        abs(dH) < threshold || continue
        OnlineStatsBase.fit!(adaptation, args...; kwargs...)
        break
    end
    a
end
OnlineStatsBase.fit!(a::IntermediateScaleAdaptation, lpdf::AbstractNUTSPosterior) = fit!(a, parent(lpdf))
OnlineStatsBase.fit!(a::IntermediateScaleAdaptation, lpdf::NUTSPosterior) = begin
    for (dH, p, g) in zip(lpdf.dH, eachcol(lpdf.recorder.positions.value), eachcol(lpdf.recorder.gradients.value))
        fit!(a, p, g; dH)
    end
    a
end
OnlineStatsBase.fit!(a::IntermediateScaleAdaptation{:nuts}, lpdf::NUTSPosterior) = begin
    lw = @broadcasted(min(0, lpdf.dH))
    D = (1 + sum(exp, lw))
    weights = @broadcasted(exp(lw) / D)
    fit!(a, lpdf.recorder.initial_position.value, lpdf.recorder.initial_gradient.value; dH=0., dw=max(0, 1 - sum(weights)))
    for (dH, p, g, dw) in zip(lpdf.dH, eachcol(lpdf.recorder.positions.value), eachcol(lpdf.recorder.gradients.value), weights)
        fit!(a, p, g; dH=0., dw)
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
compatible(args::IntermediateScaleAdaptation...) = compatible(last.(getproperty.(args, :adaptations))...)
compatible(args::NutpieScaleAdaptation...) = compatibility(args...) >= 0
compatibility(args::NutpieScaleAdaptation...) = begin 
    ARGS = merge(args...)
    maximum(args) do arg 
        mykldivergence(arg, ARGS.position_variances) - mykldivergence(arg, ARGS)
    end
    # maximum(Base.Fix2(mykldivergence, ARGS.position_variances), args) - maximum(Base.Fix2(mykldivergence, ARGS), args)
end
compatibility2(arg, merged) = mykldivergence(arg, merged.position_variances) - mykldivergence(arg, merged)
cond_compatibility(x, y) = diagonal_cond(@broadcasted($marginal_scales!(x) / $marginal_scales!(y)))
marginal_scales(x::Diagonal) = parent(x)
marginal_scales!(x::Diagonal) = parent(x)
diagonal_cond(x) = \(extrema(x)...)
mykldivergence(x, y) = @bsum(
    mykldivergence(
        $marginal_locations(x),
        $marginal_scales(x),
        $marginal_locations(y),
        $marginal_scales(y)
    )
)
mykldivergence(loc1, scale1, loc2, scale2) = if all(isfinite, (loc1, scale1, loc2, scale2))
    Distributions.kldivergence(Normal(loc1, scale1), Normal(loc2, scale2))
else
    NaN
end
marginal_locations(a::NutpieScaleAdaptation) = @broadcasted(
    $marginal_locations(a.position_variances) + $marginal_variances(a) * $marginal_locations(a.gradient_variances)
)
marginal_locations(a::RegularizedVariances) = marginal_locations(a.variances)
marginal_locations(a::Variances) = a.location
marginal_scales(a::RegularizedVariances) = @broadcasted(sqrt($var(a)))
Base.length(a::NutpieScaleAdaptation) = length(a.gradient_variances)
Base.length(a::RegularizedVariances) = length(a.variances)
regularizing_n(a::RegularizedVariances) = a.regularizing_n
regularizing_n(a::NutpieScaleAdaptation) = regularizing_n(a.position_variances)
checkunique(args) = begin
    @assert all(==(args[1]), args[2:end])
    args[1]
end
checkunique(f, args) = checkunique(map(f, args))
Base.length(a::Variances) = length(a.location)
Base.eachindex(a::Variances) = eachindex(a.location)

Base.merge(a::NutpieScaleAdaptation, rgs::NutpieScaleAdaptation...) = merge!(
    NutpieScaleAdaptation(
        checkunique(length, (a, rgs...));
        regularizing_n=checkunique(regularizing_n.((a, rgs...)))
    ), a, rgs...
)
Base.merge!(a::NutpieScaleAdaptation, rgs::NutpieScaleAdaptation...) = begin 
    args = (a, rgs...)
    merge!(getproperty.(args, :position_variances)...) 
    merge!(getproperty.(args, :gradient_variances)...)
    a
end
Base.merge!(a::RegularizedVariances, rgs::RegularizedVariances...) = begin
    merge!(getproperty.((a, rgs...), :variances)...)
    a
end
safe_divide(x, y) = x == 0 ? zero(x) : x / y
Base.merge!(a::Variances, rgs::Variances...) = begin 
    args = (a, rgs...)
    w2s = safe_divide.(nobs.(rgs), (nobs.(rgs) .+ cumsum(nobs.(args)[1:end-1])))
    @assert all(isfinite, w2s) w2s
    for i in eachindex(a)
        loc = a.location[i]
        ss = a.squared_scale[i]
        for (w2, arg) in zip(w2s, rgs)
            w1 = 1 - w2
            loc2 = arg.location[i]
            ss2 = arg.squared_scale[i]
            dloc = loc2 - loc
            loc = smooth(loc, loc2, w2)
            ss = smooth(ss, ss2, w2) + dloc^2 * w2 * w1
        end
        a.location[i] = loc
        a.squared_scale[i] = ss
    end
    a.n[] = sum(nobs, args)
    a
end
unsmooth(x12, x2, w2; w1=1-w2) = (x12 - w2 * x2) / w1
unmerge!(a::NutpieScaleAdaptation, rgs::NutpieScaleAdaptation...) = begin 
    args = (a, rgs...)
    unmerge!(getproperty.(args, :position_variances)...) 
    unmerge!(getproperty.(args, :gradient_variances)...)
    a
end
unmerge!(a::RegularizedVariances, rgs::RegularizedVariances...) = begin
    unmerge!(getproperty.((a, rgs...), :variances)...)
    a
end
unmerge!(a::Variances, arg::Variances) = begin 
    @assert nobs(a) >= nobs(arg)
    w1 = (nobs(a) - nobs(arg)) / nobs(a)
    w2 = 1 - w1
    for i in eachindex(a)
        loc = a.location[i]
        ss = a.squared_scale[i]
        loc2 = arg.location[i]
        ss2 = arg.squared_scale[i]
        a.location[i] = unsmooth(loc, loc2, w2; w1)
        a.squared_scale[i] = unsmooth(ss, ss2, w2; w1) - ((loc2 - loc) / w1)^2*w2
    end
    a.n[] -= nobs(arg)
    a
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
OnlineStatsBase.nobs(a::RegularizedVariances) = nobs(a.variances)
OnlineStatsBase.nobs(a::StanScaleAdaptation) = nobs(a.variances)
OnlineStatsBase.nobs(a::NutpieScaleAdaptation) = nobs(a.position_variances)# + nobs(a.gradient_variances)
@views OnlineStatsBase.nobs(a::IntermediateScaleAdaptation) = sum(nobs, a.adaptations[2:end])

OnlineStatsBase.fit!(
    a::StanScaleAdaptation, ::AbstractNUTSPosterior, position_and_gradient::DynamicHMC.EvaluatedLogDensity
) = OnlineStatsBase.fit!(a, position_and_gradient.q)
OnlineStatsBase.fit!(
    a::NutpieScaleAdaptation, ::AbstractNUTSPosterior, position_and_gradient::DynamicHMC.EvaluatedLogDensity
) = OnlineStatsBase.fit!(a, position_and_gradient.q, position_and_gradient.∇ℓq)
OnlineStatsBase.fit!(
    a::IntermediateScaleAdaptation, p::AbstractNUTSPosterior, ::DynamicHMC.EvaluatedLogDensity
) = OnlineStatsBase.fit!(a, p)