struct Progress{P,I}
    parent::P
    info::I
end
Base.parent(x::Progress) = x.parent
info(x::Progress) = x.info
initialize_progress!(::Nothing, args...; kwargs...) = nothing
update_progress!(::Nothing, args...; kwargs...) = nothing
fail_progress!(args...; kwargs...) = nothing
finalize_progress!(::Nothing, args...; kwargs...) = nothing
with_progress(f, args...; kwargs...) = begin 
    progress = initialize_progress!(args...; kwargs...)
    try
        return f(progress)
    catch e
        fail_progress!(progress, e)
        rethrow()
    finally
        finalize_progress!(progress)
    end
end


round2(x::OnlineStatsBase.Mean) = round2(mean(x))
round2(x::Real) = round(x; sigdigits=2)
round2(x::Integer) = round(x)
round2(x::Union{Tuple,NamedTuple,AbstractArray}) = map(round2, x)
round2(x::Missing) = x
round2(x::String) = x

struct UncertainFrequency
    obs::Int64
    n::Int64
end
Base.string(uf::UncertainFrequency) = "$(uf.obs) out of $(uf.n) ($(short_string(100*quantile(Beta(1+uf.obs, 1+uf.n-uf.obs), .05))) - $(short_string(100*quantile(Beta(1+uf.obs, 1+uf.n-uf.obs), .95)))%)" 
struct SamplingPerformance
    stepsize::Float64
    steps_per_draw::Float64
end
Base.string(x::SamplingPerformance) = "$(short_string(x.steps_per_draw)) steps per draw (stepsize = $(short_string(x.stepsize)))"
struct ActiveTransformation{K}
    kinetic_energy::K
    scale_changes::Vector{Float64}
end
struct Speed
    n::Int
    dt::Float64
    Speed(n, dt::UInt64) = new(n, Float64(dt))
end
Base.string(x::Speed) = "$(x.n) in $(short_string(x.dt/1e9)) seconds ($(short_string(x.n/(x.dt/1e9))) / s)"
Base.string(x::ActiveTransformation) = "$(short_string(x.kinetic_energy.M⁻¹.m1)) (marginal scale changes = $(short_string(x.scale_changes)))"
short_string(x::WarmupHMC.MatrixFactorization{<:Any, <:LinearAlgebra.Transpose}) = short_string(parent(x.m1))
short_string(x::Pathfinder.WoodburyPDRightFactor) = "Pathfinder($(size(x.V, 1)))"
short_string(x::WarmupHMC.MatrixFactorization{<:Any, <:WarmupHMC.SuccessiveReflections}) = "Adaptive($(length(x.m1.reflections)))"
short_string(x::Diagonal) = "Diagonal($(short_string(diag(x))))"
short_string(x::AbstractVector) = "[" * if length(x) > 7
    join(map(short_string, x[1:3]), ", ") * ", ..., " * join(map(short_string, x[end-2:end]), ", ")
else
    join(map(short_string, x), ", ")
end * "]"
short_string(x::Real) = begin
    rv = string(round2(x))
    endswith(rv, ".0") && length(rv) > 3 ? rv[1:end-2] : rv
end
short_string(x::Integer) = if x >= 1e9
    short_string(x / 1e9) * "G"
elseif x >= 1e6
    short_string(x / 1e6) * "M"
elseif x >= 1e3
    short_string(x / 1e3) * "k"
else
    string(x)
end
short_string(x::Beta) = "$(short_string(x.α)) out of $(short_string(x.α+x.β)) ($(short_string(100*quantile(x, .05))) - $(short_string(100*quantile(x, .95)))%)" 
short_string(x) = string(x)
short_string(x::Pair) = "$(short_string(first(x))) => $(short_string(last(x)))"
short_string(x::NamedTuple) = "(;" * join([
    short_string(key) * "=" * short_string(value)
    for (key, value) in pairs(x)
], ", ") * ")"

struct Fraction{T}
    value::T
end
short_string(x::Fraction) = short_string(x.value * 100) * "%"
Base.isless(x::Fraction, y::Fraction) = isless(x.value, y.value)
Base.isequal(x::Fraction, y::Fraction) = isequal(x.value, y.value)

struct Speeds
    n::Vector{Int}
    dt::Float64
    Speeds(n, dt::UInt64) = new(n, Float64(dt))
end
@views Base.string(x::Speeds) = begin
    p = sortperm(x.n)
    dt = (time_ns() - x.dt)/1e9
    "$(short_string(x.n[p])) in $(short_string(dt)) seconds ($(short_string(x.n[p]./(dt))) / s)"
end
struct Speeds2{T}
    n::T
    dt::Float64
    Speeds2(n, dt::UInt64) = new{typeof(n)}(n, Float64(dt))
end
@views Base.string(x::Speeds2) = begin
    p = sortperm(x.n)
    dt = (time_ns() - x.dt)/1e9
    "$(short_string(x.n[p])) in $(short_string(dt)) seconds ($(short_string(x.n[p]./(dt))) / s)"
end
