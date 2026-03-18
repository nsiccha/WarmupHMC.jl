# Progress functions come from Treebars (using Treebars in WarmupHMC.jl).
# This file only defines domain-specific display types and short_string/round2 extensions.

import Treebars: short_string, round2

round2(x::OnlineStatsBase.Mean) = round2(mean(x))

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
short_string(x::Beta) = "$(short_string(x.α)) out of $(short_string(x.α+x.β)) ($(short_string(100*quantile(x, .05))) - $(short_string(100*quantile(x, .95)))%)"

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
