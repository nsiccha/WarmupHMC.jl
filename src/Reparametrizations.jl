# Generic adaptive nonlinear reparametrization machinery
# Migrated from LocalScalesHMC.jl/julia/reparametrizations.jl

maybecall(f::Function, args...; kwargs...) = f(args...; kwargs...)
maybecall(x, args...; kwargs...) = x

"""
    ReparametrizedProblem(reparametrizer, problem, ad_backend=nothing)

Wrap a `LogDensityProblems`-compatible `problem` with a nonlinear reparametrization.
The `reparametrizer` (typically an [`IndexedReparametrization`](@ref)) transforms the
parameter vector before evaluating the log density, adding the log-Jacobian correction.

Gradients are computed by differentiating only through the reparametrization transform
(using `ad_backend`, e.g. `AutoMooncake()` from DifferentiationInterface.jl), while
reusing the inner problem's native gradient. This allows use with FFI-based backends
like BridgeStan.

# Example
```julia
using WarmupHMC, DifferentiationInterface, Mooncake
ir = IndexedReparametrization([
    i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
        x -> x[loc_idx], x -> x[scale_idx])
    for i in param_indices
])
rp = ReparametrizedProblem(ir, my_problem, AutoMooncake())
result = adaptive_warmup_mcmc(rng, rp)
```
"""
struct ReparametrizedProblem{R,P,B}
    reparametrizer::R
    problem::P
    ad_backend::B
end
ReparametrizedProblem(r, p) = ReparametrizedProblem(r, p, nothing)
reparametrizer(p::ReparametrizedProblem) = p.reparametrizer
reparametrizer(p::WrappedLogDensityProblem) = reparametrizer(parent(p))
reparametrizer(::Any) = IndexedReparametrization([])
LogDensityProblems.capabilities(::Type{<:ReparametrizedProblem{R,P}}) where {R,P} = LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(p::ReparametrizedProblem) = LogDensityProblems.dimension(p.problem)
LogDensityProblems.logdensity(p::ReparametrizedProblem, x::AbstractVector) = begin
    ljac, y = p.reparametrizer(x)
    ljac + LogDensityProblems.logdensity(p.problem, y)
end
LogDensityProblems.logdensity_and_gradient(p::ReparametrizedProblem, x::AbstractVector) = begin
    # Differentiate only through the reparametrization, not through the inner problem.
    # The inner problem (e.g. BridgeStan) provides its own gradients via FFI.
    _logdensity_and_gradient_reparam(p, x)
end
# Fallback — overridden by DifferentiationInterfaceExt
_logdensity_and_gradient_reparam(p::ReparametrizedProblem, x) =
    error("ReparametrizedProblem requires DifferentiationInterface to compute gradients. Load DifferentiationInterface and pass an AD backend to ReparametrizedProblem.")

# --- Abstract reparametrization interface ---

abstract type AbstractReparametrization end
(t::AbstractReparametrization)(x) = with_logabsdet_jacobian!(copy(x), t, x)
with_logabsdet_jacobian!(Y::AbstractMatrix, t, X::AbstractMatrix) = begin
    map(eachcol(Y), eachcol(X)) do y, x
        with_logabsdet_jacobian!(y, t, x)[1]
    end, Y
end

"""
    PartiallyCentered(c)

Centering parameter for hierarchical reparametrization, where `c ∈ [0, 1]`.
`c = 0` is fully non-centered, `c = 1` is fully centered.

For a parameter `x` with location `loc` and log-scale `log_scale`, the transform from
`PartiallyCentered(source)` to `PartiallyCentered(target)` is:

    y = target * loc + (x - source * loc) * exp(log_scale * (target - source))

with log-Jacobian `log_scale * (target - source)`.

During warmup, the optimizer tries multiple candidate centering values and picks
the one minimizing a correlation-based loss.
"""
struct PartiallyCentered{C}
    c::C
end
reparam(target::PartiallyCentered, source::PartiallyCentered, x::Real, loc::Real, log_scale::Real) = begin
    ljac = log_scale * (target.c - source.c)
    ljac, target.c * loc + xexpy(x - source.c * loc, ljac)
end
reparam(target::PartiallyCentered, source::PartiallyCentered, x::Real, g::Real, loc::Real, log_scale::Real) = begin
    ljac = log_scale * (target.c - source.c)
    s = exp(ljac)
    ljac, target.c * loc + (x - source.c * loc) * s, g / s
end

"""
    Reparametrization(target, source, args...)

Maps between two [`PartiallyCentered`](@ref) parametrizations. The `args` are either
constant values or functions `x -> x[i]` that extract the location and log-scale
from the full parameter vector.

# Example
```julia
# Reparametrize dimensions 1:8, with location at x[9] and log-scale at x[10]
Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0), x -> x[9], x -> x[10])
```

During adaptation, the `source` centering is updated to minimize a loss function
while `target` stays fixed.
"""
struct Reparametrization{T,S,A}
    target::T
    source::S
    args::A
    Reparametrization(target, source, args...) = new{typeof(target), typeof(source), typeof(args)}(target, source, args)
end
InverseFunctions.inverse((;target, source, args)::Reparametrization) = Reparametrization(source, target, args...)
reparam_rargs((;source, args)::Reparametrization, xi, x) = (source, xi, map(Base.Fix2(maybecall, x), args)...)
reparam_rargs((;source, args)::Reparametrization, xi, gi, x) = (source, xi, gi, map(Base.Fix2(maybecall, x), args)...)
reparam(r::Reparametrization, xi, x) = reparam(r.target, reparam_rargs(r, xi, x)...)
reparam(r::Reparametrization, xi, gi, x) = reparam(r.target, reparam_rargs(r, xi, gi, x)...)

"""
    IndexedReparametrization(pairs)

Maps dimension indices to [`Reparametrization`](@ref) objects. This is the main container
passed to [`ReparametrizedProblem`](@ref).

`pairs` is a vector of `idx => Reparametrization(...)` entries. Dimensions not listed
are passed through unchanged.

# Example
```julia
# Eight schools: reparametrize dims 1:8 with shared location (dim 9) and scale (dim 10)
ir = IndexedReparametrization(
    1:8 .=> Ref(Reparametrization(
        PartiallyCentered(1.0), PartiallyCentered(1.0),
        x -> x[9], x -> x[10]
    ))
)
```
"""
struct IndexedReparametrization{P} <: AbstractReparametrization
    pairs::P
end
with_logabsdet_jacobian!(y::AbstractVector, (;pairs)::IndexedReparametrization, x::AbstractVector) = begin
    ljac = 0.
    for (idx, value) in pairs
        tmp, y[idx] = reparam(value, x[idx], x)
        ljac += tmp
    end
    ljac, y
end
InverseFunctions.inverse((;pairs)::IndexedReparametrization) = IndexedReparametrization([
    idx => inverse(value) for (idx, value) in pairs
])

# --- Online reparametrization loss tracking ---

struct OnlineReparametrizationLoss{L<:OnlineStatsBase.Mean,C<:OnlineStatsBase.CovMatrix}
    ljac::L
    cov::C
end
OnlineStatsBase.nobs((;ljac)::OnlineReparametrizationLoss) = OnlineStatsBase.nobs(ljac)
OnlineReparametrizationLoss(::AbstractMatrix) = OnlineReparametrizationLoss()
OnlineReparametrizationLoss(::AbstractMatrix, ::AbstractMatrix) = OnlineReparametrizationLoss()
OnlineReparametrizationLoss() = OnlineReparametrizationLoss(OnlineStatsBase.Mean(), OnlineStatsBase.CovMatrix())
OnlineStatsBase.fit!((;ljac, cov)::OnlineReparametrizationLoss, obs) = map(OnlineStatsBase.fit!, (ljac, cov), (obs[1], [obs[2], obs[3]]))
reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss; w1=0, w2=1-w1) = (
    w1 * (-mean(ljac) + .5 * log(Statistics.cov(cov)[1, 1])) + w2 * Statistics.cor(cov)[1, 2]
)
scale_estimate(orl::OnlineReparametrizationLoss) = begin
    c = Statistics.cov(orl.cov)
    (c[1,1] / c[2,2])^.25
end

# --- OnlineReparametrizer: fits multiple candidates ---

struct OnlineReparametrizer{P}
    pairs::P
end
OnlineStatsBase.fit!((;pairs)::OnlineReparametrizer, args...) = for (candidate, accumulator) in pairs
    OnlineStatsBase.fit!(accumulator, reparam(candidate, args...))
end
OnlineStatsBase.nobs((;pairs)::OnlineReparametrizer) = length(pairs) == 0 ? 0 : OnlineStatsBase.nobs(pairs[1][2])
minimizer((;pairs)::OnlineReparametrizer; kwargs...) = argmin(p -> reparametrization_loss(last(p); kwargs...), pairs)
scale_estimate(or::OnlineReparametrizer; kwargs...) = scale_estimate(last(minimizer(or; kwargs...)))

reparametrization_candidates(::PartiallyCentered; n=11) = Iterators.map(PartiallyCentered, range(0, 1, n))
OnlineReparametrizer((;source)::Reparametrization, xg...; kwargs...) = OnlineReparametrizer(source, xg...; kwargs...)
OnlineReparametrizer(source::PartiallyCentered, xg...; kwargs...) = OnlineReparametrizer([
    target => OnlineReparametrizationLoss(xg...)
    for target in reparametrization_candidates(source; kwargs...)
])

# --- Batch optimization ---

optimize!((;pairs)::IndexedReparametrization, xg::AbstractMatrix...; loss_kwargs=(;), kwargs...) = begin
    pairs .= Base.broadcasted(pairs) do (idx, value)
        or = OnlineReparametrizer(value, xg...; kwargs...)
        for xgi in zip(eachcol.(xg)...)
            OnlineStatsBase.fit!(or, reparam_rargs(value, getindex.(xgi, idx)..., first(xgi))...)
        end
        OnlineStatsBase.nobs(or) > 2 || return idx => value
        new_value = Reparametrization(value.target, first(minimizer(or; loss_kwargs...)), value.args...)
        trans = Reparametrization(new_value.source, value.source, value.args...)
        for xgi in zip(eachcol.(xg)...)
            setindex!.(xgi, reparam(trans, getindex.(xgi, idx)..., first(xgi))[2:end], idx)
        end
        idx => new_value
    end
    IndexedReparametrization(pairs)
end

# --- Online fitting for IndexedReparametrization ---

OnlineReparametrizer((;pairs)::IndexedReparametrization; kwargs...) = OnlineReparametrizer([
    idx => OnlineReparametrizer(value; kwargs...)
    for (idx, value) in pairs
])
OnlineStatsBase.fit!(ir::IndexedReparametrization, ors::OnlineReparametrizer, xg::AbstractMatrix...; loss_kwargs=(;), kwargs...) = begin
    ir.pairs .= Base.broadcasted(ir.pairs, ors.pairs) do (idx, value), (_, or)
        for xgi in zip(eachcol.(xg)...)
            OnlineStatsBase.fit!(or, reparam_rargs(value, getindex.(xgi, idx)..., first(xgi))...)
        end
        OnlineStatsBase.nobs(or) > 2 || return idx => value
        new_value = Reparametrization(value.target, first(minimizer(or; loss_kwargs...)), value.args...)
        trans = Reparametrization(new_value.source, value.source, value.args...)
        for xgi in zip(eachcol.(xg)...)
            setindex!.(xgi, reparam(trans, getindex.(xgi, idx)..., first(xgi))[2:end], idx)
        end
        idx => new_value
    end
    ir
end

# --- Hooks for adaptive_warmup_mcmc ---

find_reparametrization!(lpdf, halo_position, halo_gradient, position_and_gradient) = begin
    ir = reparametrizer(lpdf)
    isempty(ir.pairs) && return position_and_gradient
    optimize!(ir, halo_position, halo_gradient)
    DynamicHMC.evaluate_ℓ(lpdf, position_and_gradient.q; strict=false)
end

reparametrize!(lpdf, posterior_position) = begin
    ir = reparametrizer(lpdf)
    isempty(ir.pairs) && return
    inv_ir = inverse(ir)
    for col in eachcol(posterior_position)
        ljac, y = inv_ir(col)
        col .= y
    end
end
