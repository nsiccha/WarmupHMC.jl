# Weighted running marginal-scale estimator for `adaptive_warmup_mcmc`'s
# alternative linear restart sources.

_positive_or_nan(x) = x > 0 ? x : oftype(x, NaN)

# ---------------------------------------------------------------------------
# General-weight running marginal-scale estimator.
#
# Records both W=sum(w) and W₂=sum(w²), so its covariance correction remains
# valid for exact NUTS proposal probabilities and trajectory reliability/exposure
# weights. `adaptive_warmup_mcmc`'s non-halo linear restart sources use it.
# ---------------------------------------------------------------------------
mutable struct WeightedScaleAdaptation{T}
    position_mean::Vector{T}
    position_m2::Vector{T}
    gradient_mean::Vector{T}
    gradient_m2::Vector{T}
    weight::T
    weight2::T
end
WeightedScaleAdaptation(dim::Integer; T=Float64) = WeightedScaleAdaptation{T}(
    zeros(T, dim), zeros(T, dim), zeros(T, dim), zeros(T, dim), zero(T), zero(T),
)

OnlineStatsBase.nobs(a::WeightedScaleAdaptation) = a.weight
effective_sample_size(a::WeightedScaleAdaptation) =
    iszero(a.weight2) ? zero(a.weight) : abs2(a.weight) / a.weight2

function OnlineStatsBase.fit!(a::WeightedScaleAdaptation,
                              position::AbstractVector,
                              gradient::AbstractVector; dw=1)
    dw >= 0 || throw(ArgumentError("observation weight must be nonnegative, got $dw"))
    length(position) == length(a.position_mean) || throw(DimensionMismatch(
        "position has length $(length(position)); expected $(length(a.position_mean))",
    ))
    length(gradient) == length(a.gradient_mean) || throw(DimensionMismatch(
        "gradient has length $(length(gradient)); expected $(length(a.gradient_mean))",
    ))
    iszero(dw) && return a
    new_weight = a.weight + dw
    fraction = dw / new_weight
    @inbounds for i in eachindex(position, gradient)
        position_delta = position[i] - a.position_mean[i]
        a.position_mean[i] += fraction * position_delta
        a.position_m2[i] += dw * position_delta * (position[i] - a.position_mean[i])

        gradient_delta = gradient[i] - a.gradient_mean[i]
        a.gradient_mean[i] += fraction * gradient_delta
        a.gradient_m2[i] += dw * gradient_delta * (gradient[i] - a.gradient_mean[i])
    end
    a.weight = new_weight
    a.weight2 += abs2(dw)
    a
end

function _weighted_variances(a::WeightedScaleAdaptation)
    if !(a.weight > 0)
        nan = oftype(a.weight, NaN)
        return (; position=fill(nan, length(a.position_mean)),
                  gradient=fill(nan, length(a.gradient_mean)))
    end
    denominator = a.weight - a.weight2 / a.weight
    if !(denominator > 0)
        nan = oftype(a.weight, NaN)
        return (; position=fill(nan, length(a.position_mean)),
                  gradient=fill(nan, length(a.gradient_mean)))
    end
    position = _positive_or_nan.(a.position_m2 ./ denominator)
    gradient = _positive_or_nan.(a.gradient_m2 ./ denominator)
    (; position, gradient)
end

"General-weight diagonal scales `(var_pos/var_grad)^(1/4)`."
marginal_scales(a::WeightedScaleAdaptation) = begin
    (; position, gradient) = _weighted_variances(a)
    (position ./ gradient) .^ 0.25
end

function reset!(a::WeightedScaleAdaptation)
    fill!(a.position_mean, 0)
    fill!(a.position_m2, 0)
    fill!(a.gradient_mean, 0)
    fill!(a.gradient_m2, 0)
    a.weight = 0
    a.weight2 = 0
    a
end
