# Pool independent streams in chain order, writing only the barrier's private
# recorder. Candidate moments combine exactly up to floating-point rounding.
function _merge_nonlinear!(a::WeightedReparametrizationLoss, b::WeightedReparametrizationLoss, rng)
    a.groups += b.groups
    iszero(b.weight) && return a
    if iszero(a.weight)
        for field in (:weight, :weight2, :mean_ljac, :mean_position, :mean_gradient,
                      :m2_position, :m2_gradient, :co_position_gradient)
            setfield!(a, field, getfield(b, field))
        end
        return a
    end
    total = a.weight + b.weight
    fraction = b.weight / total
    cross_weight = a.weight * fraction
    dx = b.mean_position - a.mean_position
    dg = b.mean_gradient - a.mean_gradient
    a.m2_position += b.m2_position + cross_weight * dx * dx
    a.m2_gradient += b.m2_gradient + cross_weight * dg * dg
    a.co_position_gradient += b.co_position_gradient + cross_weight * dx * dg
    a.mean_ljac += fraction * (b.mean_ljac - a.mean_ljac)
    a.mean_position += fraction * dx
    a.mean_gradient += fraction * dg
    a.weight = total
    a.weight2 += b.weight2
    a
end

function _merge_nonlinear!(a::OnlineReparametrizer, b::OnlineReparametrizer, rng)
    for i in eachindex(a.pairs, b.pairs)
        key, dest = a.pairs[i]
        other, src = b.pairs[i]
        key == other || throw(ArgumentError("cannot pool different nonlinear candidates or indices: $key and $other"))
        _merge_nonlinear!(dest, src, rng)
    end
    a
end

# Each slot already represents a weighted draw within its stream. Select the
# source independently for each slot, proportional to the two streams' masses.
function _merge_nonlinear!(a::WeightedLeafSample, b::WeightedLeafSample, rng)
    iszero(b.total) && return a
    size(a.position) == size(b.position) || throw(DimensionMismatch("nonlinear leaf sample sizes differ"))
    total = a.total + b.total
    p = b.total / total
    for j in axes(a.position, 2)
        if iszero(a.total) || rand(rng) < p
            a.position[:, j] .= view(b.position, :, j)
            a.gradient[:, j] .= view(b.gradient, :, j)
        end
    end
    a.total = total
    a
end

function _merge_nonlinear!(a::ChainEvidence, b::ChainEvidence, rng)
    _merge_nonlinear!(a.pairs, b.pairs, rng)
    _merge_nonlinear!(a.sample, b.sample, rng)
    a
end
