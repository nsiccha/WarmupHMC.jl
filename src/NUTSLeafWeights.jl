mutable struct NUTSLeaves{T}
    position::ElasticMatrix{T,Vector{T}}
    gradient::ElasticMatrix{T,Vector{T}}
    dH::Vector{T}
    weights::Vector{T}
end

NUTSLeaves(dimension::Integer; T=Float64) = NUTSLeaves(
    ElasticMatrix{T,Vector{T}}(undef, dimension, 0),
    ElasticMatrix{T,Vector{T}}(undef, dimension, 0),
    T[],
    T[],
)

Base.length(leaves::NUTSLeaves) = length(leaves.dH)

function reset!(leaves::NUTSLeaves)
    reset!(leaves.position)
    reset!(leaves.gradient)
    empty!(leaves.dH)
    empty!(leaves.weights)
    leaves
end

function record_leaf!(leaves::NUTSLeaves, position, gradient, dH)
    append!(leaves.position, position)
    append!(leaves.gradient, gradient)
    push!(leaves.dH, dH)
    leaves
end

record_leaf!(leaves::NUTSLeaves, z, dH) =
    record_leaf!(leaves, z.Q.q, z.Q.∇ℓq, dH)

"""
    leaf_weights!(weights, dH, depth)

Compute the marginal proposal probability of every leaf visited by a
`DynamicHMC.sample_tree` traversal. `dH[1]` is the initial state and the next
`2^depth - 1` entries are the successfully doubled subtrees, in visitation
order. Any later entries came from the invalid final doubling and receive zero
weight.

Within a new subtree, DynamicHMC uses multinomial weights `exp(dH)`. At each
tree doubling it uses biased progressive sampling, switching to the new subtree
with probability `min(exp(log_weight_new - log_weight_previous), 1)`. This
routine marginalizes both choices. Including the initial state is essential:
its residual "stay" probability makes the returned weights sum to one.
"""
function leaf_weights!(weights::Vector{T}, dH::AbstractVector, depth::Integer) where {T<:AbstractFloat}
    depth >= 0 || throw(ArgumentError("tree depth must be nonnegative, got $depth"))
    n_valid = 1 << depth
    length(dH) >= n_valid || throw(DimensionMismatch(
        "tree depth $depth requires at least $n_valid leaves, got $(length(dH))",
    ))
    all(x -> isfinite(x) || x == -Inf, @view(dH[1:n_valid])) || throw(ArgumentError(
        "valid leaf log weights must be finite or -Inf",
    ))
    isfinite(dH[1]) || throw(ArgumentError("the initial leaf must have finite log weight"))

    resize!(weights, length(dH))
    fill!(weights, zero(T))
    weights[1] = one(T)

    log_previous = dH[1]
    first_new = 2
    for level in 0:(depth - 1)
        n_new = 1 << level
        new_idxs = first_new:(first_new + n_new - 1)
        log_new = logsumexp(@view dH[new_idxs])
        log_switch = min(log_new - log_previous, zero(log_new))
        switch_probability = exp(log_switch)
        stay_probability = -expm1(log_switch)

        @views weights[1:(first_new - 1)] .*= stay_probability
        if !iszero(switch_probability)
            @views @. weights[new_idxs] = switch_probability * exp(dH[new_idxs] - log_new)
        end

        log_previous = logaddexp(log_previous, log_new)
        first_new += n_new
    end
    weights
end

function finalize_leaf_weights!(leaves::NUTSLeaves, depth::Integer)
    leaf_weights!(leaves.weights, leaves.dH, depth)
    leaves
end

@views leaf_state(leaves::NUTSLeaves, i::Integer) = (
    position=leaves.position[:, i],
    gradient=leaves.gradient[:, i],
    dH=leaves.dH[i],
    is_initial=i == 1,
)

"""
    expected_stat(f, leaves)

Evaluate `f` on each leaf state and return its exact expectation under the
NUTS proposal probabilities computed by [`leaf_weights!`](@ref). Zero-weight
leaves from an invalid final doubling are not evaluated.
"""
function expected_stat(f, leaves::NUTSLeaves)
    first_positive = findfirst(>(zero(eltype(leaves.weights))), leaves.weights)
    isnothing(first_positive) && throw(ArgumentError("leaf weights have not been finalized"))
    result = leaves.weights[first_positive] * f(leaf_state(leaves, first_positive))
    for i in (first_positive + 1):length(leaves)
        weight = leaves.weights[i]
        iszero(weight) || (result += weight * f(leaf_state(leaves, i)))
    end
    result
end

function sample_leaf(rng, leaves::NUTSLeaves)
    threshold = rand(rng)
    cumulative = 0.0
    last_positive = nothing
    for (i, weight) in pairs(leaves.weights)
        if weight > 0
            last_positive = i
            cumulative += weight
            threshold < cumulative && return i
        end
    end
    isnothing(last_positive) && throw(ArgumentError("leaf weights have not been finalized"))
    last_positive
end
