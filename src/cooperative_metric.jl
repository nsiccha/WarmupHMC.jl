# Geometry is fitted once from the pre-restart state at a batch barrier. Only
# the scale actually installed on a chain survives the barrier and is saved.

# Allocate the exact available budget in one pass. Round cumulative quotas so
# hundreds of small responsibilities cannot each round to zero (or one). A
# capacity constraint reserves enough places to finish with the remaining donors.
function _halo_quotas(weights, capacities, budget)
    remaining = sum((h for (w, h) in zip(weights, capacities) if w > 0); init=0)
    goal = min(budget, remaining)
    quotas = zeros(Int, length(weights))
    iszero(goal) && return quotas
    total_weight = sum(w for (w, h) in zip(weights, capacities) if h > 0)
    cumulative, used = 0.0, 0
    for i in eachindex(quotas)
        w, h = weights[i], capacities[i]
        w > 0 && h > 0 || continue
        cumulative += w
        remaining -= h
        requested = round(Int, goal * (cumulative / total_weight)) - used
        quotas[i] = clamp(requested, max(0, goal - used - remaining), min(h, goal - used))
        used += quotas[i]
    end
    quotas
end

function _cluster_sample(chains, weights, capacities, budget; within_target=false)
    quotas = _halo_quotas(weights, capacities, budget)
    p = Matrix{Float64}(undef, first(chains).dimension, sum(quotas))
    g = similar(p)
    offset = 0
    for (c, h, m) in zip(chains, capacities, quotas)
        iszero(m) && continue
        # A singleton takes the midpoint. Larger quotas span the retained ring
        # without duplicating any leaf. Copy into buffers owned by this fit:
        # update_loss! rotates both positions and gradients in place.
        cols = m == 1 ? (cld(h, 2):cld(h, 2)) : round.(Int, range(1, h; length=m))
        dest = offset+1:offset+m
        @views p[:, dest] .= c.recording_lpdf.halo_position[:, cols]
        @views g[:, dest] .= c.recording_lpdf.halo_gradient[:, cols]
        # Family members have different targets. Remove each donor's own
        # mean, including for a single selected leaf from a longer history.
        if within_target
            @views p[:, dest] .-= c.moments.position_mean
            @views g[:, dest] .-= c.moments.gradient_mean
        end
        offset += m
    end
    p, g
end

function _fit_cluster_factor!(p, g, max_reflections)
    d, n = size(p)
    # Two observations per searched direction, and at most one reflection per
    # pair of observations. A bounded search needs no full-rank d × d estimate.
    n >= 2 * min(d, _DEFAULT_N_DIRECTIONS) && max_reflections > 0 || return nothing
    L = MatrixFactorization(SuccessiveReflections(d), Diagonal(ones(d)))
    # Reject unresolved/constant spreads before passing such a block to TSVD.
    isfinite(update_loss!(L.m2, p, g)) || return nothing
    loss = update_loss!(L, p, g; max_reflections=min(max_reflections, n ÷ 2))
    isfinite(loss) && !isempty(L.m1.reflections) ? L : nothing
end

function _cluster_geometry(clusters, chains, cfg; eligible=nothing, within_target=false)
    current = map(_coop_geometry_current, chains)
    donors = isnothing(eligible) ? current : (current .& eligible)
    R = clusters.responsibilities
    # Keep the receiver responsibilities even if that chain cannot donate.
    stats = if all(donors) && !within_target
        clusters.stats
    else
        [_combine_moments!(CoopMoments(first(chains).dimension), (c.moments for c in chains),
                           view(R, :, k) .* donors; within_target) for k in axes(R, 2)]
    end
    factors = Vector{Union{Nothing,ScaleThenReflect{Float64}}}(nothing, size(R, 2))
    if cfg.max_reflections > 0
        capacities = map(eachindex(chains)) do i
            c = chains[i]
            usable = donors[i] && (!within_target || c.moments.weight > 0)
            usable ? size(c.recording_lpdf.halo_position, 2) : 0
        end
        _parallel_foreach(axes(R, 2), cfg.n_cores) do k
            p, g = _cluster_sample(chains, view(R, :, k), capacities, cfg.recording_target; within_target)
            factors[k] = _fit_cluster_factor!(p, g, cfg.max_reflections)
        end
    end
    (; chains, responsibilities=R, stats, factors, within_target)
end

# Ordinary adaptation pools diagonal evidence over the receiver's memberships
# and takes reflections from its dominant cluster. Rescue may name one donor
# cluster explicitly, irrespective of the receiver's responsibility there.
function _cluster_metric(geometry, i)
    weights = view(geometry.responsibilities, i, :)
    current = geometry.chains[i].scale
    estimate = _combine_moments!(CoopMoments(size(current, 1)), geometry.stats, weights; geometry.within_target)
    _cluster_metric_choice(current, geometry.factors[argmax(weights)], estimate)
end
_cluster_metric(geometry, i, k) =
    _cluster_metric_choice(geometry.chains[i].scale, geometry.factors[k], geometry.stats[k])

# All shared-target rescues use the same measured donors. In a family a
# receiving slot has at most eight nearby donors. Cache each distinct fit for
# this barrier so even simultaneous rescues use only pre-restart evidence.
function _rescue_metric(geometry, i, donor, cfg, factors)
    factor = if cfg.max_reflections > 0
        key = cfg.family ? (donor.cluster, Tuple(donor.donors)) : donor.cluster
        get!(factors, key) do
            chains = view(geometry.chains, donor.donors)
            weights = view(geometry.responsibilities, donor.donors, donor.cluster)
            capacities = [size(c.recording_lpdf.halo_position, 2) for c in chains]
            p, g = _cluster_sample(chains, weights, capacities, cfg.recording_target;
                                   within_target=cfg.family)
            _fit_cluster_factor!(p, g, cfg.max_reflections)
        end
    end
    _cluster_metric_choice(geometry.chains[i].scale, factor, donor.moments)
end

function _cluster_metric_choice(current, factor, estimate)
    if isnothing(factor)
        scales = _moment_scales(estimate)
        # With no measured coordinate, a fitted shape still in use has no
        # challenger. A step-size-only restart may reuse the same scale.
        any(isfinite, scales) || return (; scale=current, weight=estimate.weight, relative_condition=1.0)
        scale = _complete_diagonal(scales, current)
        # The diagonal-only rule judges only coordinates with evidence. Filling
        # an unknown coordinate with its old value must not add a ratio of one.
        cc = current isa Diagonal ? _relative_condition(scales, current) :
                                    _relative_metric_condition(current, scale)
    else
        scale = factor
        cc = _relative_metric_condition(current, scale)
    end
    (; scale, weight=estimate.weight, relative_condition=cc)
end

function _complete_diagonal(scales, current)
    all(isfinite, scales) && return Diagonal(scales)
    fallback = current isa Diagonal ? current.diag : sqrt.(_metric_variances(current))
    Diagonal([isfinite(s) ? s : old for (s, old) in zip(scales, fallback)])
end

_metric_variances(L::Diagonal) = abs2.(L.diag)

# diag(L L') in the chain's coordinates. The principal scales are in the
# rotated coordinates and can put a missing coordinate's variance on the wrong
# axis. Conjugating C by a Householder I - 2vv' adds 4v .* ((v'Cv)v - Cv) to
# its diagonal. Evaluate Cv through the suffix of reflections already applied.
function _metric_variances(L::ScaleThenReflect)
    R = L.m1
    variances = abs2.(L.m2.diag)
    diagonal = copy(variances)
    floor = minimum(variances)
    v, z = similar(diagonal), similar(diagonal)
    for k in reverse(eachindex(R.idxs))
        idx, r = R.idxs[k], R.reflections[k]
        fill!(v, 0)
        v[idx] .= r
        suffix = k+1:length(R.idxs)
        _apply_reflections!(z, R, v, suffix)
        z .*= variances
        _apply_reflections!(z, R, z, reverse(suffix))
        q = dot(v, z)
        for (j, u) in zip(idx, r)
            diagonal[j] = max(floor, diagonal[j] + 4u * (q * u - z[j]))
        end
    end
    diagonal
end
