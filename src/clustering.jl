# Cluster chains by mass-matrix compatibility, so each cluster pools its chains'
# draws into ONE estimate. This is the "carefully but optimistically" pool
# construction: optimistically merge all chains into one pool, then carefully
# `unmerge!` the least-compatible ones until the pool is self-consistent.
#
# Pool-construction strategy is decision `ofj104`: `assign_clusters` below is the
# greedy baseline (ported from cooperative-clusters:src/cooperative_warmup_mcmc.jl
# `cluster!`). A look-behind / retrospective strategy is a drop-in alternative —
# the standalone driver takes the clustering as a swappable function.

"""
    assign_clusters(adaptations; metric=cond_compatibility, threshold=sqrt(2)) -> Vector{Vector{Int}}

Greedy "carefully but optimistically" clustering of chains by mass-matrix
compatibility. Optimistically pool ALL remaining chains' `adaptations`
([`NutpieScaleAdaptation`](@ref)s) into one estimate, then repeatedly drop
(`unmerge!`) the least-compatible chain — the one with the highest `metric`
against the pool — while that score exceeds `threshold`. When every survivor is
within `threshold` they form a cluster; recurse on the dropped chains.

Returns a partition of `eachindex(adaptations)` into clusters, largest first.
`metric` / `threshold` are tuneable (decision `y4o8i`); the default
[`cond_compatibility`](@ref) at `√2` mirrors the old default.
"""
assign_clusters(adaptations::AbstractVector; metric=cond_compatibility, threshold=sqrt(2.0)) = begin
    clusters = Vector{Int}[]
    remaining = Set(eachindex(adaptations))
    while !isempty(remaining)
        candidates = sort!(collect(remaining))
        pool = pooled(adaptations[candidates]...)     # optimistic: merge everyone
        while true
            if length(candidates) == 1
                push!(clusters, copy(candidates))
                setdiff!(remaining, candidates)
                break
            end
            scores = [metric(adaptations[i], pool) for i in candidates]
            worst_pos = argmax(scores)
            if scores[worst_pos] <= threshold                 # all survivors compatible
                push!(clusters, copy(candidates))
                setdiff!(remaining, candidates)
                break
            else                                              # carefully drop the worst
                unmerge!(pool, adaptations[candidates[worst_pos]])
                deleteat!(candidates, worst_pos)
            end
        end
    end
    sort!(clusters; by=length, rev=true)
    clusters
end

# ---------------------------------------------------------------------------
# Look-behind (retrospective) clustering — the second strategy (decision
# `ofj104`: "include both"). Greedy above makes forward, never-reconsidered
# drops; look-behind starts from a partition and REVISITS every chain, moving it
# to the cluster it best fits. The scoring `criterion` is the swappable knob
# (decision `hwfj4p`, user: "try a few and see what sticks").
# ---------------------------------------------------------------------------

# A criterion scores chain `ai`'s fit to a candidate cluster whose OTHER members'
# adaptations are `others` (i itself excluded). Lower is better; the
# condition-number metric is ≥ 1, so 1.0 is a perfect fit.

"Leave-one-out: `ai` vs the pool of the cluster's OTHER members (no self-inclusion bias)."
loo_criterion(others::AbstractVector, ai; metric=cond_compatibility) = metric(ai, pooled(others...))
"Complete-linkage: `ai`'s WORST pairwise score against the cluster's other members."
linkage_criterion(others::AbstractVector, ai; metric=cond_compatibility) = maximum(aj -> metric(ai, aj), others)
"Inclusive: `ai` vs the pool INCLUDING itself (self-inclusion bias, most lenient)."
inclusive_criterion(others::AbstractVector, ai; metric=cond_compatibility) = metric(ai, pooled(ai, others...))

"""
    lookbehind_clusters(adaptations; metric=cond_compatibility, threshold=sqrt(2),
                        criterion=loo_criterion, init=assign_clusters, max_passes=10)

Retrospective ("look-behind") alternative to greedy [`assign_clusters`](@ref):
start from `init`'s partition, then repeatedly REVISIT every chain and move it to
the cluster it best fits — the LARGEST cluster it is within `threshold` of (the
chain excluded from that cluster's pool), scored by `criterion`, ties broken by
lower score. A chain that fits no existing cluster becomes its own singleton.
Unlike greedy's forward, never-reconsidered drops, this repairs chains that
greedy's processing order pulled into the wrong cluster.

`criterion` is the swappable scoring knob (decision `hwfj4p`): [`loo_criterion`](@ref)
(default, leave-one-out), [`linkage_criterion`](@ref) (complete-linkage), or
[`inclusive_criterion`](@ref). Same signature as `assign_clusters`, so it is a
drop-in `cluster_fn` for the sampler. Returns a partition, largest cluster first.
"""
lookbehind_clusters(adaptations::AbstractVector;
    metric=cond_compatibility, threshold=sqrt(2.0),
    criterion=loo_criterion, init=assign_clusters, max_passes=10) = begin
    n = length(adaptations)
    n == 0 && return Vector{Int}[]
    assign = Vector{Int}(undef, n)
    for (ci, members) in enumerate(init(adaptations; metric, threshold)), i in members
        assign[i] = ci
    end
    next_id = maximum(assign) + 1
    for _ in 1:max_passes
        changed = false
        for i in 1:n
            best_id = 0
            best_key = (0, 0.0)        # (n_others_in_cluster, -score): maximize size, then minimize score
            for cj in unique(assign)
                others = [adaptations[k] for k in 1:n if assign[k] == cj && k != i]
                isempty(others) && continue
                score = criterion(others, adaptations[i]; metric)
                score <= threshold || continue
                key = (length(others), -score)
                if best_id == 0 || key > best_key
                    best_id = cj; best_key = key
                end
            end
            new_id = if best_id != 0
                best_id                                    # fits an existing cluster → join the best
            elseif count(==(assign[i]), assign) == 1
                assign[i]                                  # already a singleton → stay put
            else
                (id = next_id; next_id += 1; id)           # fits nothing → break off into a fresh singleton
            end
            new_id == assign[i] || (assign[i] = new_id; changed = true)
        end
        changed || break
    end
    groups = [findall(==(cj), assign) for cj in unique(assign)]
    sort!(groups; by=length, rev=true)
    groups
end
