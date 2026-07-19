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
