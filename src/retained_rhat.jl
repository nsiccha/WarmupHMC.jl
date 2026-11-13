# Split rank-normalized R-hat for unequal retained histories. MCMCDiagnosticTools
# accepts rectangular arrays, so using it on a common tail would omit precisely
# the transients this check must see. The variance estimator is the equal-chain-
# weight generalization in Stan 2.6.2, "Generalized R-hat for Ragged Chains":
# https://statmodeling.stat.columbia.edu/wp-content/uploads/2015/04/stan-reference-2.6.2.pdf#page=420
#   W = mean(s²_j), V = mean((n_j - 1)/n_j * s²_j) + var(mean_j).
# It agrees with ordinary split R-hat at equal even lengths. Odd histories split
# into floor(n/2) and ceil(n/2), retaining the middle draw too. Ranking and folding
# use every retained draw; neither the histories nor sampler state are mutated.

function _ragged_rhat(values, segments)
    means = [mean(view(values, r)) for r in segments]
    variances = [var(view(values, r)) for r in segments]
    within = mean(variances)
    within > 0 || return Inf  # Constant histories cannot establish mixing.
    total = mean((1 - inv(length(r))) * v for (r, v) in zip(segments, variances)) + var(means)
    sqrt(total / within)
end

function _rank_normalize_retained(values)
    ranks = StatsBase.tiedrank(values)
    n = length(values)
    map!(r -> quantile(Normal(), (r - 3 / 8) / (n + 1 / 4)), ranks, ranks)
end

function _retained_rhat(draws)
    holding = filter(d -> size(d, 2) > 0, draws)
    isempty(holding) && return Inf
    # Do not silently exclude a short chain that will be returned to the caller.
    any(d -> size(d, 2) <= 10, holding) && return Inf
    dimension = size(first(holding), 1)
    all(d -> size(d, 1) == dimension, holding) || throw(DimensionMismatch("chain dimensions differ"))
    segments = UnitRange{Int}[]
    offset = 0
    for d in holding
        n = size(d, 2)
        push!(segments, offset + 1:offset + n ÷ 2, offset + n ÷ 2 + 1:offset + n)
        offset += n
    end
    # Work one coordinate at a time: O(total draws + chains) temporary storage,
    # O(d * total draws * log(total draws)) time, with no chain-pair comparisons.
    values = Vector{Float64}(undef, offset)
    worst = 0.0
    for i in 1:dimension
        offset = 0
        for d in holding
            n = size(d, 2)
            copyto!(view(values, offset + 1:offset + n), view(d, i, :))
            offset += n
        end
        all(isfinite, values) || return Inf
        bulk = _ragged_rhat(_rank_normalize_retained(values), segments)
        center = median(values)
        map!(x -> abs(x - center), values, values)
        tail = _ragged_rhat(_rank_normalize_retained(values), segments)
        worst = max(worst, bulk, tail)
        isfinite(worst) || return Inf
    end
    worst
end
