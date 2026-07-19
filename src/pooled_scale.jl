# Pooled mass-matrix estimation primitives for the cooperative samplers.
#
# Chains that sample the same geometry can POOL their draws into ONE mass-matrix
# estimate; the pool is built "carefully but optimistically" by the clustering
# driver, which optimistically merges all chains and then carefully `unmerge!`s
# the least-compatible ones. These are the shared, mergeable/unmergeable
# estimators + compatibility metric that make that possible.
#
# Reimplemented cleanly (plain Julia) from the buried
# `cooperative-clusters:src/scale_adaptation.jl`, keeping its math:
#   * Welford weighted online mean/variance (`Variances`),
#   * Chan's parallel-variance `merge!` and its exact inverse `unmerge!`,
#   * shrinkage toward a prior (`RegularizedVariances`),
#   * the nutpie marginal scale `(var_pos/var_grad)^(1/4)` — which coincides
#     with `adaptive_warmup_mcmc`'s `sqrt(std(pos)/std(grad))`,
#   * a condition-number compatibility metric, exposed as a tuneable knob.
# Decisions: estimator `lnmhbn` (Nutpie), metric `y4o8i` ("make tuneable").

_smooth(a, b, w) = a + w * (b - a)
_unsmooth(ab, b, w) = (ab - w * b) / (1 - w)   # inverse of _smooth in its first arg

# ---------------------------------------------------------------------------
# Welford weighted online mean + variance, with a mergeable/unmergeable state.
# `squared_scale` is the weighted mean of squared deviations (biased variance);
# `var` applies the Bessel correction.
# ---------------------------------------------------------------------------
mutable struct Variances{T}
    location::Vector{T}       # running mean per dimension
    squared_scale::Vector{T}  # running biased variance per dimension
    n::Base.RefValue{T}       # total (possibly fractional) weight seen
end
Variances(dim::Int; T=Float64) = Variances{T}(zeros(T, dim), zeros(T, dim), Ref(zero(T)))

OnlineStatsBase.nobs(v::Variances) = v.n[]
Base.length(v::Variances) = length(v.location)
Base.eachindex(v::Variances) = eachindex(v.location)
Statistics.mean(v::Variances) = v.location

OnlineStatsBase.fit!(v::Variances, y::AbstractVector; dw=1) = begin
    @assert dw >= 0 dw
    n = v.n[] += dw
    w = dw / n
    @inbounds for i in eachindex(y)
        yi = y[i]
        preloc = v.location[i]
        postloc = v.location[i] = _smooth(preloc, yi, w)
        v.squared_scale[i] = _smooth(v.squared_scale[i], (yi - preloc) * (yi - postloc), w)
    end
    v
end

_positive_or_nan(x) = x > 0 ? x : oftype(x, NaN)
# Bessel-corrected variance; NaN where non-positive (too few obs / degenerate).
Statistics.var(v::Variances) = begin
    n = v.n[]
    b = n / (n - 1)
    _positive_or_nan.(v.squared_scale .* b)
end

# Chan's parallel combination: fold rgs... into `a` in one pass.
Base.merge!(a::Variances, rgs::Variances...) = begin
    args = (a, rgs...)
    ns = OnlineStatsBase.nobs.(args)
    # weight of each rg against the running total accumulated so far
    w2s = ((n, cum) -> n == 0 ? zero(n) : n / (n + cum)).(ns[2:end], cumsum(collect(ns[1:end-1])))
    @inbounds for i in eachindex(a)
        loc = a.location[i]
        ss = a.squared_scale[i]
        for (w2, arg) in zip(w2s, rgs)
            w1 = 1 - w2
            dloc = arg.location[i] - loc
            loc = _smooth(loc, arg.location[i], w2)
            ss = _smooth(ss, arg.squared_scale[i], w2) + dloc^2 * w2 * w1
        end
        a.location[i] = loc
        a.squared_scale[i] = ss
    end
    a.n[] = sum(ns)
    a
end

# Exact inverse of `merge!(a, arg)`: remove `arg`'s contribution from `a`.
unmerge!(a::Variances, arg::Variances) = begin
    @assert OnlineStatsBase.nobs(a) >= OnlineStatsBase.nobs(arg)
    w1 = (a.n[] - arg.n[]) / a.n[]
    w2 = 1 - w1
    @inbounds for i in eachindex(a)
        loc = a.location[i]
        loc2 = arg.location[i]
        a.location[i] = _unsmooth(loc, loc2, w2)
        a.squared_scale[i] = _unsmooth(a.squared_scale[i], arg.squared_scale[i], w2) - ((loc2 - loc) / w1)^2 * w2
    end
    a.n[] -= arg.n[]
    a
end

Base.copy!(dest::Variances, src::Variances) = begin
    copy!(dest.location, src.location)
    copy!(dest.squared_scale, src.squared_scale)
    dest.n[] = src.n[]
    dest
end
reset!(v::Variances) = (fill!(v.location, 0); fill!(v.squared_scale, 0); v.n[] = 0; v)

# ---------------------------------------------------------------------------
# Variance estimate shrunk toward a prior `regularizing_var` with pseudo-count
# `regularizing_n` — stabilises early/low-count estimates.
# ---------------------------------------------------------------------------
struct RegularizedVariances{T}
    regularizing_n::T
    regularizing_var::T
    variances::Variances{T}
end
RegularizedVariances(dim::Int; T=Float64, regularizing_n=5.0, regularizing_var=1e-3) =
    RegularizedVariances{T}(regularizing_n, regularizing_var, Variances(dim; T))

OnlineStatsBase.nobs(v::RegularizedVariances) = OnlineStatsBase.nobs(v.variances)
Base.length(v::RegularizedVariances) = length(v.variances)
OnlineStatsBase.fit!(v::RegularizedVariances, y::AbstractVector; kwargs...) = (OnlineStatsBase.fit!(v.variances, y; kwargs...); v)
Statistics.var((; regularizing_n, regularizing_var, variances)::RegularizedVariances) = begin
    n = OnlineStatsBase.nobs(variances)
    w = n / (n + regularizing_n)
    _smooth.(regularizing_var, var(variances), w)
end
Base.merge!(a::RegularizedVariances, rgs::RegularizedVariances...) = (merge!(a.variances, getfield.(rgs, :variances)...); a)
unmerge!(a::RegularizedVariances, arg::RegularizedVariances) = (unmerge!(a.variances, arg.variances); a)
Base.copy!(dest::RegularizedVariances, src::RegularizedVariances) = (copy!(dest.variances, src.variances); dest)
reset!(v::RegularizedVariances) = (reset!(v.variances); v)

# ---------------------------------------------------------------------------
# Nutpie mass-matrix estimator (decision `lnmhbn`): combines position AND
# gradient variances. Marginal (diagonal) scale = (var_pos / var_grad)^(1/4),
# which equals `adaptive_warmup_mcmc`'s `sqrt(std(pos)/std(grad))`.
# Pool chains by `merge!`; drop a chain from a pool by `unmerge!`.
# ---------------------------------------------------------------------------
struct NutpieScaleAdaptation{T}
    position_variances::RegularizedVariances{T}
    gradient_variances::RegularizedVariances{T}
end
NutpieScaleAdaptation(dim::Int; kwargs...) =
    NutpieScaleAdaptation(RegularizedVariances(dim; kwargs...), RegularizedVariances(dim; kwargs...))

OnlineStatsBase.nobs(a::NutpieScaleAdaptation) = OnlineStatsBase.nobs(a.position_variances)
Base.length(a::NutpieScaleAdaptation) = length(a.position_variances)
OnlineStatsBase.fit!(a::NutpieScaleAdaptation, position::AbstractVector, gradient::AbstractVector; kwargs...) = begin
    OnlineStatsBase.fit!(a.position_variances, position; kwargs...)
    OnlineStatsBase.fit!(a.gradient_variances, gradient; kwargs...)
    a
end
"Diagonal mass-matrix variances: `sqrt(var(position)/var(gradient))`."
marginal_variances(a::NutpieScaleAdaptation) = sqrt.(var(a.position_variances) ./ var(a.gradient_variances))
"Diagonal mass-matrix scales (std): `(var_pos/var_grad)^(1/4)`."
marginal_scales(a::NutpieScaleAdaptation) = sqrt.(marginal_variances(a))

Base.merge!(a::NutpieScaleAdaptation, rgs::NutpieScaleAdaptation...) = begin
    merge!(a.position_variances, getfield.(rgs, :position_variances)...)
    merge!(a.gradient_variances, getfield.(rgs, :gradient_variances)...)
    a
end
unmerge!(a::NutpieScaleAdaptation, arg::NutpieScaleAdaptation) = begin
    unmerge!(a.position_variances, arg.position_variances)
    unmerge!(a.gradient_variances, arg.gradient_variances)
    a
end
Base.copy!(dest::NutpieScaleAdaptation, src::NutpieScaleAdaptation) = begin
    copy!(dest.position_variances, src.position_variances)
    copy!(dest.gradient_variances, src.gradient_variances)
    dest
end
reset!(a::NutpieScaleAdaptation) = (reset!(a.position_variances); reset!(a.gradient_variances); a)

# Non-mutating pool of one or more adaptations into a fresh estimate.
pooled(a::NutpieScaleAdaptation, rgs::NutpieScaleAdaptation...) = begin
    rv = a.position_variances
    fresh = NutpieScaleAdaptation(length(a); regularizing_n=rv.regularizing_n, regularizing_var=rv.regularizing_var)
    merge!(fresh, a, rgs...)
end

# ---------------------------------------------------------------------------
# Compatibility — is a chain's estimate consistent with a pool's? (decision
# `y4o8i`: condition-number metric, exposed as a tuneable knob.) A metric is any
# `(a, b) -> Real` scoring how far apart two estimates' marginal scales are;
# `compatible` thresholds it. Swap `metric`/`threshold` to tune.
# ---------------------------------------------------------------------------
"Condition number of a positive vector: `maximum/minimum`."
diagonal_cond(x) = ((lo, hi) = extrema(x); hi / lo)
"Condition number of the ratio of two adaptations' marginal scales (1 ⇔ proportional)."
cond_compatibility(a, b) = diagonal_cond(marginal_scales(a) ./ marginal_scales(b))
"Are `a` and `b` compatible? `metric(a,b) <= threshold`. Both are tuneable."
compatible(a, b; metric=cond_compatibility, threshold=sqrt(2.0)) = metric(a, b) <= threshold
