"""
    fit_gaussian_moments(draws; weights, max_reflections=16, maxiter=100, progress=nothing)

Fit a Gaussian moment summary to finite real `d × N` draws, one observation per
column. Return `(; mean, scale, diagnostics)`, all plain data, with newly owned
arrays and no retained draws. `scale_from_data(summary.scale)` reconstructs a
factor `L` of the approximate covariance `L * L'`. Inputs are never modified.

`weights` is a length-`N` vector of nonnegative finite masses with positive sum;
it is normalized internally. Moments are population moments of this discrete
distribution, **without** a sample-size or effective-sample-size correction.
Both inputs are converted to `Float64`; unrepresentable moments are rejected.

The fit learns up to `min(max_reflections, d-1)` leading covariance directions
by bounded power iteration, represented as successive Householder reflections.
It then matches the weighted marginal variances in that rotated frame. No
gradients, pilot, density evaluations, or sampler adaptation are involved.
`max_reflections=0` fits the original coordinatewise variances.

The weighted centered cloud is copied once. At fixed reflection and iteration
bounds, work and temporary storage are linear in both `d` and `N`: work is
O(d N (1 + max_reflections * maxiter)), workspace O(d N + d k + N), and returned
storage O(d (1 + k)), where `k ≤ max_reflections`. No dense covariance is formed.
A private fixed random stream initializes the iterations; the caller's random
stream is untouched and repeated calls with the same inputs are deterministic.
`progress` accepts a Treebars parent or backend, with counters for the fitted
directions and the current direction's iterations.

`diagnostics` contains `reflections`, per-direction `iterations`,
`relative_residuals`, `converged`, and `effective_sample_size`. `converged`
means each fitted direction's relative eigen-residual is at most `1e-8`;
it does **not** certify covariance accuracy or that a leading eigenvector was
found. Reaching `maxiter` returns the bounded approximation with
`converged=false`, not an unreported retry. There is no retained iteration trace.

A finite reflection budget discards remaining cross-covariances, even for a
Gaussian cloud. Close eigenvalues can require more iterations, and leading
directions can miss narrow correlations. This is an empirical moment
approximation, not an accuracy guarantee or a general posterior density fit.
The mean is fitted independently of the reflection budget.

Zero marginal spread or a fitted direction whose spread is at floating-point
roundoff is an `ArgumentError`, not a hidden variance floor. The roundoff test
applies only after reflections and uses `8eps(Float64) * norm(weighted_cloud)`
per reflection in the normalized fitting frame. No regularization is added.
"""
function fit_gaussian_moments(draws::AbstractMatrix{<:Real};
                              weights::AbstractVector{<:Real},
                              max_reflections::Integer=_DEFAULT_MAX_REFLECTIONS,
                              maxiter::Integer=100, progress=nothing)
    Base.require_one_based_indexing(draws, weights)
    d, n = size(draws)
    d > 0 && n > 0 || throw(ArgumentError("draws must have at least one row and column"))
    length(weights) == n || throw(DimensionMismatch("weights must have one entry per draw ($n)"))
    max_reflections >= 0 || throw(ArgumentError("max_reflections must be nonnegative"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))

    w = Vector{Float64}(weights)
    all(x -> isfinite(x) && x >= 0, w) || throw(ArgumentError("weights must be finite and nonnegative"))
    largest_weight = maximum(w)
    largest_weight > 0 || throw(ArgumentError("weights must have positive total mass"))
    # Scaling before summing also accepts masses whose unnormalized sum overflows.
    w ./= largest_weight
    w ./= sum(w)
    effective_sample_size = inv(sum(abs2, w))
    cloud = Matrix{Float64}(draws)
    all(isfinite, cloud) || throw(ArgumentError("draws must be finite and representable in Float64"))

    # Center about an observed point before taking the mean: large common offsets
    # do not consume the precision of the spread. Only the owned cloud is written.
    anchor = findfirst(>(0), w)
    location = copy(view(cloud, :, anchor))
    for j in axes(cloud, 2)
        if iszero(w[j])
            fill!(view(cloud, :, j), 0)
        else
            view(cloud, :, j) .-= location
        end
    end
    offset = cloud * w
    location .+= offset
    for j in axes(cloud, 2)
        iszero(w[j]) || (view(cloud, :, j) .-= offset)
    end
    all(isfinite, location) && all(isfinite, cloud) || throw(ArgumentError(
        "centered moments are not representable in Float64"))
    magnitude = maximum(abs, cloud)
    magnitude > 0 || throw(ArgumentError("the weighted cloud has zero spread"))
    cloud ./= magnitude
    for j in axes(cloud, 2)
        view(cloud, :, j) .*= sqrt(w[j])
    end
    # A distant draw with tiny probability can set the unweighted scale above.
    # Rescale after weighting too, so covariance products do not square a cloud
    # whose entries are all tiny. Only this global unit changes, not the moments.
    weighted_magnitude = maximum(abs, cloud)
    weighted_magnitude > 0 || throw(ArgumentError("the weighted cloud has zero spread"))
    cloud ./= weighted_magnitude
    magnitude *= weighted_magnitude
    all(i -> norm(view(cloud, i, :)) > 0, 1:d) || throw(ArgumentError(
        "the weighted cloud has a coordinate with zero spread"))

    indices = Vector{Int}[]
    reflections = Vector{Float64}[]
    iterations = Int[]
    relative_residuals = Float64[]
    v, cv, residual_buffer, u = zeros(d), zeros(d), zeros(d), zeros(n)
    rng = Random.Xoshiro(0)
    tolerance = 1e-8
    cloud_norm = norm(cloud)
    @progress progress "Moment directions" for first_coordinate in 1:min(max_reflections, d - 1)
        remaining = first_coordinate:d
        block = view(cloud, remaining, :)
        direction, product = view(v, remaining), view(cv, remaining)
        error = view(residual_buffer, remaining)
        Random.randn!(rng, direction)
        normalize!(direction)
        residual, steps = Inf, 0
        @progress "Power iterations" for iteration in 1:maxiter
            mul!(u, block', direction)
            mul!(product, block, u)
            λ = dot(direction, product)
            product_norm = norm(product)
            product_norm > 0 || throw(ArgumentError(
                "the weighted cloud has a numerically unresolved covariance direction"))
            # Evaluate the residual on the same direction that will be retained.
            @. error = product - λ * direction
            residual = norm(error) / product_norm
            steps = iteration
            (residual <= tolerance || iteration == maxiter) && break
            direction .= product ./ product_norm
        end
        push!(iterations, steps)
        push!(relative_residuals, residual)

        # H v = -sign(v[1]) e₁, with the sign that avoids cancellation. Deflation
        # excludes that coordinate on the next pass; no d × d Q is constructed.
        reflector = copy(direction)
        reflector[1] += copysign(1.0, reflector[1])
        normalize!(reflector)
        push!(indices, collect(remaining))
        push!(reflections, reflector)
        mul!(u, block', reflector)
        @progress nothing for j in axes(block, 2), i in axes(block, 1)
            block[i, j] -= 2 * reflector[i] * u[j]
        end
    end

    k = length(reflections)
    roundoff = 8eps(Float64) * k * cloud_norm
    diagonal = [norm(view(cloud, i, :)) for i in 1:d]
    all(>(roundoff), diagonal) || throw(ArgumentError(
        "a fitted direction has zero or numerically unresolved weighted spread; no variance floor was added"))
    diagonal .*= magnitude
    all(_usable_scale, diagonal) || throw(ArgumentError(
        "fitted scales are not finite positive Float64 values"))
    scale = (; diagonal, indices, reflections)
    diagnostics = (; reflections=k, iterations, relative_residuals,
                   converged=all(<=(tolerance), relative_residuals), effective_sample_size)
    (; mean=location, scale, diagnostics)
end
