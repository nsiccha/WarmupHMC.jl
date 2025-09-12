struct NUTSPosterior{P,T} <: WrappedLogDensityProblem{P}
    posterior::P
    position::ElasticMatrix{T,Vector{T}}
    gradient::ElasticMatrix{T,Vector{T}}
    momentum::Matrix{T}
    dH::Vector{T}
    dangle::Vector{T}
    n_right::Ref{Int}
    c1::Vector{T}
    c2::Vector{T}
end
Base.parent(lpdf::NUTSPosterior) = lpdf.posterior
NUTSPosterior(lpdf) = begin
    n = LogDensityProblems.dimension(lpdf)
    NUTSPosterior(
        lpdf, 
        ElasticMatrix(zeros((n, 0))), ElasticMatrix(zeros((n, 0))), 
        zeros((n, 2)), zeros(0), zeros(0), Ref(0),
        zeros(n), zeros(n)
    )
end
reset!(x::Vector{<:Number}) = empty!(x)
reset!(x::Vector{<:AbstractArray}) = map(reset!, x)
reset!(x::NamedTuple) = map(reset!, x)
reset!(lpdf::NUTSPosterior) = begin
    map(reset!, (lpdf.position, lpdf.gradient, lpdf.dH, lpdf.dangle))
    lpdf.momentum .= 0
    lpdf.n_right[] = 0
end

velocity!((;M⁻¹)::DynamicHMC.GaussianKineticEnergy, momentum; velocity) = mul!(velocity, M⁻¹, momentum)
kick((;ℓ, κ)::DynamicHMC.Hamiltonian, (;p, Q)::DynamicHMC.PhasePoint, stepsize) = @.(p + .5 * stepsize * Q.∇ℓq)
leapfrog!((;ℓ, κ)::DynamicHMC.Hamiltonian, (;p, Q)::DynamicHMC.PhasePoint, stepsize; velocity) = begin
    @assert isfinite(Q.ℓq)
    # @. p += .5 * stepsize * Q.∇ℓq
    p = @.(p + .5 * stepsize * Q.∇ℓq)
    # @. Q.q += stepsize * $velocity!(κ, p; velocity)
    q = @.(Q.q + stepsize * $velocity!(κ, p; velocity))
    Q = DynamicHMC.evaluate_ℓ(ℓ, q)
    @. p += .5 * stepsize * Q.∇ℓq
    # DynamicHMC relies on Q.q and p being new vectors
    DynamicHMC.PhasePoint(Q, p)
end
kinetic_energy!(energy::DynamicHMC.GaussianKineticEnergy, momentum; velocity) = .5 * dot(momentum, velocity!(energy, momentum; velocity))
logdensity!((;κ)::DynamicHMC.Hamiltonian, (;p, Q)::DynamicHMC.PhasePoint; velocity) = begin 
    (; ℓq) = Q
    isfinite(ℓq) || return oftype(ℓq, -Inf)
    K = kinetic_energy!(κ, p; velocity)
    ℓq - (isfinite(K) ? K : oftype(K, Inf))
end
finiteorzero(x) = isfinite(x) ? x : zero(x)
angle(p1, p2) = finiteorzero(acos(clamp(@bsum(p1 * p2 / ($norm(p1) * $norm(p2))), -1, 1)))
# M = W*W' => inv(M) = inv(W')*inv(W) => W' * inv(M) = inv(W)
transformed_velocity!((;W)::DynamicHMC.GaussianKineticEnergy, momentum; velocity) = ldiv!(velocity, W, momentum)
transformed_angle!(
    energy::DynamicHMC.GaussianKineticEnergy, momentum1, momentum2; 
    velocity1, velocity2
) = angle(
    transformed_velocity!(energy, momentum1; velocity=velocity1),
    transformed_velocity!(energy, momentum2; velocity=velocity2)
)

@views DynamicHMC.move(trajectory::DynamicHMC.TrajectoryNUTS{DynamicHMC.Hamiltonian{K,P}}, z, fwd) where {K, P<:NUTSPosterior} = begin
    (;H, ϵ) = trajectory
    lpdf = H.ℓ
    if length(lpdf.dH) == 0
        lpdf.momentum .= 0.
    end
    z = leapfrog!(H, z, fwd ? ϵ : -ϵ; velocity=lpdf.c1)
    dx = lpdf.momentum[:, fwd ? 2 : 1] 
    dx .+= z.p
    f! = fwd ? append! : prepend!
    f!(lpdf.position, z.Q.q)
    f!(lpdf.gradient, z.Q.∇ℓq)
    f!(lpdf.dH, logdensity!(H, z; velocity=lpdf.c1) - trajectory.π₀)
    f!(lpdf.dangle, 2*transformed_angle!(H.κ, dx, z.p; velocity1=lpdf.c1, velocity2=lpdf.c2))
    fwd && (lpdf.n_right[] += 1)
    z
end
n_total(lpdf::NUTSPosterior) = length(lpdf.dH)
n_right(lpdf::NUTSPosterior) = lpdf.n_right[]
n_left(lpdf::NUTSPosterior) = n_total(lpdf) - n_right(lpdf)
skip_zero(x) = x + (x >= 0)
idxs(lpdf::NUTSPosterior) = Base.broadcasted(skip_zero, -n_left(lpdf):n_right(lpdf)-1)
signed_idx(lpdf::NUTSPosterior, x::AbstractVector) = begin 
    idx = findfirst(==(x), eachcol(lpdf.position))
    isnothing(idx) && return 0
    idxs(lpdf)[idx]
    # return idx > n_left(lpdf) ? idx - n_left(lpdf) : -(n_left(lpdf) - idx + 1)
end
abs_idx(lpdf::NUTSPosterior, x::AbstractVector) = abs(signed_idx(lpdf, x))
# dts(lpdf::NUTSPosterior; dt=1.) = dt .* vcat(-lpdf.idxs[1]:-1, 1:lpdf.idxs[2])

levelwise(f; n_right, n_levels) = begin 
    left_start = 2^n_levels-1 - n_right
    right_start = left_start + 1
    level_n = 1
    for level in 1:n_levels
        if (n_right & level_n) != 0
            f(level, right_start:right_start+level_n-1)
            right_start += level_n
        else
            f(level, left_start-level_n+1:left_start)
            left_start -= level_n
        end
        level_n *= 2
    end
end

@views leaf_weights!(weights, lpdf::NUTSPosterior, termination::DynamicHMC.InvalidTree; cache) = begin 
    # return leaf_weights(termination, lpdf.dH; n_right=lpdf.n_right[])
    (;left, right) = termination
    n_steps = n_total(lpdf)
    resize!(weights, n_steps)
    weights .= 0
    n_right = lpdf.n_right[]
    valid_idxs = if !(min(left, right) <= 0 <= max(left, right))
        n_steps_left = 2^trunc(Int, log2(n_steps))-1
        if left > 0 
            @assert left <= right
            n_right -= n_steps - n_steps_left
            1:n_steps_left
        else
            @assert right <= left
            n_steps-n_steps_left+1:n_steps
        end
    else
        1:n_steps
    end
    n_valid = length(valid_idxs)
    n_valid == 0 && return weights
    (;dH) = lpdf
    n_levels = Int(log2(1+n_valid))
    resize!(cache, n_levels)
    log_right_weights = cache
    levelwise(;n_right, n_levels) do level, idxs
        log_right_weights[level] = logsumexp(dH[valid_idxs[idxs]])
        @. weights[valid_idxs[idxs]] = dH[valid_idxs[idxs]] - log_right_weights[level]
    end

    log_switch_weights = cache
    log_left_weight = 0.
    for level in 1:n_levels
        tmp = log_right_weights[level]
        log_switch_weights[level] = min(tmp - log_left_weight, 0.)
        log_left_weight = logaddexp(log_left_weight, tmp)
    end

    log_stay_weights = log_switch_weights
    log_remainder = log1mexp(log_stay_weights[end])
    for level in reverse(1:n_levels-1)
        tmp = log_switch_weights[level]
        log_stay_weights[level] += log_remainder
        log_remainder += log1mexp(tmp)
    end

    levelwise(;n_right, n_levels) do level, idxs
        @. weights[valid_idxs[idxs]] = exp(weights[valid_idxs[idxs]] + log_stay_weights[level])
    end
    weights
end

sample_tree!(lpdf, position::AbstractVector; kwargs...) = sample_tree!(
    lpdf, DynamicHMC.evaluate_ℓ(lpdf, position; strict=true); kwargs...
)
sample_tree!(
    lpdf, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    scale=Diagonal(ones(LogDensityProblems.dimension(lpdf))), 
    kwargs...
) = sample_tree!(
    DynamicHMC.Hamiltonian(
         DynamicHMC.GaussianKineticEnergy(
            WarmupHMC.MatrixFactorization(scale, scale'), 
            WarmupHMC.MatrixInverse(scale')
        ), 
         lpdf
    ), 
    position_and_gradient;
    kwargs...
)
sample_tree!(
    hamiltonian::DynamicHMC.Hamiltonian, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    max_depth=32,
    rng, algorithm=DynamicHMC.NUTS(;max_depth), stepsize=1.
) = DynamicHMC.sample_tree(
    rng, algorithm, hamiltonian, position_and_gradient, stepsize
)