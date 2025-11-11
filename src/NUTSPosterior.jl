abstract type AbstractNUTSPosterior{P} <: WrappedLogDensityProblem{P} end
Base.parent(lpdf::AbstractNUTSPosterior) = lpdf.posterior
cache(lpdf::AbstractNUTSPosterior) = cache(parent(lpdf))
logdensity!(lpdf::AbstractNUTSPosterior, position_and_gradient_and_momentum) = logdensity!(parent(lpdf), position_and_gradient_and_momentum)
rand_momentum(rng, lpdf::AbstractNUTSPosterior) = rand_momentum(rng, parent(lpdf))

problem(trajectory::DynamicHMC.TrajectoryNUTS) = problem(trajectory.H)
problem(hamiltonian::DynamicHMC.Hamiltonian) = hamiltonian.ℓ
stepsize(trajectory::DynamicHMC.TrajectoryNUTS) = trajectory.ϵ

# energy.W = MatrixInverse(scale')
# transformed_velocity = xi
# velocity!(velocity, scale, transformed_velocity) = 
# random_momentum!(rng::AbstractRNG, energy::DynamicHMC.GaussianKineticEnergy; cache) = mul!(energy.W, randn!(rng, cache))
# velocity!(velocity, scale, momentum) = mul!(velocity, MatrixFactorization(scale, scale'), momentum)
# kinetic_energy!(scale, momentum; cache) = .5 * sum(abs2, mul!(cache, scale', momentum))
# move!!(lpdf, position_and_gradient_and_momentum::DynamicHMC.PhasePoint; stepsize) = begin 
#     (;Q, p) = position_and_gradient_and_momentum
#     @assert isfinite(Q.ℓq)
#     @. p += .5 * stepsize * Q.∇ℓq
#     @. Q.q += stepsize * p
#     Q = DynamicHMC.evaluate_ℓ(ℓ, q)
#     @. p += .5 * stepsize * Q.∇ℓq
#     DynamicHMC.PhasePoint(Q, p)
# end

DynamicHMC.move(
    trajectory::DynamicHMC.TrajectoryNUTS{DynamicHMC.Hamiltonian{K,P}}, 
    position_and_gradient_and_momentum::DynamicHMC.PhasePoint, 
    fwd
) where {K, P<:AbstractNUTSPosterior} = move!!(
    problem(trajectory), deepcopy(position_and_gradient_and_momentum); 
    stepsize=(fwd ? +1 : -1) * stepsize(trajectory)
)

"""
momentum = scale' \\ transformed_velocity
velocity = scale * scale' * momentum = scale * transformed_velocity
kinetic_energy = dot(momentum, velocity) = sum(abs2, transformed_velocity)
kick: momentum += stepsize * grad
kick: transformed_velocity += stepsize * scale' * grad
drift: position += stepsize * velocity
drift: position += stepsize * scale * transformed_velocity
"""
struct PreconditionedNUTSPosterior{P,S,C} <: AbstractNUTSPosterior{P}
    posterior::P
    scale::S
    cache::C
end
PreconditionedNUTSPosterior(lpdf, scale) = PreconditionedNUTSPosterior(lpdf, scale, zeros(ldpdim(lpdf)))
scale(lpdf::PreconditionedNUTSPosterior) = lpdf.scale
scale(lpdf::AbstractNUTSPosterior) = scale(parent(lpdf))
scale_cache(lpdf::PreconditionedNUTSPosterior) = lpdf.cache
scale_cache(lpdf::AbstractNUTSPosterior) = scale_cache(parent(lpdf))

kick!(transformed_velocity, gradient; stepsize, scale, cache) = begin
    @. transformed_velocity += stepsize * $mul!(cache, scale', gradient)
end
halfkick!(args...; stepsize, kwargs...) = kick!(args...; stepsize=.5*stepsize, kwargs...)
drift!(position, transformed_velocity; stepsize, scale, cache) = begin 
    @. position += stepsize * $mul!(cache, scale, transformed_velocity)
end
move!!(lpdf::PreconditionedNUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint; stepsize) = begin 
    (;cache, scale) = lpdf
    (;Q, p) = position_and_gradient_and_momentum
    @assert isfinite(Q.ℓq)
    halfkick!(p, Q.∇ℓq; stepsize, scale, cache)
    drift!(Q.q, p; stepsize, scale, cache)
    Q = DynamicHMC.evaluate_ℓ(lpdf, Q.q)
    halfkick!(p, Q.∇ℓq; stepsize, scale, cache)
    DynamicHMC.PhasePoint(Q, p)
end
logdensity!(::PreconditionedNUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint) = finiteor(
    position_and_gradient_and_momentum.Q.ℓq - .5 * sum(abs2, position_and_gradient_and_momentum.p), 
    -Inf
)
rand_momentum(rng, lpdf::PreconditionedNUTSPosterior) = randn(rng, ldpdim(lpdf))

struct AdaptiveNUTSPosterior{P,T} <: AbstractNUTSPosterior{P}
    posterior::P
    H_extrema::Vector{T}
    recovery_probability::Ref{T}
    info::Vector{Int}
end
info(lpdf::AdaptiveNUTSPosterior) = (;n_substeps=lpdf.info[1], n_steps=lpdf.info[2], current_refinement=Int(log2(lpdf.info[1])), recovery_probability=lpdf.recovery_probability[])
AdaptiveNUTSPosterior(lpdf) = AdaptiveNUTSPosterior(lpdf, zeros(2), Ref(0.), [1, 0, 0])
reset!(lpdf::AdaptiveNUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint) = begin 
    reset!(parent(lpdf), position_and_gradient_and_momentum)
    lpdf.H_extrema .= logdensity!(lpdf, position_and_gradient_and_momentum)
    lpdf.info .= (1, 0, 0)
end
remake!!(lpdf::AdaptiveNUTSPosterior) = remake!!(parent(lpdf), remake!!(lpdf, parent(parent(lpdf))))
remake!!(lpdf::AdaptiveNUTSPosterior, posterior) = AdaptiveNUTSPosterior(posterior, lpdf.H_extrema, lpdf.recovery_probability, lpdf.info)

n_substeps(lpdf::AdaptiveNUTSPosterior) = lpdf.info[1]
n_steps(lpdf::AdaptiveNUTSPosterior) = lpdf.info[2]
n_refinements(lpdf::AdaptiveNUTSPosterior) = lpdf.info[3]
recovery_probability(lpdf::AdaptiveNUTSPosterior) = lpdf.recovery_probability[]::Float64
divergent(lpdf::AdaptiveNUTSPosterior) = divergent(parent(lpdf))
log_acceptance_rate(lpdf::AdaptiveNUTSPosterior) = log_acceptance_rate(parent(lpdf)) + log(n_steps(parent(lpdf))) - log(n_steps(lpdf))
max_abs_dH(lpdf::AdaptiveNUTSPosterior) = max_abs_dH(parent(lpdf))
leaf_weights!(lpdf::AdaptiveNUTSPosterior) = begin
    leaf_weights!(parent(lpdf))
    parent(lpdf).weights .*= recovery_probability(lpdf)
end

rejects!(lpdf::AdaptiveNUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint) = rejects!(
    lpdf, logdensity!(lpdf, position_and_gradient_and_momentum)
)
rejects!(lpdf::AdaptiveNUTSPosterior, H) = begin 
    lpdf.info[2] += 1
    lo, hi = lpdf.H_extrema .= extrema((lpdf.H_extrema[1], lpdf.H_extrema[2], H))
    # (hi - lo > 1000 && H - parent(lpdf).H0[] > -1000) && @info "hi - lo is $(hi-lo), but H - H0 only $(H - parent(lpdf).H0[])"
    hi - lo > 1000
end
reject(z::DynamicHMC.PhasePoint) = DynamicHMC.PhasePoint(DynamicHMC.EvaluatedLogDensity(z.Q.q, -Inf, z.Q.∇ℓq), z.p)
move!!(lpdf::AdaptiveNUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint; stepsize) = begin 
    stepsize /= n_substeps(lpdf)
    for _ in 1:n_substeps(lpdf)
        position_and_gradient_and_momentum = move!!(parent(lpdf), position_and_gradient_and_momentum; stepsize)
        rejects!(lpdf, position_and_gradient_and_momentum) && return reject(position_and_gradient_and_momentum)
    end
    position_and_gradient_and_momentum
end


struct NUTSPosterior{P,T,R} <: AbstractNUTSPosterior{P}
    posterior::P
    H0::Ref{T}
    dH::Vector{T}
    info::Vector{Int}
    cache::Vector{T}
    weights::Vector{T}
    recorder::R
end
NUTSPosterior(lpdf; R, kwargs...) = NUTSPosterior(lpdf, R(LogDensityProblems.dimension(lpdf); kwargs...); kwargs...)
NUTSPosterior(lpdf, recorder; T=Float64) = NUTSPosterior(
    lpdf, 
    Ref(zero(T)),
    zeros(T, 0), 
    zeros(Int, 3),
    zeros(T, 0), 
    zeros(T, 0),
    recorder,
)
remake!!(lpdf::NUTSPosterior, posterior) = NUTSPosterior(posterior, lpdf.H0, lpdf.dH, lpdf.info, lpdf.cache, lpdf.weights, lpdf.recorder)
recorder(lpdf::NUTSPosterior) = lpdf.recorder
recorder(lpdf::AbstractNUTSPosterior) = recorder(parent(lpdf))

n_steps(lpdf::NUTSPosterior) = length(lpdf.dH)
divergent(lpdf::NUTSPosterior) = lpdf.info[2] == lpdf.info[3]
log_acceptance_rate(lpdf::NUTSPosterior) = logsumexp(@broadcasted(min(0, lpdf.dH))) - log(n_steps(lpdf))
max_abs_dH(lpdf::NUTSPosterior) = maximum(abs, lpdf.dH)
expected_stat(f, lpdf::AbstractNUTSPosterior) = expected_stat(f, parent(lpdf))
expected_stat(f, lpdf::NUTSPosterior) = sum(1:n_steps(lpdf)) do i
    lpdf.weights[i] * f(lpdf, i)
end
@views squared_jump(lpdf::NUTSPosterior, i) = @bsum(abs2(lpdf.recorder.positions.value[:, i] - lpdf.recorder.initial_position.value))
@views squared_gradient_jump(lpdf::NUTSPosterior, i) = @bsum(abs2(lpdf.recorder.gradients.value[:, i] - lpdf.recorder.initial_gradient.value))
@views squared_igradient_jump(lpdf::NUTSPosterior, i) = 1/@bsum(1/abs2(lpdf.recorder.gradients.value[:, i] - lpdf.recorder.initial_gradient.value))
squared_dual_jump1(lpdf::NUTSPosterior, i) = sqrt(squared_jump(lpdf, i) * squared_gradient_jump(lpdf, i))
squared_dual_jump2(lpdf::NUTSPosterior, i) = sqrt(squared_jump(lpdf, i) * squared_igradient_jump(lpdf, i))

for base in (:jump, :gradient_jump, :igradient_jump, :dual_jump1, :dual_jump2)
    a, s, q = Symbol.(("absolute_", "squared_", "quad_"), base)
    @eval begin 
        $(a)(lpdf::NUTSPosterior, i) = sqrt($s(lpdf, i))
        $(q)(lpdf::NUTSPosterior, i) = $s(lpdf, i)^2
    end
    for b in (a, s, q)
        @eval $(Symbol("expected_", b))(lpdf::AbstractNUTSPosterior) = expected_stat($b, lpdf)
    end
end

move!!(lpdf::NUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint; stepsize, kwargs...) = begin 
    f! = stepsize > 0 ? append! : prepend!
    pre_move!!(lpdf, position_and_gradient_and_momentum, f!)
    position_and_gradient_and_momentum = move!!(parent(lpdf), position_and_gradient_and_momentum; stepsize, kwargs...)
    post_move!!(lpdf, position_and_gradient_and_momentum, f!)
    position_and_gradient_and_momentum
end
reset!(lpdf::NUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint) = begin
    lpdf.H0[] = logdensity!(lpdf, position_and_gradient_and_momentum)
    empty!(lpdf.dH)
    lpdf.info .= 0
    # resize!(lpdf.cache, LogDensityProblems.dimension(lpdf))
    initialize!(lpdf.recorder, position_and_gradient_and_momentum)
end
# random_momentum!(rng::AbstractRNG, energy::DynamicHMC.GaussianKineticEnergy; cache) = mul!(energy.W, randn!(rng, cache))
# DynamicHMC relies on Q.q and p being new vectors
# leapfrog(H::DynamicHMC.Hamiltonian, z::DynamicHMC.PhasePoint, stepsize; velocity) = leapfrog!!(H, deepcopy(z), stepsize; velocity)
# leapfrog!!((;ℓ, κ)::DynamicHMC.Hamiltonian, (;Q, p)::DynamicHMC.PhasePoint, stepsize; velocity) = begin
#     @assert isfinite(Q.ℓq)
#     @. p += .5 * stepsize * Q.∇ℓq
#     @. Q.q += stepsize * $velocity!(κ, p; velocity)
#     Q = DynamicHMC.evaluate_ℓ(ℓ, q)
#     @. p += .5 * stepsize * Q.∇ℓq
#     DynamicHMC.PhasePoint(Q, p)
# end
# M = W*W' => inv(M) = inv(W')*inv(W) => W' * inv(M) = inv(W)
# transformed_velocity!((;W)::DynamicHMC.GaussianKineticEnergy, momentum; velocity) = ldiv!(velocity, W, momentum)
# angle(p1, p2) = finiteorzero(acos(clamp(@bsum(p1 * p2 / ($norm(p1) * $norm(p2))), -1, 1)))
# transformed_angle!(
#     energy::DynamicHMC.GaussianKineticEnergy, momentum1, momentum2; 
#     velocity1, velocity2
# ) = angle(
#     transformed_velocity!(energy, momentum1; velocity=velocity1),
#     transformed_velocity!(energy, momentum2; velocity=velocity2)
# )
# logdensity!((;κ)::DynamicHMC.Hamiltonian, (;p, Q)::DynamicHMC.PhasePoint; velocity) = begin 
#     (; ℓq) = Q
#     isfinite(ℓq) || return oftype(ℓq, -Inf)
#     K = kinetic_energy!(κ, p; velocity)
#     ℓq - (isfinite(K) ? K : oftype(K, Inf))
# end
pre_move!!(lpdf::NUTSPosterior, z, f!) = pre_move!!(lpdf.recorder, z, f!)
post_move!!(lpdf::NUTSPosterior, z, f!) = begin 
    post_move!!(lpdf.recorder, z, f!)
    f!(lpdf.dH, logdensity!(lpdf, z) - lpdf.H0[])
    f! == append! && (lpdf.info[1] += 1)
end
n_total(lpdf::NUTSPosterior) = length(lpdf.dH)
n_right(lpdf::NUTSPosterior) = lpdf.info[1]
n_left(lpdf::NUTSPosterior) = n_total(lpdf) - n_right(lpdf)
n_total(lpdf::AbstractNUTSPosterior) = n_total(parent(lpdf))
n_right(lpdf::AbstractNUTSPosterior) = n_right(parent(lpdf))
n_left(lpdf::AbstractNUTSPosterior) = n_left(parent(lpdf))
skip_zero(x) = x + (x >= 0)
idxs(lpdf::NUTSPosterior) = Base.broadcasted(skip_zero, -n_left(lpdf):n_right(lpdf)-1)
signed_idx(lpdf::NUTSPosterior, x::AbstractVector) = begin 
    idx = findfirst(==(x), eachcol(lpdf.position))
    isnothing(idx) && return 0
    idxs(lpdf)[idx]
    # return idx > n_left(lpdf) ? idx - n_left(lpdf) : -(n_left(lpdf) - idx + 1)
end
abs_idx(lpdf::NUTSPosterior, x::AbstractVector) = abs(signed_idx(lpdf, x))
# idx(lpdf::NUTSPosterior, x::AbstractVector) = 
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

leaf_weights!(lpdf::AbstractNUTSPosterior) = leaf_weights!(parent(lpdf))
leaf_weights!(lpdf::NUTSPosterior) = leaf_weights!(lpdf.weights, lpdf)
@views leaf_weights!(weights, lpdf::NUTSPosterior) = begin 
    (;cache) = lpdf
    n_right, left, right = lpdf.info
    n_steps = n_total(lpdf)
    resize!(weights, n_steps)
    weights .= 0
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


sample!(lpdf, position::AbstractVector; kwargs...) = sample!(lpdf, DynamicHMC.evaluate_ℓ(lpdf, position; strict=true); kwargs...)
sample!(
    lpdf, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    scale=Diagonal(ones(ldpdim(lpdf))), 
    kwargs...
) = sample!(PreconditionedNUTSPosterior(lpdf, scale, zeros(ldpdim(lpdf))), position_and_gradient; kwargs...)
sample!(
    lpdf::PreconditionedNUTSPosterior, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    R, 
    kwargs...
) = sample!(NUTSPosterior(lpdf; R), position_and_gradient; kwargs...)
sample!(
    lpdf::AbstractNUTSPosterior, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    rng,
    momentum=rand_momentum(rng, lpdf),
    directions=rand(rng, DynamicHMC.Directions),
    kwargs...
) = sample!(lpdf, DynamicHMC.PhasePoint(position_and_gradient, momentum); rng, directions, kwargs...).Q

sample!(
    lpdf::NUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint; 
    rng, stepsize, directions, max_abs_dH=1000., max_depth=10, lw=true
) = begin 
    reset!(lpdf, position_and_gradient_and_momentum)
    position_and_gradient_and_momentum, _, (;left, right), _ = DynamicHMC.sample_trajectory(
        rng,
        DynamicHMC.TrajectoryNUTS(
            DynamicHMC.Hamiltonian(DynamicHMC.GaussianKineticEnergy(ldpdim(lpdf)), lpdf), 
            logdensity!(lpdf, position_and_gradient_and_momentum), 
            stepsize, 
            -max_abs_dH, 
            Val(:generalized)
        ),
        position_and_gradient_and_momentum,
        max_depth,
        directions
    )
    lpdf.info[2:3] .= left, right
    lw && leaf_weights!(lpdf)
    position_and_gradient_and_momentum
end
sample!(
    lpdf::AdaptiveNUTSPosterior, position_and_gradient_and_momentum::DynamicHMC.PhasePoint; 
    rng, stepsize, directions, max_refinements=0, lw=true, kwargs...
) = begin
    reset!(lpdf, position_and_gradient_and_momentum)
    H0 = lpdf.H_extrema[1]
    new_position_and_gradient_and_momentum = position_and_gradient_and_momentum 
    for i in 0:max_refinements
        lpdf.H_extrema .= H0
        lpdf.info[1] = 2^i
        new_position_and_gradient_and_momentum = sample!(
            remake!!(lpdf), position_and_gradient_and_momentum; 
            rng, stepsize, directions, lw=false, kwargs...
        )
        divergent(lpdf) || break
        # error(parent(lpdf).info)
    end
    lpdf.info[3] = Int(log2(lpdf.info[1]))
    rp = recovery_probability!(lpdf, new_position_and_gradient_and_momentum; rng, stepsize, kwargs...)
    lw && leaf_weights!(lpdf)
    if rand(rng) < rp
        new_position_and_gradient_and_momentum
    else
        position_and_gradient_and_momentum
    end

end
reverse_directions(lpdf, new_position_and_gradient_and_momentum::DynamicHMC.PhasePoint) = 
reverse_directions(lpdf, new_position_and_gradient_and_momentum.Q)
reverse_directions(lpdf, new_position_and_gradient::DynamicHMC.EvaluatedLogDensity) = 
reverse_directions(lpdf, new_position_and_gradient.q)
reverse_directions(lpdf::AbstractNUTSPosterior, new_position::AbstractVector) = reverse_directions(parent(lpdf), new_position)
reverse_directions(lpdf::NUTSPosterior, new_position::AbstractVector) = begin 
    n_right, left, right = lpdf.info
    n_steps = n_total(lpdf)
    idx = findfirst(==(new_position), eachcol(recorder(lpdf).positions.value))
    pos = isnothing(idx) ? 0 : idxs(lpdf)[idx]
    if !(min(left, right) <= 0 <= max(left, right))
        n_steps_missing = 2^ceil(Int, log2(n_steps+1)) - 1 - n_steps
        if left > 0 
            @assert left <= right
            directions(n_right - pos + n_steps_missing)
        else
            @assert right <= left
            directions(n_right - pos)
        end
    else
        directions(n_right - pos)
    end
end 
directions(n_right) = begin 
    rv = UInt32(0)
    level_n = UInt32(1)
    for _ in 1:32
        rv |= (n_right & level_n)
        level_n <<= 1
    end
    DynamicHMC.Directions(rv)
end
n_right(directions::DynamicHMC.Directions) = begin 
    rv = 0
    level_n = UInt32(1)
    for _ in 1:32
        fwd, directions = DynamicHMC.next_direction(directions)
        fwd && (rv |= level_n)
        level_n <<= 1
    end
    rv
end
recovery_probability!(
    lpdf::AdaptiveNUTSPosterior, new_position_and_gradient_and_momentum::DynamicHMC.PhasePoint;
    rng, stepsize, kwargs...
) = if divergent(lpdf)
    # @warn "Diverged, but not trying to recover at $(info(lpdf))"
    lpdf.recovery_probability[] = 1.
elseif lpdf.info[1] == 1
    lpdf.recovery_probability[] = 1.
else
    directions = reverse_directions(lpdf, new_position_and_gradient_and_momentum)
    rlpdf = NUTSPosterior(remake!!(lpdf, parent(parent(lpdf))), (;))
    max_refinements = n_refinements(lpdf) - 1
    H0 = logdensity!(rlpdf, new_position_and_gradient_and_momentum)
    for i in 0:max_refinements
        lpdf.H_extrema .= H0
        lpdf.info[1] = 2^i
        sample!(
            rlpdf, new_position_and_gradient_and_momentum; 
            rng, stepsize, directions, kwargs...
        )
        if !divergent(rlpdf)
            # parent(lpdf).dH .= -Inf
            # @error "Not recovering at $(max_refinements=>info(lpdf))"
            return lpdf.recovery_probability[] = 0.
        end
    end
    # @info "Recovering at $(max_refinements=>info(lpdf))"
    lpdf.recovery_probability[] = 1.
end

# begin 
#     position_and_gradient_and_momentum = DynamicHMC.PhasePoint(
#         position_and_gradient, rand_momentum(rng, lpdf)
#     )
#     position_and_gradient_and_momentum sample_trajectory(rng, trajectory, z, max_depth, directions)

# end
# sample!(
#     lpdf::PreconditionedNUTSPosterior, position_and_gradient::DynamicHMC.EvaluatedLogDensity;
#     rng, stepsize, 
# ) = begin 
#     position_and_gradient_and_momentum = DynamicHMC.PhasePoint(
#         position_and_gradient, randn(rng)
#     )
#     sample_trajectory(rng, trajectory, z, max_depth, directions)
#     trajectory = 
# end

# sample_tree!(lpdf, position::AbstractVector; kwargs...) = sample_tree!(
#     lpdf, DynamicHMC.evaluate_ℓ(lpdf, position; strict=true); kwargs...
# )
# sample_tree!(
#     lpdf, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
#     scale=Diagonal(ones(ldpdim(lpdf))), 
#     kwargs...
# ) = sample_tree!(
#     DynamicHMC.Hamiltonian(
#         DynamicHMC.GaussianKineticEnergy(ldpdim(lpdf)), 
#         PreconditionedNUTSPosterior(lpdf, scale, zeros(ldpdim(lpdf)))
#     ), 
#     position_and_gradient;
#     kwargs...
# )

# sample_tree!(
#     hamiltonian::DynamicHMC.Hamiltonian, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
#     max_depth=32,
#     rng, algorithm=DynamicHMC.NUTS(;max_depth), stepsize=1.
# ) = DynamicHMC.sample_tree(
#     rng, algorithm, hamiltonian, position_and_gradient, stepsize
# )
# sample_tree!(
#     hamiltonian::DynamicHMC.Hamiltonian{K,P}, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
#     max_depth=32,
#     rng, algorithm=DynamicHMC.NUTS(;max_depth), stepsize=1.
# )  where {K, P<:NUTSPosterior} = begin 
#     lpdf = hamiltonian.ℓ
#     reset!(lpdf, position_and_gradient)
#     position_and_gradient, stats = DynamicHMC.sample_tree(
#         rng, algorithm, hamiltonian, position_and_gradient, stepsize
#     )
#     lpdf.info[2:3] .= stats.termination.left, stats.termination.right
#     position_and_gradient, stats
# end

find_initial_stepsize(
    lpdf, position_and_gradient::DynamicHMC.EvaluatedLogDensity;
    scale=Diagonal(ones(LogDensityProblems.dimension(lpdf))),
    rng
) = begin
    kinetic_energy = DynamicHMC.GaussianKineticEnergy(
        WarmupHMC.MatrixFactorization(scale, scale'), 
        WarmupHMC.MatrixInverse(scale')
    )
    DynamicHMC.find_initial_stepsize(
        DynamicHMC.InitialStepsizeSearch(), 
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf), 
            DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy))
        )
    )
end


