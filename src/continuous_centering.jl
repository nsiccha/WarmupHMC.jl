# Per-cell stochastic Gauss–Newton on (centeredness, position offset, gradient
# offset, log inverse mass). Sum normal equations within a proposal group before
# taking a step. Only 4 × 4 *per cell*, independent of the target dimension.
struct _JointTree
    moments::WeightedScaleAdaptation{Float64}
    gradient::Matrix{Float64}
    information::Array{Float64,3}
end
_JointTree(d, n) = _JointTree(WeightedScaleAdaptation(d), zeros(4, n), zeros(4, 4, n))

mutable struct _JointCentering
    parameters::Matrix{Float64}
    information::Array{Float64,3}
    moments::WeightedScaleAdaptation{Float64}
    initial_scale::Vector{Float64}
    adapt_controls::Bool
    trees::Int
    supplied::Int
    installations::Vector{Int}
    frozen_after::Union{Nothing,Int}
end
const _JointLeaves = _CenteringLeaves{_JointCentering}
_continuous_schedule(leaves) = false
_continuous_schedule(::_JointLeaves) = true

function _check_continuous_input(lpdf, schedule, adapt, evidence, ensemble)
    if schedule !== :continuous
        isnothing(ensemble) || throw(ArgumentError("centering_initial_ensemble requires centering_schedule=:continuous"))
        return
    end
    isnothing(evidence) || evidence === :nuts_weighted || throw(ArgumentError(
        "continuous adaptation uses NUTS proposal weights"))
    ir = reparametrizer(lpdf)
    if adapt && has_reparametrization(ir)
        ir isa IndexedReparametrization && isempty(ir.chains) &&
            all(p -> first(p) isa Integer && last(p) isa Reparametrization &&
                     last(p).source isa PartiallyCentered, ir.pairs) || throw(ArgumentError(
                "continuous centering adapts indexed scalar constant controls; use nonlinear_adapt=false for a fixed frame"))
        candidate_scoring_plan(lpdf) isa WindowSelectionPlan && throw(ArgumentError(
            "continuous centering requires a per-node scorer"))
    end
    isnothing(ensemble) && return
    all(k -> hasproperty(ensemble, k), (:positions, :gradients, :source_controls)) ||
        throw(ArgumentError("centering_initial_ensemble needs positions, gradients and source_controls"))
    d = LogDensityProblems.dimension(lpdf)
    P, G = ensemble.positions, ensemble.gradients
    P isa AbstractMatrix{<:Real} && G isa AbstractMatrix{<:Real} &&
        size(P) == size(G) && size(P, 1) == d && size(P, 2) > 0 ||
        throw(DimensionMismatch("initializer positions and gradients must be matching dimension × draws matrices"))
    all(isfinite, P) && all(isfinite, G) || throw(ArgumentError("nonfinite initializer ensemble"))
    ir isa IndexedReparametrization && isempty(ir.chains) &&
        all(p -> last(p) isa Reparametrization && last(p).source isa PartiallyCentered, ir.pairs) ||
        throw(ArgumentError("supply cached rows for a plain or indexed constant source frame"))
    collect(ensemble.source_controls) == [last(p).source.c for p in ir.pairs] ||
        throw(ArgumentError("initializer ensemble source_controls differ from the target's starting source frame"))
end

function _continuous_leaves(lpdf, adapt, linear_source, saved)
    d = LogDensityProblems.dimension(lpdf)
    ir = reparametrizer(lpdf)
    n = adapt && has_reparametrization(ir) ? length(ir.pairs) : 0
    policy = if isnothing(saved)
        p = zeros(4, n)
        H = zeros(4, 4, n)
        for j in 1:n
            p[1,j] = last(ir.pairs[j]).source.c
            for k in 1:4
                H[k,k,j] = 1
            end
        end
        _JointCentering(p, H, WeightedScaleAdaptation(d), ones(d), adapt, 0, 0, Int[], nothing)
    else
        saved.adapt_controls == adapt && size(saved.parameters) == (4,n) || throw(ArgumentError(
            "resume continuous centering with the saved nonlinear_adapt and source pairs"))
        _JointCentering(saved.parameters, saved.information, saved.moments, saved.initial_scale,
            adapt, saved.trees, saved.supplied, saved.installations, saved.frozen_after)
    end
    linear = linear_source === :halo ? nothing : NUTSLeaves(d)
    _CenteringLeaves(policy, _JointTree(d, n), _JointTree(d, n), linear, 0, 1, -Inf, -Inf)
end

function reset!(x::_JointTree)
    reset!(x.moments)
    fill!(x.gradient, 0)
    fill!(x.information, 0)
    x
end
function _scale_mass!(x::_JointTree, w)
    iszero(w) && return reset!(x)
    _forget!(x.moments, w)
    x.gradient .*= w
    x.information .*= w
    x
end

function _merge_scale_moments!(a, b)
    iszero(b.weight) && return a
    total = a.weight + b.weight
    f = b.weight / total
    for (m, v, mb, vb) in ((a.position_mean, a.position_m2, b.position_mean, b.position_m2),
                           (a.gradient_mean, a.gradient_m2, b.gradient_mean, b.gradient_m2))
        for i in eachindex(m)
            diff = mb[i] - m[i]
            v[i] += vb[i] + (a.weight * f) * diff^2
            m[i] += f * diff
        end
    end
    a.weight = total
    a.weight2 += b.weight2
    a
end
function _merge_nonlinear!(a::_JointTree, b::_JointTree, rng)
    _merge_scale_moments!(a.moments, b.moments)
    a.gradient .+= b.gradient
    a.information .+= b.information
    a
end

# The public plan may supply its own score for any real c, or defer to the
# indexed map. Read the parent accessors in model coordinates for direct scores.
_joint_frame(::DirectCandidateScoring, ir, q, g) = nothing
_joint_frame(plan::CandidateScoringPlan, ir, q, g) = plan.prepare(ir, q, g)
_joint_score(::DirectCandidateScoring, frame, j, idx, rule, candidate) = nothing
_joint_score(plan::CandidateScoringPlan, frame, j, idx, rule, candidate) =
    plan.score(frame, j, idx, rule, candidate)

function _joint_scores(plan, frame, j, idx, rule, c, q, g, ir, model)
    h = cbrt(eps(Float64))
    lo, hi = max(0.0, c-h), min(1.0, c+h)
    observations = map((c, lo, hi)) do x
        candidate = PartiallyCentered(x)
        obs = _joint_score(plan, frame, j, idx, rule, candidate)
        if isnothing(obs)
            isnothing(model[]) && (model[] = last(ir(q)))
            obs = reparam(candidate, _framed_rargs(rule, q[idx], g[idx], _pair_frames(rule, model[]))...)
        end
        obs
    end
    (_, x, s), (_, xl, sl), (_, xh, sh) = observations
    x, s, (xh-xl)/(hi-lo), (sh-sl)/(hi-lo)
end

function _observe_tree_summary!(tree::_JointTree, policy::_JointCentering, lpdf, q, g, w)
    OnlineStatsBase.fit!(tree.moments, q, g; dw=w)
    isempty(policy.parameters) && return
    ir, plan = reparametrizer(lpdf), candidate_scoring_plan(lpdf)
    frame = _joint_frame(plan, ir, q, g)
    model = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    for (j, (idx, rule)) in enumerate(ir.pairs)
        c, a, b, r = @view policy.parameters[:,j]
        x, s, dx, ds = _joint_scores(plan, frame, j, idx, rule, c, q, g, ir, model)
        k = exp(-r/2)
        u, v = (x-a)*k, (s-b)/k
        ju, jv = (dx*k, -k, 0., -u/2), (ds/k, 0., -1/k, v/2)
        for i in 1:4
            tree.gradient[i,j] += w * (ju[i]*u + jv[i]*v)
            for h in 1:4
                tree.information[i,h,j] += w * (ju[i]*ju[h] + jv[i]*jv[h])
            end
        end
    end
end

# The live diagonal is the sampler's factor L; inverse mass = L * L'.
# The current point and all chart-dependent optimizer parameters publish as
# one transaction. On rejection only ordinary evidence in the old frame advances.
function _joint_fit!(s, tree, lpdf, point, scale)
    _merge_scale_moments!(s.moments, tree.moments)
    next_scale = similar(scale)
    _continuous_scales!(next_scale, s.moments, s.initial_scale, 1.0, scale)
    if isempty(s.parameters)
        copyto!(scale, next_scale)
        return point, false
    end
    ir = reparametrizer(lpdf)
    next = copy(s.parameters)
    information = copy(s.information)
    moved = false
    for (j, (idx, rule)) in enumerate(ir.pairs)
        H = @view information[:,:,j]
        h = @view tree.information[:,:,j]
        grad = @view tree.gradient[:,j]
        all(isfinite, h) && all(isfinite, grad) || continue
        H .+= h
        # A cell whose score is independent of centeredness uses the
        # ordinary diagonal estimator, also used when controls are fixed.
        iszero(h[1,1]) && continue
        delta = cholesky(Symmetric(H)) \ grad
        proposal = (@view s.parameters[:,j]) .- delta ./ max(1.0, abs(delta[4]))
        proposal[1] = clamp(proposal[1], 0.0, 1.0)
        factor = exp(proposal[4]/2)
        all(isfinite, proposal) && isfinite(factor) && factor > 0 || continue
        next[:,j] .= proposal
        next_scale[idx] = factor
        moved |= proposal[1] != rule.source.c
    end
    if moved
        old = snapshot_reparametrization(ir)
        try
            for (j, (idx, rule)) in enumerate(ir.pairs)
                ir.pairs[j] = idx => Reparametrization(rule.target, PartiallyCentered(next[1,j]), rule.args...)
            end
            _synchronize_scoring!(lpdf)
            proposed_point, failure = _checked_centering_point(lpdf, old, point)
            if !isnothing(failure)
                _restore_reparametrization!(ir, old)
                _synchronize_scoring!(lpdf)
                @warn "Rejected continuous centering update: $(failure.reason)" failure.fields...
                return point, false
            end
            point = proposed_point
        catch
            _restore_reparametrization!(ir, old)
            _synchronize_scoring!(lpdf)
            rethrow()
        end
    end
    copyto!(s.parameters, next)
    copyto!(s.information, information)
    copyto!(scale, next_scale)
    point, moved
end

function _initialize_continuous!(leaves, lpdf, point, scale, ensemble)
    s = leaves.schedule
    s.initial_scale .= scale
    ir = reparametrizer(lpdf)
    for j in axes(s.parameters, 2)
        idx = first(ir.pairs[j])
        s.parameters[4,j] = 2log(scale[idx])
    end
    isnothing(ensemble) && return point
    P, G = ensemble.positions, ensemble.gradients
    tree = leaves.prefix
    for j in axes(P,2)
        OnlineStatsBase.fit!(tree.moments, view(P,:,j), view(G,:,j))
    end
    _continuous_scales!(scale, tree.moments, s.initial_scale, 1., scale)
    for j in axes(s.parameters, 2)
        idx = first(ir.pairs[j])
        s.parameters[2,j] = tree.moments.position_mean[idx]
        s.parameters[3,j] = tree.moments.gradient_mean[idx]
        s.parameters[4,j] = 2log(scale[idx])
    end
    reset!(tree)
    for j in axes(P,2)
        _observe_tree_summary!(tree, s, lpdf, view(P,:,j), view(G,:,j), 1.0)
    end
    s.supplied = size(P,2)
    first(_joint_fit!(s, tree, lpdf, point, scale))
end

function _begin_centering_tree!(leaves::_JointLeaves, state)
    if isnothing(leaves.schedule.frozen_after) && state.current_transition_counter >= state.stepsize_adaptation_limit
        leaves.schedule.frozen_after = state.total_transition_counter - 1
    end
    reset!(leaves)
end
# No stale halo survives an installation. Collect the diagnostic halo once
# the chain is sampling in its final frame; continuous fitting needs no replay.
_record_halo(leaves) = true
_record_halo(leaves::_JointLeaves) = !isnothing(leaves.schedule.frozen_after)
record_nonlinear!(::NonlinearRecorder, lpdf, ::_JointLeaves, stepsize) = nothing

function _update_centering!(state, leaves::_JointLeaves)
    s = leaves.schedule
    scale = state.scale_options.diagonal.diag
    old_log_scale = sum(log, scale)
    state.position_and_gradient, moved = _joint_fit!(s, leaves.prefix, parent(state.recording_lpdf),
                                                     state.position_and_gradient, scale)
    s.trees += 1
    moved && push!(s.installations, state.total_transition_counter)
    # Rescale dual averaging's anchor and both log steps. Its count and
    # acceptance-error history continue across changes to the Hamiltonian.
    shift = (old_log_scale - sum(log, scale)) / length(scale)
    a = state.stepsize_state
    state.stepsize_state = DynamicHMC.DualAveragingState(
        a.μ+shift, a.m, a.H̄, a.logϵ+shift, a.logϵ̄+shift)
    state.stepsize = DynamicHMC.current_ϵ(state.stepsize_state)
end
function _centering_checkpoint(leaves::_JointLeaves)
    s = leaves.schedule
    (; continuous_centering=(; s.parameters, s.information, s.moments, s.initial_scale,
        s.adapt_controls, s.trees, s.supplied, s.installations, s.frozen_after))
end
function _centering_result(leaves::_JointLeaves)
    s = leaves.schedule
    (; centering_schedule=:continuous, centering_installations=copy(s.installations),
       centering_frozen_after=s.frozen_after, adaptation_trees=s.trees, initializer_rows=s.supplied)
end
