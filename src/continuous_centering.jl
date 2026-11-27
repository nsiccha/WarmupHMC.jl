# Per-cell stochastic Gauss–Newton on (centeredness, position offset, gradient
# offset, log inverse mass). Sum normal equations within a proposal group before
# taking a step. Only 4 × 4 *per cell*, independent of the target dimension.
struct _JointTree
    moments::WeightedScaleAdaptation{Float64}
    gradient::Matrix{Float64}
    information::Array{Float64,3}
end
_JointTree(d, n) = _JointTree(WeightedScaleAdaptation(d), zeros(4, n), zeros(4, 4, n))

# Positions drawn by NUTS itself are representatives of the proposal group.
# Keep a bounded set of these alongside initializer representatives. They let
# us compare the *same model points* on either side of a chart change.
const _JOINT_ANCHORS = 16
mutable struct _FrameAnchors
    q::Matrix{Float64}
    g::Matrix{Float64}
    seeds::Int
    seen::Int
end
_FrameAnchors(d, n) = _FrameAnchors(zeros(d,n), zeros(d,n), 0, 0)

function _anchor_point!(a, point)
    a.seen += 1
    j = a.seeds + mod1(a.seen, _JOINT_ANCHORS)
    a.q[:,j] .= point.q
    a.g[:,j] .= point.∇ℓq
end

function _anchor_sample(a, supplied)
    live = min(a.seen, _JOINT_ANCHORS)
    cols = 1:a.seeds+live
    # The seed and trajectory strata retain their own total weights; neither
    # a large ensemble nor a long chain acquires an extra vote from thinning.
    w = vcat(fill(supplied / max(1,a.seeds), a.seeds),
             fill(a.seen / max(1,live), live))
    view(a.q,:,cols), view(a.g,:,cols), w
end

mutable struct _JointCentering
    parameters::Matrix{Float64}
    information::Array{Float64,3}
    moments::WeightedScaleAdaptation{Float64}
    initial_scale::Vector{Float64}
    anchors::_FrameAnchors
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
        _JointCentering(p, H, WeightedScaleAdaptation(d), ones(d), _FrameAnchors(d, n == 0 ? 0 : _JOINT_ANCHORS), adapt, 0, 0, Int[], nothing)
    else
        saved.adapt_controls == adapt && size(saved.parameters) == (4,n) || throw(ArgumentError(
            "resume continuous centering with the saved nonlinear_adapt and source pairs"))
        _JointCentering(saved.parameters, saved.information, saved.moments, saved.initial_scale,
            saved.anchors, adapt, saved.trees, saved.supplied, saved.installations, saved.frozen_after)
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

function _visit_joint_scores(visit, parameters, lpdf, q, g)
    ir, plan = reparametrizer(lpdf), candidate_scoring_plan(lpdf)
    frame = _joint_frame(plan, ir, q, g)
    model = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    for (j, (idx, rule)) in enumerate(ir.pairs)
        visit(j, _joint_scores(plan, frame, j, idx, rule, parameters[1,j], q, g, ir, model))
    end
end

function _normal_equations!(gradient, information, j, parameters, obs, w)
    _, a, b, r = @view parameters[:,j]
    x, s, dx, ds = obs
    k = exp(-r/2)
    u, v = (x-a)*k, (s-b)/k
    ju, jv = (dx*k, -k, 0., -u/2), (ds/k, 0., -1/k, v/2)
    for i in 1:4
        gradient[i,j] += w * (ju[i]*u + jv[i]*v)
        for h in 1:4
            information[i,h,j] += w * (ju[i]*ju[h] + jv[i]*jv[h])
        end
    end
end

function _observe_tree_summary!(tree::_JointTree, policy::_JointCentering, lpdf, q, g, w)
    OnlineStatsBase.fit!(tree.moments, q, g; dw=w)
    isempty(policy.parameters) && return
    _visit_joint_scores(policy.parameters, lpdf, q, g) do j, obs
        _normal_equations!(tree.gradient, tree.information, j, policy.parameters, obs, w)
    end
end

function _anchor_scores(p, lpdf, q, g)
    out = Array{NTuple{4,Float64}}(undef, size(p,2), size(q,2))
    for k in axes(q,2)
        _visit_joint_scores(p, lpdf, view(q,:,k), view(g,:,k)) do j, obs
            out[j,k] = obs
        end
    end
    out
end

# An arbitrary accessor or scorer can depend on another control. Compare full
# transported gradients, and compare each scorer at the same candidate in both
# frames. Finite differences of scores carry O(eps/h) roundoff.
_same_proxy(x,y) = all(i -> isapprox(x[i], y[i]; rtol=1e-8, atol=1e-9), 1:4)
_same_frame_value(x,y) = isapprox(x,y; rtol=32eps(Float64), atol=32eps(Float64))

function _transport_joint_history!(s, next, information, lpdf, old, previous_scores, weights)
    q, g, _ = _anchor_sample(s.anchors, s.supplied)
    qnew, gnew = similar(q), similar(g)
    _jointly_transport_halo!(lpdf, old, q, g, qnew, gnew)
    all(isfinite, qnew) && all(isfinite, gnew) || return false
    isnothing(_pool_roundtrip_rejection(old, reparametrizer(lpdf), q, qnew)) || return false
    scores = _anchor_scores(next, lpdf, qnew, gnew)
    all(obs -> all(isfinite, obs), scores) || return false
    # For an affected marginal use only observations in the installed chart.
    # An invariant marginal retains the full exact tree-weighted history.
    recharted = WeightedScaleAdaptation(size(q,1))
    for k in eachindex(weights)
        OnlineStatsBase.fit!(recharted, view(qnew,:,k), view(gnew,:,k); dw=weights[k])
    end
    for (before, after, mean, m2, refmean, refm2) in
        ((q, qnew, s.moments.position_mean, s.moments.position_m2, recharted.position_mean, recharted.position_m2),
         (g, gnew, s.moments.gradient_mean, s.moments.gradient_m2, recharted.gradient_mean, recharted.gradient_m2))
        for i in axes(before,1)
            if any(k -> !_same_frame_value(before[i,k], after[i,k]), axes(before,2))
                mean[i], m2[i] = refmean[i], refm2[i]
            end
        end
    end
    for j in axes(next,2)
        all(k -> _same_proxy(previous_scores[j,k], scores[j,k]), eachindex(weights)) && continue
        # Rebuild offsets, balance and curvature for a dependent scoring cell.
        # The previous information and the new local derivatives can differ
        # even if its installed control didn't move.
        loss = WeightedReparametrizationLoss()
        for k in eachindex(weights)
            x, grad, _, _ = scores[j,k]
            OnlineStatsBase.fit!(loss, (0.,x,grad); weight=weights[k], count=false)
        end
        next[2,j], next[3,j] = loss.mean_position, loss.mean_gradient
        scale = scale_estimate(loss)
        isfinite(scale) && scale > 0 && (next[4,j] = 2log(scale))
        H = view(information,:,:,j)
        H .= I(4)
        scratch = zeros(4,1)
        for k in eachindex(weights)
            _normal_equations!(scratch, view(information,:,:,j:j), 1, view(next,:,j:j), scores[j,k], weights[k])
        end
    end
    copyto!(q,qnew)
    copyto!(g,gnew)
    true
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
    balance = falses(size(next,2))
    moved = false
    for (j, (idx, rule)) in enumerate(ir.pairs)
        H = @view information[:,:,j]
        h = @view tree.information[:,:,j]
        grad = @view tree.gradient[:,j]
        all(isfinite, h) && all(isfinite, grad) || continue
        # A cell whose score is independent of centeredness uses the
        # ordinary diagonal estimator, also used when controls are fixed.
        iszero(h[1,1]) && continue
        fitted = H + h
        all(isfinite, fitted) || continue
        chol = cholesky(Symmetric(fitted); check=false)
        issuccess(chol) || continue
        delta = chol \ grad
        proposal = (@view s.parameters[:,j]) .- delta ./ max(1.0, abs(delta[4]))
        proposal[1] = clamp(proposal[1], 0.0, 1.0)
        factor = exp(proposal[4]/2)
        all(isfinite, proposal) && isfinite(factor) && factor > 0 || continue
        next[:,j] .= proposal
        H .= fitted
        balance[j] = true
        moved |= proposal[1] != rule.source.c
    end
    if moved
        old = snapshot_reparametrization(ir)
        q, g, weights = _anchor_sample(s.anchors, s.supplied)
        previous_scores = _anchor_scores(next, lpdf, q, g)
        try
            for (j, (idx, rule)) in enumerate(ir.pairs)
                ir.pairs[j] = idx => Reparametrization(rule.target, PartiallyCentered(next[1,j]), rule.args...)
            end
            _synchronize_scoring!(lpdf)
            proposed_point, failure = _checked_centering_point(lpdf, old, point)
            if isnothing(failure) && !_transport_joint_history!(s, next, information, lpdf, old, previous_scores, weights)
                failure = (; reason="a historical anchor could not be transported", fields=(;))
            end
            if !isnothing(failure)
                _restore_reparametrization!(ir, old)
                _synchronize_scoring!(lpdf)
                @warn "Rejected continuous centering update: $(failure.reason)" failure.fields...
                copyto!(scale, next_scale)
                return point, false
            end
            point = proposed_point
        catch
            _restore_reparametrization!(ir, old)
            _synchronize_scoring!(lpdf)
            rethrow()
        end
    end
    moved && _continuous_scales!(next_scale, s.moments, s.initial_scale, 1.0, scale)
    for (j, (idx, _)) in enumerate(ir.pairs)
        balance[j] && (next_scale[idx] = exp(next[4,j]/2))
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
    if !isempty(s.parameters)
        # First update can re-score *all* supplied rows in the installed
        # chart. Subsequent moves carry a bounded deterministic stratum.
        s.anchors = _FrameAnchors(Matrix{Float64}(P), Matrix{Float64}(G), s.supplied, 0)
    end
    point, _ = _joint_fit!(s, tree, lpdf, point, scale)
    if !isempty(s.parameters)
        n = min(_JOINT_ANCHORS, s.supplied)
        cols = n == 1 ? [1] : round.(Int, range(1, s.supplied; length=n))
        a = _FrameAnchors(length(scale), n + _JOINT_ANCHORS)
        a.seeds = n
        a.q[:,1:n] .= view(s.anchors.q,:,cols)
        a.g[:,1:n] .= view(s.anchors.g,:,cols)
        s.anchors = a
    end
    point
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
    isempty(s.parameters) || _anchor_point!(s.anchors, state.position_and_gradient)
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
    (; continuous_centering=(; s.parameters, s.information, s.moments, s.initial_scale, s.anchors,
        s.adapt_controls, s.trees, s.supplied, s.installations, s.frozen_after))
end
function _centering_result(leaves::_JointLeaves)
    s = leaves.schedule
    (; centering_schedule=:continuous, centering_installations=copy(s.installations),
       centering_frozen_after=s.frozen_after, adaptation_trees=s.trees, initializer_rows=s.supplied)
end
