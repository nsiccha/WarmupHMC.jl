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
    weights::Vector{Float64}
    live_weight::Float64
    seeds::Int
    seen::Int
end
_FrameAnchors(d, n) = _FrameAnchors(zeros(d,n), zeros(d,n), zeros(n), 0., 0, 0)

function _anchor_point!(a, point, weight)
    a.seen += 1
    j = a.seeds + mod1(a.seen, _JOINT_ANCHORS)
    a.weights[j] = weight
    a.live_weight += weight
    a.q[:,j] .= point.q
    a.g[:,j] .= point.∇ℓq
end

function _anchor_sample(a)
    live = min(a.seen, _JOINT_ANCHORS)
    cols = 1:a.seeds+live
    # The seed and trajectory strata retain their own total weights; neither
    # a large ensemble nor a long chain acquires an extra vote from thinning.
    w = copy(view(a.weights,cols))
    if live > 0
        retained = view(w,a.seeds+1:length(w))
        retained .*= a.live_weight / sum(retained)
    end
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
        all(p -> last(p).source.c isa AbstractFloat && 0 <= last(p).source.c <= 1, ir.pairs) ||
            throw(ArgumentError("continuous centering needs floating-point source controls in [0, 1]"))
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
    if hasproperty(ensemble, :source_indices)
        collect(ensemble.source_indices) == first.(ir.pairs) ||
            throw(ArgumentError("initializer ensemble source_indices differ from the target's ordered coordinate indices"))
    end
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
    x.moments.weight *= w
    x.moments.weight2 *= w^2
    x.moments.position_m2 .*= w
    x.moments.gradient_m2 .*= w
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
function _direct_joint_args(ir, model, rule, idx, q, g)
    isnothing(model[]) && (model[] = last(ir(q)))
    _framed_rargs(rule, q[idx], g[idx], _pair_frames(rule, model[]))
end

function _joint_observation(::DirectCandidateScoring, frame, j, idx, rule, c, q, g, ir, model)
    reparam(PartiallyCentered(c), _direct_joint_args(ir, model, rule, idx, q, g)...)
end
function _joint_observation(plan::CandidateScoringPlan, frame, j, idx, rule, c, q, g, ir, model)
    obs = plan.score(frame, j, idx, rule, PartiallyCentered(c))
    isnothing(obs) ? _joint_observation(DIRECT_CANDIDATE_SCORING, frame, j, idx, rule, c, q, g, ir, model) : obs
end

function _joint_scores(::DirectCandidateScoring, frame, j, idx, rule, c, q, g, ir, model)
    source, xi, gi, location, log_scale = _direct_joint_args(ir, model, rule, idx, q, g)
    _, x, s = reparam(PartiallyCentered(c), source, xi, gi, location, log_scale)
    x, s, location + log_scale*(x - c*location), -log_scale*s
end

function _joint_scores(plan::CandidateScoringPlan, frame, j, idx, rule, c, q, g, ir, model)
    h = cbrt(eps(Float64))
    lo, hi = max(0.0, c-h), min(1.0, c+h)
    observations = map((c, lo, hi)) do x
        _joint_observation(plan, frame, j, idx, rule, x, q, g, ir, model)
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

function _map_joint_marginal(before, after, weights, history_mean, history_m2, exact)
    paired = WeightedReparametrizationLoss()
    for k in eachindex(weights)
        OnlineStatsBase.fit!(paired, (0.,before[k],after[k]); weight=weights[k], count=false)
    end
    # A positive spread cannot be estimated from one anchor or a constant
    # marginal. Preserve the historical spread and apply the observed shift.
    ratio = sqrt(max(0., paired.m2_gradient) / max(0., paired.m2_position))
    scale = isfinite(ratio) && ratio > 0 ? copysign(ratio, paired.co_position_gradient) : 1.
    exact && return (; mean=paired.mean_gradient, m2=paired.m2_gradient, scale)
    (; mean=paired.mean_gradient + scale*(history_mean-paired.mean_position),
       m2=scale^2*history_m2, scale)
end

function _anchor_information(parameters, scores, weights)
    info = reshape(Matrix{Float64}(I,4,4),4,4,1)
    grad = zeros(4,1)
    for k in eachindex(weights)
        _normal_equations!(grad, info, 1, parameters, scores[k], weights[k])
    end
    view(info,:,:,1)
end

function _transport_joint_history!(s, next, information, lpdf, old, previous_scores, q, g, weights)
    qnew, gnew = similar(q), similar(g)
    _jointly_transport_halo!(lpdf, old, q, g, qnew, gnew)
    all(isfinite, qnew) && all(isfinite, gnew) || return false
    isnothing(_pool_roundtrip_rejection(old, reparametrizer(lpdf), q, qnew)) || return false
    scores = _anchor_scores(next, lpdf, qnew, gnew)
    all(obs -> all(isfinite, obs), scores) || return false
    # At initialization the paired samples cover every row. Later they calibrate
    # an affine marginal map for the full discounted history, including old
    # trees whose individual positions and gradients are no longer stored.
    exact = iszero(s.anchors.seen)
    moments = deepcopy(s.moments)
    for (before, after, means, m2s) in
        ((q, qnew, moments.position_mean, moments.position_m2),
         (g, gnew, moments.gradient_mean, moments.gradient_m2))
        for i in axes(before,1)
            source, target = view(before,i,:), view(after,i,:)
            all(k -> _same_frame_value(source[k], target[k]), eachindex(weights)) && continue
            mapped = _map_joint_marginal(source, target, weights, means[i], m2s[i], exact)
            means[i], m2s[i] = mapped.mean, mapped.m2
            isfinite(mapped.mean) && isfinite(mapped.m2) || return false
        end
    end
    for j in axes(next,2)
        before, after = view(previous_scores,j,:), view(scores,j,:)
        all(k -> _same_proxy(before[k], after[k]), eachindex(weights)) && continue
        # Rechart the parameter offsets and balance from paired score moments.
        # Compare Jacobian information at the old and new offsets in the
        # corresponding charts. A congruence carries the full previous curvature.
        old_parameters = copy(view(next,:,j:j))
        x = _map_joint_marginal(first.(before), first.(after), weights, next[2,j], 0., exact)
        gscore = _map_joint_marginal(getindex.(before,2), getindex.(after,2), weights, next[3,j], 0., exact)
        next[2,j], next[3,j] = x.mean, gscore.mean
        if exact
            x.m2 > 0 && gscore.m2 > 0 && (next[4,j] = log(x.m2/gscore.m2)/2)
        else
            next[4,j] += log(abs(x.scale)) - log(abs(gscore.scale))
        end
        all(isfinite, view(next,:,j)) && isfinite(exp(next[4,j]/2)) && exp(next[4,j]/2) > 0 || return false
        new_info = _anchor_information(view(next,:,j:j), after, weights)
        H = view(information,:,:,j)
        if exact
            H .= new_info
        else
            old_info = _anchor_information(old_parameters, before, weights)
            oldchol = cholesky(Symmetric(old_info); check=false)
            newchol = cholesky(Symmetric(new_info); check=false)
            issuccess(oldchol) && issuccess(newchol) || return false
            transform = newchol.L / oldchol.L
            H .= Symmetric(transform * H * transform')
        end
        all(isfinite, H) && isposdef(Symmetric(H)) || return false
    end
    s.moments = moments
    copyto!(q,qnew)
    copyto!(g,gnew)
    true
end

# The live diagonal is the sampler's factor L; inverse mass = L * L'.
# The current point and all chart-dependent optimizer parameters publish as
# one transaction. On rejection only ordinary evidence in the old frame advances.
function _joint_fit!(s, tree, lpdf, point, scale; batch=nothing)
    _merge_scale_moments!(s.moments, tree.moments)
    prior_weight = isnothing(batch) ? 1. : 0.
    next_scale = similar(scale)
    _continuous_scales!(next_scale, s.moments, s.initial_scale, prior_weight, scale)
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
        if !isnothing(batch)
            next[:,j] .= view(batch,:,j)
            H .+= h
            balance[j] = true
            moved |= next[1,j] != rule.source.c
            continue
        end
        # A cell whose score is independent of centeredness uses the
        # ordinary diagonal estimator, also used when controls are fixed.
        iszero(h[1,1]) && continue
        fitted = H + h
        all(isfinite, fitted) || continue
        chol = cholesky(Symmetric(fitted); check=false)
        issuccess(chol) || continue
        delta = chol \ grad
        proposal = (@view s.parameters[:,j]) .- delta ./ max(1.0, abs(delta[4]))
        proposal[1] = oftype(rule.source.c, clamp(proposal[1], 0.0, 1.0))
        factor = exp(proposal[4]/2)
        all(isfinite, proposal) && isfinite(factor) && factor > 0 || continue
        next[:,j] .= proposal
        H .= fitted
        balance[j] = true
        moved |= proposal[1] != rule.source.c
    end
    if moved
        old = snapshot_reparametrization(ir)
        q, g, weights = _anchor_sample(s.anchors)
        previous_scores = _anchor_scores(next, lpdf, q, g)
        try
            for (j, (idx, rule)) in enumerate(ir.pairs)
                ir.pairs[j] = idx => Reparametrization(rule.target, PartiallyCentered(oftype(rule.source.c, next[1,j])), rule.args...)
            end
            _synchronize_scoring!(lpdf)
            proposed_point, failure = _checked_centering_point(lpdf, old, point)
            if isnothing(failure) && !_transport_joint_history!(s, next, information, lpdf, old, previous_scores, q, g, weights)
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
    moved && _continuous_scales!(next_scale, s.moments, s.initial_scale, prior_weight, scale)
    for (j, (idx, _)) in enumerate(ir.pairs)
        balance[j] && (next_scale[idx] = exp(next[4,j]/2))
    end
    copyto!(s.parameters, next)
    copyto!(s.information, information)
    copyto!(scale, next_scale)
    point, moved
end

# Once c is fixed the minimizing offsets are the two sample means and
# exp(r) = sd(x)/sd(g). Optimize the resulting one-dimensional profile using
# only cached scores, then build the information matrix once at the fit.
function _batch_profile(lpdf, j, P, G, c)
    ir, plan = reparametrizer(lpdf), candidate_scoring_plan(lpdf)
    idx, rule = ir.pairs[j]
    fit = WeightedReparametrizationLoss()
    for k in axes(P,2)
        q, g = view(P,:,k), view(G,:,k)
        frame = _joint_frame(plan, ir, q, g)
        model = Ref{Union{Nothing,Vector{Float64}}}(nothing)
        obs = _joint_observation(plan, frame, j, idx, rule, c, q, g, ir, model)
        OnlineStatsBase.fit!(fit, obs; count=false)
    end
    fit
end

# The direct scorer uses the model-space parents of a cached row at every
# candidate c. These parents stay fixed until the batch has finished fitting.
# Project each row once across all cells, and retain just one cell's scalar
# arguments during its search. A custom scorer keeps its transient frame API.
_batch_model_positions(::CandidateScoringPlan, ir, P) = nothing
_batch_model_positions(::DirectCandidateScoring, ir, P) =
    [convert(Vector{Float64}, last(ir(q))) for q in eachcol(P)]
_batch_cell_profile(::Nothing, lpdf, j, P, G) = c -> _batch_profile(lpdf, j, P, G, c)
function _batch_cell_profile(model_positions, lpdf, j, P, G)
    idx, rule = reparametrizer(lpdf).pairs[j]
    args = [_framed_rargs(rule, P[idx,k], G[idx,k], _pair_frames(rule, model_positions[k]))
            for k in axes(P,2)]
    function profile(c)
        candidate = PartiallyCentered(c)
        fit = WeightedReparametrizationLoss()
        for row in args
            OnlineStatsBase.fit!(fit, reparam(candidate, row...); count=false)
        end
        fit
    end
end

function _batch_control(profile, start)
    loss(c) = begin
        f = profile(c)
        v = sqrt(max(0., f.m2_position)) * sqrt(max(0., f.m2_gradient))
        isfinite(v) ? v : Inf
    end
    best, bestloss = start, loss(start)
    nodes = range(0., 1.; length=9)
    costs = loss.(nodes)
    for k in eachindex(nodes)
        c, value = nodes[k], costs[k]
        left, right = max(1,k-1), min(length(nodes),k+1)
        if value < bestloss
            best, bestloss = c, value
        end
        if value <= min(costs[left], costs[right]) && value < max(costs[left], costs[right])
            # Golden-section refinement of an interior minimum. Always compare
            # both boundaries and the supplied control, including on a flat fit.
            a, b = nodes[left], nodes[right]
            t = (sqrt(5.) - 1)/2
            x, y = b-t*(b-a), a+t*(b-a)
            fx, fy = loss(x), loss(y)
            while b-a > sqrt(eps(Float64))
                if fx <= fy
                    b, y, fy = y, x, fx
                    x = b-t*(b-a)
                    fx = loss(x)
                else
                    a, x, fx = x, y, fy
                    y = a+t*(b-a)
                    fy = loss(y)
                end
            end
            c, value = fx <= fy ? (x,fx) : (y,fy)
        end
        if value < bestloss
            best, bestloss = c, value
        end
    end
    best
end

function _batch_offsets!(params, j, fit, fallback)
    all(isfinite, (fit.mean_position, fit.mean_gradient)) || throw(ArgumentError(
        "nonfinite initializer scoring observations at pair $j"))
    params[2,j], params[3,j] = fit.mean_position, fit.mean_gradient
    factor = scale_estimate(fit)
    params[4,j] = 2log(isfinite(factor) && factor > 0 ? factor : fallback)
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
    s.supplied = size(P,2)
    for k in axes(P,2)
        OnlineStatsBase.fit!(s.moments, view(P,:,k), view(G,:,k))
    end
    _continuous_scales!(scale, s.moments, s.initial_scale, 0., scale)
    s.initial_scale .= scale
    isempty(s.parameters) && return point
    batch = copy(s.parameters)
    model_positions = _batch_model_positions(candidate_scoring_plan(lpdf), ir, P)
    for (j, (idx, rule)) in enumerate(ir.pairs)
        profile = _batch_cell_profile(model_positions, lpdf, j, P, G)
        start = rule.source.c
        _batch_offsets!(s.parameters, j, profile(start), scale[idx])
        c = oftype(start, _batch_control(profile, start))
        batch[1,j] = c
        _batch_offsets!(batch, j, profile(c), scale[idx])
    end
    tree = leaves.prefix
    for k in axes(P,2)
        _visit_joint_scores(batch, lpdf, view(P,:,k), view(G,:,k)) do j, obs
            _normal_equations!(tree.gradient, tree.information, j, batch, obs, 1.)
        end
    end
    # The first update can re-score every supplied row in the installed chart.
    s.anchors = _FrameAnchors(Matrix{Float64}(P), Matrix{Float64}(G), ones(s.supplied), 0., s.supplied, 0)
    point, _ = _joint_fit!(s, tree, lpdf, point, scale; batch)
    s.initial_scale .= scale
    n = min(_JOINT_ANCHORS, s.supplied)
    cols = n == 1 ? [1] : round.(Int, range(1, s.supplied; length=n))
    a = _FrameAnchors(length(scale), n + _JOINT_ANCHORS)
    a.seeds = n
    a.weights[1:n] .= s.supplied/n
    a.q[:,1:n] .= view(s.anchors.q,:,cols)
    a.g[:,1:n] .= view(s.anchors.g,:,cols)
    s.anchors = a
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

function _update_centering!(state, leaves::_JointLeaves, trajectory_stepsize)
    s = leaves.schedule
    scale = state.scale_options.diagonal.diag
    weight = _trajectory_weight(state.nonlinear_recorder.trajectory_weighting, trajectory_stepsize)
    # Discount historical tree and initializer data in the same units. Keep
    # the unit ridge on the Gauss–Newton solve as a numerical regularizer.
    decay = exp2(-1 / _DEFAULT_CONTINUOUS_HALF_LIFE)
    _forget!(s.moments, decay)
    for H in eachslice(s.information; dims=3)
        H .*= decay
        for k in axes(H,1)
            H[k,k] += 1 - decay
        end
    end
    s.anchors.weights .*= decay
    s.anchors.live_weight *= decay
    weight == 1 || _scale_mass!(leaves.prefix, weight)
    isempty(s.parameters) || _anchor_point!(s.anchors, state.position_and_gradient, weight)
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

# Capture paired source-frame positions and gradients before finalization maps
# the public posterior positions to model coordinates. An interrupted warm-up
# with no retained draws can still hand its current point to an initializer.
_centering_handoff(leaves, state) = (;)
function _centering_handoff(::_JointLeaves, state)
    ir = reparametrizer(state.lpdf)
    point = state.position_and_gradient
    recorder = state.recording_lpdf
    frozen = !isnothing(recorder.leaves.schedule.frozen_after)
    ensemble = if ir isa IndexedReparametrization && isempty(ir.chains) &&
                  all(p -> first(p) isa Integer && last(p) isa Reparametrization && last(p).source isa PartiallyCentered, ir.pairs)
        retained = !isempty(recorder.posterior_position)
        positions = retained ? Matrix(recorder.posterior_position) : reshape(copy(point.q),:,1)
        gradients = retained ? Matrix(recorder.posterior_gradient) : reshape(copy(point.∇ℓq),:,1)
        (; positions, gradients, source_indices=Int[first(p) for p in ir.pairs],
           source_controls=[last(p).source.c for p in ir.pairs])
    end
    init = (; position=copy(point.q), squared_scale=abs2.(state.scale_options.diagonal.diag))
    (; centering_handoff=(; init, ensemble, stepsize=state.stepsize, frozen,
                           reparam_sources=reparam_sources(state.lpdf)))
end
