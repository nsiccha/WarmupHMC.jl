# The schedule is saved at completed transitions. The two speculative summaries
# below live only for the duration of sample_tree and contain no leaf vectors.
mutable struct _CenteringSchedule
    persistence::Int
    max_updates::Union{Nothing,Int}
    winners::Vector{Float64}
    streaks::Vector{Int}
    installations::Vector{Int}
    frozen_after::Union{Nothing,Int}
end

mutable struct _CenteringLeaves{O,L}
    schedule::_CenteringSchedule
    prefix::O
    doubling::O
    linear::L
    seen::Int
    prefix_size::Int
    prefix_logmass::Float64
    doubling_logmass::Float64
end

function _check_centering_schedule(schedule, persistence, max_updates)
    schedule in (:window, :online) || throw(ArgumentError(
        "centering_schedule must be :window or :online (got $schedule)"))
    schedule === :window && !(isnothing(persistence) && isnothing(max_updates)) &&
        throw(ArgumentError("centering_persistence and centering_max_updates require centering_schedule=:online"))
    for (name, n) in ((:centering_persistence, persistence), (:centering_max_updates, max_updates))
        isnothing(n) || (n isa Integer && !(n isa Bool) && 0 < n <= typemax(Int)) ||
            throw(ArgumentError("$name must be a positive integer"))
    end
end

function _check_online_target(lpdf, adapt, evidence)
    ir = reparametrizer(lpdf)
    adapt && ir isa IndexedReparametrization && !isempty(ir.pairs) && isempty(ir.chains) &&
        all(p -> last(p).source isa PartiallyCentered, ir.pairs) ||
        throw(ArgumentError("online centering requires nonlinear_adapt=true and indexed constant centering pairs without Markov-chain controls"))
    candidate_scoring_plan(lpdf) isa WindowSelectionPlan && throw(ArgumentError(
        "WindowSelectionPlan selects at a window boundary; use centering_schedule=:window"))
    evidence === :nuts_weighted || throw(ArgumentError(
        "online centering requires nonlinear_evidence=:nuts_weighted"))
end

function _centering_leaves(lpdf, recorder, linear_source, schedule, persistence, max_updates,
                           adapt; saved=nothing)
    _check_centering_schedule(schedule, persistence, max_updates)
    d = LogDensityProblems.dimension(lpdf)
    schedule === :window && return NUTSLeaves(d)
    _check_online_target(lpdf, adapt, recorder.mode)
    n = length(recorder.online.pairs)
    policy = if isnothing(saved)
        _CenteringSchedule(something(persistence, 3), max_updates, fill(NaN, n),
                           zeros(Int, n), Int[], nothing)
    else
        (isnothing(persistence) || persistence == saved.persistence) &&
            (isnothing(max_updates) || max_updates == saved.max_updates) ||
            throw(ArgumentError("resume with the saved centering_persistence and centering_max_updates"))
        length(saved.winners) == length(saved.streaks) == n ||
            throw(DimensionMismatch("saved online centering state does not match the supplied pairs"))
        _CenteringSchedule(saved.persistence, saved.max_updates, saved.winners, saved.streaks,
                           saved.installations, saved.frozen_after)
    end
    # Only a caller selecting a buffered *linear* evidence source needs the raw nodes.
    linear = linear_source === :halo ? nothing : NUTSLeaves(d)
    summaries = ntuple(_ -> OnlineReparametrizer(reparametrizer(lpdf)), 2)
    _CenteringLeaves(policy, summaries..., linear, 0, 1, -Inf, -Inf)
end

function reset!(leaves::_CenteringLeaves)
    leaves.seen = 0
    leaves.prefix_size = 1
    leaves.prefix_logmass = leaves.doubling_logmass = -Inf
    if isnothing(leaves.schedule.frozen_after)
        reset!(leaves.prefix)
        reset!(leaves.doubling)
    end
    isnothing(leaves.linear) || reset!(leaves.linear)
    leaves
end

function _scale_mass!(loss::WeightedReparametrizationLoss, scale)
    iszero(scale) && return reset!(loss)
    loss.weight *= scale
    loss.weight2 *= scale^2
    loss.m2_position *= scale
    loss.m2_gradient *= scale
    loss.co_position_gradient *= scale
    loss
end
function _scale_mass!(or::OnlineReparametrizer, scale)
    for (_, item) in or.pairs
        _scale_mass!(item, scale)
    end
end

# A further visit proves that the preceding doubling succeeded. At the final
# boundary only stats.depth can distinguish a completed invalid doubling.
function _accept_doubling!(leaves)
    a, b = leaves.prefix_logmass, leaves.doubling_logmass
    log_switch = min(0.0, b - a)
    _scale_mass!(leaves.prefix, -expm1(log_switch))
    _scale_mass!(leaves.doubling, exp(log_switch))
    _merge_nonlinear!(leaves.prefix, leaves.doubling, nothing)
    reset!(leaves.doubling)
    leaves.prefix_logmass = logaddexp(a, b)
    leaves.doubling_logmass = -Inf
    leaves.prefix_size *= 2
end

function _observe_trajectory_leaf!(leaves::_CenteringLeaves, lpdf, z, dh)
    isnothing(leaves.linear) || record_leaf!(leaves.linear, z, dh)
    isnothing(leaves.schedule.frozen_after) || return
    leaves.seen == 2 * leaves.prefix_size && _accept_doubling!(leaves)
    leaves.seen += 1
    if leaves.seen == 1
        isfinite(dh) || throw(ArgumentError("the initial leaf must have finite log weight"))
        leaves.prefix_logmass = dh
        summary, weight = leaves.prefix, 1.0
    else
        # Divergent/nonfinite evaluations have no candidate observation.
        isfinite(dh) || return
        old = leaves.doubling_logmass
        total = logaddexp(old, dh)
        _scale_mass!(leaves.doubling, exp(old - total))
        leaves.doubling_logmass = total
        summary, weight = leaves.doubling, exp(dh - total)
    end
    iszero(weight) || _observe_nonlinear!(candidate_scoring_plan(lpdf), reparametrizer(lpdf),
                                         summary, z.Q.q, z.Q.∇ℓq, weight)
end

function finalize_leaf_weights!(leaves::_CenteringLeaves, depth)
    isnothing(leaves.linear) || finalize_leaf_weights!(leaves.linear, depth)
    isnothing(leaves.schedule.frozen_after) || return
    valid = 1 << depth
    if valid == 2 * leaves.prefix_size && leaves.seen == valid
        _accept_doubling!(leaves)
    end
    valid == leaves.prefix_size || error("inconsistent NUTS tree depth $depth after $(leaves.seen) leaves")
end

function record_nonlinear!(recorder::NonlinearRecorder, lpdf, leaves::_CenteringLeaves, stepsize)
    isnothing(leaves.schedule.frozen_after) || return
    weight = _trajectory_weight(recorder.trajectory_weighting, stepsize)
    _scale_mass!(leaves.prefix, weight)
    _mark_group!(leaves.prefix)
    _merge_nonlinear!(recorder.online, leaves.prefix, nothing)
end

_linear_leaves(leaves) = leaves
_linear_leaves(leaves::_CenteringLeaves) = leaves.linear
_window_centering(leaves) = true
_window_centering(::_CenteringLeaves) = false
_centering_checkpoint(leaves) = (;)
function _centering_checkpoint(leaves::_CenteringLeaves)
    s = leaves.schedule
    (; online_centering=(; s.persistence, s.max_updates, s.winners, s.streaks,
                           s.installations, s.frozen_after))
end
_centering_result(leaves) = (;)
_centering_result(leaves::_CenteringLeaves) = (;
    centering_installations=copy(leaves.schedule.installations),
    centering_frozen_after=leaves.schedule.frozen_after)

_begin_centering_tree!(leaves, state) = reset!(leaves)
function _begin_centering_tree!(leaves::_CenteringLeaves, state)
    s = leaves.schedule
    # Leave a complete adaptation observation after the last possible change,
    # then finalize the step size in that frame. Freeze across later restarts.
    if isnothing(s.frozen_after) && state.current_transition_counter >= state.stepsize_adaptation_limit - 1
        s.frozen_after = state.total_transition_counter - 1
    end
    reset!(leaves)
end

# Only stable, finite winners propose changes. Compute them against one frame,
# then validate all proposed controls in a single transaction.
function _stable_centering_pairs!(s, ir, online)
    proposed = copy(ir.pairs)
    for (k, (index, candidates)) in enumerate(online.pairs)
        OnlineStatsBase.nobs(candidates) > 2 || continue
        best = minimizer(candidates)
        c = isnothing(best) ? NaN : first(best).c
        s.streaks[k] = c == s.winners[k] ? s.streaks[k] + 1 : 1
        s.winners[k] = c
        if !isnothing(best) && s.streaks[k] >= s.persistence
            _, rule = ir.pairs[k]
            proposed[k] = index => Reparametrization(rule.target, first(best), rule.args...)
        end
    end
    proposed
end

_update_centering!(state, leaves) = nothing
function _update_centering!(state, leaves::_CenteringLeaves)
    s = leaves.schedule
    isnothing(s.frozen_after) || return
    recording = state.recording_lpdf
    lpdf = parent(recording)
    ir = reparametrizer(lpdf)
    proposed = _stable_centering_pairs!(s, ir, state.nonlinear_recorder.online)
    proposed == ir.pairs && return
    old = snapshot_reparametrization(ir)
    positions, gradients = recording.halo_position, recording.halo_gradient
    new_positions, new_gradients = similar(positions), similar(gradients)
    try
        ir.pairs .= proposed
        _arbitrate_blocks!(lpdf, old, positions, gradients)
        ir.pairs == old.pairs && return
        _synchronize_scoring!(lpdf)
        point = _validated_transport!(lpdf, old, positions, gradients, new_positions,
                                      new_gradients, state.position_and_gradient)
        ir.pairs == old.pairs && return  # rejected; old-frame evidence remains usable
        copyto!(positions, new_positions)
        copyto!(gradients, new_gradients)
        state.position_and_gradient = point
    catch
        _restore_reparametrization!(ir, old)
        _synchronize_scoring!(lpdf)
        rethrow()
    end

    push!(s.installations, state.total_transition_counter)
    if length(s.installations) == s.max_updates
        s.frozen_after = state.total_transition_counter
    end
    reset!(state.nonlinear_recorder)
    fill!(s.winners, NaN)
    fill!(s.streaks, 0)
    reset!(state.transformed_adaptation)
    # The transported halo remains available for the next ordinary metric fit.
    # Start the new nonlinear coordinates with a unit Euclidean metric.
    fill!(state.scale_options.diagonal.diag, 1)
    state.active_transformation = :diagonal
    state.kinetic_energy = state.energy_options.diagonal
    state.stepsize_state = DynamicHMC.initial_adaptation_state(
        state.stepsize_adaptation, DynamicHMC.final_ϵ(state.stepsize_state))
    state.stepsize = DynamicHMC.current_ϵ(state.stepsize_state)
end
