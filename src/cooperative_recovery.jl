# Decisions at a barrier use the same pre-restart observations for every slot.
# A short warm-up or a small cluster alone is not evidence of failure.
_coop_divergent_fraction(c) = c.n_divergent / max(1, c.current_transition_counter)
function _coop_ess_per_draw(c)
    c.n_samples > 10 || return NaN
    ess = c.monitor_ess ? c.ess : _coordinate_ess(chain_draws(c))
    _min_ess(ess) / c.n_samples
end

function _coop_recovery_context(run, targets)
    chains, R = run.chains, run.clusters.responsibilities
    eligible = findall(_coop_geometry_current, chains)
    length(eligible) >= 3 || return nothing
    measured = filter(i -> chains[i].current_transition_counter > 10, eligible)
    length(measured) >= 2 || return nothing
    reference_step = median([chains[i].stepsize for i in measured])
    cluster = zeros(Int, length(chains))
    counts = zeros(Int, size(R, 2))
    for i in eligible
        k = argmax(view(R, i, :))
        cluster[i] = k
        counts[k] += 1
    end
    majority = argmax(counts)
    2counts[majority] > length(eligible) || return nothing
    # Most barriers need only the counts and step-size comparison. Fit donor
    # geometry (and compute any unmonitored ESS) only if someone may need it.
    candidate = any(measured) do i
        c = chains[i]
        _running(c) && (c.stepsize < reference_step / 10 || _coop_divergent_fraction(c) > 0.4 ||
            (counts[cluster[i]] == 1 && c.n_samples > 10 && isnothing(c.rescue_prior)))
    end
    candidate || return nothing
    donors = filter(measured) do i
        c = chains[i]
        cluster[i] == majority && c.stepsize >= reference_step / 10 &&
            _coop_divergent_fraction(c) <= 0.4 && c.moments.weight >= 2 &&
            c.n_samples > 10 && c.n_divergent_samples / c.n_samples <= 0.02
    end
    length(donors) >= 2 || return nothing
    weights = R[donors, majority]
    moments = _combine_moments!(CoopMoments(first(chains).dimension),
        (chains[i].moments for i in donors), weights)
    scales = _moment_scales(moments)
    any(isfinite, scales) || return nothing
    log_kappa = sum(w * chains[i].log_kappa for (w, i) in zip(weights, donors)) / sum(weights)
    efficiencies = filter(isfinite, [_coop_ess_per_draw(chains[i]) for i in donors])
    reference_efficiency = isempty(efficiencies) ? NaN : median(efficiencies)
    (; cluster, counts, majority, donors, moments, log_kappa, reference_step, reference_efficiency)
end

const _NO_COOP_RECOVERY = (; kind=:none, reason=nothing, extreme=false)
_coop_recovery_action(::Nothing, chain, batch) = _NO_COOP_RECOVERY
_coop_rescue_geometry(context, chain) =
    (; cluster=context.majority, context.donors, context.moments, context.log_kappa)

function _coop_recovery_action(context, chain, batch)
    k = context.cluster[chain.chain_index]
    k > 0 || return _NO_COOP_RECOVERY
    singleton = k != context.majority && context.counts[k] == 1
    _coop_recovery_action(chain, batch, context.reference_step, context.reference_efficiency, singleton)
end

function _coop_recovery_action(chain, batch, reference_step, reference_efficiency, singleton::Bool)
    chain.current_transition_counter > 10 || return _NO_COOP_RECOVERY
    small_step = chain.stepsize < reference_step / 10
    divergent = _coop_divergent_fraction(chain) > 0.4
    small_step || divergent || singleton || return _NO_COOP_RECOVERY
    efficiency = divergent ? NaN : _coop_ess_per_draw(chain)
    slow = efficiency < reference_efficiency / 10
    # A locally efficient minority mode can have a genuinely narrower scale.
    # Its fixed-kernel draws are direct counterevidence to step-size collapse.
    efficient = isfinite(efficiency) && efficiency >= reference_efficiency / 10
    collapsed = small_step && !efficient
    extreme = collapsed || divergent
    bad_draws = chain.n_samples > 10 && chain.n_divergent_samples / chain.n_samples > 0.02
    if !extreme && !(singleton && (bad_draws || slow) && isnothing(chain.rescue_prior))
        return _NO_COOP_RECOVERY
    end
    reason = divergent ? :divergences : collapsed ? :stepsize : bad_draws ? :divergent_draws : :low_ess_per_draw
    previous = isempty(chain.batches) ? (;) : last(chain.batches)
    replace = extreme && get(previous, :extreme, false) && previous.recovery === :rescue &&
        previous.batch == batch - 1
    (; kind=replace ? :replace : :rescue, reason, extreme)
end

# A replacement is a new chain at the old logical index. Only slot-level
# accounting and family membership survive; no draw, moment or fitted prior
# from an abandoned generation is input to the new one.
_coop_replacement_slot!(fresh, old, targets) = _coop_carry_freeze!(fresh, old)

# Save ordinary starts as data. Executable or opaque initializers belong to
# the caller, and are needed again only if a later barrier replaces that slot.
const _SUPPLY_COOP_INIT = (; supply_cooperative_init=true)
_checkpoint_cold_init(init) = _SUPPLY_COOP_INIT
_checkpoint_cold_init(init::Union{Missing,Real,AbstractVector{<:Real}}) = init
_checkpoint_cold_init(init::Distribution) = isbitstype(typeof(init)) ? init : _SUPPLY_COOP_INIT
_checkpoint_cold_init(init::AbstractVector) = Any[_checkpoint_cold_init(x) for x in init]
function _checkpoint_cold_init(init::NamedTuple)
    hasproperty(init, :position) && hasproperty(init, :squared_scale) || return _SUPPLY_COOP_INIT
    (; init.position, init.squared_scale)
end
_require_cold_init(init, index) = init
_require_cold_init(::typeof(_SUPPLY_COOP_INIT), index) = throw(ArgumentError(
    "chain $index needs replacement. Resume with the original init so the replacement uses its configured start."))

function _replace_coop_chain!(run, chain, reason)
    (; chain_index, generation, abandoned) = chain
    restarts = filter(b -> b.restart, chain.batches)
    # Pre-recovery checkpoints did not record the divergence count at a
    # restart. Preserve that uncertainty for a generation started by old code.
    generation_divergences = all(b -> haskey(b, :n_divergent), restarts) ?
        chain.n_divergent + sum(b -> b.n_divergent, restarts; init=0) : missing
    prior_evaluations = isempty(abandoned) ? 0 : last(abandoned).total_evaluation_counter
    summary = (; generation, reason, batch=run.batch, round=run.round, chain.stepsize,
        n_transitions=chain.current_transition_counter, chain.n_divergent, chain.n_samples,
        generation_divergences, generation_evaluations=chain.total_evaluation_counter - prior_evaluations,
        chain.n_divergent_samples, chain.total_evaluation_counter, chain.total_transition_counter,
        chain.n_restarts, batches=copy(chain.batches))
    if generation >= 2
        push!(abandoned, summary)
        chain.status = :recovery_failed
        _slot_failure_ends_run(run.lpdf) && (run.stop_reason = :recovery_failed)
        # Its checks leave the median the remaining slots still qualify against.
        append!(chain.freeze.abandoned_scores, chain.freeze.scores)
        empty!(chain.freeze.scores)
        reset!(chain.recording_lpdf)
        _reset_moments!(chain.moments)
        fill!(chain.ess, 0.0)
        chain.n_samples = chain.n_divergent_samples = 0
        chain.dropped_posterior_position = zeros(chain.dimension, 0)
        chain.dropped_posterior_gradient = zeros(chain.dimension, 0)
        chain.dropped_n_divergent_samples = 0
    else
        rng = Random.Xoshiro(rand(run.rng, UInt64))
        target = deepcopy(_coop_target(run.lpdf, chain_index))
        init = _coop_cold_init(run.init, chain_index, rng)
        frame = isnothing(run.reparam_lpdf) ? chain.lpdf : run.reparam_lpdf
        current_sources = reparam_sources(frame)
        fresh = _coop_chain(chain_index, rng, target, init, run.cfg; chain.progress,
            current_sources, chain.start_time, adapt_centering=!isnothing(run.reparam_lpdf))
        fresh.geometry_current = !isnothing(run.reparam_lpdf) || chain.geometry_current
        fresh.generation = generation + 1
        fresh.abandoned = push!(copy(abandoned), summary)
        fresh.total_evaluation_counter = chain.total_evaluation_counter
        fresh.total_transition_counter = chain.total_transition_counter
        _coop_replacement_slot!(fresh, chain, run.lpdf)
        run.chains[chain_index] = fresh
    end
    run.clusters = CoopClusters()
    run.last_pooled_ess = NaN
    nothing
end

function _replace_gate_failures!(run)
    for chain in run.chains
        chain.status === :needs_replacement || continue
        # A cold init may itself be expensive. Honour cancellation and hard
        # budgets before starting a new generation.
        _stop_requested(run) && break
        _replace_coop_chain!(run, chain, :first_draw_gate)
    end
end

# A previously observed chain cannot disappear from the agreement check by
# discarding its evidence and reaching target_ess before it has sampled again.
_has_coop_recovery(c) = c.generation > 1 || c.status in (:needs_replacement, :recovery_failed) ||
    any(b -> get(b, :recovery, :none) in (:rescue, :refit), c.batches)
_awaiting_recovery_draws(c) = c.n_samples <= 10 && _has_coop_recovery(c)
