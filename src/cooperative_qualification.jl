# Compare only disposable warm-up states. Each receiving conditional supplies
# the coordinate frame and the density for every comparison.
const _COOP_CHECK_DRAWS = 12
const _COOP_MAX_METRIC_REVIEWS = 2

function _qualification_charge!(chain, total, n)
    chain.total_evaluation_counter += n
    chain.round_evaluations += n
    Threads.atomic_add!(total, n)
end

function _pilot_level(chain, history)
    half = length(history) ÷ 2
    early, late = view(history, 1:half), view(history, half+1:length(history))
    level = median(late)
    # A d-dimensional normal has log-density variance d/2. Use the larger
    # of that reference spread and the observed robust spread on this target.
    spread = max(1.0, sqrt(chain.dimension / 2), 1.4826 * median(abs.(late .- level)))
    (; level, tolerance=6spread, drift=abs(level - median(early)))
end

function _pilot_neighbors(family::CoopFamily, i, n)
    family.pilot_partners[i]
end
_pilot_neighbors(target, i, n) = (mod1(i + k, n) for k in 1:min(8, n-1))

function _best_peer_level(run, chain)
    best = -Inf
    new_ir = reparametrizer(chain.lpdf)
    for j in _pilot_neighbors(run.lpdf, chain.chain_index, length(run.chains))
        peer = run.chains[j]
        anchor = peer.freeze.anchor
        isempty(anchor) && continue
        peer.status in (:piloting, :sampling, :done) || continue
        old_ir = reparametrizer(peer.lpdf)
        q = if has_reparametrization(old_ir) || has_reparametrization(new_ir)
            dest = similar(anchor)
            try
                _transport_position!(dest, similar(dest), new_ir, old_ir, anchor)
            catch err
                err isa DomainError || rethrow()
                continue
            end
            dest
        else
            copy(anchor)
        end
        all(isfinite, q) || continue
        _stop_requested(run) && return nothing
        pg = _evaluate_if_finite(chain.lpdf, q)
        _qualification_charge!(chain, run.total_evaluations, 1)
        chain.freeze.peer_evaluations += 1
        isnothing(pg) || (best = max(best, pg.ℓq))
    end
    best
end

# Review the movement scale in the chain's own frame. A scaled gradient
# based on the old kernel can look small precisely because that kernel does
# not move in the direction with the large gradient.
function _underresolved_scale(chain, scales)
    chain.scale isa Diagonal || return false
    for i in eachindex(scales)
        s = scales[i]
        isfinite(s) && s > 10chain.scale.diag[i] &&
            abs(s * chain.moments.gradient_mean[i]) > 6 && return true
    end
    false
end

function _review_pilot_metric!(chain, scale, cfg)
    if chain.freeze.reviews >= _COOP_MAX_METRIC_REVIEWS
        chain.status = :needs_replacement
        return
    end
    chain.freeze.reviews += 1
    chain.freeze.total_reviews += 1
    cc = chain.scale isa Diagonal ? _relative_condition(scale.diag, chain.scale) :
        _relative_metric_condition(chain.scale, scale)
    _restart!(chain, scale, cc, cfg)
    chain.freeze.warmup = max(chain.freeze.warmup, _COOP_CHECK_DRAWS)
end

function _accept_pilot!(chain)
    chain.status = :sampling
    chain.sampling_start = length(chain.acceptances)
    chain.freeze.probe = nothing
    empty!(chain.freeze.probe_history)
    chain.freeze.probe_moves = 0
end

function _check_pilot_positions!(run)
    # Score the entire barrier before any chain invalidates its endpoint by
    # restarting. Every worker count and iteration order sees the same anchors.
    pending = filter(c -> c.status === :piloting && c.freeze.pending, run.chains)
    peers = map(c -> _best_peer_level(run, c), pending)
    for (c, best) in zip(pending, peers)
        isnothing(best) && continue
        f = c.freeze
        f.pending = false
        local_level = _pilot_level(c, f.history)
        gap = best - local_level.level
        scales = _moment_scales(c.moments)
        suspect = gap > local_level.tolerance || local_level.drift > local_level.tolerance ||
            _underresolved_scale(c, scales)
        if suspect && iszero(f.reviews)
            _review_pilot_metric!(c, _complete_diagonal(scales, c.scale), run.cfg)
        elseif suspect || gap < -local_level.tolerance
            # An unavailable neighbor, or a persistent disagreement between
            # modes, calls for another random stream on this very conditional.
            c.status = :probing
        else
            _accept_pilot!(c)
        end
    end
end

function _probe_transition!(chain, cfg, total)
    f = chain.freeze
    target = CountingPosterior(chain.lpdf)
    if isnothing(f.probe)
        rng = Random.Xoshiro(rand(chain.rng, UInt64))
        scales = _moment_scales(chain.moments)
        init = (; position=copy(chain.position_and_gradient.q),
                  squared_scale=[isfinite(s) ? s^2 : 1.0 for s in scales])
        f.probes += 1
        try
            f.probe = ContinuousNUTSState(rng, target; init, cfg.target_acceptance_rate,
                max_tree_depth=cfg.algorithm.max_depth, blas_threads=nothing)
        catch err
            err isa DynamicHMC.DynamicHMCError || rethrow()
            f.probe_evaluations += target.count[]
            _qualification_charge!(chain, total, target.count[])
            chain.status = :needs_replacement
            return chain
        end
    end
    probe = f.probe
    previous = probe.position
    step = continuous_nuts_step(probe, target; blas_threads=nothing)
    f.probe_moves += Int(probe.position != previous)
    pg = _evaluate_if_finite(target, probe.position)
    push!(f.probe_history, isnothing(pg) ? -Inf : pg.ℓq)
    f.probe_evaluations += target.count[]
    _qualification_charge!(chain, total, target.count[])
    chain.total_transition_counter += 1
    chain.n_nonfinite_leaves += step.nonfinite
    f.warmup_transitions += 1
    f.divergences += Int(step.divergent)
    if length(f.probe_history) >= _COOP_CHECK_DRAWS
        local_level = _pilot_level(chain, f.history)
        witness = _pilot_level(chain, f.probe_history)
        tolerance = max(local_level.tolerance, witness.tolerance)
        # The witness is still tuning its step size. Actual moves show it
        # explored the target even when its early acceptance rates were low.
        if all(isfinite, f.probe_history) && f.probe_moves >= 2 &&
                probe.n_divergent < length(f.probe_history) ÷ 2 &&
                abs(witness.level - local_level.level) <= tolerance &&
                max(witness.drift, local_level.drift) <= tolerance
            _accept_pilot!(chain)
        else
            # Install only the probe's geometry. The designated production
            # chain stays at its own point and must qualify a new segment.
            _review_pilot_metric!(chain, probe.scale, cfg)
        end
    end
    chain
end
