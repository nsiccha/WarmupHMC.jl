# Cooperative multi-chain sampler.
#
# `adaptive_warmup_mcmc` (src/adaptive_warmup_mcmc.jl) runs ONE chain as a
# sequence of warm-up windows; at each window boundary (the "checkpoint") it
# decides restart-warmup vs keep-sampling from the marginal-scale condition
# number. This file makes that per-window step RESUMABLE (`CooperativeChain` +
# `advance_window!`) so a scheduler can, at each checkpoint, keep `n_cores`
# busy by choosing among {continue this chain, resume a parked chain, start a
# new chain} (and park a stuck one) to maximize joint ESS per unit time.
#
# The low-level pieces (`initialize_mcmc`, the `scale_options`/`energy_options`
# construction, the recording pipeline, `find_reparametrization!`,
# `update_loss!`) are reused verbatim from adaptive_warmup_mcmc.jl; the
# monolith there is deliberately left untouched (it is test-covered and
# dev-pathed into Bruno). De-duplication can follow once this path is green.

"""
    CooperativeChain

Resumable state of a single adaptive-warmup NUTS chain. Holds everything the
outer window loop of `adaptive_warmup_mcmc` mutates, so the chain can be
advanced exactly one window at a time via [`advance_window!`](@ref), paused,
and resumed to an identical state.

`status` is one of `:warming` (last checkpoint restarted warm-up),
`:sampling` (last checkpoint kept the kernel and is collecting draws),
`:done` (collected `n_draws`), or `:stuck` (marked for parking by the
scheduler).
"""
mutable struct CooperativeChain{R,L,RL,A,SA,SO,EO,NT}
    # --- configuration (set once) ---
    const rng::R
    # Stable 1-based identity of this chain, equal to its index into `rngs`.
    # `_plan!` only ever increments `n_started`, so an index is never reused: a
    # replaced chain takes the NEXT rng rather than a freed slot. That is what
    # makes an on-disk `chain_<i>/` layout safe — `chain_3/cp_window_5.jls` and
    # `chain_3/cp_window_2.jls` always belong to the same chain. 0 means unset.
    const chain_index::Int
    const lpdf::L
    const recording_lpdf::RL           # RecordingPosterior2 wrapping lpdf
    const algorithm::A                 # DynamicHMC.NUTS
    const stepsize_adaptation::SA      # DynamicHMC.DualAveraging
    const dimension::Int
    const recording_target::Int
    const stepsize_adaptation_limit::Int
    const variance_cond_target::Float64
    const nonlinear_adapt::Bool
    const monitor_ess::Bool
    const n_draws::Int
    const max_window_evaluations::Int  # cap on the per-window eval budget (keeps checkpoints frequent)
    const scale_options::SO
    const energy_options::EO
    const kwargs::NT
    const start_time::UInt64
    # --- mutable window-loop state ---
    position_and_gradient
    active_transformation::Symbol
    # Untyped, for the same reason as `AWMState.kinetic_energy`: reassigned at
    # every window boundary across `energy_options` entries, which are
    # heterogeneously typed (each transformation yields a distinct
    # GaussianKineticEnergy type), so no single concrete type fits. A concrete
    # `::K` here made every restart that switched transformation throw
    # `MethodError: Cannot convert`.
    kinetic_energy
    stepsize::Float64
    stepsize_state
    n_evaluations::Int                 # current window's gradient-eval budget
    variance_memory::Vector{Float64}
    variance_position
    variance_gradient
    variance_cond::Float64
    scale_changes::Vector{Float64}
    total_evaluation_counter::Int
    outer_counter::Int
    current_transition_counter::Int
    total_transition_counter::Int
    n_divergent::Int
    n_divergent_samples::Int
    steps_per_draw
    ess::Vector{Float64}
    restart::Bool
    n_samples::Int
    status::Symbol
    # Why this chain was abandoned, recorded AT the moment `status` became `:stuck`
    # rather than re-derived later — a stored decision cannot drift from the
    # decision that was actually acted on. `nothing` while the chain is alive.
    stuck_reason::Union{Nothing,Symbol}
    checkpoints::Vector{NamedTuple}     # per-window log for the scheduler/selector
    # Draws DROPPED at the most recent restart — same contract as
    # `AWMState.dropped_posterior_position`, and for the same reason: this chain's
    # payload is built in `_release!`, i.e. after `advance_window!` has already
    # emptied the recorder, so a restarting window would otherwise checkpoint zero
    # draws. Inert: nothing reads them back.
    dropped_posterior_position::Matrix{Float64}
    dropped_posterior_gradient::Matrix{Float64}
    dropped_n_divergent_samples::Int
end

"Positions collected as posterior draws so far (dimension × n_samples)."
chain_draws(chain::CooperativeChain) = chain.recording_lpdf.posterior_position
"Number of posterior draws collected so far."
n_chain_draws(chain::CooperativeChain) = size(chain_draws(chain), 2)

"""
    cooperative_chain(rng, lpdf; n_draws, n_evaluations, recording_target,
                      stepsize_adaptation_limit, target_acceptance_rate,
                      max_tree_depth, init, variance_cond_target,
                      nonlinear_adapt, monitor_ess, progress, kwargs...)

Build a [`CooperativeChain`](@ref) ready to run its first warm-up window. Mirrors
the setup section of `adaptive_warmup_mcmc` (same defaults, same RNG-consuming
call order), so `run_chain!` on the result reproduces the single-chain result.
"""
cooperative_chain(
    rng, lpdf;
    chain_index=0,
    n_draws=1000,
    n_evaluations=1000,
    recording_target=1000,
    stepsize_adaptation_limit=50,
    target_acceptance_rate=.8,
    max_tree_depth=10,
    init=missing,
    variance_cond_target=2.,
    nonlinear_adapt=true,
    monitor_ess=false,
    max_window_evaluations=typemax(Int),
    progress=nothing,
    kwargs...
) = begin
    start_time = time_ns()
    stepsize_adaptation = DynamicHMC.DualAveraging(δ=target_acceptance_rate)
    algorithm = DynamicHMC.NUTS(;max_depth=max_tree_depth)
    dimension = LogDensityProblems.dimension(lpdf)
    recorder = LimitedRecorder2(recording_target)
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    scale_options = (;
        diagonal=_initial_diagonal_scale(squared_scale),
        pathfinder=_initial_pathfinder_scale(squared_scale, dimension),
        adaptive=MatrixFactorization(SuccessiveReflections(dimension), Diagonal(ones(dimension)))
    )
    energy_options = map(scale_options) do L
        DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))
    end
    active_transformation = :pathfinder
    kinetic_energy = energy_options[active_transformation]
    variance_memory = zeros(dimension)
    variance_position = OnlineStatsBase.Group([OnlineStatsBase.Variance() for i in 1:dimension])
    variance_gradient = OnlineStatsBase.Group([OnlineStatsBase.Variance() for i in 1:dimension])
    # Find an initial stepsize (consumes rng via rand_p, exactly as adaptive does).
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    position_gradient_and_momentum = DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy))
    stepsize = DynamicHMC.find_initial_stepsize(
        DynamicHMC.InitialStepsizeSearch(),
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf), position_gradient_and_momentum
        )
    )
    stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
    CooperativeChain(
        rng, chain_index, lpdf, recording_lpdf, algorithm, stepsize_adaptation, dimension,
        recording_target, stepsize_adaptation_limit, variance_cond_target,
        nonlinear_adapt, monitor_ess, n_draws, max_window_evaluations,
        scale_options, energy_options, NamedTuple(kwargs), start_time,
        position_and_gradient, active_transformation, kinetic_energy,
        stepsize, stepsize_state, n_evaluations, variance_memory,
        variance_position, variance_gradient, Inf, Float64[],
        0, 0, 0, 0, 0, 0, OnlineStatsBase.Mean(), zeros(dimension),
        true, 0, :warming, nothing, NamedTuple[],
        Matrix{Float64}(undef, dimension, 0), Matrix{Float64}(undef, dimension, 0), 0,
    )
end

"""
    advance_window!(chain) -> Symbol

Run exactly ONE warm-up/sampling window of `chain` and return at the
checkpoint, updating `chain.status`. One call == one iteration of the outer
loop of `adaptive_warmup_mcmc`: sample transitions until the window's
gradient-eval budget is hit (adapting the step size for the first
`stepsize_adaptation_limit` transitions, collecting draws thereafter), then at
the checkpoint recompute the marginal-scale condition number, and — unless the
chain is already `:done` — double the budget and, if still restarting,
re-select the linear transformation.

Returns `chain.status` (`:done`, `:sampling`, or `:warming`). A no-op returning
`:done` if the chain has already collected `n_draws`.
"""
advance_window!(chain::CooperativeChain) = begin
    n_chain_draws(chain) >= chain.n_draws && return chain.status = :done
    (;rng, algorithm, recording_lpdf, stepsize_adaptation, stepsize_adaptation_limit,
      n_draws, recording_target, variance_cond_target, nonlinear_adapt, monitor_ess,
      scale_options, energy_options, dimension) = chain
    (;halo_position, halo_gradient, posterior_position, posterior_gradient) = recording_lpdf

    chain.outer_counter += 1
    hamiltonian = DynamicHMC.Hamiltonian(chain.kinetic_energy, recording_lpdf)
    current_evaluation_counter = 0
    while size(posterior_position, 2) < n_draws && current_evaluation_counter < chain.n_evaluations
        chain.current_transition_counter += 1
        chain.total_transition_counter += 1
        reset!(recording_lpdf.leaves)   # per-transition: fresh leaf buffer before the tree (dev's protocol)
        chain.position_and_gradient, stats = DynamicHMC.sample_tree(
            rng, algorithm, hamiltonian, chain.position_and_gradient, chain.stepsize
        )
        finalize_leaf_recording!(recording_lpdf, stats.depth)   # sample one leaf ∝ proper weight → halo
        chain.total_evaluation_counter += stats.steps
        current_evaluation_counter += stats.steps
        OnlineStatsBase.fit!(chain.steps_per_draw, stats.steps)
        is_divergent = DynamicHMC.is_divergent(stats.termination)
        is_divergent && (chain.n_divergent += 1)
        if chain.current_transition_counter < stepsize_adaptation_limit
            chain.stepsize_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, chain.stepsize_state, stats.acceptance_rate)
            chain.stepsize = DynamicHMC.current_ϵ(chain.stepsize_state)
        elseif chain.current_transition_counter == stepsize_adaptation_limit
            chain.stepsize = DynamicHMC.final_ϵ(chain.stepsize_state)
        else
            append!(posterior_position, chain.position_and_gradient.q)
            append!(posterior_gradient, chain.position_and_gradient.∇ℓq)
            is_divergent && (chain.n_divergent_samples += 1)
        end
        if current_evaluation_counter >= chain.n_evaluations
            scale = scale_options[chain.active_transformation]
            for (pi, gi) in zip(eachcol(halo_position), eachcol(halo_gradient))
                ldiv!(chain.variance_memory, scale, pi)
                OnlineStatsBase.fit!(chain.variance_position, chain.variance_memory)
                mul!(chain.variance_memory, scale', gi)
                OnlineStatsBase.fit!(chain.variance_gradient, chain.variance_memory)
            end
            chain.variance_memory .= sqrt.(std.(chain.variance_position.stats) ./ std.(chain.variance_gradient.stats))
            for i in 1:dimension
                chain.variance_position.stats[i] = OnlineStatsBase.Variance()
                chain.variance_gradient.stats[i] = OnlineStatsBase.Variance()
            end
            lmin, lmax = extrema(chain.variance_memory)
            chain.variance_cond = lmax / lmin
            pushfirst!(chain.scale_changes, sqrt(chain.variance_cond))
            chain.restart = chain.variance_cond >= variance_cond_target
        end
        chain.n_samples = size(posterior_position, 2)
    end
    if monitor_ess && chain.n_samples > 10
        chain.ess .= sort!(MCMCDiagnosticTools.ess(reshape(posterior_position', (:, 1, dimension))))
    end

    # --- checkpoint bookkeeping ---
    chain.status = chain.n_samples >= n_draws ? :done : (chain.restart ? :warming : :sampling)
    log_checkpoint!(chain)
    chain.status == :done && return chain.status

    chain.n_evaluations = min(chain.n_evaluations * 2, chain.max_window_evaluations)
    chain.restart || return chain.status

    # Restart the warm-up window: re-adapt the transformation, drop prior draws.
    # Preserve them first — `_release!` builds this chain's payload only after
    # this call returns, so the reset below would otherwise checkpoint a chain
    # with full resume state and zero draws.
    chain.dropped_posterior_position = Matrix{Float64}(posterior_position)
    chain.dropped_posterior_gradient = Matrix{Float64}(recording_lpdf.posterior_gradient)
    chain.dropped_n_divergent_samples = chain.n_divergent_samples
    chain.stepsize = DynamicHMC.final_ϵ(chain.stepsize_state)
    chain.stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, chain.stepsize)
    chain.stepsize = DynamicHMC.current_ϵ(chain.stepsize_state)
    chain.current_transition_counter = 0
    chain.steps_per_draw = OnlineStatsBase.Mean()
    chain.n_divergent = 0
    chain.n_divergent_samples = 0
    # Same invariant as the adaptive restart block (`0f0f0c0`): `reset!(recording_lpdf)`
    # below empties `posterior_position`, and `n_samples` mirrors its column count, so
    # leaving it stale exports a chain claiming draws it no longer holds — via
    # `chain_result`, `_joint_ess_proxy`, and (once this lands) the on-disk payload.
    chain.n_samples = 0
    nonlinear_adapt && (chain.position_and_gradient = find_reparametrization!(chain.lpdf, halo_position, halo_gradient, chain.position_and_gradient))
    chain.active_transformation = argmin(
        map(L->update_loss!(L, halo_position, halo_gradient; chain.kwargs...), scale_options)
    )
    chain.kinetic_energy = energy_options[chain.active_transformation]
    reset!(recording_lpdf)
    chain.status
end

"Append a per-window summary to `chain.checkpoints` for the scheduler/selector."
log_checkpoint!(chain::CooperativeChain) = push!(chain.checkpoints, (;
    window=chain.outer_counter,
    n_samples=chain.n_samples,
    min_ess=chain.monitor_ess && chain.n_samples > 10 ? chain.ess[1] : NaN,
    variance_cond=chain.variance_cond,
    n_divergent_samples=chain.n_divergent_samples,
    total_evaluation_counter=chain.total_evaluation_counter,
    steps_per_draw=mean(chain.steps_per_draw),
    status=chain.status,
))

"""
    run_chain!(chain) -> chain

Drive `chain` to completion window-by-window (the resumable equivalent of the
single-chain `adaptive_warmup_mcmc` loop). Used to validate that the resumable
stepper reproduces the monolith.

Terminal finalizer: it maps the collected draws back to the original
parametrization in place (like the monolith's final step). Do NOT also call
[`chain_result`](@ref) on the same chain — that reparametrizes a second time and
corrupts the draws. The scheduler path (`advance_window!` + `chain_result`)
never calls `run_chain!`, so this only matters if you drive a chain by hand.
"""
run_chain!(chain::CooperativeChain) = begin
    while advance_window!(chain) != :done
    end
    chain.nonlinear_adapt && reparametrize!(chain.lpdf, chain_draws(chain))
    chain
end

# =============================================================================
# Abandon-stuck predicate (the `abandon-stuck` scope)
# =============================================================================

"""
    stuck_reason(chain; min_windows=4, divergence_rate=0.4, cond_stall_ratio=0.9,
                 min_divergence_samples=30) -> Union{Nothing,Symbol}

Which pathology (if any) makes `chain` stuck — `:geometry_stall`,
`:divergence_blowup`, or `nothing`. Single source of truth: [`is_stuck`](@ref) is
derived from it, so a reason recorded in a checkpoint can never disagree with the
decision to abandon, and a consumer can explain WHY a chain was dropped.

Heuristic: has `chain` stopped making useful progress and should be abandoned?
Two pathologies, read from the per-window `chain.checkpoints` log:

* **Geometry stall** — the last `min_windows` checkpoints all restarted warm-up
  (`:warming`, never reached `:sampling`) and the marginal-scale condition
  number has not improved (latest ≥ `cond_stall_ratio` × the value
  `min_windows` windows ago): the chain cannot find a workable transformation.
* **Divergence blow-up** — among the draws collected since the last restart the
  divergent fraction exceeds `divergence_rate` (needs ≥ `min_divergence_samples`
  draws to judge on).

"""
stuck_reason(chain::CooperativeChain; min_windows=4, divergence_rate=0.4,
             cond_stall_ratio=0.9, min_divergence_samples=30) = begin
    cps = chain.checkpoints
    length(cps) >= min_windows || return nothing
    recent = @view cps[end-min_windows+1:end]
    if all(cp -> cp.status === :warming, recent)
        first_cond, last_cond = first(recent).variance_cond, last(recent).variance_cond
        if isfinite(first_cond) && isfinite(last_cond) && last_cond >= cond_stall_ratio * first_cond
            return :geometry_stall
        end
    end
    if chain.n_samples >= min_divergence_samples &&
       chain.n_divergent_samples / chain.n_samples > divergence_rate
        return :divergence_blowup
    end
    nothing
end

"Has `chain` stopped making useful progress? See [`stuck_reason`](@ref) for which pathology."
is_stuck(chain::CooperativeChain; kwargs...) = !isnothing(stuck_reason(chain; kwargs...))

# =============================================================================
# Cooperative scheduler
# =============================================================================

# Shared scheduler state. All mutation goes through `state.lock`; `advance_window!`
# runs OUTSIDE the lock (it only touches its own chain), so the CPU-bound sampling
# of different chains proceeds in parallel.
mutable struct CooperativeState{RS,L,CFG}
    const rngs::RS
    const lpdf::L
    const chain_cfg::CFG
    const n_cores::Int
    const pool_target::Int          # target number of alive (warming/sampling) chains
    const n_draws::Int              # per-chain draw cap
    const target_ess::Float64
    const eval_budget::Int
    const time_budget::Float64
    const start_time::UInt64
    const lock::ReentrantLock
    const checkpoint_dir::Union{Nothing,String}
    chains::Vector{CooperativeChain}
    busy::Base.IdSet{CooperativeChain}
    n_started::Int                  # chains created (== length(chains) once installs settle)
    n_starting::Int                 # reserved-but-not-yet-installed starts
    stopped::Bool
end

_alive(state::CooperativeState) =
    count(c -> c.status === :sampling || c.status === :warming, state.chains) + state.n_starting
_total_evals(state::CooperativeState) = sum(c -> c.total_evaluation_counter, state.chains; init=0)
_elapsed(state::CooperativeState) = (time_ns() - state.start_time) / 1e9

# CHEAP proxy: sum of each chain's own min-ESS (`c.ess[1]`), over ALL sampling/done
# chains. Ignores between-chain disagreement and short-chain noise, so it
# OVER-counts — i.e. it is an upper bound on the true pooled ESS, which is exactly
# what makes it a valid stop GATE (proxy < target ⟹ pooled < target). Reads only
# scalars, so the benign race against a concurrently-advancing chain is harmless.
_joint_ess_proxy(state::CooperativeState) = sum(state.chains; init=0.0) do c
    (c.status === :sampling || c.status === :done) && c.n_samples > 10 ? c.ess[1] : 0.0
end

# ACCURATE joint ESS: pooled `MCMCDiagnosticTools.ess` over the stacked draws of
# the sampling/done chains (min over parameters) — the definition promised in
# decision tlg8p5. Chains being advanced right now (`state.busy`) are EXCLUDED:
# their draw matrix is being `append!`ed OUTSIDE the lock, so reading it here
# would race a resize. Non-busy chains are quiescent, hence safe to read under the
# lock. Draws are truncated to the shortest included chain and stacked as
# (draws, chains, params). Slightly conservative during a run (a busy chain's
# draws don't count until it next idles); exact in `_finalize` (nothing is busy).
_pooled_ess(state::CooperativeState) = begin
    used = Matrix{Float64}[]
    for c in state.chains
        (c in state.busy) && continue
        (c.status === :sampling || c.status === :done) || continue
        n_chain_draws(c) > 10 || continue
        push!(used, chain_draws(c)[:, :])   # copy the current draws (safe: c is quiescent)
    end
    isempty(used) && return 0.0
    m = minimum(d -> size(d, 2), used)
    m > 3 || return 0.0
    dim = size(first(used), 1)
    stacked = Array{Float64}(undef, m, length(used), dim)
    for (j, d) in enumerate(used)
        stacked[:, j, :] = @view(d[:, end-m+1:end])'
    end
    minimum(MCMCDiagnosticTools.ess(stacked))
end

_should_stop(state::CooperativeState) = begin
    state.stopped && return true
    _total_evals(state) >= state.eval_budget && return true
    _elapsed(state) >= state.time_budget && return true
    # Only pay for the accurate pooled ESS once the cheap upper-bound gate clears.
    if isfinite(state.target_ess) && _joint_ess_proxy(state) >= state.target_ess
        _pooled_ess(state) >= state.target_ess && return true
    end
    false
end

# Decide the next action for a freed core. Runs under `state.lock`. Returns
# `:stop`, `:idle`, `(:advance, chain)`, or `(:start, rng_index)`. This is where
# the continue / resume-another / start-new / (park-stuck) choice is made: the
# least-progressed sampling chain wins (balancing pooled ESS), warming chains
# come next, and a new chain starts only when the alive pool is under target.
_plan!(state::CooperativeState) = begin
    _should_stop(state) && (state.stopped = true; return :stop)
    best = nothing
    best_key = (typemax(Int), typemax(Int))
    for c in state.chains
        (c in state.busy) && continue
        (c.status === :sampling || c.status === :warming) || continue
        n_chain_draws(c) < state.n_draws || continue
        key = (c.status === :sampling ? 0 : 1, n_chain_draws(c))
        if best === nothing || key < best_key
            best = c; best_key = key
        end
    end
    if best !== nothing
        push!(state.busy, best)
        return (:advance, best)
    end
    if _alive(state) < state.pool_target && state.n_started < length(state.rngs)
        state.n_started += 1
        state.n_starting += 1
        return (:start, state.n_started)
    end
    (isempty(state.busy) && state.n_starting == 0) ? (state.stopped = true; :stop) : :idle
end

_install!(state::CooperativeState, chain) = (push!(state.chains, chain); state.n_starting -= 1)
# Returns the work needed to persist this chain's checkpoint, or `nothing`.
# The PAYLOAD IS BUILT UNDER THE LOCK (a pure read of chain state, cheap) but the
# disk write happens outside it — see `_flush_checkpoint!`. Building it later
# would race: once `_release!` returns, another worker may pick the chain up via
# `_plan!` and start mutating it mid-serialization.
_release!(state::CooperativeState, chain) = begin
    delete!(state.busy, chain)
    if chain.status === :sampling || chain.status === :warming
        why = stuck_reason(chain)
        if !isnothing(why)
            chain.status = :stuck
            chain.stuck_reason = why
        end
    end
    paths = _chain_checkpoint_paths(state.checkpoint_dir, chain)
    isnothing(paths) ? nothing : (paths, cooperative_checkpoint_payload(chain))
end

# Atomic temp+rename, so a crash mid-write cannot truncate a checkpoint — the
# whole point of checkpointing a run that may die. Shared with the adaptive path.
_flush_checkpoint!(::Nothing) = nothing
_flush_checkpoint!(((dir, window_path, latest_path), payload)) = begin
    mkpath(dir)
    _atomic_serialize(window_path, payload)
    _atomic_serialize(latest_path, payload)
    nothing
end

_worker!(state::CooperativeState) = while true
    action = @lock state.lock _plan!(state)
    if action === :stop
        return
    elseif action === :idle
        sleep(0.002)
    else
        tag, val = action
        if tag === :start
            # Heavy per-chain init (Pathfinder) happens OUTSIDE the lock.
            chain = cooperative_chain(state.rngs[val], deepcopy(state.lpdf); chain_index=val, state.chain_cfg...)
            @lock state.lock _install!(state, chain)
        else # :advance
            advance_window!(val)
            pending = @lock state.lock _release!(state, val)
            _flush_checkpoint!(pending)
        end
    end
end

# --- Run manifest / summary ----------------------------------------------------
#
# TWO WRITE-ONCE JSON files at the checkpoint_dir root, never rewritten:
#
#   run_manifest.json  at run START    — run identity + the criteria in force
#   run_summary.json   at FINALIZE     — the terminal answer
#
# EXISTENCE of run_summary.json IS the run-completed signal: a consumer polling
# for "is this run done?" needs no parsing and no field. An earlier draft had a
# single manifest rewritten at finalize; that reintroduces the cp_latest.jls
# torn-read hazard in the one file every consumer must read to identify a run,
# so it was rejected in review.
#
# JSON rather than .jls so a web process can read run identity WITHOUT a Julia
# runtime — that is the whole point, and it is why this hand-rolls a tiny writer
# instead of taking a JSON dependency: the contents are a flat, fully-controlled
# set of scalars and small arrays, and WarmupHMC is heading for General
# registration where every added dep is a liability.

_json_esc(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")
_json_val(::Nothing) = "null"
_json_val(x::Bool) = x ? "true" : "false"
_json_val(x::Real) = isfinite(x) ? string(x) : "null"   # Inf/NaN are not JSON
_json_val(x::Union{AbstractString,Symbol}) = "\"" * _json_esc(x) * "\""
_json_val(x::AbstractVector) = "[" * join(map(_json_val, x), ",") * "]"
_json_val(x) = _json_val(string(x))
_json_object(pairs) = "{" * join(
    [ "\"" * _json_esc(k) * "\":" * _json_val(v) for (k, v) in pairs ], ",") * "}"

# Atomic write, same reason as the checkpoints: a crash during a WRITE-ONCE file
# must not strand a half-written one that a consumer would read as authoritative.
# `pairs` is deliberately Pair{String,Any} at every call site: a literal array
# mixing an Int with an Inf promotes to Float64, which would silently render
# `schema_version` as `1.0` and break a consumer parsing it as an integer.
_write_json(path, pairs) = begin
    mkpath(dirname(path))
    tmp, io = mktemp(dirname(path); cleanup=false)
    try
        write(io, _json_object(pairs))
        close(io)
        mv(tmp, path; force=true)
    catch
        close(io); rm(tmp; force=true); rethrow()
    end
    nothing
end

"Write `run_manifest.json` once, at run start. Never rewritten."
write_run_manifest(::Nothing, state) = nothing
write_run_manifest(dir::AbstractString, state::CooperativeState) = _write_json(
    joinpath(dir, "run_manifest.json"), Pair{String,Any}[
        "schema_version" => checkpoint_schema_version(),
        "sampler" => "cooperative",
        "n_chains_requested" => length(state.rngs),
        "n_cores" => state.n_cores,
        "pool_target" => state.pool_target,
        # The stopping criteria ACTUALLY in force. Non-finite means "not set":
        # JSON has no Inf, so an unset bound reads as null rather than a lie.
        "n_draws" => state.n_draws == typemax(Int) ? nothing : state.n_draws,
        "target_ess" => state.target_ess,
        "n_evaluations_budget" => state.eval_budget == typemax(Int) ? nothing : state.eval_budget,
        "time_budget" => state.time_budget,
        # Xoshiro prints its full internal state, so this is the reproducible
        # per-chain seed, not a label.
        "chain_rngs" => map(string, state.rngs),
        "start_time_unix" => time(),
    ])

"""
    write_run_summary(dir, state, result) -> nothing

Write `run_summary.json` once, at finalize. Its EXISTENCE is the run-completed
signal — a consumer need not parse it to answer "is this run done?".

`stop_reason` here is RUN-level and answers a different question from a chain's
`stop_reason`: a chain can be final while the pool keeps going.
"""
write_run_summary(::Nothing, state, result) = nothing
write_run_summary(dir::AbstractString, state::CooperativeState, result) = _write_json(
    joinpath(dir, "run_summary.json"), Pair{String,Any}[
        "schema_version" => checkpoint_schema_version(),
        "sampler" => "cooperative",
        "is_final" => true,
        "stop_reason" => string(run_stop_reason(state)),
        "n_started" => result.n_started,
        "n_used" => result.n_used,
        "n_stuck" => result.n_stuck,
        "joint_ess" => result.joint_ess,
        "total_evaluation_counter" => result.total_evaluation_counter,
        "elapsed" => result.elapsed,
        "end_time_unix" => time(),
    ])

"""
    run_stop_reason(state) -> Symbol

Which bound actually ended the run: `:eval_budget`, `:time_budget`,
`:target_ess`, or `:exhausted` (every chain reached `n_draws` or was abandoned).
Checked in the same order as `_should_stop` so the reported reason matches the
one that fired.
"""
run_stop_reason(state::CooperativeState) =
    _total_evals(state) >= state.eval_budget ? :eval_budget :
    _elapsed(state) >= state.time_budget ? :time_budget :
    (isfinite(state.target_ess) && _pooled_ess(state) >= state.target_ess) ? :target_ess :
    :exhausted

"""
    checkpoint_schema_version() -> Int

Version of the on-disk checkpoint/manifest contract, shared by ALL samplers.

Policy (the contract, not just the number):

* **Additive changes do NOT bump it.** New keys may appear at any time, so
  readers MUST ignore unknown keys.
* **Changing an existing key's meaning or type, or removing it, DOES bump it.**
  A bump means exactly "re-read the contract".
* Therefore a reader seeing a version NEWER than it knows should **fail loudly**:
  since additions never bump, a higher version can only mean existing semantics
  changed.
"""
checkpoint_schema_version() = 2

"""
    cooperative_checkpoint_payload(chain) -> NamedTuple

Serializable snapshot of `chain` at a window boundary.

Deliberately EXCLUDES `lpdf` (possibly a non-serializable BridgeStan handle —
resume re-supplies it, exactly as the adaptive payload does).

`status` and `stuck_reason` are AS OF THIS CHECKPOINT and are never backfilled:
if a chain is abandoned at window 7, `cp_window_3.jls` still reads `:warming`.
That keeps written checkpoints immutable, which is what lets a consumer pin a
`(config, checkpoint#)` view and trust an old file.

`is_final` means THIS CHAIN will produce no more draws — not that the run is
over. A chain can be final while the pool keeps going; the run-level answer
needs scheduler intent and lives in the run summary instead.

`posterior_position` is empty at every checkpoint whose window restarted (the
reset precedes the write); `dropped_posterior_position` holds what that restart
discarded. Same additive contract and same consumer rule as the adaptive
payload — see the comment above `checkpoint_payload`.
"""
cooperative_checkpoint_payload(chain::CooperativeChain) = (;
    schema_version=checkpoint_schema_version(),
    sampler=:cooperative,
    chain_index=chain.chain_index,
    status=chain.status,
    stuck_reason=chain.stuck_reason,
    is_final=chain.status === :done || chain.status === :stuck,
    stop_reason=chain.status === :done ? :n_draws :
                chain.status === :stuck ? :stuck : :running,
    window=chain.outer_counter,
    halo_position=chain.recording_lpdf.halo_position,
    halo_gradient=chain.recording_lpdf.halo_gradient,
    posterior_position=chain.recording_lpdf.posterior_position,
    posterior_gradient=chain.recording_lpdf.posterior_gradient,
    recorder=chain.recording_lpdf.recorder,
    chain.rng, chain.position_and_gradient, chain.active_transformation,
    chain.scale_options, chain.stepsize, chain.stepsize_state,
    chain.n_evaluations, chain.variance_memory, chain.variance_position,
    chain.variance_gradient, chain.variance_cond, chain.scale_changes,
    chain.total_evaluation_counter, chain.current_transition_counter,
    chain.total_transition_counter, chain.n_divergent, chain.n_divergent_samples,
    chain.steps_per_draw, chain.ess, chain.restart, chain.n_samples,
    chain.n_draws, chain.dimension, chain.checkpoints,
    chain.dropped_posterior_position, chain.dropped_posterior_gradient,
    chain.dropped_n_divergent_samples,
    reparam_sources=reparam_sources(chain.lpdf),
)

# Per-chain checkpoint paths. `chain_<i>/` reuses the adaptive layout, and the
# index is stable and never reused (see `CooperativeChain.chain_index`), so
# `chain_3/cp_window_5.jls` and `chain_3/cp_window_2.jls` are the same chain.
_chain_checkpoint_paths(::Nothing, ::CooperativeChain) = nothing
_chain_checkpoint_paths(dir::AbstractString, chain::CooperativeChain) = begin
    d = joinpath(dir, "chain_$(chain.chain_index)")
    (d, joinpath(d, "cp_window_$(chain.outer_counter).jls"), joinpath(d, "cp_latest.jls"))
end

"""
    chain_result(chain) -> NamedTuple

Adaptive-style per-chain result. Maps draws back to the original
parametrization (when `nonlinear_adapt`), then reports the collected positions/
gradients plus diagnostics and the per-window `checkpoints` log.

Terminal finalizer: it reparametrizes the draws in place, so call it exactly
once per chain and never after [`run_chain!`](@ref) (double reparametrization
corrupts the draws). The scheduler drives chains with `advance_window!` alone —
which does not reparametrize — so `_finalize` is the single such call.
"""
chain_result(chain::CooperativeChain) = begin
    draws = chain_draws(chain)
    chain.nonlinear_adapt && size(draws, 2) > 0 && reparametrize!(chain.lpdf, draws)
    (;
        # `state.chains` is in install-COMPLETION order, not index order, so
        # `results[i]` is not chain `i`. Carry the identity explicitly.
        chain_index=chain.chain_index,
        posterior_position=draws,
        posterior_gradient=chain.recording_lpdf.posterior_gradient,
        ess=chain.ess,
        stepsize=chain.stepsize,
        active_transformation=chain.active_transformation,
        status=chain.status,
        n_samples=chain.n_samples,
        n_divergent_samples=chain.n_divergent_samples,
        total_evaluation_counter=chain.total_evaluation_counter,
        n_windows=chain.outer_counter,
        checkpoints=chain.checkpoints,
    )
end

_finalize(state::CooperativeState) = begin
    # Nothing is busy post-@sync, so _pooled_ess sees every usable chain.
    joint_ess = _pooled_ess(state)
    joint_ess_proxy = _joint_ess_proxy(state)
    total_evaluation_counter = _total_evals(state)
    elapsed = _elapsed(state)
    results = map(chain_result, state.chains)
    (;
        results,
        n_started=length(state.chains),
        n_used=count(r -> (r.status === :sampling || r.status === :done) && size(r.posterior_position, 2) > 0, results),
        n_stuck=count(r -> r.status === :stuck, results),
        joint_ess,          # accurate pooled ESS over stacked draws (tlg8p5 definition)
        joint_ess_proxy,    # cheap sum-of-per-chain-min-ESS gate value, for reference
        total_evaluation_counter,
        elapsed,
    )
end

"""
    cooperative_warmup_mcmc(rngs, lpdf; n_cores=length(rngs), target_ess=Inf,
        n_evaluations_budget=typemax(Int), time_budget=Inf, min_chains=min(length(rngs),4),
        max_window_evaluations=4000, n_draws=typemax(Int), nonlinear_adapt=true,
        progress=nothing, kwargs...)

Cooperative multi-chain adaptive-warmup NUTS. Keeps up to `n_cores` chains
advancing concurrently; at each warm-up **window boundary (checkpoint)** a
scheduler decides, per freed core, whether to CONTINUE that chain, RESUME
another (a less-progressed one), or START a new one, and PARKS chains flagged
stuck by [`is_stuck`](@ref) — greedily maximizing joint (pooled) ESS per
gradient evaluation.

Each chain is a resumable [`CooperativeChain`](@ref) built from the same
adaptive procedure as [`adaptive_warmup_mcmc`](@ref) (the low-level pieces are
shared), advanced one window at a time via [`advance_window!`](@ref).

* `rngs` — a vector of RNGs; its length bounds the number of distinct chains.
* `n_cores` — how many chains run at once.
* Stops when the pooled ESS reaches `target_ess`, the total gradient-evaluation
  budget `n_evaluations_budget` is spent, or `time_budget` seconds elapse. **At
  least one bound must be finite.** `target_ess` is measured as the accurate
  pooled `MCMCDiagnosticTools.ess` over stacked draws, but only over chains that
  are momentarily IDLE at the checkpoint (a chain being advanced has its draw
  buffer mutated off-lock and can't be read safely). So with the cores saturated
  the visible estimate lags the true total — the run **over-delivers** (realized
  pooled ESS can exceed `target_ess` by up to roughly `n_cores`×, never less).
  A future refinement (a per-chain last-committed-draws snapshot updated under
  the lock) would tighten this; for now prefer bounding by budget/time when you
  want a hard compute ceiling.
* `max_window_evaluations` caps each window's eval budget so checkpoints stay
  frequent enough to reschedule.
* Extra `kwargs` are forwarded to every chain (`recording_target`,
  `target_acceptance_rate`, `max_tree_depth`, `variance_cond_target`, `init`, …).

Real parallelism needs Julia started with threads (`julia -t N`); correctness
holds at any thread count. Each chain gets a `deepcopy` of `lpdf`.

Returns a `NamedTuple`: per-chain `results` (adaptive-style, see
[`chain_result`](@ref)) plus the accurate pooled `joint_ess` (over stacked
draws), the cheap `joint_ess_proxy` (sum of per-chain min-ESS, the stop gate),
`n_started`, `n_used`, `n_stuck`, `total_evaluation_counter`, `elapsed`.
"""
cooperative_warmup_mcmc(rngs::AbstractVector, lpdf;
    n_cores=length(rngs),
    target_ess=Inf,
    n_evaluations_budget=typemax(Int),
    time_budget=Inf,
    min_chains=min(length(rngs), 4),
    max_window_evaluations=4000,
    n_draws=typemax(Int),
    nonlinear_adapt=true,
    progress=nothing,
    checkpoint_dir=nothing,
    # Discard whatever `checkpoint_dir` already holds and start fresh. `resume` is
    # accepted so the error is a clear "not supported yet" rather than a silent
    # no-op; the guard itself matters regardless — see `guard_run_dir!`.
    resume=false,
    overwrite=false,
    # Forwarded verbatim to the Pathfinder initializer; see `_check_kwargs`.
    pathfinder_kw=(;),
    kwargs...
) = begin
    _check_kwargs(:cooperative_warmup_mcmc, kwargs)
    guard_run_dir!(checkpoint_dir, resume, overwrite, :cooperative)
    @assert isfinite(target_ess) || n_evaluations_budget != typemax(Int) || isfinite(time_budget) "cooperative_warmup_mcmc needs at least one finite stopping bound (target_ess, n_evaluations_budget, or time_budget)."
    chain_cfg = (; n_draws, max_window_evaluations, nonlinear_adapt, monitor_ess=true, kwargs..., pathfinder_kw...)
    pool_target = min(length(rngs), max(min_chains, n_cores))
    state = CooperativeState(
        rngs, lpdf, chain_cfg, n_cores, pool_target, n_draws,
        Float64(target_ess), n_evaluations_budget, Float64(time_budget),
        time_ns(), ReentrantLock(), checkpoint_dir,
        CooperativeChain[], Base.IdSet{CooperativeChain}(), 0, 0, false,
    )
    write_run_manifest(checkpoint_dir, state)
    @sync for _ in 1:n_cores
        Threads.@spawn _worker!(state)
    end
    result = _finalize(state)
    write_run_summary(checkpoint_dir, state, result)
    result
end
