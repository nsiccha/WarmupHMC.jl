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
mutable struct CooperativeChain{R,L,RL,A,SA,SO,EO,K,NT}
    # --- configuration (set once) ---
    const rng::R
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
    kinetic_energy::K
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
    checkpoints::Vector{NamedTuple}     # per-window log for the scheduler/selector
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
    recorder = LimitedRecorder2(recording_target, n_evaluations ÷ recording_target)
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    scale_options = (;
        diagonal=Diagonal(sqrt.(diag(squared_scale))::Vector{Float64}),
        pathfinder=MatrixFactorization(factorize(squared_scale).L, Diagonal(ones(dimension))),
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
        rng, lpdf, recording_lpdf, algorithm, stepsize_adaptation, dimension,
        recording_target, stepsize_adaptation_limit, variance_cond_target,
        nonlinear_adapt, monitor_ess, n_draws, max_window_evaluations,
        scale_options, energy_options, NamedTuple(kwargs), start_time,
        position_and_gradient, active_transformation, kinetic_energy,
        stepsize, stepsize_state, n_evaluations, variance_memory,
        variance_position, variance_gradient, Inf, Float64[],
        0, 0, 0, 0, 0, 0, OnlineStatsBase.Mean(), zeros(dimension),
        true, 0, :warming, NamedTuple[],
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
        chain.position_and_gradient, stats = DynamicHMC.sample_tree(
            rng, algorithm, hamiltonian, chain.position_and_gradient, chain.stepsize
        )
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
    recording_lpdf.recorder.thin = max(1, chain.n_evaluations ÷ recording_target)
    chain.restart || return chain.status

    # Restart the warm-up window: re-adapt the transformation, drop prior draws.
    chain.stepsize = DynamicHMC.final_ϵ(chain.stepsize_state)
    chain.stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, chain.stepsize)
    chain.stepsize = DynamicHMC.current_ϵ(chain.stepsize_state)
    chain.current_transition_counter = 0
    chain.steps_per_draw = OnlineStatsBase.Mean()
    chain.n_divergent = 0
    chain.n_divergent_samples = 0
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
    is_stuck(chain; min_windows=4, divergence_rate=0.4, cond_stall_ratio=0.9,
             min_divergence_samples=30) -> Bool

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
is_stuck(chain::CooperativeChain; min_windows=4, divergence_rate=0.4,
         cond_stall_ratio=0.9, min_divergence_samples=30) = begin
    cps = chain.checkpoints
    length(cps) >= min_windows || return false
    recent = @view cps[end-min_windows+1:end]
    if all(cp -> cp.status === :warming, recent)
        first_cond, last_cond = first(recent).variance_cond, last(recent).variance_cond
        if isfinite(first_cond) && isfinite(last_cond) && last_cond >= cond_stall_ratio * first_cond
            return true
        end
    end
    if chain.n_samples >= min_divergence_samples &&
       chain.n_divergent_samples / chain.n_samples > divergence_rate
        return true
    end
    false
end

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
    chains::Vector{CooperativeChain}
    busy::Base.IdSet{CooperativeChain}
    n_started::Int                  # chains created (== length(chains) once installs settle)
    n_starting::Int                 # reserved-but-not-yet-installed starts
    stopped::Bool
end

_alive(state::CooperativeState) =
    count(c -> c.status === :sampling || c.status === :warming, state.chains) + state.n_starting
_joint_ess(state::CooperativeState) = sum(state.chains; init=0.0) do c
    (c.status === :sampling || c.status === :done) && c.n_samples > 10 ? c.ess[1] : 0.0
end
_total_evals(state::CooperativeState) = sum(c -> c.total_evaluation_counter, state.chains; init=0)
_elapsed(state::CooperativeState) = (time_ns() - state.start_time) / 1e9

_should_stop(state::CooperativeState) = state.stopped ||
    _joint_ess(state) >= state.target_ess ||
    _total_evals(state) >= state.eval_budget ||
    _elapsed(state) >= state.time_budget

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
_release!(state::CooperativeState, chain) = begin
    delete!(state.busy, chain)
    if (chain.status === :sampling || chain.status === :warming) && is_stuck(chain)
        chain.status = :stuck
    end
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
            chain = cooperative_chain(state.rngs[val], deepcopy(state.lpdf); state.chain_cfg...)
            @lock state.lock _install!(state, chain)
        else # :advance
            advance_window!(val)
            @lock state.lock _release!(state, val)
        end
    end
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
    joint_ess = _joint_ess(state)
    total_evaluation_counter = _total_evals(state)
    elapsed = _elapsed(state)
    results = map(chain_result, state.chains)
    (;
        results,
        n_started=length(state.chains),
        n_used=count(r -> (r.status === :sampling || r.status === :done) && size(r.posterior_position, 2) > 0, results),
        n_stuck=count(r -> r.status === :stuck, results),
        joint_ess,
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
  least one bound must be finite.**
* `max_window_evaluations` caps each window's eval budget so checkpoints stay
  frequent enough to reschedule.
* Extra `kwargs` are forwarded to every chain (`recording_target`,
  `target_acceptance_rate`, `max_tree_depth`, `variance_cond_target`, `init`, …).

Real parallelism needs Julia started with threads (`julia -t N`); correctness
holds at any thread count. Each chain gets a `deepcopy` of `lpdf`.

Returns a `NamedTuple`: per-chain `results` (adaptive-style, see
[`chain_result`](@ref)) plus pooled `joint_ess`, `n_started`, `n_used`,
`n_stuck`, `total_evaluation_counter`, `elapsed`.
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
    kwargs...
) = begin
    @assert isfinite(target_ess) || n_evaluations_budget != typemax(Int) || isfinite(time_budget) "cooperative_warmup_mcmc needs at least one finite stopping bound (target_ess, n_evaluations_budget, or time_budget)."
    chain_cfg = (; n_draws, max_window_evaluations, nonlinear_adapt, monitor_ess=true, kwargs...)
    pool_target = min(length(rngs), max(min_chains, n_cores))
    state = CooperativeState(
        rngs, lpdf, chain_cfg, n_cores, pool_target, n_draws,
        Float64(target_ess), n_evaluations_budget, Float64(time_budget),
        time_ns(), ReentrantLock(),
        CooperativeChain[], Base.IdSet{CooperativeChain}(), 0, 0, false,
    )
    @sync for _ in 1:n_cores
        Threads.@spawn _worker!(state)
    end
    _finalize(state)
end
