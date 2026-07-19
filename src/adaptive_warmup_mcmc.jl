initialize_mcmc(lpdf, ::Missing; kwargs...) = initialize_mcmc(lpdf, 2.; kwargs...)
initialize_mcmc(lpdf, init::Real; kwargs...) = initialize_mcmc(lpdf, Uniform(-init,+init); kwargs...)
initialize_mcmc(lpdf, init::Distribution; rng, ntries=10, kwargs...) = for i in 1:ntries
    try
        return initialize_mcmc(lpdf, rand(rng, init, LogDensityProblems.dimension(lpdf)); rng, kwargs...)
    catch
        i == ntries && rethrow()
        @warn "Initialization failed the $i-th time, trying again..."
    end
end
pathfinder_callback(progress) = (state, args...) -> (update_progress!(progress, state.iter); false)
initialize_mcmc(lpdf, init::AbstractVector; rng, progress, maxiters=100, kwargs...) = with_progress(progress, maxiters; description="Pathfinder", transient=true) do pprogress
    # Work around https://github.com/roualdes/bridgestan/issues/272
    LogDensityProblems.logdensity_and_gradient(lpdf, init)
    initialize_mcmc(
        lpdf,
        mypathfinder(lpdf; rng, init, callback=pathfinder_callback(pprogress), maxiters, kwargs...);
        kwargs...
    )
end
initialize_mcmc(lpdf, init::PathfinderResult; kwargs...) = begin
    @assert length(init.elbo_estimates) > 0
    position = collect(init.draws[:, 1])::Vector{Float64}
    dimension = length(position)
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    squared_scale = init.fit_distribution.Σ
    scale = MatrixFactorization(factorize(squared_scale).L, Diagonal(ones(dimension)))
    initialize_mcmc(lpdf, (;position, position_and_gradient, scale, squared_scale))
end
initialize_mcmc(lpdf, init::NamedTuple; kwargs...) = init
"Set other defaults and works around https://github.com/mlcolab/Pathfinder.jl/issues/248"
mypathfinder(args...;
    ndraws=1, ndraws_elbo=1, ntries=1,
    history_length=6,
    optimizer=Pathfinder.Optim.LBFGS(;
        m=history_length,
        linesearch=Pathfinder.LineSearches.HagerZhang(),
        alphaguess=Pathfinder.LineSearches.InitialHagerZhang()
    ),
    kwargs...
) = pathfinder(
    args...;
    ndraws, ntries, ndraws_elbo, optimizer, kwargs...
)

# ── Windowed adaptive warm-up: explicit state object + stage functions ──────
#
# The single-chain sampler is factored into a state object plus three stage
# functions so that the two checkpoint boundaries the caller cares about —
# CP-0 "after initialization" and CP-N "after each outer (warm-up-window)
# iteration" — become explicit seams for callbacks / resume / on-disk state.
# This factoring is byte-for-byte identical to the previous monolith: no
# arithmetic is reordered, every mutation keeps its original order, and the one
# `rng` object stays shared between the driver and the recording posterior.
#
#   init_state          → setup + Pathfinder init + initial step size  (→ CP-0)
#   run_outer_iteration! → one warm-up window: transitions, adaptation  (→ CP-N)
#   finalize_warmup!    → back-transform + assemble the return NamedTuple
#
# `AWMState` carries EVERY value that lives across a checkpoint boundary. Config
# fields (set once) precede dynamic fields (mutated during warm-up). `lpdf`,
# `progress` and `start_time` are runtime handles that are re-supplied rather
# than restored on resume (the inner problem may wrap a non-serializable native
# gradient; wall-clock timing is display-only).
mutable struct AWMState{L,K,A,DA,P,R,RL,SO,EO,VP,VG,POS,PG,SS,MN}
    # ── config (set once) ──
    lpdf::L
    n_draws::Int
    stepsize_adaptation_limit::Int
    variance_cond_target::Float64
    nonlinear_adapt::Bool
    monitor_ess::Bool
    recording_target::Int
    kwargs::K
    algorithm::A
    stepsize_adaptation::DA
    dimension::Int
    # ── runtime handles (re-supplied on resume, not restored) ──
    progress::P
    start_time::UInt64
    # ── dynamic (mutated during warm-up) ──
    rng::R
    recording_lpdf::RL
    position::POS
    scale_options::SO
    energy_options::EO
    active_transformation::Symbol
    # Untyped: reassigned across `energy_options` entries, which are
    # heterogeneously typed (each transformation yields a distinct
    # GaussianKineticEnergy type), so no single concrete type fits.
    kinetic_energy::Any
    variance_memory::Vector{Float64}
    variance_position::VP
    variance_gradient::VG
    variance_cond::Float64
    scale_changes::Vector{Float64}
    position_and_gradient::PG
    stepsize::Float64
    stepsize_state::SS
    n_evaluations::Int
    total_evaluation_counter::Int
    outer_counter::Int
    current_transition_counter::Int
    total_transition_counter::Int
    ess::Vector{Float64}
    steps_per_draw::MN
    n_divergent::Int
    n_divergent_samples::Int
    restart::Bool
    n_samples::Int
end

# Setup + initialization, up to and including the initial step-size search and
# the first progress tick. Returns the state as of CP-0 ("after Pathfinder").
init_state(
    rng, lpdf, progress;
    n_draws, n_evaluations, recording_target, stepsize_adaptation_limit,
    target_acceptance_rate, max_tree_depth, init, monitor_ess,
    nonlinear_adapt, variance_cond_target, kwargs...
) = begin
    start_time = time_ns()
    # Standard Stepsize Search
    stepsize_search = DynamicHMC.InitialStepsizeSearch()
    # Standard Dual Averaging
    stepsize_adaptation = DynamicHMC.DualAveraging(δ=target_acceptance_rate)
    # Standard NUTS
    algorithm = DynamicHMC.NUTS(;max_depth=max_tree_depth)
    # The dimension of the posterior
    dimension = LogDensityProblems.dimension(lpdf)
    # A thin wrapper around the posterior that enables us to record the intermediate positions and gradients
    recorder = LimitedRecorder2(recording_target)
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    # Use Stan's initialization procedure if no initial position is given
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    # We currently learn three linear transformation options
    scale_options = (;
        # Corresponds to a standard diagonal mass matrix
        diagonal=Diagonal(sqrt.(diag(squared_scale))::Vector{Float64}),
        # Corresponds to Pathfinder's linear transformation with an added diagonal scaling term that can be updated
        pathfinder=MatrixFactorization(factorize(squared_scale).L, Diagonal(ones(dimension))),
        # Something new. Corresponds to a sequence of Householder reflections, followed by a diagonal scaling term.
        # Both the reflections and the diagonal scaling term will be updated.
        adaptive=MatrixFactorization(SuccessiveReflections(dimension), Diagonal(ones(dimension)))
    )
    # This is needed to make DynamicHMC "accept" our linear transformations
    energy_options = map(scale_options) do L
        DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))
    end
    # At the beginning, we will use Pathfinder's transformation.
    active_transformation = :pathfinder # Pathfinder
    kinetic_energy = energy_options[active_transformation]
    # Online variance recorders
    variance_memory = zeros(dimension)
    variance_position = OnlineStatsBase.Group([OnlineStatsBase.Variance() for i in 1:dimension])
    variance_gradient = OnlineStatsBase.Group([OnlineStatsBase.Variance() for i in 1:dimension])
    variance_cond = Inf
    scale_changes = Float64[]

    # The below tries to find a good initial stepsize.
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    position_gradient_and_momentum = DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy))
    stepsize = DynamicHMC.find_initial_stepsize(
        stepsize_search,
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf), position_gradient_and_momentum
        )
    )
    # For monitoring purposes: Keep track of the number of gradient evaluations during warm-up
    total_evaluation_counter = 0
    # For monitoring purposes: Keep track of the number of warm-up windows so far
    outer_counter = 0
    # For monitoring purposes: Keep track of the number of the total number of MCMC transitions
    current_transition_counter = 0
    total_transition_counter = 0
    # For monitoring purposes: Keep track of the minimal effective sample size so far
    ess = zeros(dimension)
    # For monitoring purposes: Keep track of the current number of gradient evaluations per MCMC transition
    steps_per_draw = OnlineStatsBase.Mean()
    # For monitoring purposes: Keep track of the number of divergences in the current WARM-UP window
    n_divergent = 0
    # For monitoring purposes: Keep track of the number of divergences in the current SAMPLING window
    n_divergent_samples = 0
    restart = true
    # We run the warm-up procedure until we have collected enough samples
    n_samples = 0
    update_progress!(progress, current_transition_counter;
        divergent_samples=UncertainFrequency(n_divergent_samples, n_samples),
        (monitor_ess ? (;ess="pending...") : (;))...,
        active_transformation=ActiveTransformation(kinetic_energy, scale_changes),
        sampling_performance=SamplingPerformance(stepsize, mean(steps_per_draw)),
        total_transition_counter,
        total_evaluation_counter,
    )
    stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
    AWMState(
        lpdf, n_draws, stepsize_adaptation_limit, variance_cond_target, nonlinear_adapt,
        monitor_ess, recording_target, (; kwargs...), algorithm, stepsize_adaptation, dimension,
        progress, start_time,
        rng, recording_lpdf, position, scale_options, energy_options, active_transformation,
        kinetic_energy, variance_memory, variance_position, variance_gradient, variance_cond,
        scale_changes, position_and_gradient, stepsize, stepsize_state, n_evaluations,
        total_evaluation_counter, outer_counter, current_transition_counter,
        total_transition_counter, ess, steps_per_draw, n_divergent, n_divergent_samples,
        restart, n_samples,
    )
end

# One warm-up/sampling window ("big iteration"): the inner transition loop plus
# the end-of-window step-size freeze, variance-condition check, and (on restart)
# reparametrization + transformation reselection. Mutates `state` in place;
# returns after one window, leaving `state` as of CP-N.
run_outer_iteration!(state::AWMState) = begin
    (; progress, start_time, lpdf, recording_lpdf, algorithm, stepsize_adaptation, dimension) = state
    # Some setup that has to happen at the beginning of every warm-up window
    state.outer_counter += 1
    hamiltonian = DynamicHMC.Hamiltonian(state.kinetic_energy, recording_lpdf)
    current_evaluation_counter = 0
    # We run the current warm-up/sampling window until
    #   a) we have collected enough samples and can break out of the outer loop as well or
    #   b) we have reached the current targeted number of gradient evaluations AND we estimate that
    #       restarting (adding a new warm-up window) is better than finishing sampling with the current adaptation
    while size(recording_lpdf.posterior_position, 2) < state.n_draws && (current_evaluation_counter < state.n_evaluations)
        state.current_transition_counter += 1
        state.total_transition_counter += 1
        # One MCMC transition
        reset!(recording_lpdf.leaves)
        state.position_and_gradient, stats = DynamicHMC.sample_tree(state.rng, algorithm, hamiltonian, state.position_and_gradient, state.stepsize)
        finalize_leaf_recording!(recording_lpdf, stats.depth)
        state.total_evaluation_counter += stats.steps
        current_evaluation_counter += stats.steps
        OnlineStatsBase.fit!(state.steps_per_draw, stats.steps)
        is_divergent = DynamicHMC.is_divergent(stats.termination)
        is_divergent && (state.n_divergent += 1)
        if state.current_transition_counter < state.stepsize_adaptation_limit
            # The current warm-up window has seen fewer MCMC transitions than our step size adaptation limit.
            # Continue adapting the step size.
            state.stepsize_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, state.stepsize_state, stats.acceptance_rate)
            state.stepsize = DynamicHMC.current_ϵ(state.stepsize_state)
        elseif state.current_transition_counter == state.stepsize_adaptation_limit
            # The current warm-up window hits the step size adaptation limit.
            # Finalize the stepsize.
            state.stepsize = DynamicHMC.final_ϵ(state.stepsize_state)
        else
            # The current warm-up window has been sampling with the same linear transformation and step size.
            # Record posterior positions, gradients and whether the current transition diverged
            append!(recording_lpdf.posterior_position, state.position_and_gradient.q)
            append!(recording_lpdf.posterior_gradient, state.position_and_gradient.∇ℓq)
            is_divergent && (state.n_divergent_samples += 1)
        end
        if current_evaluation_counter >= state.n_evaluations
            scale = state.scale_options[state.active_transformation]
            for (pi, gi) in zip(eachcol(recording_lpdf.halo_position), eachcol(recording_lpdf.halo_gradient))
                ldiv!(state.variance_memory, scale, pi)
                OnlineStatsBase.fit!(state.variance_position, state.variance_memory)
                mul!(state.variance_memory, scale', gi)
                OnlineStatsBase.fit!(state.variance_gradient, state.variance_memory)
            end
            state.variance_memory .= sqrt.(std.(state.variance_position.stats) ./ std.(state.variance_gradient.stats))
            for i in 1:dimension
                state.variance_position.stats[i] = OnlineStatsBase.Variance()
                state.variance_gradient.stats[i] = OnlineStatsBase.Variance()
            end
            lmin, lmax = extrema(state.variance_memory)
            state.variance_cond = lmax / lmin
            pushfirst!(state.scale_changes, sqrt(state.variance_cond))
            state.restart = state.variance_cond >= state.variance_cond_target
        end
        state.n_samples = size(recording_lpdf.posterior_position, 2)
        update_progress!(progress, state.current_transition_counter;
            divergent_samples=UncertainFrequency(state.n_divergent_samples, state.n_samples),
            active_transformation=ActiveTransformation(state.kinetic_energy, state.scale_changes),
            sampling_performance=SamplingPerformance(state.stepsize, mean(state.steps_per_draw)),
            total_transition_counter=Speed(state.total_transition_counter, time_ns()-start_time),
            total_evaluation_counter=Speed(state.total_evaluation_counter, time_ns()-start_time),
        )
    end
    if state.monitor_ess && state.n_samples > 10
        state.ess .= sort!(MCMCDiagnosticTools.ess(reshape(recording_lpdf.posterior_position', (:, 1, dimension))))
        update_progress!(progress, nothing;
            ess=short_string(state.ess) * " from $(state.n_samples) samples.",
        )
    end
    state.n_samples < state.n_draws || return state
    # Double the targeted number of GRADIENT EVALUATIONS in the next warm-up window
    state.n_evaluations *= 2
    state.restart || return state
    state.stepsize = DynamicHMC.final_ϵ(state.stepsize_state)
    state.stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, state.stepsize)
    state.stepsize = DynamicHMC.current_ϵ(state.stepsize_state)
    # Reset the so far recorded intermediate and MCMC positions and gradients
    state.current_transition_counter = 0
    state.steps_per_draw = OnlineStatsBase.Mean()
    state.n_divergent = 0
    state.n_divergent_samples = 0
    # Update the linear transformation candidates and estimate the transformation loss,
    # using the INTERMEDIATE POSITIONS AND GRADIENTS.
    state.nonlinear_adapt && (state.position_and_gradient = find_reparametrization!(lpdf, recording_lpdf.halo_position, recording_lpdf.halo_gradient, state.position_and_gradient))
    # Update the new linear transformation to be the one with the minimal estimated transformation loss.
    state.active_transformation = argmin(
        map(L->update_loss!(L, (recording_lpdf.halo_position), (recording_lpdf.halo_gradient); state.kwargs...), state.scale_options)
    )
    state.kinetic_energy = state.energy_options[state.active_transformation]
    update_progress!(progress, nothing;
        active_transformation=ActiveTransformation(state.kinetic_energy, state.scale_changes),
    )
    reset!(recording_lpdf)
    state
end

# Final progress line, back-transform the posterior draws into the original
# parametrization, and assemble the returned NamedTuple.
finalize_warmup!(state::AWMState) = begin
    (; progress, recording_lpdf, lpdf) = state
    update_progress!(progress, (state.monitor_ess ? "min. ESS: $(short_string(state.ess[1])), " : "") * "divergent: $(short_string(100*state.n_divergent_samples/state.n_samples))%")
    state.nonlinear_adapt && reparametrize!(lpdf, recording_lpdf.posterior_position)
    (;
        initial_position=state.position,
        halo_position=recording_lpdf.halo_position,
        halo_gradient=recording_lpdf.halo_gradient,
        posterior_position=recording_lpdf.posterior_position,
        posterior_gradient=recording_lpdf.posterior_gradient,
        ess=state.ess,
        scale_options=state.scale_options,
        active_transformation=state.active_transformation,
        stepsize=state.stepsize,
        total_evaluation_counter=state.total_evaluation_counter,
        n_divergent_samples=state.n_divergent_samples,
        position_and_gradient=state.position_and_gradient,
        scale_changes=state.scale_changes,
    )
end

# Extract the reparametrizer's DYNAMIC scalar state — the per-index `source`
# centering that `optimize!` mutates in place. The lpdf itself (possibly a
# non-serializable native problem) and the reparametrizer's index-extraction
# closures are NOT serialized; on resume these `source` values are restored onto
# a freshly supplied lpdf. Empty for a plain lpdf.
reparam_sources(lpdf) = [idx => value.source for (idx, value) in reparametrizer(lpdf).pairs]

# A serializable snapshot of everything needed to resume at a checkpoint,
# EXCLUDING the lpdf and runtime handles (`progress`/`start_time`). The recording
# posterior is decomposed into its arrays + recorder; its wrapped lpdf and its
# rng (shared with `state.rng`, serialized once here) are omitted and rewired on
# resume. `energy_options`/`kinetic_energy` are omitted too: both are pure
# functions of `scale_options` (which they alias) and are rebuilt on resume.
checkpoint_payload(state::AWMState, stage::Symbol) = (;
    stage,
    state.n_draws, state.stepsize_adaptation_limit, state.variance_cond_target,
    state.nonlinear_adapt, state.monitor_ess, state.recording_target, state.kwargs,
    state.algorithm, state.stepsize_adaptation, state.dimension,
    halo_position=state.recording_lpdf.halo_position,
    halo_gradient=state.recording_lpdf.halo_gradient,
    posterior_position=state.recording_lpdf.posterior_position,
    posterior_gradient=state.recording_lpdf.posterior_gradient,
    recorder=state.recording_lpdf.recorder,
    state.rng, state.position, state.scale_options, state.active_transformation,
    state.variance_memory, state.variance_position, state.variance_gradient,
    state.variance_cond, state.scale_changes, state.position_and_gradient, state.stepsize,
    state.stepsize_state, state.n_evaluations, state.total_evaluation_counter,
    state.outer_counter, state.current_transition_counter, state.total_transition_counter,
    state.ess, state.steps_per_draw, state.n_divergent, state.n_divergent_samples,
    state.restart, state.n_samples,
    reparam_sources=reparam_sources(state.lpdf),
)

# Opt-in on-disk checkpoint write at a boundary. Pure read of `state` + file
# writes — consumes no rng and mutates nothing, so it never perturbs the run.
# Writes both a stage-specific file (`cp_init.jls` / `cp_window_<n>.jls`, for
# "enter at a specific state") and an overwritten `cp_latest.jls`.
_write_checkpoint(::Nothing, state::AWMState, stage::Symbol) = nothing
_write_checkpoint(dir, state::AWMState, stage::Symbol) = begin
    mkpath(dir)
    payload = checkpoint_payload(state, stage)
    name = stage === :init ? "cp_init.jls" : "cp_window_$(state.outer_counter).jls"
    serialize(joinpath(dir, name), payload)
    serialize(joinpath(dir, "cp_latest.jls"), payload)
    nothing
end

# Observational checkpoint callback. Fires at the two checkpoint boundaries
# (`stage = :init` after CP-0, `stage = :window` after each CP-N) with the live
# `state`. It is OBSERVATIONAL: it may read `state` and request an early stop by
# returning `true`, but must not mutate `state` (mutation would break the
# byte-identity guarantee, which is then the caller's responsibility). The
# default (`nothing`) is never called, so the default path is byte-identical.
_fire_callback(::Nothing, state::AWMState, stage::Symbol) = false
_fire_callback(callback, state::AWMState, stage::Symbol) = callback(state, stage) === true

# Restore the reparametrizer's scalar `source` centerings (from `reparam_sources`)
# onto a freshly-supplied lpdf, in place. The lpdf brings its own reparametrizer
# structure (targets + index-extraction closures); only the mutated `source`
# scalars are overwritten. No-op for a plain lpdf (empty `sources`).
restore_reparam_sources!(lpdf, sources) = begin
    isempty(sources) && return lpdf
    ir = reparametrizer(lpdf)
    ir.pairs .= [
        idx => Reparametrization(value.target, src, value.args...)
        for ((idx, value), (_, src)) in zip(ir.pairs, sources)
    ]
    lpdf
end

# Reconstruct a live `AWMState` from a deserialized checkpoint `p` and a freshly
# supplied `lpdf`. Rewires the recording posterior around `lpdf` (sharing the one
# deserialized rng between driver and recorder), rebuilds `energy_options` from
# `scale_options`, recomputes `kinetic_energy`, and restores the reparametrizer
# scalars onto `lpdf`. `progress`/`start_time` are fresh runtime handles.
restore_state(p, lpdf, progress) = begin
    restore_reparam_sources!(lpdf, p.reparam_sources)
    recording_lpdf = RecordingPosterior2(
        lpdf, p.halo_position, p.halo_gradient, p.posterior_position, p.posterior_gradient,
        NUTSLeaves(p.dimension), p.recorder, p.rng,
    )
    energy_options = map(p.scale_options) do L
        DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))
    end
    kinetic_energy = energy_options[p.active_transformation]
    AWMState(
        lpdf, p.n_draws, p.stepsize_adaptation_limit, p.variance_cond_target, p.nonlinear_adapt,
        p.monitor_ess, p.recording_target, p.kwargs, p.algorithm, p.stepsize_adaptation, p.dimension,
        progress, time_ns(),
        p.rng, recording_lpdf, p.position, p.scale_options, energy_options, p.active_transformation,
        kinetic_energy, p.variance_memory, p.variance_position, p.variance_gradient, p.variance_cond,
        p.scale_changes, p.position_and_gradient, p.stepsize, p.stepsize_state, p.n_evaluations,
        p.total_evaluation_counter, p.outer_counter, p.current_transition_counter,
        p.total_transition_counter, p.ess, p.steps_per_draw, p.n_divergent, p.n_divergent_samples,
        p.restart, p.n_samples,
    )
end

"""
    adaptive_warmup_mcmc(rng, lpdf; kwargs...)
    adaptive_warmup_mcmc(rngs::AbstractArray, lpdf_or_lpdfs; parallel=true, kwargs...)

Run windowed adaptive NUTS warm-up + sampling against the
`LogDensityProblems`-compatible `lpdf`, returning a `NamedTuple` of
posterior positions/gradients plus diagnostics. Multi-chain dispatch
broadcasts over `rngs` and (optionally) per-chain log densities.

The warm-up procedure is windowed and inspired by [Stan](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)'s
and [nutpie](https://github.com/pymc-devs/nutpie)'s warm-up procedures, but differs in several important ways:

* Initializes via Pathfinder (LBFGS-based variational approximation).
* Warm-up windows target a number of GRADIENT EVALUATIONS rather than
  MCMC transitions. Default 1000, doubled after every window.
* Uses POSITIONS AND GRADIENTS (like nutpie), plus one state from every NUTS
  tree traversal. The state is drawn from the exact marginal proposal
  probabilities induced by all leaf Hamiltonian errors, including the
  probability of staying at the initial state. Up to `recording_target` such
  states are kept.
* Learns three candidate linear transformations in parallel at the end
  of every warm-up window:
    * Pathfinder's initial transformation + an updated diagonal scaling,
    * A standard diagonal "mass matrix",
    * A novel, adaptive sequence of Householder reflections followed by
      diagonal scaling.

  Selection minimises `loss(p', g') = sum(abs2(log(std(p') * std(g'))))`
  on the transformed intermediate positions/gradients — zero for an
  uncorrelated Normal target.
* Adapts step size for only the first `stepsize_adaptation_limit`
  transitions per window (default 50), then freezes the step size and
  treats subsequent transitions as posterior samples.
* Stops warm-up adaptively: if the marginal-scale condition number drops
  below `variance_cond_target` (default `2.0`), no new window starts.

If `nonlinear_adapt=true` (the default) and `lpdf` wraps a
[`ReparametrizedProblem`](@ref), the active [`IndexedReparametrization`](@ref)
is optimised at the end of every warm-up window, and posterior samples
are transformed back to the original parametrization before returning.

# `init` kwarg

`init` controls per-chain initialization:

* `missing` (default) — random `Uniform(-2, +2)` start, then Pathfinder.
* a `Real` — random `Uniform(-init, +init)` start, then Pathfinder.
* a `Distribution` — sample from it, then Pathfinder.
* an `AbstractVector` — use as the unconstrained starting position, then
  Pathfinder.
* a `PathfinderResult` — take the first draw, skip running Pathfinder.
* a `NamedTuple` — interpret as a pre-built initialization
  (`position`, `position_and_gradient`, `scale`, `squared_scale`); skips
  Pathfinder entirely.

For the multi-chain method, pass either a scalar to broadcast or a
`Vector` of length `length(rngs)` for per-chain initial values. There is
no separate `initial_params` kwarg.

# Selected keyword arguments

* `n_draws=1000` — number of posterior draws to collect.
* `n_evaluations=1000` — gradient-evaluation budget for the first
  window; doubled each subsequent window.
* `recording_target=1000` — maximum number of acceptance-weighted NUTS leaf
  positions/gradients to keep.
* `stepsize_adaptation_limit=50` — per-window cap on step-size
  adaptation transitions.
* `target_acceptance_rate=0.8`, `max_tree_depth=10` — standard NUTS knobs.
* `nonlinear_adapt=true` — whether to activate the reparametrization
  hooks (no-op when `lpdf` carries no reparametrization).
* `variance_cond_target=2.0` — restart threshold on the marginal-scale
  condition number.
* `progress=nothing`, `description="MCMC"`, `monitor_ess` — progress and
  diagnostic reporting via Treebars.
* `parallel=true` (multi-chain only) — run chains on `Threads.@threads`.
* `callback=nothing` — observational checkpoint callback (see below).
* `checkpoint_dir=nothing` — opt-in on-disk checkpointing (see below).

# Checkpoints, callbacks and resume

The warm-up has two checkpoint boundaries: **CP-0**, right after initialization
(Pathfinder), and **CP-N**, after each outer warm-up window ("big iteration").
Three opt-in, independent mechanisms hang off these boundaries; all default to
off, and with them off the run is byte-for-byte identical to the plain sampler.

* **`callback=(state, stage) -> should_stop`** fires at each boundary
  (`stage ∈ (:init, :window)`) with the live state. It is *observational*: it
  may read `state` and request an early stop by returning `true`, but must not
  mutate `state` (mutation makes byte-identity the caller's responsibility).
* **`checkpoint_dir=path`** serializes a resumable snapshot at each boundary
  (`cp_init.jls`, `cp_window_<n>.jls`, and an overwritten `cp_latest.jls`). The
  snapshot excludes the (possibly non-serializable) inner problem and stores the
  reparametrizer only as its scalar `source` centerings. The multi-chain method
  writes chain `i` under `path/chain_<i>/`.
* **[`resume_warmup_mcmc`](@ref)`(lpdf, cp_path; ...)`** re-supplies `lpdf` and
  continues from a checkpoint, returning the same result as an uninterrupted
  run. The multi-chain form takes the parent `dir` and resumes each chain from
  its `chain_<i>/` subdirectory.

# Returns

For the single-chain method, a `NamedTuple` with fields including
`initial_position`, `halo_position`, `halo_gradient`,
`posterior_position`, `posterior_gradient`, `ess`, `scale_options`,
`active_transformation`, `stepsize`, `total_evaluation_counter`,
`n_divergent_samples`, `position_and_gradient`, `scale_changes`.
For the multi-chain method, a `Vector` of such `NamedTuple`s.
"""
adaptive_warmup_mcmc(
    rng, lpdf;
    # The number of posterior draws
    n_draws=1000,
    # The number of GRADIENT EVALUATIONS in the first window
    n_evaluations=1000,
    # The upper limit of (intermediate) positions and gradients that will be recorded and then used for adaptation
    recording_target=1000,
    # The maximum number of transitions (per window) for which the stepsize gets adapted
    stepsize_adaptation_limit=50,
    target_acceptance_rate=.8,
    max_tree_depth=10,
    init=missing,
    progress=nothing,
    description="MCMC",
    monitor_ess=!isnothing(progress),
    nonlinear_adapt=true,
    variance_cond_target=2.,
    # Observational checkpoint callback `(state, stage) -> should_stop`; see
    # `_fire_callback`. Default `nothing` keeps the run byte-identical.
    callback=nothing,
    # Opt-in on-disk checkpointing: a directory to write resumable state into at
    # each checkpoint. Default `nothing` writes nothing and keeps the run
    # byte-identical (the write is a pure read + file I/O).
    checkpoint_dir=nothing,
    kwargs...
    # For monitoring purposes: Displays the progress and additional info
) = with_progress(progress, n_draws+stepsize_adaptation_limit; description) do progress
    state = init_state(
        rng, lpdf, progress;
        n_draws, n_evaluations, recording_target, stepsize_adaptation_limit,
        target_acceptance_rate, max_tree_depth, init, monitor_ess,
        nonlinear_adapt, variance_cond_target, kwargs...
    )
    _write_checkpoint(checkpoint_dir, state, :init)                    # CP-0
    stop = _fire_callback(callback, state, :init)
    while !stop && size(state.recording_lpdf.posterior_position, 2) < state.n_draws
        run_outer_iteration!(state)
        _write_checkpoint(checkpoint_dir, state, :window)             # CP-N
        stop = _fire_callback(callback, state, :window)
    end
    finalize_warmup!(state)
end

"""
    resume_warmup_mcmc(lpdf, checkpoint_path; progress=nothing, description="MCMC",
                       callback=nothing, checkpoint_dir=nothing)

Resume [`adaptive_warmup_mcmc`](@ref) from an on-disk checkpoint written by the
`checkpoint_dir` kwarg. `lpdf` is re-supplied by the caller (the checkpoint does
not store the — possibly non-serializable — inner problem; only the
reparametrizer's scalar state is restored onto it). The run continues from the
saved boundary and returns the same `NamedTuple` as a single, uninterrupted run:
for a fixed seed, resuming is byte-for-byte identical to running straight
through. `callback` and `checkpoint_dir` behave as in `adaptive_warmup_mcmc`.
"""
resume_warmup_mcmc(lpdf, checkpoint_path;
    progress=nothing, description="MCMC", callback=nothing, checkpoint_dir=nothing,
) = begin
    payload = deserialize(checkpoint_path)
    with_progress(progress, payload.n_draws+payload.stepsize_adaptation_limit; description) do progress
        state = restore_state(payload, lpdf, progress)
        # Continue from AFTER the resumed boundary: the original run already fired
        # the callback / wrote the checkpoint there, so we do not re-fire it here.
        stop = false
        while !stop && size(state.recording_lpdf.posterior_position, 2) < state.n_draws
            run_outer_iteration!(state)
            _write_checkpoint(checkpoint_dir, state, :window)
            stop = _fire_callback(callback, state, :window)
        end
        finalize_warmup!(state)
    end
end

"""
    resume_warmup_mcmc(lpdfs::AbstractArray, dir; checkpoint_name="cp_latest.jls",
                       parallel=true, progress=nothing, description="MCMC",
                       callback=nothing, checkpoint_dir=nothing)

Multi-chain resume: resume chain `i` from `dir/chain_<i>/<checkpoint_name>` — the
per-chain layout written by the multi-chain `adaptive_warmup_mcmc(rngs, lpdfs;
checkpoint_dir=dir)`. Returns a `Vector` of per-chain result `NamedTuple`s.
"""
resume_warmup_mcmc(lpdfs::AbstractArray, dir; checkpoint_name="cp_latest.jls",
    parallel=true, progress=nothing, description="MCMC", callback=nothing, checkpoint_dir=nothing,
) = with_progress(progress, length(lpdfs); description) do progress
    n_chains = length(lpdfs)
    rv = Vector{Any}(missing, n_chains)
    if parallel
        Threads.@threads for i in 1:n_chains
            rv[i] = resume_warmup_mcmc(lpdfs[i], joinpath(_chain_dir(dir, i), checkpoint_name);
                progress, description=description*".$i", callback, checkpoint_dir=_chain_dir(checkpoint_dir, i))
            update_progress!(progress)
        end
    else
        for i in 1:n_chains
            rv[i] = resume_warmup_mcmc(lpdfs[i], joinpath(_chain_dir(dir, i), checkpoint_name);
                progress, description=description*".$i", callback, checkpoint_dir=_chain_dir(checkpoint_dir, i))
            update_progress!(progress)
        end
    end
    identity.(rv)
end
ensurevector(x, n) = Fill(x, n)
ensurevector(x::AbstractVector, n) = begin
    @assert length(x) == n
    x
end
# Per-chain checkpoint subdirectory, so chains never collide on `cp_*.jls`.
_chain_dir(::Nothing, i) = nothing
_chain_dir(dir, i) = joinpath(dir, "chain_$i")

adaptive_warmup_mcmc(rngs::AbstractArray, lpdf; kwargs...) = adaptive_warmup_mcmc(rngs, fill(lpdf, size(rngs)); kwargs...)
adaptive_warmup_mcmc(rngs::AbstractArray, lpdfs::AbstractArray; parallel=true, progress=nothing,
monitor_ess=!isnothing(progress), description="MCMC", init=missing, checkpoint_dir=nothing, kwargs...) = with_progress(progress, length(rngs); description) do progress
    n_chains = length(rngs)
    rv = Vector{Any}(missing, n_chains)
    init = ensurevector(init, n_chains)
    if parallel
        Threads.@threads for i in 1:n_chains
            rv[i] = adaptive_warmup_mcmc(rngs[i], lpdfs[i]; progress, monitor_ess, description=description*".$i", init=init[i], checkpoint_dir=_chain_dir(checkpoint_dir, i), kwargs...)
            update_progress!(progress)
        end
    else
        for i in 1:n_chains
            rv[i] = adaptive_warmup_mcmc(rngs[i], lpdfs[i]; progress, monitor_ess, description=description*".$i", init=init[i], checkpoint_dir=_chain_dir(checkpoint_dir, i), kwargs...)
            update_progress!(progress)
        end
    end
    if !isnothing(progress)
        n_divergent_samples = sum(rvi->rvi.n_divergent_samples, rv)
        n_samples = sum(rvi->size(rvi.posterior_position, 2), rv)
        update_progress!(
            progress,
            (monitor_ess ? "min. ESS: $(short_string(minimum((MCMCDiagnosticTools.ess(permutedims(stack(getproperty.(rv, :posterior_position)), (2, 3, 1))))))), " : "") * "divergent: $(short_string(100*n_divergent_samples/n_samples))%"
        )
    end
    identity.(rv)
end
