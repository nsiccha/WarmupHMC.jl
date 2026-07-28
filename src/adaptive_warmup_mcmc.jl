_initial_diagonal_scale(squared_scale::AbstractMatrix) =
    Diagonal(sqrt.(Float64.(diag(squared_scale))))

function _initial_pathfinder_scale(squared_scale::AbstractMatrix, dimension)
    decomposition = factorize(squared_scale)
    factor = decomposition isa Diagonal ? _initial_diagonal_scale(squared_scale) : decomposition.L
    MatrixFactorization(factor, Diagonal(ones(dimension)))
end

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
    scale = _initial_pathfinder_scale(squared_scale, dimension)
    initialize_mcmc(lpdf, (;position, position_and_gradient, scale, squared_scale))
end
initialize_mcmc(lpdf, init::NamedTuple; kwargs...) = begin
    required = (:position, :squared_scale)
    all(k -> hasproperty(init, k), required) || throw(ArgumentError(
        "init NamedTuple must contain `position` and `squared_scale`; got keys $(keys(init))"
    ))

    dimension = LogDensityProblems.dimension(lpdf)
    position = init.position
    position isa AbstractVector || throw(ArgumentError(
        "init.position must be an AbstractVector of length $dimension; got $(typeof(position))"
    ))
    length(position) == dimension || throw(ArgumentError(
        "init.position must have length $dimension; got length $(length(position))"
    ))

    squared_scale = init.squared_scale
    if squared_scale isa AbstractVector
        length(squared_scale) == dimension || throw(ArgumentError(
            "init.squared_scale as a diagonal variance vector must have length $dimension; " *
            "got length $(length(squared_scale))"
        ))
        squared_scale = Diagonal(collect(Float64, squared_scale))
    elseif squared_scale isa AbstractMatrix
        size(squared_scale) == (dimension, dimension) || throw(ArgumentError(
            "init.squared_scale as a full squared-scale matrix must have size " *
            "($dimension, $dimension); got size $(size(squared_scale))"
        ))
    else
        throw(ArgumentError(
            "init.squared_scale must be either a diagonal variance vector of length $dimension " *
            "or a full squared-scale matrix of size ($dimension, $dimension); " *
            "got $(typeof(squared_scale))"
        ))
    end

    merge(init, (;position, squared_scale))
end
"Set other defaults and works around https://github.com/mlcolab/Pathfinder.jl/issues/248"
mypathfinder(args...;
    ndraws=1, ndraws_elbo=1, ntries=1,
    history_length=6,
    # WarmupHMC targets already provide `logdensity_and_gradient`. Telling
    # Optimization to synthesize another gradient makes its default
    # AutoForwardDiff path call `logdensity` with Dual-valued parameters,
    # which native-backed targets such as BridgeStan cannot accept.
    adtype=Pathfinder.SciMLBase.NoAD(),
    optimizer=Pathfinder.Optim.LBFGS(;
        m=history_length,
        linesearch=Pathfinder.LineSearches.HagerZhang(),
        alphaguess=Pathfinder.LineSearches.InitialHagerZhang()
    ),
    kwargs...
) = pathfinder(
    args...;
    ndraws, ntries, ndraws_elbo, adtype, optimizer, kwargs...
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
const _LINEAR_RESTART_SOURCES = (
    :halo,
    :all_good_leaves,
    :nuts_weighted,
)
const _LINEAR_TRAJECTORY_WEIGHTINGS = (:unit, :stepsize)

function _validate_linear_restart_source(source, weighting)
    source in _LINEAR_RESTART_SOURCES || throw(ArgumentError(
        "linear_restart_source must be one of " *
        "$(join(repr.(_LINEAR_RESTART_SOURCES), ", ")); got $(repr(source))",
    ))
    source === :halo && weighting !== :unit && throw(ArgumentError(
        "linear_trajectory_weighting=$(repr(weighting)) cannot affect " *
        "linear_restart_source=:halo; use :unit or opt into a running source",
    ))
    source
end
function _validate_linear_trajectory_weighting(weighting)
    weighting in _LINEAR_TRAJECTORY_WEIGHTINGS || throw(ArgumentError(
        "linear_trajectory_weighting must be one of " *
        "$(join(repr.(_LINEAR_TRAJECTORY_WEIGHTINGS), ", ")); got $(repr(weighting))",
    ))
    weighting
end
_uses_running_linear_restart(source) = source !== :halo

mutable struct AWMState{L,K,A,DA,P,R,RL,NR,SO,EO,VP,VG,TA,POS,PG,SS,MN}
    # ── config (set once) ──
    lpdf::L
    n_draws::Int
    stepsize_adaptation_limit::Int
    variance_cond_target::Float64
    linear_restart_source::Symbol
    linear_trajectory_weighting::Symbol
    linear_metric_fallback::Bool
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
    nonlinear_recorder::NR
    position::POS
    scale_options::SO
    energy_options::EO
    active_transformation::Symbol
    # Untyped: reassigned across `energy_options` entries, which are
    # heterogeneously typed (each transformation yields a distinct
    # GaussianKineticEnergy type), so no single concrete type fits.
    kinetic_energy::Any
    variance_memory::Vector{Float64}
    transformed_position_memory::Vector{Float64}
    variance_position::VP
    variance_gradient::VG
    transformed_adaptation::TA
    variance_cond::Float64
    scale_changes::Vector{Float64}
    linear_metric_fallbacks::Int
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
    # ── draws DROPPED at the most recent restart (consumer-facing, inert on resume) ──
    # A restarting window empties the recorder (`reset!(recording_lpdf)`) and the
    # checkpoint is serialized AFTER that, so CP-N would otherwise pair complete
    # resume state with ZERO retained draws. These hold exactly what that reset
    # threw away: the post-stepsize-adaptation draws of the epoch that just
    # ended, sampled under the metric and step size that were frozen for it.
    #
    # They are NEVER read back into the sampler — `restore_state` carries them
    # forward verbatim and nothing else touches them — so a resumed run cannot
    # mistake them for draws it still holds. `dimension × 0` until the first
    # restart.
    dropped_posterior_position::Matrix{Float64}
    dropped_posterior_gradient::Matrix{Float64}
    # `n_divergent_samples` as of that reset, so the dropped draws' divergence
    # rate is recoverable (the live counter is zeroed with them).
    dropped_n_divergent_samples::Int
end

function _fit_transformed_observation!(adaptation, position_memory, gradient_memory, scale,
                                       position, gradient, weight)
    iszero(weight) && return adaptation
    ldiv!(position_memory, scale, position)
    mul!(gradient_memory, scale', gradient)
    OnlineStatsBase.fit!(adaptation, position_memory, gradient_memory; dw=weight)
end

const _GOOD_LEAF_MIN_DH = log(1e-2)

function _fit_transformed_trajectory!(adaptation, position_memory, gradient_memory,
                                      scale, leaves,
                                      source, weighting, stepsize)
    source === :halo && return zero(stepsize)
    trajectory_weight = weighting === :unit ? one(stepsize) : stepsize
    total_weight = zero(stepsize)
    if source === :all_good_leaves
        for i in 2:length(leaves)
            leaves.dH[i] > _GOOD_LEAF_MIN_DH || continue
            _fit_transformed_observation!(
                adaptation, position_memory, gradient_memory, scale,
                view(leaves.position, :, i), view(leaves.gradient, :, i),
                trajectory_weight,
            )
            total_weight += trajectory_weight
        end
    else
        source === :nuts_weighted || throw(ArgumentError(
            "unsupported running linear restart source $(repr(source))",
        ))
        for i in eachindex(leaves.weights)
            weight = trajectory_weight * leaves.weights[i]
            iszero(weight) && continue
            _fit_transformed_observation!(
                adaptation, position_memory, gradient_memory, scale,
                view(leaves.position, :, i), view(leaves.gradient, :, i), weight,
            )
            total_weight += weight
        end
    end
    total_weight
end

function _halo_restart_scales!(out, position_stats, gradient_stats, memory,
                               scale, halo_position, halo_gradient)
    for (position, gradient) in zip(eachcol(halo_position), eachcol(halo_gradient))
        ldiv!(memory, scale, position)
        OnlineStatsBase.fit!(position_stats, memory)
        mul!(memory, scale', gradient)
        OnlineStatsBase.fit!(gradient_stats, memory)
    end
    out .= sqrt.(std.(position_stats.stats) ./ std.(gradient_stats.stats))
    for i in eachindex(position_stats.stats)
        position_stats.stats[i] = OnlineStatsBase.Variance()
        gradient_stats.stats[i] = OnlineStatsBase.Variance()
    end
    out
end

function _linear_restart_scales!(out, source, transformed_adaptation,
                                 position_stats, gradient_stats, memory, scale,
                                 halo_position, halo_gradient)
    if source === :halo
        _halo_restart_scales!(
            out, position_stats, gradient_stats, memory,
            scale, halo_position, halo_gradient,
        )
    else
        out .= marginal_scales(transformed_adaptation)
    end
    out
end

function _linear_restart_condition(scales)
    all(x -> isfinite(x) && x > 0, scales) || return Inf
    lo, hi = extrema(scales)
    hi / lo
end

function _apply_linear_metric_fallback!(scale_options, old_diagonal, correction;
                                        enabled, old_active, new_active,
                                        nonlinear_changed)
    enabled && old_active === :diagonal && new_active === :diagonal &&
        !nonlinear_changed && all(x -> isfinite(x) && x > 0, correction) || return false
    scale_options.diagonal.diag .= old_diagonal .* correction
    true
end

# Setup + initialization, up to and including the initial step-size search and
# the first progress tick. Returns the state as of CP-0 ("after Pathfinder").
init_state(
    rng, lpdf, progress;
    n_draws, n_evaluations, recording_target, stepsize_adaptation_limit,
    target_acceptance_rate, max_tree_depth, init, monitor_ess,
    nonlinear_adapt, nonlinear_evidence, nonlinear_trajectory_weighting,
    nonlinear_good_leaf_threshold, variance_cond_target,
    linear_restart_source=:halo, linear_trajectory_weighting=:unit,
    linear_metric_fallback=true, kwargs...
) = begin
    _validate_linear_trajectory_weighting(linear_trajectory_weighting)
    _validate_linear_restart_source(linear_restart_source, linear_trajectory_weighting)
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
    recorder = LimitedRecorder2(
        recording_target,
        # The initial "thinning" of intermediate positions and gradients: one
        # retained state per `thin` leaf evaluations, so a window of
        # `n_evaluations` gradient evaluations fills the whole ring.
        max(1, n_evaluations ÷ recording_target),
    )
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    nonlinear_recorder = NonlinearRecorder(
        lpdf;
        mode=nonlinear_evidence,
        trajectory_weighting=nonlinear_trajectory_weighting,
        good_leaf_threshold=nonlinear_good_leaf_threshold,
    )
    # Use Stan's initialization procedure if no initial position is given
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    # We currently learn three linear transformation options
    scale_options = (;
        # Corresponds to a standard diagonal mass matrix
        diagonal=_initial_diagonal_scale(squared_scale),
        # Corresponds to Pathfinder's linear transformation with an added diagonal scaling term that can be updated
        pathfinder=_initial_pathfinder_scale(squared_scale, dimension),
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
    transformed_position_memory = zeros(dimension)
    variance_position = OnlineStatsBase.Group([OnlineStatsBase.Variance() for i in 1:dimension])
    variance_gradient = OnlineStatsBase.Group([OnlineStatsBase.Variance() for i in 1:dimension])
    transformed_adaptation = WeightedScaleAdaptation(dimension)
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
        lpdf, n_draws, stepsize_adaptation_limit, variance_cond_target,
        linear_restart_source, linear_trajectory_weighting,
        linear_metric_fallback, nonlinear_adapt,
        monitor_ess, recording_target, (; kwargs...), algorithm, stepsize_adaptation, dimension,
        progress, start_time,
        rng, recording_lpdf, nonlinear_recorder,
        position, scale_options, energy_options, active_transformation,
        kinetic_energy, variance_memory, transformed_position_memory,
        variance_position, variance_gradient,
        transformed_adaptation, variance_cond, scale_changes, 0,
        position_and_gradient, stepsize, stepsize_state, n_evaluations,
        total_evaluation_counter, outer_counter, current_transition_counter,
        total_transition_counter, ess, steps_per_draw, n_divergent, n_divergent_samples,
        restart, n_samples,
        Matrix{Float64}(undef, dimension, 0), Matrix{Float64}(undef, dimension, 0), 0,
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
        trajectory_stepsize = state.stepsize
        state.position_and_gradient, stats = DynamicHMC.sample_tree(state.rng, algorithm, hamiltonian, state.position_and_gradient, state.stepsize)
        finalize_leaf_recording!(recording_lpdf, stats.depth)
        state.nonlinear_adapt && record_nonlinear!(
            state.nonlinear_recorder,
            lpdf,
            recording_lpdf.leaves,
            state.stepsize,
        )
        state.total_evaluation_counter += stats.steps
        current_evaluation_counter += stats.steps
        OnlineStatsBase.fit!(state.steps_per_draw, stats.steps)
        is_divergent = DynamicHMC.is_divergent(stats.termination)
        is_divergent && (state.n_divergent += 1)
        if _uses_running_linear_restart(state.linear_restart_source)
            _fit_transformed_trajectory!(
                state.transformed_adaptation,
                state.transformed_position_memory, state.variance_memory,
                state.scale_options[state.active_transformation],
                recording_lpdf.leaves, state.linear_restart_source,
                state.linear_trajectory_weighting, trajectory_stepsize,
            )
        end
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
            _linear_restart_scales!(
                state.variance_memory, state.linear_restart_source,
                state.transformed_adaptation,
                state.variance_position, state.variance_gradient,
                state.variance_memory, scale,
                recording_lpdf.halo_position, recording_lpdf.halo_gradient,
            )
            state.variance_cond = _linear_restart_condition(state.variance_memory)
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
    # Recompute the thinning factor for the intermediate positions and gradients,
    # so the larger window still fills exactly the `recording_target`-slot ring.
    recording_lpdf.recorder.thin = max(1, state.n_evaluations ÷ state.recording_target)
    state.restart || return state
    # Preserve what this restart is about to throw away, BEFORE any counter is
    # zeroed or `reset!(recording_lpdf)` runs at the end of this function. The
    # checkpoint is written after this call returns, so without this copy a
    # restarting window's payload carries no draws at all — see
    # `AWMState.dropped_posterior_position` and `checkpoint_payload`.
    state.dropped_posterior_position = Matrix{Float64}(recording_lpdf.posterior_position)
    state.dropped_posterior_gradient = Matrix{Float64}(recording_lpdf.posterior_gradient)
    state.dropped_n_divergent_samples = state.n_divergent_samples
    state.stepsize = DynamicHMC.final_ϵ(state.stepsize_state)
    state.stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, state.stepsize)
    state.stepsize = DynamicHMC.current_ϵ(state.stepsize_state)
    # Reset the so far recorded intermediate and MCMC positions and gradients
    state.current_transition_counter = 0
    state.steps_per_draw = OnlineStatsBase.Mean()
    state.n_divergent = 0
    state.n_divergent_samples = 0
    # `reset!(recording_lpdf)` below empties `posterior_position`, and `n_samples`
    # mirrors `size(posterior_position, 2)` — so it must be zeroed with the other
    # counters. Leaving it stale is invisible internally (every read is preceded by
    # a fresh assignment in the transition loop) but IS visible to consumers: the
    # CP-N checkpoint is serialized after this reset, so the payload would pair a
    # non-zero `n_samples` with zero retained draws.
    state.n_samples = 0
    # Update the linear transformation candidates and estimate the transformation loss,
    # using the INTERMEDIATE POSITIONS AND GRADIENTS.
    uses_running_restart = _uses_running_linear_restart(state.linear_restart_source)
    old_active = state.active_transformation
    old_diagonal = uses_running_restart ? copy(state.scale_options.diagonal.diag) : nothing
    old_sources = uses_running_restart ? reparam_sources(lpdf) : nothing
    state.nonlinear_adapt && (state.position_and_gradient = find_reparametrization!(
        lpdf, state.nonlinear_recorder, recording_lpdf.halo_position,
        recording_lpdf.halo_gradient, state.position_and_gradient,
    ))
    # Update the new linear transformation to be the one with the minimal estimated transformation loss.
    state.active_transformation = argmin(
        map(L->update_loss!(L, (recording_lpdf.halo_position), (recording_lpdf.halo_gradient); state.kwargs...), state.scale_options)
    )
    if uses_running_restart
        nonlinear_changed = reparam_sources(lpdf) != old_sources
        fallback_applied = _apply_linear_metric_fallback!(
            state.scale_options, old_diagonal, state.variance_memory;
            enabled=state.linear_metric_fallback,
            old_active,
            new_active=state.active_transformation,
            nonlinear_changed,
        )
        fallback_applied && (state.linear_metric_fallbacks += 1)
    end
    state.kinetic_energy = state.energy_options[state.active_transformation]
    update_progress!(progress, nothing;
        active_transformation=ActiveTransformation(state.kinetic_energy, state.scale_changes),
    )
    uses_running_restart && reset!(state.transformed_adaptation)
    reset!(recording_lpdf)
    state
end

# Final progress line, back-transform the posterior draws into the original
# parametrization, and assemble the returned NamedTuple.
finalize_warmup!(state::AWMState) = begin
    (; progress, recording_lpdf, lpdf) = state
    update_progress!(progress, (state.monitor_ess ? "min. ESS: $(short_string(state.ess[1])), " : "") * "divergent: $(short_string(100*state.n_divergent_samples/state.n_samples))%")
    # NOT gated on `nonlinear_adapt`. That flag gates whether the centering is
    # FITTED; it must not gate whether the returned draws are reported in the
    # model's own parametrization. The sampler works in the reparametrizer's
    # SOURCE frame whether or not anything was adapted, so gating this too meant
    # `nonlinear_adapt=false` silently returned source-frame draws with nothing
    # in the return value saying so — and the obvious consumer, a pinned-`c`
    # control arm compared against an adapted arm, then summarised two different
    # frames as if they were one.
    #
    # Safe unconditionally: `reparametrizer(::Any)` is an empty
    # `IndexedReparametrization` and `reparametrize!` returns immediately on
    # `isempty(ir.pairs)`, so this is a no-op for a plain lpdf and for a
    # `ReparametrizedProblem` carrying no spec.
    reparametrize!(lpdf, recording_lpdf.posterior_position)
    (;
        initial_position=state.position,
        halo_position=recording_lpdf.halo_position,
        halo_gradient=recording_lpdf.halo_gradient,
        posterior_position=recording_lpdf.posterior_position,
        posterior_gradient=recording_lpdf.posterior_gradient,
        ess=state.ess,
        scale_options=state.scale_options,
        active_transformation=state.active_transformation,
        linear_restart_source=state.linear_restart_source,
        linear_trajectory_weighting=state.linear_trajectory_weighting,
        linear_metric_fallback=state.linear_metric_fallback,
        linear_metric_fallbacks=state.linear_metric_fallbacks,
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

# A serializable snapshot of the TRANSIENT SAMPLER STATE at a checkpoint —
# nothing else.
#
# Three categories are deliberately absent:
#
#   * the lpdf and runtime handles (`progress`/`start_time`) — the lpdf is
#     possibly non-serializable, and the caller re-supplies it;
#   * `energy_options`/`kinetic_energy` — pure functions of `scale_options`
#     (which they alias), rebuilt on resume;
#   * **CONFIG** — `n_draws`, `stepsize_adaptation_limit`, `variance_cond_target`,
#     `linear_restart_source`, `linear_trajectory_weighting`,
#     `linear_metric_fallback`, `nonlinear_adapt`,
#     `monitor_ess`, `recording_target`, `kwargs`, `algorithm` and
#     `stepsize_adaptation` all used to live here. They are caller-owned
#     INPUT, exactly like the lpdf, and persisting one but not the other was
#     arbitrary. Taking config from the resuming CALL instead is what makes
#     post-hoc reconfiguration possible at all — most importantly resuming with a
#     larger `n_draws` to keep sampling rather than re-running a fixed-length
#     batch. See `resume=` on `adaptive_warmup_mcmc`.
#
# `dimension` is kept, but as a VALIDATION anchor rather than config: it must
# still match the freshly-supplied lpdf.
#
# The old `stage` field is gone. It was written at every boundary and read by
# nothing — the filename (`cp_init.jls` vs `cp_window_<n>.jls`) already carries it.
#
# READING DRAWS OUT OF A CHECKPOINT. `posterior_position` holds the draws the
# run still HAS, and is empty at every checkpoint whose window restarted (the
# reset happens before the write). `dropped_posterior_position` holds the ones
# that restart discarded. So a consumer materializing partial results wants:
#
#     draws = isempty(payload.posterior_position) ?
#         get(payload, :dropped_posterior_position, nothing) : payload.posterior_position
#
# Both are in the sampler's WORKING parametrization — `finalize_warmup!` is what
# applies the back-transform, and it never runs for a checkpoint; `reparam_sources`
# is in the payload for exactly that. The dropped draws are legitimate MCMC draws
# for a shorter run under a less-adapted metric: within an epoch the kernel is
# fixed (metric and step size frozen once `stepsize_adaptation_limit` is passed),
# which is precisely why the sampler recorded them.
#
# The three `dropped_*` keys are ADDITIVE — `checkpoint_schema_version` does NOT
# move for them, so existing readers are unaffected and new readers use
# `get(payload, key, default)`.
#
# `nonlinear_recorder.mode` and `custom_candidate_scoring` are both DECLARED
# state. The EFFECTIVE evidence mode is a function of the two rather than a
# stored key, because a plan silently reads `:linear_pool` as online
# all-good-leaf evidence — see `_effective_nonlinear_evidence` in
# `Reparametrizations.jl` for the rule and for why storing it is the wrong move.
checkpoint_payload(state::AWMState) = (;
    schema_version=checkpoint_schema_version(),
    sampler=:adaptive,
    state.dimension,
    halo_position=state.recording_lpdf.halo_position,
    halo_gradient=state.recording_lpdf.halo_gradient,
    posterior_position=state.recording_lpdf.posterior_position,
    posterior_gradient=state.recording_lpdf.posterior_gradient,
    recorder=state.recording_lpdf.recorder,
    nonlinear_recorder=state.nonlinear_recorder,
    state.rng, state.position, state.scale_options, state.active_transformation,
    state.variance_memory, state.variance_position, state.variance_gradient,
    linear_recorder=(;
        source=state.linear_restart_source,
        trajectory_weighting=state.linear_trajectory_weighting,
        adaptation=state.transformed_adaptation,
    ),
    state.variance_cond, state.scale_changes,
    state.linear_metric_fallbacks, state.position_and_gradient, state.stepsize,
    state.stepsize_state, state.n_evaluations, state.total_evaluation_counter,
    state.outer_counter, state.current_transition_counter, state.total_transition_counter,
    state.ess, state.steps_per_draw, state.n_divergent, state.n_divergent_samples,
    state.restart, state.n_samples,
    state.dropped_posterior_position, state.dropped_posterior_gradient,
    state.dropped_n_divergent_samples,
    reparam_sources=reparam_sources(state.lpdf),
    custom_candidate_scoring=_has_custom_candidate_scoring(state.lpdf),
)

# Opt-in on-disk checkpoint write at a boundary. Pure read of `state` + file
# writes — consumes no rng and mutates nothing, so it never perturbs the run.
# Writes both a stage-specific file (`cp_init.jls` / `cp_window_<n>.jls`, for
# "enter at a specific state") and an overwritten `cp_latest.jls`.
#
# Writes are ATOMIC: serialize to a temp file in the SAME directory, then rename.
# A rename within one filesystem is atomic, so a crash (or a kill) mid-write can
# never leave a truncated file behind — a reader sees either the previous
# complete checkpoint or the new one. This matters most for `cp_latest.jls`,
# which is what `resume_warmup_mcmc` reads by default: a plain `serialize`
# straight to that path leaves it corrupt if the process dies mid-write, i.e.
# exactly the crash the checkpoint exists to survive.
_atomic_serialize(path, payload) = begin
    tmp, io = mktemp(dirname(path); cleanup=false)
    try
        serialize(io, payload)
        close(io)
        mv(tmp, path; force=true)
    catch
        close(io)
        rm(tmp; force=true)
        rethrow()
    end
    nothing
end

_write_checkpoint(::Nothing, state::AWMState, stage::Symbol) = nothing
_write_checkpoint(dir, state::AWMState, stage::Symbol) = begin
    mkpath(dir)
    payload = checkpoint_payload(state)
    name = stage === :init ? "cp_init.jls" : "cp_window_$(state.outer_counter).jls"
    _atomic_serialize(joinpath(dir, name), payload)
    _atomic_serialize(joinpath(dir, "cp_latest.jls"), payload)
    nothing
end

_checkpoint_files(dir) = isdir(dir) ?
    sort!(filter(f -> startswith(f, "cp_") && endswith(f, ".jls"), readdir(dir))) : String[]

"""
    resolve_checkpoint_dir(dir, resume, overwrite) -> payload_or_nothing

Decide what pointing a sampler at `checkpoint_dir` means, and refuse to guess.

A checkpoint directory that already holds state is ambiguous: the caller might
mean "continue that run" or "throw it away and start over", and silently picking
either one is a way to lose a long run. So a non-empty directory REQUIRES
`resume=true` or `overwrite=true`; without one, this errors and names what it
found. An empty or absent directory needs no flag.

Returns the deserialized `cp_latest.jls` payload to resume from, or `nothing` to
start fresh.
"""
resolve_checkpoint_dir(::Nothing, resume::Bool, overwrite::Bool) = begin
    (resume || overwrite) && throw(ArgumentError(
        "`resume`/`overwrite` are meaningless without `checkpoint_dir`."))
    nothing
end
resolve_checkpoint_dir(dir, resume::Bool, overwrite::Bool) = begin
    resume && overwrite && throw(ArgumentError(
        "`resume=true` and `overwrite=true` are mutually exclusive."))
    existing = _checkpoint_files(dir)
    if isempty(existing)
        resume && throw(ArgumentError(
            "`resume=true` but no checkpoints were found in $(repr(dir))."))
        return nothing
    end
    overwrite && (foreach(f -> rm(joinpath(dir, f)), existing); return nothing)
    resume || throw(ArgumentError("""
    $(repr(dir)) already contains $(length(existing)) checkpoint(s): $(join(existing, ", ")).

    Refusing to guess. Pass one of:
      resume=true     continue that run (config comes from THIS call, so a larger
                      `n_draws` keeps sampling rather than restarting)
      overwrite=true  discard them and start fresh
    """))
    latest = joinpath(dir, "cp_latest.jls")
    isfile(latest) || throw(ArgumentError(
        "`resume=true` but $(repr(latest)) is missing (found: $(join(existing, ", ")))."))
    deserialize(latest)
end

"""
    guard_run_dir!(dir, resume, overwrite, sampler::Symbol)

The multi-chain / whole-run counterpart of [`resolve_checkpoint_dir`](@ref), for
samplers that lay out `dir/chain_<i>/` subdirectories rather than a single
`cp_*.jls` set.

Same rule: a directory that already holds a run is ambiguous, so it requires an
explicit `resume=true` or `overwrite=true`. This is worth having even where
`resume` is not yet implemented — without it, pointing a second run at a live
checkpoint directory silently interleaves two runs' chain state.
"""
guard_run_dir!(::Nothing, resume::Bool, overwrite::Bool, sampler::Symbol) = begin
    (resume || overwrite) && throw(ArgumentError(
        "`resume`/`overwrite` are meaningless without `checkpoint_dir`."))
    nothing
end
guard_run_dir!(dir, resume::Bool, overwrite::Bool, sampler::Symbol) = begin
    resume && overwrite && throw(ArgumentError(
        "`resume=true` and `overwrite=true` are mutually exclusive."))
    existing = isdir(dir) ?
        sort!(filter(f -> startswith(f, "chain_") || startswith(f, "run_"), readdir(dir))) : String[]
    if isempty(existing)
        resume && throw(ArgumentError(
            "`resume=true` but no run was found in $(repr(dir))."))
        return nothing
    end
    if overwrite
        foreach(f -> rm(joinpath(dir, f); recursive=true), existing)
        return nothing
    end
    resume && throw(ArgumentError(
        "`resume=true` is not supported by `$(sampler)_warmup_mcmc` yet — only " *
        "`adaptive_warmup_mcmc` can resume. Use `overwrite=true` to start fresh, " *
        "or point `checkpoint_dir` somewhere empty."))
    throw(ArgumentError("""
    $(repr(dir)) already contains a run: $(join(existing, ", ")).

    Refusing to guess — a second run writing here would interleave with it. Pass
    `overwrite=true` to discard it, or point `checkpoint_dir` somewhere empty.
    """))
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
    ir = reparametrizer(lpdf)
    if !isempty(sources)
        ir.pairs .= [
            idx => Reparametrization(value.target, src, value.args...)
            for ((idx, value), (_, src)) in zip(ir.pairs, sources)
        ]
    end
    _synchronize_scoring!(lpdf)
    lpdf
end

"""
    checkpoint_sampler(payload) -> Symbol

Which sampler wrote `payload`. Checkpoints written before the `sampler` tag
existed carry no field; adaptive was the only writer then, so an absent tag reads
as `:adaptive`. That keeps every pre-tag checkpoint readable — no migration.
"""
checkpoint_sampler(p) = hasproperty(p, :sampler) ? p.sampler : :adaptive

"""
    check_checkpoint_compatible(payload, reader::Symbol, accepted)

The masquerade guard. Every sampler writes into the same `chain_<i>/cp_*.jls`
layout, so a directory alone cannot say which sampler produced it — a reader must
check the tag rather than discover the mismatch as a missing field several frames
deep.

`accepted` encodes the one-way lattice: adaptive reads only its own state,
cooperative may additionally adopt adaptive state, clustered may read anything.
The asymmetry is not stylistic — a clustered chain's scale is pooled across its
cluster-mates (`cluster_and_adapt!`), so clustered state is only meaningful as a
whole ensemble and must never be resumed as a lone adaptive or cooperative chain.
"""
check_checkpoint_compatible(p, reader::Symbol, accepted) = begin
    writer = checkpoint_sampler(p)
    writer in accepted && return nothing
    throw(ArgumentError("""
    This checkpoint was written by `$(writer)_warmup_mcmc`, which `$(reader)_warmup_mcmc` cannot resume.

      accepted here: $(join(string.(accepted), ", "))

    Sampler state does not transfer in this direction. Resume it with \
    `$(writer)_warmup_mcmc`, or point `checkpoint_dir` at a fresh directory.
    """))
end

function _check_candidate_scoring_compatible(p, lpdf)
    checkpoint_custom = get(p, :custom_candidate_scoring, false)
    supplied_custom = _has_custom_candidate_scoring(lpdf)
    checkpoint_custom == supplied_custom && return nothing
    if checkpoint_custom
        throw(ArgumentError(
            "this checkpoint was written with a custom CandidateScoringPlan, but " *
            "the supplied log density has only the default scoring plan; reconstruct " *
            "the ReparametrizedProblem with scoring_plan=plan before resuming",
        ))
    end
    throw(ArgumentError(
        "this checkpoint was written with the default candidate scoring plan, but " *
        "the supplied log density has a custom CandidateScoringPlan; resume with the " *
        "same scoring strategy that wrote the checkpoint",
    ))
end

# Reconstruct a live `AWMState` from a deserialized checkpoint `p`, a freshly
# supplied `lpdf`, and CALLER-SUPPLIED CONFIG. Rewires the recording posterior
# around `lpdf` (sharing the one deserialized rng between driver and recorder),
# rebuilds `energy_options` from `scale_options`, recomputes `kinetic_energy`, and
# restores the reparametrizer scalars onto `lpdf`. `progress`/`start_time` are
# fresh runtime handles.
#
# `algorithm` and `stepsize_adaptation` are rebuilt from the scalars the caller
# passed rather than read back from the payload. That is safe in both directions:
# `DualAveragingState`'s type does not depend on δ, so a rebuilt adaptation always
# accepts the persisted `stepsize_state`.
restore_state(p, lpdf, progress;
    n_draws=1000, stepsize_adaptation_limit=50, variance_cond_target=2.,
    linear_restart_source=nothing, linear_trajectory_weighting=nothing,
    linear_metric_fallback=true,
    nonlinear_adapt=true, monitor_ess=!isnothing(progress),
    target_acceptance_rate=.8, max_tree_depth=10, recording_target=nothing,
    nonlinear_evidence=nothing, nonlinear_trajectory_weighting=nothing,
    nonlinear_good_leaf_threshold=nothing,
    kwargs...
) = begin
    check_checkpoint_compatible(p, :adaptive, (:adaptive,))
    _check_candidate_scoring_compatible(p, lpdf)
    lpdf_dimension = LogDensityProblems.dimension(lpdf)
    p.dimension == lpdf_dimension || throw(DimensionMismatch(
        "checkpoint holds a $(p.dimension)-dimensional problem but the supplied " *
        "lpdf has dimension $lpdf_dimension."
    ))
    # `recording_target` sizes the recorder's ring buffer, and `outer_count` (the
    # next destination) is persisted with it — so shrinking it below the retained
    # contents would corrupt the ring. `nothing` (the default) inherits the
    # persisted value, which keeps a plain resume working when the original run
    # used a non-default target; an explicit DIFFERENT value is refused rather
    # than silently discarding recorded halo state.
    isnothing(recording_target) || recording_target == p.recorder.target || throw(ArgumentError(
        "`recording_target` cannot change on resume (checkpoint has " *
        "$(p.recorder.target), got $recording_target): the recorder's ring buffer " *
        "and its retained contents are part of the persisted state. Omit it to inherit."
    ))
    restore_reparam_sources!(lpdf, p.reparam_sources)
    recording_lpdf = RecordingPosterior2(
        lpdf, p.halo_position, p.halo_gradient, p.posterior_position, p.posterior_gradient,
        NUTSLeaves(p.dimension), p.recorder, p.rng,
    )
    saved_nonlinear_recorder = get(p, :nonlinear_recorder, nothing)
    restored_mode = something(
        nonlinear_evidence,
        isnothing(saved_nonlinear_recorder) ? :linear_pool : saved_nonlinear_recorder.mode,
    )
    restored_threshold = something(
        nonlinear_good_leaf_threshold,
        isnothing(saved_nonlinear_recorder) ? log(1e-2) : saved_nonlinear_recorder.good_leaf_threshold,
    )
    restored_weighting = something(
        nonlinear_trajectory_weighting,
        isnothing(saved_nonlinear_recorder) ? :unit : saved_nonlinear_recorder.trajectory_weighting,
    )
    can_reuse_nonlinear = !isnothing(saved_nonlinear_recorder) &&
        saved_nonlinear_recorder.mode === restored_mode &&
        saved_nonlinear_recorder.trajectory_weighting === restored_weighting &&
        saved_nonlinear_recorder.good_leaf_threshold == restored_threshold
    nonlinear_recorder = can_reuse_nonlinear ? saved_nonlinear_recorder : NonlinearRecorder(
        lpdf;
        mode=restored_mode,
        trajectory_weighting=restored_weighting,
        good_leaf_threshold=restored_threshold,
    )
    saved_linear_recorder = get(p, :linear_recorder, nothing)
    restored_linear_source = something(
        linear_restart_source,
        isnothing(saved_linear_recorder) ? :halo : saved_linear_recorder.source,
    )
    restored_linear_weighting = something(
        linear_trajectory_weighting,
        isnothing(saved_linear_recorder) ? :unit : saved_linear_recorder.trajectory_weighting,
    )
    _validate_linear_trajectory_weighting(restored_linear_weighting)
    _validate_linear_restart_source(restored_linear_source, restored_linear_weighting)
    can_reuse_linear = !isnothing(saved_linear_recorder) &&
        saved_linear_recorder.source === restored_linear_source &&
        saved_linear_recorder.trajectory_weighting === restored_linear_weighting
    energy_options = map(p.scale_options) do L
        DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))
    end
    kinetic_energy = energy_options[p.active_transformation]
    transformed_adaptation = can_reuse_linear ?
        saved_linear_recorder.adaptation : WeightedScaleAdaptation(p.dimension)
    AWMState(
        lpdf, n_draws, stepsize_adaptation_limit, variance_cond_target,
        restored_linear_source, restored_linear_weighting,
        linear_metric_fallback, nonlinear_adapt,
        monitor_ess, p.recorder.target, (; kwargs...),
        DynamicHMC.NUTS(; max_depth=max_tree_depth),
        DynamicHMC.DualAveraging(δ=target_acceptance_rate), p.dimension,
        progress, time_ns(),
        p.rng, recording_lpdf, nonlinear_recorder,
        p.position, p.scale_options, energy_options, p.active_transformation,
        kinetic_energy, p.variance_memory, zeros(p.dimension),
        p.variance_position, p.variance_gradient,
        transformed_adaptation, p.variance_cond, p.scale_changes,
        get(p, :linear_metric_fallbacks, 0),
        p.position_and_gradient, p.stepsize, p.stepsize_state, p.n_evaluations,
        p.total_evaluation_counter, p.outer_counter, p.current_transition_counter,
        p.total_transition_counter, p.ess, p.steps_per_draw, p.n_divergent, p.n_divergent_samples,
        p.restart, p.n_samples,
        # Carried forward, never read back into the sampler. `get` (not `p.x`) so
        # a checkpoint written before these keys existed still resumes — they are
        # ADDITIVE, which is why `checkpoint_schema_version` does NOT move.
        get(p, :dropped_posterior_position, Matrix{Float64}(undef, p.dimension, 0)),
        get(p, :dropped_posterior_gradient, Matrix{Float64}(undef, p.dimension, 0)),
        get(p, :dropped_n_divergent_samples, 0),
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
  `linear_restart_source=:halo` preserves the existing thinned-halo criterion.
  Opt-in `:nuts_weighted` uses each trajectory's exact NUTS proposal
  distribution, while `:all_good_leaves` weights every eligible noninitial
  leaf equally. `linear_trajectory_weighting=:unit` gives each trajectory or
  eligible leaf unit exposure; `:stepsize` multiplies those weights by the
  trajectory step size. There is no adaptation-phase or validity-fraction
  branch: unit versus step-size weighting is an empirical choice throughout.

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
* a `NamedTuple` — interpret as a pre-built initialization and skip
  Pathfinder entirely. The required keys are `position`, an unconstrained
  vector of target dimension, and `squared_scale`, either a same-length vector
  of diagonal variances or a full target-dimension × target-dimension matrix.
  A `Diagonal` matrix is accepted. Optional extra keys are preserved.

For the multi-chain method, pass either a scalar to broadcast or a
`Vector` of length `length(rngs)` for per-chain initial values. There is
no separate `initial_params` kwarg.

# The multi-chain method and `lpdf`

`adaptive_warmup_mcmc(rngs, lpdf)` gives each chain its own `deepcopy(lpdf)`,
exactly as [`cooperative_warmup_mcmc`](@ref) and [`clustered_warmup_mcmc`](@ref)
do, so chain `i` is byte-identical to running that chain on its own and the
`lpdf` you pass is never mutated. This matters when the lpdf carries adaptation
state — a [`ReparametrizedProblem`](@ref)'s
[`IndexedReparametrization`](@ref) is optimised in place at every window
boundary — and it is why the chains do not have to share one centering.

Pass `lpdfs::AbstractArray` instead (`adaptive_warmup_mcmc(rngs, lpdfs)`) to
control the per-chain problems yourself: independently built problems for
independent adaptation, or `fill(lpdf, n)` to deliberately share one object.

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
* `nonlinear_evidence=:linear_pool` — evidence used to fit the nonlinear
  reparametrization. The default preserves the existing bounded halo fit.
  `:all_good_leaves` streams every noninitial leaf whose log Hamiltonian error
  exceeds `nonlinear_good_leaf_threshold=log(1e-2)`, with equal within-trajectory
  weight. `:nuts_weighted` streams every leaf with its exact marginal NUTS
  proposal probability. Neither streaming mode retains leaves after the current
  trajectory.
* `nonlinear_trajectory_weighting=:unit` — whole-trajectory multiplier for a
  streaming evidence mode, independent of the selected evidence source.
  The only alternative is `:stepsize`; a policy never changes when step-size
  adaptation ends.
  These settings change only the nonlinear fit: selection/application remains
  subordinate to a restart independently requested by the linear criterion.
* `variance_cond_target=2.0` — restart threshold on the marginal-scale
  condition number.
* `linear_restart_source=:halo` — source for that condition number. The
  running alternatives above replace (rather than supplement) the halo-only
  criterion and retain `sum(w)` plus `sum(w^2)` for general weighted moments.
* `linear_trajectory_weighting=:unit` — trajectory exposure for a running
  source; set to `:stepsize` to use step-size exposure. `:halo` requires
  `:unit`, because it has no trajectory weights to modify.
* `linear_metric_fallback=true` — for an opt-in running estimator, if a restart
  keeps the diagonal family and nonlinear parametrization unchanged, apply the
  running transformed marginal correction to that diagonal metric. Other
  families continue to use their ordinary halo fit.
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
    # The upper limit of (intermediate) positions and gradients that will be recorded and then used for adaptation.
    # `nothing` means "unset": 1000 for a fresh run, and on `resume=true` it
    # inherits the checkpoint's value (which cannot be changed — see `restore_state`).
    recording_target=nothing,
    # The maximum number of transitions (per window) for which the stepsize gets adapted
    stepsize_adaptation_limit=50,
    target_acceptance_rate=.8,
    max_tree_depth=10,
    init=missing,
    progress=nothing,
    description="MCMC",
    monitor_ess=!isnothing(progress),
    nonlinear_adapt=true,
    # Nonlinear evidence source. `nothing` means `:linear_pool` for a fresh run
    # and inherits the persisted source on resume.
    nonlinear_evidence=nothing,
    # `nothing` means `:unit` for a fresh run and inherits the persisted policy
    # on resume.
    nonlinear_trajectory_weighting=nothing,
    nonlinear_good_leaf_threshold=nothing,
    variance_cond_target=2.,
    linear_restart_source=nothing,
    linear_trajectory_weighting=nothing,
    linear_metric_fallback=true,
    # Observational checkpoint callback `(state, stage) -> should_stop`; see
    # `_fire_callback`. Default `nothing` keeps the run byte-identical.
    callback=nothing,
    # Opt-in on-disk checkpointing: a directory to write resumable state into at
    # each checkpoint. Default `nothing` writes nothing and keeps the run
    # byte-identical (the write is a pure read + file I/O).
    checkpoint_dir=nothing,
    # Continue from `checkpoint_dir`'s `cp_latest.jls` instead of initializing.
    # Config comes from THIS call, so resuming with a larger `n_draws` keeps
    # sampling rather than re-running a fixed-length batch.
    resume=false,
    # Discard whatever `checkpoint_dir` already holds and start fresh.
    overwrite=false,
    # Keywords forwarded verbatim to the Pathfinder initializer. This is the
    # escape hatch that lets a bare unrecognised keyword be an ERROR (see
    # `_check_kwargs`) without closing off Pathfinder's open kwarg surface.
    pathfinder_kw=(;),
    kwargs...
    # For monitoring purposes: Displays the progress and additional info
) = begin
    _check_kwargs(:adaptive_warmup_mcmc, kwargs)
    resumed = resolve_checkpoint_dir(checkpoint_dir, resume, overwrite)
    with_progress(progress, n_draws+stepsize_adaptation_limit; description) do progress
        # On resume, do NOT re-fire the init callback or rewrite CP-0: the
        # original run already did both at that boundary.
        state, stop = if isnothing(resumed)
            s = init_state(
                rng, lpdf, progress;
                n_draws, n_evaluations, recording_target=something(recording_target, 1000),
                stepsize_adaptation_limit,
                target_acceptance_rate, max_tree_depth, init, monitor_ess,
                nonlinear_adapt,
                nonlinear_evidence=something(nonlinear_evidence, :linear_pool),
                nonlinear_trajectory_weighting=something(
                    nonlinear_trajectory_weighting, :unit,
                ),
                nonlinear_good_leaf_threshold=something(
                    nonlinear_good_leaf_threshold, log(1e-2),
                ),
                variance_cond_target,
                linear_restart_source=something(linear_restart_source, :halo),
                linear_trajectory_weighting=something(linear_trajectory_weighting, :unit),
                linear_metric_fallback,
                kwargs..., pathfinder_kw...
            )
            _write_checkpoint(checkpoint_dir, s, :init)                    # CP-0
            s, _fire_callback(callback, s, :init)
        else
            restore_state(
                resumed, lpdf, progress;
                n_draws, stepsize_adaptation_limit, variance_cond_target,
                linear_restart_source, linear_trajectory_weighting,
                linear_metric_fallback,
                nonlinear_adapt, monitor_ess, target_acceptance_rate,
                max_tree_depth, recording_target, nonlinear_evidence,
                nonlinear_trajectory_weighting, nonlinear_good_leaf_threshold,
                kwargs...
            ), false
        end
        while !stop && size(state.recording_lpdf.posterior_position, 2) < state.n_draws
            run_outer_iteration!(state)
            _write_checkpoint(checkpoint_dir, state, :window)             # CP-N
            stop = _fire_callback(callback, state, :window)
        end
        finalize_warmup!(state)
    end
end

"""
    resume_warmup_mcmc(lpdf, checkpoint_path; progress=nothing, description="MCMC",
                       callback=nothing, checkpoint_dir=nothing)

!!! warning "Deprecated"
    Resuming is no longer a separate function. Point the sampler itself at the
    directory instead:

    ```julia
    adaptive_warmup_mcmc(rng, lpdf; checkpoint_dir=dir, resume=true)
    ```

    That form takes its config from the CALL, so resuming with a larger `n_draws`
    keeps sampling rather than re-running a fixed-length batch. This function
    remains for existing callers and is implemented on the same machinery.

Resume [`adaptive_warmup_mcmc`](@ref) from an on-disk checkpoint written by the
`checkpoint_dir` kwarg. `lpdf` is re-supplied by the caller (the checkpoint does
not store the — possibly non-serializable — inner problem; only the
reparametrizer's scalar state is restored onto it). The run continues from the
saved boundary.

Config kwargs (`n_draws`, `stepsize_adaptation_limit`, …) are accepted here too
and default to the same values as `adaptive_warmup_mcmc` — they are NO LONGER
read from the checkpoint, so a resume that relied on the payload carrying the
original run's config must now pass it explicitly.
"""
resume_warmup_mcmc(lpdf, checkpoint_path;
    progress=nothing, description="MCMC", callback=nothing, checkpoint_dir=nothing,
    n_draws=1000, stepsize_adaptation_limit=50, kwargs...
) = begin
    payload = deserialize(checkpoint_path)
    with_progress(progress, n_draws+stepsize_adaptation_limit; description) do progress
        state = restore_state(payload, lpdf, progress;
            n_draws, stepsize_adaptation_limit, monitor_ess=!isnothing(progress), kwargs...)
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

# Scalar-lpdf multi-chain entry point: each chain gets its OWN `deepcopy(lpdf)`,
# matching `cooperative_warmup_mcmc` and `clustered_warmup_mcmc`.
#
# This used to be a `fill`, which stores the SAME object in every slot. A stateful
# lpdf was then shared by every chain — notably a `ReparametrizedProblem`, whose
# `IndexedReparametrization` is mutated IN PLACE (`optimize!`: `pairs .= ...`) at
# every window boundary while `logdensity`/`logdensity_and_gradient` read those same
# pairs on the gradient hot path. Under the default `parallel=true` that is a data
# race; even single-threaded, chain i>1 sampled under whatever centering chain i-1
# last wrote, and every chain's `reparam_sources` checkpoint recorded the same
# last-writer values. Both were silent.
#
# Callers that genuinely want the chains to share one object can still say so
# explicitly by passing `fill(lpdf, n)` to the `lpdfs::AbstractArray` method.
adaptive_warmup_mcmc(rngs::AbstractArray, lpdf; kwargs...) =
    adaptive_warmup_mcmc(rngs, map(_ -> deepcopy(lpdf), rngs); kwargs...)
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
