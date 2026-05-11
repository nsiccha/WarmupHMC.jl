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
* Uses POSITIONS AND GRADIENTS (like nutpie), plus the
  INTERMEDIATE POSITIONS AND GRADIENTS visited during NUTS tree
  traversal (selected pseudo-randomly, only if the Hamiltonian error is
  small enough). Up to `recording_target` intermediate states are kept.
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
* `recording_target=1000` — maximum number of intermediate
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
    kwargs...
    # For monitoring purposes: Displays the progress and additional info
) = with_progress(progress, n_draws+stepsize_adaptation_limit; description) do progress
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
        # As above
        recording_target,
        # The initial "thinning" of intermediate positions and gradients 
        n_evaluations ÷ recording_target, 
    )
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    # Use Stan's initialization procedure if no initial position is given
    # ismissing(init) && (init = rand(rng, Uniform(-2,+2), dimension))
    # Work around https://github.com/roualdes/bridgestan/issues/272
    # LogDensityProblems.logdensity_and_gradient(recording_lpdf, init)
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
    # At later stages of warm-up, the estimated losses corresponding to each transformation will be written to `transformation_losses` 
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
    # stepsize = with_progress(progress; description="Initial stepsize", transient=true) do _
    stepsize = DynamicHMC.find_initial_stepsize(
        stepsize_search, 
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf), position_gradient_and_momentum
        )
    )
    # end
    # The below variables give us access to the matrices into which the intermediate positions and gradients and the MCMC positions an gradients will be written
    (;
        # Intermediate positions
        halo_position, 
        # Intermediate gradients
        halo_gradient, 
        # MCMC positions
        posterior_position,
        # MCMC gradients
        posterior_gradient
    ) = recording_lpdf
    # This will try to predict the future number of steps needed, if we adopt the new linear transformation.
    # It's currently doing something slightly silly, and I will change what exactly it does. 
    # depth_predictor = DepthPredictor(max_tree_depth)
    # For monitoring purposes: Keep track of the number of gradient evaluations during warm-up
    total_evaluation_counter = 0
    # For monitoring purposes: Keep track of the number of warm-up windows so far
    outer_counter = 0
    # For monitoring purposes: Keep track of the number of the total number of MCMC transitions
    current_transition_counter = 0
    total_transition_counter = 0
    # progress = report ? ProgressMeter.Progress(n_draws; dt=1e-3, desc="Sampling...") : nothing
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
    while size(posterior_position, 2) < n_draws
        # Some setup that has to happen at the beginning of every warm-up window
        outer_counter += 1
        hamiltonian = DynamicHMC.Hamiltonian(kinetic_energy, recording_lpdf)
        current_evaluation_counter = 0
        # We run the current warm-up/sampling window until 
        #   a) we have collected enough samples and can break out of the outer loop as well or
        #   b) we have reached the current targeted number of gradient evaluations AND we estimate that 
        #       restarting (adding a new warm-up window) is better than finishing sampling with the current adaptation
        while size(posterior_position, 2) < n_draws && (current_evaluation_counter < n_evaluations)
            current_transition_counter += 1
            total_transition_counter += 1
            # One MCMC transition
            position_and_gradient, stats = DynamicHMC.sample_tree(rng, algorithm, hamiltonian, position_and_gradient, stepsize)
            total_evaluation_counter += stats.steps
            current_evaluation_counter += stats.steps
            OnlineStatsBase.fit!(steps_per_draw, stats.steps)
            is_divergent = DynamicHMC.is_divergent(stats.termination)
            is_divergent && (n_divergent += 1)
            if current_transition_counter < stepsize_adaptation_limit
                # The current warm-up window has seen fewer MCMC transitions than our step size adaptation limit.
                # Continue adapting the step size.
                stepsize_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, stepsize_state, stats.acceptance_rate)
                stepsize = DynamicHMC.current_ϵ(stepsize_state)
                # nuts_state = merge(nuts_state, (;stepsize))
            elseif current_transition_counter == stepsize_adaptation_limit
                # The current warm-up window hits the step size adaptation limit.
                # Finalize the stepsize.
                stepsize = DynamicHMC.final_ϵ(stepsize_state)
                # nuts_state = merge(nuts_state, (;stepsize))
            else
                # The current warm-up window has been sampling with the same linear transformation and step size.
                # Record posterior positions, gradients and whether the current transition diverged
                append!(posterior_position, position_and_gradient.q)
                append!(posterior_gradient, position_and_gradient.∇ℓq)
                # append!(posterior_position, nuts_state.current.position)
                # append!(posterior_gradient, nuts_state.current.log_density_gradient)
                is_divergent && (n_divergent_samples += 1)
            end
            if current_evaluation_counter >= n_evaluations
                scale = scale_options[active_transformation]
                for (pi, gi) in zip(eachcol(halo_position), eachcol(halo_gradient))
                    ldiv!(variance_memory, scale, pi)
                    OnlineStatsBase.fit!(variance_position, variance_memory)
                    mul!(variance_memory, scale', gi)
                    OnlineStatsBase.fit!(variance_gradient, variance_memory)
                end
                variance_memory .= sqrt.(std.(variance_position.stats) ./ std.(variance_gradient.stats))
                for i in 1:dimension
                    variance_position.stats[i] = OnlineStatsBase.Variance()
                    variance_gradient.stats[i] = OnlineStatsBase.Variance()
                end
                lmin, lmax = extrema(variance_memory)
                variance_cond = lmax / lmin
                pushfirst!(scale_changes, sqrt(variance_cond))
                restart = variance_cond >= variance_cond_target
            end
            n_samples = size(posterior_position, 2)
            update_progress!(progress, current_transition_counter;
                divergent_samples=UncertainFrequency(n_divergent_samples, n_samples),
                active_transformation=ActiveTransformation(kinetic_energy, scale_changes),
                sampling_performance=SamplingPerformance(stepsize, mean(steps_per_draw)),
                total_transition_counter=Speed(total_transition_counter, time_ns()-start_time),
                total_evaluation_counter=Speed(total_evaluation_counter, time_ns()-start_time),
            )
        end
        if monitor_ess && n_samples > 10
            ess .= sort!(MCMCDiagnosticTools.ess(reshape(posterior_position', (:, 1, dimension))))
            update_progress!(progress, nothing;
                ess=short_string(ess) * " from $n_samples samples.",
            )
        end
        n_samples < n_draws || continue
        # Double the targeted number of GRADIENT EVALUATIONS in the next warm-up window
        n_evaluations *= 2
        # Recompute the thinning factor for the intermediate positions and gradients
        recorder.thin = n_evaluations ÷ recording_target
        restart || continue
        stepsize = DynamicHMC.final_ϵ(stepsize_state)
        stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
        stepsize = DynamicHMC.current_ϵ(stepsize_state)
        # Reset the so far recorded intermediate and MCMC positions and gradients
        current_transition_counter = 0
        steps_per_draw = OnlineStatsBase.Mean()
        n_divergent = 0
        n_divergent_samples = 0
        # Update the linear transformation candidates and estimate the transformation loss,
        # using the INTERMEDIATE POSITIONS AND GRADIENTS.
        nonlinear_adapt && (position_and_gradient = find_reparametrization!(lpdf, halo_position, halo_gradient, position_and_gradient))
        # Update the new linear transformation to be the one with the minimal estimated transformation loss.
        active_transformation = argmin(
            map(L->update_loss!(L, (halo_position), (halo_gradient); kwargs...), scale_options)
        )
        kinetic_energy = energy_options[active_transformation]
        update_progress!(progress, nothing;
            active_transformation=ActiveTransformation(kinetic_energy, scale_changes),
        )
        reset!(recording_lpdf)
    end
    update_progress!(progress, (monitor_ess ? "min. ESS: $(short_string(ess[1])), " : "") * "divergent: $(short_string(100*n_divergent_samples/n_samples))%")
    nonlinear_adapt && reparametrize!(lpdf, posterior_position)
    (;initial_position=position, halo_position, halo_gradient, posterior_position, posterior_gradient, ess, scale_options, active_transformation, stepsize, total_evaluation_counter, n_divergent_samples, position_and_gradient, scale_changes)
end
ensurevector(x, n) = Fill(x, n)
ensurevector(x::AbstractVector, n) = begin 
    @assert length(x) == n
    x
end
adaptive_warmup_mcmc(rngs::AbstractArray, lpdf; kwargs...) = adaptive_warmup_mcmc(rngs, fill(lpdf, size(rngs)); kwargs...) 
adaptive_warmup_mcmc(rngs::AbstractArray, lpdfs::AbstractArray; parallel=true, progress=nothing, 
monitor_ess=!isnothing(progress), description="MCMC", init=missing, kwargs...) = with_progress(progress, length(rngs); description) do progress 
    n_chains = length(rngs)
    rv = Vector{Any}(missing, n_chains)
    init = ensurevector(init, n_chains)
    if parallel
        Threads.@threads for i in 1:n_chains
            rv[i] = adaptive_warmup_mcmc(rngs[i], lpdfs[i]; progress, monitor_ess, description=description*".$i", init=init[i], kwargs...)
            update_progress!(progress)
        end
    else
        for i in 1:n_chains
            rv[i] = adaptive_warmup_mcmc(rngs[i], lpdfs[i]; progress, monitor_ess, description=description*".$i", init=init[i], kwargs...)
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
