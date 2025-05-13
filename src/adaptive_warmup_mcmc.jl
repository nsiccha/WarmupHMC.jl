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
pathfinder_callback(progress) = nothing
initialize_mcmc(lpdf, init::AbstractVector; rng, progress, kwargs...) = with_progress(progress, 1_000; description="Pathfinder", transient=true) do pprogress
    # Work around https://github.com/roualdes/bridgestan/issues/272
    LogDensityProblems.logdensity_and_gradient(lpdf, init)
    initialize_mcmc(
        lpdf, 
        mypathfinder(lpdf; rng, init, callback=pathfinder_callback(pprogress), kwargs...);
        kwargs...
    )
end
initialize_mcmc(lpdf, init::PathfinderResult; kwargs...) = begin 
    position = collect(init.draws[:, 1])::Vector{Float64}
    squared_scale = init.fit_distribution.Σ
    initialize_mcmc(lpdf, (;position, squared_scale))
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
Adaptively 

* learn a (currently only) linear transformation of the posterior that simplifies MCMC sampling,
* learn a NUTS step size (using standard Dual Averaging), and
* return samples from the posterior. 

The warm-up procedure is windowed and insipred by [Stan](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)'s and [nutpie](https://github.com/pymc-devs/nutpie)'s warm-up procedures, but differs in several important ways:

* We initialize using Pathfinder!
* Our warm-up windows aim to reach a certain number of GRADIENT EVALUATIONS , instead of a certain number of MCMC transitions (Stan).
We start with a (default) target of 1000 gradient evaluations, and double that target after each warm-up window.
* Instead of only using the posterior positions (Stan), we use the posterior POSITIONS AND GRADIENTS (like e.g. nutpie).
* Instead of only using the MCMC/posterior positions and gradients (Stan and nutpie), 
we also store and use the INTERMEDIATE POSITIONS AND GRADIENTS, i.e. the ones that MCMC visits before returning the "final" new position. 
We store up to (a default of) 1000 intermediate positions and gradients. 
The stored intermediate positions get selected (pseudo-)randomly, and only get selected if the Hamiltonian error is small enough.
* Instead of only learning a single (linear) transformation and upating that one repeatedly, we learn several transformations in parallel and
at the end of each warm-up window select the one that minimizes a loss function. Currently, we learn three different linear transformations:
    * Pathfinder's initial transformation, enriched by an updated additional diagonal scaling,
    * A standard diagonal "mass matrix".
    * A novel, adaptive sequence of Householder transformations followed by diagonal scaling.

  The loss function that gets used to select the used linear transformation tries to reward transformations which turn the transformed posterior
into something that's close to a Normal distribution without correlation. 
For transformed positions p' and gradients g', the loss function is 
    `loss(p', g') = sum((std(p') * std(g')))`, 
which is zero for Normal distributions with zero correlations, independently of the sampled positions and gradients. 
* Instead of running the warm-up for a fixed number of windows, 
we try to estimate when continuing warming up is harmful/useless and stop warming up then. To facilitate this, we
    * only ever adapt the stepsize until (a default of) 50 MCMC transitions have ocurred in the current warm-up window,
    * and try to predict cost (in terms of gradient evaluations) for finishing sampling with the current kernel vs. restarting warm-up with a new window.
    The way we do this prediction will probably be changed in the future.

  By only ever having 50 stepsize adaptation MCMC transitions, we can start collecting posterior samples early.
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
    finish = false
    # We run the warm-up procedure until we have collected enough samples

    n_samples = 0
    update_progress!(progress, current_transition_counter;
        divergent_samples=UncertainFrequency(n_divergent_samples, n_samples),
        (monitor_ess ? (;ess="pending...") : (;))...,
        total_transition_counter,
        total_evaluation_counter,
        sampling_performance=SamplingPerformance(stepsize, mean(steps_per_draw)),
        active_transformation=ActiveTransformation(kinetic_energy, scale_changes),
    )
    while size(posterior_position, 2) < n_draws
        # Some setup that has to happen at the beginning of every warm-up window
        outer_counter += 1
        hamiltonian = DynamicHMC.Hamiltonian(kinetic_energy, recording_lpdf)
        stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
        stepsize = DynamicHMC.current_ϵ(stepsize_state)
        # Reset the so far recorded intermediate and MCMC positions and gradients
        reset!(recording_lpdf)
        current_transition_counter = 0
        steps_per_draw = OnlineStatsBase.Mean()
        n_divergent = 0
        n_divergent_samples = 0
        current_evaluation_counter = 0
        # We run the current warm-up/sampling window until 
        #   a) we have collected enough samples and can break out of the outer loop as well or
        #   b) we have reached the current targeted number of gradient evaluations AND we estimate that 
        #       restarting (adding a new warm-up window) is better than finishing sampling with the current adaptation
        while size(posterior_position, 2) < n_draws && (current_evaluation_counter < n_evaluations || finish)
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
            if current_evaluation_counter >= n_evaluations && !finish
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
                finish = variance_cond < variance_cond_target
                finish || (stepsize = DynamicHMC.final_ϵ(stepsize_state))
            end
            n_samples = size(posterior_position, 2)
            update_progress!(progress, current_transition_counter;
                total_transition_counter=Speed(total_transition_counter, time_ns()-start_time),
                total_evaluation_counter=Speed(total_evaluation_counter, time_ns()-start_time),
                sampling_performance=SamplingPerformance(stepsize, mean(steps_per_draw)),
                divergent_samples=UncertainFrequency(n_divergent_samples, n_samples),
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
    end
    update_progress!(progress, (monitor_ess ? "min. ESS: $(short_string(ess[1])), " : "") * "divergent: $(short_string(100*n_divergent_samples/n_samples))%")
    reparametrize!(lpdf, posterior_position)#, posterior_gradient)
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