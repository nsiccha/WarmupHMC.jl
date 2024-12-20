"""
Adaptively learn 
* a (currently only) linear transformation of the posterior that simplifies MCMC sampling,
* a NUTS step size (using standard Dual Averaging)
and return samples from the posterior. 

The warm-up procedure is windowed and somehow insipred by [Stan](https://mc-stan.org/docs/reference-manual/mcmc.html#automatic-parameter-tuning)'s and [nutpie](https://github.com/pymc-devs/nutpie)'s warm-up procedures, but differs in several important ways:

* We initialize using Pathfinder!
* Our warm-up windows aim to reach a certain number of GRADIENT EVALUATIONS , instead of a certain number of MCMC transitions (Stan).
We start with a (default) target of 1000 gradient evaluations, and double that target after each warm-up window.
* Instead of only using the posterior positions (Stan), we use the posterior POSITIONS AND GRADIENTS (like e.g. nutpie).
* Instead of only using the MCMC/POSTERIOR positions and gradients (Stan and nutpie), 
we also store and use the INTERMEDIATE positions and gradients, i.e. the ones that MCMC visits before returning the "final" new position. 
We store up to (a default of) 1000 intermediate positions and gradients. 
The stored intermediate positions get selected (pseudo-)randomly, and only get selected if the Hamiltonian error is small enough.
* Instead of only learning a single (linear) transformation and upating that one repeatedly, we learn several transformations in parallel and
at the end of each warm-up window select the one that minimizes a loss function. Currently, we learn three different linear transformations:
    * Pathfinder's initial transformation, enriched by an updated additional diagonal scaling
    * A standard diagonal "mass matrix".
    * A novel, adaptive sequence of Householder transformations followed by diagonal scaling.
The loss function that gets used to select the used linear transformation tries to reward transformations which turn the transformed posterior
into something that's close to a Normal distribution without correlation. 
For transformed positions p' and gradients g', the loss function is 
    loss(p', g') = sum(log(std(p') * std(g'))^2), 
which is zero for Normal distributions with zero correlations, independently of the samples positions and gradients. 
* Instead of running the warm-up for a fixed number of windows, 
we try to estimate when continuing warming up is harmful/useless and stop warming up then. To facilitate this, we
    * only ever adapt the stepsize until (a default of) 50 MCMC transitions have ocurred in the current warm-up window,
    * and try to predict cost (in terms of gradient evaluations) for finishing sampling with the current kernel vs. restarting warm-up with a new window.
    The way we do this prediction will probably be changed in the future.
By only ever having 50 stepsize adaptation MCMC transitions, we can start collecting posterio samples early.
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
    stepsize_search=DynamicHMC.InitialStepsizeSearch(),
    init=missing, 
    report=true, 
    monitor_ess=report,
    ) = begin 
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
    recording_lpdf = RecordingPosterior2(lpdf; recorder)
    # Use Stan's initialization procedure if no initial position is given
    ismissing(init) && (init = rand(rng, Uniform(-2,+2), dimension))
    # Run pathfinder once to get a linear transformation and a better initial position
    pathfinder_result = pathfinder(lpdf; rng, ndraws=1, init)
    # The better initial position
    position = collect(pathfinder_result.draws[:, 1])::Vector{Float64}
    # Pathfinder's linear transformation
    pathfinder_transformation = factorize(
        pathfinder_result.fit_distribution.Σ).L::Transpose{Float64, Pathfinder.WoodburyPDRightFactor{Float64, Diagonal{Float64, Vector{Float64}},
        LinearAlgebra.QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}, UpperTriangular{Float64, Matrix{Float64}}}
    }
    # We currently learn three linear transformation options
    scale_options = (
        # Corresponds to a standard diagonal mass matrix
        Diagonal(sqrt.(diag(pathfinder_result.fit_distribution.Σ))::Vector{Float64}),
        # Corresponds to Pathfinder's linear transformation with an added diagonal scaling term that can be updated
        MatrixFactorization(pathfinder_transformation, Diagonal(ones(dimension))),
        # Something new. Corresponds to a sequence of Householder reflections, followed by a diagonal scaling term. 
        # Both the reflections and the diagonal scaling term will be updated. 
        MatrixFactorization(SuccessiveReflections(dimension), Diagonal(ones(dimension)))
    )
    # This is needed to make DynamicHMC "accept" our linear transformations
    energy_options = map(scale_options) do L
        DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))
    end
    # At the beginning, we will use Pathfinder's transformation.
    # At later stages of warm-up, the estimated losses corresponding to each transformation will be written to `transformation_losses` 
    transformation_losses = (0., 0., 0.)
    active_option = 2 # Pathfinder
    kinetic_energy = energy_options[active_option]
    # The below tries to find a good initial stepsize. 
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    position_gradient_and_momentum = DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy))
    stepsize = DynamicHMC.find_initial_stepsize(
        stepsize_search, 
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf), position_gradient_and_momentum
        )
    )
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
    depth_predictor = DepthPredictor(max_tree_depth)
    # For monitoring purposes: Keep track of the number of gradient evaluations during warm-up
    total_evaluation_counter = 0
    # For monitoring purposes: Keep track of the number of warm-up windows so far
    outer_counter = 0
    # For monitoring purposes: Keep track of the number of the total number of MCMC transitions
    total_transition_counter = 0
    # For monitoring purposes: Displays the progress and additional info
    progress = report ? ProgressMeter.Progress(n_draws; dt=1e-3, desc="Sampling...") : nothing
    # For monitoring purposes: Keep track of the minimal effective sample size so far
    min_ess = missing
    # For monitoring purposes: Keep track of the current number of gradient evaluations per MCMC transition
    steps_per_draw = missing
    # For monitoring purposes: Keep track of the number of divergences in the current WARM-UP window
    n_divergent = missing
    # For monitoring purposes: Keep track of the number of divergences in the current SAMPLING window
    n_divergent_samples = missing
    # We run the warm-up procedure until we have collected enough samples
    while size(posterior_position, 2) < n_draws
        # Some setup that has to happen at the beginning of every warm-up window
        outer_counter += 1
        hamiltonian = DynamicHMC.Hamiltonian(kinetic_energy, recording_lpdf)
        stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
        stepsize = DynamicHMC.current_ϵ(stepsize_state)
        # Reset the so far recorded intermediate and MCMC positions and gradients
        reset!(recording_lpdf)
        current_transition_counter = 0
        steps_per_draw = current_steps(depth_predictor)
        finish_cost = restart_cost = (n_draws + stepsize_adaptation_limit) * steps_per_draw
        n_divergent = 0
        n_divergent_samples = 0
        current_evaluation_counter = 0
        # We run the current warm-up/sampling window until 
        #   a) we have collected enough samples and can break out of the outer loop as well or
        #   b) we have reached the current targeted number of gradient evaluations AND we estimate that 
        #       restarting (adding a new warm-up window) is better than finishing sampling with the current adaptation
        while size(posterior_position, 2) < n_draws && (
            current_evaluation_counter < n_evaluations || finish_cost < restart_cost 
        )
            current_transition_counter += 1
            total_transition_counter += 1
            # One MCMC transition
            position_and_gradient, stats = DynamicHMC.sample_tree(rng, algorithm, hamiltonian, position_and_gradient, stepsize)
            total_evaluation_counter += stats.steps
            current_evaluation_counter += stats.steps
            # Add information about the current transition to our predictor
            record!(depth_predictor, stats)
            is_divergent = DynamicHMC.is_divergent(stats.termination)
            is_divergent && (n_divergent += 1)
            if current_transition_counter < stepsize_adaptation_limit
                # The current warm-up window has seen fewer MCMC transitions than our step size adaptation limit.
                # Continue adapting the step size.
                stepsize_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, stepsize_state, stats.acceptance_rate)
                stepsize = DynamicHMC.current_ϵ(stepsize_state)
            elseif current_transition_counter == stepsize_adaptation_limit
                # The current warm-up window hits the step size adaptation limit.
                # Finalize the stepsize.
                stepsize = DynamicHMC.final_ϵ(stepsize_state)
            else
                # The current warm-up window has been sampling with the same linear transformation and step size.
                # Record posterior positions, gradients and whether the current transition diverged
                append!(posterior_position, position_and_gradient.q)
                append!(posterior_gradient, position_and_gradient.∇ℓq)
                is_divergent && (n_divergent_samples += 1)
            end
            # Update the current steps per draw and predicted steps per draw should we add a new warm-up window,
            # as well as the estimated cost for finishing sampling with the current kernel or restarting warm-up. 
            # I think this actually only has to happen once, will change at some point in the future.
            steps_per_draw = current_steps(depth_predictor)
            potential_steps_per_draw = potential_steps(depth_predictor)
            finish_cost = (n_draws + stepsize_adaptation_limit - current_transition_counter) * steps_per_draw
            restart_cost = (n_draws + stepsize_adaptation_limit) * potential_steps_per_draw
            report && ProgressMeter.update!(progress, size(posterior_position, 2), showvalues=pairs(
                merge(
                    round2((;outer_counter, current_transition_counter, total_transition_counter, transformation_losses, finish_cost, restart_cost, total_evaluation_counter, steps_per_draw, potential_steps_per_draw, stepsize, n_divergent, n_divergent_samples, min_ess, recorder=(;recorder.target, recorder.thin, recorder.outer_count, recorder.inner_count, recorder.triggered, recorded=size(halo_position, 2)))),
                )
            ))
        end
        advance!(depth_predictor)
        # Double the targeted number of GRADIENT EVALUATIONS in the next warm-up window
        n_evaluations *= 2
        # Recompute the thinning factor for the intermediate positions and gradients
        recorder.thin = n_evaluations ÷ recording_target
        # Update the linear transformation candidates and estimate the transformation loss,
        # using the INTERMEDIATE POSITIONS AND GRADIENTS.
        # AT THIS POINT WE CAN ALSO LEARN NON-LINEAR TRANSFORMATIONS, 
        # AND AGAIN SELECT THE ONE THAT MINIMIZES THE TRANSFORMATION LOSS.
        transformation_losses = map(L->update_loss!(L, (halo_position), (halo_gradient)), scale_options)
        # Update the new linear transformation to be the one with the minimal estimated transformation loss.
        active_option = argmin(transformation_losses)
        kinetic_energy = energy_options[active_option]
        (monitor_ess && size(posterior_position, 2) > 10) && (min_ess = minimum(
            MCMCDiagnosticTools.ess(reshape(posterior_position', (:, 1, dimension)))
        ))
    end
    report && ProgressMeter.update!(progress, size(posterior_position, 2), showvalues=pairs(
        merge(
            round2((;outer_counter, total_transition_counter, transformation_losses, total_evaluation_counter, steps_per_draw, stepsize, n_divergent_samples, min_ess)),
        )
    ))
    recording_lpdf
end