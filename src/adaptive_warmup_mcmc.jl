round2(x::Real) = round(x; sigdigits=2)
round2(x::Integer) = round(x)
round2(x::Union{Tuple,NamedTuple,AbstractArray}) = map(round2, x)
round2(x::Missing) = x

grad_cov_ev(p, g) = eigen(Symmetric(cov(g')), size(g,1):size(g,1)).vectors[:, 1]
update_loss!(t::SuccessiveReflections, p, g; threshold=log(2), v_f=grad_cov_ev, idx_f=v->argmax(v.^2)) = begin 
    (;idxs, reflections, s1, s2, transformation_losses) = t
    dimension = LinearAlgebra.checksquare(t)
    s1 .= std.(eachrow(p))
    s2 .= std.(eachrow(g))
    transformation_losses .= abs.(log.(s1 .* s2))
    bad_idxs = collect(1:dimension)
    # empty!(splits1)
    empty!(idxs)
    empty!(reflections)
    @views while length(bad_idxs) > 0
        filter!(i->transformation_losses[i]>=threshold, bad_idxs)
        length(bad_idxs) == 0 && break

        bad_p = p[bad_idxs, :]
        bad_g = g[bad_idxs, :]
        v = v_f(bad_p, bad_g)
        l = abs(log(std(v' * bad_p) * std(v' * bad_g)))
        l > threshold && break

        push!(idxs, copy(bad_idxs))
        v[idx_f(v)] -= 1
        normalize!(v)
        push!(reflections, v)
        vp = v' * bad_p
        vg = v' * bad_g
        bad_p .-= 2 .* v * vp 
        bad_g .-= 2 .* v * vg 
        s1[bad_idxs] .= std.(eachrow(bad_p))
        s2[bad_idxs] .= std.(eachrow(bad_g))
        transformation_losses[bad_idxs] .= abs.(log.(s1[bad_idxs] .* s2[bad_idxs]))
    end
    t
end
update_loss!(t::Diagonal, p, g) = sum(1:LinearAlgebra.checksquare(t)) do i 
    pi, gi = view(p, i, :), view(g, i, :)
    s1, s2 = std(pi), std(gi)
    t[i,i] = sqrt(s1 / s2)
    log(s1 * s2) ^ 2
end
update_loss!(t::MatrixFactorization, p, g) = update_loss!(t.m2, t.m1 \ p, t.m1' * g)
update_loss!(t::ScaleThenReflect, p, g; kwargs...) = begin
    update_loss!(t.m1, p, g; kwargs...)
    t.m2.diag .= sqrt.(t.m1.s1 ./ t.m1.s2)
    sum(log.(t.m1.s1 .* t.m1.s2) .^ 2)
end
adaptive_warmup_mcmc(
    rng, lpdf; 
    n_draws=1000, n_evaluations=1000, 
    target_acceptance_rate=.8, max_tree_depth=10,
    stepsize_search=DynamicHMC.InitialStepsizeSearch(),
    stepsize_adaptation_limit=50, init=missing, report=true, monitor_ess=report,
    recording_target=1000
    ) = begin 

    stepsize_adaptation = DynamicHMC.DualAveraging(δ=target_acceptance_rate)
    algorithm = DynamicHMC.NUTS(;max_depth=max_tree_depth)
    dimension = LogDensityProblems.dimension(lpdf)
    recorder = LimitedRecorder2(recording_target, n_evaluations ÷ recording_target, 1, 0, false, false)
    recording_lpdf = RecordingPosterior2(lpdf; recorder)
    ismissing(init) && (init = rand(rng, Uniform(-2,+2), dimension))
    pathfinder_result = pathfinder(lpdf; rng, ndraws=1, init)
    position = collect(pathfinder_result.draws[:, 1])::Vector{Float64}
    pathfinder_transformation = factorize(pathfinder_result.fit_distribution.Σ).L::Transpose{Float64, Pathfinder.WoodburyPDRightFactor{Float64, Diagonal{Float64, Vector{Float64}}, LinearAlgebra.QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}, UpperTriangular{Float64, Matrix{Float64}}}}
    scale_options = (
        Diagonal(sqrt.(diag(pathfinder_result.fit_distribution.Σ))::Vector{Float64}),
        MatrixFactorization(pathfinder_transformation, Diagonal(ones(dimension))),
        MatrixFactorization(SuccessiveReflections(dimension), Diagonal(ones(dimension)))
    )
    energy_options = map(scale_options) do L
        DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))
    end
    transformation_losses = (0., 0., 0.)
    active_option = 2
    kinetic_energy = energy_options[active_option]
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    position_gradient_and_momentum = DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy))
    stepsize = DynamicHMC.find_initial_stepsize(
        stepsize_search, 
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf), position_gradient_and_momentum
        )
    )
    (;halo_position, halo_gradient, posterior_position, posterior_gradient) = recording_lpdf
    depth_predictor = DepthPredictor(max_tree_depth)
    total_evaluation_counter = 0
    outer_counter = 0
    total_transition_counter = 0
    progress = report ? ProgressMeter.Progress(n_draws; dt=1e-3, desc="Sampling...") : nothing
    min_ess = missing
    steps_per_draw = missing
    n_divergent = missing
    n_divergent_samples = missing
    while size(posterior_position, 2) < n_draws
        outer_counter += 1
        hamiltonian = DynamicHMC.Hamiltonian(kinetic_energy, recording_lpdf)
        stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
        stepsize = DynamicHMC.current_ϵ(stepsize_state)
        reset!(recording_lpdf)
        current_transition_counter = 0
        steps_per_draw = current_steps(depth_predictor)
        finish_cost = restart_cost = (n_draws + stepsize_adaptation_limit) * steps_per_draw
        n_divergent = 0
        n_divergent_samples = 0
        current_evaluation_counter = 0
        while size(posterior_position, 2) < n_draws && (
            current_evaluation_counter < n_evaluations || finish_cost < restart_cost 
        )
            current_transition_counter += 1
            total_transition_counter += 1
            position_and_gradient, stats = DynamicHMC.sample_tree(rng, algorithm, hamiltonian, position_and_gradient, stepsize)
            total_evaluation_counter += stats.steps
            current_evaluation_counter += stats.steps
            @assert total_evaluation_counter >= size(halo_position, 2) println(total_evaluation_counter, "<", size(halo_position, 2))
            record!(depth_predictor, stats)
            is_divergent = DynamicHMC.is_divergent(stats.termination)
            is_divergent && (n_divergent += 1)
            if current_transition_counter < stepsize_adaptation_limit
                stepsize_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, stepsize_state, stats.acceptance_rate)
                stepsize = DynamicHMC.current_ϵ(stepsize_state)
            elseif current_transition_counter == stepsize_adaptation_limit
                stepsize = DynamicHMC.final_ϵ(stepsize_state)
                map(reset!, (posterior_position, posterior_gradient))
            else
                append!(posterior_position, position_and_gradient.q)
                append!(posterior_gradient, position_and_gradient.∇ℓq)
                is_divergent && (n_divergent_samples += 1)
            end
            steps_per_draw = current_steps(depth_predictor)
            potential_steps_per_draw = potential_steps(depth_predictor)#(8 + steps_per_draw) / 2
            finish_cost = (n_draws + stepsize_adaptation_limit - current_transition_counter) * steps_per_draw
            restart_cost = (n_draws + stepsize_adaptation_limit) * potential_steps_per_draw
            report && ProgressMeter.update!(progress, size(posterior_position, 2), showvalues=pairs(
                merge(
                    round2((;outer_counter, current_transition_counter, total_transition_counter, transformation_losses, finish_cost, restart_cost, total_evaluation_counter, steps_per_draw, potential_steps_per_draw, stepsize, n_divergent, n_divergent_samples, min_ess, recorder=(;recorder.target, recorder.thin, recorder.outer_count, recorder.inner_count, recorder.triggered, recorded=size(halo_position, 2)))),
                )
            ))
        end
        # display((;recorder.target, recorder.thin, recorder.outer_count, recorder.inner_count, recorder.triggered, recorded=size(halo_position, 2)))
        # depth_count .*= length(depth_count) / sum(depth_count) 
        advance!(depth_predictor)
        n_evaluations *= 2
        recorder.thin = n_evaluations ÷ recording_target
        transformation_losses = map(L->update_loss!(L, (halo_position), (halo_gradient)), scale_options)
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