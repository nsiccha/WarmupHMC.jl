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
        nonlinear_adapt, monitor_ess, n_draws, scale_options, energy_options,
        NamedTuple(kwargs), start_time,
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

    chain.n_evaluations *= 2
    recording_lpdf.recorder.thin = chain.n_evaluations ÷ recording_target
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
"""
run_chain!(chain::CooperativeChain) = begin
    while advance_window!(chain) != :done
    end
    chain.nonlinear_adapt && reparametrize!(chain.lpdf, chain_draws(chain))
    chain
end
