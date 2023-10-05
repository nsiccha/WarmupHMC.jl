module WarmupHMCDynamicHMCExt

using WarmupHMC, DynamicHMC, UnPack, Random

import WarmupHMC: reparametrize, find_reparametrization, mcmc_with_reparametrization, mcmc_keep_reparametrization

import DynamicHMC: default_warmup_stages, default_reporter, NUTS, SamplingLogDensity, _warmup, mcmc, WarmupState, initialize_warmup_state, warmup, InitialStepsizeSearch, TuningNUTS, _empty_posterior_matrix, TreeStatisticsNUTS, Hamiltonian, initial_adaptation_state, make_mcmc_reporter, evaluate_ℓ, current_ϵ, sample_tree, adapt_stepsize, report, REPORT_SIGDIGITS, GaussianKineticEnergy, regularize_M⁻¹, sample_M⁻¹, final_ϵ, mcmc_steps, mcmc_next_step

function mcmc_with_reparametrization(rng, ℓ, N; initialization = (),
    warmup_stages = default_warmup_stages(),
    algorithm = NUTS(), reporter = default_reporter(), kwargs...)
    @unpack final_reparametrization_state, inference = mcmc_keep_reparametrization(
        rng, ℓ, N; initialization = initialization,
        warmup_stages = warmup_stages, algorithm = algorithm,
        reporter = reporter, kwargs...
    )
#    final_warmup_state = final_reparametrization_state.warmup_state
# @unpack κ, ϵ = final_warmup_state
    (; inference..., final_reparametrization_state)
end

function mcmc_keep_reparametrization(rng::AbstractRNG, ℓ, N::Integer;
                          initialization = (),
                          warmup_stages = default_warmup_stages(),
                          algorithm = NUTS(),
                          reporter = default_reporter(), kwargs...)
    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    initial_reparametrization_state = initialize_reparametrization_state(rng, ℓ; initialization...)
    warmup, reparametrization_state = my_warmup(sampling_logdensity, warmup_stages, initial_reparametrization_state; kwargs...)
    inference = mcmc(sampling_logdensity, N, reparametrization_state)
    (; initial_reparametrization_state, warmup, final_reparametrization_state = reparametrization_state, inference, sampling_logdensity)
end

struct ReparametrizationState{R,W<:WarmupState}
    reparametrization::R
    warmup_state::W
end

function initialize_reparametrization_state(rng, ℓ; kwargs...)
    ReparametrizationState(
        ℓ, 
        initialize_warmup_state(rng, ℓ; kwargs...)
    )
end

function my_warmup(sampling_logdensity, stages, initial_warmup_state; kwargs...)
    foldl(stages; init = ((), initial_warmup_state)) do acc, stage
        stages_and_results, warmup_state = acc
        results, warmup_state′ = warmup(sampling_logdensity, stage, warmup_state; kwargs...)
        stage_information = (stage, results, warmup_state = warmup_state′)
        (stages_and_results..., stage_information), warmup_state′
    end
end

function warmup(sampling_logdensity, stage::Nothing, reparametrization_state::ReparametrizationState; kwargs...)
    @unpack reparametrization, warmup_state = reparametrization_state
    w, warmup_state = warmup(sampling_logdensity, stage, warmup_state)
    return w, ReparametrizationState(reparametrization, warmup_state)
end
function warmup(sampling_logdensity, stage::InitialStepsizeSearch, reparametrization_state::ReparametrizationState; kwargs...)
    @unpack reparametrization, warmup_state = reparametrization_state
    w, warmup_state = warmup(sampling_logdensity, stage, warmup_state)
    return w, ReparametrizationState(reparametrization, warmup_state)
end

handle_reparametrization(reference, source, target, Q, reference_draws) = begin
    try 
        return target, evaluate_ℓ(target, reparametrize(source, target, Q.q); strict = true)
    catch e
        @warn """
Failed to reparametrize to reference parameters: 
    $source 
    $target 
    $draw
    $e
Trying to recover...
        """
    end
    for rdraw in reverse(eachcol(reference_draws))
        try 
            return target, evaluate_ℓ(target, reparametrize(reference, target, rdraw); strict = true)
        catch e
            continue
        end
    end
    @warn """Failed to reparametrize all draws so far. Not reparametrizing!"""
    return target, Q
end

function warmup(sampling_logdensity, tuning::TuningNUTS{M}, reparametrization_state::ReparametrizationState; kwargs...) where {M}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack reparametrization, warmup_state = reparametrization_state
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    posterior_matrix = _empty_posterior_matrix(Q, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, reparametrization)
    ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N;
                                       currently_warmup = true,
                                       tuning = M ≡ Nothing ? "stepsize" : "stepsize, $(M) metric and parametrization")
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        posterior_matrix[:, i] = reparametrize(reparametrization, ℓ, Q.q)
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    if M ≢ Nothing
        reparametrization, Q = handle_reparametrization(
            ℓ, 
            reparametrization,
            find_reparametrization(
                reparametrization, reparametrize(ℓ, reparametrization, posterior_matrix); kwargs...
            ),
            Q,
            posterior_matrix
        )
        κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, reparametrize(ℓ, reparametrization, posterior_matrix)), λ))
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ, reparametrization = WarmupHMC.reparametrization_parameters(reparametrization))
    end
    ((; posterior_matrix, tree_statistics, ϵs), ReparametrizationState(reparametrization, WarmupState(Q, κ, final_ϵ(ϵ_state))))
end

function mcmc(sampling_logdensity, N, reparametrization_state::ReparametrizationState)
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack reparametrization, warmup_state = reparametrization_state
    sampling_logdensity = SamplingLogDensity(rng, reparametrization, algorithm, reporter)
    @unpack Q = warmup_state
    posterior_matrix = _empty_posterior_matrix(Q, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; currently_warmup = false)
    steps = mcmc_steps(sampling_logdensity, warmup_state)
    # Q = evaluate_ℓ(reparametrization, reparametrize(ℓ, reparametrization, Q.q); strict = true)
    for i in 1:N
        Q, tree_statistics[i] = mcmc_next_step(steps, Q)
        posterior_matrix[:, i] = reparametrize(reparametrization, ℓ, Q.q)
        report(mcmc_reporter, i)
    end
    (; posterior_matrix, tree_statistics)
end

end