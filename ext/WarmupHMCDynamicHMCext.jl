module WarmupHMCDynamicHMCext

using WarmupHMC, DynamicHMC, UnPack

import WarmupHMC: reparametrize, find_reparametrization, mcmc_with_reparametrization

import DynamicHMC: default_warmup_stages, default_reporter, NUTS, SamplingLogDensity, _warmup, mcmc, WarmupState, initialize_warmup_state, warmup, TuningNUTS, _empty_posterior_matrix, TreeStatisticsNUTS, Hamiltonian, initial_adaptation_state, make_mcmc_reporter, evaluate_ℓ, current_ϵ, sample_tree, adapt_stepsize, report, GaussianKineticEnergy, regularize_M⁻¹, sample_M⁻¹, final_ϵ, mcmc_steps

function mcmc_with_reparametrization(rng, ℓ, N; initialization = (),
    warmup_stages = default_warmup_stages(),
    algorithm = NUTS(), reporter = default_reporter())
@unpack final_warmup_state, inference =
mcmc_keep_reparametrization(rng, ℓ, N; initialization = initialization,
   warmup_stages = warmup_stages, algorithm = algorithm,
   reporter = reporter)
@unpack κ, ϵ = final_warmup_state
(; inference..., κ, ϵ)
end

function mcmc_keep_reparametrization(rng::AbstractRNG, ℓ, N::Integer;
                          initialization = (),
                          warmup_stages = default_warmup_stages(),
                          algorithm = NUTS(),
                          reporter = default_reporter())
    sampling_logdensity = SamplingLogDensity(rng, ℓ, algorithm, reporter)
    initial_warmup_state = initialize_reparametrization_state(rng, ℓ; initialization...)
    warmup, warmup_state = _warmup(sampling_logdensity, warmup_stages, initial_warmup_state)
    inference = mcmc(sampling_logdensity, N, warmup_state)
    (; initial_warmup_state, warmup, final_warmup_state = warmup_state, inference,
     sampling_logdensity)
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


function warmup(sampling_logdensity, stage, reparametrization_state::ReparametrizationState)
    @unpack reparametrization, warmup_state = reparametrization_state
    warmup, warmup_state = warmup(sampling_logdensity, stage, warmup_state)
    return warmup, ReparametrizationState(reparametrization, warmup_state)
end

function warmup(sampling_logdensity, tuning::TuningNUTS{M}, reparametrization_state::ReparametrizationState) where {M}
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
                                       tuning = M ≡ Nothing ? "stepsize" : "stepsize and $(M) metric")
    Q = evaluate_ℓ(reparametrization, reparametrize(ℓ, reparametrization, Q.q); strict = true)
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        posterior_matrix[:, i] = reparametrize(reparametrization, ℓ, Q.q)
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    Q = evaluate_ℓ(ℓ, reparametrize(reparametrization, ℓ, Q.q); strict = true)
    if M ≢ Nothing
        reparametrization = find_reparametrization(ℓ, posterior_matrix)
        κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, reparametrize(ℓ, reparametrization, posterior_matrix)), λ))
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
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
    Q = evaluate_ℓ(reparametrization, reparametrize(ℓ, reparametrization, Q.q); strict = true)
    for i in 1:N
        Q, tree_statistics[i] = mcmc_next_step(steps, Q)
        posterior_matrix[:, i] = reparametrize(reparametrization, ℓ, Q.q)
        report(mcmc_reporter, i)
    end
    (; posterior_matrix, tree_statistics)
end

end