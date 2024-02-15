module WarmupHMCDynamicHMCExt

using WarmupHMC, DynamicHMC, UnPack, Random, NaNStatistics

import WarmupHMC: reparametrize, find_reparametrization, mcmc_with_reparametrization, mcmc_keep_reparametrization, reparametrization_parameters, find_reparametrization_and_reparametrize, TuningConfig

import DynamicHMC: default_warmup_stages, default_reporter, NUTS, SamplingLogDensity, _warmup, mcmc, WarmupState, initialize_warmup_state, warmup, InitialStepsizeSearch, TuningNUTS, _empty_posterior_matrix, TreeStatisticsNUTS, Hamiltonian, initial_adaptation_state, make_mcmc_reporter, evaluate_ℓ, current_ϵ, sample_tree, adapt_stepsize, report, REPORT_SIGDIGITS, GaussianKineticEnergy, regularize_M⁻¹, sample_M⁻¹, final_ϵ, mcmc_steps, mcmc_next_step, Symmetric, Diagonal, is_divergent

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

finite_evaluate_ℓ(reparametrization, posterior_matrix) = begin
    for draw in reverse(eachcol(posterior_matrix))
        try 
            return evaluate_ℓ(reparametrization, collect(draw); strict = true)
        catch e
            @debug """
    Failed to evaluate log density: 
        $reparametrization
        $draw
        $(WarmupHMC.exception_to_string(e))
    Trying to recover...
            """
        end
    end
    return evaluate_ℓ(reparametrization, collect(posterior_matrix[:, 1]))
end

nansample_M⁻¹(::Type{Diagonal}, posterior_matrix) = Diagonal(vec(nanvar(posterior_matrix; dims = 2)))
nansample_M⁻¹(::Type{Symmetric}, posterior_matrix) = Symmetric(nancov(posterior_matrix; dims = 2))

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
        posterior_matrix[:, i] = Q.q
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    if M ≢ Nothing
        # new_reparametrization = find_reparametrization(reparametrization, posterior_matrix; kwargs...)
        # posterior_matrix = reparametrize(reparametrization, new_reparametrization, posterior_matrix)
        reparametrization, posterior_matrix = find_reparametrization_and_reparametrize(
            reparametrization, posterior_matrix; kwargs...
        )
        Q = finite_evaluate_ℓ(reparametrization, posterior_matrix)
        κ = GaussianKineticEnergy(regularize_M⁻¹(nansample_M⁻¹(M, posterior_matrix), λ))
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ, reparametrization = reparametrization_parameters(reparametrization))
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
    for i in 1:N
        Q, tree_statistics[i] = mcmc_next_step(steps, Q)
        posterior_matrix[:, i] = reparametrize(reparametrization, ℓ, Q.q)
        report(mcmc_reporter, i)
    end
    (; posterior_matrix, tree_statistics)
end

import WarmupHMC: InfoStruct, TuningConfig

function warmup(sampling_logdensity, tuning::TuningConfig, reparametrization_state::ReparametrizationState; kwargs...)
    state = TuningState(sampling_logdensity, tuning, reparametrization_state)
    while !done(state)
        step!(state)
    end
    debug(state)
    ((; state.tree_statistics), ReparametrizationState(state))
end

mutable struct TuningState{T,I<:NamedTuple} <: InfoStruct
    info::I
    TuningState{T}(info::I) where {T,I<:NamedTuple} = new{T,I}(info)
    TuningState{T}(;kwargs...) where {T} = TuningState{T}((;kwargs...))
    TuningState{T}(other::TuningState) where {T} = TuningState{T}(other.info)
end
Base.setproperty!(source::T, key::Symbol, x) where {T<:TuningState} = hasfield(T, key) ? setfield!(source, key, x) : setfield!(source, :info, merge(source.info, (;key=>x)))
update!(what::TuningState, info::NamedTuple) = begin 
    what.info = merge(what.info, info)
end
update!(what::TuningState, other::TuningState) = update!(what, other.info)
debug(state::TuningState) = begin 
    dt = final_ϵ(state)
    n_draws = length(state.tree_statistics)
    if n_draws > 0
        n_steps = sum(getproperty.(state.tree_statistics, :steps))
        frac_divergent = sum(is_divergent.(getproperty.(state.tree_statistics, :termination))) / n_draws
        display((;dt, n_draws, n_steps, avg_n_steps=n_steps/n_draws, frac_divergent))
    end
end


TuningState(sampling_logdensity, tuning::TuningConfig{T}, reparametrization_state) where {T} = begin 
    @unpack rng, algorithm = sampling_logdensity
    @unpack reparametrization, warmup_state = reparametrization_state
    @unpack Q, κ, ϵ = warmup_state
    TuningState{T}(;
        tuning.info...,
        rng, algorithm, reparametrization, Q, κ, ϵ,
        posterior_matrix=posterior_matrix(tuning, Q),
        tree_statistics=Vector{TreeStatisticsNUTS}(),
        H=Hamiltonian(κ, reparametrization),
        ϵ_state=initial_adaptation_state(tuning.stepsize_adaptation, ϵ),
        counter=1
    )
end
# n_draws(state::TuningState) = size(state.posterior_matrix, 2)
done(state::TuningState) = state.counter > state.target
step!(state::TuningState) = begin 
    Q, stats = sample_tree(state.rng, state.algorithm, Hamiltonian(state), state.Q, current_ϵ(state.ϵ_state))
    handle_transition!(state, Q, stats)
end
Hamiltonian(state::TuningState) = state.H
handle_transition!(state::TuningState, Q, stats) = begin
    handle_draw!(state, Q.q)
    ϵ_state = adapt_stepsize(state.stepsize_adaptation, state.ϵ_state, stats.acceptance_rate)
    state.info = merge(state.info, (;Q, ϵ_state))
    push!(state.tree_statistics, stats)
end
posterior_matrix(::TuningConfig, ::Any) = error("unimplemented")
handle_draw!(::TuningState, q) = error("unimplemented")
ReparametrizationState(state::TuningState) = ReparametrizationState(
    state.reparametrization, 
    WarmupState(
        state.Q, 
        GaussianKineticEnergy(state),
        final_ϵ(state)
    )
)
final_ϵ(state::TuningState) = final_ϵ(state.ϵ_state)
GaussianKineticEnergy(state::TuningState) = state.κ

TuningConfig{:step}(target::Int, stepsize_adaptation=DualAveraging()) = TuningConfig{:step}(;target, stepsize_adaptation)
posterior_matrix(::TuningConfig{:step}, ::Any) = nothing
handle_draw!(state::TuningState{:step}, ::Any) = begin 
    state.counter += 1
end

TuningConfig{:diagonal}(target::Int, stepsize_adaptation=DualAveraging()) = TuningConfig{:diagonal}(;target, stepsize_adaptation)
posterior_matrix(cfg::TuningConfig{:diagonal}, Q) = _empty_posterior_matrix(Q, cfg.target)
handle_draw!(state::TuningState{:diagonal}, q) = begin 
    state.posterior_matrix[:, state.counter] .= q
    state.counter += 1
end
GaussianKineticEnergy(state::TuningState{:diagonal}) = GaussianKineticEnergy(
    regularize_M⁻¹(Diagonal(vec(nanvar(state.posterior_matrix; dims = 2))), 5/state.target)
)

TuningConfig{:mad}(target::Int, thin=1, stepsize_adaptation=DualAveraging(γ=.1)) = TuningConfig{:mad}(;target, stepsize_adaptation, thin, inner=0, accs=fill(-Inf, target))
posterior_matrix(cfg::TuningConfig{:mad}, Q) = _empty_posterior_matrix(Q, cfg.target)
Hamiltonian(state::TuningState{:mad}) = Hamiltonian(
    state.κ, RecordingPosterior(state, state.reparametrization),
)
record!(state::TuningState{:mad}, draw, acc) = begin
    state.inner += 1
    if state.inner == state.thin
        state.inner = 0
        state.counter += 1
    end
    idx = argmin(state.accs)
    if acc > state.accs[idx] 
        state.posterior_matrix[:, idx] .= draw
        state.accs[idx] = acc
    end
end
handle_draw!(::TuningState{:mad}, ::Any) = nothing
GaussianKineticEnergy(state::TuningState{:mad}) = GaussianKineticEnergy(
    Diagonal(vec(nanmad(state.posterior_matrix; dims = 2)).^2)
)

TuningConfig{:reparam}(target::Int, stepsize_adaptation=DualAveraging(); kwargs...) = TuningConfig{:reparam}(;target, stepsize_adaptation, reparametrization_kwargs=kwargs)
posterior_matrix(cfg::TuningConfig{:reparam}, Q) = _empty_posterior_matrix(Q, cfg.target)
handle_draw!(state::TuningState{:reparam}, q) = begin 
    state.posterior_matrix[:, state.counter] .= q
    state.counter += 1
end
ReparametrizationState(state::TuningState{:reparam}) = ReparametrizationState(
    TuningState{:diagonal}(reparam(state))
)
reparam(state::TuningState{T}) where {T} = begin 
    reparametrization, posterior_matrix = find_reparametrization_and_reparametrize(
        state.reparametrization, state.posterior_matrix; state.reparametrization_kwargs...
    )
    Q = finite_evaluate_ℓ(reparametrization, posterior_matrix)
    TuningState{T}(merge(state.info, (;reparametrization, posterior_matrix, Q)))
end

TuningConfig{:mad_reparam}(target::Int, thin=1, stepsize_adaptation=DualAveraging(γ=.1); kwargs...) = TuningConfig{:mad_reparam}(;target, stepsize_adaptation, thin, inner=0, accs=fill(-Inf, target), reparametrization_kwargs=kwargs)
posterior_matrix(cfg::TuningConfig{:mad_reparam}, Q) = _empty_posterior_matrix(Q, cfg.target)
Hamiltonian(state::TuningState{:mad_reparam}) = Hamiltonian(
    state.κ, RecordingPosterior(state, state.reparametrization),
)
record!(state::TuningState{:mad_reparam}, args...) = begin
    tmp = TuningState{:mad}(state)
    record!(tmp, args...)
    update!(state, tmp)
end
handle_draw!(::TuningState{:mad_reparam}, ::Any) = nothing
ReparametrizationState(state::TuningState{:mad_reparam}) = begin
    ReparametrizationState(
        TuningState{:mad}(reparam(state))
)
end



TuningConfig{:mod_step}(f::Function) = TuningConfig{:mod_step}(;f, target=0, stepsize_adaptation=DualAveraging())
posterior_matrix(::TuningConfig{:mod_step}, ::Any) = nothing
final_ϵ(state::TuningState{:mod_step}) = state.f(state.ϵ)

TuningConfig{:adaptive}(break_condition::Function, stages) = TuningConfig{:adaptive}(;break_condition, stages)
warmup(sampling_logdensity, tuning::TuningConfig{:adaptive}, reparametrization_state::ReparametrizationState; kwargs...) = begin 
    tree_statistics = Vector{TreeStatisticsNUTS}()
    for stage in tuning.stages
        info, reparametrization_state = warmup(sampling_logdensity, stage, reparametrization_state)
        append!(tree_statistics, info.tree_statistics)
        tuning.break_condition(info.tree_statistics) && break
    end
    ((; tree_statistics), reparametrization_state)
end
TuningConfig{:adaptive}(n_draws::Int, stages) = TuningConfig{:adaptive}(ts->length(ts) > n_draws / 4, stages)
TuningConfig{:adaptive}(type::Symbol, args...; kwargs...) = TuningConfig{:adaptive}(Val{type}(), args...; kwargs...)
TuningConfig{:adaptive}(::Val{T}, n_draws=1000, target=1023, args...; kwargs...) where {T} = TuningConfig{:adaptive}(n_draws, [
    TuningConfig{T}(target, thin, args...; kwargs...)
    for thin in [1,2,4,8,16,32,64,128,256]
])



mutable struct RecordingPosterior{R,P}
    recorder::R
    posterior::P
end
using LogDensityProblems
LogDensityProblems.capabilities(::Type{<:RecordingPosterior}) = LogDensityProblems.LogDensityOrder{2}()
LogDensityProblems.dimension(source::RecordingPosterior) = LogDensityProblems.dimension(source.posterior)
LogDensityProblems.logdensity(source::RecordingPosterior, draw::AbstractVector) = LogDensityProblems.logdensity(
    source.posterior, draw
)
LogDensityProblems.logdensity_and_gradient(source::RecordingPosterior, draw::AbstractVector) = LogDensityProblems.logdensity_and_gradient(
    source.posterior, draw#record!(source, draw)
)
record!(source::RecordingPosterior, args...) = record!(source.recorder, args...) 

import DynamicHMC: TrajectoryNUTS, leaf, logdensity, leaf_acceptance_statistic, leaf_turn_statistic


function leaf(trajectory::TrajectoryNUTS{Hamiltonian{K,P}}, z, is_initial) where {K, P<:RecordingPosterior}
    @unpack H, π₀, min_Δ, turn_statistic_configuration = trajectory
    Δ = is_initial ? zero(π₀) : logdensity(H, z) - π₀
    !is_initial && record!(H.ℓ, z.Q.q, Δ)
    isdiv = Δ < min_Δ
    v = leaf_acceptance_statistic(Δ, is_initial)
    if isdiv
        nothing, v
    else
        τ = leaf_turn_statistic(turn_statistic_configuration, H, z)
        (z, Δ, τ), v
    end
end


end