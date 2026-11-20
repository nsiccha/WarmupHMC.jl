# Cooperative multi-chain warm-up: the chains adapt together.
#
# Metric adaptation:
#   * every leapfrog state of every chain feeds running position and gradient
#     moments, from which the Nutpie diagonal scale
#     `(var(position) / var(gradient))^(1/4)` follows;
#   * the moments ACCUMULATE from the chain's last restart on, over batches whose
#     budget doubles with no cap;
#   * at a batch end a chain normally restarts when its pooled estimate is incompatible
#     with the metric it samples under: relative condition number × step-size gain
#     > `restart_threshold`. The original used √2; the default is 2, the value
#     `adaptive_warmup_mcmc` uses for the same statistic, because √2 restarted on
#     estimate noise alone (snag `cooperative-warm-96ff19ae`);
#   * persistent extreme failures can start a new generation at the same index.
# Step-size adaptation and pooling:
#   * the step size: a per-chain fit of κ in acceptance ≈ 2Φ(-κε²), with the pooled
#     log κ as prior and a backoff on failed transitions;
#   * the kernel weight between a chain and a pool, on the RMS standardized
#     distance between their means.
# Pool structure:
#   * pooling goes through soft CLUSTERS, so it costs O(chains × clusters × d),
#     linear in the number of chains and in the dimension. Once the chains have
#     mixed there is one cluster: the original's single shared estimate;
#   * no dense matrices;
#   * one seed RNG gives every chain its stream.
#
# Rounds and batches. A round gives every running chain `n_evaluations` gradient
# evaluations and ends at a barrier; batch k is 2^(k-1) rounds. Between barriers a
# chain reads only its own state and values fixed at the last barrier, so a run is
# deterministic per seed at any thread count. A time or evaluation budget and an
# interrupt are the exception: they can end the final round at a point that
# depends on the schedule.

# =============================================================================
# Running moments of leapfrog states
# =============================================================================

"""
    CoopMoments(dimension)

Running per-coordinate means and summed squared deviations of positions and
gradients (Welford), over `weight` leapfrog states. Moments of disjoint states
combine exactly (`_combine_moments!`), with fractional weights, which is how
chains pool into clusters and clusters into a chain's estimate.
"""
mutable struct CoopMoments
    weight::Float64
    position_mean::Vector{Float64}
    position_m2::Vector{Float64}
    gradient_mean::Vector{Float64}
    gradient_m2::Vector{Float64}
end
CoopMoments(dimension::Integer) =
    CoopMoments(0.0, zeros(dimension), zeros(dimension), zeros(dimension), zeros(dimension))

_reset_moments!(m::CoopMoments) = begin
    m.weight = 0
    fill!(m.position_mean, 0); fill!(m.position_m2, 0)
    fill!(m.gradient_mean, 0); fill!(m.gradient_m2, 0)
    m
end

_fit_moments!(m::CoopMoments, position, gradient) = begin
    m.weight += 1
    w = 1 / m.weight
    @inbounds for i in eachindex(m.position_mean)
        dx = position[i] - m.position_mean[i]
        m.position_mean[i] += w * dx
        m.position_m2[i] += dx * (position[i] - m.position_mean[i])
        dg = gradient[i] - m.gradient_mean[i]
        m.gradient_mean[i] += w * dg
        m.gradient_m2[i] += dg * (gradient[i] - m.gradient_mean[i])
    end
    m
end

# Every good leaf of the transition just taken: not the initial point, and an
# energy error above the halo's threshold (`record!`), which also excludes
# divergent and out-of-support leaves, whose gradients are never meaningful.
_fit_leaves!(m::CoopMoments, leaves::NUTSLeaves) = begin
    for k in 2:length(leaves)
        dH = leaves.dH[k]
        isfinite(dH) && dH > log(1e-2) || continue
        _fit_moments!(m, view(leaves.position, :, k), view(leaves.gradient, :, k))
    end
    m
end

# `out` := Σⱼ wⱼ partsⱼ, the moments of the union of the parts' states with each
# state of part j counted wⱼ times. Parts without states, or with zero weight,
# contribute nothing.
_combine_moments!(out::CoopMoments, parts, weights; within_target=false) = begin
    _reset_moments!(out)
    for (m, w) in zip(parts, weights)
        (w > 0 && m.weight > 0) || continue
        ww = w * m.weight
        out.weight += ww
        out.position_mean .+= ww .* m.position_mean
        out.gradient_mean .+= ww .* m.gradient_mean
    end
    out.weight > 0 || return out
    out.position_mean ./= out.weight
    out.gradient_mean ./= out.weight
    for (m, w) in zip(parts, weights)
        (w > 0 && m.weight > 0) || continue
        if within_target
            @. out.position_m2 += w * m.position_m2
            @. out.gradient_m2 += w * m.gradient_m2
        else
            @. out.position_m2 += w * (m.position_m2 + m.weight * abs2(m.position_mean - out.position_mean))
            @. out.gradient_m2 += w * (m.gradient_m2 + m.weight * abs2(m.gradient_mean - out.gradient_mean))
        end
    end
    out
end

# The Nutpie diagonal scale per coordinate, `(var(position) / var(gradient))^(1/4)`,
# which equals `sqrt(std(position) / std(gradient))` (adaptive's restart estimate).
# `NaN` where the moments cannot say: fewer than two states, or a degenerate spread.
_moment_scales(m::CoopMoments) = map(m.position_m2, m.gradient_m2) do p, g
    s = (p / g)^(1 / 4)
    m.weight >= 2 && isfinite(s) && s > 0 ? s : NaN
end

# Position variance per coordinate, `NaN` without two states.
_moment_variances(m::CoopMoments) =
    m.weight >= 2 ? m.position_m2 ./ m.weight : fill(NaN, length(m.position_m2))

# =============================================================================
# Step size: acceptance ≈ (1 - π)·2Φ(-κε²) (acceptance model with a failure floor)
# =============================================================================
#
# The model `2Φ(-κε²)` explains every rejection by the step size. A target
# whose trajectories leave its support (or hit another step-size-independent
# failure) rejects at ANY step size, and that model then shrinks ε without bound:
# on a standard normal truncated at its mode it fell from adaptive's 0.28 to 0.001,
# and the chains restarted at every batch end. So the fit also carries a floor
# (1 - π) for failures the step size does not cause, solved in closed form for each
# κ, and ε is chosen for the step-size-driven part only.

const _STANDARD_NORMAL = Normal()

_kappa_acceptance(log_kappa, ε) = 2cdf(_STANDARD_NORMAL, -exp(log_kappa) * ε^2)
"The step size the model `acceptance = 2Φ(-κε²)` gives for `target` acceptance."
_kappa_stepsize(log_kappa, target) = sqrt(-quantile(_STANDARD_NORMAL, target / 2) / exp(log_kappa))
"The log κ for which `ε` has `target` acceptance."
_kappa_log(ε, target) = log(-quantile(_STANDARD_NORMAL, target / 2)) - 2log(ε)

# The floor `1 - π` that best fits the pairs for a given log κ: the weighted least
# squares scale of the model's predictions, kept within [0.05, 1].
_kappa_floor(log_kappa, stepsizes, acceptances, weights) = begin
    num = den = 0.0
    for (e, a, w) in zip(stepsizes, acceptances, weights)
        m = _kappa_acceptance(log_kappa, e)
        num += w * a * m
        den += w * m * m
    end
    den > 0 ? clamp(num / den, 0.05, 1.0) : 1.0
end

# Weighted least squares of the (step size, acceptance) pairs against the model,
# with a Normal prior on log κ. The grid spans both the prior and the previous
# estimate, so a chain whose own evidence disagrees with its pool can still move.
# `coarse_points` and `fine_points` size the two grids; the defaults are the
# cooperative sampler's. The continuous sampler (`continuous_nuts.jl`) fits on
# every transition and passes smaller grids.
_fit_log_kappa(stepsizes, acceptances, weights, previous, prior, prior_weight;
               coarse_points=61, fine_points=21) = begin
    loss(lk) = begin
        c = _kappa_floor(lk, stepsizes, acceptances, weights)
        sum(w * abs2(a - c * _kappa_acceptance(lk, e)) for (e, a, w) in zip(stepsizes, acceptances, weights); init=0.0) +
            prior_weight * 0.05 * abs2(lk - prior)
    end
    center = (previous + prior) / 2
    half_width = 3 + abs(previous - prior) / 2
    coarse = center .+ range(-half_width, half_width; length=coarse_points)
    best = coarse[argmin(loss.(coarse))]
    fine = best .+ range(-step(coarse), step(coarse); length=fine_points)
    fine[argmin(loss.(fine))]
end

# Efficiency of a step size under the model: distance per leapfrog step times the
# predicted acceptance (the floor cancels in a ratio). The original measured this
# gain with its squared-jump regression; here the κ fit supplies it.
_kappa_efficiency(log_kappa, ε) = ε * _kappa_acceptance(log_kappa, ε)

# =============================================================================
# Chains
# =============================================================================

_coop_energy(scale) = DynamicHMC.GaussianKineticEnergy(MatrixFactorization(scale, scale'), MatrixInverse(scale'))

"""
    CoopChain

One chain of a [`cooperative_warmup_mcmc`](@ref) run. `status` is `:warming`
(adapting its step size since its last restart), `:sampling` (kernel fixed,
keeping draws) or `:done` (holds `n_draws` draws).
"""
mutable struct CoopChain{R,L,RL,N}
    const chain_index::Int
    const rng::R
    const lpdf::L
    const recording_lpdf::RL
    const nonlinear_recorder::N
    geometry_current::Bool          # may lend source-frame geometry to the shared fit
    const dimension::Int
    const monitor_ess::Bool
    const start_time::UInt64
    # This chain's Treebars node (`nothing` without progress); a runtime handle,
    # never checkpointed.
    const progress::Any
    position_and_gradient
    # The metric's scale `L` (`L*L'` approximates the posterior covariance) and its
    # kinetic energy.
    scale::Union{Diagonal{Float64,Vector{Float64}},ScaleThenReflect{Float64}}
    kinetic_energy
    stepsize::Float64
    log_kappa::Float64
    kappa_prior::Float64            # pooled log κ, fixed at the last barrier
    backoff::Float64                # halved by a failed warm-up transition
    stepsizes::Vector{Float64}      # every transition since the last restart
    acceptances::Vector{Float64}
    sampling_start::Int             # pairs up to this index are warm-up
    moments::CoopMoments            # leapfrog states since the last restart
    status::Symbol
    current_transition_counter::Int
    total_transition_counter::Int
    total_evaluation_counter::Int
    round_evaluations::Int
    n_divergent::Int
    n_divergent_samples::Int
    n_samples::Int
    steps_per_draw
    ess::Vector{Float64}            # per coordinate, source frame, coordinate order
    scale_changes::Vector{Float64}  # relative condition number at each restart
    n_restarts::Int
    batches::Vector{NamedTuple}     # one entry per batch end
    dropped_posterior_position::Matrix{Float64}
    dropped_posterior_gradient::Matrix{Float64}
    dropped_n_divergent_samples::Int
    freeze::CoopFreeze
    generation::Int
    abandoned::Vector{NamedTuple}
    rescue_prior::Union{Nothing,Float64}
end

chain_draws(chain::CoopChain) = chain.recording_lpdf.posterior_position
n_chain_draws(chain::CoopChain) = size(chain_draws(chain), 2)
_running(chain::CoopChain) = chain.status in (:warming, :sampling)
_coop_geometry_current(chain::CoopChain) = chain.geometry_current && chain.status in (:warming, :sampling, :done)

function _coop_nonlinear_recorder(p, lpdf, capacity, settings, rng)
    recorder = _restore_nonlinear_recorder(p, lpdf, capacity; settings...)
    if recorder !== get(p, :nonlinear_recorder, nothing) &&
            _effective_nonlinear_evidence(recorder, lpdf) !== :linear_pool
        # Independent reservoirs within chains, independent slot choices at
        # the barrier. On resume the saved reservoir already has its stream.
        sample = _leaf_sample(recorder.online)
        isnothing(sample) || Random.seed!(sample.rng, rand(rng, UInt64))
    end
    recorder
end

# Build chain `index` on its own stream: deep-copied density, initialization
# (Pathfinder unless `init` says otherwise), diagonal scale from the
# initializer's squared scale, initial step size and the log κ that gives it.
_coop_chain(index, rng, lpdf, init, cfg; progress, adapt_centering=false, waiting=false,
            current_sources=nothing, start_time=time_ns()) = begin
    dimension = LogDensityProblems.dimension(lpdf)
    recording_lpdf = RecordingPosterior2(lpdf;
        recorder=LimitedRecorder2(cfg.recording_target, max(1, cfg.n_evaluations ÷ cfg.recording_target)), rng)
    if waiting
        # Leave the target and its initializer untouched until its donor is ready.
        scale = Diagonal(ones(dimension))
        kinetic_energy = _coop_energy(scale)
        position_and_gradient = nothing
        stepsize = log_kappa = NaN
    else
        (; position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, cfg.initializer_kw...)
        position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
        if !isnothing(current_sources) && reparam_sources(lpdf) != current_sources
            original = snapshot_reparametrization(reparametrizer(lpdf))
            restore_reparam_sources!(lpdf, current_sources)
            position_and_gradient, rejection = _checked_centering_point(lpdf, original, position_and_gradient)
            isnothing(rejection) || throw(ArgumentError(
                "replacement init for chain $index could not enter the current coordinates: $(rejection.reason)"))
            # The initializer's scale belongs to its original coordinates.
            squared_scale = Diagonal(ones(dimension))
        end
        scale = _initial_diagonal_scale(squared_scale)
        kinetic_energy = _coop_energy(scale)
        stepsize = DynamicHMC.find_initial_stepsize(
            DynamicHMC.InitialStepsizeSearch(),
            DynamicHMC.local_log_acceptance_ratio(
                _hamiltonian(kinetic_energy, lpdf),
                DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy)),
            ),
        )
        log_kappa = _kappa_log(stepsize, cfg.target_acceptance_rate)
    end
    nonlinear_recorder = adapt_centering ? _coop_nonlinear_recorder(
        (;), lpdf, cfg.recording_target, cfg.nonlinear_settings, rng) : nothing
    chain = CoopChain(
        index, rng, lpdf, recording_lpdf, nonlinear_recorder, true, dimension, cfg.monitor_ess, start_time, progress,
        position_and_gradient, scale, kinetic_energy, stepsize, log_kappa, log_kappa, 1.0,
        Float64[], Float64[], 0, CoopMoments(dimension), waiting ? :waiting : :warming,
        0, 0, 0, 0, 0, 0, 0, OnlineStatsBase.Mean(), zeros(dimension), Float64[], 0,
        NamedTuple[], Matrix{Float64}(undef, dimension, 0), Matrix{Float64}(undef, dimension, 0), 0,
        CoopFreeze(get(cfg, :stepsize_adaptation_limit, 50)),
        1, NamedTuple[], nothing,
    )
    _initial_chain_progress!(progress, nothing, chain)
    update_progress!(progress, waiting ? "waiting for first wave" : "warming · initialized")
    chain
end

# One NUTS transition under the chain's current kernel. Warm-up transitions refit
# κ and set the next step size; the transition that ends warm-up fixes it, and
# every later one keeps a draw.
_coop_transition!(chain::CoopChain, cfg, total_evaluations) = begin
    recording_lpdf = chain.recording_lpdf
    reset!(recording_lpdf.leaves)
    ε = chain.stepsize
    chain.position_and_gradient, stats = DynamicHMC.sample_tree(
        chain.rng, cfg.algorithm, _hamiltonian(chain.kinetic_energy, recording_lpdf),
        chain.position_and_gradient, ε,
    )
    finalize_leaf_recording!(recording_lpdf, stats.depth)
    isnothing(chain.nonlinear_recorder) ||
        record_nonlinear!(chain.nonlinear_recorder, chain.lpdf, recording_lpdf.leaves, ε)
    _fit_leaves!(chain.moments, recording_lpdf.leaves)
    steps = stats.steps
    chain.total_evaluation_counter += steps
    chain.round_evaluations += steps
    Threads.atomic_add!(total_evaluations, steps)
    chain.current_transition_counter += 1
    chain.total_transition_counter += 1
    OnlineStatsBase.fit!(chain.steps_per_draw, steps)
    divergent = DynamicHMC.is_divergent(stats.termination)
    if divergent
        chain.n_divergent += 1
        chain.freeze.divergences += 1
    end
    acceptance = stats.acceptance_rate
    push!(chain.stepsizes, ε)
    push!(chain.acceptances, acceptance)
    if chain.status === :warming
        chain.freeze.warmup_transitions += 1
        # A divergence alone is not a step-size failure: leaving the support
        # diverges at any step size.
        failed = acceptance < 0.1
        chain.backoff = failed ? chain.backoff / 2 : min(1.0, 1.5chain.backoff)
        n = length(chain.acceptances)
        weights = cfg.kappa_decay .^ (n-1:-1:0)
        chain.log_kappa = _fit_log_kappa(chain.stepsizes, chain.acceptances, weights,
            chain.log_kappa, chain.kappa_prior, cfg.kappa_prior_weight)
        chain.stepsize = _kappa_stepsize(chain.log_kappa, cfg.target_acceptance_rate) * chain.backoff
        if chain.current_transition_counter >= chain.freeze.warmup
            # Decide at the barrier against all checks, including simultaneous ones.
            score = sum(abs2, chain.scale' * chain.position_and_gradient.∇ℓq) / chain.dimension
            push!(chain.freeze.scores, score)
            chain.freeze.pending = true
            chain.freeze.clean = !divergent && acceptance >= 0.6
        end
    else
        f = chain.freeze
        length(f.pilot) < f.pilot_quota && push!(f.pilot, copy(chain.position_and_gradient.q))
        if chain.n_samples < cfg.n_draws
            append!(recording_lpdf.posterior_position, chain.position_and_gradient.q)
            append!(recording_lpdf.posterior_gradient, chain.position_and_gradient.∇ℓq)
            divergent && (chain.n_divergent_samples += 1)
            chain.n_samples += 1
        end
        chain.n_samples >= cfg.n_draws && length(f.pilot) >= f.pilot_quota && (chain.status = :done)
    end
    _transition_progress!(chain.progress, nothing, chain)
    chain
end

# The step-size gain of the restart test: how much more efficient the step size
# the κ fit prescribes would be than the one in use (≥ 1). The fit takes every
# pair since the last restart: the sampling phase holds ε fixed, so only the
# warm-up pairs separate κ from the failure floor.
#
# A chain still in step-size warm-up has no fixed step size to judge, so it
# restarts on metric disagreement alone (gain 1). An infinite gain here would
# restart a chain before its step-size adaptation can finish whenever adaptation
# takes more evaluations than an early batch contains.
_stepsize_gain(chain::CoopChain, cfg) = begin
    chain.status === :warming && return 1.0
    chain.sampling_start < length(chain.acceptances) || return 1.0
    log_kappa = _fit_log_kappa(chain.stepsizes, chain.acceptances, Ones(length(chain.acceptances)),
        chain.log_kappa, chain.kappa_prior, cfg.kappa_prior_weight)
    ε = chain.stepsize
    ε_target = _kappa_stepsize(log_kappa, cfg.target_acceptance_rate)
    max(1.0, _kappa_efficiency(log_kappa, ε_target) / _kappa_efficiency(log_kappa, ε))
end

# Relative condition number of a new diagonal scale against the current one:
# max/min of the per-coordinate scale ratios, over the coordinates the estimate
# can judge.
_relative_condition(new_scales::AbstractVector, scale::Diagonal) = begin
    lo, hi = Inf, -Inf
    for (n, s) in zip(new_scales, scale.diag)
        isfinite(n) || continue
        r = n / s
        lo = min(lo, r); hi = max(hi, r)
    end
    isfinite(lo) ? hi / lo : 1.0
end

_principal_scales(L::Diagonal) = L.diag
_principal_scales(L::ScaleThenReflect) = L.m2.diag

# Install the selected factor and re-base κ to the carried-over step size.
_restart!(chain::CoopChain, new_scales::AbstractVector, cc, cfg) =
    _restart!(chain, _complete_diagonal(new_scales, chain.scale), cc, cfg)
_restart!(chain::CoopChain, scale::AbstractMatrix, cc, cfg) = begin
    old = chain.scale
    # Several chains may select the same factor. Each published chain state
    # owns its mutable reflection vectors and diagonal.
    chain.scale = deepcopy(scale)
    chain.kinetic_energy = _coop_energy(chain.scale)
    chain.stepsize *= sqrt(minimum(_principal_scales(old)) / minimum(_principal_scales(scale)))
    chain.log_kappa = _kappa_log(chain.stepsize, cfg.target_acceptance_rate)
    chain.backoff = 1.0
    chain.rescue_prior = nothing
    empty!(chain.stepsizes); empty!(chain.acceptances)
    chain.sampling_start = 0
    _reset_moments!(chain.moments)
    isnothing(chain.nonlinear_recorder) || reset!(chain.nonlinear_recorder)
    recording_lpdf = chain.recording_lpdf
    chain.dropped_posterior_position = Matrix{Float64}(recording_lpdf.posterior_position)
    chain.dropped_posterior_gradient = Matrix{Float64}(recording_lpdf.posterior_gradient)
    chain.dropped_n_divergent_samples = chain.n_divergent_samples
    reset!(recording_lpdf)
    chain.status = :warming
    chain.current_transition_counter = 0
    chain.freeze.extensions = 0
    empty!(chain.freeze.pilot)
    chain.n_divergent = 0
    chain.n_divergent_samples = 0
    chain.n_samples = 0
    chain.steps_per_draw = OnlineStatsBase.Mean()
    chain.n_restarts += 1
    pushfirst!(chain.scale_changes, cc)
    _transformation_progress!(chain.progress, chain)
    # The monitored ESS described the dropped draws: neither the result nor the
    # node may report it for the new ones.
    fill!(chain.ess, 0.0)
    chain.monitor_ess && update_progress!(chain.progress, nothing; ess="pending...")
    chain
end

# =============================================================================
# Soft clusters
# =============================================================================

"""
    CoopClusters

The pool's soft clusters. `centers[k]` is a position; `responsibilities[i, k]` is
chain i's weight on cluster k (each row sums to one); `stats[k]` pools the
chains' moments with those weights. A chain far from every center (RMS
standardized distance above 2) seeds a new cluster, up to `max_clusters`; two
centers within 1 merge. Every step is O(chains × clusters × dimension).
"""
mutable struct CoopClusters
    centers::Vector{Vector{Float64}}
    responsibilities::Matrix{Float64}
    stats::Vector{CoopMoments}
end
CoopClusters() = CoopClusters(Vector{Float64}[], zeros(0, 0), CoopMoments[])

# A chain's location for clustering: the mean of its states since its last
# restart, or its current position before it has any.
_chain_location(chain::CoopChain) =
    chain.moments.weight > 0 ? chain.moments.position_mean : chain.position_and_gradient.q

# Per-coordinate spread that standardizes distances: the root mean over chains of
# each chain's own position variance (per-chain estimate), falling back to the
# metric's scale for a chain without states yet.
_cluster_spread(chains) = begin
    total = zeros(first(chains).dimension)
    for c in chains
        v = _moment_variances(c.moments)
        if any(x -> !(isfinite(x) && x > 0), v)
            fallback = _metric_variances(c.scale)
            @. v = ifelse(isfinite(v) & (v > 0), v, fallback)
        end
        total .+= v
    end
    sqrt.(total ./ length(chains))
end

_standardized_distance(a, b, spread) = mean(abs2((x - y) / s) for (x, y, s) in zip(a, b, spread))

_responsibilities(locations, centers, spread, bandwidth) = begin
    D = [_standardized_distance(x, c, spread) for x in locations, c in centers]
    R = exp.(-(D .- minimum(D; dims=2)) ./ (2bandwidth^2))   # row-shifted: no underflow
    R ./ sum(R; dims=2)
end

_pool_clusters!(clusters::CoopClusters, chains, responsibilities; within_target=false) = begin
    clusters.responsibilities = responsibilities
    d = first(chains).dimension
    clusters.stats = [_combine_moments!(CoopMoments(d), (c.moments for c in chains), view(responsibilities, :, k); within_target)
                      for k in axes(responsibilities, 2)]
    clusters
end

_update_active_clusters!(clusters::CoopClusters, chains, cfg; within_target=false) = begin
    spread = _cluster_spread(chains)
    locations = map(_chain_location, chains)
    centers = clusters.centers
    for x in locations
        far = isempty(centers) || minimum(c -> _standardized_distance(x, c, spread), centers) > 4
        far && length(centers) < cfg.max_clusters && push!(centers, copy(x))
    end
    _pool_clusters!(clusters, chains, _responsibilities(locations, centers, spread, cfg.bandwidth); within_target)
    # Move each center to its cluster's mean, merge centers that came within 1,
    # drop clusters no chain belongs to, then pool once more.
    for (k, s) in enumerate(clusters.stats)
        s.weight > 0 && (centers[k] = copy(s.position_mean))
    end
    occupied = vec(sum(clusters.responsibilities; dims=1)) .> 1e-8
    keep = copy(occupied)
    for k in eachindex(centers), l in k+1:length(centers)
        keep[k] && keep[l] && _standardized_distance(centers[k], centers[l], spread) < 1 && (keep[l] = false)
    end
    clusters.centers = centers[keep]
    _pool_clusters!(clusters, chains, _responsibilities(locations, clusters.centers, spread, cfg.bandwidth); within_target)
end

# Waiting slots have no geometry or step-size evidence. Keep the final weight
# rows indexed by the original target, including runs stopped in the first wave.
function _update_clusters!(clusters::CoopClusters, chains, cfg)
    within_target = get(cfg, :family, false)
    active = findall(_coop_geometry_current, chains)
    if isempty(active)
        empty!(clusters.centers)
        empty!(clusters.stats)
        clusters.responsibilities = zeros(length(chains), 0)
        return clusters
    end
    _update_active_clusters!(clusters, view(chains, active), cfg; within_target)
    if length(active) != length(chains)
        R = zeros(length(chains), length(clusters.centers))
        R[active, :] = clusters.responsibilities
        clusters.responsibilities = R
    end
    clusters
end

include("cooperative_metric.jl")

# Each chain's κ prior: the responsibility-weighted mean of its clusters' mean
# log κ. O(chains × clusters).
_update_kappa_priors!(clusters::CoopClusters, chains) = begin
    R = clusters.responsibilities
    # A cluster whose responsibilities all underflowed carries no weight anywhere.
    cluster_log_kappa = map(axes(R, 2)) do k
        mass = sum(view(R, :, k))
        mass > 0 ? sum(R[i, k] * chains[i].log_kappa for i in eachindex(chains) if R[i,k] > 0) / mass : 0.0
    end
    for (i, c) in enumerate(chains)
        _coop_geometry_current(c) || continue
        c.kappa_prior = c.status === :warming && !isnothing(c.rescue_prior) ? c.rescue_prior :
            sum(R[i, k] * cluster_log_kappa[k] for k in axes(R, 2) if R[i, k] > 0)
    end
end

# =============================================================================
# The run
# =============================================================================

mutable struct CoopRun{S,L,C,I}
    const rng::S                    # seeds each chain's stream at its creation
    const lpdf::L
    const cfg::C
    const init::I
    const progress::Any             # the run's node, `nothing` without progress
    const description::String
    const start_time::UInt64
    const prior_elapsed::Float64
    # The shared nonlinear reparametrization: refit on this master copy, then
    # restored onto every chain's own copy. `nothing` when it is not adapted.
    const reparam_lpdf::Any
    chains::Vector{CoopChain}       # in index order
    clusters::CoopClusters
    total_evaluations::Threads.Atomic{Int}
    round::Int
    batch::Int
    stop_reason::Union{Nothing,Symbol}
    last_pooled_ess::Float64
end

_elapsed(run::CoopRun) = run.prior_elapsed + (time_ns() - run.start_time) / 1e9
_running_chains(run::CoopRun) = filter(_running, run.chains)

# The `init` chain `index` starts from: one shared init (a position vector of reals
# included), or a per-chain vector of inits.
_chain_init(init::AbstractVector{<:Real}, index) = init
_chain_init(init::AbstractVector, index) = init[index]
_chain_init(init, index) = init
_coop_target(lpdf, index) = lpdf
_coop_cold_init(init, index, rng) = _require_cold_init(_chain_init(init, index), index)

# Create chains `indices` in parallel on `cfg.n_cores` workers. Their streams are
# drawn from the seed RNG serially first, in index order, so the seed-to-chain
# map does not depend on scheduling (decision `0m8p12j`, prong `rng`).
_create_chains!(run::CoopRun, indices) = begin
    streams = [Random.Xoshiro(rand(run.rng, UInt64)) for _ in indices]
    nodes = [initialize_progress!(run.progress; description="$(run.description).$i") for i in indices]
    created = Vector{CoopChain}(undef, length(indices))
    _parallel_foreach(eachindex(indices), run.cfg.n_cores) do j
        i, rng = indices[j], streams[j]
        family = run.lpdf isa CoopFamily
        source = family ? run.lpdf.source[i] : 0
        waiting = !iszero(source)
        target = deepcopy(_coop_target(run.lpdf, i))
        init = waiting ? nothing : _coop_cold_init(run.init, i, rng)
        master = run.reparam_lpdf
        chain = _coop_chain(i, rng, target, init, run.cfg; progress=nodes[j], waiting,
            adapt_centering=!isnothing(master),
            current_sources=isnothing(master) ? nothing : reparam_sources(master))
        if family
            chain.freeze.source = source
            chain.freeze.wave = waiting ? 2 : 1
            chain.freeze.pilot_quota = !waiting && run.cfg.two_waves ? _COOP_PILOT_DRAWS : 0
        end
        created[j] = chain
    end
    append!(run.chains, created)
    created
end

_parallel_foreach(f, items, n_workers) = begin
    items = collect(items)
    (n_workers <= 1 || Threads.nthreads() == 1 || length(items) <= 1) && return foreach(f, items)
    queue = Channel{eltype(items)}(length(items))
    foreach(i -> put!(queue, i), items)
    close(queue)
    @sync for _ in 1:min(n_workers, length(items))
        Threads.@spawn for i in queue
            f(i)
        end
    end
    nothing
end

_stop_requested(run::CoopRun) =
    _interrupted(run.progress) ||
    run.total_evaluations[] >= run.cfg.n_evaluations_budget ||
    _elapsed(run) >= run.cfg.time_budget

# One round: every running chain takes transitions until it has spent
# `n_evaluations` this round, holds `n_draws` draws, or the run must stop.
_run_round!(run::CoopRun) = begin
    cfg = run.cfg
    chains = _running_chains(run)
    foreach(c -> (c.round_evaluations = 0), chains)
    unfinished = chains
    while !isempty(unfinished) && !_stop_requested(run)
        _parallel_foreach(unfinished, cfg.n_cores) do chain
            update_progress!(chain.progress, "$(chain.status) · batch $(run.batch)")
            while _running(chain) && !chain.freeze.pending &&
                    chain.round_evaluations < cfg.n_evaluations && !_stop_requested(run)
                _coop_transition!(chain, cfg, run.total_evaluations)
            end
        end
        any(c -> c.freeze.pending, chains) || break
        _qualify_freezes!(run.chains)
        unfinished = filter(c -> _running(c) && c.round_evaluations < cfg.n_evaluations, chains)
    end
    # As at an adaptive window boundary (`_check_current_point`): every chain that
    # sampled in this round, including one that finished in it, is evaluated once
    # more at its current point, serially at the barrier.
    for chain in chains
        _check_current_point(chain.lpdf, chain.position_and_gradient,
            "cooperative_warmup_mcmc, chain $(chain.chain_index) at the end of round $(run.round)")
    end
end

# Rounds of batch k: 2^(k-1). Batch k ends after round 2^k - 1.
_batch_ends_after(round) = ispow2(round + 1)

include("cooperative_recovery.jl")

# Each running chain's restart, decided from its cluster-pooled estimate, all from
# the same pre-restart state.
_batch_end!(run::CoopRun) = begin
    cfg = run.cfg
    geometry = _cluster_geometry(run.clusters, run.chains, cfg; within_target=cfg.family)
    recovery = _coop_recovery_context(run, run.lpdf)
    rescue_factors = Dict{Any,Union{Nothing,ScaleThenReflect{Float64}}}()
    decide(i, chain) = begin
        _running(chain) && _coop_geometry_current(chain) || return nothing
        action = _coop_recovery_action(recovery, chain, run.batch)
        cfg.family && chain.status === :sampling && action.kind === :none && return nothing
        donor = action.kind === :rescue ? _coop_rescue_geometry(recovery, chain) : nothing
        metric = isnothing(donor) ? _cluster_metric(geometry, i) :
            _rescue_metric(geometry, i, donor, cfg, rescue_factors)
        (; scale, weight, relative_condition) = metric
        sc = _stepsize_gain(chain, cfg)
        restart = action.kind === :rescue ||
            (action.kind === :none && weight >= 2 && relative_condition * sc > cfg.restart_threshold)
        (; chain, new_scales=scale, cc=relative_condition, sc, action, donor, restart)
    end
    decisions = [decide(i, c) for (i, c) in enumerate(run.chains)]
    running = filter(!isnothing, decisions)
    # Refit the shared reparametrization only when every running chain restarts
    # anyway, so no kept draw is ever left in a stale frame. After a change the
    # chains re-cluster in the new frame and take their metrics from their
    # transported halos, as `adaptive_warmup_mcmc` does after its refit.
    if !isnothing(run.reparam_lpdf) && !isempty(running) && all(d -> d.restart, running) &&
            all(c -> !_running(c) || _coop_geometry_current(c), run.chains) &&
            !isnothing(_refit_reparametrization!(run))
        run.clusters = CoopClusters()
        _update_clusters!(run.clusters, run.chains, cfg)
        transported = _cluster_geometry(run.clusters, run.chains, cfg; within_target=cfg.family)
        decisions = map(eachindex(decisions)) do i
            d = decisions[i]
            isnothing(d) && return nothing
            # A change of coordinates supersedes a rescue proposed in the
            # previous frame. Install the new metric and discard that prior.
            action = d.action.kind === :rescue ? (; kind=:refit, d.action.reason, extreme=false) : d.action
            merge(d, (; new_scales=_cluster_metric(transported, i).scale, action,
                       donor=nothing, centering_refit=true))
        end
    end
    for d in decisions
        isnothing(d) && continue
        chain = d.chain
        push!(chain.batches, (; batch=run.batch, chain.generation, status=chain.status, relative_condition=d.cc,
            stepsize_gain=d.sc, restart=d.restart, centering_refit=get(d, :centering_refit, false),
            recovery=d.action.kind, recovery_reason=d.action.reason, extreme=d.action.extreme,
            n_transitions=chain.current_transition_counter, n_divergent=chain.n_divergent, n_samples=chain.n_samples,
            total_evaluation_counter=chain.total_evaluation_counter, stepsize=chain.stepsize,
            min_ess=chain.monitor_ess && chain.n_samples > 10 ? _min_ess(chain.ess) : NaN))
        if d.action.kind === :replace
            _replace_coop_chain!(run, chain, d.action.reason)
        elseif d.restart
            _restart!(chain, d.new_scales, d.cc, cfg)
            if d.action.kind === :rescue
                chain.rescue_prior = chain.kappa_prior = chain.log_kappa = d.donor.log_kappa
                chain.stepsize = _kappa_stepsize(chain.log_kappa, cfg.target_acceptance_rate)
            end
        end
    end
    # Keep the halo ring (centering evidence) spanning one batch of the next size.
    next_batch_evaluations = cfg.n_evaluations * 2^run.batch
    for c in run.chains
        c.recording_lpdf.recorder.thin = max(1, next_batch_evaluations ÷ cfg.recording_target)
    end
    run.batch += 1
end

# Refit the shared reparametrization from the running chains' evidence
# on the master copy. On a change, restore the new
# controls onto every running chain, move its current point into the new frame,
# and refill its moments from its transported halo. A transported point that is
# not finite rolls the whole refit back. Returns the new controls, or `nothing`
# when they did not change.
_refit_reparametrization!(run::CoopRun) = begin
    master = run.reparam_lpdf
    chains = _running_chains(run)
    isempty(chains) && return nothing
    all(_coop_geometry_current, chains) || return nothing
    recorder = deepcopy(first(chains).nonlinear_recorder)
    columns = [size(c.recording_lpdf.halo_position, 2) for c in chains]
    if _effective_nonlinear_evidence(recorder, master) === :linear_pool
        sum(columns) > 0 || return nothing
    else
        rng = isnothing(_leaf_sample(recorder.online)) ? nothing : Random.Xoshiro(rand(run.rng, UInt64))
        for c in Iterators.drop(chains, 1)
            _merge_nonlinear!(recorder.online, c.nonlinear_recorder.online, rng)
        end
    end
    before = deepcopy(reparam_sources(master))
    old_ir = snapshot_reparametrization(reparametrizer(master))
    halo_position = reduce(hcat, [Matrix(c.recording_lpdf.halo_position) for c in chains])
    halo_gradient = reduce(hcat, [Matrix(c.recording_lpdf.halo_gradient) for c in chains])
    first_point = find_reparametrization!(master, recorder, halo_position, halo_gradient, first(chains).position_and_gradient)
    after = reparam_sources(master)
    after == before && return nothing
    points = [first_point]
    for c in Iterators.drop(chains, 1)
        pg, rejection = _checked_centering_point(master, old_ir, c.position_and_gradient)
        if !isnothing(rejection)
            restore_reparam_sources!(master, before)
            @warn "Rejected the pooled centering update: $(rejection.reason)" chain=c.chain_index rejection.fields...
            return nothing
        end
        push!(points, pg)
    end
    offset = 0
    for (c, pg, n) in zip(chains, points, columns)
        restore_reparam_sources!(c.lpdf, after)
        c.position_and_gradient = pg
        cols = offset+1:offset+n
        c.recording_lpdf.halo_position .= view(halo_position, :, cols)
        c.recording_lpdf.halo_gradient .= view(halo_gradient, :, cols)
        _reset_moments!(c.moments)
        for j in cols
            _fit_moments!(c.moments, view(halo_position, :, j), view(halo_gradient, :, j))
        end
        offset += n
    end
    # A finished chain keeps its own frame for its draws; its moments, in the old
    # frame, no longer pool.
    for c in run.chains
        _running(c) || (c.geometry_current = false)
    end
    after
end

# =============================================================================
# Stopping and ESS
# =============================================================================

# The pooled ESS of the best subset of at least `min_chains` chains holding more
# than 10 draws: rank by per-chain minimum ESS, then truncate to the shortest
# tail (`_stack_chain_tails`). Subsets of doubling size are tried, so this stays
# O(log chains) pooled ESS evaluations. Return original chain indices.
_best_pooled_ess(draws, min_chains; individual=nothing) = begin
    usable = findall(d -> size(d, 2) > 10, draws)
    length(usable) >= min_chains || return (0.0, Int[])
    scores = map(usable) do i
        e = _min_ess(isnothing(individual) ? _coordinate_ess(draws[i]) : individual[i])
        isfinite(e) ? e : -Inf
    end
    usable = usable[sortperm(scores; rev=true)]
    best_ess, best_k = 0.0, 0
    k = min_chains
    while true
        stacked = _stack_chain_tails(view(draws, view(usable, 1:k)); min_chain_draws=11)
        if !isnothing(stacked)
            e = _min_ess(vec(MCMCDiagnosticTools.ess(stacked)))
            if isfinite(e) && e > best_ess
                best_ess, best_k = e, k
            end
        end
        k == length(usable) && break
        k = min(2k, length(usable))
    end
    best_ess, usable[1:best_k]
end

_min_pooled_chains(run::CoopRun) = min(2, run.cfg.n_chains)

# The stop must diagnose the same coordinates as chain_result, without applying
# its terminal, in-place transform to the live chain's retained history.
function _model_chain_draws(chain::CoopChain)
    draws = chain_draws(chain)
    has_reparametrization(reparametrizer(chain.lpdf)) || return draws
    model_draws = copy(draws)
    reparametrize!(chain.lpdf, model_draws)
    model_draws
end

# Per-chain monitored ESS and the stop decision, at a round barrier. The stop
# reason is recorded HERE, where the decision is made.
_check_stop!(run::CoopRun) = begin
    cfg = run.cfg
    for c in run.chains
        c.monitor_ess && c.n_samples > 10 && (c.ess .= _coordinate_ess(chain_draws(c)))
    end
    reason = any(c -> c.status === :recovery_failed, run.chains) ? :recovery_failed :
        _interrupted(run.progress) ? :interrupted :
        run.total_evaluations[] >= cfg.n_evaluations_budget ? :eval_budget :
        _elapsed(run) >= cfg.time_budget ? :time_budget : nothing
    if isnothing(reason) && !(run.lpdf isa CoopFamily) && isfinite(cfg.target_ess) && !any(_awaiting_recovery_draws, run.chains)
        holding = [c for c in run.chains if c.status !== :warming && c.n_samples > 10]
        proxy = sum(c -> _min_ess(c.ess), holding; init=0.0)
        if proxy >= cfg.target_ess && length(holding) >= _min_pooled_chains(run)
            # A subset can supply the requested ESS, but cannot vouch for a
            # different chain or for a transient outside its common tail.
            draws = map(_model_chain_draws, run.chains)
            run.last_pooled_ess = first(_best_pooled_ess(draws, _min_pooled_chains(run)))
            if run.last_pooled_ess >= cfg.target_ess && _retained_rhat(draws) < cfg.rhat_threshold
                reason = :target_ess
            end
        end
    end
    isnothing(reason) && all(c -> c.status === :done, run.chains) && (reason = :n_draws)
    run.stop_reason = reason
    !isnothing(reason)
end

# =============================================================================
# Progress
# =============================================================================

_coop_chains_label(run::CoopRun) = begin
    n(status) = count(c -> c.status === status, run.chains)
    parts = ["$(n(s)) $(s)" for s in (:sampling, :warming, :waiting, :needs_replacement, :recovery_failed, :done) if n(s) > 0]
    k = length(run.clusters.centers)
    "$(length(run.chains)) chains: " * join(parts, ", ") * "; $k cluster" * (k == 1 ? "" : "s")
end

_coop_batch_label(run::CoopRun) = begin
    rounds = 2^(run.batch - 1)
    "$(run.batch): $rounds round$(rounds == 1 ? "" : "s") of $(short_string(run.cfg.n_evaluations)) evaluations per chain"
end

_coop_pooled_ess_label(run::CoopRun) = run.lpdf isa CoopFamily ? "separate targets" : isfinite(run.cfg.target_ess) ?
    (isnan(run.last_pooled_ess) ? "not computed yet; target $(short_string(run.cfg.target_ess))" :
        "$(short_string(run.last_pooled_ess)) at the last check; target $(short_string(run.cfg.target_ess))") :
    "no target_ess"

_coop_evaluations_label(run::CoopRun) = begin
    n = run.total_evaluations[]
    rate = _rate_or_pace(n, _elapsed(run))
    run.cfg.n_evaluations_budget == typemax(Int) ? "$(short_string(n)) ($rate), no budget" :
        "$(short_string(n)) of $(short_string(run.cfg.n_evaluations_budget)) ($rate)"
end

_coop_seconds(seconds) = short_duration(Dates.Millisecond(round(Int, 1000 * seconds)))
_coop_elapsed_label(run::CoopRun) = isfinite(run.cfg.time_budget) ?
    "$(_coop_seconds(_elapsed(run))) of $(_coop_seconds(run.cfg.time_budget))" :
    "$(_coop_seconds(_elapsed(run))), no time budget"

# The run's labels, made before any chain node so they render first; every later
# `_run_progress!` updates them in place.
_initial_run_labels!(::Nothing, cfg) = nothing
_initial_run_labels!(progress, cfg) = update_progress!(progress, nothing;
    chains="initializing $(cfg.n_chains) chains", batch="not started", pooled_ess="not computed yet",
    evaluations="0", elapsed="0s")

_run_progress!(run::CoopRun) = begin
    isnothing(run.progress) && return nothing
    update_progress!(run.progress, nothing;
        chains=_coop_chains_label(run),
        batch=_coop_batch_label(run),
        pooled_ess=_coop_pooled_ess_label(run),
        evaluations=_coop_evaluations_label(run),
        elapsed=_coop_elapsed_label(run),
    )
end

_chain_progress_summary(chain::CoopChain) = chain.n_samples == 0 ? "no draws retained" :
    "$(chain.n_samples) draws, " *
    (chain.monitor_ess && chain.n_samples > 10 ? "min. ESS: $(short_string(_min_ess(chain.ess))), " : "") *
    "divergent: $(short_string(100 * chain.n_divergent_samples / chain.n_samples))%"

_finish_chain_progress!(chain::CoopChain, what) = begin
    isnothing(chain.progress) && return nothing
    _transition_progress!(chain.progress, nothing, chain)
    chain.monitor_ess && (chain.n_samples > 10 ? _ess_progress!(chain.progress, chain) :
        update_progress!(chain.progress, nothing; ess="no estimate from $(chain.n_samples) draws"))
    update_progress!(chain.progress, "$what: $(_chain_progress_summary(chain))")
    finalize_progress!(chain.progress)
end

_round_progress!(run::CoopRun) = begin
    _run_progress!(run)
    for c in run.chains
        c.monitor_ess && c.n_samples > 10 && _ess_progress!(c.progress, c)
    end
end

# =============================================================================
# Result
# =============================================================================

"""
    chain_result(chain) -> NamedTuple

One chain's part of a [`cooperative_warmup_mcmc`](@ref) result. Terminal: it
back-transforms the draws in place to the model frame, so call it once.
"""
chain_result(chain::CoopChain) = begin
    draws = chain_draws(chain)
    size(draws, 2) > 0 && reparametrize!(chain.lpdf, draws)
    (;
        chain_index=chain.chain_index,
        posterior_position=draws,
        posterior_gradient=chain.recording_lpdf.posterior_gradient,
        ess=_returned_ess(chain.ess, draws, chain.monitor_ess),
        stepsize=chain.stepsize,
        scale=chain.scale,
        status=chain.status,
        n_samples=chain.n_samples,
        n_divergent_samples=chain.n_divergent_samples,
        total_evaluation_counter=chain.total_evaluation_counter,
        n_restarts=chain.n_restarts,
        accounting=(; wave=chain.freeze.wave, source=chain.freeze.source,
            warmup_transitions=chain.freeze.warmup_transitions, divergences=chain.freeze.divergences,
            freeze_extensions=chain.freeze.total_extensions, pilot_draws=length(chain.freeze.pilot),
            abandoned_freeze_scores=chain.freeze.abandoned_scores,
            rescues=chain.freeze.prior_rescues + _coop_rescue_count(chain.batches),
            replacements=chain.generation - 1,
            gradient_evaluations=chain.total_evaluation_counter),
        batches=chain.batches,
        chain.generation, chain.abandoned,
    )
end

_finalize(run::CoopRun) = begin
    for c in run.chains
        _finish_chain_progress!(c, c.status === :done ? "done" : "$(c.status) when the run stopped")
    end
    results = map(run.chains) do chain
        r = chain_result(chain)
        if run.lpdf isa CoopFamily
            id = isnothing(run.lpdf.ids) ? chain.chain_index : run.lpdf.ids[chain.chain_index]
            return merge(r, (; target_id=id))
        end
        r
    end
    individual = run.cfg.monitor_ess ? [r.ess for r in results] : nothing
    joint_ess, pooled_chains = run.lpdf isa CoopFamily ? (nothing, Int[]) :
        _best_pooled_ess([r.posterior_position for r in results], _min_pooled_chains(run); individual)
    (;
        results,
        stop_reason=run.stop_reason,
        n_used=count(r -> r.n_samples > 0, results),
        joint_ess,
        n_pooled=length(pooled_chains),
        n_batches=run.batch - 1,
        total_evaluation_counter=run.total_evaluations[],
        elapsed=_elapsed(run),
        pooled_chains,
    )
end

_finish_run_progress!(run::CoopRun, result) = begin
    isnothing(run.progress) && return nothing
    _run_progress!(run)
    n = sum(r -> r.n_samples, result.results; init=0)
    divergent = n == 0 ? "no draws retained" :
        "divergent: $(short_string(100 * sum(r -> r.n_divergent_samples, result.results) / n))%"
    ess = isnothing(result.joint_ess) ? "" : "joint ESS $(short_string(result.joint_ess)) over $(result.n_pooled) chains, "
    update_progress!(run.progress,
        "$(result.stop_reason): $ess" *
        "$(result.n_used) of $(length(result.results)) chains hold draws, $divergent")
end

# =============================================================================
# Checkpoints and resume
# =============================================================================
#
# With `checkpoint_dir`, every round barrier writes
#   run_latest.jls           the whole run (every chain, the seed RNG, the clusters),
#                            which is all `resume=true` reads, so it never sees a
#                            torn mix of chain files;
#   chain_<i>/cp_latest.jls  this chain's payload, for readers of a live run;
# and every batch end also writes the immutable chain_<i>/cp_window_<batch>.jls.
# The chain payload keeps the keys the other samplers' readers use
# (`posterior_position`, `dropped_*`, `status`, `is_final`, `scale_options` /
# `active_transformation`, `reparam_sources`, ...), so `FixedKernelNUTSState`,
# `back_transform` and the consumer readers of warmuphmc-use §3 read it as before.
# run_manifest.json (at the start) and run_summary.json (at the finish) are
# write-once; the summary's existence marks a finished run.

_coop_scale_payload(L::Diagonal) = (; scale_options=(; diagonal=L), active_transformation=:diagonal)
_coop_scale_payload(L::ScaleThenReflect) = (; scale_options=(; adaptive=L), active_transformation=:adaptive)

function _coop_checkpoint_version(chain::CoopChain)
    streaming = !isnothing(chain.nonlinear_recorder) &&
        _effective_nonlinear_evidence(chain.nonlinear_recorder, chain.lpdf) !== :linear_pool
    changed = _has_coop_recovery(chain) || !(chain.scale isa Diagonal) || !chain.geometry_current ||
        chain.freeze.pending || chain.freeze.extensions > 0 || chain.status === :waiting || streaming
    changed ? checkpoint_schema_version(Val(:cooperative)) : checkpoint_schema_version()
end

_coop_run_version(run) = run.lpdf isa CoopFamily ? checkpoint_schema_version(Val(:cooperative)) :
    maximum(_coop_checkpoint_version, run.chains; init=checkpoint_schema_version())

# Built-in container types let earlier readers inspect the sampler and schema
# before deciding whether they can resume this checkpoint.
_freeze_payload(f::CoopFreeze) = (; (k => getfield(f, k) for k in fieldnames(CoopFreeze))...)

function _restore_freeze(payload, warmup)
    state = CoopFreeze(warmup)
    saved = get(payload, :freeze, nothing)
    for k in fieldnames(CoopFreeze)
        hasproperty(saved, k) && setproperty!(state, k, getproperty(saved, k))
    end
    state
end

_coop_payload(chain::CoopChain, run::CoopRun) = (;
    schema_version=run.lpdf isa CoopFamily ? checkpoint_schema_version(Val(:cooperative)) :
        _coop_checkpoint_version(chain),
    sampler=:cooperative,
    chain_index=chain.chain_index,
    (run.lpdf isa CoopFamily ? (; target_id=run.lpdf.ids[chain.chain_index]) : (;))...,
    chain.generation, chain.abandoned, chain.rescue_prior,
    status=chain.status,
    stuck_reason=chain.status === :recovery_failed ? last(chain.abandoned).reason : nothing,
    is_final=chain.status in (:done, :recovery_failed),
    stop_reason=chain.status === :done ? :n_draws : chain.status === :recovery_failed ? :recovery_failed : :running,
    window=run.batch - 1,
    round=run.round,
    dimension=chain.dimension,
    chain.rng, chain.position_and_gradient, chain.stepsize,
    _coop_scale_payload(chain.scale)...,
    chain.log_kappa, chain.kappa_prior, chain.backoff, chain.stepsizes, chain.acceptances,
    freeze=_freeze_payload(chain.freeze), chain.sampling_start, chain.moments, chain.current_transition_counter,
    chain.total_transition_counter, chain.total_evaluation_counter, chain.n_divergent,
    chain.n_divergent_samples, chain.n_samples, chain.steps_per_draw, chain.ess,
    chain.scale_changes, chain.n_restarts, chain.batches,
    posterior_position=chain.recording_lpdf.posterior_position,
    posterior_gradient=chain.recording_lpdf.posterior_gradient,
    halo_position=chain.recording_lpdf.halo_position,
    halo_gradient=chain.recording_lpdf.halo_gradient,
    recorder=chain.recording_lpdf.recorder,
    chain.nonlinear_recorder, chain.geometry_current,
    chain.dropped_posterior_position, chain.dropped_posterior_gradient,
    chain.dropped_n_divergent_samples,
    reparam_sources=reparam_sources(chain.lpdf),
    custom_candidate_scoring=_has_custom_candidate_scoring(chain.lpdf),
    run_elapsed=_elapsed(run),
)

_write_checkpoints!(::CoopRun, ::Nothing; batch_end) = nothing
_write_checkpoints!(run::CoopRun, dir::AbstractString; batch_end) = begin
    payloads = [_coop_payload(c, run) for c in run.chains]
    for p in payloads
        chain_dir = joinpath(dir, "chain_$(p.chain_index)")
        mkpath(chain_dir)
        batch_end && _atomic_serialize(joinpath(chain_dir, "cp_window_$(p.window).jls"), p)
        _atomic_serialize(joinpath(chain_dir, "cp_latest.jls"), p)
    end
    _atomic_serialize(joinpath(dir, "run_latest.jls"), (;
        schema_version=maximum(p -> p.schema_version, payloads), sampler=:cooperative,
        run.round, run.batch, total_evaluations=run.total_evaluations[], run.rng,
        family=_family_checkpoint(run.lpdf),
        replacement_init=_checkpoint_cold_init(run.init),
        run.clusters, chains=payloads, run.last_pooled_ess, run_elapsed=_elapsed(run),
        master_sources=isnothing(run.reparam_lpdf) ? nothing : reparam_sources(run.reparam_lpdf),
    ))
end

_write_coop_manifest(::Nothing, run) = nothing
_write_coop_manifest(dir::AbstractString, run::CoopRun) = _write_json(
    joinpath(dir, "run_manifest.json"), Pair{String,Any}[
        "schema_version" => _coop_run_version(run),
        "sampler" => "cooperative",
        "n_chains" => run.cfg.n_chains,
        (run.lpdf isa CoopFamily ? ["target_ids" => run.lpdf.ids] : Pair{String,Any}[])...,
        "n_cores" => run.cfg.n_cores,
        # The stopping criteria in force; JSON has no Inf, so an unset bound is null.
        "n_draws" => run.cfg.n_draws == typemax(Int) ? nothing : run.cfg.n_draws,
        "target_ess" => run.cfg.target_ess,
        "rhat_threshold" => run.cfg.rhat_threshold,
        "n_evaluations_budget" => run.cfg.n_evaluations_budget == typemax(Int) ? nothing : run.cfg.n_evaluations_budget,
        "time_budget" => run.cfg.time_budget,
        "n_evaluations" => run.cfg.n_evaluations,
        # Xoshiro prints its full state: the reproducible seed, not a label.
        "seed_rng" => string(run.rng),
        "start_time_unix" => time(),
    ])

_write_coop_summary(::Nothing, run, result) = nothing
_write_coop_summary(dir::AbstractString, run::CoopRun, result) = _write_json(
    joinpath(dir, "run_summary.json"), Pair{String,Any}[
        "schema_version" => _coop_run_version(run),
        "sampler" => "cooperative",
        "is_final" => true,
        "stop_reason" => string(result.stop_reason),
        "n_chains" => length(result.results),
        "n_used" => result.n_used,
        "joint_ess" => result.joint_ess,
        "n_pooled" => result.n_pooled,
        "pooled_chains" => result.pooled_chains,
        "chain_generations" => [r.generation for r in result.results],
        "failed_chains" => [r.chain_index for r in result.results if r.status === :recovery_failed],
        "n_batches" => result.n_batches,
        "total_evaluation_counter" => result.total_evaluation_counter,
        "elapsed" => result.elapsed,
        "end_time_unix" => time(),
    ])

# A chain from its payload and a fresh `lpdf`; the configuration comes from the
# resuming call. A finished chain resumes sampling when the call raises `n_draws`.
_restore_coop_chain(p, lpdf, cfg; progress, adapt_centering, geometry_current) = begin
    check_checkpoint_compatible(p, :cooperative, (:cooperative,))
    _check_schema_version(p)
    d = LogDensityProblems.dimension(lpdf)
    p.dimension == d || throw(DimensionMismatch(
        "the checkpoint holds a $(p.dimension)-dimensional problem but the supplied lpdf has dimension $d."))
    _check_candidate_scoring_compatible(p, lpdf)
    restore_reparam_sources!(lpdf, p.reparam_sources)
    nonlinear_recorder = adapt_centering ? _coop_nonlinear_recorder(
        p, lpdf, p.recorder.target, cfg.nonlinear_settings, p.rng) : nothing
    recording_lpdf = RecordingPosterior2(lpdf, p.halo_position, p.halo_gradient,
        p.posterior_position, p.posterior_gradient, NUTSLeaves(d), p.recorder, p.rng)
    freeze = _restore_freeze(p, cfg.stepsize_adaptation_limit)
    status = p.status in (:done, :sampling) ?
        (p.n_samples >= cfg.n_draws && length(freeze.pilot) >= freeze.pilot_quota ? :done : :sampling) : p.status
    scale = p.scale_options[p.active_transformation]
    chain = CoopChain(
        p.chain_index, p.rng, lpdf, recording_lpdf, nonlinear_recorder, geometry_current,
        d, cfg.monitor_ess, time_ns(), progress,
        p.position_and_gradient, scale, _coop_energy(scale), p.stepsize, p.log_kappa, p.kappa_prior,
        p.backoff, p.stepsizes, p.acceptances, p.sampling_start, p.moments, status,
        p.current_transition_counter, p.total_transition_counter, p.total_evaluation_counter, 0,
        p.n_divergent, p.n_divergent_samples, p.n_samples, p.steps_per_draw, p.ess,
        p.scale_changes, p.n_restarts, p.batches, p.dropped_posterior_position,
        p.dropped_posterior_gradient, p.dropped_n_divergent_samples, freeze,
        get(p, :generation, 1), get(p, :abandoned, NamedTuple[]), get(p, :rescue_prior, nothing),
    )
    _initial_chain_progress!(progress, nothing, chain)
    update_progress!(progress, "$(chain.status) · restored at batch $(p.window + 1)")
    chain
end

_restore_coop_run(dir, lpdf, cfg, init, progress, description, adapt_centering) = begin
    state = deserialize(joinpath(dir, "run_latest.jls"))
    check_checkpoint_compatible(state, :cooperative, (:cooperative,))
    _check_schema_version(state)
    length(state.chains) == cfg.n_chains || throw(ArgumentError(
        "resume needs the run's n_chains = $(length(state.chains)); got n_chains = $(cfg.n_chains)."))
    isequal(get(state, :family, nothing), _family_checkpoint(lpdf)) ||
        throw(ArgumentError("resume needs the checkpoint's ordered targets, features and first wave"))
    nodes = [initialize_progress!(progress; description="$description.$(p.chain_index)") for p in state.chains]
    controls = if isnothing(state.master_sources)
        first(p.reparam_sources for p in state.chains if get(p, :geometry_current, true))
    else
        state.master_sources
    end
    chains = [_restore_coop_chain(p, deepcopy(_coop_target(lpdf, p.chain_index)), cfg; progress=node, adapt_centering,
                geometry_current=get(p, :geometry_current, cfg.family || p.reparam_sources == controls))
              for (p, node) in zip(state.chains, nodes)]
    master = adapt_centering ? restore_reparam_sources!(deepcopy(lpdf), controls) : nothing
    cold_init = ismissing(init) ? get(state, :replacement_init, missing) : init
    stop_reason = any(c -> c.status === :recovery_failed, chains) ? :recovery_failed : nothing
    CoopRun(state.rng, lpdf, cfg, cold_init, progress, string(description), time_ns(), state.run_elapsed, master,
        CoopChain[chains...], state.clusters, Threads.Atomic{Int}(state.total_evaluations),
        state.round, state.batch, stop_reason, state.last_pooled_ess)
end

# =============================================================================
# Entry point
# =============================================================================

"""
    cooperative_warmup_mcmc(rng, lpdf; n_chains=4, n_cores=n_chains, target_ess=Inf,
        rhat_threshold=1.01, n_evaluations_budget=typemax(Int), time_budget=Inf,
        n_draws=typemax(Int), kwargs...)

Cooperative multi-chain NUTS warm-up and sampling on one target: `n_chains`
chains adapt TOGETHER and keep sampling until a stopping bound is met.

Every leapfrog state of every chain feeds running position and gradient moments.
Chains pool them through soft clusters (chains near each other in the posterior
share; once they have mixed, all share one estimate), and each chain's metric is
fitted from its dominant cluster's recent position and gradient pairs. Each cluster
takes at most `recording_target` leaves, apportioned by responsibility, and fits
up to `max_reflections` Householder reflections (default 16). With too little
evidence or no retained reflection, a chain uses its pooled Nutpie diagonal.
`max_reflections=0` selects diagonal-only adaptation. Work proceeds in rounds of
`n_evaluations` gradient evaluations per chain; batch k is 2^(k-1) rounds. At a
batch end a chain normally restarts warm-up under its pooled estimate when that
estimate's relative condition number against its current metric, times the
predicted step-size gain, exceeds `restart_threshold` (2). A restart drops the
chain's draws; otherwise it keeps sampling under its fixed kernel, so every kept
draw is an MCMC draw of an unchanging kernel. Step sizes follow a per-chain fit
of acceptance ≈ 2Φ(-κε²) with the pooled κ as prior.

A poorly sampling singleton may restart at its own position with the majority
cluster's geometry and step-size prior. Persistent extreme divergence or step-size
collapse at consecutive batch ends with a rescue between them starts a fresh
generation at the same index, using the slot's original init and a new seed-RNG
stream. Each slot gets one replacement. Exhaustion stops the run with
`:recovery_failed` and zero retained draws for the failed slot. Each result
records its `generation` (starting at one) and `abandoned` attempt summaries.

* `rng` seeds every chain: chain i's stream is drawn from it when chain i is
  created, so one seed reproduces the run at any thread count.
* Stops on `target_ess` when the pooled ESS of at least `min(2, n_chains)`
  chains reaches it AND the maximum rank-normalized split R-hat over every
  retained history is below `rhat_threshold` (default 1.01, finite and > 1). Both checks use
  the returned model frame. R-hat uses all draws, even with unequal history
  lengths: split-chain variances and means have equal weight, and odd histories
  retain their middle draw. The maximum includes both ordinary rank-normalized
  values and ranks of absolute deviations from the pooled median (folding).
  Every nonempty chain needs more than 10 draws;
  an empty chain that has been rescued or replaced must sample again before an
  ESS stop. Other empty chains do not contribute. This diagnostic cannot detect
  an unvisited mode.
* Also stops when `n_evaluations_budget` gradient evaluations are spent,
  `time_budget` seconds pass, every chain holds `n_draws` draws, or
  `Treebars.request_interrupt!` is called on `progress`, all checked at round
  barriers (a budget or an interrupt also stops a round early). At least one
  bound must be finite. The returned `stop_reason` names the one that fired.
  The `target_ess` stop needs `monitor_ess` (the default).
* `init` (as for [`adaptive_warmup_mcmc`](@ref)) is shared by every chain, or a
  vector of per-chain inits; a position vector of reals is one shared start.
* With `nonlinear_adapt=true`, every running chain contributes to one shared
  centering refit at a batch end where all of them already need to restart.
  `nonlinear_evidence=:linear_pool` uses their combined retained halos.
  A [`CandidateScoringPlan`](@ref) scores `:all_good_leaves` online even
  under that default. `:all_good_leaves` uses noninitial leaves above
  `nonlinear_good_leaf_threshold=log(1e-2)`. `:nuts_weighted` uses the
  NUTS proposal weights. `nonlinear_trajectory_weighting=:unit` gives no
  additional trajectory weight, and `:stepsize` multiplies by its step size.
  A [`WindowSelectionPlan`](@ref) receives the combined halo in linear mode
  or a weighted sample of `recording_target` columns in either streaming mode.
* `blas_threads=1` holds BLAS at that many threads for the whole call and
  restores the caller's count afterwards, as for
  [`adaptive_warmup_mcmc`](@ref); `nothing` leaves BLAS untouched.

## Checkpointing and resume

`checkpoint_dir=path` writes, at every round barrier, `path/run_latest.jls` (the
whole run) and `path/chain_<i>/cp_latest.jls`, plus `path/chain_<i>/cp_window_<k>.jls`
at the end of batch k, all atomically; `run_manifest.json` at the start and
`run_summary.json` at the finish are write-once. A later call with `resume=true`
continues an unfinished run from its last round barrier: every chain's state,
draws and stream, the seed RNG, the clusters and the counters are restored, so
the run continues as if it had not stopped. The configuration comes from the
resuming call (a larger `n_draws` extends finished chains; `n_chains` must
match). The nonlinear recorder resumes with its accumulated evidence and
random stream. Changing its mode, weighting or threshold starts fresh evidence.
Older checkpoints use the original halo default.
Numeric starts and explicit positions with squared scales are saved
for future replacements. Supply custom initializers again on resume. The
evaluation budget continues from the restored total, and
`time_budget` from the run's live time. A finished run (`run_summary.json`
exists) is refused; `overwrite=true` discards the directory's run.

Returns `(; results, stop_reason, n_used, joint_ess, n_pooled, n_batches,
total_evaluation_counter, elapsed, pooled_chains)`. `results[i]` is chain `i`'s
`chain_result`, its draws in the model frame. `joint_ess` is the pooled
ESS of the best tested prefix ranked by individual minimum ESS, over a common
tail of at least `min(2, n_chains)` chains. `pooled_chains` gives their indices.
"""
cooperative_warmup_mcmc(rng::Random.AbstractRNG, lpdf;
    n_chains::Integer=4,
    n_cores::Integer=n_chains,
    target_ess=Inf,
    rhat_threshold=1.01,
    n_evaluations_budget=typemax(Int),
    time_budget=Inf,
    n_draws=typemax(Int),
    n_evaluations=1000,
    stepsize_adaptation_limit=50,
    target_acceptance_rate=0.8,
    max_tree_depth=10,
    restart_threshold=2.0,
    max_reflections::Integer=_DEFAULT_MAX_REFLECTIONS,
    max_clusters=8,
    bandwidth=1.0,
    kappa_prior_weight=2.0,
    kappa_decay=0.9,
    recording_target=1000,
    init=missing,
    nonlinear_adapt=true,
    nonlinear_evidence=nothing,
    nonlinear_trajectory_weighting=nothing,
    nonlinear_good_leaf_threshold=nothing,
    monitor_ess=true,
    progress=nothing,
    description="MCMC",
    checkpoint_dir=nothing,
    resume=false,
    overwrite=false,
    pathfinder_kw=(;),
    blas_threads=_DEFAULT_BLAS_THREADS,
    kwargs...
) = _with_blas_threads(blas_threads) do
    _check_kwargs(:cooperative_warmup_mcmc, kwargs)
    family = lpdf isa CoopFamily
    any(k -> k in (:features, :n_wave1, :target_ids), keys(kwargs)) &&
        throw(ArgumentError("features, n_wave1 and target_ids apply to a vector of targets"))
    _refuse_unsafe_concurrent_gradients(:cooperative_warmup_mcmc, family ? lpdf.targets : (lpdf,),
        _concurrent_chains(n_chains, n_cores))
    guard_run_dir!(checkpoint_dir, resume, overwrite, :cooperative)
    max_reflections >= 0 || throw(ArgumentError("max_reflections must be nonnegative"))
    recording_target > 0 || throw(ArgumentError("recording_target must be positive"))
    n_evaluations > 0 || throw(ArgumentError("n_evaluations must be positive"))
    n_chains >= 1 || throw(ArgumentError("cooperative_warmup_mcmc needs n_chains ≥ 1, got $n_chains."))
    isfinite(rhat_threshold) && rhat_threshold > 1 || throw(ArgumentError(
        "cooperative_warmup_mcmc needs a finite rhat_threshold > 1, got $rhat_threshold."))
    isfinite(target_ess) || n_evaluations_budget != typemax(Int) || isfinite(time_budget) || n_draws != typemax(Int) ||
        throw(ArgumentError("cooperative_warmup_mcmc needs at least one finite stopping bound " *
            "(target_ess, n_evaluations_budget, time_budget or n_draws)."))
    # The target_ess stop is checked only once the chains' monitored ESS adds up to
    # it; unmonitored, that sum stays zero and the stop never fires.
    isfinite(target_ess) && !monitor_ess && throw(ArgumentError(
        "cooperative_warmup_mcmc's target_ess stop needs the per-chain ESS: drop `monitor_ess=false`."))
    init isa AbstractVector && !(init isa AbstractVector{<:Real}) && length(init) != n_chains &&
        throw(ArgumentError("a per-chain `init` vector needs one entry per chain: got $(length(init)) for n_chains = $n_chains."))
    _check_nonlinear_adaptable(lpdf, nonlinear_adapt)
    adapt_centering = nonlinear_adapt && has_reparametrization(reparametrizer(lpdf))
    _check_nonlinear_evidence(something(nonlinear_evidence, :linear_pool),
                              something(nonlinear_trajectory_weighting, :unit))
    nonlinear_settings = (; nonlinear_evidence, nonlinear_trajectory_weighting, nonlinear_good_leaf_threshold)
    cfg = (; family, two_waves=family && any(!iszero, lpdf.source), n_chains, n_cores=max(1, n_cores), target_ess=Float64(target_ess),
        rhat_threshold=Float64(rhat_threshold), n_evaluations_budget,
        time_budget=Float64(time_budget), n_draws, n_evaluations, stepsize_adaptation_limit,
        target_acceptance_rate, algorithm=DynamicHMC.NUTS(; max_depth=max_tree_depth),
        restart_threshold, max_reflections, max_clusters, bandwidth, kappa_prior_weight, kappa_decay,
        recording_target, monitor_ess, nonlinear_settings, initializer_kw=(; kwargs..., pathfinder_kw...))
    with_progress(progress; description) do progress
        _initial_run_labels!(progress, cfg)
        run = if resume
            _restore_coop_run(checkpoint_dir, lpdf, cfg, init, progress, description, adapt_centering)
        else
            fresh = CoopRun(rng, lpdf, cfg, init, progress, string(description), time_ns(), 0.0,
                adapt_centering ? deepcopy(lpdf) : nothing,
                CoopChain[], CoopClusters(), Threads.Atomic{Int}(0), 0, 1, nothing, NaN)
            _write_coop_manifest(checkpoint_dir, fresh)
            _create_chains!(fresh, 1:n_chains)
            _update_clusters!(fresh.clusters, fresh.chains, cfg)
            _update_kappa_priors!(fresh.clusters, fresh.chains)
            family && _write_checkpoints!(fresh, checkpoint_dir; batch_end=false)
            fresh
        end
        _run_progress!(run)
        while isnothing(run.stop_reason)
            run.round += 1
            _run_round!(run)
            _replace_gate_failures!(run)
            if run.cfg.two_waves && !_stop_requested(run)
                _start_second_wave!(run)
                _replace_gate_failures!(run)
            end
            _update_clusters!(run.clusters, run.chains, cfg)
            if _check_stop!(run)
                if run.stop_reason === :recovery_failed || (family && run.stop_reason === :n_draws)
                    _write_checkpoints!(run, checkpoint_dir; batch_end=false)
                end
                break
            end
            batch_end = _batch_ends_after(run.round)
            if batch_end
                _batch_end!(run)
                _update_clusters!(run.clusters, run.chains, cfg)
            end
            _update_kappa_priors!(run.clusters, run.chains)
            _write_checkpoints!(run, checkpoint_dir; batch_end)
            _round_progress!(run)
        end
        result = _finalize(run)
        _finish_run_progress!(run, result)
        _write_coop_summary(checkpoint_dir, run, result)
        result
    end
end

cooperative_warmup_mcmc(rngs::AbstractVector, lpdf; kwargs...) = throw(ArgumentError(
    "cooperative_warmup_mcmc takes ONE seed RNG and `n_chains`, not a vector of RNGs: " *
    "`cooperative_warmup_mcmc(Xoshiro(seed), lpdf; n_chains=$(length(rngs)), …)`. " *
    "Each chain's stream is drawn from the seed RNG when the chain is created (decision `0m8p12j`)."))
