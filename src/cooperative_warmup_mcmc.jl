# Cooperative multi-chain warm-up: the chains adapt together.
#
# This is the user's original `CooperativeIterativeSampler` (archived at
# `7be3f39^:src/cooperative_warmup_mcmc.jl`) merged with Bruno's cut sampler
# (BrunoStan `cooperative_chains`, Bruno `73c5dc06`), per the user's direction of
# 2026-10-21 22:52 and design decisions `0m327rr` / `0m8p12j`. It replaces the
# July scheduler around independent adaptive chains, whose 4000-evaluation window
# cap and geometry-stall parking returned no usable chain on ODLPlaus' AP2 (snag
# `cooperative-pool-272534b8`).
#
# From the original:
#   * every leapfrog state of every chain feeds running position and gradient
#     moments, from which the Nutpie diagonal scale
#     `(var(position) / var(gradient))^(1/4)` follows;
#   * the moments ACCUMULATE from the chain's last restart on, over batches whose
#     budget doubles with no cap;
#   * at a batch end a chain restarts only when its pooled estimate is incompatible
#     with the metric it samples under: relative condition number × step-size gain
#     > √2 (`restart_threshold`);
#   * no chain is ever parked or abandoned.
# From Bruno's sampler:
#   * the step size: a per-chain fit of κ in acceptance ≈ 2Φ(-κε²), with the pooled
#     log κ as prior and a backoff on failed transitions;
#   * the kernel weight between a chain and a pool, on the RMS standardized
#     distance between their means.
# From the user's notes on `0m327rr`:
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
_combine_moments!(out::CoopMoments, parts, weights) = begin
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
        @. out.position_m2 += w * (m.position_m2 + m.weight * abs2(m.position_mean - out.position_mean))
        @. out.gradient_m2 += w * (m.gradient_m2 + m.weight * abs2(m.gradient_mean - out.gradient_mean))
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
# Step size: acceptance ≈ (1 - π)·2Φ(-κε²) (Bruno's model, with a failure floor)
# =============================================================================
#
# Bruno's model `2Φ(-κε²)` explains every rejection by the step size. A target
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
_fit_log_kappa(stepsizes, acceptances, weights, previous, prior, prior_weight) = begin
    loss(lk) = begin
        c = _kappa_floor(lk, stepsizes, acceptances, weights)
        sum(w * abs2(a - c * _kappa_acceptance(lk, e)) for (e, a, w) in zip(stepsizes, acceptances, weights); init=0.0) +
            prior_weight * 0.05 * abs2(lk - prior)
    end
    center = (previous + prior) / 2
    half_width = 3 + abs(previous - prior) / 2
    coarse = center .+ range(-half_width, half_width; length=61)
    best = coarse[argmin(loss.(coarse))]
    fine = best .+ range(-step(coarse), step(coarse); length=21)
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
mutable struct CoopChain{R,L,RL}
    const chain_index::Int
    const rng::R
    const lpdf::L
    const recording_lpdf::RL
    const dimension::Int
    const monitor_ess::Bool
    const start_time::UInt64
    # This chain's Treebars node (`nothing` without progress); a runtime handle,
    # never checkpointed.
    const progress::Any
    position_and_gradient
    # The metric's scale `L` (`L*L'` approximates the posterior covariance) and its
    # kinetic energy.
    scale::Diagonal{Float64,Vector{Float64}}
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
end

chain_draws(chain::CoopChain) = chain.recording_lpdf.posterior_position
n_chain_draws(chain::CoopChain) = size(chain_draws(chain), 2)
_running(chain::CoopChain) = chain.status !== :done

# Build chain `index` on its own stream: deep-copied density, initialization
# (Pathfinder unless `init` says otherwise), diagonal scale from the
# initializer's squared scale, initial step size and the log κ that gives it.
_coop_chain(index, rng, lpdf, init, cfg; progress) = begin
    dimension = LogDensityProblems.dimension(lpdf)
    recording_lpdf = RecordingPosterior2(lpdf;
        recorder=LimitedRecorder2(cfg.recording_target, max(1, cfg.n_evaluations ÷ cfg.recording_target)), rng)
    (; position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, cfg.initializer_kw...)
    scale = _initial_diagonal_scale(squared_scale)
    kinetic_energy = _coop_energy(scale)
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    stepsize = DynamicHMC.find_initial_stepsize(
        DynamicHMC.InitialStepsizeSearch(),
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(kinetic_energy, lpdf),
            DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy)),
        ),
    )
    log_kappa = _kappa_log(stepsize, cfg.target_acceptance_rate)
    chain = CoopChain(
        index, rng, lpdf, recording_lpdf, dimension, cfg.monitor_ess, time_ns(), progress,
        position_and_gradient, scale, kinetic_energy, stepsize, log_kappa, log_kappa, 1.0,
        Float64[], Float64[], 0, CoopMoments(dimension), :warming,
        0, 0, 0, 0, 0, 0, 0, OnlineStatsBase.Mean(), zeros(dimension), Float64[], 0,
        NamedTuple[], Matrix{Float64}(undef, dimension, 0), Matrix{Float64}(undef, dimension, 0), 0,
    )
    _initial_chain_progress!(progress, nothing, chain)
    update_progress!(progress, "warming · initialized")
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
        chain.rng, cfg.algorithm, DynamicHMC.Hamiltonian(chain.kinetic_energy, recording_lpdf),
        chain.position_and_gradient, ε,
    )
    finalize_leaf_recording!(recording_lpdf, stats.depth)
    _fit_leaves!(chain.moments, recording_lpdf.leaves)
    steps = stats.steps
    chain.total_evaluation_counter += steps
    chain.round_evaluations += steps
    Threads.atomic_add!(total_evaluations, steps)
    chain.current_transition_counter += 1
    chain.total_transition_counter += 1
    OnlineStatsBase.fit!(chain.steps_per_draw, steps)
    divergent = DynamicHMC.is_divergent(stats.termination)
    divergent && (chain.n_divergent += 1)
    acceptance = stats.acceptance_rate
    push!(chain.stepsizes, ε)
    push!(chain.acceptances, acceptance)
    if chain.status === :warming
        # A divergence alone is not a step-size failure: leaving the support
        # diverges at any step size.
        failed = acceptance < 0.1
        chain.backoff = failed ? chain.backoff / 2 : min(1.0, 1.5chain.backoff)
        n = length(chain.acceptances)
        weights = cfg.kappa_decay .^ (n-1:-1:0)
        chain.log_kappa = _fit_log_kappa(chain.stepsizes, chain.acceptances, weights,
            chain.log_kappa, chain.kappa_prior, cfg.kappa_prior_weight)
        chain.stepsize = _kappa_stepsize(chain.log_kappa, cfg.target_acceptance_rate) * chain.backoff
        if chain.current_transition_counter >= cfg.stepsize_adaptation_limit
            chain.status = :sampling
            chain.sampling_start = n
        end
    else
        append!(recording_lpdf.posterior_position, chain.position_and_gradient.q)
        append!(recording_lpdf.posterior_gradient, chain.position_and_gradient.∇ℓq)
        divergent && (chain.n_divergent_samples += 1)
        chain.n_samples += 1
        chain.n_samples >= cfg.n_draws && (chain.status = :done)
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
# restarts on metric disagreement alone (gain 1). The original instead gave it an
# infinite gain, restarting every chain whose step-size adaptation had not
# finished. That throws away a long warm-up at every early batch end: on ODLPlaus
# AP2, 50 warm-up transitions of ~300 leapfrog steps need ~15k evaluations, more
# than any of the first four batches holds.
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
_relative_condition(new_scales, scale::Diagonal) = begin
    lo, hi = Inf, -Inf
    for (n, s) in zip(new_scales, diag(scale))
        isfinite(n) || continue
        r = n / s
        lo = min(lo, r); hi = max(hi, r)
    end
    isfinite(lo) ? hi / lo : 1.0
end

# Install `new_scales` (coordinates without an estimate keep theirs), drop the
# draws, and start warm-up again. The step size carries over by the original's
# rule, scaled by the change of the smallest scale; κ is re-based to it.
_restart!(chain::CoopChain, new_scales, cc, cfg) = begin
    old = diag(chain.scale)
    scales = [isfinite(n) ? n : o for (n, o) in zip(new_scales, old)]
    chain.scale = Diagonal(scales)
    chain.kinetic_energy = _coop_energy(chain.scale)
    chain.stepsize *= sqrt(minimum(old) / minimum(scales))
    chain.log_kappa = _kappa_log(chain.stepsize, cfg.target_acceptance_rate)
    chain.backoff = 1.0
    empty!(chain.stepsizes); empty!(chain.acceptances)
    chain.sampling_start = 0
    _reset_moments!(chain.moments)
    recording_lpdf = chain.recording_lpdf
    chain.dropped_posterior_position = Matrix{Float64}(recording_lpdf.posterior_position)
    chain.dropped_posterior_gradient = Matrix{Float64}(recording_lpdf.posterior_gradient)
    chain.dropped_n_divergent_samples = chain.n_divergent_samples
    reset!(recording_lpdf)
    chain.status = :warming
    chain.current_transition_counter = 0
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
# each chain's own position variance (Bruno's choice), falling back to the
# metric's scale for a chain without states yet.
_cluster_spread(chains) = begin
    total = zeros(first(chains).dimension)
    for c in chains
        v = _moment_variances(c.moments)
        s = diag(c.scale)
        @. total += ifelse(isfinite(v) & (v > 0), v, abs2(s))
    end
    sqrt.(total ./ length(chains))
end

_standardized_distance(a, b, spread) = mean(abs2((x - y) / s) for (x, y, s) in zip(a, b, spread))

_responsibilities(locations, centers, spread, bandwidth) = begin
    D = [_standardized_distance(x, c, spread) for x in locations, c in centers]
    R = exp.(-(D .- minimum(D; dims=2)) ./ (2bandwidth^2))   # row-shifted: no underflow
    R ./ sum(R; dims=2)
end

_pool_clusters!(clusters::CoopClusters, chains, responsibilities) = begin
    clusters.responsibilities = responsibilities
    d = first(chains).dimension
    clusters.stats = [_combine_moments!(CoopMoments(d), (c.moments for c in chains), view(responsibilities, :, k))
                      for k in axes(responsibilities, 2)]
    clusters
end

_update_clusters!(clusters::CoopClusters, chains, cfg) = begin
    spread = _cluster_spread(chains)
    locations = map(_chain_location, chains)
    centers = clusters.centers
    for x in locations
        far = isempty(centers) || minimum(c -> _standardized_distance(x, c, spread), centers) > 4
        far && length(centers) < cfg.max_clusters && push!(centers, copy(x))
    end
    _pool_clusters!(clusters, chains, _responsibilities(locations, centers, spread, cfg.bandwidth))
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
    _pool_clusters!(clusters, chains, _responsibilities(locations, clusters.centers, spread, cfg.bandwidth))
end

# Chain i's pooled estimate: its clusters' moments, weighted by its responsibilities.
_pooled_estimate(clusters::CoopClusters, i) =
    _combine_moments!(CoopMoments(length(first(clusters.centers))), clusters.stats,
        view(clusters.responsibilities, i, :))

# Each chain's κ prior: the responsibility-weighted mean of its clusters' mean
# log κ. O(chains × clusters).
_update_kappa_priors!(clusters::CoopClusters, chains) = begin
    R = clusters.responsibilities
    # A cluster whose responsibilities all underflowed carries no weight anywhere.
    cluster_log_kappa = map(axes(R, 2)) do k
        mass = sum(view(R, :, k))
        mass > 0 ? sum(R[i, k] * chains[i].log_kappa for i in eachindex(chains)) / mass : 0.0
    end
    for (i, c) in enumerate(chains)
        c.kappa_prior = sum(R[i, k] * cluster_log_kappa[k] for k in axes(R, 2) if R[i, k] > 0)
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

# Create chains `indices` in parallel on `cfg.n_cores` workers. Their streams are
# drawn from the seed RNG serially first, in index order, so the seed-to-chain
# map does not depend on scheduling (decision `0m8p12j`, prong `rng`).
_create_chains!(run::CoopRun, indices) = begin
    streams = [Random.Xoshiro(rand(run.rng, UInt64)) for _ in indices]
    nodes = [initialize_progress!(run.progress; description="$(run.description).$i") for i in indices]
    created = Vector{CoopChain}(undef, length(indices))
    _parallel_foreach(eachindex(indices), run.cfg.n_cores) do j
        created[j] = _coop_chain(indices[j], streams[j], deepcopy(run.lpdf),
            _chain_init(run.init, indices[j]), run.cfg; progress=nodes[j])
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
    _parallel_foreach(chains, cfg.n_cores) do chain
        update_progress!(chain.progress, "$(chain.status) · batch $(run.batch)")
        while _running(chain) && chain.round_evaluations < cfg.n_evaluations && !_stop_requested(run)
            _coop_transition!(chain, cfg, run.total_evaluations)
        end
        update_progress!(chain.progress, "$(chain.status) · batch $(run.batch), waiting for the round")
    end
end

# Rounds of batch k: 2^(k-1). Batch k ends after round 2^k - 1.
_batch_ends_after(round) = ispow2(round + 1)

# Each running chain's restart, decided from its cluster-pooled estimate, all from
# the same pre-restart state.
_batch_end!(run::CoopRun) = begin
    cfg = run.cfg
    decide(i, chain) = begin
        _running(chain) || return nothing
        estimate = _pooled_estimate(run.clusters, i)
        new_scales = _moment_scales(estimate)
        cc = _relative_condition(new_scales, chain.scale)
        sc = _stepsize_gain(chain, cfg)
        (; chain, new_scales, cc, sc, restart=estimate.weight >= 2 && cc * sc > cfg.restart_threshold)
    end
    decisions = [decide(i, c) for (i, c) in enumerate(run.chains)]
    running = filter(!isnothing, decisions)
    # Refit the shared reparametrization only when every running chain restarts
    # anyway, so no kept draw is ever left in a stale frame. After a change the
    # chains re-cluster in the new frame and take their metrics from their
    # transported halos, as `adaptive_warmup_mcmc` does after its refit.
    if !isnothing(run.reparam_lpdf) && !isempty(running) && all(d -> d.restart, running) &&
            !isnothing(_refit_reparametrization!(run))
        run.clusters = CoopClusters()
        _update_clusters!(run.clusters, run.chains, cfg)
        decisions = [isnothing(d) ? d : merge(d, (; new_scales=_moment_scales(_pooled_estimate(run.clusters, i)),
                                                 centering_refit=true))
                     for (i, d) in enumerate(decisions)]
    end
    for d in decisions
        isnothing(d) && continue
        chain = d.chain
        push!(chain.batches, (; batch=run.batch, status=chain.status, relative_condition=d.cc,
            stepsize_gain=d.sc, restart=d.restart, centering_refit=get(d, :centering_refit, false),
            n_samples=chain.n_samples,
            total_evaluation_counter=chain.total_evaluation_counter, stepsize=chain.stepsize,
            min_ess=chain.monitor_ess && chain.n_samples > 10 ? _min_ess(chain.ess) : NaN))
        d.restart && _restart!(chain, d.new_scales, d.cc, cfg)
    end
    # Keep the halo ring (centering evidence) spanning one batch of the next size.
    next_batch_evaluations = cfg.n_evaluations * 2^run.batch
    for c in run.chains
        c.recording_lpdf.recorder.thin = max(1, next_batch_evaluations ÷ cfg.recording_target)
    end
    run.batch += 1
end

# Refit the shared reparametrization from every running chain's retained halo
# (the `:linear_pool` evidence), on the master copy. On a change, restore the new
# controls onto every running chain, move its current point into the new frame,
# and refill its moments from its transported halo. A transported point that is
# not finite rolls the whole refit back. Returns the new controls, or `nothing`
# when they did not change.
_refit_reparametrization!(run::CoopRun) = begin
    master = run.reparam_lpdf
    chains = _running_chains(run)
    columns = [size(c.recording_lpdf.halo_position, 2) for c in chains]
    sum(columns) > 0 || return nothing
    before = deepcopy(reparam_sources(master))
    old_ir = snapshot_reparametrization(reparametrizer(master))
    halo_position = reduce(hcat, [Matrix(c.recording_lpdf.halo_position) for c in chains])
    halo_gradient = reduce(hcat, [Matrix(c.recording_lpdf.halo_gradient) for c in chains])
    recorder = NonlinearRecorder(master; capacity=size(halo_position, 2))
    find_reparametrization!(master, recorder, halo_position, halo_gradient, first(chains).position_and_gradient)
    after = reparam_sources(master)
    after == before && return nothing
    points = [_transport_active_evaluation(master, old_ir, c.position_and_gradient) for c in chains]
    if !all(pg -> isfinite(pg.ℓq) && all(isfinite, pg.∇ℓq), points)
        restore_reparam_sources!(master, before)
        @warn "Rejected the pooled centering update: a chain's current point is not finite in the new coordinates; retaining the previous ones"
        return nothing
    end
    offset = 0
    for (c, pg, n) in zip(chains, points, columns)
        restore_reparam_sources!(c.lpdf, after)
        c.position_and_gradient = pg
        _reset_moments!(c.moments)
        for j in offset+1:offset+n
            _fit_moments!(c.moments, view(halo_position, :, j), view(halo_gradient, :, j))
        end
        offset += n
    end
    # A finished chain keeps its own frame for its draws; its moments, in the old
    # frame, no longer pool.
    foreach(c -> _running(c) || _reset_moments!(c.moments), run.chains)
    after
end

# =============================================================================
# Stopping and ESS
# =============================================================================

# The pooled ESS of the best subset of at least `min_chains` chains holding more
# than 10 draws: the chains with the most draws, truncated to the shortest one's
# tail (`_stack_chain_tails`). Subsets of doubling size are tried, so this stays
# O(log chains) ESS evaluations. Returns `(ess, n_chains)`.
_best_pooled_ess(draws, min_chains) = begin
    usable = sort!([d for d in draws if size(d, 2) > 10]; by=d -> -size(d, 2))
    length(usable) >= min_chains || return (0.0, 0)
    best = (0.0, 0)
    k = min_chains
    while true
        stacked = _stack_chain_tails(view(usable, 1:k); min_chain_draws=11)
        if !isnothing(stacked)
            e = _min_ess(vec(MCMCDiagnosticTools.ess(stacked)))
            isfinite(e) && e > first(best) && (best = (e, k))
        end
        k == length(usable) && break
        k = min(2k, length(usable))
    end
    best
end

_min_pooled_chains(run::CoopRun) = min(2, run.cfg.n_chains)

# Per-chain monitored ESS and the stop decision, at a round barrier. The stop
# reason is recorded HERE, where the decision is made.
_check_stop!(run::CoopRun) = begin
    cfg = run.cfg
    for c in run.chains
        c.monitor_ess && c.n_samples > 10 && (c.ess .= _coordinate_ess(chain_draws(c)))
    end
    reason = _interrupted(run.progress) ? :interrupted :
        run.total_evaluations[] >= cfg.n_evaluations_budget ? :eval_budget :
        _elapsed(run) >= cfg.time_budget ? :time_budget : nothing
    if isnothing(reason) && isfinite(cfg.target_ess)
        holding = [c for c in run.chains if c.status !== :warming && c.n_samples > 10]
        proxy = sum(c -> _min_ess(c.ess), holding; init=0.0)
        if proxy >= cfg.target_ess && length(holding) >= _min_pooled_chains(run)
            run.last_pooled_ess = first(_best_pooled_ess(map(chain_draws, holding), _min_pooled_chains(run)))
            run.last_pooled_ess >= cfg.target_ess && (reason = :target_ess)
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
    parts = ["$(n(s)) $(s)" for s in (:sampling, :warming, :done) if n(s) > 0]
    k = length(run.clusters.centers)
    "$(length(run.chains)) chains: " * join(parts, ", ") * "; $k cluster" * (k == 1 ? "" : "s")
end

_coop_batch_label(run::CoopRun) = begin
    rounds = 2^(run.batch - 1)
    "$(run.batch): $rounds round$(rounds == 1 ? "" : "s") of $(short_string(run.cfg.n_evaluations)) evaluations per chain"
end

_coop_pooled_ess_label(run::CoopRun) = isfinite(run.cfg.target_ess) ?
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
        batches=chain.batches,
    )
end

_finalize(run::CoopRun) = begin
    for c in run.chains
        _finish_chain_progress!(c, c.status === :done ? "done" : "$(c.status) when the run stopped")
    end
    results = map(chain_result, run.chains)
    joint_ess, n_pooled = _best_pooled_ess([r.posterior_position for r in results], _min_pooled_chains(run))
    (;
        results,
        stop_reason=run.stop_reason,
        n_used=count(r -> r.n_samples > 0, results),
        joint_ess,
        n_pooled,
        n_batches=run.batch - 1,
        total_evaluation_counter=run.total_evaluations[],
        elapsed=_elapsed(run),
    )
end

_finish_run_progress!(run::CoopRun, result) = begin
    isnothing(run.progress) && return nothing
    _run_progress!(run)
    n = sum(r -> r.n_samples, result.results; init=0)
    divergent = n == 0 ? "no draws retained" :
        "divergent: $(short_string(100 * sum(r -> r.n_divergent_samples, result.results) / n))%"
    update_progress!(run.progress,
        "$(result.stop_reason): joint ESS $(short_string(result.joint_ess)) over $(result.n_pooled) chains, " *
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

_coop_payload(chain::CoopChain, run::CoopRun) = (;
    schema_version=checkpoint_schema_version(),
    sampler=:cooperative,
    chain_index=chain.chain_index,
    status=chain.status,
    stuck_reason=nothing,
    is_final=chain.status === :done,
    stop_reason=chain.status === :done ? :n_draws : :running,
    window=run.batch - 1,
    round=run.round,
    dimension=chain.dimension,
    chain.rng, chain.position_and_gradient, chain.stepsize,
    scale_options=(; diagonal=chain.scale), active_transformation=:diagonal,
    chain.log_kappa, chain.kappa_prior, chain.backoff, chain.stepsizes, chain.acceptances,
    chain.sampling_start, chain.moments, chain.current_transition_counter,
    chain.total_transition_counter, chain.total_evaluation_counter, chain.n_divergent,
    chain.n_divergent_samples, chain.n_samples, chain.steps_per_draw, chain.ess,
    chain.scale_changes, chain.n_restarts, chain.batches,
    posterior_position=chain.recording_lpdf.posterior_position,
    posterior_gradient=chain.recording_lpdf.posterior_gradient,
    halo_position=chain.recording_lpdf.halo_position,
    halo_gradient=chain.recording_lpdf.halo_gradient,
    recorder=chain.recording_lpdf.recorder,
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
        schema_version=checkpoint_schema_version(), sampler=:cooperative,
        run.round, run.batch, total_evaluations=run.total_evaluations[], run.rng,
        run.clusters, chains=payloads, run.last_pooled_ess, run_elapsed=_elapsed(run),
        master_sources=isnothing(run.reparam_lpdf) ? nothing : reparam_sources(run.reparam_lpdf),
    ))
end

_write_coop_manifest(::Nothing, run) = nothing
_write_coop_manifest(dir::AbstractString, run::CoopRun) = _write_json(
    joinpath(dir, "run_manifest.json"), Pair{String,Any}[
        "schema_version" => checkpoint_schema_version(),
        "sampler" => "cooperative",
        "n_chains" => run.cfg.n_chains,
        "n_cores" => run.cfg.n_cores,
        # The stopping criteria in force; JSON has no Inf, so an unset bound is null.
        "n_draws" => run.cfg.n_draws == typemax(Int) ? nothing : run.cfg.n_draws,
        "target_ess" => run.cfg.target_ess,
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
        "schema_version" => checkpoint_schema_version(),
        "sampler" => "cooperative",
        "is_final" => true,
        "stop_reason" => string(result.stop_reason),
        "n_chains" => length(result.results),
        "n_used" => result.n_used,
        "joint_ess" => result.joint_ess,
        "n_pooled" => result.n_pooled,
        "n_batches" => result.n_batches,
        "total_evaluation_counter" => result.total_evaluation_counter,
        "elapsed" => result.elapsed,
        "end_time_unix" => time(),
    ])

# A chain from its payload and a fresh `lpdf`; the configuration comes from the
# resuming call. A finished chain resumes sampling when the call raises `n_draws`.
_restore_coop_chain(p, lpdf, cfg; progress) = begin
    check_checkpoint_compatible(p, :cooperative, (:cooperative,))
    _check_schema_version(p)
    d = LogDensityProblems.dimension(lpdf)
    p.dimension == d || throw(DimensionMismatch(
        "the checkpoint holds a $(p.dimension)-dimensional problem but the supplied lpdf has dimension $d."))
    _check_candidate_scoring_compatible(p, lpdf)
    restore_reparam_sources!(lpdf, p.reparam_sources)
    recording_lpdf = RecordingPosterior2(lpdf, p.halo_position, p.halo_gradient,
        p.posterior_position, p.posterior_gradient, NUTSLeaves(d), p.recorder, p.rng)
    status = p.status === :warming ? :warming : p.n_samples >= cfg.n_draws ? :done : :sampling
    scale = p.scale_options.diagonal
    chain = CoopChain(
        p.chain_index, p.rng, lpdf, recording_lpdf, d, cfg.monitor_ess, time_ns(), progress,
        p.position_and_gradient, scale, _coop_energy(scale), p.stepsize, p.log_kappa, p.kappa_prior,
        p.backoff, p.stepsizes, p.acceptances, p.sampling_start, p.moments, status,
        p.current_transition_counter, p.total_transition_counter, p.total_evaluation_counter, 0,
        p.n_divergent, p.n_divergent_samples, p.n_samples, p.steps_per_draw, p.ess,
        p.scale_changes, p.n_restarts, p.batches, p.dropped_posterior_position,
        p.dropped_posterior_gradient, p.dropped_n_divergent_samples,
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
    nodes = [initialize_progress!(progress; description="$description.$(p.chain_index)") for p in state.chains]
    chains = [_restore_coop_chain(p, deepcopy(lpdf), cfg; progress=node) for (p, node) in zip(state.chains, nodes)]
    master = adapt_centering ? restore_reparam_sources!(deepcopy(lpdf), state.master_sources) : nothing
    CoopRun(state.rng, lpdf, cfg, init, progress, string(description), time_ns(), state.run_elapsed, master,
        CoopChain[chains...], state.clusters, Threads.Atomic{Int}(state.total_evaluations),
        state.round, state.batch, nothing, state.last_pooled_ess)
end

# =============================================================================
# Entry point
# =============================================================================

"""
    cooperative_warmup_mcmc(rng, lpdf; n_chains=4, n_cores=n_chains, target_ess=Inf,
        n_evaluations_budget=typemax(Int), time_budget=Inf, n_draws=typemax(Int), kwargs...)

Cooperative multi-chain NUTS warm-up and sampling on one target: `n_chains`
chains adapt TOGETHER and keep sampling until a stopping bound is met.

Every leapfrog state of every chain feeds running position and gradient moments.
Chains pool them through soft clusters (chains near each other in the posterior
share; once they have mixed, all share one estimate), and each chain's metric is
the Nutpie diagonal scale of its pooled moments. Work proceeds in rounds of
`n_evaluations` gradient evaluations per chain; batch k is 2^(k-1) rounds. At a
batch end a chain restarts warm-up under its pooled estimate only when that
estimate's relative condition number against its current metric, times the
predicted step-size gain, exceeds `restart_threshold` (√2). A restart drops the
chain's draws; otherwise it keeps sampling under its fixed kernel, so every kept
draw is an MCMC draw of an unchanging kernel. Step sizes follow a per-chain fit
of acceptance ≈ 2Φ(-κε²) with the pooled κ as prior. No chain is ever parked or
abandoned.

* `rng` seeds every chain: chain i's stream is drawn from it when chain i is
  created, so one seed reproduces the run at any thread count.
* Stops when the pooled ESS of at least `min(2, n_chains)` chains reaches
  `target_ess`, `n_evaluations_budget` gradient evaluations are spent,
  `time_budget` seconds pass, every chain holds `n_draws` draws, or
  `Treebars.request_interrupt!` is called on `progress`, all checked at round
  barriers (a budget or an interrupt also stops a round early). At least one
  bound must be finite. The returned `stop_reason` names the one that fired.
* `init` (as for [`adaptive_warmup_mcmc`](@ref)) is shared by every chain, or a
  vector of per-chain inits; a position vector of reals is one shared start.

## Checkpointing and resume

`checkpoint_dir=path` writes, at every round barrier, `path/run_latest.jls` (the
whole run) and `path/chain_<i>/cp_latest.jls`, plus `path/chain_<i>/cp_window_<k>.jls`
at the end of batch k, all atomically; `run_manifest.json` at the start and
`run_summary.json` at the finish are write-once. A later call with `resume=true`
continues an unfinished run from its last round barrier: every chain's state,
draws and stream, the seed RNG, the clusters and the counters are restored, so
the run continues as if it had not stopped. The configuration comes from the
resuming call (a larger `n_draws` extends finished chains; `n_chains` must
match), the evaluation budget continues from the restored total, and
`time_budget` from the run's live time. A finished run (`run_summary.json`
exists) is refused; `overwrite=true` discards the directory's run.

Returns `(; results, stop_reason, n_used, joint_ess, n_pooled, n_batches,
total_evaluation_counter, elapsed)`. `results[i]` is chain `i`'s
[`chain_result`](@ref), its draws in the model frame. `joint_ess` is the pooled
ESS over the `n_pooled` chains with the most draws (at least `min(2, n_chains)`).
"""
cooperative_warmup_mcmc(rng::Random.AbstractRNG, lpdf;
    n_chains::Integer=4,
    n_cores::Integer=n_chains,
    target_ess=Inf,
    n_evaluations_budget=typemax(Int),
    time_budget=Inf,
    n_draws=typemax(Int),
    n_evaluations=1000,
    stepsize_adaptation_limit=50,
    target_acceptance_rate=0.8,
    max_tree_depth=10,
    restart_threshold=sqrt(2),
    max_clusters=8,
    bandwidth=1.0,
    kappa_prior_weight=2.0,
    kappa_decay=0.9,
    recording_target=1000,
    init=missing,
    nonlinear_adapt=true,
    monitor_ess=true,
    progress=nothing,
    description="MCMC",
    checkpoint_dir=nothing,
    resume=false,
    overwrite=false,
    pathfinder_kw=(;),
    kwargs...
) = begin
    _check_kwargs(:cooperative_warmup_mcmc, kwargs)
    guard_run_dir!(checkpoint_dir, resume, overwrite, :cooperative)
    n_chains >= 1 || throw(ArgumentError("cooperative_warmup_mcmc needs n_chains ≥ 1, got $n_chains."))
    isfinite(target_ess) || n_evaluations_budget != typemax(Int) || isfinite(time_budget) || n_draws != typemax(Int) ||
        throw(ArgumentError("cooperative_warmup_mcmc needs at least one finite stopping bound " *
            "(target_ess, n_evaluations_budget, time_budget or n_draws)."))
    init isa AbstractVector && !(init isa AbstractVector{<:Real}) && length(init) != n_chains &&
        throw(ArgumentError("a per-chain `init` vector needs one entry per chain: got $(length(init)) for n_chains = $n_chains."))
    _check_nonlinear_adaptable(lpdf, nonlinear_adapt)
    adapt_centering = nonlinear_adapt && has_reparametrization(reparametrizer(lpdf))
    # Online candidate scoring keeps its evidence in a per-chain accumulator,
    # which the pooled refit cannot merge yet: it would find no evidence and never
    # adapt. Refuse rather than silently not adapting.
    adapt_centering && _uses_online_candidate_scoring(candidate_scoring_plan(lpdf)) && throw(ArgumentError(
        "cooperative_warmup_mcmc cannot yet adapt a reparametrization scored by a `CandidateScoringPlan` " *
        "(its evidence is per chain and is not pooled). Pass `nonlinear_adapt=false` to sample under the " *
        "centering you supplied, or use `adaptive_warmup_mcmc`."))
    cfg = (; n_chains, n_cores=max(1, n_cores), target_ess=Float64(target_ess), n_evaluations_budget,
        time_budget=Float64(time_budget), n_draws, n_evaluations, stepsize_adaptation_limit,
        target_acceptance_rate, algorithm=DynamicHMC.NUTS(; max_depth=max_tree_depth),
        restart_threshold, max_clusters, bandwidth, kappa_prior_weight, kappa_decay,
        recording_target, monitor_ess, initializer_kw=(; kwargs..., pathfinder_kw...))
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
            fresh
        end
        _run_progress!(run)
        while true
            run.round += 1
            _run_round!(run)
            _update_clusters!(run.clusters, run.chains, cfg)
            _check_stop!(run) && break
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
