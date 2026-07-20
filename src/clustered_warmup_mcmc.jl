# Truly-cooperative (clustered) multi-chain sampler.
#
# DISTINCT from the side-by-side `cooperative_warmup_mcmc`: there, cooperation
# lives only in the SCHEDULER (which chain to advance / when to stop) and every
# chain adapts its OWN mass matrix from its own three `scale_options`. Here the
# chains POOL their draws to estimate the mass matrix TOGETHER — the "really
# cooperative" sampler the user asked to revive. Each window:
#
#   1. every chain samples one window under its current diagonal scale, feeding
#      its halo (intermediate positions AND gradients, nutpie-style) into a
#      fresh per-window `NutpieScaleAdaptation` — in ORIGINAL space, so the
#      estimates are directly poolable/comparable across chains;
#   2. the chains are CLUSTERED by mass-matrix compatibility (`cluster_fn`,
#      greedy `assign_clusters` default) — "carefully but optimistically";
#   3. each cluster POOLS its members' estimates into ONE shared diagonal scale
#      (`pooled` + `marginal_scales`);
#   4. a chain whose pooled scale differs from what it is sampling under by more
#      than `threshold` (condition number of the ratio) RESTARTS onto the new
#      scale — resets step size, discards its warm-up draws; otherwise it keeps
#      sampling and accumulating draws under a frozen scale.
#
# Shared primitives (NOT a joint function): `initialize_mcmc`, the recording
# pipeline (`RecordingPosterior2`/`LimitedRecorder2`), and the DynamicHMC kernel
# are reused verbatim from adaptive_warmup_mcmc.jl; the pooled estimator
# (`NutpieScaleAdaptation`) and clustering (`assign_clusters`) are in
# pooled_scale.jl / clustering.jl. There is deliberately NO single joint target
# of evaluations shared with the side-by-side sampler.
#
# Two knobs are pluggable with sensible defaults (the framework theme):
#   * `cluster_fn`  — pool-construction strategy (decision `ofj104`: greedy
#     default; a look-behind strategy `lzetez`/`hwfj4p` slots in unchanged);
#   * `weighting`   — which states, and how weighted, feed the estimate
#     (decision `1s8blg3`: |dH|-halo all-ones default; a leaf-weights scheme
#     `myvkgz` slots in as a different `weighting`).
# Decisions: `lnmhbn` nutpie, `4hobr0` per-chain doubling, `19clurb` cond>√2
# restart, `1nfpfei` in-memory, `1bfkc9a` all-clusters-labeled output.

"Diagonal Gaussian kinetic energy with metric `scale*scale'` (matches the monolith's `:diagonal` option)."
_diagonal_energy(scale::Diagonal) =
    DynamicHMC.GaussianKineticEnergy(MatrixFactorization(scale, scale'), MatrixInverse(scale'))

"Reset ONLY the halo ring buffer (its leaves + recorder), leaving the collected draws intact."
_reset_halo!(p::RecordingPosterior2) = (reset!(p.halo_position); reset!(p.halo_gradient); reset!(p.leaves); reset!(p.recorder); p)

"""
    default_weighting(halo_position, halo_gradient) -> weights

Default state-weighting for the pooled mass-matrix estimate (decision
`1s8blg3`): every recorded halo state gets weight 1. On `dev` the recorder
([`RecordingPosterior2`](@ref)) already stores each halo state by sampling one
NUTS leaf per transition PROPORTIONAL to its proper marginal proposal
probability (`sample_leaf`/`finalize_leaf_weights!`, the `myvkgz` leaf-weights
work), so equal weights here already realise proper leaf-weighting via that
selection. A weighting is any `(halo_position, halo_gradient) -> AbstractVector`
of per-column weights; swap in another scheme via the `weighting` kwarg.
"""
default_weighting(halo_position, halo_gradient) = Ones(size(halo_position, 2))

"""
    ClusteredChain

Resumable state of one chain in the clustered cooperative sampler. Unlike
[`CooperativeChain`](@ref) it does NOT carry the three self-adapted
`scale_options`: its diagonal `scale` is supplied externally by its cluster's
pooled estimate. Each window it (re)fills a fresh `adaptation` from its halo, in
original space, for the driver to cluster and pool.

`status` ∈ `:warming` (last checkpoint restarted onto a new pooled scale) /
`:sampling` (scale stable, accumulating draws) / `:done` (`n_draws` collected).
"""
mutable struct ClusteredChain{R,L,RL,A,SA,K,W}
    # --- configuration (set once) ---
    const rng::R
    const lpdf::L
    const recording_lpdf::RL           # RecordingPosterior2 wrapping lpdf
    const algorithm::A                 # DynamicHMC.NUTS
    const stepsize_adaptation::SA      # DynamicHMC.DualAveraging
    const dimension::Int
    const recording_target::Int
    const stepsize_adaptation_limit::Int
    const n_draws::Int
    const max_window_evaluations::Int
    const weighting::W                 # (halo_position, halo_gradient) -> per-state weights
    # --- mutable window-loop state ---
    position_and_gradient
    scale::Diagonal{Float64,Vector{Float64}}   # current diagonal mass-matrix scale (from the cluster pool)
    kinetic_energy::K
    stepsize::Float64
    stepsize_state
    n_evaluations::Int                 # current window's gradient-eval budget (doubles per window)
    adaptation::NutpieScaleAdaptation{Float64}   # THIS window's estimate (reset+refilled every window)
    cluster_id::Int
    total_evaluation_counter::Int
    current_transition_counter::Int    # transitions since the last restart (drives stepsize freeze)
    n_divergent_samples::Int
    n_samples::Int                     # draws collected since the last restart
    status::Symbol
    checkpoints::Vector{NamedTuple}
end

"Positions collected as posterior draws since the last restart (dimension × n_samples)."
chain_draws(chain::ClusteredChain) = chain.recording_lpdf.posterior_position
n_chain_draws(chain::ClusteredChain) = size(chain_draws(chain), 2)

"""
    clustered_chain(rng, lpdf; n_draws, n_evaluations, recording_target,
                    stepsize_adaptation_limit, target_acceptance_rate,
                    max_tree_depth, max_window_evaluations, init,
                    regularizing_n, regularizing_var, weighting, progress, kwargs...)

Build a [`ClusteredChain`](@ref) ready for its first window. Initializes via
Pathfinder (shared `initialize_mcmc`) and starts from the diagonal of the
Pathfinder covariance — the same starting scale as the monolith's `:diagonal`
option.
"""
clustered_chain(
    rng, lpdf;
    n_draws=1000,
    n_evaluations=1000,
    recording_target=1000,
    stepsize_adaptation_limit=50,
    target_acceptance_rate=.8,
    max_tree_depth=10,
    max_window_evaluations=4000,
    init=missing,
    regularizing_n=5.0,
    regularizing_var=1e-3,
    weighting=default_weighting,
    progress=nothing,
    kwargs...
) = begin
    stepsize_adaptation = DynamicHMC.DualAveraging(δ=target_acceptance_rate)
    algorithm = DynamicHMC.NUTS(;max_depth=max_tree_depth)
    dimension = LogDensityProblems.dimension(lpdf)
    recorder = LimitedRecorder2(recording_target)   # ring of `recording_target` leaf-weighted halo states
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    scale = Diagonal(sqrt.(diag(squared_scale))::Vector{Float64})
    kinetic_energy = _diagonal_energy(scale)
    adaptation = NutpieScaleAdaptation(dimension; regularizing_n, regularizing_var)
    # Initial stepsize (consumes rng via rand_p, exactly as the monolith does).
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    pgm = DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, kinetic_energy))
    stepsize = DynamicHMC.find_initial_stepsize(
        DynamicHMC.InitialStepsizeSearch(),
        DynamicHMC.local_log_acceptance_ratio(DynamicHMC.Hamiltonian(kinetic_energy, lpdf), pgm)
    )
    stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
    ClusteredChain(
        rng, lpdf, recording_lpdf, algorithm, stepsize_adaptation, dimension,
        recording_target, stepsize_adaptation_limit, n_draws, max_window_evaluations,
        weighting,
        position_and_gradient, scale, kinetic_energy, stepsize, stepsize_state,
        n_evaluations, adaptation, 0, 0, 0, 0, 0, :warming, NamedTuple[],
    )
end

"""
    advance_chain!(chain) -> chain

Run exactly ONE window: sample transitions under the chain's current scale until
its per-window gradient-eval budget is hit (adapting the step size for the first
`stepsize_adaptation_limit` transitions since the last restart, collecting draws
thereafter), then FOLD this window's halo (intermediate positions AND gradients,
in ORIGINAL space, weighted by `chain.weighting`) into the ACCUMULATING
`chain.adaptation`. The halo is reset per window so folding never double-counts;
the estimate is never reset (original-space variances are target-invariant, so
more windows only sharpen it). Leaves clustering + scale adoption to
[`cluster_and_adapt!`](@ref); this call touches only its own chain, so it is
safe to run for different chains in parallel.
"""
advance_chain!(chain::ClusteredChain) = begin
    n_chain_draws(chain) >= chain.n_draws && (chain.status = :done; return chain)
    (;rng, algorithm, recording_lpdf, stepsize_adaptation, stepsize_adaptation_limit, n_draws) = chain
    _reset_halo!(recording_lpdf)   # fresh per-window halo, so accumulation never double-counts
    (;halo_position, halo_gradient, posterior_position, posterior_gradient) = recording_lpdf
    hamiltonian = DynamicHMC.Hamiltonian(chain.kinetic_energy, recording_lpdf)
    current_evaluation_counter = 0
    while size(posterior_position, 2) < n_draws && current_evaluation_counter < chain.n_evaluations
        chain.current_transition_counter += 1
        reset!(recording_lpdf.leaves)   # per-transition: fresh leaf buffer before the tree (dev's protocol)
        chain.position_and_gradient, stats = DynamicHMC.sample_tree(
            rng, algorithm, hamiltonian, chain.position_and_gradient, chain.stepsize
        )
        finalize_leaf_recording!(recording_lpdf, stats.depth)   # sample one leaf ∝ proper weight → halo
        chain.total_evaluation_counter += stats.steps
        current_evaluation_counter += stats.steps
        is_divergent = DynamicHMC.is_divergent(stats.termination)
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
        chain.n_samples = size(posterior_position, 2)
    end
    # Fold THIS window's halo into the accumulating pooled-estimate input (original space).
    weights = chain.weighting(halo_position, halo_gradient)
    @inbounds for (i, (pi, gi)) in enumerate(zip(eachcol(halo_position), eachcol(halo_gradient)))
        OnlineStatsBase.fit!(chain.adaptation, pi, gi; dw=weights[i])
    end
    chain
end

"""
    cluster_and_adapt!(chains; cluster_fn=assign_clusters, metric=cond_compatibility,
                       threshold=sqrt(2)) -> clusters

Cluster `chains` by mass-matrix compatibility, pool each cluster's per-window
estimates into ONE diagonal scale, and have each chain ADOPT that pooled scale
iff it differs from what the chain is currently sampling under by more than
`threshold` (condition number of the marginal-scale ratio, decision `19clurb`).
Adopting = restart: reset step-size adaptation, discard the chain's warm-up
draws, mark `:warming`. A chain whose pooled scale is already within `threshold`
keeps its frozen scale and accumulates draws (`:sampling`). Sets each chain's
`cluster_id`. Returns the cluster partition (indices into `chains`).

`cluster_fn` is the swappable pool-construction strategy (decision `ofj104`);
`metric`/`threshold` are the tuneable compatibility knobs (decision `y4o8i`).
"""
cluster_and_adapt!(chains::AbstractVector{<:ClusteredChain};
    cluster_fn=assign_clusters, metric=cond_compatibility, threshold=sqrt(2.0)) = begin
    adaptations = [chain.adaptation for chain in chains]
    clusters = cluster_fn(adaptations; metric, threshold)
    for (cid, members) in enumerate(clusters)
        pooled_adaptation = pooled(adaptations[members]...)
        new_scale = marginal_scales(pooled_adaptation)
        for m in members
            chain = chains[m]
            chain.cluster_id = cid
            chain.n_samples >= chain.n_draws && (chain.status = :done; continue)
            # cond of (pooled scale / current scale): >threshold ⇒ the pool wants a
            # materially different geometry ⇒ restart onto it.
            if diagonal_cond(new_scale ./ diag(chain.scale)) > threshold
                adopt_scale!(chain, new_scale)
                chain.status = :warming
            else
                chain.status = :sampling
            end
            log_checkpoint!(chain)
            chain.n_evaluations = min(chain.n_evaluations * 2, chain.max_window_evaluations)   # decision 4hobr0
        end
    end
    clusters
end

"Adopt a new diagonal `scale` and restart: rebuild the kernel, reset step-size adaptation, discard warm-up draws."
adopt_scale!(chain::ClusteredChain, scale_vec::AbstractVector) = begin
    chain.scale = Diagonal(collect(Float64, scale_vec))
    chain.kinetic_energy = _diagonal_energy(chain.scale)
    chain.stepsize = DynamicHMC.final_ϵ(chain.stepsize_state)
    chain.stepsize_state = DynamicHMC.initial_adaptation_state(chain.stepsize_adaptation, chain.stepsize)
    chain.stepsize = DynamicHMC.current_ϵ(chain.stepsize_state)
    chain.current_transition_counter = 0
    chain.n_divergent_samples = 0
    reset!(chain.recording_lpdf)
    chain.n_samples = 0
    chain
end

log_checkpoint!(chain::ClusteredChain) = push!(chain.checkpoints, (;
    cluster_id=chain.cluster_id,
    n_samples=chain.n_samples,
    total_evaluation_counter=chain.total_evaluation_counter,
    n_divergent_samples=chain.n_divergent_samples,
    status=chain.status,
))

# --- pooled per-cluster diagnostics over the collected draws ---------------

"Pooled ESS + split-Rhat over the stacked draws of a cluster's chains (`ess` = min, `rhat` = max over parameters); NaN if too few draws."
cluster_diagnostics(chains::AbstractVector{<:ClusteredChain}) = begin
    usable = [chain_draws(c) for c in chains if n_chain_draws(c) > 3]
    isempty(usable) && return (; ess=NaN, rhat=NaN, n_draws=0, n_chains=0)
    m = minimum(d -> size(d, 2), usable)
    m > 3 || return (; ess=NaN, rhat=NaN, n_draws=0, n_chains=length(usable))
    dim = size(first(usable), 1)
    stacked = Array{Float64}(undef, m, length(usable), dim)      # (draws, chains, params)
    for (j, d) in enumerate(usable)
        stacked[:, j, :] = @view(d[:, end-m+1:end])'
    end
    (;
        ess=minimum(MCMCDiagnosticTools.ess(stacked)),
        rhat=maximum(MCMCDiagnosticTools.rhat(stacked)),
        n_draws=m,
        n_chains=length(usable),
    )
end

"""
    clustered_warmup_mcmc(rngs, lpdf; n_draws=1000, max_windows=16,
        cluster_fn=assign_clusters, weighting=default_weighting,
        metric=cond_compatibility, threshold=sqrt(2), parallel=true,
        n_evaluations_budget=typemax(Int), progress=nothing, kwargs...)

Truly-cooperative clustered NUTS warm-up + sampling. Chains POOL their draws to
estimate the mass matrix together, in pools built "carefully but optimistically"
by `cluster_fn`; each cluster shares ONE diagonal scale. See the file header for
the per-window loop.

Distinct from [`cooperative_warmup_mcmc`](@ref) (side-by-side, per-chain
self-adaptation) — the two share primitives, not a joint driver.

* `rngs` — one RNG per chain (its length is the chain count).
* `n_draws` — per-chain draw target (drives the stop; a chain stops sampling
  once it has this many draws SINCE ITS LAST RESTART).
* `max_windows` — hard cap on cooperative windows.
* `n_evaluations_budget` — optional global gradient-eval ceiling.
* `cluster_fn` / `weighting` / `metric` / `threshold` — the tuneable knobs
  (defaults: greedy clustering, |dH|-halo weights, cond-number metric at √2).
* `parallel=true` — advance chains on `Threads.@threads` within each window.
* extra `kwargs` (`recording_target`, `target_acceptance_rate`, `max_tree_depth`,
  `init`, `regularizing_n`, `regularizing_var`, …) are forwarded per chain.

Returns a `NamedTuple`: `clusters` (a per-cluster `NamedTuple` of `chain_indices`
+ pooled `ess`/`rhat`/`n_draws`, decision `1bfkc9a`), the per-chain
`chains`/`results`, `n_windows`, and `total_evaluation_counter`.

## Resumability

The run is assembled from resumable pieces (decision `1nfpfei` scoped *disk*
resume out of v1; in-memory resume is free here): [`clustered_chains`](@ref)
builds the per-chain state, [`clustered_step!`](@ref) advances every chain one
window and re-clusters, and [`clustered_output`](@ref) finalizes. All mutable
state — including each chain's RNG — lives in the returned `chains`, so a run
split across several `clustered_step!` calls is BIT-IDENTICAL to a single loop;
hold the `chains`, checkpoint or inspect, and continue where you left off.
"""
clustered_warmup_mcmc(rngs::AbstractVector, lpdf;
    n_draws=1000,
    max_windows=16,
    cluster_fn=assign_clusters,
    weighting=default_weighting,
    metric=cond_compatibility,
    threshold=sqrt(2.0),
    parallel=true,
    n_evaluations_budget=typemax(Int),
    init=missing,
    progress=nothing,
    kwargs...
) = begin
    chains = clustered_chains(rngs, lpdf; n_draws, weighting, init, parallel, progress, kwargs...)
    clusters = [collect(eachindex(chains))]
    n_windows = 0
    for _ in 1:max_windows
        n_windows += 1
        clusters = clustered_step!(chains; cluster_fn, metric, threshold, parallel)
        all(c -> c.status === :done, chains) && break
        sum(c -> c.total_evaluation_counter, chains) >= n_evaluations_budget && break
    end
    clustered_output(chains, clusters, n_windows)
end

"""
    clustered_chains(rngs, lpdf; n_draws=1000, weighting=default_weighting,
                     init=missing, parallel=true, progress=nothing, kwargs...) -> Vector{ClusteredChain}

Build the per-chain state for a clustered run — the resumable handle. Each chain
gets a `deepcopy` of `lpdf`. Advance the result with [`clustered_step!`](@ref).
"""
clustered_chains(rngs::AbstractVector, lpdf; n_draws=1000, weighting=default_weighting,
    init=missing, parallel=true, progress=nothing, kwargs...) = begin
    n_chains = length(rngs)
    inits = ensurevector(init, n_chains)
    chains = Vector{ClusteredChain}(undef, n_chains)
    _pforeach(parallel, 1:n_chains) do i
        chains[i] = clustered_chain(rngs[i], deepcopy(lpdf); n_draws, weighting, init=inits[i], progress, kwargs...)
    end
    chains
end

"""
    clustered_step!(chains; cluster_fn=assign_clusters, metric=cond_compatibility,
                    threshold=sqrt(2), parallel=true) -> clusters

Advance EVERY chain one window ([`advance_chain!`](@ref)) then cluster + adopt
pooled scales ([`cluster_and_adapt!`](@ref)); returns the cluster partition. This
is ONE cooperative window and the resumable unit — call it repeatedly (all state
lives in `chains`, RNG included, so splitting the loop is bit-identical).
"""
clustered_step!(chains::AbstractVector{<:ClusteredChain};
    cluster_fn=assign_clusters, metric=cond_compatibility, threshold=sqrt(2.0), parallel=true) = begin
    _pforeach(parallel, eachindex(chains)) do i
        advance_chain!(chains[i])
    end
    cluster_and_adapt!(chains; cluster_fn, metric, threshold)
end

"Finalize a clustered run into the public result (per-cluster labeled output + per-chain results)."
clustered_output(chains::AbstractVector{<:ClusteredChain}, clusters, n_windows) = (;
    clusters=map(enumerate(clusters)) do (cid, members)
        (; cluster_id=cid, chain_indices=members, cluster_diagnostics(chains[members])...)
    end,
    chains,
    results=map(clustered_result, chains),
    n_windows,
    total_evaluation_counter=sum(c -> c.total_evaluation_counter, chains; init=0),
)

"Per-chain result: the collected draws + diagnostics (mirrors the adaptive/cooperative result shape)."
clustered_result(chain::ClusteredChain) = (;
    posterior_position=chain_draws(chain),
    posterior_gradient=chain.recording_lpdf.posterior_gradient,
    scale=diag(chain.scale),
    stepsize=chain.stepsize,
    cluster_id=chain.cluster_id,
    status=chain.status,
    n_samples=chain.n_samples,
    n_divergent_samples=chain.n_divergent_samples,
    total_evaluation_counter=chain.total_evaluation_counter,
    checkpoints=chain.checkpoints,
)

# `_pforeach(f, parallel, itr)` via do-block: run `f` over `itr`, threaded when `parallel`.
_pforeach(f, parallel::Bool, itr) = if parallel
    Threads.@threads for x in itr
        f(x)
    end
else
    for x in itr
        f(x)
    end
end
