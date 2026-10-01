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
`1s8blg3`): every recorded halo state gets weight 1. The recorder
([`RecordingPosterior2`](@ref)) keeps one random good leaf per `thin` leaf
evaluations, not a proposal-weighted leaf per transition, so equal weights here
weight every retained good leaf equally. A weighting is any `(halo_position, halo_gradient) -> AbstractVector`
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
    # Draws DROPPED at the most recent scale adoption — same contract as
    # `AWMState.dropped_posterior_position`. `adopt_scale!` empties the recorder
    # and `_write_clustered_checkpoints` runs after the whole cluster round, so
    # an adopting chain would otherwise checkpoint zero draws. Inert on read-back.
    dropped_posterior_position::Matrix{Float64}
    dropped_posterior_gradient::Matrix{Float64}
    dropped_n_divergent_samples::Int
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
    # Ring of `recording_target` halo states; one retained per `thin` leaf evaluations.
    recorder = LimitedRecorder2(recording_target, max(1, n_evaluations ÷ recording_target))
    recording_lpdf = RecordingPosterior2(lpdf; recorder, rng)
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    scale = _initial_diagonal_scale(squared_scale)
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
        Matrix{Float64}(undef, dimension, 0), Matrix{Float64}(undef, dimension, 0), 0,
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
        finalize_leaf_recording!(recording_lpdf, stats.depth)   # finalize leaf weights; the halo was filled during the tree
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
            # Keep the halo filling the whole ring as the window budget grows.
            chain.recording_lpdf.recorder.thin = max(1, chain.n_evaluations ÷ chain.recording_target)
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
    # Preserve the discarded draws before the counter and the recorder are cleared
    # — the checkpoint for this window is written after the cluster round returns.
    chain.dropped_posterior_position = Matrix{Float64}(chain_draws(chain))
    chain.dropped_posterior_gradient = Matrix{Float64}(chain.recording_lpdf.posterior_gradient)
    chain.dropped_n_divergent_samples = chain.n_divergent_samples
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
* `checkpoint_dir` — opt-in on-disk checkpointing; see *Checkpointing* below.
* extra `kwargs` (`recording_target`, `target_acceptance_rate`, `max_tree_depth`,
  `init`, `regularizing_n`, `regularizing_var`, …) are forwarded per chain.

Returns a `NamedTuple`: `clusters` (a per-cluster `NamedTuple` of `chain_indices`
+ pooled `ess`/`rhat`/`n_draws`, decision `1bfkc9a`), the per-chain
`chains`/`results`, `n_windows`, and `total_evaluation_counter`.

## Checkpointing and crash-resume

`checkpoint_dir=path` writes the SAME layout and payload contract as
[`cooperative_warmup_mcmc`](@ref), so one consumer read path serves both:
`path/chain_<i>/cp_window_<n>.jls` + `cp_latest.jls` after every window, plus
write-once `run_manifest.json` (at start) and `run_summary.json` (at finalize)
at the root. Payloads carry `sampler === :clustered` and the clustered-only
`cluster_id`; see [`clustered_checkpoint_payload`](@ref).

Like the cooperative sampler this writes no `cp_init.jls` — the first
checkpoint is `cp_window_1.jls`, so a reader must not require one.

Point a later call at the same directory with `resume=true` to continue after a
crash: the whole ensemble is restored together (each chain's pooled-estimate
accumulator with it) and the run continues for the remaining windows.
`max_windows` and `n_draws` are totals across the resume — raise either to
extend; the eval budget continues from the restored counters, and a restored
ensemble that already meets a bound stops before stepping. `resume=true`
continues an unfinished run only: a directory holding `run_summary.json` is
finalized and refused, since that file is write-once. `overwrite=true` discards
the directory's run and starts fresh; without either flag a non-empty directory
is refused. A custom `weighting` must be re-passed on resume (it has
no payload copy). Pointing [`resume_warmup_mcmc`](@ref) at this directory
throws — the payload is a different shape, not a subset; resume through this
sampler's own `resume=true`.

## Resumability

The run is assembled from resumable pieces: [`clustered_chains`](@ref) builds
the per-chain state, [`clustered_step!`](@ref) advances every chain one window
and re-clusters, and [`clustered_output`](@ref) finalizes. All mutable state —
including each chain's RNG — lives in the returned `chains`, so a run split
across several `clustered_step!` calls is BIT-IDENTICAL to a single loop; hold
the `chains`, checkpoint or inspect, and continue where you left off. Disk
resume above is the crash-safe counterpart (decision `1nfpfei` scoped it out of
v1; this entry point is the follow-up).
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
    checkpoint_dir=nothing,
    # Continue from `checkpoint_dir`'s per-chain checkpoints instead of starting
    # over. `max_windows`/`n_draws` are totals across the resume; config comes
    # from THIS call.
    resume=false,
    # Discard whatever `checkpoint_dir` already holds and start fresh.
    overwrite=false,
    # Forwarded verbatim to the Pathfinder initializer; see `_check_kwargs`.
    pathfinder_kw=(;),
    kwargs...
) = begin
    # Validate BEFORE any side effect, so a rejected call writes no manifest.
    _check_kwargs(:clustered_warmup_mcmc, kwargs)
    guard_run_dir!(checkpoint_dir, resume, overwrite, :clustered)
    chains, windows_done = resume ?
        _restore_clustered_pool(checkpoint_dir, rngs, lpdf;
            n_draws, weighting, init, progress, pathfinder_kw, kwargs...) :
        (clustered_chains(rngs, lpdf; n_draws, weighting, init, parallel, progress, pathfinder_kw, kwargs...), 0)
    # The manifest is write-once: a resumed run keeps the original's identity
    # and criteria, it does not rewrite them with this call's.
    resume || write_clustered_run_manifest(checkpoint_dir, rngs,
        (; n_draws, max_windows, n_evaluations_budget, threshold))
    # A resumed run with no windows left never re-clusters, so its output
    # partition comes from the as-checkpointed cluster ids.
    clusters = resume ? _clusters_from_ids(chains) : [collect(eachindex(chains))]
    n_windows = windows_done
    # A restored ensemble can ALREADY meet a stopping bound (a crash between its
    # last checkpoint and finalize, or a resuming call with a lower budget), so a
    # resume checks before stepping — stopping where the uninterrupted run did
    # rather than running one window past it. A fresh run has done no work yet,
    # so its loop is unchanged.
    early = resume ? _clustered_stop_reason(chains, n_evaluations_budget) : nothing
    stop_reason = something(early, :max_windows)
    for _ in 1:(isnothing(early) ? max(0, max_windows - windows_done) : 0)
        n_windows += 1
        clusters = clustered_step!(chains; cluster_fn, metric, threshold, parallel)
        # AFTER the step, so `cluster_id` and any restart are already reflected.
        _write_clustered_checkpoints(checkpoint_dir, chains, n_windows)
        reason = _clustered_stop_reason(chains, n_evaluations_budget)
        if !isnothing(reason)
            stop_reason = reason
            break
        end
    end
    out = clustered_output(chains, clusters, n_windows)
    write_clustered_run_summary(checkpoint_dir, out, stop_reason)
    out
end

# --- On-disk checkpointing (opt-in, `checkpoint_dir=`) ------------------------
#
# Deliberately the SAME layout and payload contract as `cooperative_warmup_mcmc`
# (`checkpoint_schema_version`, whose policy is explicitly "shared by ALL
# samplers"), so one consumer read path serves every sampler that checkpoints:
# per-chain `chain_<i>/cp_window_<n>.jls` + `cp_latest.jls`, plus the write-once
# JSON files at the `checkpoint_dir` root.
#
# **No count of those JSON files is given here on purpose.** The set is defined
# in another file as much as in this one, so a third file added to the
# cooperative sampler would falsify a number written here without its author
# ever opening this file. A census in prose cannot fail — it silently returns a
# subset. Ask instead:
#
#     grep -rn '_write_json(' src/
#
# `_write_json` is the single writer every write-once root JSON goes through, so
# a site cannot exist without matching it. The filenames are the wrong anchor
# for the same reason: a third file is by definition named something the pattern
# does not already know.
#
# `chain_index` is the chain's position in the `chains` vector. The cooperative
# scheduler needs a STORED index because it installs chains in completion order;
# here `clustered_chains` builds the vector once and never reorders it, so
# position is a stable identity for the life of the run.
#
# Like every other payload in this package this EXCLUDES `lpdf`, and that is
# load-bearing here rather than incidental: `ClusteredChain` holds `const lpdf`,
# which naive whole-struct serialization would drag in (possibly a
# non-serializable BridgeStan handle). Selecting fields into a NamedTuple
# sidesteps it, which is why checkpoint WRITING needs no change to the struct —
# only RESUME, which must reconstruct a chain, still does.

"""
    clustered_checkpoint_payload(chain, chain_index, window) -> NamedTuple

Serializable snapshot of a clustered chain at a window boundary, in the shared
checkpoint contract (`sampler === :clustered`).

Carries `cluster_id` — the contract's clustered-only key — recording which pool
this chain sampled under AS OF this checkpoint. Like `status` it is never
backfilled: a chain that moves pools at window 5 still reads its old
`cluster_id` in `cp_window_4.jls`. Written checkpoints stay immutable, which is
what lets a consumer pin a `(config, checkpoint#)` view and trust an old file.

`stuck_reason` is always `nothing` and `is_final` never reports abandonment:
the clustered sampler restarts chains onto a new pooled scale rather than
abandoning them, so it has no `:stuck` status. The keys are present anyway so a
generic reader needs no per-sampler branch.

Excludes `lpdf` (resume re-supplies it), exactly as the adaptive and cooperative
payloads do.
"""
clustered_checkpoint_payload(chain::ClusteredChain, chain_index::Int, window::Int) = (;
    schema_version=checkpoint_schema_version(),
    sampler=:clustered,
    chain_index,
    status=chain.status,
    stuck_reason=nothing,
    is_final=chain.status === :done,
    stop_reason=chain.status === :done ? :n_draws : :running,
    window,
    cluster_id=chain.cluster_id,
    halo_position=chain.recording_lpdf.halo_position,
    halo_gradient=chain.recording_lpdf.halo_gradient,
    posterior_position=chain.recording_lpdf.posterior_position,
    posterior_gradient=chain.recording_lpdf.posterior_gradient,
    recorder=chain.recording_lpdf.recorder,
    chain.rng, chain.position_and_gradient, chain.scale, chain.stepsize,
    chain.stepsize_state, chain.n_evaluations, chain.adaptation,
    chain.total_evaluation_counter, chain.current_transition_counter,
    chain.n_divergent_samples, chain.n_samples, chain.n_draws, chain.dimension,
    chain.checkpoints,
    chain.dropped_posterior_position, chain.dropped_posterior_gradient,
    chain.dropped_n_divergent_samples,
    reparam_sources=reparam_sources(chain.lpdf),
    custom_candidate_scoring=_has_custom_candidate_scoring(chain.lpdf),
)

"""
    restore_clustered_chain(payload, lpdf; n_draws, stepsize_adaptation_limit,
                            target_acceptance_rate, max_tree_depth,
                            max_window_evaluations, weighting, kwargs...) -> ClusteredChain

Reconstruct a live [`ClusteredChain`](@ref) from a deserialized clustered
checkpoint payload and a freshly supplied `lpdf` (one `deepcopy` per chain, as
on the fresh path) — the clustered counterpart of `restore_state`, under the
same payload invariant: learned state from the payload, configuration from THIS
call.

The restored chain carries its accumulating pooled-estimate input
(`adaptation`) with it, so a resumed ensemble re-clusters from the same
estimates the crash interrupted — per-chain resume is meaningless here except
as part of the whole-ensemble restore (`cluster_and_adapt!` pools across
cluster-mates).

* `n_draws` is a total: a chain restored `:done` resumes `:sampling` when the
  call raises it past the retained draws (resume-and-extend).
* `recording_target`, `regularizing_n` and `regularizing_var` default to
  `nothing`, which inherits the checkpoint's values; an explicit different value
  is refused — all three are baked into persisted accumulators.
* `weighting` has no payload copy (an arbitrary function is not serializable
  state) and always comes from the call: re-pass a custom weighting on resume,
  or the resumed windows accumulate under the default.
* `n_evaluations`, `init`, `progress` and `pathfinder_kw` are accepted and
  ignored — the window budget continues from the payload, and initialization
  already happened — so the resuming call can repeat the original verbatim.
"""
restore_clustered_chain(p, lpdf;
    expected_chain_index=nothing,   # pool resume passes the slot; direct callers omit it
    n_draws=1000,
    n_evaluations=1000,             # ignored: the payload's (doubled) budget continues
    recording_target=nothing,       # nothing inherits the checkpoint's ring size
    stepsize_adaptation_limit=50,
    target_acceptance_rate=.8,
    max_tree_depth=10,
    max_window_evaluations=4000,
    init=missing,                   # ignored: initialization already happened
    regularizing_n=nothing,         # nothing inherits the checkpoint's prior strength
    regularizing_var=nothing,       # nothing inherits the checkpoint's prior scale
    weighting=default_weighting,    # always from the call — see above
    progress=nothing,               # ignored: init-only handle
    pathfinder_kw=(;),              # ignored: init-only, accepted so the call repeats verbatim
    kwargs...                       # init-only remainder (see `clustered_chain`); dropped, not stored
) = begin
    check_checkpoint_compatible(p, :clustered, (:clustered,))
    _check_schema_version(p)
    # Slot check AFTER the tag check: a foreign payload has no `chain_index` at
    # all, and its masquerade error is the informative one.
    found_index = get(p, :chain_index, nothing)
    isnothing(expected_chain_index) || found_index == expected_chain_index || throw(ArgumentError(
        "chain_$expected_chain_index holds " *
        (isnothing(found_index) ? "a payload with no chain index" :
            "a checkpoint for chain $found_index") *
        ": chain directories were moved or mixed. Restore them and resume again."
    ))
    _check_candidate_scoring_compatible(p, lpdf)
    lpdf_dimension = LogDensityProblems.dimension(lpdf)
    p.dimension == lpdf_dimension || throw(DimensionMismatch(
        "checkpoint holds a $(p.dimension)-dimensional problem but the supplied " *
        "lpdf has dimension $lpdf_dimension."
    ))
    # Same rule as `restore_state`: the recorder's target sizes a persisted ring,
    # so it can only be inherited, never changed.
    isnothing(recording_target) || recording_target == p.recorder.target || throw(ArgumentError(
        "`recording_target` cannot change on resume (checkpoint has " *
        "$(p.recorder.target), got $recording_target): the recorder's ring buffer " *
        "and its retained contents are part of the persisted state. Omit it to inherit."
    ))
    # The regularizing pair is the prior baked into the persisted pooled-estimate
    # accumulator — changing it mid-run would silently re-weight every estimate.
    saved_regularizing = p.adaptation.position_variances
    for (name, value, saved) in (
        (:regularizing_n, regularizing_n, saved_regularizing.regularizing_n),
        (:regularizing_var, regularizing_var, saved_regularizing.regularizing_var),
    )
        isnothing(value) || value == saved || throw(ArgumentError(
            "`$name` cannot change on resume (checkpoint has $saved, got $value): " *
            "it is baked into the persisted pooled-estimate accumulator. Omit it to inherit."
        ))
    end
    restore_reparam_sources!(lpdf, p.reparam_sources)
    recording_lpdf = RecordingPosterior2(
        lpdf, p.halo_position, p.halo_gradient, p.posterior_position, p.posterior_gradient,
        NUTSLeaves(p.dimension), p.recorder, p.rng,
    )
    # Resume-and-extend, as on the other two samplers.
    status = p.status === :done && size(p.posterior_position, 2) < n_draws ? :sampling : p.status
    ClusteredChain(
        p.rng, lpdf, recording_lpdf,
        DynamicHMC.NUTS(; max_depth=max_tree_depth),
        DynamicHMC.DualAveraging(δ=target_acceptance_rate), p.dimension,
        p.recorder.target, stepsize_adaptation_limit, n_draws, max_window_evaluations,
        weighting,
        p.position_and_gradient, p.scale, _diagonal_energy(p.scale),
        p.stepsize, p.stepsize_state, p.n_evaluations, p.adaptation,
        p.cluster_id, p.total_evaluation_counter,
        p.current_transition_counter, p.n_divergent_samples, p.n_samples,
        status, p.checkpoints,
        p.dropped_posterior_position, p.dropped_posterior_gradient,
        p.dropped_n_divergent_samples,
    )
end

# Rebuild the clustered ensemble from a run directory: slot `i` restores from
# `chain_<i>/`, or starts fresh in place when it has no restorable checkpoint
# (crash before its first window completed — its previous occupant wrote
# nothing, so reusing the slot cannot mix two chains' draws). The ensemble is
# fixed — unlike the cooperative scheduler nothing refills it later — so every
# slot is filled here. Returns `(chains, windows_done)`, where `windows_done`
# is the highest completed window over the restored chains (fresh slots
# contribute 0); the resumed run's `max_windows` counts from there as a total.
_restore_clustered_pool(dir, rngs, lpdf; n_draws, weighting, init, progress, pathfinder_kw, kwargs...) = begin
    idxs = _chain_dir_indices(dir)
    n = max(length(rngs), isempty(idxs) ? 0 : maximum(idxs))
    inits = ensurevector(init, n)
    chains = Vector{ClusteredChain}(undef, n)
    windows_done = 0
    for i in 1:n
        p = _read_chain_payload(dir, i)
        if isnothing(p)
            i > length(rngs) && throw(ArgumentError(
                "chain_$i has no checkpoint and needs a fresh start, but only " *
                "$(length(rngs)) rngs were supplied. Pass at least $i rngs to resume."
            ))
            @warn "chain_$i has no checkpoint (crash before its first write); starting it fresh" maxlog=10
            chains[i] = clustered_chain(rngs[i], deepcopy(lpdf);
                n_draws, weighting, init=inits[i], progress, pathfinder_kw, kwargs...)
        else
            chains[i] = restore_clustered_chain(p, deepcopy(lpdf);
                expected_chain_index=i, n_draws, weighting, init=inits[i], pathfinder_kw, kwargs...)
            windows_done = max(windows_done, p.window)
        end
    end
    (chains, windows_done)
end

# The bound a clustered ensemble has met, checked in the run loop's order, or
# `nothing` to keep going.
_clustered_stop_reason(chains, n_evaluations_budget) =
    all(c -> c.status === :done, chains) ? :n_draws :
    sum(c -> c.total_evaluation_counter, chains) >= n_evaluations_budget ? :eval_budget :
    nothing

# Partition chain indices by their as-checkpointed `cluster_id`, for a resumed
# run with no windows left to run (no re-clustering happens, so the output
# partition comes from the payloads). Ordered by id, so the output's per-cluster
# `cluster_id` (its position) agrees with each chain's own `cluster_id` —
# `cluster_and_adapt!` numbers clusters 1..K. Fresh slots carry `cluster_id == 0`
# — never pooled — so each stands alone, after the pooled clusters.
_clusters_from_ids(chains) = begin
    ids = sort!(unique(c.cluster_id for c in chains if c.cluster_id > 0))
    clusters = [[i for (i, c) in enumerate(chains) if c.cluster_id == id] for id in ids]
    append!(clusters, [[i] for (i, c) in enumerate(chains) if c.cluster_id <= 0])
end

# Atomic temp+rename per file, so a crash mid-write cannot strand a truncated
# checkpoint — same writer the adaptive and cooperative paths use.
_write_clustered_checkpoints(::Nothing, chains, window) = nothing
_write_clustered_checkpoints(dir::AbstractString, chains::AbstractVector{<:ClusteredChain}, window::Int) = begin
    for (i, chain) in enumerate(chains)
        d = joinpath(dir, "chain_$i")
        mkpath(d)
        payload = clustered_checkpoint_payload(chain, i, window)
        _atomic_serialize(joinpath(d, "cp_window_$window.jls"), payload)
        _atomic_serialize(joinpath(d, "cp_latest.jls"), payload)
    end
    nothing
end

"Write `run_manifest.json` once, at run start. Never rewritten (see the cooperative writer for why)."
write_clustered_run_manifest(::Nothing, rngs, cfg) = nothing
write_clustered_run_manifest(dir::AbstractString, rngs, cfg) = _write_json(
    joinpath(dir, "run_manifest.json"), Pair{String,Any}[
        "schema_version" => checkpoint_schema_version(),
        "sampler" => "clustered",
        "n_chains_requested" => length(rngs),
        # The stopping criteria ACTUALLY in force; JSON has no Inf, so an unset
        # bound reads as null rather than a lie.
        "n_draws" => cfg.n_draws,
        "max_windows" => cfg.max_windows,
        "n_evaluations_budget" => cfg.n_evaluations_budget == typemax(Int) ? nothing : cfg.n_evaluations_budget,
        "threshold" => cfg.threshold,
        "chain_rngs" => map(string, rngs),
        "start_time_unix" => time(),
    ])

"""
    write_clustered_run_summary(dir, out, stop_reason) -> nothing

Write `run_summary.json` once, at finalize. Its EXISTENCE is the run-completed
signal, so a consumer answering "is this run done?" needs no parsing.

Carries `cluster_assignments` — the final pooling. That is genuinely run-level
state: a per-chain checkpoint records the `cluster_id` a chain believed it was
in, but only the run knows the partition those ids resolve to.
"""
write_clustered_run_summary(::Nothing, out, stop_reason) = nothing
write_clustered_run_summary(dir::AbstractString, out, stop_reason) = _write_json(
    joinpath(dir, "run_summary.json"), Pair{String,Any}[
        "schema_version" => checkpoint_schema_version(),
        "sampler" => "clustered",
        "is_final" => true,
        "stop_reason" => string(stop_reason),
        "n_windows" => out.n_windows,
        "n_chains" => length(out.results),
        "n_clusters" => length(out.clusters),
        "cluster_assignments" => [r.cluster_id for r in out.results],
        "total_evaluation_counter" => out.total_evaluation_counter,
        "end_time_unix" => time(),
    ])

"""
    clustered_chains(rngs, lpdf; n_draws=1000, weighting=default_weighting,
                     init=missing, parallel=true, progress=nothing, kwargs...) -> Vector{ClusteredChain}

Build the per-chain state for a clustered run — the resumable handle. Each chain
gets a `deepcopy` of `lpdf`. Advance the result with [`clustered_step!`](@ref).
"""
clustered_chains(rngs::AbstractVector, lpdf; n_draws=1000, weighting=default_weighting,
    init=missing, parallel=true, progress=nothing, pathfinder_kw=(;), kwargs...) = begin
    _check_kwargs(:clustered_chains, kwargs)
    n_chains = length(rngs)
    inits = ensurevector(init, n_chains)
    chains = Vector{ClusteredChain}(undef, n_chains)
    _pforeach(parallel, 1:n_chains) do i
        chains[i] = clustered_chain(rngs[i], deepcopy(lpdf); n_draws, weighting, init=inits[i], progress, kwargs..., pathfinder_kw...)
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
