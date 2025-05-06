regweights(w0, w1; f=mean) = if length(w0) == 0
    0.
else
    n1 = 1 + sum(w0) do wi
        logistic(wi-w1)
    end
    n2 = 2 + length(w0) - n1
    f(Beta(n1, n2))
end
passchanged(old, new) = new, old != new
adaptive_pathfinder(lpdf; n_eval, n_chains, rng=Random.default_rng(), warn=true, progress=nothing, threshold=1e-2) = with_progress(progress, n_eval; description="Adaptive pathfinder") do progress
    warn && @warn """
        This method will try to find good initialization points by repeatedly discarding initalization points it deems bad - however, we may be throwing out import modes!

        If run in parallel, this method is not reproducible!

        Disable this warning by passing warn=false.
    """
    l = ReentrantLock()
    active_chains = UInt64[]
    max_median = -Inf
    median_changes = UncertainFrequency(1, 1)
    n_compute_chains = 2 * n_chains
    update_progress!(progress, nothing; 
        final_weights="pending...", 
        sampling_weights="pending...", 
        betas="pending...",
        n_compute_chains,
        median_changes,
    )
    chains = Dict{UInt64,Union{}}()
    dlpdf(i, x=chains[i].fit_distribution.μ) = LogDensityProblems.logdensity(lpdf, x) - logpdf(chains[i].fit_distribution, x)
    total_n = 0
    betas() = map(idx->chains[idx].beta[], active_chains)
    final_weights() = normalize(map(idx->chains[idx].final_weight[], active_chains), 1)
    sampling_weights() = StatsBase.weights(map(idx->chains[idx].sampling_weight[], active_chains))
    update_chain!(chain, x; median_changed) = if length(chain.log_weights) > 0 
        beta = if !median_changed
            (;α, β) = chain.beta[]
            dalpha = logistic(x - max_median)
            Beta(α + dalpha, β + (1-dalpha))
        else
            alpha = 1 + sum(chain.log_weights) do log_weight
                logistic(log_weight - max_median)
            end
            Beta(alpha, 2 + length(chain.log_weights) - alpha)
        end
        chain.beta[] = beta 
        chain.sampling_weight[] = if mean(beta) < threshold 
            # @info "Discarding chain..."
            if chain.sampling_weight[] > 0
                n_compute_chains += 1
            end
            0.
        else
            quantile(beta, .99)
        end
        chain.final_weight[] = quantile(beta, .01)
    end
    Threads.@threads for i_ in 1:n_chains
        x = zeros(LogDensityProblems.dimension(lpdf))
        while total_n < n_eval
            i, new = lock(l) do
                i, new = if length(active_chains) < n_compute_chains
                    rand(rng, UInt64), true
                else
                    sample(active_chains, sampling_weights()), false
                end
                if new
                    chains = BangBang.setindex!!(chains, (;
                        fit_distribution=missing,
                        log_weights=Float64[],
                        median=Ref(-Inf),
                        sampling_weight=Ref(0.),
                        final_weight=Ref(0.),
                        beta=Ref(Beta(1.,1.))
                    ), i)
                    push!(active_chains, i)
                end
                i, new
            end
            if new 
                pr = with_progress(progress, 1_000; description="Pathfinder.$i", transient=true) do pprogress
                    pathfinder(lpdf; ndraws=1, callback=pathfinder_callback(pprogress))
                end
                lock(l) do 
                    chains = BangBang.setindex!!(chains, BangBang.setproperty!!(chains[i], :fit_distribution, pr.fit_distribution), i)
                end
            end
            rvi = if new
                dlpdf(i)
            else
                dlpdf(i, Random.rand!(chains[i].fit_distribution, x))
            end
            lock(l) do
                total_n += 1
                insert!(chains[i].log_weights, searchsortedfirst(chains[i].log_weights, rvi), rvi)
                chains[i].median[] = quantile(chains[i].log_weights, .5; sorted=true) 
                max_median, median_changed = passchanged(max_median, maximum(idx->chains[idx].median[], active_chains))
                median_changes = UncertainFrequency(median_changes.obs + median_changed, median_changes.n+1)
                if !median_changed
                    update_chain!(chains[i], rvi; median_changed)
                else
                    for idx in active_chains
                        # (idx == i || chains[idx].sampling_weight[] > 0) || continue 
                        update_chain!(chains[idx], rvi; median_changed)
                    end
                end
                weights = final_weights()
                p = sortperm(weights, rev=true)
                update_progress!(
                    progress, total_n; 
                    final_weights=weights[p], 
                    sampling_weights=sampling_weights()[p], 
                    betas=betas()[p],
                    n_compute_chains=UncertainFrequency(sum(chain->chain.final_weight[]>1e-2, values(chains)), n_compute_chains),
                    median_changes
                )
            end
        end
    end
    chains = [chains[active_chains[pi]] for pi in sortperm(final_weights(), rev=true)]
    chains = filter(chain->chain.final_weight[]>1e-2, chains)
    (warn && length(chains) < n_chains) && @warn "Only $(length(chains)) chains remaining out of $n_chains."
    map(1:n_chains) do idx
        chain = chains[(idx-1) % length(chains)+1]
        (;
            position=rand(rng, chain.fit_distribution), 
            location=chain.fit_distribution.μ, 
            squared_scale=chain.fit_distribution.Σ
        )
    end
end