abstract type AbstractIterativeSampler end

"""
A cooperative (as in parallel) iterative (as in resumable) sampler.

Configure via e.g. `sampler = WarmupHMC.CooperativeIterativeSampler(lpdf, Xoshiro.(1:32))`
where `lpdf` satisfies the `LogDensityProblems.jl` interface and the second argument `Xoshiro.(1:32)` determines the number of chains.

The main configuration options include 

* `n_stepsize_adaptations`: the number of transitions reserved for adapting the step size,
* `n_evaluations_per_chain`: the number of gradient evaluations in the initial batch (doubles with every batch/iteration),
* `jitter`: a distribution of the LOG-jitter or `nothing` for no jitter after step size adaptation (mainly useful to visualize the optimization landscape).


Use either via 
```
for batch in sampler 
    # do something with the current batch of draws
end
```

or, automatically resuming from and storing persistent state at `path`, via

```
WarmupHMC.sample_resumably(sampler; path) do state 
    # do something with the current state

    # returning `false` or `nothing` stops sampling 
    false
end
```
"""
struct CooperativeIterativeSampler{C<:NamedTuple} <: AbstractIterativeSampler
    config::C
    CooperativeIterativeSampler(config::NamedTuple) = new{typeof(config)}(config)
end
CooperativeIterativeSampler(
    lpdf, rngs;
    regularizing_n=0, n_stepsize_adaptations=100, max_refinements=0, jitter=nothing, compatibility_threshold=sqrt(2),
    n_evaluations_per_chain=100, progress=nothing
) = CooperativeIterativeSampler((;
    lpdf, rngs, regularizing_n, n_stepsize_adaptations, max_refinements, jitter, compatibility_threshold, n_evaluations_per_chain, progress
))

# Base.iterate(s::AbstractIterativeSampler; kwargs...) = Base.iterate(s, initial_state(s; kwargs...); kwargs...)
Base.iterate(s::AbstractIterativeSampler, ::Nothing=nothing; kwargs...) = Base.iterate(s, initial_state(s; kwargs...); kwargs...)

progressasyncmap(f, it; progress, kwargs...) =  with_progress(progress, length(it); kwargs...) do progress
    asyncmap(it) do i
        rv = f(i, progress)
        update_progress!(progress)
        rv
    end
end
progressmap(f, it; progress, kwargs...) =  with_progress(progress, length(it); kwargs...) do progress
    map(it) do i
        rv = f(i, progress)
        update_progress!(progress)
        rv
    end
end
# progressasyncmap!(f, results, it=results; progress, kwargs...) =  with_progress(progress, length(it); kwargs...) do progress
#     asyncmap!(results, it) do i
#         rv = f(i, progress)
#         update_progress!(progress)
#         rv
#     end
# end

initial_state(s::CooperativeIterativeSampler; progress=s.config.progress, kwargs...) = begin 
    (;lpdf, rngs, regularizing_n, n_evaluations_per_chain) = s.config
    n_chains = length(rngs)
    dim = LogDensityProblems.dimension(lpdf)
    chains = progressasyncmap(1:n_chains; progress, description="Initialize chains") do idx, progress
        rng = rngs[idx]
        scale = Diagonal(ones(dim))
        problem = AdaptiveNUTSPosterior(
            NUTSPosterior(
                PreconditionedNUTSPosterior(deepcopy(lpdf), scale); 
                R=composite_recorder(:everything)
            )
        )
        (;position) = initialize_mcmc(problem, missing; rng, progress)
        position_and_gradient = DynamicHMC.evaluate_ℓ(problem, position; strict=true)
        initial_stepsize = find_initial_stepsize(problem, position_and_gradient; rng)
        stepsize_adaptation = SquaredJumpStepsizeAdaptation(initial_stepsize)
        scale_adaptation = IntermediateScaleAdaptation(dim; regularizing_n)
        draws = ElasticMatrix(zeros((dim, 0)))
        is_busy = false
        n_evaluations = 0
        n_transitions = 0
        final_stepsize = 0
        lock = ReentrantLock()
        cluster = 0
        (;
            scale, problem, position, position_and_gradient, 
            initial_stepsize, stepsize_adaptation, scale_adaptation, draws, is_busy,
            n_evaluations, n_transitions, final_stepsize, lock, cluster,
        )
    end
    clusters = progressasyncmap(1:0; progress, description="Initialize clusters") do idx, progress
        chains = Int64[]
        lock = ReentrantLock()
        stepsize_adaptation = SquaredJumpStepsizeAdaptation(0.)
        scale_adaptation = IntermediateScaleAdaptation(dim; regularizing_n)
        (;stepsize_adaptation, scale_adaptation, lock, chains)
    end
    max_depth = trunc(Int, log2(n_evaluations_per_chain))
    iteration = 0
    (;
        n_chains, dim, n_evaluations_per_chain, max_depth, iteration, 
        chains, clusters, 
    )
end
Base.iterate(s::CooperativeIterativeSampler, state::NamedTuple; progress=s.config.progress, transient=true) = begin
    (;n_stepsize_adaptations, max_refinements, regularizing_n, jitter, compatibility_threshold) = s.config
    (;
        n_chains, dim, n_evaluations_per_chain, max_depth, iteration, 
        chains, clusters, 
    ) = state
    tasks = Dict{Int,Task}()
    target_evaluations = n_chains * n_evaluations_per_chain
    iteration_lock = ReentrantLock()
    n_evaluations = Ref(0)
    progressasyncmap(eachindex(clusters)) do idx, progress
        error(clusters[idx].chains)
    end
    # if final_stepsize[] == 0
    #     stepsize_adaptation = SquaredJumpStepsizeAdaptation(initial_stepsize)
    #     scale_adaptation = IntermediateScaleAdaptation(dim; regularizing_n)
    #     for draws_ in draws
    #         resize!(draws_, (dim, 0))
    #     end
    # end
    iteration += 1
    with_progress(progress, target_evaluations; description="CooperativeIterativeSampler($n_chains@$iteration)", transient) do progress
        start_time = time_ns()
        while true 
            chain_idx = 0
            @lock iteration_lock begin 
                n_evaluations[] >= target_evaluations && break
                for idx in 1:n_chains
                    @lock chains[idx].lock begin
                        is_busy[idx] && continue
                        if chain_idx == 0 || n_evaluations[chain_idx] > n_evaluations[idx]
                            chain_idx = idx
                        end
                    end
                end
                if chain_idx != 0
                    is_busy[chain_idx] = true
                end
            end
            if chain_idx == 0
                all(istaskfailed, values(tasks)) && error("All tasks failed!", map(fetch, values(tasks)))
                sleep(.001)
                continue
            end
            tasks[chain_idx] = Threads.@spawn let chain_idx = $chain_idx
                local stepsize = @lock chain_lock if final_stepsize[] == 0
                    propose!(stepsize_adaptation; q=.99)
                else
                    append!(draws[chain_idx], position_and_gradients[chain_idx].q)
                    if isnothing(jitter)
                        final_stepsize[]
                    else
                        final_stepsize[] * exp(rand(rngs[chain_idx], jitter))
                    end
                end
                position_and_gradients[chain_idx] = sample!(
                    problems[chain_idx], position_and_gradients[chain_idx]; rng=rngs[chain_idx], stepsize, max_depth, max_refinements
                )
                @lock chain_lock begin
                    is_busy[chain_idx] = false
                    n_evaluations[chain_idx] += n_steps(problems[chain_idx])
                    n_transitions[chain_idx] += 1
                    local dt = time_ns()-start_time
                    update_progress!(
                        progress, min(sum(n_evaluations), target_evaluations);
                        sampling_performance=SamplingPerformance(final_stepsize[], sum(n_evaluations) / sum(n_transitions)),
                        usable_draws_counter=Speed(sum(Base.Fix2(size, 2), draws), dt),
                        total_evaluation_counter=Speed(sum(n_evaluations), dt),
                        total_transition_counter=Speed(sum(n_transitions), dt),
                    )
                    fit!(scale_adaptation, problems[chain_idx], position_and_gradients[chain_idx])
                    fit!(stepsize_adaptation, problems[chain_idx]; stepsize)
                    if nobs(stepsize_adaptation) == n_stepsize_adaptations && final_stepsize[] == 0.
                        final_stepsize[] = finalize!(stepsize_adaptation)
                    end
                end
            end
        end
    end
    @lock chain_lock begin 
        cc = cond_compatibility(scale_adaptation, scale)
        sc = stepsize_compatibility!(stepsize_adaptation, final_stepsize[])
        if cc * sc > compatibility_threshold
            min_prev = minimum(parent(scale))
            parent(scale) .= marginal_scales!(scale_adaptation)
            initial_stepsize = finalize!(stepsize_adaptation) * sqrt(min_prev / minimum(parent(scale)))
            final_stepsize[] = 0.
            max_depth = trunc(Int, log2(n_evaluations_per_chain))
        end
        update_progress!(progress;
            relative_condition_number=cc,
            potential_step_size_gain=sc,
            restarted=final_stepsize[] == 0.
        )
        n_evaluations_per_chain *= 2
        (;draws), (;
            scale, problems, rngs, n_chains, dim, position_and_gradients, initial_stepsize, stepsize_adaptation, scale_adaptation,
            n_evaluations_per_chain, max_depth, iteration,
            draws, is_busy, n_evaluations, final_stepsize
        )
    end
end

function store end
restore(sampler::AbstractIterativeSampler, path::AbstractString) = restore(sampler, restore(path))
restore(::AbstractIterativeSampler, ::Nothing) = nothing
restore(sampler::CooperativeIterativeSampler, state::NamedTuple) = begin
    state.is_busy .= false 
    state.problems .= [
        AdaptiveNUTSPosterior(
            NUTSPosterior(
                PreconditionedNUTSPosterior(deepcopy(sampler.config.lpdf), state.scale); 
                R=composite_recorder(:everything)
            )
        )
         for _ in 1:state.n_chains
    ]
    state
end
sample_resumably(callback, sampler::AbstractIterativeSampler, n_iterations; path, progress=sampler.config.progress) = with_progress(progress, n_iterations) do progress
    state = restore(sampler, path)
    (;iteration) = something(state, (;iteration=0))
    update_progress!(progress, iteration)
    !isnothing(state) && something(callback(state), false) && return state
    for i in 1+iteration:n_iterations
        _, state = iterate(sampler, state; progress)
        update_progress!(progress, i)
        store(path, state)
        something(callback(state), false) && break
    end
    state
end


# Base.iterate(s::IterativeSampler, state::NamedTuple) = begin 
#     stepsize = propose!(stepsize_adaptation)
#     position_and_gradient = sample!(
#         problem, position_and_gradient; rng, stepsize, max_depth, max_refinements
#     )
#     n_evaluations += n_steps(problem)
#     n_transitions += 1
#     fit!(scale_adaptation, problem, position_and_gradient)
#     fit!(stepsize_adaptation, problem; stepsize)
#     if nobs(stepsize_adaptation) == n_stepsize_adaptations && final_stepsize == 0.
#         final_stepsize[] = finalize!(stepsize_adaptation)
#     end
#     # update_progress!(progress)
# end
whilefirst(f, state) = while true#!isnothing(state)
    cond, state = f(state)
    cond || return state
end
maybereset!!(chain) = chain
step!!(state::NamedTuple; kwargs...) = step!!(state.sampler, state; kwargs...)
step!!(sampler::AbstractIterativeSampler; kwargs...) = step!!(sampler, initial_state(sampler); kwargs...)

struct ParallelStepsizeAdaptation2{P<:AbstractStepsizeAdaptation,L<:Base.AbstractLock} <: AbstractStepsizeAdaptation
    parent::P
    lock::L
end
Base.parent(a::ParallelStepsizeAdaptation2) = a.parent
Base.lock(a::ParallelStepsizeAdaptation2) = lock(a.lock)
Base.unlock(a::ParallelStepsizeAdaptation2) = unlock(a.lock)
propose!(a::ParallelStepsizeAdaptation2, args...; kwargs...) = @lock a propose!(parent(a), args...; kwargs...)
OnlineStatsBase.fit!(a::ParallelStepsizeAdaptation2, args...; kwargs...) = @lock a OnlineStatsBase.fit!(parent(a), args...; kwargs...)
finalize!(a::ParallelStepsizeAdaptation2, args...; kwargs...) = @lock a finalize!(parent(a), args...; kwargs...)

struct ClusteredIterativeSampler{C<:NamedTuple} <: AbstractIterativeSampler
    config::C
    ClusteredIterativeSampler(config::NamedTuple) = new{typeof(config)}(config)
end
initial_state(s::ClusteredIterativeSampler) = begin 
    chains = map(s.config.samplers) do sampler
        tmp = initial_state(sampler)
        merge(tmp, (;
            stepsize_adaptation=ParallelStepsizeAdaptation2(tmp.stepsize_adaptation, ReentrantLock()),
            cluster_idx=0
        ))
    end
    (;chains, target_evaluations=sum(chain->chain.target_evaluations, chains))
end
struct IterativeSampler{C<:NamedTuple} <: AbstractIterativeSampler
    config::C
    IterativeSampler(config::NamedTuple) = new{typeof(config)}(config)
end
IterativeSampler(
    lpdf;
    rng, regularizing_n=0, n_stepsize_adaptations=100, max_refinements=0, jitter=nothing, compatibility_threshold=sqrt(2),
    target_evaluations=1000, progress=nothing
) = IterativeSampler((;
    lpdf, rng, regularizing_n, n_stepsize_adaptations, max_refinements, jitter, compatibility_threshold, target_evaluations, progress
))

initial_state(s::IterativeSampler) = begin 
    (;lpdf, rng, regularizing_n, target_evaluations) = s.config
    dim = LogDensityProblems.dimension(lpdf)
    scale = Diagonal(ones(dim))
    problem = AdaptiveNUTSPosterior(
        NUTSPosterior(
            PreconditionedNUTSPosterior(deepcopy(lpdf), scale); 
            R=composite_recorder(:everything)
        )
    )
    position_and_gradient = DynamicHMC.EvaluatedLogDensity(zeros(dim), -Inf, zeros(dim))
    initial_stepsize = 1.#find_initial_stepsize(problem, position_and_gradient; rng)
    stepsize_adaptation = SquaredJumpStepsizeAdaptation(initial_stepsize)
    scale_adaptation = IntermediateScaleAdaptation(dim; regularizing_n)
    draws = ElasticMatrix(zeros((dim, 0)))
    n_evaluations = 0
    n_transitions = 0
    final_stepsize = 0
    iteration = 0
    (;
        sampler=s, rng, target_evaluations,
        scale, problem, position_and_gradient, 
        initial_stepsize, stepsize_adaptation, scale_adaptation, draws, 
        n_evaluations, n_transitions, final_stepsize, iteration, 
        max_depth=10, max_refinements=0, stable_for=0
    )
end
step!!(s::IterativeSampler, state::NamedTuple; progress=s.config.progress) = begin
    (;
        scale_adaptation, stepsize_adaptation, position_and_gradient, problem, rng, max_depth, max_refinements,
        n_evaluations, n_transitions, draws
    ) = state
    if !isfinite(position_and_gradient.ℓq)
        (;position) = initialize_mcmc(problem; rng, progress)
        position_and_gradient = DynamicHMC.evaluate_ℓ(problem, position; strict=true)
    end
    frozen, stepsize = propose!(stepsize_adaptation)
    position_and_gradient = sample!(problem, position_and_gradient; rng, stepsize, max_depth, max_refinements)
    n_evaluations += n_steps(problem)
    n_transitions += 1
    fit!(scale_adaptation, problem, position_and_gradient)
    fit!(stepsize_adaptation, problem; stepsize)
    frozen && append!(draws, position_and_gradient.q)
    update_progress!(progress, n_evaluations; stepsize, n_evaluations, n_draws=size(draws, 2), n_transitions)
    merge(state, (;stepsize, position_and_gradient, n_evaluations, n_transitions))
end
step!!(s::ClusteredIterativeSampler, state::NamedTuple; progress=s.config.progress, transient=false) = with_progress(progress, state.target_evaluations) do iprogress
    (;chains, target_evaluations) = state
    n_evaluations = Threads.Atomic{Int}(0)
    n_chains = length(chains)
    chains = progressasyncmap(chains; progress=iprogress, transient, description="chains") do chain, cprogress
        with_progress(cprogress, target_evaluations ÷ n_chains; transient, description="$((;chain.cluster_idx, chain.stable_for))") do eprogress
            whilefirst(maybereset!!(chain)) do chain 
                pre_n_evaluations = chain.n_evaluations
                chain = step!!(chain; progress=eprogress)
                Threads.atomic_add!(n_evaluations, chain.n_evaluations - pre_n_evaluations)
                update_progress!(iprogress, n_evaluations[])
                n_evaluations[] < target_evaluations => chain
            end
        end
    end
    cluster!(chains)
    (;chains, target_evaluations=2*target_evaluations)
end
findminval(f, domain) = mapfoldl(v->(f(v), v), _rf_findminval, domain)
_rf_findminval((fm, im), (fx, ix)) = Base.isgreater(fm, fx) ? (fx, ix) : (fm, im)
cluster!(chains) = begin
    clusters = []
    remaining = Set(eachindex(chains)) 
    while length(remaining) > 0
        candidates = copy(remaining)
        scale_adaptation = merge(map(idx->chains[idx].scale_adaptation, collect(candidates))...)
        stepsize_adaptation = ParallelStepsizeAdaptation2(SquaredJumpStepsizeAdaptation(1.), ReentrantLock())
        while true
            val, idx = findminval(candidates) do idx 
                WarmupHMC.compatibility2(chains[idx].scale_adaptation.adaptations[2], scale_adaptation) 
            end
            if val > 0 || length(candidates) == 1
                push!(clusters, (;candidates, scale_adaptation, stepsize_adaptation))
                setdiff!(remaining, candidates)
                break
            else
                WarmupHMC.unmerge!(scale_adaptation, chains[idx].scale_adaptation.adaptations[2])
                pop!(candidates, idx)
            end
        end
    end
    sort!(clusters; by=c->length(c.candidates), rev=true)
    for (cluster_idx, cluster) in enumerate(clusters)
        (;candidates, scale_adaptation) = cluster
        for idx in candidates
            (;scale, stepsize_adaptation) = chain = chains[idx]
            cc = cond_compatibility(scale_adaptation, scale)
            restart = cc > sqrt(2)
            if restart
                min_prev = minimum(parent(scale))
                parent(scale) .= marginal_scales(scale_adaptation) 
                initial_stepsize = finalize!(stepsize_adaptation)[2] * sqrt(min_prev / minimum(parent(scale)))
                reset!(chain.scale_adaptation)
                reset!(chain.draws)
                chains[idx] = merge(
                    chain, 
                    (;cluster_idx, stable_for=0, initial_stepsize, cluster.stepsize_adaptation)
                )
            else
                chains[idx] = merge(
                    chain, 
                    (;cluster_idx, stable_for=chain.stable_for+1)
                )
            end
        end
    end
end

