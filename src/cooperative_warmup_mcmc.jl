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
    n_evaluations_per_chain=100
) = CooperativeIterativeSampler((;
    lpdf, rngs, regularizing_n, n_stepsize_adaptations, max_refinements, jitter, compatibility_threshold, n_evaluations_per_chain
))

Base.iterate(s::AbstractIterativeSampler) = Base.iterate(s, initial_state(s))
Base.iterate(s::AbstractIterativeSampler, ::Nothing) = Base.iterate(s, initial_state(s))

initial_state(s::CooperativeIterativeSampler) = begin 
    (;lpdf, rngs, regularizing_n, n_evaluations_per_chain) = s.config
    n_chains = length(rngs)
    dim = LogDensityProblems.dimension(lpdf)
    scale = Diagonal(ones(dim))
    problems = [
        AdaptiveNUTSPosterior(
            NUTSPosterior(
                PreconditionedNUTSPosterior(deepcopy(lpdf), scale); 
                R=composite_recorder(:everything)
            )
        )
         for _ in 1:n_chains
    ]
    position_and_gradients = asyncmap(1:n_chains) do idx 
        (;position) = initialize_mcmc(problems[idx], missing; rng=rngs[idx], progress=nothing)
        DynamicHMC.evaluate_ℓ(problems[idx], position; strict=true)
    end
    initial_stepsize = median(asyncmap(1:n_chains) do idx 
        find_initial_stepsize(problems[idx], position_and_gradients[idx]; rng=rngs[idx])
    end)
    stepsize_adaptation = SquaredJumpStepsizeAdaptation(initial_stepsize)
    scale_adaptation = IntermediateScaleAdaptation(dim; regularizing_n)
    max_depth = trunc(Int, log2(n_evaluations_per_chain))
    iteration = 1
    draws = [ElasticMatrix(zeros((dim, 0))) for _ in 1:n_chains]
    is_busy = fill(false, n_chains)
    n_evaluations = fill(0, n_chains)
    final_stepsize = Ref(0.)
    (;
        scale, problems, rngs, n_chains, dim, position_and_gradients, initial_stepsize, stepsize_adaptation, scale_adaptation,
        n_evaluations_per_chain, max_depth, iteration,
        draws, is_busy, n_evaluations, final_stepsize
    )
end
Base.iterate(s::CooperativeIterativeSampler, state::NamedTuple) = begin
    (;n_stepsize_adaptations, max_refinements, regularizing_n, jitter, compatibility_threshold) = s.config
    tasks = Dict{Int,Task}()
    (;
        scale, problems, rngs, n_chains, dim, position_and_gradients, initial_stepsize, stepsize_adaptation, scale_adaptation,
        n_evaluations_per_chain, max_depth, iteration,
        draws, is_busy, n_evaluations, final_stepsize
    ) = state
    target_evaluations = n_chains * n_evaluations_per_chain
    n_evaluations .= 0
    chain_lock = ReentrantLock()
    if final_stepsize[] == 0
        stepsize_adaptation = SquaredJumpStepsizeAdaptation(initial_stepsize)
        scale_adaptation = IntermediateScaleAdaptation(dim; regularizing_n)
        for draws_ in draws
            resize!(draws_, (dim, 0))
        end
    end
    ProgressLogging.@withprogress name="CooperativeIterativeSampler($n_chains@$iteration)" while true 
        chain_idx = 0
        @lock chain_lock begin 
            sum(n_evaluations) >= target_evaluations && break
            for idx in 1:n_chains
                is_busy[idx] && continue
                if chain_idx == 0 || n_evaluations[chain_idx] > n_evaluations[idx]
                    chain_idx = idx
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
                ProgressLogging.@logprogress sum(n_evaluations) / target_evaluations
                fit!(scale_adaptation, problems[chain_idx], position_and_gradients[chain_idx])
                fit!(stepsize_adaptation, problems[chain_idx]; stepsize)
                if nobs(stepsize_adaptation) == n_stepsize_adaptations && final_stepsize[] == 0.
                    final_stepsize[] = select!(stepsize_adaptation)
                end
            end
        end
    end
    @lock chain_lock begin 
        cc = cond_compatibility(scale_adaptation, scale)
        sc = stepsize_compatibility!(stepsize_adaptation, final_stepsize[])
        if cc * sc > compatibility_threshold
            @info "Restarting sampling @ $iteration $((;cc, sc))"
            min_prev = minimum(parent(scale))
            parent(scale) .= marginal_scales!(scale_adaptation)
            initial_stepsize = select!(stepsize_adaptation) * sqrt(min_prev / minimum(parent(scale)))
            final_stepsize[] = 0.
            max_depth = trunc(Int, log2(n_evaluations_per_chain))
        else
            # @info "Continuing sampling @ $iteration $((;cc, sc))"
        end
        n_evaluations_per_chain *= 2
        iteration += 1
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
sample_resumably(callback, sampler::AbstractIterativeSampler; path) = begin 
    state = restore(sampler, path)
    while isnothing(state) || something(callback(state), false)
        _, state = iterate(sampler, state)
        store(path, state)
    end
    state
end