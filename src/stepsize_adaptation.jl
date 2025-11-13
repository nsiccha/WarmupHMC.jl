abstract type AbstractStepsizeAdaptation end
OnlineStatsBase.value(a::AbstractStepsizeAdaptation) = select!(a)

struct NoStepsizeAdaptation{T} <: AbstractStepsizeAdaptation
    stepsize::T
end
OnlineStatsBase.fit!(a::NoStepsizeAdaptation, args...) = a
OnlineStatsBase.value(a::NoStepsizeAdaptation) = a.stepsize
propose!(a::NoStepsizeAdaptation) = a.stepsize

struct DualAveragingStepsizeAdaptation{T}  <: AbstractStepsizeAdaptation
    config::DynamicHMC.DualAveraging{T}
    state::Ref{DynamicHMC.DualAveragingState{T}}
    DualAveragingStepsizeAdaptation(stepsize::Real; target_acceptance_rate=.8) = DualAveragingStepsizeAdaptation(
        DynamicHMC.DualAveraging(δ=target_acceptance_rate),
        stepsize
    )
    DualAveragingStepsizeAdaptation(config::DynamicHMC.DualAveraging, stepsize::Real) = new{typeof(stepsize)}(
        config, Ref(DynamicHMC.initial_adaptation_state(config, stepsize))
    )
end
OnlineStatsBase.fit!(a::DualAveragingStepsizeAdaptation, lpdf::AbstractNUTSPosterior; kwargs...) = OnlineStatsBase.fit!(a, exp(log_acceptance_rate(lpdf)))
OnlineStatsBase.fit!(a::DualAveragingStepsizeAdaptation, acceptance_rate::Real) = begin 
    a.state[] = DynamicHMC.adapt_stepsize(a.config, a.state[], acceptance_rate)
    a
end
OnlineStatsBase.fit!(a::DualAveragingStepsizeAdaptation, ::NUTSPosterior, stats) = OnlineStatsBase.fit!(
    a, stats.acceptance_rate
)
propose!(a::DualAveragingStepsizeAdaptation) = DynamicHMC.current_ϵ(a.state[])
select!(a::DualAveragingStepsizeAdaptation) = DynamicHMC.final_ϵ(a.state[])


abstract type AbstractStepsizeRegression end

OnlineStatsBase.fit!(a::AbstractStepsizeRegression, lpdf::AbstractNUTSPosterior; stepsize) = begin
    a.gp[] = prepare!!(a.gp[], log(stepsize))
    if ispow2(length(a.gp[].cache.n))
        a.gp[] = rescale!!(a.gp[]; n=length(a.gp[].cache.n))
    end
    condition!(a, lpdf; stepsize)
end
OnlineStatsBase.nobs(a::AbstractStepsizeRegression) = nobs(a.gp[])

struct SquaredJumpStepsizeRegression{G<:IPGPRegression} <: AbstractStepsizeRegression
    gp::Ref{G}
end
eff_link(x,y) = y/exp(.5x)
eff_unlink(x, y) = y * exp(.5x)
ylink(::typeof(eff_link)) = eff_unlink
SquaredJumpStepsizeRegression(stepsize::Real; x_scale=.5, y_scale=.5, link=eff_link, functions=(one, )) = SquaredJumpStepsizeRegression(Ref(IPGPRegression(
    log(stepsize/2), log(stepsize*2);
    kernel=SquaredExponentialKernel(log(2)*x_scale, y_scale), 
    link, functions
)))
condition!(a::SquaredJumpStepsizeRegression, lpdf::AbstractNUTSPosterior; stepsize) = begin 
    N = ylink(a.gp[], log(stepsize), 1/n_steps(lpdf))
    condition!(a.gp[], log(stepsize); n=1, sum=expected_squared_jump(lpdf)*N, sum_abs2=expected_quad_jump(lpdf)*N^2)
end

struct LogAcceptanceRateStepsizeRegression{G<:IPGPRegression} <: AbstractStepsizeRegression
    gp::Ref{G}
end
acc_link(x, y) = clamp(logitexp(y), -8, +8)
acc_unlink(x, y) = logistic(y)
ylink(::typeof(acc_link)) = acc_unlink
LogAcceptanceRateStepsizeRegression(stepsize::Real; x_scale=2, y_scale=1.) = LogAcceptanceRateStepsizeRegression(Ref(IPGPRegression(
    log(stepsize/2), log(stepsize*2);
    kernel=IntegratedSquaredExponentialKernel(2, log(2)*x_scale, y_scale), 
    link=acc_link, functions=(one, Base.Fix2(-, log(stepsize)))
)))
condition!(a::LogAcceptanceRateStepsizeRegression, lpdf::AbstractNUTSPosterior; stepsize) = condition!(
    a.gp[], log(stepsize), log_acceptance_rate(lpdf)
)

struct SquaredJumpStepsizeAdaptation{S<:SquaredJumpStepsizeRegression,L<:LogAcceptanceRateStepsizeRegression,C} <: AbstractStepsizeAdaptation
    sj::S
    lar::L
    cache::C
end
SquaredJumpStepsizeAdaptation(stepsize) = SquaredJumpStepsizeAdaptation(
    SquaredJumpStepsizeRegression(stepsize),
    LogAcceptanceRateStepsizeRegression(stepsize),
    (;
        queue=Set(stepsize), visited=Set{typeof(stepsize)}(), 
        max_sj=Ref(-Inf), max_lar=Ref(-Inf)
    )
)
loss!(a::SquaredJumpStepsizeAdaptation, stepsize; q=.5, ar_penalty=.5) = begin 
    log_stepsize = log(stepsize)
    min(
        ylink(
            a.sj.gp[], log_stepsize; f=(gp, x)->qlocation!(gp, x, q)
        ), 
        a.cache.max_sj[]
    ) * min(
        ylink(
            a.lar.gp[], log_stepsize; f=(gp, x)->qlocation!(gp, x, q)
        ),
        ar_penalty
    )
end
propose!(a::SquaredJumpStepsizeAdaptation; q=.99, ar_penalty=.5, max_factor=64) = begin 
    rv = if a.cache.max_sj[] == -Inf
        maximum(a.cache.queue)
    elseif a.cache.max_lar[] < log(ar_penalty)
        minimum(a.cache.queue)
    else
        argmax(a.cache.queue) do stepsize
            loss!(a, stepsize; q, ar_penalty)
        end
    end
    push!(a.cache.visited, rv)
    pop!(a.cache.queue, rv)
    if rv == minimum(a.cache.visited) 
        push!(a.cache.queue, .5 * rv)
    else
        push!(a.cache.queue, sqrt(rv * maximum(filter(<(rv), a.cache.visited))))
    end
    if rv == maximum(a.cache.visited) 
        rv < max_factor * minimum(a.cache.visited) && push!(a.cache.queue, 2 * rv)
    else
        push!(a.cache.queue, sqrt(rv * minimum(filter(>(rv), a.cache.visited))))
    end
    rv
end
select!(a::SquaredJumpStepsizeAdaptation; q=.5, kwargs...) = propose!(a::SquaredJumpStepsizeAdaptation; q, kwargs...)
OnlineStatsBase.fit!(a::SquaredJumpStepsizeAdaptation, lpdf::AbstractNUTSPosterior; stepsize) = begin 
    a.cache.max_sj[] = max(a.cache.max_sj[], expected_squared_jump(lpdf) / n_steps(lpdf))
    a.cache.max_lar[] = max(a.cache.max_lar[], log_acceptance_rate(lpdf))
    fit!(a.sj, lpdf; stepsize)
    fit!(a.lar, lpdf; stepsize)
end
stepsize_compatibility!(a::SquaredJumpStepsizeAdaptation, stepsize) = if stepsize == 0
    Inf
else
    loss!(a, select!(a)) / loss!(a, stepsize)
end

OnlineStatsBase.nobs(a::SquaredJumpStepsizeAdaptation) = nobs(a.sj)
