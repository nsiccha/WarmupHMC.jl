abstract type AbstractStepsizeAdaptation end

# struct DualAveraging end
struct BayesianOptimizationStepsizeAdaptation{T,B,S,D} <: AbstractStepsizeAdaptation 
    soft_threshold::T
    hard_threshold::T
    p::Vector{Int}
    dlinear::Vector{T}
    dzero::Vector{T}
    log_stepsizes::Vector{T}
    bo_gp::B
    sel_gp::S
    div_gp::D
end
exp_then_zero(x, threshold) = x < threshold ? exp(x) : zero(x)
one_then_zero(x, threshold) = x <= threshold ? one(x) : zero(x)

rescale!!((;soft_threshold, hard_threshold, p, dlinear, dzero, log_stepsizes, bo_gp, sel_gp, div_gp)::BayesianOptimizationStepsizeAdaptation; n=0, kwargs...) = BayesianOptimizationStepsizeAdaptation(
    soft_threshold, hard_threshold, p, dlinear, dzero, log_stepsizes, rescale!!(bo_gp; n, kwargs...), sel_gp, rescale!!(div_gp; n, kwargs...)
)
observe!!(a::BayesianOptimizationStepsizeAdaptation; stepsize, dH, weighted_jump, weighted_angle, log_stepsize=log(stepsize)) = begin 
    (;soft_threshold, hard_threshold, p, dlinear, dzero, log_stepsizes, bo_gp, div_gp) = a
    # (maximum(abs, dH) >= 1000 && length(dH) == 1) && (hard_threshold = min(hard_threshold, log_stepsize))
    idx = searchsortedfirst(view(log_stepsizes, p), log_stepsize)
    if idx > length(p) || log_stepsizes[p[idx]] != log_stepsize
        insert!(p, idx, length(p)+1)
        push!(log_stepsizes, log_stepsize)
        push!(dlinear, @bsum(abs2(weighted_jump - .5 * stepsize)))
        push!(dzero, sum(abs2, weighted_jump))
        soft_threshold, loss_star = log_stepsizes[p[1]]-log(2), 0.
        loss = 0.
        for (ii, i) in enumerate(p)
            loss += dlinear[i] - dzero[i]
            if loss <= loss_star
                soft_threshold, loss_star = log_stepsizes[i], loss
                # soft_threshold, loss_star = log_stepsizes[p[min(ii+1, length(p))]], loss
            end
        end 
    else
        # condition!(bo_gp, log_stepsize, weighted_angle)
        condition!(bo_gp, log_stepsize, [mean(weighted_angle)])
        condition!(div_gp, log_stepsize, 1e-32 + maximum(Base.Fix2(min, 1e12) ∘ abs, dH))
        return a
        dlinear[p[idx]] += @bsum(abs2(weighted_jump - .5 * stepsize))
        dzero[p[idx]] += sum(abs2, weighted_jump)
    end
    (;bo_gp, div_gp) = rescale!!(a)
    bo_gp = if maximum(abs, dH) >= 1000# && length(dH) == 1
        # @assert log_stepsize < hard_threshold
        hard_threshold = min(log_stepsize, hard_threshold)
        prepare!!(bo_gp, log_stepsize)
        # prepare!!(
        #     bo_gp, 
        #     inducing_x(
        #         min(hard_threshold-log(2), bo_gp.inducing_x[1]), 
        #         # min(hard_threshold+log(2), bo_gp.inducing_x[end]),
        #         hard_threshold, 
        #         x_scale(bo_gp.kernel); 
        #         n_pad=0
        #     )
        # )
    else
        prepare!!(bo_gp, log_stepsize)
    end
    # bo_gp = prepare!!(bo_gp, log_stepsize)#Base.Fix2(one_then_zero, soft_threshold),))
    # div_gp = prepare!!(div_gp, bo_gp.inducing_x)
    div_gp = prepare!!(div_gp, log_stepsize)
    # @info prescale => y_scale(div_gp.kernel)
    # condition!(bo_gp, log_stepsize, weighted_angle)
    condition!(bo_gp, log_stepsize, [mean(weighted_angle)])
    # mean(weighted_angle) > 0 && condition!(bo_gp, log_stepsize, [log(mean(weighted_angle))])
    # display(maximum(Base.Fix2(min, 1e12) ∘ abs, dH))
    condition!(div_gp, log_stepsize, 1e-32 + maximum(Base.Fix2(min, 1e12) ∘ abs, dH))
    BayesianOptimizationStepsizeAdaptation(
        soft_threshold, hard_threshold, p, dlinear, dzero, log_stepsizes, bo_gp, a.sel_gp, div_gp
    )
end
propose!(a::BayesianOptimizationStepsizeAdaptation) = begin 
    (;soft_threshold, hard_threshold, log_stepsizes, p) = a
    isfinite(hard_threshold) || return 2 * exp(log_stepsizes[p[end]])
    hard_threshold == log_stepsizes[p[1]] && return .5 * exp(log_stepsizes[p[1]])
    # soft_threshold >= log_stepsizes[p[end]] && return 2 * exp(log_stepsizes[p[end]])
    # soft_threshold <= log_stepsizes[p[1]] && return .5 * exp(log_stepsizes[p[1]])

    x_star = x = log_stepsizes[p[1]] - log(2)
    utility_star = utility!(a, x)
    for i in p
        cx = .5 * (x + log_stepsizes[i])
        x = log_stepsizes[i]
        # cx >= hard_threshold && return exp(x_star)
        utility = utility!(a, cx)
        if utility > utility_star
            x_star = cx
            utility_star = utility
        end
    end
    cx = x + log(2)
    utility = utility!(a, cx)
    if utility > utility_star
        x_star = cx
        utility_star = utility
    end
    return exp(x_star)
end
select!(a::BayesianOptimizationStepsizeAdaptation) = exp(argmax(Base.Fix1(selection_utility!, a), a.log_stepsizes))
divergence_risk(div_gp, log_stepsize; q=.5, threshold=1000) = ccdf(
    Normal(
        quantile(mlocation!(div_gp, log_stepsize), q), 
        obsscale!(div_gp.clr)
    ), 
    div_gp.link(log_stepsize, threshold)
)
divergence_risks(div_gp, log_stepsize; q=.05, threshold=1000) = ccdf.(
    Normal.(
        quantile.(Ref(mlocation!(div_gp, log_stepsize)), (q, 1-q)), 
        obsscale!(div_gp.clr)
    ), 
    div_gp.link(log_stepsize, threshold)
)
utility!(
    (;soft_threshold, hard_threshold, p, log_stepsizes, bo_gp, div_gp)::BayesianOptimizationStepsizeAdaptation, log_stepsize; 
    scale_fac=2, stepsize_pow=1, dx_pow=0, n_steps=1000, n_divergent=0, lower=-Inf
) = if log_stepsize >= hard_threshold
    0.
else
    lidx = searchsortedlast(view(log_stepsizes, p), log_stepsize)
    dx = if lidx == 0
        log_stepsizes[p[1]] - log_stepsize
    elseif lidx == length(p)
        log_stepsize - log_stepsizes[p[end]]
    else
        .5 * (log_stepsizes[p[lidx+1]] - log_stepsizes[p[lidx]])
    end
    # return qlocation!(bo_gp, log_stepsize, .95) * exp(stepsize_pow*log_stepsize)*dx^dx_pow
    loc, scale = locscale!(bo_gp, log_stepsize)
    # ls = locscale!(div_gp, log_stepsize)
    # div_risk = 1 - cdf(Normal(2log_stepsize + ls[1], sqrt(ls[2]^2+obsscale!(div_gp.clr)^2)), log(1000))
    lo, hi = divergence_risks(div_gp, log_stepsize)
    # @info (log_stepsize=>(;hi, lo, dp=hi-lo))
    div_risk = 1-pcdf!(div_gp, log_stepsize, 1000)
    # if log_stepsize > soft_threshold-log(2)/2
    #     log_stepsize = soft_threshold-log(2)/2
    #     scale = min(scale, locscale!(bo_gp, log_stepsize)[2])
    # end
    dist = truncated(
        Normal(
            (loc+scale_fac*scale)*exp(stepsize_pow*log_stepsize), 
            obsscale!(bo_gp.clr)*exp(stepsize_pow*log_stepsize)
        );
        lower
    )
    return mean(dist)*cdf(Binomial(n_steps, lo), n_divergent)*exp(dx_pow*dx)#^dx_pow
    (loc+scale_fac*scale)*exp(stepsize_pow*log_stepsize)*cdf(Binomial(n_steps, lo), n_divergent)#*dx^dx_pow#*(hi-lo)
end

selection_utility!(
    (;hard_threshold, soft_threshold, log_stepsizes, bo_gp, sel_gp, div_gp)::BayesianOptimizationStepsizeAdaptation, log_stepsize; 
    scale_fac=-2, stepsize_pow=1, n_steps=1000, n_divergent=0, lower=0. #   -Inf
) = if log_stepsize >= hard_threshold
    0
else
    loc, scale = locscale!(bo_gp, log_stepsize)
    lo, hi = divergence_risks(div_gp, log_stepsize)
    div_risk = 1-pcdf!(div_gp, log_stepsize, 1000)
    dist = truncated(
        Normal(
            (loc+scale_fac*scale)*exp(stepsize_pow*log_stepsize), 
            obsscale!(bo_gp.clr)*exp(stepsize_pow*log_stepsize)
        );
        lower
    )
    return mean(dist)*cdf(Binomial(n_steps, hi), n_divergent)
    # div_risk = 0
    # log_stepsize = min(log_stepsize, soft_threshold-log(2)/2)
    # return max(0, location!(bo_gp, log_stepsize))*exp(log_stepsize)*cdf(Binomial(n_steps, div_risk), n_divergent)
    (loc+scale_fac*scale)*exp(stepsize_pow*log_stepsize)#*cdf(Binomial(n_steps, hi), n_divergent)#*(hi-lo)#clamp(1 - 100div_risk, 0, 1)
end