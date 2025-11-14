
# abstract type AbstractJointAdaptation end

# struct ScaleStepsizeAdaptation{S<:AbstractScaleAdaptation,SS<:AbstractStepsizeAdaptation} <: AbstractJointAdaptation
#     scale::S
#     stepsize::SS
# end

# OnlineStatsBase.fit!(a::ScaleStepsizeAdaptation, problem; scale, stepsize) = begin
#     OnlineStatsBase.fit!(a.scale, problem)
#     OnlineStatsBase.fit!(a.stepsize, problem; stepsize, factor=1.)
#     # # lmin, lmax = extrema(parent(scale) ./ marginal_scales!(a.scale))
#     # # cond = lmax / lmin
#     # # @assert all(>(0), parent(scale))
#     # # @assert all(>(0), marginal_scales!(a.scale)) a.scale
#     # shift = log((minimum(parent(scale)) ./ minimum(marginal_scales!(a.scale))))
#     # @assert isfinite(shift)
#     # mul = cond(Diagonal(collect(marginal_scales!(a.scale)))) / cond((scale))
#     # @assert isfinite(mul)
#     # # @info (;lmin, lmax, cond, shift, mul)
#     # gp = a.stepsize.gp[]
#     # gp.inducing_x .+= shift
#     # gp.cache.x .+= shift
#     # gp.cache.sum .*= mul/exp(shift)
#     # gp.cache.sum_abs2 .*= (mul/exp(shift))^2
#     # refit!(gp)
#     # if ispow2(length(gp.cache.x)) 
#     #     a.stepsize.gp[] = rescale!!(gp; n=length(gp.cache.x))
#     # end
#     a
# end