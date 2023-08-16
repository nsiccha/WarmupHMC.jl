module WarmupHMC

export regularize, to_x1, to_xc, klp, klps, approximately_whitened#, klps_plot!

using DynamicObjects
using Random, Distributions, LinearAlgebra
using LogDensityProblems

regularize(sample_covariance, no_draws, regularization_no_draws=5, regularization_constant=1e-3) = (
  no_draws / ((no_draws + regularization_no_draws) * (no_draws - 1)) * sample_covariance + regularization_constant * (regularization_no_draws / (no_draws + regularization_no_draws)) * I
)

# to_x1(xc, log_sd, centeredness) = xc * exp(log_sd * (1 - centeredness))
# xc = mu * c + sigma^c * x0
# x0 = (x1 - mu) / sigma
# xc = mu * c + sigma^(c-1) * (x1 - mu)
to_xc(x1, mean, log_sd, centeredness) = (
    mean * centeredness + exp(log_sd * (centeredness - 1)) * (x1 - mean)
)
to_x1(xc, mean, log_sd, centeredness) = (
    mean + (xc - mean * centeredness) * exp(log_sd * (1 - centeredness))
)
to_xc(previous_xc, mean, log_sd, target_centeredness, previous_centeredness) = (
    mean * target_centeredness 
    + exp(log_sd * (target_centeredness - previous_centeredness)) 
        * (previous_xc - mean * previous_centeredness)
)
# to_xc(
#     to_x1(xcp, mean, log_sd, previous_centeredness),
#     mean, log_sd, target_centeredness
# )
# to_xc(x1, log_sd, centeredness) = x1 * exp(log_sd * (centeredness - 1))
# klp(x1, log_sd, centeredness) = klp(to_xc.(x1, log_sd, centeredness), exp.(log_sd .* centeredness))
klp(x1s, means, log_sds, centeredness) = klp(
    to_xc.(x1s, means, log_sds, centeredness), 
    means .* centeredness, 
    exp.(log_sds .* centeredness)
)
klp(xc, meanc, sdc) = mean(logpdf.(Normal.(meanc, sdc), xc)) + log(std(xc))
klps(x1s, means, log_sds, cs) = [klp(x1s, means, log_sds, c) for c in cs]

# @dynamic_object DiagonalScaling <: Reparametrization scaling::Vector
# reparametrize(what::DiagonalScaling, parameters) = what.scaling .* parameters
# unreparametrize(what::DiagonalScaling, reparameters) = what.scaling .\ reparameters
# logjacobian(what::DiagonalScaling, ::Any, ::Any) = 0

struct HouseholderReflector{T} <: AbstractMatrix{T}
    v::Vector{T}
end
HouseholderReflector(source::AbstractVector, target::AbstractVector) = HouseholderReflector(normalize(source - target))
HouseholderReflector(source::AbstractVector, target::Integer) = begin 
    e = zeros(length(source))
    e[target] = 1
    HouseholderReflector(source, e)
end
Base.:*(lhs::HouseholderReflector, rhs::AbstractVector) = rhs - 2 * lhs.v * (lhs.v' * rhs)
Base.:*(lhs::HouseholderReflector, rhs::AbstractMatrix) = rhs - 2 * lhs.v * (lhs.v' * rhs)
Base.:*(lhs::HouseholderReflector, rhs::Diagonal) = rhs - 2 * lhs.v * (lhs.v' * rhs)
Base.transpose(lhs::HouseholderReflector) = lhs
Base.adjoint(lhs::HouseholderReflector) = lhs
Base.size(lhs::HouseholderReflector) = (length(lhs.v), length(lhs.v))

do_nothing(args...; kwargs...) = nothing
projected_out(x, n::AbstractVector) = x - n * (n' * x)

function approximate_whitening(
    logdensity; 
    dt0=1e-6, rng=Xoshiro(0), n_parameters = LogDensityProblems.dimension(logdensity), n_iterations=n_parameters, x=randn(rng, n_parameters),
    dt_speedup=8, dt_mul=1. / dt_speedup, pack=do_nothing,
    twosided=false, vinit=:random, vrefresh=:all
)
    lpdfg(x) = LogDensityProblems.logdensity_and_gradient(logdensity, x)[2]
    
    dt = collect(Diagonal(fill(dt_mul, n_parameters)))

    a = lpdfg(x)
    v = if vinit == :random
        randn(rng, n_parameters)
    elseif vinit == :igradient
        1 ./ a
    else
        0 .* a
    end
    pack(x, a)
    for iteration in 1:n_iterations
        dt[iteration:end, :] *= dt0
        
        # dx = dt' * (v + .5 * (dt*a))
        dxv = dt' * v
        dxa = dt' * (dt * .5a)
        xr = x + dxv + dxa
        ar = lpdfg(xr)
        if vrefresh == :all
            v = randn(rng, n_parameters)
        else
            v += .5 * (dt * (a + ar))
            if vrefresh == :fast
                v[1:iteration] .= randn(rng, iteration)
            elseif vrefresh == :slow
                v[iteration:end] .= randn(rng, 1+n_parameters-iteration)
            end
        end

        dx, da, dir = if twosided == true || twosided == :first && iteration == 1
            xl = x - dxv + dxa
            al = lpdfg(xl)
            xr-xl,ar-al,ar-2a+al
        else
            xr-x,ar-a,ar-a
        end
        x = xr
        a = ar
        pack(x, a)
        dx = dt * dx
        da = dt * da
        dir = dt * dir
        dir[1:iteration-1] .= 0
        dir = dir |> normalize
        scale = sqrt(abs(dot(dir, dx) / dot(dir, da)))
        if dir[iteration] != 1
            hr = HouseholderReflector(dir, iteration)
            dt = hr * dt
            if vrefresh != :all
                v = hr * v
            end
        end
        dt[iteration:end, :] ./= dt0
        dt[iteration, :] .*= scale
        dt0 = dt_speedup*scale
    end
    dt ./= dt_mul
    return dt
end

struct ProductPosterior{T}
    dists::Vector{T}
end
LogDensityProblems.dimension(what::ProductPosterior) = length(what.dists)
LogDensityProblems.capabilities(::Type{<:ProductPosterior}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(what::ProductPosterior, x) = sum(logpdf.(what.dists, x))
LogDensityProblems.logdensity_and_gradient(what::ProductPosterior, x) = LogDensityProblems.logdensity(what, x), gradlogpdf.(what.dists, x)

struct ScaledLogDensity{L, T} 
    logdensity::L
    transform::T
end
# ScaledLogDensity(what::ScaledLogDensity, transform) = ScaledLogDensity(
#     what.logdensity, what.transform * transform
# )
LogDensityProblems.dimension(what::ScaledLogDensity) = LogDensityProblems.dimension(
    what.logdensity
)
LogDensityProblems.capabilities(::Type{<:ScaledLogDensity{L,T}}) where {L,T} = LogDensityProblems.capabilities(L)
transform(what::ScaledLogDensity, xi) = what.transform * xi
LogDensityProblems.logdensity(what::ScaledLogDensity, xi) = LogDensityProblems.logdensity(what.logdensity, transform(what, xi))
LogDensityProblems.logdensity_and_gradient(what::ScaledLogDensity, xi) = begin 
    l, g = LogDensityProblems.logdensity_and_gradient(what.logdensity, transform(what, xi))
    l, what.transform' * g
end

approximately_whitened(logdensity; kwargs...) = ScaledLogDensity(
    logdensity, approximate_whitening(logdensity; kwargs...)'
)



end
