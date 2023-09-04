module WarmupHMC

export regularize, to_x1, to_xc, klp, klps, approximately_whitened, mcmc_with_reparametrization, ConvenientLogDensityProblem#, klps_plot!

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
    dt0=1e-6, rng=Xoshiro(0), n_parameters = LogDensityProblems.dimension(logdensity), n_repetitions=1, n_iterations=n_parameters, x=randn(rng, n_parameters),
    dt_speedup=8, dt_mul=1. / dt_speedup, pack=do_nothing,
    twosided=false, vinit=:random, vrefresh=:all, sparse=:false, escalate_velocity=false,
    tinit=:uniform, vscale=:uniform, mh=true
)
    lpdfg(x) = LogDensityProblems.logdensity_and_gradient(logdensity, x)
    
    dt = collect(Diagonal(fill(dt_mul, n_parameters)))

    lp, a = lpdfg(x)
    if tinit == :igradient
        dt = collect(Diagonal(dt_mul * abs.(minimum(a) ./ a)))
    end
    v = if vinit == :random
        randn(rng, n_parameters)
    elseif vinit == :igradient
        1 ./ a
    else
        0 .* a
    end
    pack(lp, x, v, a, 1.)
    oscale = missing
    for iteration in 1:n_iterations
        if vscale == :igradient
            v = v ./ (dt * a)
        elseif vscale isa Real
            v = v ./ abs.(dt * a).^vscale
        end
        dt[iteration:end, :] *= dt0
        
        # dx = dt' * (v + .5 * (dt*a))
        dxv = dt' * v
        dxa = dt' * (dt * .5a)
        xr = x + dxv + dxa
        lpr, ar = lpdfg(xr)
        vr = v + .5 * (dt * (a + ar))

        dx, da, dir = if twosided == true || twosided == :first && iteration == 1
            xl = x - dxv + dxa
            al = lpdfg(xl)
            xr-xl,ar-al,ar-2a+al
        else
            xr-x,ar-a,ar-a
        end
        acceptance_rate = exp((.5sum(vr.^2) - lpr) - (.5sum(v.^2) - lp))
        accept = !mh || rand(rng) < acceptance_rate
        if accept
            lp = lpr
            x = xr
            v = vr
            a = ar
        end
        pack(lp, x, v, a, acceptance_rate)
        if vrefresh == :all || !accept
            v = randn(rng, n_parameters)
        else
            if vrefresh == :fast
                v[1:iteration] .= randn(rng, iteration)
            elseif vrefresh == :slow
                v[iteration:end] .= randn(rng, 1+n_parameters-iteration)
            end
        end
        dx = dt * dx
        da = dt * da
        dir = dt * dir
        dir[1:iteration-1] .= 0
        if sparse
            idx = argmax(abs.(dir))
            dir .= 0
            dir[idx] = 1
        else
            dir = dir |> normalize
        end
        scale = sqrt(abs(dot(dir, dx) / dot(dir, da)))
        rscale = ismissing(oscale) ? 1 : (scale / oscale)
        oscale = scale
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
        if escalate_velocity
            v[iteration+1:end] *= rscale
        end
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


struct Ignore <: Real end
Base.:+(lhs::Ignore, ::Real) = lhs
Base.:-(lhs::Ignore, ::Real) = lhs

reparametrization_parameters(::Any) = []
unconstrained_reparametrization_parameters(source::Any) = reparametrization_parameters(source)
reparametrize(source::Any, ::Any) = source
unconstrained_reparametrize(source::Any, parameters::AbstractVector) = reparametrize(source, parameters)
lja_reparametrize(source, target, draws::AbstractMatrix) = begin 
    rv = lja_reparametrize.([source], [target], eachcol(draws))
    first.(rv), hcat(last.(rv)...)
end
lja(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[1]
lja(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[1]
# lja(source, target, draws::AbstractMatrix) = lja.([source], [target], eachcol(draws))
reparametrize(source::Any, target::Any, draw::AbstractVector) = lja_reparametrize(source, target, draw)[2]
reparametrize(source::Any, target::Any, draws::AbstractMatrix) = lja_reparametrize(source, target, draws)[2]
# reparametrize(source, target, draws::AbstractMatrix) = hcat(
    # reparametrize.([source], [target], eachcol(draws))...
# )
reparametrization_loss(source, target, draws::AbstractMatrix) = begin 
    ljas, reparametrized = lja_reparametrize(source, target, draws)
    mean(ljas) + sum(log.(std(reparametrized, dims=2)))
end
reparametrization_loss_function(source, draws::AbstractMatrix) = begin 
    loss(v) = reparametrization_loss(source, reparametrize(source, v), draws)
end
unconstrained_reparametrization_loss_function(source, draws::AbstractMatrix) = begin 
    loss(v) = reparametrization_loss(source, unconstrained_reparametrize(source, v), draws)
end
find_reparametrization(source, ::AbstractMatrix) = source
find_reparametrization(kind::Symbol, source, draws::AbstractMatrix; kwargs...) = find_reparametrization(Val{kind}(), source, draws; kwargs...)
mcmc_with_reparametrization(args...; kwargs...) = missing


logdensity_and_stuff(source, draw::AbstractVector) = (
    LogDensityProblems.logdensity(source, draw), nothing
)
logdensity_and_stuff(source, draws::AbstractMatrix) = begin 
    rv = logdensity_and_stuff.([source], eachcol(draws))
    first.(rv), last.(rv)
end

struct ConvenientLogDensityProblem{P,L,I}
    prior::P
    likelihood::L
    draw_boundaries::Vector{I}
    parameter_boundaries::Vector{I}
end
ConvenientLogDensityProblem(prior, likelihood) = begin
    ConvenientLogDensityProblem(
        prior, likelihood,  
        vcat(0, cumsum(LogDensityProblems.dimension.(prior))), 
        vcat(0, cumsum(length.(reparametrization_parameters.(prior)))), 
    )
end
LogDensityProblems.dimension(source::ConvenientLogDensityProblem) = sum(LogDensityProblems.dimension.(source.prior))
subdraws(source::ConvenientLogDensityProblem, draw::AbstractVector) = view.([draw], range.(1 .+ source.draw_boundaries[1:end-1], source.draw_boundaries[2:end]))
subparameters(source::ConvenientLogDensityProblem, parameters) = view.([parameters], range.(1 .+ source.parameter_boundaries[1:end-1], source.parameter_boundaries[2:end]))
reparametrization_parameters(source::ConvenientLogDensityProblem) = vcat(
    reparametrization_parameters.(source.prior)...
)

@views LogDensityProblems.logdensity(source::ConvenientLogDensityProblem, draw::AbstractVector) = begin 
    intermediates = logdensity_and_stuff.(
        source.prior, subdraws(source, draw)
    )
    sum(first.(intermediates)) + sum(source.likelihood(last.(intermediates)...))
end
reparametrize(source::ConvenientLogDensityProblem, parameters) = ConvenientLogDensityProblem(reparametrize.(source.prior, subparameters(source, parameters)), source.likelihood, source.draw_boundaries, source.parameter_boundaries)
reparametrize(source::ConvenientLogDensityProblem, target::ConvenientLogDensityProblem, draw::AbstractVector) = vcat(reparametrize.(source.prior, target.prior, subdraws(source, draw))...)
lja(source::ConvenientLogDensityProblem, target::ConvenientLogDensityProblem, draw::AbstractVector) = sum(lja.(source.prior, target.prior, subdraws(source, draw)))
end

