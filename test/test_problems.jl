# Shared LogDensityProblems fixtures + helpers for the WarmupHMC test suite.
# Kept at top level (struct defs + method extensions can't live inside a @testset).

"""
    DiagGaussian(mu, sigma)

A `LogDensityProblems`-compatible diagonal Gaussian target with an analytic
gradient. Matches `Distributions.MvNormal(mu, Diagonal(sigma.^2))` exactly
(the normalizing constant is included). Used as a cheap, gradient-providing
inner problem for the reparametrization / AD / end-to-end tests.
"""
struct DiagGaussian{V<:AbstractVector}
    mu::V
    sigma::V
end
LogDensityProblems.capabilities(::Type{<:DiagGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::DiagGaussian) = length(p.mu)
LogDensityProblems.logdensity(p::DiagGaussian, x) =
    -sum(abs2, (x .- p.mu) ./ p.sigma) / 2 - sum(log, p.sigma) - length(x) / 2 * log(2π)
function LogDensityProblems.logdensity_and_gradient(p::DiagGaussian, x)
    z = (x .- p.mu) ./ p.sigma
    (-sum(abs2, z) / 2 - sum(log, p.sigma) - length(x) / 2 * log(2π), -z ./ p.sigma)
end

"Central finite-difference gradient of scalar `f` at `x`."
function fd_gradient(f, x; h=1e-6)
    g = similar(x, float(eltype(x)))
    for i in eachindex(x)
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    g
end
