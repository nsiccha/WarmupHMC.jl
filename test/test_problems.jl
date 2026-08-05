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

"""
    NaNProblem(dimension)

A target whose log density AND gradient are `NaN` everywhere — the exact shape a
consumer's error handler produces when it catches the model's own exception and
returns a number instead of rethrowing (Bruno's `BridgeStanProblem` does this
around BridgeStan). Used to pin the initialization error contract.
"""
struct NaNProblem
    dimension::Int
end
LogDensityProblems.capabilities(::Type{NaNProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::NaNProblem) = p.dimension
LogDensityProblems.logdensity(::NaNProblem, x) = NaN
LogDensityProblems.logdensity_and_gradient(p::NaNProblem, x) = (NaN, fill(NaN, p.dimension))

"""
    BadGradientProblem(inner, index)

`inner` with a FINITE log density but component `index` of the gradient poisoned
with `Inf`. The initialization check has to fail on either half, and a test that
only ever supplies a non-finite log density cannot tell whether the gradient half
is wired up at all.
"""
struct BadGradientProblem{P}
    inner::P
    index::Int
end
LogDensityProblems.capabilities(::Type{<:BadGradientProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::BadGradientProblem) = LogDensityProblems.dimension(p.inner)
LogDensityProblems.logdensity(p::BadGradientProblem, x) = LogDensityProblems.logdensity(p.inner, x)
function LogDensityProblems.logdensity_and_gradient(p::BadGradientProblem, x)
    logdensity, gradient = LogDensityProblems.logdensity_and_gradient(p.inner, x)
    poisoned = collect(float(eltype(gradient)), gradient)
    poisoned[p.index] = Inf
    (logdensity, poisoned)
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

# A capturing Treebars backend for the `stream_mcmc` progress test: flattens every
# arg and kwarg VALUE from every progress call into `seen`, so a test can assert on
# what the sampler emitted (Treebars stringifies the display types before they
# reach the backend, so field values arrive as their `string(...)` forms). Reached
# through `WarmupHMC.Treebars` so the test project needs no direct Treebars dep.
const _Treebars = WarmupHMC.Treebars
struct CaptureProgress
    seen::Vector{Any}
end
CaptureProgress() = CaptureProgress(Any[])
_capture!(p::CaptureProgress, args, kwargs) = begin
    for a in args; push!(p.seen, a); end
    for kv in kwargs; push!(p.seen, kv[1] => kv[2]); end
end
_Treebars.initialize_progress!(p::CaptureProgress, args...; kwargs...) = (_capture!(p, args, kwargs); p)
_Treebars.update_progress!(p::CaptureProgress, args...; kwargs...) = (_capture!(p, args, kwargs); nothing)
_Treebars.finalize_progress!(p::CaptureProgress, args...; kwargs...) = (_capture!(p, args, kwargs); nothing)
_Treebars.fail_progress!(p::CaptureProgress, args...; kwargs...) = (_capture!(p, args, kwargs); nothing)
Base.show(io::IO, ::_Treebars.ProgressNode{<:CaptureProgress}) = print(io, "CaptureProgress")
