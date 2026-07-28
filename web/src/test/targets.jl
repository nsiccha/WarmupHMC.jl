# Shared, self-contained test targets (no BridgeStan, no PosteriorDB).
#
# Guarded so several test files can include it without redefining the structs.
if !@isdefined(TEST_TARGETS_LOADED)

using LogDensityProblems

# --- Diagonal Gaussian ------------------------------------------------------
# Wide scale spread so the marginal-scale condition number exceeds
# `variance_cond_target` and the outer loop restarts at least once.
struct DiagGaussian{V}
    sigma::V
end
LogDensityProblems.dimension(g::DiagGaussian) = length(g.sigma)
LogDensityProblems.capabilities(::Type{<:DiagGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(g::DiagGaussian, x) = -sum(abs2, x ./ g.sigma) / 2
LogDensityProblems.logdensity_and_gradient(g::DiagGaussian, x) =
    (LogDensityProblems.logdensity(g, x), -x ./ g.sigma .^ 2)

# --- Neal's funnel ----------------------------------------------------------
# v ~ N(0, 3), xᵢ | v ~ N(0, exp(v/2)).  Coordinate 1 is `v`, coordinates
# 2:k+1 are the `xᵢ` — the layout the `funnel` branch of
# `posteriordb_reparametrizations.jl` assumes.
#
# The marginal of `v` is EXACTLY N(0, 3) whatever `k` is, and it is the
# coordinate a centered funnel fails on: a sampler that cannot descend into the
# neck under-covers the low-`v` region and reports too small a standard
# deviation. That makes `std(v)` a known-truth check rather than a self-comparison.
struct Funnel
    k::Int
end
LogDensityProblems.dimension(f::Funnel) = f.k + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(f::Funnel, x)
    v = x[1]; xs = @view x[2:end]
    -0.5 * v^2 / 9 - sum(xi -> 0.5 * xi^2 * exp(-v) + v / 2, xs)
end
function LogDensityProblems.logdensity_and_gradient(f::Funnel, x)
    v = x[1]; xs = @view x[2:end]
    lp = LogDensityProblems.logdensity(f, x)
    g = similar(x)
    g[1] = -v / 9 + 0.5 * exp(-v) * sum(abs2, xs) - f.k / 2
    g[2:end] .= .-xs .* exp(-v)
    (lp, g)
end

# --- Eight schools ----------------------------------------------------------
# The canonical hierarchical test case, in the exact unconstrained layout the
# `eight_schools` branch of `posteriordb_reparametrizations.jl` assumes:
#
#     x[1:8] = θ (or its non-centered counterpart)   x[9] = μ   x[10] = log τ
#
# Priors follow the PosteriorDB `eight_schools_centered` / `_noncentered` pair:
# μ ~ N(0, 5), τ ~ HalfCauchy(0, 5) (sampled as log τ, with the log-Jacobian).
const EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

# `centered=true`  → x[1:8] are θ directly              (the hard geometry)
# `centered=false` → x[1:8] are the standardized θ̃      (the easy geometry)
struct EightSchools
    centered::Bool
end
LogDensityProblems.dimension(::EightSchools) = 10
LogDensityProblems.capabilities(::Type{EightSchools}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(m::EightSchools, x)
    t = @view x[1:8]; mu = x[9]; log_tau = x[10]
    tau = exp(log_tau)
    theta = m.centered ? t : mu .+ tau .* t
    lp = -0.5 * (mu / 5)^2                              # μ ~ N(0, 5)
    lp += -log1p((tau / 5)^2) + log_tau                 # τ ~ HalfCauchy(0,5), + Jacobian
    lp += m.centered ? -sum(abs2, (theta .- mu) ./ tau) / 2 - 8 * log_tau :
                       -sum(abs2, t) / 2
    lp += -sum(abs2, (EIGHT_SCHOOLS_Y .- theta) ./ EIGHT_SCHOOLS_SIGMA) / 2
    lp
end

# Analytic gradient, so the target itself never needs an AD backend (the
# reparametrization wrapper supplies its own).
function LogDensityProblems.logdensity_and_gradient(m::EightSchools, x)
    t = @view x[1:8]; mu = x[9]; log_tau = x[10]
    tau = exp(log_tau)
    theta = m.centered ? collect(t) : mu .+ tau .* t
    g = zeros(eltype(x), 10)
    # likelihood: d/dθ
    dtheta = (EIGHT_SCHOOLS_Y .- theta) ./ EIGHT_SCHOOLS_SIGMA .^ 2
    if m.centered
        z = (theta .- mu) ./ tau
        dtheta .-= z ./ tau
        g[1:8] .= dtheta
        g[9] = sum(z) / tau
        g[10] = sum(abs2, z) - 8
    else
        g[1:8] .= dtheta .* tau .- t
        g[9] = sum(dtheta)
        g[10] = sum(dtheta .* tau .* t)
    end
    g[9] += -mu / 25
    g[10] += 1 - 2 * (tau / 5)^2 / (1 + (tau / 5)^2)
    (LogDensityProblems.logdensity(m, x), g)
end

const TEST_TARGETS_LOADED = true
end
