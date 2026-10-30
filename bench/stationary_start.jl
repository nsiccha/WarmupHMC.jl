# Pathfinder from a stationary start (snag `pathfinder-start-890fde27`): what the
# recovery in `_pathfinder_from_start` buys. Run after resolving the test
# environment (see .github/workflows/test.yml):
#   julia --project=test bench/stationary_start.jl
#
# Three readings on a 20-dimensional diagonal Gaussian whose scales span e^±1.5,
# and on a product of Student-t marginals with the same scales (not Gaussian,
# heavier tails, the same stationary mode):
#   1. the gradient threshold below which Optim's L-BFGS declares convergence at
#      iteration 0, so that Pathfinder's trajectory has length zero;
#   2. the quality of Pathfinder's metric from the mode plus a Gaussian jitter of
#      several sizes (Pathfinder's own `ntries`/`init_sampler` retry, which is
#      what the recovery reproduces with its own loop), against a random ±2
#      start and against the unit metric of the fallback;
#   3. what the recovery costs in evaluations through `initialize_mcmc` itself.
using WarmupHMC, LogDensityProblems, Random, LinearAlgebra, Statistics, Printf

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
# Independent Student-t(ν) marginals with scale `sigma`: mode 0, sd sigma·√(ν/(ν-2)).
struct DiagStudent{V<:AbstractVector}
    sigma::V
    nu::Float64
end
LogDensityProblems.capabilities(::Type{<:DiagStudent}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::DiagStudent) = length(p.sigma)
LogDensityProblems.logdensity(p::DiagStudent, x) =
    -(p.nu + 1) / 2 * sum(log1p, abs2.(x ./ p.sigma) ./ p.nu) - sum(log, p.sigma)
function LogDensityProblems.logdensity_and_gradient(p::DiagStudent, x)
    z = x ./ p.sigma
    (LogDensityProblems.logdensity(p, x), -(p.nu + 1) .* z ./ (p.nu .+ abs2.(z)) ./ p.sigma)
end
# Evaluation counts, in globals so that a `deepcopy` of the problem (Pathfinder's
# own retry) still counts here.
const N_GRADIENT = Ref(0)
const N_DENSITY = Ref(0)
struct Counting{P}
    inner::P
end
LogDensityProblems.capabilities(::Type{<:Counting}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::Counting) = LogDensityProblems.dimension(p.inner)
LogDensityProblems.logdensity(p::Counting, x) = (N_DENSITY[] += 1; LogDensityProblems.logdensity(p.inner, x))
LogDensityProblems.logdensity_and_gradient(p::Counting, x) =
    (N_GRADIENT[] += 1; LogDensityProblems.logdensity_and_gradient(p.inner, x))
counted(f) = (N_GRADIENT[] = 0; N_DENSITY[] = 0; r = f(); (r, N_GRADIENT[], N_DENSITY[]))

quietly(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

const d = 20
const sigma = exp.(range(-1.5, 1.5, length=d))
const mode = zeros(d)

println("=== 1. the stationary threshold (Gaussian): trajectory length by |g|max at the start ===")
let target = DiagGaussian(zeros(d), sigma)
    for eps in (1e-3, 1e-5, 1e-7, 1e-8, 1e-9)
        x0 = fill(eps, d) .* sigma .^ 2   # gradient = -eps in every coordinate
        r = quietly(() -> WarmupHMC.mypathfinder(target; rng=Xoshiro(1), init=x0, maxiters=100))
        @printf("  |g|max = %.0e  -> L-BFGS iterations %3d  fit: %s\n", eps, length(r.optim_trace) - 1,
            isempty(r.elbo_estimates) ? "none" : "yes")
    end
    println("  _stationary_start threshold: ", WarmupHMC.Pathfinder.Optim.Options().g_abstol)
end

struct Jitter{V}
    center::V
    scale::Float64
end
(s::Jitter)(rng, point) = (point .= s.center .+ s.scale .* randn(rng, length(point)); point)

for (name, target, sd) in (("Gaussian", DiagGaussian(zeros(d), sigma), sigma),
                           ("Student-t(4)", DiagStudent(sigma, 4.0), sigma .* sqrt(4 / 2)))
    println("\n=== 2. $name: Pathfinder's metric from the mode plus a jitter (20 seeds) ===")
    println("  columns: mean and max |log(fitted scale / posterior sd)|, Mahalanobis²/d of the")
    println("  draw, gradient and density evaluations, L-BFGS iterations")
    @printf("  %8s %10s %10s %10s %8s %8s %8s\n", "sd", "logratio", "maxlogr", "mahal/d", "grads", "dens", "iters")
    for scale in (0.01, 0.05, 0.1, 0.3, 1.0, 3.0)
        lr = Float64[]; mlr = Float64[]; mah = Float64[]; grads = Int[]; dens = Int[]; iters = Int[]
        for seed in 1:20
            r, ng, nd = counted(() -> quietly(() -> WarmupHMC.mypathfinder(Counting(target);
                rng=Xoshiro(seed), init=mode, maxiters=100, ntries=2, init_sampler=Jitter(mode, scale))))
            isempty(r.elbo_estimates) && continue
            s = sqrt.(diag(r.fit_distribution.Σ))
            push!(lr, mean(abs, log.(s ./ sd))); push!(mlr, maximum(abs, log.(s ./ sd)))
            push!(mah, sum(abs2, r.draws[:, 1] ./ sd) / d)
            push!(grads, ng); push!(dens, nd); push!(iters, length(r.optim_trace) - 1)
        end
        @printf("  %8.2f %10.3f %10.3f %10.3f %8.1f %8.1f %8.1f   (fits %d/20)\n", scale,
            mean(lr), mean(mlr), mean(mah), mean(grads), mean(dens), mean(iters), length(lr))
    end
    @printf("  unit metric (the fallback): logratio %.3f  maxlogr %.3f\n",
        mean(abs, log.(1 ./ sd)), maximum(abs, log.(1 ./ sd)))
    let lr = Float64[], mlr = Float64[], grads = Int[], dens = Int[]
        for seed in 1:20
            x0 = rand(Xoshiro(100 + seed), d) .* 4 .- 2
            r, ng, nd = counted(() -> quietly(() -> WarmupHMC.mypathfinder(Counting(target);
                rng=Xoshiro(seed), init=x0, maxiters=100)))
            s = sqrt.(diag(r.fit_distribution.Σ))
            push!(lr, mean(abs, log.(s ./ sd))); push!(mlr, maximum(abs, log.(s ./ sd)))
            push!(grads, ng); push!(dens, nd)
        end
        @printf("  random ±2 start:            logratio %.3f  maxlogr %.3f  grads %.1f  dens %.1f\n",
            mean(lr), mean(mlr), mean(grads), mean(dens))
    end

    println("\n=== 3. $name: initialize_mcmc from the mode, through the recovery (20 seeds) ===")
    let lr = Float64[], grads = Int[], dens = Int[]
        for seed in 1:20
            init, ng, nd = counted(() -> quietly(() -> WarmupHMC.initialize_mcmc(Counting(target), mode;
                rng=Xoshiro(seed), progress=nothing)))
            s = sqrt.(diag(init.squared_scale))
            push!(lr, mean(abs, log.(s ./ sd))); push!(grads, ng); push!(dens, nd)
        end
        @printf("  logratio %.3f  grads %.1f  dens %.1f\n", mean(lr), mean(grads), mean(dens))
    end
end
