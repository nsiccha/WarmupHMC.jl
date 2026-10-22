# End-to-end comparison of the `:adaptive` metric fit variants inside
# `adaptive_warmup_mcmc`.
#
#   WHMC_TSVD_ARM=current julia --project=docs/benchmark docs/benchmark/two_stage_tsvd_sample.jl
#
# Investigation todo `WarmupHMC:two-stage-tsvd/todos/2026-10-22T10-55-12-401-1ruyk25`.
# The fit variants are defined in `two_stage_tsvd_common.jl`; this driver swaps
# the one both samplers fit as `:adaptive` by redefining the internal
# `WarmupHMC._adaptive_scale` in THIS process, before any sampler call. Run one
# process per arm. The package default is untouched.
#
# Knobs:
#   WHMC_TSVD_ARM      current | scaled | two_stage      (required arm)
#   WHMC_TSVD_SEEDS    pinned seeds 1:N per target       (default 6)
#   WHMC_TSVD_DRAWS    n_draws                           (default 1000)
#   WHMC_TSVD_TARGETS  comma-separated target names      (default: all below)
#   WHMC_TSVD_OUT      output file (Serialization of the row vector); required
#
# Per run, besides the sampler's own numbers (gradient evaluations per minimum
# bulk ESS, over the whole run and over the retained epoch only), every variant
# is also fitted OFFLINE to the run's final halo and judged on the run's
# retained draws (`frame_loss` on positions + gradients the fit never saw).
# Those offline fits are paired: all three variants see the same evidence.
# Their `fit_ms` is one timed call (the first run of a process includes
# compilation; the report takes medians over seeds).
#
# Every run fixes `nonlinear_adapt=false`: this measures the linear metric only.

using LinearAlgebra
BLAS.set_num_threads(1)
include("common.jl")
include("two_stage_tsvd_common.jl")
using Serialization

const ARM = get(ENV, "WHMC_TSVD_ARM", "")
const ARM_MAKE = only(filter(v -> v.name == ARM, VARIANTS)).make
const N_SEEDS = parse(Int, get(ENV, "WHMC_TSVD_SEEDS", "6"))
const N_DRAWS = parse(Int, get(ENV, "WHMC_TSVD_DRAWS", "1000"))
const OUT = ENV["WHMC_TSVD_OUT"]

# Select the arm. Both samplers build `:adaptive` through `_adaptive_scale`.
@eval WarmupHMC _adaptive_scale(dimension) = $ARM_MAKE(dimension)

struct GaussianTarget{M}
    precision::M
end
LogDensityProblems.dimension(g::GaussianTarget) = size(g.precision, 1)
LogDensityProblems.capabilities(::Type{<:GaussianTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(g::GaussianTarget, x) = -dot(x, g.precision, x) / 2
LogDensityProblems.logdensity_and_gradient(g::GaussianTarget, x) =
    (LogDensityProblems.logdensity(g, x), -(g.precision * x))

# name => (problem, precision or nothing)
gaussian(structure, scaling, d) = begin
    P = Matrix(inv(gaussian_covariance(structure, scaling, d, 1)))
    P = (P + P') / 2
    (GaussianTarget(P), P)
end
const ALL_TARGETS = [
    "gauss:pairs:spread:20", "gauss:dense:spread:20", "gauss:dense:equal:20",
    "gauss:ar1:spread:50", "gauss:hub:spread:50", "gauss:factor:spread:50",
    "gauss:pairs:spread:200", "gauss:ar1:spread:200",
    "funnel",
    "kilpisjarvi_mod-kilpisjarvi", "diamonds-diamonds",
    "eight_schools-eight_schools_centered", "seeds_data-seeds_centered_model",
    "radon_mn-radon_partially_pooled_centered",
]
const SELECTED = let raw = get(ENV, "WHMC_TSVD_TARGETS", "")
    isempty(raw) ? ALL_TARGETS : String.(strip.(split(raw, ",")))
end

make_target(name) =
    if startswith(name, "gauss:")
        _, st, sc, d = split(name, ":")
        gaussian(st, sc, parse(Int, d))
    elseif name == "funnel"
        (Funnel(10), nothing)
    else
        (first(stan_problem(name)), nothing)
    end

function run_one(name, problem, P, seed)
    windows = NamedTuple[]
    callback = (state, stage) -> begin
        stage === :window && push!(windows, (;
            active=state.active_transformation,
            refl=W.n_reflections(state.scale_options.adaptive),
            restart=state.restart,
        ))
        false
    end
    local r
    wall = @elapsed r = adaptive_warmup_mcmc(
        Xoshiro(seed), problem;
        n_draws=N_DRAWS, nonlinear_adapt=false, monitor_ess=false,
        progress=nothing, callback, UNBOUNDED...,
    )
    draws = Matrix{Float64}(r.posterior_position)
    ess = ess_per_coordinate(draws)
    min_ess = minimum(ess)
    hx, hg = Matrix{Float64}(r.halo_position), Matrix{Float64}(r.halo_gradient)
    pg = Matrix{Float64}(r.posterior_gradient)
    offline = map(VARIANTS) do v
        t = @elapsed ((L, loss) = fit(v.make, hx, hg; UNBOUNDED...))
        (; variant=v.name, loss, loss_out=frame_loss(L, draws, pg),
           refl=W.n_reflections(L), bad=bad_left(L, hx, hg), fit_ms=1e3t,
           kappa=isnothing(P) ? NaN : kappa(L, P))
    end
    (; target=name, arm=ARM, seed, ok=true, wall,
       grad_evals=Int(r.total_evaluation_counter),
       sampling_grad_evals=Int(r.sampling_evaluation_counter),
       n_draws=size(draws, 2), min_ess, median_ess=median(ess),
       evals_per_min_ess=r.total_evaluation_counter / min_ess,
       sampling_evals_per_min_ess=r.sampling_evaluation_counter / min_ess,
       divergences=Int(r.n_divergent_samples),
       active=r.active_transformation,
       adaptive_refl=W.n_reflections(r.scale_options.adaptive),
       n_windows=length(windows), windows,
       kappa_final=isnothing(P) ? NaN : kappa(r.scale_options[r.active_transformation], P),
       halo_n=size(hx, 2), offline)
end

rows = NamedTuple[]
for name in SELECTED
    problem, P = make_target(name)
    for seed in 1:N_SEEDS
        row = try
            run_one(name, problem, P, seed)
        catch err
            (; target=name, arm=ARM, seed, ok=false, error=sprint(showerror, err))
        end
        push!(rows, row)
        serialize(OUT, rows)
        if row.ok
            @printf("%-45s %-9s seed=%d  evals/minESS=%8.1f  sampling=%7.1f  minESS=%7.1f  active=%-10s refl=%3d  div=%d  wall=%.1fs\n",
                    name, ARM, seed, row.evals_per_min_ess, row.sampling_evals_per_min_ess,
                    row.min_ess, row.active, row.adaptive_refl, row.divergences, row.wall)
        else
            println("$name $ARM seed=$seed FAILED: ", row.error)
        end
        flush(stdout)
    end
end
