# End-to-end comparison of the `:adaptive` reflection-fit variants
# (`reflection_search_common.jl`) inside `adaptive_warmup_mcmc`. One process per
# arm:
#
#   WHMC_RS_ARM=topk WHMC_BENCH_OUT=<dir> julia --project=bench/research bench/research/reflection_search_sample.jl
#
# Knobs: WHMC_RS_ARM (a variant name, required), WHMC_RS_SEEDS (default 6),
# WHMC_RS_SEED0 (first seed, default 1), WHMC_RS_DRAWS (n_draws, default 1000),
# WHMC_RS_TARGETS (comma-separated), WHMC_RS_TAG (output name suffix).
#
# The arm is installed in THIS process only: `WarmupHMC._adaptive_scale` is
# redefined to build the arm's candidate structure, and a narrower method of
# `update_loss!(::SuccessiveReflections, ...)` adds the arm's keywords to every
# reflection fit (the window refits and the restart test's refits alike). The
# package default is untouched; the `shipped` arm installs nothing. Every run
# fixes `nonlinear_adapt=false`: this measures the linear metric only.
#
# Writes `<out>/sample_<arm>.jls`; `reflection_search_report.jl` summarizes.

using LinearAlgebra
BLAS.set_num_threads(1)
include("common.jl")
include("reflection_search_common.jl")
using Serialization, Random

const ARM = variant(ENV["WHMC_RS_ARM"])
const N_SEEDS = parse(Int, get(ENV, "WHMC_RS_SEEDS", "6"))
const SEED0 = parse(Int, get(ENV, "WHMC_RS_SEED0", "1"))
const N_DRAWS = parse(Int, get(ENV, "WHMC_RS_DRAWS", "1000"))
const OUT = joinpath(ENV["WHMC_BENCH_OUT"], "sample_$(ARM.name)$(get(ENV, "WHMC_RS_TAG", "")).jls")
const ARM_KW = ARM.kw

if ARM.name != "shipped"
    @eval WarmupHMC _adaptive_scale(dimension) = $(ARM.make)(dimension)
    function WarmupHMC.update_loss!(t::WarmupHMC.SuccessiveReflections, p::AbstractMatrix, g::AbstractMatrix; kwargs...)
        invoke(WarmupHMC.update_loss!, Tuple{WarmupHMC.SuccessiveReflections,Any,Any}, t, p, g; kwargs..., ARM_KW...)
    end
end

function run_one(name, problem, seed)
    windows = NamedTuple[]
    callback = (state, stage) -> begin
        stage === :window && push!(windows, (; active=state.active_transformation,
            refl=W.n_reflections(state.scale_options.adaptive), restart=state.restart))
        false
    end
    local r
    wall = @elapsed r = adaptive_warmup_mcmc(Xoshiro(seed), problem; n_draws=N_DRAWS,
        nonlinear_adapt=false, monitor_ess=false, progress=nothing, callback)
    draws = Matrix{Float64}(r.posterior_position)
    ess = ess_per_coordinate(draws)
    (; target=name, arm=ARM.name, seed, ok=true, wall,
       grad_evals=Int(r.total_evaluation_counter),
       sampling_grad_evals=Int(r.sampling_evaluation_counter),
       n_draws=size(draws, 2), min_ess=minimum(ess), median_ess=median(ess),
       evals_per_min_ess=r.total_evaluation_counter / minimum(ess),
       sampling_evals_per_min_ess=r.sampling_evaluation_counter / minimum(ess),
       steps_per_draw=r.sampling_evaluation_counter / size(draws, 2),
       divergences=Int(r.n_divergent_samples), active=r.active_transformation,
       adaptive_refl=W.n_reflections(r.scale_options.adaptive),
       n_windows=length(windows), windows)
end

rows = NamedTuple[]
for name in selected_targets()
    problem = make_problem(name)
    for seed in SEED0:SEED0+N_SEEDS-1
        row = try
            run_one(name, problem, seed)
        catch err
            (; target=name, arm=ARM.name, seed, ok=false, error=sprint(showerror, err))
        end
        push!(rows, row)
        serialize(OUT, rows)
        if row.ok
            @printf("%-45s %-15s seed=%d evals/minESS=%8.1f steps/draw=%6.1f minESS=%7.1f active=%-10s refl=%3d div=%d wall=%.1fs\n",
                    name, ARM.name, seed, row.evals_per_min_ess, row.steps_per_draw, row.min_ess,
                    row.active, row.adaptive_refl, row.divergences, row.wall)
        else
            println(name, " ", ARM.name, " seed=", seed, " FAILED: ", row.error)
        end
        flush(stdout)
    end
end
