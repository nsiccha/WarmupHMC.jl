# Capture the halo of every metric refit of real `adaptive_warmup_mcmc` runs, so
# the reflection-fit variants can be compared offline on exactly the evidence
# the sampler refit on (`reflection_search_refits.jl`).
#
#   WHMC_BENCH_OUT=<dir> julia --project=bench/research bench/research/reflection_search_halos.jl
#
# Knobs: WHMC_RS_SEEDS (default 4), WHMC_RS_DRAWS (n_draws, default 1000),
# WHMC_RS_TARGETS (comma-separated, default all of `ALL_TARGETS`).
#
# The sampler runs unchanged (the shipped metric, defaults apart from
# `nonlinear_adapt=false`, which isolates the linear metric). A narrower method
# of the internal `WarmupHMC._select_transformation!`, defined in THIS process
# only, copies the halo the window refits on and then `invoke`s the shipped
# method, so selection and sampling are exactly the shipped ones. (The final
# window never refits, so its halo is not evidence the sampler fits on.)
#
# Writes `<out>/halos/<target>__<seed>.jls`: the refits in order, each
# `(; window, x, g, active_before, active_after)`, plus the run's summary.

using LinearAlgebra
BLAS.set_num_threads(1)
include("common.jl")
include("reflection_search_common.jl")
using Serialization, Random

const N_SEEDS = parse(Int, get(ENV, "WHMC_RS_SEEDS", "4"))
const N_DRAWS = parse(Int, get(ENV, "WHMC_RS_DRAWS", "1000"))
const OUT = joinpath(ENV["WHMC_BENCH_OUT"], "halos")
mkpath(OUT)

const REFITS = NamedTuple[]
const WINDOW = Ref(0)

function WarmupHMC._select_transformation!(scale_options::NamedTuple, active::Symbol,
                                           positions::AbstractMatrix, gradients::AbstractMatrix; kwargs...)
    x, g = Matrix{Float64}(positions), Matrix{Float64}(gradients)
    next = invoke(WarmupHMC._select_transformation!, Tuple{Any,Any,Any,Any},
                  scale_options, active, positions, gradients; kwargs...)
    size(x, 2) > 1 && push!(REFITS, (; window=WINDOW[], x, g, active_before=active, active_after=next))
    next
end

for name in selected_targets()
    problem = make_problem(name)
    for seed in 1:N_SEEDS
        empty!(REFITS)
        WINDOW[] = 0
        callback = (state, stage) -> (stage === :window && (WINDOW[] += 1); false)
        summary = try
            wall = @elapsed r = adaptive_warmup_mcmc(Xoshiro(seed), problem; n_draws=N_DRAWS,
                nonlinear_adapt=false, monitor_ess=false, progress=nothing, callback)
            (; ok=true, wall, grad_evals=r.total_evaluation_counter,
               active=r.active_transformation, refl=W.n_reflections(r.scale_options.adaptive))
        catch err
            (; ok=false, error=sprint(showerror, err))
        end
        serialize(joinpath(OUT, "$(name)__$(seed).jls"),
                  (; target=name, seed, d=LogDensityProblems.dimension(problem), summary, refits=copy(REFITS)))
        println(rpad(name, 45), " seed=", seed, " refits=", length(REFITS), " ", summary)
        flush(stdout)
    end
end
