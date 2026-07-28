# Const vs Duplicated across every benchmark target AND both centering endpoints.
#
# The docstring's claim that `Const` is correct-and-fast rests on one 11-d funnel.
# Two things could make that unrepresentative: dimension (10 -> 89 here) and the
# SOURCE centering, since WarmupHMC and I measured different ratios (22x vs 8.8x)
# on the same target with different `c`. Both are varied below.
include(joinpath(@__DIR__, "common.jl"))

# Same knob as the driver, for the same reason: these scripts overwrite their
# JSON unconditionally, so a quick low-round smoke run silently replaces the
# checked-in record with numbers too thin to mean anything.
const OUT_DIR = env_dir("WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT_DIR)
using Printf, Enzyme

const CONST_BE = AutoEnzyme(; function_annotation = Enzyme.Const)
const DUP_BE   = AutoEnzyme(; function_annotation = Enzyme.Duplicated)
const FD_BE    = AutoForwardDiff()

function ns_per_grad(p, xs)
    LogDensityProblems.logdensity_and_gradient(p, xs[1])   # warm
    t0 = time_ns()
    for x in xs; LogDensityProblems.logdensity_and_gradient(p, x); end
    (time_ns() - t0) / length(xs)
end

rows = []
function bench_spec(label, problem, spec, dim, csrc)
    xs = [randn(Xoshiro(100 + i), dim) for i in 1:2000]
    sp = with_source(spec, csrc)
    gs = Dict{String,Any}()
    for (nm, be) in (("fd", FD_BE), ("const", CONST_BE), ("dup", DUP_BE))
        p = ReparametrizedProblem(sp, problem, be)
        _, g = LogDensityProblems.logdensity_and_gradient(p, xs[1])
        gs[nm] = (ns_per_grad(p, xs), g)
    end
    # correctness: all three must agree
    dmax = max(maximum(abs.(gs["const"][2] .- gs["fd"][2])),
               maximum(abs.(gs["dup"][2]   .- gs["fd"][2])))
    push!(rows, (label, dim, csrc, gs["fd"][1], gs["const"][1], gs["dup"][1], dmax))
    @printf("%-46s d=%-4d c=%.1f  fd=%8.1f const=%8.1f dup=%9.1f  dup/const=%5.1fx  const/fd=%.2fx  max|Δg|=%.2e\n",
            label, dim, csrc, gs["fd"][1], gs["const"][1], gs["dup"][1],
            gs["dup"][1]/gs["const"][1], gs["const"][1]/gs["fd"][1], dmax)
end

for t in TARGETS
    prob, dim, jdata = stan_problem(t.name)
    spec = native_spec(t.name, dim, jdata)
    isnothing(spec) && continue
    for c in (opposite_c(spec), 0.5)          # endpoint and interior
        bench_spec(t.name, prob, spec, dim, c)
    end
end
let f = Funnel(9), dim = LogDensityProblems.dimension(Funnel(9))
    spec = funnel_spec(f, 0.0)
    for c in (0.0, 0.5)
        bench_spec("funnel", f, spec, dim, c)
    end
end

import JSON
# `git_provenance()` BEFORE the `open` — see the trap documented on it in
# `common.jl`. `open(path, "w")` truncates immediately, so a provenance call
# inside this block would see this very file as an uncommitted change.
const PROV = git_provenance()
open(joinpath(OUT_DIR, "annotation_sweep.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "Enzyme function_annotation Const vs Duplicated, per target and source centering",
        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        PROV...,
        "rows" => [Dict("target"=>r[1], "dim"=>r[2], "c_source"=>r[3],
                        "ns_forwarddiff"=>r[4], "ns_const"=>r[5], "ns_duplicated"=>r[6],
                        "max_grad_diff"=>r[7]) for r in rows]), 2)
end
println("\nwrote results/annotation_sweep.json")
