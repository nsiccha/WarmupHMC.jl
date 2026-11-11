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
# Since `1a395ce` a bare `AutoEnzyme()` works at both AD sites, and the docs tell
# users to drop `function_annotation` / `mode`. These two rows are what back that
# advice with a measurement instead of prose.
const BARE_BE  = AutoEnzyme()
const RTA_BE   = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                             function_annotation = Enzyme.Const)
const BACKENDS = [("fd", FD_BE), ("const", CONST_BE), ("dup", DUP_BE),
                  ("bare", BARE_BE), ("rta", RTA_BE)]

# Same knobs as `capture_boxing.jl`. Every round times every backend on the SAME
# positions, and the order is rotated per round so no backend keeps the warm or
# cold slot; the reported figure is the per-backend median over rounds.
const NCALLS = parse(Int, get(ENV, "NCALLS", "2000"))
const ROUNDS = parse(Int, get(ENV, "ROUNDS", "5"))

function ns_per_grad(p, xs)
    LogDensityProblems.logdensity_and_gradient(p, xs[1])   # warm
    t0 = time_ns()
    for x in xs; LogDensityProblems.logdensity_and_gradient(p, x); end
    (time_ns() - t0) / length(xs)
end

rows = []
function bench_spec(label, problem, spec, dim, csrc)
    xs = [randn(Xoshiro(100 + i), dim) for i in 1:NCALLS]
    sp = with_source(spec, csrc)
    names = first.(BACKENDS)
    probs = Dict(nm => ReparametrizedProblem(sp, problem, be) for (nm, be) in BACKENDS)
    grads = Dict(nm => LogDensityProblems.logdensity_and_gradient(probs[nm], xs[1])[2] for nm in names)
    times = Dict(nm => Float64[] for nm in [names; "inner"])
    for round in 1:ROUNDS
        for nm in circshift(names, round - 1)
            push!(times[nm], ns_per_grad(probs[nm], xs))
        end
        # The wrapped model's own gradient, so the wrapper's cost has a unit.
        push!(times["inner"], ns_per_grad(problem, xs))
    end
    ns = Dict(nm => median(v) for (nm, v) in times)
    # correctness: every Enzyme arm must agree with ForwardDiff
    dmax = maximum(maximum(abs.(grads[nm] .- grads["fd"])) for nm in names if nm != "fd")
    push!(rows, (label, dim, csrc, ns["fd"], ns["const"], ns["dup"], dmax,
                 ns["bare"], ns["rta"], ns["inner"]))
    @printf("%-46s d=%-4d c=%.1f  fd=%8.1f const=%8.1f dup=%9.1f bare=%8.1f rta=%8.1f inner=%8.1f  const/bare=%.2fx  max|Δg|=%.2e\n",
            label, dim, csrc, ns["fd"], ns["const"], ns["dup"], ns["bare"], ns["rta"], ns["inner"],
            ns["const"]/ns["bare"], dmax)
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
        "note" => "Enzyme function_annotation Const vs Duplicated vs bare AutoEnzyme() (and Const + runtime activity), per target and source centering; per-backend median over rounds with rotated order",
        "rounds" => ROUNDS, "ncalls" => NCALLS,
        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        PROV...,
        "rows" => [Dict("target"=>r[1], "dim"=>r[2], "c_source"=>r[3],
                        "ns_forwarddiff"=>r[4], "ns_const"=>r[5], "ns_duplicated"=>r[6],
                        "max_grad_diff"=>r[7], "ns_bare"=>r[8],
                        "ns_runtime_activity"=>r[9], "ns_inner"=>r[10]) for r in rows]), 2)
end
println("\nwrote results/annotation_sweep.json")
