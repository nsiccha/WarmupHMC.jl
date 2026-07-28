# Where does the wrapper's per-gradient time actually go?
#
# `_logdensity_and_gradient_reparam` (src/Reparametrizations.jl:148) calls
#
#     value_and_gradient(reparam_objective, p.ad_backend, x)
#
# with NO prep object, so DifferentiationInterface re-runs `prepare_gradient`
# on every single gradient evaluation. If preparation is a large share of the
# cost and costs more under Enzyme than ForwardDiff, then "Enzyme is slower at
# d~88" is a statement about how the package CALLS DI, not about reverse mode.
# That distinction decides whether the measurement is a property or an artifact,
# so it gets tested rather than assumed.
#
# Decomposes the per-call cost three ways, on the objective the package actually
# differentiates:
#   prep_ns      - prepare_gradient alone
#   prepped_ns   - value_and_gradient reusing a prep object
#   unprepped_ns - value_and_gradient with no prep (what the package does today)
#
# CORRECTNESS NOTE: the prepped timing is a COST PROBE, not a proposed fix. The
# objective closes over `g_y`, which changes every call, and a prep object built
# for one closure instance may hold a reference to it. So the prepped gradient
# is verified against the unprepped one here, and where it DISAGREES that is
# reported rather than hidden -- a cheaper number computed from a stale closure
# is not a speedup.
include(joinpath(@__DIR__, "common.jl"))

# Same knob as the driver, for the same reason: these scripts overwrite their
# JSON unconditionally, so a quick low-round smoke run silently replaces the
# checked-in record with numbers too thin to mean anything.
const OUT_DIR = env_dir("WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT_DIR)
using Printf, Enzyme, DifferentiationInterface
import JSON

const BE = ["fd" => AutoForwardDiff(),
            "const" => AutoEnzyme(; function_annotation = Enzyme.Const)]
const N = parse(Int, get(ENV, "NCALLS", "400"))

# WHY THIS IS REPLICATED, having previously been a single loop per cell.
#
# Each of the three costs below used to be ONE timed pass of N calls, and
# `prep_share` was the ratio of two such passes. Two independent runs of the
# identical script disagreed by more than a factor of THIRTY on one cell
# (`eight_schools`/`fd` read 113% of the call, then 3629%), and half the ten
# cells moved by more than 2x. A GC pause or a first-touch compile landing in
# one unreplicated pass is indistinguishable from a cost, and RESULTS.md was
# quoting the output to three significant figures ("26.1 of 28.6 µs").
#
# So: ROUNDS repeats, medians reported, and the RAW rounds persisted beside
# them. The order of the three timed blocks ROTATES per round, because they are
# not symmetric -- whichever runs first in a process pays for anything not yet
# specialised, and a fixed order charges that permanently to `prep`.
#
# A share above 100% is NOT automatically a measurement error, and the raw
# rounds are what let a reader tell the two apart: `value_and_gradient` without
# a prep object is free to take a lighter path than an explicit
# `prepare_gradient` builds, so prep genuinely CAN cost more than the unprepped
# call it is nominally a part of. A stable 101% means that; a 3629% that reads
# 113% on the next run means the timer caught something else.
const ROUNDS = parse(Int, get(ENV, "ROUNDS", "5"))

rows = []

function probe(label, problem, spec, dim, csrc)
    reparam = with_source(spec, csrc)
    xs = [randn(Xoshiro(2000 + i), dim) for i in 1:N]

    # Rebuild exactly what the package differentiates, for one x.
    function objective_at(x)
        ljac, y = reparam(x)
        _, g_y = LogDensityProblems.logdensity_and_gradient(problem, y)
        x_ -> begin
            ljac_, y_ = reparam(x_)
            ljac_ + dot(g_y, y_)
        end
    end

    for (nm, be) in BE
        f0 = objective_at(xs[1])
        prep = prepare_gradient(f0, be, xs[1])
        value_and_gradient(f0, prep, be, xs[1])
        value_and_gradient(f0, be, xs[1])

        timed = Dict("prep" => () -> for x in xs
                         f = objective_at(x)
                         prepare_gradient(f, be, x)
                     end,
                     "prepped" => () -> for x in xs
                         f = objective_at(x)
                         value_and_gradient(f, prep, be, x)
                     end,
                     "unprepped" => () -> for x in xs
                         f = objective_at(x)
                         value_and_gradient(f, be, x)
                     end)
        blocks = ["prep", "prepped", "unprepped"]
        samples = Dict(b => Float64[] for b in blocks)
        for r in 1:ROUNDS, b in circshift(blocks, r)
            t = time_ns()
            timed[b]()
            push!(samples[b], (time_ns() - t) / N)
        end
        prep_ns = median(samples["prep"])
        prepped_ns = median(samples["prepped"])
        unprepped_ns = median(samples["unprepped"])

        # Does prep reuse across a changed closure even give the right answer?
        xt = xs[end]
        ft = objective_at(xt)
        g_ref = value_and_gradient(ft, be, xt)[2]
        g_pre = value_and_gradient(ft, prep, be, xt)[2]
        agree = maximum(abs.(g_pre .- g_ref))

        # `samples` is additive; every key that existed keeps its name and its
        # meaning (a per-call nanosecond cost), so a reader indexing
        # `row["prep_share"]` is unaffected by the replication.
        push!(rows, Dict("target" => label, "dim" => dim, "c_source" => csrc,
                         "backend" => nm, "prep_ns" => prep_ns,
                         "prepped_ns" => prepped_ns, "unprepped_ns" => unprepped_ns,
                         "prep_share" => prep_ns / unprepped_ns,
                         "rounds" => ROUNDS, "samples" => samples,
                         "reuse_grad_diff" => agree))
        @printf("%-14s d=%-3d c=%.1f %-6s | prep %9.0f  prepped %9.0f  unprepped %9.0f | prep is %3.0f%% of the call (rounds %3.0f-%3.0f%%) | reuse |Δg| %.1e %s\n",
                label, dim, csrc, nm, prep_ns, prepped_ns, unprepped_ns,
                100 * prep_ns / unprepped_ns,
                100 * minimum(samples["prep"]) / unprepped_ns,
                100 * maximum(samples["prep"]) / unprepped_ns, agree,
                agree > 1e-8 ? "<-- WRONG, prep reuse is not valid here" : "(ok)")
        flush(stdout)
    end
end

for t in TARGETS
    prob, dim, jdata = stan_problem(t.name)
    spec = native_spec(t.name, dim, jdata)
    isnothing(spec) && continue
    probe(t.name, prob, spec, dim, opposite_c(spec))
end
let f = Funnel(9), dim = LogDensityProblems.dimension(Funnel(9))
    probe("funnel", f, funnel_spec(f, 0.0), dim, 0.0)
end

open(joinpath(OUT_DIR, "prep_cost.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "Share of the wrapper's per-gradient cost spent in DI preparation. " *
                  "The package calls value_and_gradient with no prep object, so it " *
                  "re-prepares on every gradient evaluation.",
        "julia" => string(VERSION), "n_calls" => N, "rounds" => ROUNDS,
        git_provenance()...,
        "rows" => rows), 2)
end
println("\nwrote results/prep_cost.json")
