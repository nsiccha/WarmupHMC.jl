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

        t = time_ns()
        for x in xs
            f = objective_at(x)
            prepare_gradient(f, be, x)
        end
        prep_ns = (time_ns() - t) / N

        t = time_ns()
        for x in xs
            f = objective_at(x)
            value_and_gradient(f, prep, be, x)
        end
        prepped_ns = (time_ns() - t) / N

        t = time_ns()
        for x in xs
            f = objective_at(x)
            value_and_gradient(f, be, x)
        end
        unprepped_ns = (time_ns() - t) / N

        # Does prep reuse across a changed closure even give the right answer?
        xt = xs[end]
        ft = objective_at(xt)
        g_ref = value_and_gradient(ft, be, xt)[2]
        g_pre = value_and_gradient(ft, prep, be, xt)[2]
        agree = maximum(abs.(g_pre .- g_ref))

        push!(rows, Dict("target" => label, "dim" => dim, "c_source" => csrc,
                         "backend" => nm, "prep_ns" => prep_ns,
                         "prepped_ns" => prepped_ns, "unprepped_ns" => unprepped_ns,
                         "prep_share" => prep_ns / unprepped_ns,
                         "reuse_grad_diff" => agree))
        @printf("%-14s d=%-3d c=%.1f %-6s | prep %9.0f  prepped %9.0f  unprepped %9.0f | prep is %3.0f%% of the call | reuse |Δg| %.1e %s\n",
                label, dim, csrc, nm, prep_ns, prepped_ns, unprepped_ns,
                100 * prep_ns / unprepped_ns, agree,
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
        "julia" => string(VERSION), "n_calls" => N,
        git_provenance()...,
        "rows" => rows), 2)
end
println("\nwrote results/prep_cost.json")
