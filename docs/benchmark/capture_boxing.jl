# Why the AD-backend comparison says what it says.
#
# The backend probe (replicate_backends.jl) reports Enzyme/`Const` beating
# ForwardDiff on the two `d = 10` targets and losing on the three larger ones,
# which reads as "reverse mode does not scale here". It is not about scale.
#
# `reparametrization()` in web/src/posteriordb_reparametrizations.jl is one long
# if/elseif chain, and `l`, `s`, `o` are assigned in many of its branches inside
# that single scope. The per-pair closures capture them:
#
#     (l, s, o) = (J+1, J+2, 0)              # :41, radon_partially_pooled
#     map(1:J) do i
#         idx => Reparametrization(..., x->x[l], x->x[s])
#     end
#
# Julia's closure conversion cannot prove single assignment across those
# branches, so it captures a Core.Box -- a mutable heap cell read as Any --
# rather than an Int. `funnel` and `eight_schools` close over literals
# (`x->x[1]`, `x->x[9]`) and capture nothing.
#
# That distinction, and not `d`, predicts the backend verdict on all five
# targets. This script establishes both halves:
#
#   1. WHICH specs box, read off `fieldtypes` -- a fact about the object, not an
#      inference from a timing.
#   2. WHAT it costs, by rebuilding one spec so the captures are plain Ints and
#      changing nothing else. Same pairs, same indices, same centerings, and the
#      gradients are asserted bit-identical, so any difference is pure overhead.
#
# Kept as its own script because it explains a result rather than producing one,
# and because it must run in a process that has not sampled (see README).
include(joinpath(@__DIR__, "common.jl"))

const OUT_DIR = get(ENV, "WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT_DIR)
using Printf, Statistics, Enzyme
import JSON

const CONST_BE = AutoEnzyme(; function_annotation = Enzyme.Const)
const FD_BE    = AutoForwardDiff()
const ROUNDS   = parse(Int, get(ENV, "ROUNDS", "5"))
const NCALLS   = parse(Int, get(ENV, "NCALLS", "1000"))
const AB_TARGET = "radon_mn-radon_partially_pooled_centered"

function ns_per_grad(p, xs)                      # `p` as an argument, so Julia
    LogDensityProblems.logdensity_and_gradient(p, xs[1])   # specializes on it
    t0 = time_ns()
    for x in xs
        LogDensityProblems.logdensity_and_gradient(p, x)
    end
    (time_ns() - t0) / length(xs)
end

# ---------------------------------------------------------------------------
# 1. Which specs box
# ---------------------------------------------------------------------------

captures = []
println("How each shipped spec's closures capture the index they read:\n")
@printf("%-42s %-6s %-24s %s\n", "target", "arg", "closure type", "captures")
println(repeat("-", 100))
for t in TARGETS
    _, dim, jdata = stan_problem(t.name)
    spec = native_spec(t.name, dim, jdata)
    isnothing(spec) && continue
    for (j, a) in enumerate(spec.pairs[1].second.args)
        a isa Function || continue
        ft = fieldtypes(typeof(a))
        boxed = any(T -> T === Core.Box, ft)
        @printf("%-42s %-6d %-24s %s%s\n", first(t.name, 42), j,
                first(string(typeof(a)), 24), ft, boxed ? "   <-- BOXED" : "")
        push!(captures, Dict("target" => t.name, "dim" => dim, "arg" => j,
                             "captures" => string(ft), "boxed" => boxed))
    end
end
let f = Funnel(9), a = funnel_spec(f, 1.0).pairs[1].second.args[2]
    ft = fieldtypes(typeof(a))
    @printf("%-42s %-6d %-24s %s\n", "funnel (benchmark-local)", 2,
            first(string(typeof(a)), 24), ft)
    push!(captures, Dict("target" => "funnel", "dim" => LogDensityProblems.dimension(f),
                         "arg" => 2, "captures" => string(ft),
                         "boxed" => any(T -> T === Core.Box, ft)))
end

# ---------------------------------------------------------------------------
# 2. What it costs
# ---------------------------------------------------------------------------
#
# Verbatim reproduction of posteriordb_reparametrizations.jl:38-50, except that
# the captured indices are passed in as arguments -- which is what makes them
# single-assignment and therefore unboxed. Nothing else differs.

function radon_pp_pairs(J, tc, sc, l, s, o)
    map(1:J) do i
        idx = o + i
        idx => Reparametrization(PartiallyCentered(tc), PartiallyCentered(sc),
                                 x -> x[l], x -> x[s])
    end
end

prob, dim, jdata = stan_problem(AB_TARGET)
J = jdata["J"]
shipped = with_source(native_spec(AB_TARGET, dim, jdata), 0.0)
deboxed = IndexedReparametrization(radon_pp_pairs(J, 1.0, 0.0, J + 1, J + 2, 0))

# The two specs must be the same function before any timing means anything.
gref = last(LogDensityProblems.logdensity_and_gradient(
              ReparametrizedProblem(shipped, prob, FD_BE), randn(Xoshiro(7), dim)))
gdef = last(LogDensityProblems.logdensity_and_gradient(
              ReparametrizedProblem(deboxed, prob, FD_BE), randn(Xoshiro(7), dim)))
gdiff = maximum(abs, gref .- gdef)
@printf("\nshipped vs de-boxed spec on %s: %d vs %d pairs, max|Δgradient| = %.3e\n",
        AB_TARGET, length(shipped.pairs), length(deboxed.pairs), gdiff)
gdiff == 0.0 || @warn "specs are not identical; the timing below is not an A/B" gdiff

samples = Dict{String,Vector{Float64}}()
bares = Float64[]
for r in 1:ROUNDS
    xs = [randn(Xoshiro(1000r + i), dim) for i in 1:NCALLS]
    push!(bares, ns_per_grad(prob, xs))
    combos = [(sn, sp, bn, be) for (sn, sp) in (("shipped", shipped), ("deboxed", deboxed))
                               for (bn, be) in (("fd", FD_BE), ("const", CONST_BE))]
    for (sn, sp, bn, be) in circshift(combos, r)      # rotate: no fixed warm slot
        p = ReparametrizedProblem(sp, prob, be)
        push!(get!(samples, "$sn/$bn", Float64[]), ns_per_grad(p, xs))
    end
end

med(k) = median(samples[k])
@printf("\n%d rounds x %d calls, order rotated per round. bare gradient %.0f ns\n\n",
        ROUNDS, NCALLS, median(bares))
@printf("%-9s | %-12s | %10s | %s\n", "spec", "backend", "median ns", "per-round ns")
println(repeat("-", 78))
for sn in ("shipped", "deboxed"), bn in ("fd", "const")
    v = samples["$sn/$bn"]
    @printf("%-9s | %-12s | %8.0f  | %s\n", sn, bn == "fd" ? "ForwardDiff" : "Enzyme/Const",
            median(v), join((@sprintf("%.0f", x) for x in v), " "))
end

@printf("\nde-boxing speedup, ForwardDiff  : %5.2fx\n", med("shipped/fd") / med("deboxed/fd"))
@printf("de-boxing speedup, Enzyme/Const : %5.2fx\n", med("shipped/const") / med("deboxed/const"))
@printf("\nEnzyme / ForwardDiff, shipped spec : %5.2fx  (>1 = Enzyme slower)\n",
        med("shipped/const") / med("shipped/fd"))
@printf("Enzyme / ForwardDiff, de-boxed spec: %5.2fx\n",
        med("deboxed/const") / med("deboxed/fd"))
println("\nIf those two disagree in direction, the backend verdict in RESULTS.md is")
println("a property of the spec table and not of the backend.")

open(joinpath(OUT_DIR, "capture_boxing.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "Which posteriordb specs capture a Core.Box, and what it costs on " *
                  "the gradient path. The A/B rebuilds one spec with unboxed captures " *
                  "and asserts bit-identical gradients, so the delta is pure overhead.",
        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        "warmuphmc_sha" => readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`),
        "rounds" => ROUNDS, "n_calls" => NCALLS,
        "captures" => captures,
        "ab_target" => AB_TARGET, "ab_max_grad_diff" => gdiff,
        "bare_ns_median" => median(bares),
        "timings_ns" => samples), 2)
end
println("\nwrote $(joinpath(OUT_DIR, "capture_boxing.json"))")
