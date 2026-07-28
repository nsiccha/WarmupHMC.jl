# Why the AD-backend comparison said what it said, and a guard so it cannot say it
# again.
#
# The backend probe (replicate_backends.jl) reported Enzyme/`Const` beating
# ForwardDiff on the two `d = 10` targets and losing on the three larger ones,
# which reads as "reverse mode does not scale here". It was not about scale.
#
# The per-pair index closures in a reparametrization spec must capture their index
# as an `Int`. They capture a `Core.Box` -- a mutable heap cell read as `Any` --
# whenever the index is a variable assigned in MORE THAN ONE PLACE within a single
# scope, because Julia's closure conversion cannot then prove single assignment.
# `reparametrization()` in web/src/posteriordb_reparametrizations.jl used to be one
# long if/elseif chain assigning `l`, `s`, `o` in five of its branches, so five of
# the nine specs boxed:
#
#     (l, s, o) = (J+1, J+2, 0)              # radon_partially_pooled
#     map(1:J) do i
#         idx => Reparametrization(..., x->x[l], x->x[s])
#     end
#
# The closure still returns the right index. Only the AD path pays -- and it pays
# ForwardDiff ~4.8x and Enzyme ~15x, which is enough to invert which backend looks
# faster. `funnel` and `eight_schools` close over literals (`x->x[1]`, `x->x[9]`)
# and were never affected; `radon_variable_intercept_slope` and `accel_gp` bind
# their indices as `do`-block parameters, which is a function scope, and already
# had the property the others lacked.
#
# Two things are worth being precise about, because both were got wrong on the way
# here:
#
#   * It is NOT tuple destructuring. `(l, s, o) = (J+1, J+2, 0)` in a scope that
#     assigns them once is clean -- verified directly, and that clean reading is
#     what briefly killed the hypothesis.
#   * It is NOT specific to top-level `if`/`elseif`. Two assigning branches inside
#     an ordinary function box just as readily, which is what `boxed_pairs` below
#     relies on.
#
# This script establishes three things:
#
#   1. GUARD -- which shipped specs box, read off `fieldtypes`. A fact about the
#      object, not an inference from a timing, and a one-line check that outranks
#      any amount of careful measurement. Exits non-zero if any spec boxes, so a
#      branch that reintroduces one is named here rather than re-derived from a
#      backend comparison months later.
#   2. COST -- boxed against unboxed, both built locally so the measurement does
#      not depend on which way the shipped table currently happens to be written.
#      Same pairs, same indices, same centerings; gradients asserted bit-identical,
#      so the delta is pure overhead.
#   3. WHICH SIDE the shipped spec is on, tying 1 and 2 together.
#
# Kept as its own script because it explains a result rather than producing one,
# and because it must run in a process that has not sampled (see README).
include(joinpath(@__DIR__, "common.jl"))

const OUT_DIR = env_dir("WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
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

isboxed(f) = any(T -> T === Core.Box, fieldtypes(typeof(f)))

# ---------------------------------------------------------------------------
# 1. Guard: which shipped specs box
# ---------------------------------------------------------------------------

captures = []
boxed_specs = String[]

# Every closure in the spec, not just the first pair's. Different pairs of one spec
# can in principle capture differently -- `accel_gp` builds a distinct closure per
# index -- and a guard that samples pair 1 would not see it.
function scan!(label, dim, spec, source)
    isnothing(spec) && return
    seen = Set{Tuple{Int,String}}()
    for (_, r) in spec.pairs, (j, a) in enumerate(r.args)
        a isa Function || continue
        ft = fieldtypes(typeof(a))
        key = (j, string(ft))
        key in seen && continue                # one row per distinct capture shape
        push!(seen, key)
        boxed = isboxed(a)
        boxed && push!(boxed_specs, "$label arg $j")
        @printf("%-46s %-4d %-22s %s%s\n", first(label, 46), j,
                first(string(typeof(a)), 22), ft, boxed ? "   <-- BOXED" : "")
        push!(captures, Dict("target" => label, "dim" => dim, "arg" => j,
                             "captures" => string(ft), "boxed" => boxed,
                             "source" => source))
    end
end

# (a) The specs the benchmark actually samples, built from real posteriordb data.
println("How each shipped spec's closures capture the index they read.\n")
println("(a) specs the benchmark samples, real posteriordb data:\n")
@printf("%-46s %-4s %-22s %s\n", "target", "arg", "closure type", "captures")
println(repeat("-", 100))
for t in TARGETS
    _, dim, jdata = stan_problem(t.name)
    scan!(t.name, dim, native_spec(t.name, dim, jdata), "posteriordb")
end
let f = Funnel(9)
    scan!("funnel (benchmark-local)", LogDensityProblems.dimension(f),
          funnel_spec(f, 1.0), "benchmark")
end

# (b) Every remaining branch of the spec table. Whether a closure captures a
# `Core.Box` is decided by which branch of `reparametrization()` is taken and by
# nothing else -- not by the values in the data -- so synthetic `stan_jdata` reaches
# the four specs the benchmark never samples without compiling four Stan models for
# them. Those four are exactly the ones no timing here could ever have caught:
# `radon_variable_slope` and `radon_hierarchical_intercept` were boxed and nothing
# in this benchmark would have said so.
const SYNTH_JDATA = Dict{String,Any}(
    "I" => 21, "J" => 85,
    "slambda_1" => collect(1.0:64.0), "slambda_sigma_1" => collect(1.0:64.0))

const SPEC_SHAPES = [
    ("funnel", 10),
    ("eight_schools-eight_schools_centered", 10),
    ("seeds_data-seeds_centered_model", 26),
    ("radon_mn-radon_partially_pooled_centered", 88),
    ("radon_mn-radon_variable_intercept_centered", 89),
    ("radon_mn-radon_variable_slope_centered", 89),
    ("radon_mn-radon_hierarchical_intercept_centered", 90),
    ("radon_mn-radon_variable_intercept_slope_centered", 175),
    ("accel_gp-accel_gp", 100),
]

println("\n(b) every branch of the spec table, synthetic data:\n")
@printf("%-46s %-4s %-22s %s\n", "branch", "arg", "closure type", "captures")
println(repeat("-", 100))
for (name, dim) in SPEC_SHAPES
    spec = reparametrization(name, dim, SYNTH_JDATA)
    isempty(spec.pairs) && error("`$name` matched no branch of reparametrization() " *
                                 "-- the guard is not covering what it claims to.")
    scan!(name, dim, spec, "synthetic")
end

# ---------------------------------------------------------------------------
# 2. Cost: boxed against unboxed, both built here
# ---------------------------------------------------------------------------
#
# Both reproduce posteriordb_reparametrizations.jl's radon_partially_pooled pairs
# exactly. They differ in one thing: where the captured index comes from.

at_index(i) = x -> x[i]

# The index arrives as a function argument, so it is single-assignment and the
# closure captures an `Int`. This is the shape the shipped table uses.
unboxed_pairs(J, tc, sc, l, s, o) = map(1:J) do i
    (o + i) => Reparametrization(PartiallyCentered(tc), PartiallyCentered(sc),
                                 at_index(l), at_index(s))
end

# The index is assigned in two branches of one scope, so closure conversion cannot
# prove single assignment and captures a `Core.Box`. The second branch is
# unreachable -- reachability is not what the analysis is about, only the number of
# assignments in the scope. This is the shape the shipped table used to have; it is
# kept so the cost stays measurable after the shipped table stopped having it.
function boxed_pairs(J, tc, sc, l0, s0, o0, which = 1)
    if which == 1
        (l, s, o) = (l0, s0, o0)
        return map(1:J) do i
            (o + i) => Reparametrization(PartiallyCentered(tc), PartiallyCentered(sc),
                                         x -> x[l], x -> x[s])
        end
    else
        (l, s, o) = (1, 2, 0)
        return map(1:J) do i
            (o + i) => Reparametrization(PartiallyCentered(tc), PartiallyCentered(sc),
                                         x -> x[l], x -> x[s])
        end
    end
end

prob, dim, jdata = stan_problem(AB_TARGET)
J = jdata["J"]
shipped = with_source(native_spec(AB_TARGET, dim, jdata), 0.0)
unboxed = IndexedReparametrization(unboxed_pairs(J, 1.0, 0.0, J + 1, J + 2, 0))
boxed   = IndexedReparametrization(boxed_pairs(  J, 1.0, 0.0, J + 1, J + 2, 0))

# The controls are only a control if they are the same function -- as each other,
# and as the spec actually shipped. Anything less and the timing below is not an A/B.
grad_of(sp) = last(LogDensityProblems.logdensity_and_gradient(
                       ReparametrizedProblem(sp, prob, FD_BE), randn(Xoshiro(7), dim)))
g_shipped, g_unboxed, g_boxed = grad_of(shipped), grad_of(unboxed), grad_of(boxed)
d_unboxed = maximum(abs, g_shipped .- g_unboxed)
d_boxed   = maximum(abs, g_shipped .- g_boxed)
@printf("\n%s: shipped %d pairs, unboxed %d, boxed %d\n",
        AB_TARGET, length(shipped.pairs), length(unboxed.pairs), length(boxed.pairs))
@printf("max|Δgradient| vs shipped: unboxed %.3e, boxed %.3e\n", d_unboxed, d_boxed)
(d_unboxed == 0.0 && d_boxed == 0.0) ||
    @warn "controls are not identical to the shipped spec; the timing below is not an A/B" d_unboxed d_boxed

shipped_boxed = isboxed(shipped.pairs[1].second.args[1])
@printf("controls box: unboxed %s, boxed %s. The SHIPPED spec boxes: %s\n",
        isboxed(unboxed.pairs[1].second.args[1]), isboxed(boxed.pairs[1].second.args[1]),
        shipped_boxed)

samples = Dict{String,Vector{Float64}}()
bares = Float64[]
for r in 1:ROUNDS
    xs = [randn(Xoshiro(1000r + i), dim) for i in 1:NCALLS]
    push!(bares, ns_per_grad(prob, xs))
    combos = [(sn, sp, bn, be) for (sn, sp) in (("boxed", boxed), ("unboxed", unboxed))
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
for sn in ("boxed", "unboxed"), bn in ("fd", "const")
    v = samples["$sn/$bn"]
    @printf("%-9s | %-12s | %8.0f  | %s\n", sn, bn == "fd" ? "ForwardDiff" : "Enzyme/Const",
            median(v), join((@sprintf("%.0f", x) for x in v), " "))
end

@printf("\nde-boxing speedup, ForwardDiff  : %5.2fx\n", med("boxed/fd") / med("unboxed/fd"))
@printf("de-boxing speedup, Enzyme/Const : %5.2fx\n", med("boxed/const") / med("unboxed/const"))
@printf("\nEnzyme / ForwardDiff, boxed   spec : %5.2fx  (>1 = Enzyme slower)\n",
        med("boxed/const") / med("boxed/fd"))
@printf("Enzyme / ForwardDiff, unboxed spec : %5.2fx\n",
        med("unboxed/const") / med("unboxed/fd"))
println("\nThose two disagree in direction. Any backend verdict measured on a boxed")
println("spec is a property of the spec table, not of the backend.")

open(joinpath(OUT_DIR, "capture_boxing.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "Which posteriordb specs capture a Core.Box, and what it costs on " *
                  "the gradient path. The A/B builds both a boxed and an unboxed " *
                  "control locally and asserts their gradients are bit-identical to " *
                  "the shipped spec's, so the delta is pure overhead and the " *
                  "measurement is independent of how the shipped table is written.",
        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        "warmuphmc_sha" => readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`),
        "rounds" => ROUNDS, "n_calls" => NCALLS,
        "captures" => captures, "boxed_specs" => boxed_specs,
        "ab_target" => AB_TARGET,
        "ab_max_grad_diff_unboxed" => d_unboxed,
        "ab_max_grad_diff_boxed" => d_boxed,
        "shipped_spec_is_boxed" => shipped_boxed,
        "bare_ns_median" => median(bares),
        "timings_ns" => samples), 2)
end
println("\nwrote $(joinpath(OUT_DIR, "capture_boxing.json"))")

# ---------------------------------------------------------------------------
# 3. Verdict
# ---------------------------------------------------------------------------

if isempty(boxed_specs)
    println("\nGUARD PASS: no shipped spec captures a Core.Box.")
else
    println("\nGUARD FAIL: $(length(boxed_specs)) shipped closure argument(s) capture a Core.Box:")
    foreach(s -> println("  ", s), boxed_specs)
    println("""
    Fix the spec so the index reaches the closure as a FUNCTION ARGUMENT rather
    than as a variable assigned in several branches of one scope -- see
    `_index_getter` / `_grouped_pairs` in web/src/posteriordb_reparametrizations.jl.
    Until then, no measurement taken through these specs says anything about an AD
    backend. The cost is above.""")
    exit(1)
end
