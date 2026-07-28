# Replication of the per-gradient backend comparison, with the measurement
# design the one-shot sweep lacked.
#
# The one-shot sweep and the benchmark driver disagreed by up to 1.75x on
# radon_vi's ForwardDiff number, which is larger than most of the backend
# effects being claimed. Two candidate causes: (a) each backend was timed once,
# in a fixed order, so any drift in machine state is charged to whichever
# backend happened to run then; (b) the two scripts drew their evaluation
# positions from different seeds, and a Stan log-density's cost can depend on
# where it is evaluated.
#
# This script removes both. Every round times all three backends on the SAME
# positions, and the backend order is ROTATED per round so no backend keeps the
# warm/cold slot. Reporting min-over-rounds as well as median: min is the round
# least disturbed by other activity, so a min-vs-median gap is itself the
# contention signal.
include(joinpath(@__DIR__, "common.jl"))

# Same knob as the driver, for the same reason: these scripts overwrite their
# JSON unconditionally, so a quick low-round smoke run silently replaces the
# checked-in record with numbers too thin to mean anything.
const OUT_DIR = env_dir("WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT_DIR)
using Printf, Enzyme
import JSON

const CONST_BE = AutoEnzyme(; function_annotation = Enzyme.Const)
const DUP_BE   = AutoEnzyme(; function_annotation = Enzyme.Duplicated)
const FD_BE    = AutoForwardDiff()
const BACKENDS = ["fd", "const", "dup"]
const BE_OF    = Dict("fd" => FD_BE, "const" => CONST_BE, "dup" => DUP_BE)

const ROUNDS = parse(Int, get(ENV, "ROUNDS", "7"))
const NCALLS = parse(Int, get(ENV, "NCALLS", "1000"))

function ns_per_grad(p, xs)
    t0 = time_ns()
    for x in xs
        LogDensityProblems.logdensity_and_gradient(p, x)
    end
    (time_ns() - t0) / length(xs)
end

rows = []

function bench(label, problem, spec, dim, csrc)
    sp = with_source(spec, csrc)
    probs = Dict(nm => ReparametrizedProblem(sp, problem, BE_OF[nm]) for nm in BACKENDS)

    # Gradient agreement, checked once, before any timing.
    gref = LogDensityProblems.logdensity_and_gradient(probs["fd"], randn(Xoshiro(7), dim))[2]
    dmax = 0.0
    for nm in BACKENDS
        g = LogDensityProblems.logdensity_and_gradient(probs[nm], randn(Xoshiro(7), dim))[2]
        dmax = max(dmax, maximum(abs.(g .- gref)))
    end

    # Warm every backend AND the bare problem before the first timed round, so
    # round 1 is not a compile measurement charged to whoever went first.
    warm = randn(Xoshiro(1), dim)
    LogDensityProblems.logdensity_and_gradient(problem, warm)
    for nm in BACKENDS
        LogDensityProblems.logdensity_and_gradient(probs[nm], warm)
    end

    samples = Dict(nm => Float64[] for nm in BACKENDS)
    bares = Float64[]
    for r in 1:ROUNDS
        xs = [randn(Xoshiro(1000r + i), dim) for i in 1:NCALLS]
        push!(bares, ns_per_grad(problem, xs))
        order = circshift(BACKENDS, r)          # rotate: no fixed slot per backend
        for nm in order
            push!(samples[nm], ns_per_grad(probs[nm], xs))
        end
    end

    stat(v) = (median(v), minimum(v), maximum(v))
    b = stat(bares)
    s = Dict(nm => stat(samples[nm]) for nm in BACKENDS)
    spread(v) = (maximum(v) - minimum(v)) / median(v)

    push!(rows, Dict("target" => label, "dim" => dim, "c_source" => csrc,
                     "rounds" => ROUNDS, "n_calls" => NCALLS,
                     "max_grad_diff" => dmax,
                     "bare_ns" => Dict("median" => b[1], "min" => b[2], "max" => b[3]),
                     [nm => Dict("median" => s[nm][1], "min" => s[nm][2], "max" => s[nm][3],
                                 "all" => samples[nm]) for nm in BACKENDS]...))

    @printf("%-14s d=%-3d c=%.1f | bare %9.0f | fd %9.0f (±%3.0f%%) const %9.0f (±%3.0f%%) dup %9.0f (±%3.0f%%) | const/fd med %.2fx min %.2fx | max|Δg| %.1e\n",
            label, dim, csrc, b[1],
            s["fd"][1], 100spread(samples["fd"]),
            s["const"][1], 100spread(samples["const"]),
            s["dup"][1], 100spread(samples["dup"]),
            s["const"][1] / s["fd"][1], s["const"][2] / s["fd"][2], dmax)
    flush(stdout)
end

@printf("%d rounds x %d calls, backend order rotated per round.\n\n", ROUNDS, NCALLS)

for t in TARGETS
    prob, dim, jdata = stan_problem(t.name)
    spec = native_spec(t.name, dim, jdata)
    isnothing(spec) && continue
    for c in (opposite_c(spec), 0.5)
        bench(t.name, prob, spec, dim, c)
    end
end
let f = Funnel(9), dim = LogDensityProblems.dimension(Funnel(9))
    spec = funnel_spec(f, 0.0)
    for c in (0.0, 0.5)
        bench("funnel", f, spec, dim, c)
    end
end

open(joinpath(OUT_DIR, "backend_replication.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "ForwardDiff vs Enzyme/Const vs Enzyme/Duplicated per gradient call. " *
                  "Rounds are interleaved and rotated so drift is not charged to one backend.",
        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        "rounds" => ROUNDS, "n_calls" => NCALLS,
        "warmuphmc_sha" => readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`),
        "rows" => rows), 2)
end
println("\nwrote results/backend_replication.json")
