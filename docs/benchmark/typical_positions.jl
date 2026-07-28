# Does the backend verdict depend on WHERE the gradient is evaluated?
#
# Two measurements of the same quantity disagree on the radon targets:
#
#   * the microbenchmarks (this repo's `gradient_overhead`, plus two independent
#     multi-round sweeps) evaluate the wrapped gradient at `randn(dim)` and make
#     Enzyme/Const ~1.3x SLOWER than ForwardDiff;
#   * the 184-run benchmark, at identical gradient counts, makes Enzyme 3-8%
#     FASTER per second on the same target and arm.
#
# The obvious suspect is the evaluation position. `randn(dim)` is not where a
# sampler spends its time -- on a hierarchical target the posterior typical set
# sits at a very different log-scale, and both the inner Stan density and the
# transform are evaluated there instead. If that is the cause, then the
# microbenchmark is answering a question nobody asked, and the number that
# belongs in RESULTS.md is the typical-set one.
#
# So: run the sampler, keep the positions it ACTUALLY visited (in the source
# frame -- `nonlinear_adapt=false` returns draws unmapped, so no back-transform
# is applied here), and time all three backends at those positions AND at
# `randn` in the same process, same rotation, same round structure.
include(joinpath(@__DIR__, "common.jl"))

# Same knob as the driver, for the same reason: these scripts overwrite their
# JSON unconditionally, so a quick low-round smoke run silently replaces the
# checked-in record with numbers too thin to mean anything.
const OUT_DIR = get(ENV, "WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT_DIR)
using Printf, Enzyme
import JSON

const CONST_BE = AutoEnzyme(; function_annotation = Enzyme.Const)
const DUP_BE   = AutoEnzyme(; function_annotation = Enzyme.Duplicated)
const FD_BE    = AutoForwardDiff()
const BACKENDS = ["fd", "const", "dup"]
const BE_OF    = Dict("fd" => FD_BE, "const" => CONST_BE, "dup" => DUP_BE)
const ROUNDS   = parse(Int, get(ENV, "ROUNDS", "5"))

function ns_per_grad(p, xs)
    t0 = time_ns()
    for x in xs
        LogDensityProblems.logdensity_and_gradient(p, x)
    end
    (time_ns() - t0) / length(xs)
end

rows = []

function compare_positions(label, problem, spec, dim)
    csrc = opposite_c(spec)
    sp = with_source(spec, csrc)

    # Positions the sampler really visits, in the SOURCE frame. Sampled with the
    # wrapper in place and `nonlinear_adapt=false`, which is exactly the
    # `fixed_noncentered` arm of the benchmark -- so these are that arm's own
    # gradient-evaluation positions, not a proxy for them.
    lpdf = ReparametrizedProblem(sp, problem, FD_BE)
    res = adaptive_warmup_mcmc(Xoshiro(1), lpdf; n_draws = 1000, nonlinear_adapt = false)
    typical = Matrix{Float64}(res.posterior_position)
    ntyp = size(typical, 2)
    ntyp > 100 || error("$label: only $ntyp draws retained, too few to time on")

    probs = Dict(nm => ReparametrizedProblem(sp, problem, BE_OF[nm]) for nm in BACKENDS)

    n = min(1000, ntyp)
    pos = Dict(
        "typical" => [typical[:, i] for i in 1:n],
        "randn"   => [randn(Xoshiro(3000 + i), dim) for i in 1:n],
    )

    warm = typical[:, 1]
    LogDensityProblems.logdensity_and_gradient(problem, warm)
    for nm in BACKENDS
        LogDensityProblems.logdensity_and_gradient(probs[nm], warm)
    end

    out = Dict{String,Any}()
    for (pname, xs) in pos
        samples = Dict(nm => Float64[] for nm in BACKENDS)
        bares = Float64[]
        for r in 1:ROUNDS
            push!(bares, ns_per_grad(problem, xs))
            for nm in circshift(BACKENDS, r)
                push!(samples[nm], ns_per_grad(probs[nm], xs))
            end
        end
        out[pname] = Dict("bare" => median(bares),
                          [nm => median(samples[nm]) for nm in BACKENDS]...)
        @printf("%-14s d=%-3d %-8s | bare %9.0f | fd %9.0f  const %9.0f  dup %9.0f | const/fd %.2fx\n",
                label, dim, pname, median(bares),
                median(samples["fd"]), median(samples["const"]), median(samples["dup"]),
                median(samples["const"]) / median(samples["fd"]))
        flush(stdout)
    end

    push!(rows, Dict("target" => label, "dim" => dim, "c_source" => csrc,
                     "n_positions" => n, "rounds" => ROUNDS,
                     "typical" => out["typical"], "randn" => out["randn"],
                     "const_over_fd_typical" => out["typical"]["const"] / out["typical"]["fd"],
                     "const_over_fd_randn" => out["randn"]["const"] / out["randn"]["fd"]))
end

@printf("%d rounds, positions from the sampler's own fixed_noncentered arm vs randn.\n\n", ROUNDS)

for t in TARGETS
    prob, dim, jdata = stan_problem(t.name)
    spec = native_spec(t.name, dim, jdata)
    isnothing(spec) && continue
    compare_positions(t.name, prob, spec, dim)
end
let f = Funnel(9), dim = LogDensityProblems.dimension(Funnel(9))
    compare_positions("funnel", f, funnel_spec(f, 1.0), dim)
end

open(joinpath(OUT_DIR, "typical_positions.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "Per-gradient backend cost at sampler-visited (source-frame) positions " *
                  "vs randn(dim). Reconciles the microbenchmark with the end-to-end runs.",
        "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        "rounds" => ROUNDS,
        "warmuphmc_sha" => readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`),
        "rows" => rows), 2)
end
println("\nwrote results/typical_positions.json")
