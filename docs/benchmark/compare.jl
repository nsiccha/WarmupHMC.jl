# Before/after comparison across two benchmark runs.
#
#   julia --project=docs/benchmark docs/benchmark/compare.jl \
#         docs/benchmark/results/before docs/benchmark/results/after
#
# Written for the halo-recording regression 34ce034 / fix c8fed88, but it is
# generic: any two result directories produced by run_reparam_benchmark.jl with
# the same seeds, targets and draw floor can be compared this way.
#
# The metric to read is ESS per 1000 gradient evaluations. Wall-clock is not
# comparable across two runs that recompiled a changed package.

import JSON
using Statistics, Printf

const A_DIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results", "before")
const B_DIR = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results", "after")

load(dir) = JSON.parse(read(joinpath(dir, "runs.json"), String))["runs"]
A, B = load(A_DIR), load(B_DIR)

num(x) = x === nothing ? NaN : Float64(x)
sel(rs, t, a) = [r for r in rs if r["target"] == t && r["arm"] == a && r["ok"]]
med(rs, k) = (v = filter(isfinite, [num(r[k]) for r in rs]); isempty(v) ? NaN : median(v))
fmt(x; d = 1) = isfinite(x) ? string(round(x; digits = d)) : "—"

pct(a, b) = (!isfinite(a) || !isfinite(b) || a == 0) ? "—" :
            @sprintf("%+.0f%%", 100 * (b - a) / a)

const ARM_ORDER = ["plain", "fixed_centered", "adaptive", "fixed_noncentered", "sibling_plain"]
targets = unique([r["target"] for r in A])

println("# Before/after: ", basename(A_DIR), " → ", basename(B_DIR))
println()
println("Min ESS per 1000 gradient evaluations, median over seeds. Wall-clock is")
println("deliberately omitted — the two runs compiled different package sources.")
println()
println("| target | arm | before | after | change |")
println("|---|---|---|---|---|")
for t in targets, a in ARM_ORDER
    ra, rb = sel(A, t, a), sel(B, t, a)
    (isempty(ra) || isempty(rb)) && continue
    x = 1000 * med(ra, "ess_min_per_grad")
    y = 1000 * med(rb, "ess_min_per_grad")
    @printf("| `%s` | %s | %s | %s | %s |\n", t, a, fmt(x; d = 2), fmt(y; d = 2), pct(x, y))
end

println()
println("Gradient evaluations spent, median over seeds — a window that adapts")
println("better should need fewer:")
println()
println("| target | arm | before | after | change |")
println("|---|---|---|---|---|")
for t in targets, a in ARM_ORDER
    ra, rb = sel(A, t, a), sel(B, t, a)
    (isempty(ra) || isempty(rb)) && continue
    x, y = med(ra, "grad_evals"), med(rb, "grad_evals")
    @printf("| `%s` | %s | %s | %s | %s |\n", t, a, fmt(x; d = 0), fmt(y; d = 0), pct(x, y))
end

# ---------------------------------------------------------------------------
# Stuck adaptation
# ---------------------------------------------------------------------------
#
# A seed is "stuck" when every coordinate's source centering finished at exactly
# the value it started from — the model's own `c_native`. That is the signature
# of optimize!'s per-index early return (`nobs(or) > 2 || return idx => value`),
# which hands back the centering unchanged when the halo pool it was given is
# too thin. If the regressed pool was the cause, these counts should fall.

stuck(r) = (cs = num.(r["c_after"]); !isempty(cs) && all(c -> c == num(r["c_native"]), cs))

println()
println("## Stuck adaptation — seeds whose `c` never moved off its starting value")
println()
println("| target | before | after |")
println("|---|---|---|")
for t in targets
    ra, rb = sel(A, t, "adaptive"), sel(B, t, "adaptive")
    (isempty(ra) || isempty(rb)) && continue
    sa = [r["seed"] for r in ra if stuck(r)]
    sb = [r["seed"] for r in rb if stuck(r)]
    lbl(s, n) = isempty(s) ? "0 / $n" : "$(length(s)) / $n (seed$(length(s) == 1 ? " " : "s ")$(join(sort(s), ", ")))"
    @printf("| `%s` | %s | %s |\n", t, lbl(sa, length(ra)), lbl(sb, length(rb)))
end

println()
println("Min ESS on the adaptive arm, worst seed vs median — the tail the stuck")
println("seeds produce:")
println()
println("| target | before worst | before median | after worst | after median |")
println("|---|---|---|---|---|")
for t in targets
    ra, rb = sel(A, t, "adaptive"), sel(B, t, "adaptive")
    (isempty(ra) || isempty(rb)) && continue
    va = filter(isfinite, [num(r["ess_min"]) for r in ra])
    vb = filter(isfinite, [num(r["ess_min"]) for r in rb])
    @printf("| `%s` | %s | %s | %s | %s |\n", t,
            fmt(minimum(va)), fmt(median(va)), fmt(minimum(vb)), fmt(median(vb)))
end

# ---------------------------------------------------------------------------
# Final centerings
# ---------------------------------------------------------------------------

println()
println("## Where adaptation settled, median `c` per seed")
println()
println("| target | before | after |")
println("|---|---|---|")
for t in targets
    ra, rb = sel(A, t, "adaptive"), sel(B, t, "adaptive")
    (isempty(ra) || isempty(rb)) && continue
    ca = sort([median(num.(r["c_after"])) for r in ra])
    cb = sort([median(num.(r["c_after"])) for r in rb])
    @printf("| `%s` | %s | %s |\n", t,
            join(string.(round.(ca; digits = 2)), " "),
            join(string.(round.(cb; digits = 2)), " "))
end
