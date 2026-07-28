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

loadall(dir) = JSON.parse(read(joinpath(dir, "runs.json"), String))
load(dir) = loadall(dir)["runs"]
RAW_A, RAW_B = loadall(A_DIR), loadall(B_DIR)
A, B = RAW_A["runs"], RAW_B["runs"]

prov(raw, k, dflt) = get(raw, k, dflt)
const SHA_A = prov(RAW_A, "warmuphmc_sha", nothing)
const SHA_B = prov(RAW_B, "warmuphmc_sha", nothing)
const AD_A  = prov(RAW_A, "ad_backend", "unrecorded (pre-dates the field — ForwardDiff)")
const AD_B  = prov(RAW_B, "ad_backend", "unrecorded (pre-dates the field — ForwardDiff)")

# Wall-clock is comparable ONLY when both runs compiled the same package source.
# Two runs of different WarmupHMC revisions differ in seconds for reasons that
# have nothing to do with what is being compared. Two runs of the SAME revision
# under different AD backends differ in seconds for exactly the reason being
# compared — and there, omitting wall-clock would throw away the result.
const SAME_SOURCE = SHA_A !== nothing && SHA_A == SHA_B
const DIRTY = prov(RAW_A, "worktree_dirty", false) === true ||
              prov(RAW_B, "worktree_dirty", false) === true

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
println("| | ", basename(A_DIR), " | ", basename(B_DIR), " |")
println("|---|---|---|")
println("| WarmupHMC | `", something(SHA_A, "unrecorded"), "` | `", something(SHA_B, "unrecorded"), "` |")
println("| AD backend | ", AD_A, " | ", AD_B, " |")
println()
DIRTY && println("> ⚠ At least one run was measured on a DIRTY worktree — its SHA does not\n> fully describe the code that produced it.\n")
if SAME_SOURCE
    println("Both runs compiled the SAME package source, so wall-clock IS comparable")
    println("here and is reported below alongside the gradient counts.")
else
    println("Min ESS per 1000 gradient evaluations, median over seeds. Wall-clock is")
    println("deliberately omitted — the two runs compiled different package sources.")
end
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
# Wall-clock — only when both runs compiled the same package source
# ---------------------------------------------------------------------------
#
# This is the half of the verdict that an AD-backend change moves. The wrapper
# re-differentiates `ljac(x) + dot(g_y, y(x))` on every gradient call, so the
# backend sets the per-call cost while leaving the number of calls alone: expect
# `grad_evals` above to be near-identical across backends and these numbers not
# to be.

if SAME_SOURCE
    println()
    println("## Wall-clock, same package source — the half a backend change moves")
    println()
    println("Min ESS per second, median over seeds:")
    println()
    println("| target | arm | before | after | change |")
    println("|---|---|---|---|---|")
    for t in targets, a in ARM_ORDER
        ra, rb = sel(A, t, a), sel(B, t, a)
        (isempty(ra) || isempty(rb)) && continue
        x, y = med(ra, "ess_min_per_s"), med(rb, "ess_min_per_s")
        @printf("| `%s` | %s | %s | %s | %s |\n", t, a, fmt(x), fmt(y), pct(x, y))
    end

    println()
    println("Seconds per run, median over seeds (lower is better):")
    println()
    println("| target | arm | before | after | change |")
    println("|---|---|---|---|---|")
    for t in targets, a in ARM_ORDER
        ra, rb = sel(A, t, a), sel(B, t, a)
        (isempty(ra) || isempty(rb)) && continue
        x, y = med(ra, "wall_s"), med(rb, "wall_s")
        @printf("| `%s` | %s | %s | %s | %s |\n", t, a, fmt(x; d = 2), fmt(y; d = 2), pct(x, y))
    end

    # The claim under test, stated as a check rather than left to the reader:
    # a backend swap must not change how many gradients the sampler asks for.
    println()
    worst = 0.0
    for t in targets, a in ARM_ORDER
        ra, rb = sel(A, t, a), sel(B, t, a)
        (isempty(ra) || isempty(rb)) && continue
        x, y = med(ra, "grad_evals"), med(rb, "grad_evals")
        (isfinite(x) && isfinite(y) && x != 0) || continue
        worst = max(worst, abs(y - x) / x)
    end
    @printf("Largest gradient-count drift across all arms: %.1f%%. ", 100 * worst)
    println(worst < 0.02 ?
            "Under 2% — the backend changed the cost per gradient, not the sampling." :
            "**Over 2% — the runs are NOT sampling-identical, so the wall-clock\ndelta above is not a clean backend comparison. Investigate before quoting it.**")
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
