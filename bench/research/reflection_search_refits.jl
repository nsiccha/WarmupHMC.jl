# Compare the `:adaptive` reflection-fit variants (`reflection_search_common.jl`)
# at every metric refit captured by `reflection_search_halos.jl`.
#
#   WHMC_BENCH_OUT=<dir> julia --project=bench/research bench/research/reflection_search_refits.jl
#
# Paired: at each refit every variant is fitted to the same halo. Per refit and
# variant:
#   refl        reflections the fit added (bound 16, shared by both stages)
#   bad         coordinates still badly fitted in its frame (log(s1·s2)² ≥ log 2)
#   loss        in-sample selection loss (what `_select_transformation!` compares)
#   loss_next   held-out: the frame's loss on the NEXT refit's halo of the same run
#               (later draws the fit never saw); NaN at a run's last refit
#   wins_diag   whether its loss beats the `:diagonal` candidate's on this halo
#   fit_ms      fastest of 3 fits after a warm-up
# Writes `<out>/refits.jls` and prints the summary tables (markdown).

using LinearAlgebra
BLAS.set_num_threads(1)
using WarmupHMC, Serialization, Statistics, Printf
include("reflection_search_common.jl")

const DIR = ENV["WHMC_BENCH_OUT"]
const RUNS = [deserialize(f) for f in sort(readdir(joinpath(DIR, "halos"); join=true)) if endswith(f, ".jls")]

diag_loss(x, g) = W.update_loss!(Diagonal(ones(size(x, 1))), x, g)

rows = NamedTuple[]
for run in RUNS
    (; target, seed, refits) = run
    for (t, r) in enumerate(refits)
        nxt = t < length(refits) ? refits[t+1] : nothing
        ld = diag_loss(r.x, r.g)
        for v in VARIANTS
            L, loss = fit(v, r.x, r.g)
            push!(rows, (; target, seed, d=size(r.x, 1), n=size(r.x, 2), refit=t, variant=v.name,
                          refl=n_reflections(L), bad=bad_left(L, r.x, r.g), loss, diag_loss=ld,
                          wins_diag=loss < ld,
                          loss_next=isnothing(nxt) ? NaN : frame_loss(L, nxt.x, nxt.g),
                          fit_ms=fit_ms(v, r.x, r.g)))
        end
    end
    println(rpad(target, 45), " seed=", seed, " refits=", length(refits)); flush(stdout)
end
serialize(joinpath(DIR, "refits.jls"), rows)

# ── summary ──────────────────────────────────────────────────────────────────
med(v) = isempty(v) ? NaN : median(v)
gm(v) = isempty(v) ? NaN : exp(mean(log.(v)))
names = [v.name for v in VARIANTS]
targets = unique(r.target for r in rows)
key(r) = (r.target, r.seed, r.refit)
by = Dict((key(r), r.variant) => r for r in rows)
refit_keys = unique(key.(rows))

println("\n## All refits pooled (paired against `shipped`)\n")
println("| variant | refits | adds reflections where shipped stopped early | refl (mean) | bad left (mean) | in-sample loss / shipped (geo. mean) | held-out loss / shipped (geo. mean) | held-out better / worse (>1%) | beats diagonal | fit ms (median) | fit ms / shipped (median) |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for v in names
    rs = [by[(k, v)] for k in refit_keys]; ss = [by[(k, "shipped")] for k in refit_keys]
    early = [s.bad >= 2 for s in ss]
    adds = count(i -> early[i] && rs[i].refl > ss[i].refl, eachindex(rs))
    held = [(a.loss_next, b.loss_next) for (a, b) in zip(rs, ss) if isfinite(a.loss_next) && isfinite(b.loss_next)]
    ratio = [a / b for (a, b) in held]
    @printf("| %s | %d | %d of %d | %.2f | %.2f | %.3f | %.3f | %d / %d | %d | %.2f | %.2f |\n",
            v, length(rs), adds, count(early), mean(r.refl for r in rs), mean(r.bad for r in rs),
            gm([a.loss / b.loss for (a, b) in zip(rs, ss)]), gm(ratio),
            count(<(0.99), ratio), count(>(1.01), ratio), count(r -> r.wins_diag, rs),
            med([r.fit_ms for r in rs]), med([a.fit_ms / b.fit_ms for (a, b) in zip(rs, ss)]))
end

println("\n## Per target (medians over its refits; held-out = loss on the next refit's halo)\n")
println("| target | d | refits | variant | refl | bad left | in-sample loss | held-out loss | held-out / shipped (geo. mean) | fit ms |")
println("|---|---|---|---|---|---|---|---|---|---|")
for t in targets
    ks = filter(k -> k[1] == t, refit_keys)
    d = by[(first(ks), "shipped")].d
    for v in names
        rs = [by[(k, v)] for k in ks]; ss = [by[(k, "shipped")] for k in ks]
        ratio = [a.loss_next / b.loss_next for (a, b) in zip(rs, ss) if isfinite(a.loss_next) && isfinite(b.loss_next)]
        @printf("| %s | %d | %d | %s | %.1f | %.1f | %.2f | %.2f | %.3f | %.2f |\n",
                t, d, length(ks), v, med([r.refl for r in rs]), med([r.bad for r in rs]),
                med([r.loss for r in rs]), med(filter(isfinite, [r.loss_next for r in rs])),
                gm(ratio), med([r.fit_ms for r in rs]))
    end
end
