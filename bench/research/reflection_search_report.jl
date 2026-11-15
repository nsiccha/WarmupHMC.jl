# Summarize `reflection_search_sample.jl` outputs (one serialized row vector per
# arm) as markdown:
#
#   julia --project=bench/research bench/research/reflection_search_report.jl <out>/sample_*.jls
#
# Each arm is its own set of runs (same seeds), so differences include
# run-to-run noise: medians with quartiles, and per-seed paired ratios against
# the `shipped` arm where it is present.

using Serialization, Statistics, Printf

rows = reduce(vcat, deserialize.(ARGS))
ok = filter(r -> r.ok, rows)
failed = filter(r -> !r.ok, rows)
targets = unique(r.target for r in rows)
arms = unique(r.arm for r in rows)

q(v, p) = isempty(v) ? NaN : quantile(v, p)
med(v) = q(v, 0.5)
gm(v) = isempty(v) ? NaN : exp(mean(log.(v)))
find_row(t, a, s) = (i = findfirst(x -> x.target == t && x.arm == a && x.seed == s, ok); isnothing(i) ? nothing : ok[i])

println("## Sampler, per target (median over seeds [q25, q75]; ratio = geometric mean over seeds of arm / shipped, same seed)\n")
println("| target | arm | runs | evals / min ESS | ratio | steps / draw | min ESS | final metric adaptive/diagonal/pathfinder | reflections | divergences | wall s |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for t in targets, a in arms
    r = filter(x -> x.target == t && x.arm == a, ok)
    isempty(r) && continue
    e = [x.evals_per_min_ess for x in r]
    ratio = Float64[]
    for x in r
        s = find_row(t, "shipped", x.seed)
        isnothing(s) || push!(ratio, x.evals_per_min_ess / s.evals_per_min_ess)
    end
    act(k) = count(x -> x.active === k, r)
    @printf("| %s | %s | %d | %.0f [%.0f, %.0f] | %s | %.1f | %.0f | %d/%d/%d | %.0f | %d | %.1f |\n",
            t, a, length(r), med(e), q(e, 0.25), q(e, 0.75),
            isempty(ratio) ? "–" : @sprintf("%.2f", gm(ratio)),
            med([x.steps_per_draw for x in r]), med([x.min_ess for x in r]),
            act(:adaptive), act(:diagonal), act(:pathfinder),
            med([x.adaptive_refl for x in r]), sum(x.divergences for x in r), med([x.wall for x in r]))
end

println("\n## All targets pooled (geometric mean over target × seed of arm / shipped)\n")
println("| arm | paired runs | evals / min ESS | wall time | better / worse by > 10% (evals / min ESS) |")
println("|---|---|---|---|---|")
for a in arms
    a == "shipped" && continue
    pairs = [(x, find_row(x.target, "shipped", x.seed)) for x in ok if x.arm == a]
    pairs = filter(p -> !isnothing(p[2]), pairs)
    isempty(pairs) && continue
    re = [x.evals_per_min_ess / s.evals_per_min_ess for (x, s) in pairs]
    rw = [x.wall / s.wall for (x, s) in pairs]
    @printf("| %s | %d | %.3f | %.3f | %d / %d |\n", a, length(pairs), gm(re), gm(rw),
            count(<(0.9), re), count(>(1.1), re))
end

isempty(failed) || (println("\n## Failed runs\n"); foreach(r -> println("- ", r.target, " ", r.arm, " seed ", r.seed, ": ", r.error), failed))
