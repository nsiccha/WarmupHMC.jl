# Summarize `two_stage_tsvd_sample.jl` outputs: one serialized row vector per arm.
#
#   julia --project=docs/benchmark docs/benchmark/two_stage_tsvd_report.jl current.jls scaled.jls two_stage.jls
#
# Two tables. The SAMPLER table compares the arms end to end (each arm is its
# own set of runs, so differences include run-to-run noise). The OFFLINE table
# is paired: within every run, all three fit variants were fitted to that run's
# final halo and judged on its retained draws, so each variant's number is
# reported as a ratio to `current` on the same evidence.

using Serialization, Statistics, Printf

rows = reduce(vcat, deserialize.(ARGS))
ok = filter(r -> r.ok, rows)
failed = filter(r -> !r.ok, rows)
targets = unique(r.target for r in rows)
arms = unique(r.arm for r in rows)

q(v, p) = isempty(v) ? NaN : quantile(v, p)
med(v) = q(v, 0.5)

println("## Sampler (median over seeds; [q25, q75])\n")
println("| target | arm | runs | evals / min ESS | sampling evals / min ESS | min ESS | active (adaptive/diagonal/pathfinder) | reflections | divergences | wall s |")
println("|---|---|---|---|---|---|---|---|---|---|")
for t in targets, a in arms
    r = filter(x -> x.target == t && x.arm == a, ok)
    isempty(r) && continue
    e = [x.evals_per_min_ess for x in r]
    s = [x.sampling_evals_per_min_ess for x in r]
    act(k) = count(x -> x.active === k, r)
    @printf("| %s | %s | %d | %.0f [%.0f, %.0f] | %.0f [%.0f, %.0f] | %.0f | %d/%d/%d | %.0f | %d | %.1f |\n",
            t, a, length(r), med(e), q(e, 0.25), q(e, 0.75), med(s), q(s, 0.25), q(s, 0.75),
            med([x.min_ess for x in r]), act(:adaptive), act(:diagonal), act(:pathfinder),
            med([x.adaptive_refl for x in r]), sum(x.divergences for x in r),
            med([x.wall for x in r]))
end

println("\n## Offline fits on each run's final halo (paired; all arms' runs pooled)\n")
println("| target | variant | runs | in-sample loss | held-out loss | held-out loss / current | kappa / current | reflections | bad left | fit ms | fit ms / current |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for t in targets
    r = filter(x -> x.target == t, ok)
    isempty(r) && continue
    for v in ("current", "scaled", "two_stage")
        get1(x) = only(filter(o -> o.variant == v, x.offline))
        cur(x) = only(filter(o -> o.variant == "current", x.offline))
        o = get1.(r); c = cur.(r)
        kr = [a.kappa / b.kappa for (a, b) in zip(o, c)]
        @printf("| %s | %s | %d | %.3f | %.3f | %.3f | %s | %.0f | %.0f | %.2f | %.2f |\n",
                t, v, length(r), med([x.loss for x in o]), med([x.loss_out for x in o]),
                med([a.loss_out / b.loss_out for (a, b) in zip(o, c)]),
                all(isnan, kr) ? "–" : @sprintf("%.3f", med(kr)),
                med([x.refl for x in o]), med([x.bad for x in o]), med([x.fit_ms for x in o]),
                med([a.fit_ms / b.fit_ms for (a, b) in zip(o, c)]))
    end
end

isempty(failed) || (println("\n## Failed runs\n"); foreach(r -> println("- ", r.target, " ", r.arm, " seed ", r.seed, ": ", r.error), failed))
