#!/usr/bin/env julia

# Prints every number `results/brm_inventory_standard/README.md` quotes.
#
# The README beside the artifact is a static census: nothing executes it, so a
# figure edited by hand there can drift from the rows it claims to summarise and
# no check will notice. This is the same arrangement `backend_bands.jl` has with
# `RESULTS.md` — run this after any regeneration and paste what it prints, rather
# than transcribing from a table on screen.
#
# It only READS. It needs the docs stack (JSON, Statistics, Printf) and nothing
# from the measurement stack, so it runs in the same environment the docs build
# uses.
#
#   julia --startup-file=no --project=docs docs/benchmark/brm_inventory_report.jl
#   julia --startup-file=no --project=docs docs/benchmark/brm_inventory_report.jl /path/to/rows.json
#
# A SECOND path is an optional BASELINE to compare per-gradient cost against —
# the previously published artifact, which is why the README can say the
# wrapper's cost changed between two pins rather than merely asserting it. That
# baseline lives in git, not on anyone's disk, so extract it there:
#
#   git show <old-sha>:docs/benchmark/results/brm_inventory_standard/rows.json > /tmp/base.json
#   julia … docs/benchmark/brm_inventory_report.jl \
#     docs/benchmark/results/brm_inventory_standard/rows.json /tmp/base.json

import JSON, Printf, SHA, Statistics

const HERE = @__DIR__
Base.include(@__MODULE__, joinpath(HERE, "brm_catalogue.jl"))

# `brm_plots.jl` needs AlgebraOfVega, which this report does not. Only the label
# table is wanted, so it is read out of that file rather than pulling the figure
# stack in — the labels are the axis names the page publishes, and duplicating
# them here would let the two drift.
const PLOT_LABELS = let
    src = read(joinpath(HERE, "brm_plots.jl"), String)
    Dict(String(m.captures[1]) => String(m.captures[2])
         for m in eachmatch(r"\"([a-z_0-9]+:[a-z_0-9]+)\"\s*=>\s*\"([^\"]+)\"", src))
end

label(spec) = get(PLOT_LABELS, spec, spec)

length(ARGS) <= 2 ||
    error("usage: brm_inventory_report.jl [rows.json [baseline-rows.json]]")
path = isempty(ARGS) ?
    joinpath(HERE, "results", "brm_inventory_standard", "rows.json") : first(ARGS)
baseline_path = length(ARGS) == 2 ? ARGS[2] : nothing
d = JSON.parsefile(path)
c = d["config"]
models = brmc_models(d)
rows = brmc_rows(d)
arms = brmc_standard_arm_order()
fails = brmc_failures(d)

pct(x) = Printf.@sprintf("%.1f%%", 100x)
r2(x) = Printf.@sprintf("%.2f", x)

# A sweep flushes per spec, so a `rows.json` read mid-run is complete, valid
# JSON describing a fraction of the design. That is useful to watch and must not
# be pasted anywhere: `run_finished_at` is the only field that distinguishes the
# two, so it is checked once here loudly rather than surfacing later as a
# `KeyError` on some field a finished run happens to have.
if !haskey(c, "run_finished_at")
    println("> **PARTIAL RUN — DO NOT PASTE.** This artifact has no ",
            "`run_finished_at`: ", length(models), " of the design's models and ",
            length(rows), " rows are present so far.\n")
end

println("## Recorded run\n")
println("- Logical compute host: `", c["host"], "`")
println("- Julia: ", c["julia"], "; BLAS threads: ", c["blas_threads"])
println("- WarmupHMC: `", c["warmuphmc_sha"], "`",
        c["warmuphmc_src_dirty"] ? " (src DIRTY)" : "")
println("- BayesianRegressionModels: `", c["brm_sha"], "`",
        c["brm_inventory_dirty"] ? " (inventory DIRTY)" : "")
println("- StanBlocks: `", c["stanblocks_sha"], "`")
println("- DynamicHMC: ", c["dynamichmc_version"])
println("- Runner + adapters: `", get(c, "runner_sha256", "not recorded"), "`")
println("- Inventory `translations.tsv`: `", c["translations_sha256"], "`")
println("- Inventory `model_matrix.tsv`: `", c["model_matrix_sha256"], "`")
println("- ", c["n_seeds"], " seeds × ", c["n_draws"], " retained draws × ",
        length(arms), " arms × ", length(models), " models = ", length(rows), " rows")
println("- One ", c["timing_preflight_draws"],
        "-draw untimed preflight per arm; recorded arm order rotates by seed")
println("- Runner elapsed: ", get(c, "total_elapsed_s", "not recorded"), " seconds")
println("- Summed recorded sampling time: ",
        round(sum(r["wall_s"] for r in rows); digits=3), " seconds")
println("- Result: ", length(rows) - length(fails), "/", length(rows),
        " usable trajectories")
println("- Total divergences, including any failed trajectory: ",
        sum(r["n_divergent"] for r in rows))
println("- `rows.json` SHA-256: `", bytes2hex(open(SHA.sha256, path)), "`")

println("\n## Explicit failures\n")
if isempty(fails)
    println("None: every model/arm/seed cell returned a usable trajectory.")
else
    for f in fails
        println("- `", f["spec"], "` / ", brmc_standard_arm_label(f["arm"]),
                " / seed ", f["seed"], ": ", f["n_constant"],
                " constant constrained coordinates, ", f["n_divergent"],
                " divergences — ", first(eachsplit(f["error"], '\n')))
    end
end

println("\n## Divergences by arm\n")
println("| default-warmup arm | divergences | trajectories with any |")
println("| --- | ---: | ---: |")
for arm in arms
    ar = [r for r in rows if r["arm"] == arm]
    println("| ", brmc_standard_arm_label(arm), " | ",
            sum(r["n_divergent"] for r in ar), " | ",
            count(r -> r["n_divergent"] > 0, ar), "/", length(ar), " |")
end

println("\n## Degeneracy by model\n")
println("| model | divergences | most dropped coordinates | shared names | failed cells |")
println("| --- | ---: | ---: | ---: | ---: |")
for m in models
    sr = [r for r in rows if r["spec"] == m["spec"]]
    println("| ", label(m["spec"]), " | ", sum(r["n_divergent"] for r in sr), " | ",
            maximum(r["n_constant"] for r in sr), " | ", m["n_names_shared"], " | ",
            count(r -> !r["ok"], sr), " |")
end

println("\n## Structure\n")
println("| model | inferred family | grouping factors (levels) | blocks | n | dim |")
println("| --- | --- | --- | --- | ---: | ---: |")
for m in models
    println("| ", label(m["spec"]), " | ", m["inferred_family"], " | ",
            brmc_group_summary(m), " | ", brmc_block_summary(m), " | ",
            m["n_obs"], " | ", m["dim_noncentered"], " |")
end

println("\n## Seed-paired median ratios\n")
for (num, den, gloss) in (
        ("warmuphmc_noncentered", "dynamichmc_noncentered",
         "WarmupHMC over DynamicHMC on the identical generated non-centered target"),
        ("warmuphmc_centered", "dynamichmc_centered",
         "the same comparison on the generated centered target"),
        ("warmuphmc_fixed_centering", "warmuphmc_noncentered",
         "cost of wrapping the non-centered target at unchanged geometry"),
        ("warmuphmc_adaptive_centering", "warmuphmc_fixed_centering",
         "effect of fitting centerings, isolated from the wrapper"),
        ("warmuphmc_adaptive_centering", "dynamichmc_noncentered",
         "the fitted nonlinear path against DynamicHMC's non-centered endpoint"),
        ("warmuphmc_adaptive_centering", "dynamichmc_centered",
         "the fitted nonlinear path against DynamicHMC's centered endpoint"))
    println("### `", num, "` / `", den, "`\n")
    println("*", gloss, ".*\n")
    println("| model | ESS/gradient | gradients | wall time | ESS/second | paired seeds |")
    println("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in brmc_standard_ratios(d, num, den)
        println("| ", label(r.spec), " | ", r2(r.ess_per_grad_ratio), "× | ",
                r2(r.grad_ratio), "× | ", r2(r.wall_ratio), "× | ",
                r2(r.ess_per_s_ratio), "× | ", r.n, " |")
    end
    println()
end

println("## Median wall seconds by arm\n")
print("| model |")
for arm in arms
    print(" ", brmc_standard_arm_label(arm), " |")
end
println()
println("| --- |", repeat(" ---: |", length(arms)))
for m in models
    print("| ", label(m["spec"]), " |")
    for arm in arms
        v = [r["wall_s"] for r in rows
             if r["spec"] == m["spec"] && r["arm"] == arm && r["ok"]]
        print(" ", Printf.@sprintf("%.3f", brmc_med(v)), " |")
    end
    println()
end

# Wall seconds divided by the exact gradient count. This is the only per-arm
# cost figure that is comparable ACROSS runs: the seed's geometry sets how many
# gradients get taken, and dividing it out leaves the price of one gradient
# through that arm's wrapping. It is what lets the README attribute a wall-time
# change to the wrapper rather than to the box or to the target.
us_per_grad(rs) = brmc_med([1e6 * r["wall_s"] / r["grad_evals"] for r in rs])
cells(rs, spec, arm) = [r for r in rs if r["spec"] == spec && r["arm"] == arm && r["ok"]]

println("\n## Median microseconds per gradient by arm\n")
base = baseline_path === nothing ? nothing : JSON.parsefile(baseline_path)
if base !== nothing
    bc = base["config"]
    println("Each cell is `baseline → this run`. Baseline `", baseline_path, "`: WarmupHMC `",
            bc["warmuphmc_sha"], "`, Julia ", bc["julia"], ", host `", bc["host"], "`.")
    println("Only models present in BOTH artifacts are listed; a differing host or")
    println("Julia version makes the comparison meaningless, so check the line above.\n")
end
print("| model |")
for arm in arms
    print(" ", brmc_standard_arm_label(arm), " |")
end
println()
println("| --- |", repeat(" ---: |", length(arms)))
for m in models
    spec = m["spec"]
    base_rows = base === nothing ? nothing : brmc_rows(base)
    if base_rows !== nothing && isempty(cells(base_rows, spec, first(arms)))
        continue  # model is new in this run; nothing to compare it against
    end
    print("| ", label(spec), " |")
    for arm in arms
        now = us_per_grad(cells(rows, spec, arm))
        if base_rows === nothing
            print(" ", Printf.@sprintf("%.1f", now), " |")
        else
            print(" ", Printf.@sprintf("%.1f → %.1f", us_per_grad(cells(base_rows, spec, arm)), now), " |")
        end
    end
    println()
end
if base !== nothing
    shared = [m["spec"] for m in models
              if !isempty(cells(brmc_rows(base), m["spec"], first(arms)))]
    println("\n", length(shared), " shared model(s); ", length(models) - length(shared),
            " model(s) in this run have no baseline and are omitted above.")
    println("\nMedian gradient counts, baseline vs this run — a per-gradient")
    println("comparison only means something where these agree:\n")
    differ = Ref(0)  # top-level `for` opens a soft scope; a plain counter would be local
    for spec in shared, arm in arms
        a = brmc_med([Float64(r["grad_evals"]) for r in cells(brmc_rows(base), spec, arm)])
        b = brmc_med([Float64(r["grad_evals"]) for r in cells(rows, spec, arm)])
        a == b && continue
        differ[] += 1
        Printf.@printf("- DIFFERS by %+.1f%%: `%s` / %s: %.0f → %.0f\n",
                       100 * (b - a) / a, spec, brmc_standard_arm_label(arm), a, b)
    end
    println("\n", length(shared) * length(arms) - differ[], " of ", length(shared) * length(arms),
            " shared model × arm median gradient counts are identical",
            differ[] == 0 ? "." : "; the rest are listed above.")
end

# The longest-running spec is the one whose wall times a co-tenant on the box
# could plausibly have perturbed, so it gets the load audit the README quotes.
# Per-gradient cost is the discriminator: contention inflates seconds without
# touching the gradient counter, so a load artefact shows up as one cell that is
# slow PER GRADIENT, while a merely hard seed is slow only in total.
longest = argmax(spec -> sum(r["wall_s"] for r in rows if r["spec"] == spec),
                 [m["spec"] for m in models])
println("\n## Wall-time spread in the longest spec\n")
Printf.@printf("`%s` — %.1f s of recorded sampling, %.0f%% of the run's total.\n\n",
               longest, sum(r["wall_s"] for r in rows if r["spec"] == longest),
               100 * sum(r["wall_s"] for r in rows if r["spec"] == longest) /
                     sum(r["wall_s"] for r in rows))
println("| arm | s/gradient range | max/median | median wall s | max wall s |")
println("| --- | --- | ---: | ---: | ---: |")
for arm in arms
    rs = cells(rows, longest, arm)
    isempty(rs) && continue
    sg = [r["wall_s"] / r["grad_evals"] for r in rs]
    w = [r["wall_s"] for r in rs]
    Printf.@printf("| %s | %.5f–%.5f | %.2f | %.2f | %.2f |\n",
                   brmc_standard_arm_label(arm), minimum(sg), maximum(sg),
                   maximum(sg) / brmc_med(sg), brmc_med(w), maximum(w))
end
println("\nWidest raw `wall_s` cells, each against its own arm's median:\n")
armmed = Dict(arm => brmc_med([r["wall_s"] for r in cells(rows, longest, arm)]) for arm in arms)
worst = sort([r for r in rows if r["spec"] == longest && r["ok"]];
             by = r -> -r["wall_s"] / armmed[r["arm"]])
for r in first(worst, 5)
    Printf.@printf("- %.2f× its arm median — %s, seed %d: %.2f s over %d gradients\n",
                   r["wall_s"] / armmed[r["arm"]], brmc_standard_arm_label(r["arm"]),
                   r["seed"], r["wall_s"], r["grad_evals"])
end
