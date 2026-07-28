# Prints the nonlinear-weighting tables and writes the derived summary JSON.
#
#   julia --project=docs/benchmark docs/benchmark/nonlinear_weighting_report.jl [rows.json]
#
# Separate from `nonlinear_weighting.jl` because that file is `load_harness`ed
# into the docs build, where anything running at include time -- reading ARGS,
# writing a file -- would run during the build. All the derivation lives there;
# this is only the part that acts.
#
# Separate from `nonlinear_weighting_run.jl` because that one re-measures: it
# needs BridgeStan, PosteriorDB and hours of quiet CPU. Reading the checked-in
# rows needs none of it, so checking or correcting a table never costs a
# benchmark run.
include(joinpath(@__DIR__, "nonlinear_weighting.jl"))

const ROWS_PATH = length(ARGS) >= 1 ? ARGS[1] : NW_ROWS_PATH
# Blank is refused, not defaulted -- see `env_dir` in common.jl for why. Inlined
# rather than shared because this script is deliberately dependency-light: the
# docs build loads its derivation, and pulling in common.jl would drag
# BridgeStan and PosteriorDB into `makedocs`.
#
# FOUR sites implement this one rule, not two. `docs/tables.jl`'s `results_dir`
# says "keep the two in lockstep" and names only `common.jl`; the other two are
# here and in `run_linear_restart_benchmark.jl` (which guards
# `WHMC_LINEAR_BENCH_OUT`, a different variable under the same rule). A lockstep
# search anchored on that sentence finds neither. If the rule ever changes,
# change all four: common.jl, tables.jl, this file, run_linear_restart_benchmark.jl.
if haskey(ENV, "WHMC_BENCH_OUT") && isempty(strip(ENV["WHMC_BENCH_OUT"]))
    error("WHMC_BENCH_OUT is set but blank; unset it or give it a real path.")
end
const OUT = get(ENV, "WHMC_BENCH_OUT", dirname(ROWS_PATH))

art = nw_load(ROWS_PATH)
rows, cfg = art["rows"], art["config"]

@printf("Nonlinear weighting study — %d runs\n", length(rows))
@printf("WarmupHMC %s\n", cfg["warmuphmc_sha"])
@printf("measured by %s, %s, host %s\n", cfg["measured_by"], cfg["measured_on"],
        cfg["host"])

println("\n## Efficiency by policy\n")
println(nw_aggregate_table(rows))

println("\n## Paired stepsize-vs-unit, by seed\n")
println(nw_paired_table(rows))

println("\n## Discriminating power\n")
println(nw_discrimination_table(rows))

println("""
A target whose arms are bit-identical contributes NOTHING to the policy
question — it cannot prefer a policy it never actually ran differently. Ties
from such a target are an artifact of the target, not evidence that the
policies are equivalent, and are excluded from the conclusion rather than
counted toward it.
""")

println("\n## Conclusion (derived)\n")
println(nw_conclusion(rows))

# The summary is written NEXT TO the rows for convenience and is GITIGNORED
# there (`results/nonlinear_weighting/.gitignore`) so it can never be committed.
# That is not tidiness: a stored summary beside the rows is a second
# representation of one fact, and nothing keeps the two in step. A reader who
# finds a stale `1.051` here next to rows that now say `1.048` has no way to
# tell which is current, and the wrong one is the one that looks authoritative.
# Regenerate it whenever you want it; never cite it in preference to the rows.
summary = nw_summary(rows, cfg)
mkpath(OUT)
const SUMMARY_PATH = joinpath(OUT, "nonlinear_weighting_summary.json")
open(SUMMARY_PATH, "w") do io
    JSON.print(io, summary, 2)
end
println("\nwrote ", SUMMARY_PATH, " (derived, gitignored — the rows are the artifact)")
