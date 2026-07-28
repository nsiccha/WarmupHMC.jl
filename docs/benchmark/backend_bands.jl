# Recompute the Enzyme-vs-ForwardDiff bands in RESULTS.md from the checked-in
# JSON, instead of copying them by hand.
#
# RESULTS.md states the trap this script exists to close, in its own words:
#
#   "A band derived from checked-in JSON is only a floor until someone
#    regenerates the JSON."
#
# It was written about somebody else's numbers and it applies to this document's
# own. A widening landed FOUR MINUTES before a regeneration moved the files it
# was computed from, and the resulting band excluded values that were in the
# files and included values that were in none of them. Nothing detected that:
# the band is prose, the files are JSON, and a hand-copied number cannot go
# stale loudly. So the band stops being prose. Run this after any regeneration
# and paste what it prints.
#
# WHAT COUNTS AS A COMPARISON. One (target, harness, condition) triple for which
# a `const`-annotated Enzyme timing and a ForwardDiff timing were taken in the
# same process, same round structure, same positions. Ratios below 1 mean Enzyme
# is faster. Four harnesses contribute:
#
#   backend_replication.json  2 per target — one centering endpoint, one c = 0.5
#   typical_positions.json    2 per target — at `randn`, and at sampler positions
#   annotation_sweep.json     2 per target — both centering endpoints
#   <backend>-<sha>/gradient_overhead.json
#                             1 per target — the driver's own block, which is
#                                 the ONLY cross-process comparison here: it
#                                 divides the Enzyme run's `live_ns` by the
#                                 ForwardDiff run's, so it is the one number a
#                                 within-process ordering effect cannot explain.
#
# SUPERSESSION IS NOT OPTIONAL HERE. Three older driver run-dir pairs sit in
# `results/`, one of them measured against the boxed specs where Enzyme read
# 1.2-5.4x SLOWER. Averaging those into the band would not look like an error --
# it would look like a wider, more honest range. So the live set is taken from
# `artifact_currency.jl`'s own `superseded_reason`, not from a glob: the two
# scripts cannot disagree about what is live, because there is one implementation
# of that question.
#
# AND ABSENCE IS AN ERROR, NOT A NARROWER BAND. Every source below is named
# explicitly and missing one is fatal. A script that globbed `results/` would
# respond to a deleted harness by quietly reporting a band over what remains,
# which is the failure mode where a check that verifies less still prints green.
include(joinpath(@__DIR__, "artifact_currency.jl"))
import JSON
using Printf

const DRIVER_SHA = "d68d680"   # the one live driver pair; see `superseded_reason`

# posteriordb names carry the data set and the parametrization; the tables use
# the model alone. Derived rather than tabulated, but NOT silently: an
# unrecognised shape is an error, because a pass-through default would put a
# 40-character row label in the table and still print green.
#
# The parametrization is not reliably a SUFFIX: `seeds_data-seeds_centered_model`
# puts it in the middle. Stripping only a trailing `_centered` would have left
# that row labelled `seeds_centered_model` — a real row under a name no table
# uses, which is why the fallback is an error and not a pass-through.
function short_target(name)
    name == "funnel" && return "funnel"
    m = match(r"^[^-]+-(.+)$", name)
    m === nothing && error("cannot shorten target name `$name` — expected " *
                           "posteriordb's `<data>-<model>`; update `short_target` " *
                           "rather than letting an unlabelled row through")
    model = replace(m.captures[1], r"_model$" => "")
    stripped = replace(model, r"_(?:non)?centered" => "")
    stripped == model && error("target `$name` names no parametrization — a band " *
                               "row must say which one it measured")
    stripped
end

"""Read a live artifact, or fail. `path` is repo-relative, as in `artifacts()`."""
function live_json(path)
    reason = superseded_reason(path)
    reason === nothing || error("`$path` is SUPERSEDED ($reason) — it must not " *
                                "contribute to a band that describes the sampler today")
    full = joinpath(REPO, path)
    isfile(full) || error("`$path` is missing — a band computed without it would " *
                          "be narrower and indistinguishable from a correct one")
    JSON.parsefile(full)
end

# (target, harness, condition, ratio)
comparisons = NamedTuple[]
add!(t, h, c, r) = push!(comparisons, (target = short_target(t), harness = h,
                                       condition = c, ratio = r))

let d = live_json(joinpath(RESULTS, "backend_replication.json"))
    for r in d["rows"]
        # This harness already stores `all` beside `median`/`min`/`max` — the
        # band takes the median, the same statistic the other three report.
        add!(r["target"], "replicate_backends", @sprintf("c=%.2f", r["c_source"]),
             r["const"]["median"] / r["fd"]["median"])
    end
end

let d = live_json(joinpath(RESULTS, "typical_positions.json"))
    for r in d["rows"]
        add!(r["target"], "typical_positions", "typical", r["const_over_fd_typical"])
        add!(r["target"], "typical_positions", "randn", r["const_over_fd_randn"])
    end
end

let d = live_json(joinpath(RESULTS, "annotation_sweep.json"))
    for r in d["rows"]
        add!(r["target"], "annotation_sweep", @sprintf("c=%.2f", r["c_source"]),
             r["ns_const"] / r["ns_forwarddiff"])
    end
end

let enz = live_json(joinpath(RESULTS, "enzyme-$DRIVER_SHA", "gradient_overhead.json")),
    fwd = live_json(joinpath(RESULTS, "forwarddiff-$DRIVER_SHA", "gradient_overhead.json"))
    fwd_of = Dict(o["target"] => o for o in fwd["overheads"])
    for o in enz["overheads"]
        f = get(fwd_of, o["target"], nothing)
        f === nothing && error("driver pair $DRIVER_SHA: `$(o["target"])` is in the " *
                               "Enzyme run but not the ForwardDiff one — the pair is " *
                               "not a matched comparison")
        add!(o["target"], "driver/gradient_overhead", "live_ns", o["live_ns"] / f["live_ns"])
    end
end

order = ["radon_partially_pooled", "radon_variable_intercept", "seeds",
         "eight_schools", "funnel"]
seen = unique(c.target for c in comparisons)
setdiff(seen, order) |> s -> isempty(s) || error("targets not in the printed order: $s")

println("Enzyme(Const) / ForwardDiff, per-gradient. Ratio < 1 means Enzyme is faster.\n")
println("| target | `d` | Enzyme ÷ ForwardDiff | reading |")
println("|---|---|---|---|")
dims = Dict{String,Int}()
let d = live_json(joinpath(RESULTS, "backend_replication.json"))
    for r in d["rows"]; dims[short_target(r["target"])] = r["dim"]; end
end
for t in order
    rs = [c.ratio for c in comparisons if c.target == t]
    isempty(rs) && error("no comparisons for `$t`")
    lo, hi = extrema(rs)
    @printf("| `%s` | %d | **%.2f–%.2f×** | Enzyme %.1f–%.1f× faster |\n",
            t, dims[t], lo, hi, 1/hi, 1/lo)
end
const total = length(comparisons)
const faster = count(c -> c.ratio < 1, comparisons)

@printf("\nEnzyme is faster in %d of the %d comparisons the four harnesses produce.\n",
        faster, total)
if faster < total
    println("\nNOT unanimous — the exceptions, which the prose must name:")
    for c in comparisons
        c.ratio < 1 || @printf("  %-26s %-24s %-8s %.3f×\n",
                               c.target, c.harness, c.condition, c.ratio)
    end
end

println("\nPer harness (every comparison, so the table above can be checked by hand):")
for h in unique(c.harness for c in comparisons)
    println("  ", h)
    for c in comparisons
        c.harness == h && @printf("    %-26s %-10s %.3f×\n", c.target, c.condition, c.ratio)
    end
end
