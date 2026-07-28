# Verify that every artifact-derived figure RESULTS.md quotes still matches the
# checked-in JSON it was read out of.
#
# WHY THIS EXISTS. `RESULTS.md` is not a built page. It is outside `docs/src/`,
# it is not in `make.jl`'s `pages` list, and it contains no `@eval`/`@example`
# fence -- so `docs/make.jl` never executes a line of it and a green docs build
# says nothing about it. Every number in it is a static census that only a human
# can correct and that nothing can ever notice going stale. `artifact_currency.jl`
# closes the neighbouring gap (is the JSON still describing today's sampler?) and
# deliberately does not close this one (does the PROSE still match the JSON?).
#
# That gap has now produced three separate defects in one day:
#
#   1. A hand-copied Enzyme/ForwardDiff band, widened four minutes before a
#      regeneration moved the files it was computed from. Fixed by generating it
#      -- `backend_bands.jl`.
#   2. A REVERSED verdict: the funnel's evaluation-position ratios read
#      0.70x/0.38x in the prose against 0.41x/0.65x in the artifact, with a
#      mechanism already attached to the wrong direction. A plausible explanation
#      is a stronger anti-check than no explanation at all.
#   3. `capture_boxing.jl`'s A/B bare gradient quoted as BOTH `27431 ns` and
#      `25863 ns`, fifteen lines apart, describing the same run. Regenerating the
#      artifact updated the second and missed the first, and the two figures are
#      too far apart on the page for a reader to collide them.
#
# All three are the same shape: a number that was true when it was typed, in a
# file where nothing can tell it has stopped being true. So the check stops being
# a reading pass.
#
# WHAT THIS CHECKS, AND WHAT IT DELIBERATELY DOES NOT. It checks figures that are
# MECHANICALLY DERIVABLE from a live artifact. It does not check prose judgements
# ("small enough to be paid out of the sampling gain"), and it cannot: those are
# arguments, not values. A number this script does not know about is unprotected
# exactly as before -- so `n_checks` is printed, and a run that verified nothing
# reports FAILED rather than green, on the same principle as
# `artifact_currency.jl`'s zero-artifact guard. Adding a figure to RESULTS.md
# without adding it here leaves it in the old regime; that is a real limit and it
# is better stated than hidden.
#
# TWO KINDS OF CHECK, and the second is the one that catches defect 3:
#
#   :literal  the computed figure, formatted the way the prose formats it, must
#             appear somewhere in RESULTS.md.
#   :pattern  EVERY match of a regex must carry the same computed value. This is
#             what makes "the same quantity quoted in two places" a checkable
#             property. A count-based check ("this string appears twice") would
#             rot the moment someone rewords a sentence; a pattern check does
#             not care how many times the quantity is mentioned, only that the
#             mentions agree with the artifact and with each other.
#
# Run after any regeneration, and before landing a prose edit that touches a
# number. Exit 0 = every checked figure agrees.
include(joinpath(@__DIR__, "artifact_currency.jl"))

import JSON
import Statistics: median

const RESULTS_MD = joinpath(REPO, "docs", "benchmark", "RESULTS.md")

# The one live driver pair. Kept in step with `backend_bands.jl`'s DRIVER_SHA
# deliberately: two scripts naming two different "live" drivers would let a
# superseded run answer for a current claim in one place and not the other.
const DRIVER_SHA = "d68d680"

"""
    live_json(path)

Load a checked-in artifact, refusing anything `artifact_currency.jl` considers
superseded. A figure in RESULTS.md describes the sampler as it is today; a
superseded run cannot confirm one, and silently reading it would make this
script agree with prose that both are wrong.
"""
function live_json(path)
    reason = superseded_reason(path)
    reason === nothing || error("`$path` is SUPERSEDED ($reason) — it cannot " *
                                "confirm a figure that claims to describe the " *
                                "sampler today")
    full = joinpath(REPO, path)
    isfile(full) || error("`$path` is missing — every figure read from it would " *
                          "go unchecked, and an unchecked figure is exactly what " *
                          "this script exists to make impossible")
    JSON.parsefile(full)
end

short_target(name) = name == "funnel" ? "funnel" :
    replace(replace(match(r"^[^-]+-(.+)$", name).captures[1], r"_model$" => ""),
            r"_(?:non)?centered" => "")

checks = NamedTuple[]
# `want` is a LIST of acceptable renderings, and passing on any one of them is
# deliberate. This script checks VALUES, not typography: RESULTS.md writes a
# ratio as a percentage or as a multiple depending on how large it is, and the
# switch is an editorial judgement with no clean threshold (`3.325` is written
# `+232%`, `8.166` is written `8.2×`). Encoding a guessed cutoff here would make
# the checker fail on correct prose, which costs more trust than the check buys
# -- the first version of this file did exactly that.
literal(label, ss...) = push!(checks, (kind = :literal, label = label, want = collect(ss)))
pattern(label, re, s) = push!(checks, (kind = :pattern, label = label, re = re, want = s))

"Both renderings of a ratio, so a check accepts whichever the prose chose."
both(r) = (string("+", round(Int, 100 * (r - 1)), "%"), string(round(r, digits = 1), "×"))

# ---------------------------------------------------------------- capture_boxing
#
# The A/B table plus the two sentences underneath it. `bare_ns_median` is the
# quantity that was quoted two ways, so it gets a :pattern check rather than a
# :literal one -- the point is not that `25863` appears, it is that NOTHING else
# is ever presented as this run's bare gradient.
let cb = live_json("docs/benchmark/results/capture_boxing.json")
    t(k) = median(cb["timings_ns"][k])
    bare = cb["bare_ns_median"]
    ns(x) = string(round(Int, x))

    pattern("capture_boxing bare gradient (every mention must agree)",
            r"bare gradient of ([0-9]+) ns", ns(bare))

    for (k, lbl) in ("boxed/fd" => "boxed ForwardDiff", "boxed/const" => "boxed Enzyme",
                     "unboxed/fd" => "unboxed ForwardDiff", "unboxed/const" => "unboxed Enzyme")
        literal("capture_boxing $lbl", ns(t(k)) * " ns")
    end

    x2(v, d = 2) = string(round(v, digits = d), "×")
    literal("capture_boxing boxed Enzyme ÷ ForwardDiff", x2(t("boxed/const") / t("boxed/fd")))
    literal("capture_boxing unboxed Enzyme ÷ ForwardDiff", x2(t("unboxed/const") / t("unboxed/fd")))
    literal("capture_boxing de-boxing speedup, ForwardDiff", x2(t("boxed/fd") / t("unboxed/fd")))
    literal("capture_boxing de-boxing speedup, Enzyme", x2(t("boxed/const") / t("unboxed/const")))

    # The wrapper-vs-bare pair. Both are ratios to `bare`, so they must be read
    # the same way; quoting one against `bare` and the other against the unboxed
    # timing is how a "16× tax" that is really 18.1× survives next to a correct
    # 11%.
    literal("capture_boxing shipped wrapper overhead vs bare",
            string(round(Int, 100 * (t("unboxed/const") / bare - 1))) * "% overhead")
    literal("capture_boxing boxed wrapper tax vs bare",
            string(round(Int, t("boxed/const") / bare)) * "× tax")
end

# ------------------------------------------------------------- gradient_overhead
#
# Per-target wrapper overhead, both backends. Either rendering is accepted; see
# the note on `literal` above for why the threshold is not guessed here.
#
# The paired-round tally is checked alongside them because it is the claim the
# per-target figures REST on. Each is a ratio of two medians, and this document
# has already carried one reversed verdict built from exactly that -- so the
# prose asserts the sign separately, from the rounds, and an assertion about 70
# numbers is worth strictly more than five summaries only while it still counts
# them correctly. Pairing is by index and that is not arbitrary: `gradient_overhead`
# times all three variants on the same `xs` within a round, rotating the order, so
# round r's live and bare are adjacent in time on identical inputs. Comparing
# across rounds instead would discard the pairing the harness deliberately builds
# and widen every spread for no reason.
paired_above = paired_total = 0
for be in ("enzyme", "forwarddiff")
    go = live_json("docs/benchmark/results/$be-$DRIVER_SHA/gradient_overhead.json")
    for row in go["overheads"]
        literal("$be overhead on $(short_target(row["target"]))", both(row["live_ratio"])...)
        ratios = row["live_ns_rounds"] ./ row["bare_ns_rounds"]
        global paired_above += count(>(1.0), ratios)
        global paired_total += length(ratios)
    end
end
literal("paired rounds putting the wrapper above bare",
        "$paired_above of the $paired_total paired rounds")

# -------------------------------------------------------------------- prep_cost
#
# The DI-preparation bullet. This artifact is the one whose generator was fixed
# for measuring each cell ONCE: two runs of identical code disagreed by a factor
# of thirty on `eight_schools`/`fd` (113% of the call, then 3629%), and the prose
# was quoting it to three significant figures. It now takes ROUNDS repeats with
# the block order rotated and keeps the raw rounds. The figures below were
# retyped from the repaired artifact and verified by hand once; checking them
# here is what stops that hand-verification from being a one-day property.
let pc = live_json("docs/benchmark/results/prep_cost.json")
    row(t, be) = only(r for r in pc["rows"] if r["target"] == t && r["backend"] == be)
    us(x) = string(round(x / 1000, digits = 1)) * " µs"

    big = ("radon_mn-radon_partially_pooled_centered" => "radon_partially_pooled",
           "radon_mn-radon_variable_intercept_centered" => "radon_variable_intercept",
           "seeds_data-seeds_centered_model" => "seeds")

    # "26.5 of 28.5 µs" — the prose drops the unit on the first figure, so the
    # check follows the prose rather than demanding a unit it does not use.
    for (t, name) in big
        r = row(t, "const")
        literal("prep_cost $name prep-of-call",
                string(round(r["prep_ns"] / 1000, digits = 1), " of ", us(r["unprepped_ns"])))
    end

    # The band across those three, as a range. Written the way a range is
    # written; both orderings would be wrong to accept, so only the sorted one is.
    shares = sort([round(Int, 100 * row(t, "const")["prep_share"]) for (t, _) in big])
    literal("prep_cost Enzyme prep share band",
            string(first(shares), "–", last(shares), "% of the whole unprepped Enzyme"))

    # The cross-backend comparison the bullet closes on.
    literal("prep_cost prepped-ForwardDiff vs unprepped-Enzyme",
            us(row("radon_mn-radon_partially_pooled_centered", "fd")["prepped_ns"]))
end

# ------------------------------------------------------------------------- run
function main_figures()
    isfile(RESULTS_MD) || (println("FAILED: $RESULTS_MD is missing."); return 1)
    text = read(RESULTS_MD, String)

    # Literals are matched against a whitespace-flattened copy as well as the
    # raw text, so a figure that happens to straddle a line wrap still matches.
    # Without this, re-wrapping a paragraph -- which changes nothing a reader
    # would call a number -- turns a passing check red, and a check that cries
    # wolf on correct prose gets deleted rather than fixed. Patterns keep their
    # own regexes, which can carry `\s+` where they need it.
    flat = replace(text, r"\s+" => " ")

    bad = String[]
    for c in checks
        if c.kind === :literal
            hit = findfirst(w -> occursin(w, text) || occursin(w, flat), c.want)
            ok = hit !== nothing
            shown = join("`" .* c.want .* "`", " or ")
            println(rpad(c.label, 52), ok ? "ok    " : "WRONG ", "expected ", shown,
                    ok ? " (found `$(c.want[hit])`)" : "")
            ok || push!(bad, "$(c.label): RESULTS.md contains neither of $shown")
        else
            ms = collect(eachmatch(c.re, text))
            if isempty(ms)
                println(rpad(c.label, 52), "WRONG ", "no mention found matching ", c.re)
                push!(bad, "$(c.label): no text matches $(c.re) — the figure it " *
                           "guards was reworded or removed, so it is now unchecked")
            else
                got = unique(m.captures[1] for m in ms)
                ok = got == [c.want]
                println(rpad(c.label, 52), ok ? "ok    " : "WRONG ",
                        "expected `", c.want, "` in all ", length(ms), " mention(s), found ",
                        join("`" .* got .* "`", ", "))
                ok || push!(bad, "$(c.label): mentions say $(join(got, ", ")); artifact says $(c.want)")
            end
        end
    end

    println()
    if isempty(checks)
        println("FAILED: zero figures were checked.")
        println("A figure check that verifies nothing must not report green.")
        return 1
    end
    if isempty(bad)
        println("ALL AGREE: $(length(checks)) quoted figure(s) in RESULTS.md match the live artifacts.")
        println("Figures this script does not know about remain unchecked — see the header.")
        return 0
    end
    println("DISAGREES: ", length(bad), " of ", length(checks), " checked figure(s)")
    for b in bad
        println("  - ", b)
    end
    println()
    println("Either the prose is stale (fix the prose) or the artifact was regenerated")
    println("and the claim genuinely moved (re-read the surrounding argument before")
    println("retyping the number — a figure that moved may have moved a conclusion).")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_figures())
end
