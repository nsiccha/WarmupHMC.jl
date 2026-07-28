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
# There is a SECOND shape, and it is NOT what defect 3 was: figures that are
# individually correct and wrong only in RELATION to one another. The `16x tax`
# beside the `11% overhead` is the clean example -- one artifact, one moment, two
# denominators, neither figure stale, only the relation false.
#
# THE TEST THAT SEPARATES THEM: WERE BOTH FIGURES CORRECT AT THE SAME TIME?
#
#   `16x` / `11%`  -- yes. Same artifact, same instant. Shape 2.
#   `27431`/`25863` -- no. `176a061` measured `bare_ns_median = 27430.907`, so
#                      `27431 ns` was right when typed; `2f39099` re-measured to
#                      `25862.969` and updated ONE of the two mentions (its diff
#                      shows line 353 change and line 338 left alone). At the
#                      moment of the defect `27431` described a run that no
#                      longer existed. That is shape 1 verbatim.
#
# The distinction decides which check you need, which is why it is worth the
# paragraph:
#
#   shape 1 -> `:pattern`. Deriving DOES close it, provided the check requires
#              every mention to AGREE rather than one to be PRESENT: a
#              containment check passes happily on a document carrying both
#              `27431` and `25863`.
#   shape 2 -> encode the denominator the sentence intends. Deriving each figure
#              independently CANNOT close it, because each one already checks out.
#
# That second row is where the general lesson lives: A COMPUTED NUMBER VOUCHES
# FOR ITSELF AND FOR NOTHING AROUND IT. The operator, the units, the direction of
# the comparison and the label beside it are all hand-written, and the correct
# value is precisely what stops a reviewer re-reading them -- so deriving the
# figure moves the risk rather than removing it. (Paid for twice: the `16x`, and
# a line in this file's own history whose bound was computed and whose `>` was
# typed.) It does not extend to shape 1, and an earlier revision of this comment
# claimed it did -- crediting `:pattern` to the shape it does not fix.
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
# PRECISION: COMPARE AT THE RENDERING, NOT AT THE FLOAT. Every check here
# formats the computed value the way the prose formats it and compares text. That
# is deliberate in both directions, and the tempting "compare the numbers
# properly" refactor is a regression.
#
# Comparing finer than the prose reddens pages that are CORRECT AS WRITTEN, and a
# checker that does that gets deleted rather than fixed. Shipped here once: an
# earlier version encoded the percent-vs-`x` switch as a threshold at 3x, so
# `eight_schools` under ForwardDiff -- 3.325, correctly written `+232%` -- went
# red on a healthy page. Hence `both(r)`, which accepts either rendering.
#
# The property this buys is worth stating, because it looks like a weakness. A
# reviewer asked whether the `18x tax` check (which encodes a DENOMINATOR -- see
# `capture_boxing boxed wrapper tax vs bare` below) quietly loses power as the
# de-boxing work shrinks the wrapper overhead toward zero, since the two rival
# denominators converge there. It does not, and the reason is general: below the
# resolution of the rendering, the two denominators produce the SAME TEXT, so
# whichever the author intended there is no wrong number on the page to catch.
# The defect and the detector are defined on the same quantity -- what is
# written -- so they cannot come apart.
#
# Measured on the live artifact so the band is not a guess. `boxed/const` renders
# `18x` against `bare`; against `unboxed/const` it also renders `18x` for
# `unboxed` in `(25330.0, 26777.5]` ns -- overhead in `(-2.06%, +3.54%]`. Today
# `unboxed = 28823.7` (11% overhead) renders `16x`, well outside, which is why
# re-injecting the original `16x tax` defect goes red. Note the band is
# TWO-SIDED and the collapse is at LOW overhead: below `25330.0` the wrapper
# would be measuring faster than the bare gradient and the two separate again at
# `19x` vs `18x`. That lower edge is unreachable in the current artifact -- every
# `unboxed/const` round spans `27667-42965` ns against `bare = 25863`, none
# faster than bare -- but write the interval, not the half-line: a one-sided
# `overhead below 3.5%` is the form that later reads as wrong.
#
# Run after any regeneration, and before landing a prose edit that touches a
# number. Exit 0 = every checked figure agrees.
include(joinpath(@__DIR__, "artifact_currency.jl"))

import JSON
import Statistics: median

const RESULTS_MD = joinpath(REPO, "docs", "benchmark", "RESULTS.md")
const RESULTS_DIR = joinpath(REPO, RESULTS)

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

# A THIRD KIND, whose falsifier is not a number moving but A FILE APPEARING.
#
# `absence(label, claim, token)` guards a sentence that says the evidence does
# NOT contain something -- "was not measured", "no numbers yet". Every check
# above recomputes a figure from an artifact and so is blind to this class by
# construction: when such a sentence goes false, no figure anywhere changes.
# What changes is that a results file starts existing.
#
# This is the exact dual of the blindness recorded in `artifact_currency.jl`'s
# neighbourhood -- an enumerator cannot detect a file's ABSENCE, because "what
# should exist" is not information it has. But it detects PRESENCE perfectly,
# and a claim of absence is falsified by a presence. So the direction that is
# hopeless for one is the cheap one here: scan `results/` for the token and go
# red if the prose still denies it.
#
# Measured on this repo before writing it: a phrase-level lint for the tells
# ("not in yet", "remains untested") returns ~80% false positives on the real
# corpus -- it matches "not inlined", "not inferred from timings", "does not
# invalidate" -- so it would redden correct prose, which is how a checker gets
# deleted. Hence one explicit binding per claim, and only two claims repo-wide.
# That is a census and it will need adding to; it is worth it at this size, and
# an unmatched claim is reported rather than skipped so a reword cannot silently
# unbind it.
#
# REACH FOR THIS ONLY WHEN THE PROSE HAS NO EXECUTION SEAM. `absence()` is an
# EXTERNAL binding: the claim lives in one file and its predicate in another, so
# they can drift, and only this script notices. Where the prose is executed at
# build time there is a strictly stronger move -- put the claim in a BRANCH that
# reads the predicate, so the claim and the predicate are the same expression:
#
#     nl === false ? "Nonlinear adaptation is off in every run..." :
#     nl === true  ? "**Nonlinear adaptation was enabled**..." :
#                    "**Whether nonlinear adaptation was on is not recorded**..."
#
# That is `docs/src/linear-restart.md`, and it CANNOT go stale -- it can only
# stop being displayed. (`nonlinear_adapt` is recorded as `false`, so the third
# branch does not render at all today.) No external check is owed, or possible,
# or needed.
#
# The reason it works is the sharper form of the whole class: this defect needs
# prose that has COMMITTED to a state. A conditional never commits -- it reports
# whichever state holds at build time. So the rule is: if the file is built, put
# the claim in a branch; if it is not built, and `RESULTS.md` is not, bind it
# here. Do not reach for `absence()` on a page that could have used a branch.
function absence(label, claim, token)
    hits = String[]
    for (root, _, files) in walkdir(RESULTS_DIR), f in files
        endswith(f, ".json") || continue
        occursin(token, read(joinpath(root, f), String)) &&
            push!(hits, relpath(joinpath(root, f), REPO))
    end
    push!(checks, (kind = :absence, label = label, want = claim, evidence = hits))
end

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
# them correctly.
#
# PAIRING IS PER TIMING LOOP, NOT PER HARNESS. Index-pairing is correct here and
# it is not arbitrary: `gradient_overhead` binds one `xs` per round and times all
# three variants on it back-to-back with the order rotated, so round r's live and
# bare are adjacent in time on identical inputs. Comparing across rounds instead
# would discard that and widen every spread for nothing.
#
# The tempting generalisation from this -- "that harness pairs, this one does
# not" -- is WRONG, and it is worth stating because I drew it and told a peer so.
# `typical_positions.jl` pairs by the same construction (it binds `xs` OUTSIDE
# the round loop, then `circshift(BACKENDS, r)` within it), so index-pairing is
# right there too. What actually decides it is the loop: figures timed inside one
# loop over shared inputs pair; figures from two SEPARATE loops do not, however
# similar they look. In `typical_positions` that boundary falls between the
# `typical` and `randn` position sets -- different loop iterations, different
# inputs, so round i of one is not co-timed with round i of the other, and those
# two columns must be compared by overlapping ranges rather than by dividing
# across them.
#
# None of this is recoverable from the JSON: the artifacts record round ORDER but
# not co-timing. Read the harness, not the file.
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

# ------------------------------------------------------------- absence claims
#
# The limitations section states what the evidence does NOT cover. Those
# sentences are the ones a NEW measurement falsifies, and nothing else in this
# file can see that happen. Live instance of the class, on a neighbouring page:
# `reparametrization.md` said "Those numbers are not in yet", true when written
# at `df35a1c` (07:57) and false from `b95f872` (14:28), when the fixed-c
# comparison artifact landed. Six hours, a green build throughout, and it was
# found by a reader following a link -- not by any check.
absence("limitation: cooperative_warmup_mcmc unmeasured",
        "`cooperative_warmup_mcmc` was not measured", "cooperative_warmup_mcmc")

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
        elseif c.kind === :absence
            # Two ways to be wrong, and they need opposite fixes. The claim can
            # have gone FALSE (a results file now carries the thing the prose
            # denies), or it can have been reworded out of existence, which
            # leaves it unguarded exactly as a reworded :pattern does.
            stated = occursin(c.want, text) || occursin(c.want, flat)
            if !stated
                println(rpad(c.label, 52), "WRONG ", "claim not found: `", c.want, "`")
                push!(bad, "$(c.label): RESULTS.md no longer contains `$(c.want)` — " *
                           "the absence claim was reworded or removed, so it is now unchecked")
            elseif !isempty(c.evidence)
                println(rpad(c.label, 52), "WRONG ", "claim is now FALSE — evidence in ",
                        join(c.evidence, ", "))
                push!(bad, "$(c.label): RESULTS.md still says `$(c.want)`, but " *
                           join(c.evidence, ", ") * " now contains it")
            else
                println(rpad(c.label, 52), "ok    ", "still true: `", c.want, "`")
            end
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
