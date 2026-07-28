# Derived summaries for `sampler_comparison.json`.
#
# Same contract as `linear_restart_summary.jl`: this file only DEFINES things,
# and depends on nothing outside `docs/Project.toml`'s JSON / Markdown / Printf /
# Statistics — it is included by Documenter's `@eval` blocks, so an include-time
# side effect would run during the docs build and a heavier dependency would put
# the measurement stack on the deployment's critical path.
#
# The rows are the data. Every figure on the page is a function of them,
# recomputed at build time, so a regenerated JSON cannot leave a stale number in
# prose. That is not a hypothetical in this repository: a hand-copied band in
# `RESULTS.md` once landed four minutes before the files it was computed from
# were regenerated, and ended up excluding values that were in the files.
#
# THE VERDICT IS COMPUTED, AND THAT IS THE POINT
#
# `comparison_verdict` counts wins from the rows rather than stating one. If a
# future measurement inverts the result, the page inverts with it. A sentence
# saying "WarmupHMC is faster" would not — which is the entire reason the README
# claim needed this file in the first place.

import Statistics

const COMPARISON_ARMS = ["warmuphmc", "dynamichmc", "advancedhmc"]

const COMPARISON_ARM_LABELS = Dict(
    "warmuphmc"   => "WarmupHMC",
    "dynamichmc"  => "DynamicHMC",
    "advancedhmc" => "AdvancedHMC",
)

"""Finite values of `key` across `rows` that recorded a successful run.

A failed arm contributes nothing rather than a zero: a sampler that threw did
not achieve an ESS of 0, it produced no measurement, and averaging in a zero
would be an invented number. The failure is reported separately by
`comparison_failures`, so it is not lost either."""
function _cmp_values(rows, key)
    Float64[
        r[key] for r in rows
        if get(r, "ok", false) && get(r, key, nothing) isa Real && isfinite(r[key])
    ]
end

_cmp_median(rows, key) = (v = _cmp_values(rows, key); isempty(v) ? NaN : Statistics.median(v))

"""Target names in the order they appear in the file, so the table follows the
run rather than an alphabetical order nobody chose."""
function comparison_targets(data::AbstractDict)
    seen = String[]
    for r in data["runs"]
        t = r["target"]
        t in seen || push!(seen, t)
    end
    seen
end

_cmp_rows(data, target, arm) =
    [r for r in data["runs"] if r["target"] == target && r["arm"] == arm]

"""
    comparison_summary(data) -> Vector of NamedTuple

One median row per (target, arm): min ESS, the two rates, and total divergences
across seeds. `n_ok` is carried so a reader can see a row backed by one seed for
what it is.
"""
function comparison_summary(data::AbstractDict)
    out = NamedTuple[]
    for t in comparison_targets(data), arm in COMPARISON_ARMS
        rows = _cmp_rows(data, t, arm)
        isempty(rows) && continue
        ok = [r for r in rows if get(r, "ok", false)]
        push!(out, (
            target = t,
            arm = arm,
            n_ok = length(ok),
            n_run = length(rows),
            ess_min = _cmp_median(rows, "ess_min"),
            ess_per_grad = _cmp_median(rows, "ess_min_per_grad"),
            ess_per_s = _cmp_median(rows, "ess_min_per_s"),
            grad_evals = _cmp_median(rows, "grad_evals"),
            n_divergent = sum(Int(get(r, "n_divergent", 0)) for r in ok; init = 0),
        ))
    end
    out
end

"""
    comparison_ratios(data; key) -> Vector of NamedTuple

WarmupHMC ÷ each reference sampler, per target, on `key` (`"ess_min_per_grad"`
or `"ess_min_per_s"`). Above 1 means WarmupHMC produced more effective sample
per unit of the resource.

Ratios of MEDIANS, not medians of ratios: the seeds are not paired across
samplers (each sampler consumes its own random stream, and AdvancedHMC draws its
own starting point), so a per-seed ratio would pair numbers that share only an
index. Stated because the two are different estimators and the choice is not
visible in the output.
"""
function comparison_ratios(data::AbstractDict; key::AbstractString = "ess_min_per_grad")
    out = NamedTuple[]
    for t in comparison_targets(data)
        w = _cmp_median(_cmp_rows(data, t, "warmuphmc"), key)
        for arm in COMPARISON_ARMS
            arm == "warmuphmc" && continue
            r = _cmp_median(_cmp_rows(data, t, arm), key)
            push!(out, (target = t, versus = arm,
                        ratio = (isfinite(w) && isfinite(r) && r > 0) ? w / r : NaN))
        end
    end
    out
end

"""Arms that failed on at least one seed, as `(target, arm, n_failed, error)`.

Surfaced rather than filtered: an arm that cannot run on a target is a result
about that arm, and a table that silently dropped it would be reporting a
comparison over a target set chosen by which runs happened to succeed."""
function comparison_failures(data::AbstractDict)
    out = NamedTuple[]
    for t in comparison_targets(data), arm in COMPARISON_ARMS
        bad = [r for r in _cmp_rows(data, t, arm) if !get(r, "ok", false)]
        isempty(bad) && continue
        push!(out, (target = t, arm = arm, n_failed = length(bad),
                    error = first(bad)["error"]))
    end
    out
end

"""
    comparison_verdict(data; key) -> NamedTuple

Wins, losses and ties for WarmupHMC on `key`, counted over targets where both
arms produced a finite figure.

`tie_band` is a relative tolerance, not a threshold tuned to anything measured:
two samplers within 10% of each other on a handful of seeds have not been shown
to differ, and calling that a win in either direction would be reading noise.
Targets where either side has no finite figure are `n_incomparable` — counted,
never silently dropped, because a shrinking denominator is how a weak result
turns into a strong-looking one.
"""
function comparison_verdict(data::AbstractDict;
                            key::AbstractString = "ess_min_per_grad",
                            tie_band::Float64 = 0.10)
    wins = losses = ties = incomparable = 0
    for r in comparison_ratios(data; key)
        if !isfinite(r.ratio)
            incomparable += 1
        elseif r.ratio > 1 + tie_band
            wins += 1
        elseif r.ratio < 1 - tie_band
            losses += 1
        else
            ties += 1
        end
    end
    (; key, tie_band, n_wins = wins, n_losses = losses, n_ties = ties,
       n_incomparable = incomparable,
       n_comparisons = wins + losses + ties)
end

"""One sentence stating what the rows show, built from the counts.

Deliberately says "on this target set" and gives the denominator. A comparison
over five targets can refute a universal claim and cannot establish one, and the
sentence should not read as though it could."""
function comparison_verdict_sentence(data::AbstractDict; kwargs...)
    v = comparison_verdict(data; kwargs...)
    what = v.key == "ess_min_per_grad" ? "effective sample per gradient evaluation" :
           v.key == "ess_min_per_s"    ? "effective sample per wall-clock second" : v.key
    v.n_comparisons == 0 && return "No comparison on $what could be made from this run."
    string("On ", what, ", WarmupHMC leads in ", v.n_wins, " of ",
           v.n_comparisons, " head-to-head comparisons on this target set, ",
           "trails in ", v.n_losses, ", and is within ",
           round(Int, 100 * v.tie_band), "% in ", v.n_ties,
           v.n_incomparable == 0 ? "." :
           string("; ", v.n_incomparable, " comparison(s) had no finite figure on one side."))
end
