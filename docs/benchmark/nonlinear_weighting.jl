# Nonlinear online trajectory weighting: does the metric policy matter?
#
# `WarmupHMC:nonlinear-online` measured two independent policy axes on the
# adaptive nonlinear path -- which leaves contribute evidence
# (`all_good_leaves` vs `nuts_weighted`) and how that evidence is scaled
# (`unit` vs `stepsize`).
#
# THE ROWS ARE THE ARTIFACT. `results/nonlinear_weighting/rows.json` holds one
# object per run plus the configuration that produced them, and nothing else;
# every number here is derived from those rows by the caller. That is
# deliberate and it is the whole design: a JSON carrying both rows and
# precomputed summaries has two representations of one fact, and nothing keeps
# them in step. A reader who finds a stored `1.051` next to rows that now say
# `1.048` cannot tell which is stale. Deriving costs milliseconds.
#
# THE CONCLUSION IS DERIVED THE SAME WAY -- `nw_conclusion(rows)` assembles it
# from the numbers, so the docs build and the web app LOAD the sentence rather
# than restate it. A conclusion re-typed downstream is the same duplication as
# a cached aggregate and goes stale the same silent way.
#
# DEFINITIONS ONLY -- this file is `load_harness`ed into the docs build (see
# `docs/tables.jl`), so nothing here may run at include time: no `ARGS`, no
# reads, no writes. Dependencies are limited to what `docs/Project.toml` has:
# JSON, Markdown, Printf, Statistics. The two entry points that DO act live
# next door:
#
#   * `nonlinear_weighting_report.jl` -- prints the tables, writes the summary
#   * `nonlinear_weighting_run.jl`    -- re-measures the rows
#
# Names are prefixed `nw_` on purpose. `load_harness` includes into the CALLING
# module, so every harness loaded by one page shares a namespace; a bare
# `conclusion` would silently collide with the next harness that defines one.
import JSON
using Markdown, Printf, Statistics

"Default location of the checked-in rows for this study."
const NW_ROWS_PATH = joinpath(@__DIR__, "results", "nonlinear_weighting", "rows.json")

"Fields compared to decide whether two policy arms produced the same run."
const NW_IDENTITY_FIELDS = ("ess_min", "ess_median", "grad_evals",
                            "n_divergent", "final_c")

"""
    nw_load(path = NW_ROWS_PATH) -> Dict

Parse the rows artifact. Returns the whole document, with `"rows"` and
`"config"`. Every other function here takes the `rows` vector.
"""
nw_load(path::AbstractString = NW_ROWS_PATH) = JSON.parsefile(path)

nw_targets(rows) = sort(unique(r["target"] for r in rows))
nw_leaf_policies(rows) = sort(unique(r["leaf_policy"] for r in rows))
nw_metric_policies(rows) = sort(unique(r["metric_policy"] for r in rows))

nw_select(rows, t, l, m) = [r for r in rows if r["target"] == t &&
                            r["leaf_policy"] == l && r["metric_policy"] == m]

"""
    nw_per_kgrad(row, key) -> Float64

Effective sample size per 1000 gradient evaluations. The gradient count is the
denominator rather than wall-clock because these runs share a host with other
work; a contended second is not comparable across runs, a gradient is.
"""
nw_per_kgrad(row, key) = 1000 * row[key] / row["grad_evals"]

"""
    nw_aggregate(rows) -> Vector{Dict}

Median efficiency per (target, leaf policy, metric policy) cell.
"""
function nw_aggregate(rows)
    out = Dict{String,Any}[]
    for t in nw_targets(rows), l in nw_leaf_policies(rows), m in nw_metric_policies(rows)
        s = nw_select(rows, t, l, m)
        isempty(s) && continue
        push!(out, Dict{String,Any}(
            "target" => t, "leaf_policy" => l, "metric_policy" => m,
            "n_runs" => length(s),
            "median_ess_min_per_kgrad" => median(nw_per_kgrad(r, "ess_min") for r in s),
            "median_ess_median_per_kgrad" => median(nw_per_kgrad(r, "ess_median") for r in s),
            "n_divergent" => sum(r["n_divergent"] for r in s)))
    end
    out
end

"""
    nw_paired(rows) -> Vector{Dict}

`stepsize` against `unit`, **paired by seed** within a leaf policy. Paired
because the seeds are shared by construction and the unpaired spread across
seeds is far larger than the effect being looked for -- an unpaired comparison
of these cells mostly measures which seeds landed in which arm.

`median_ratio_min > 1` favours `stepsize`.

The **five-number summary is reported alongside the median, not instead of it**,
because on this data the median alone flattens the result into a non-result: the
all-good min-ESS ratios run from 0.31 to 1.52 around a median of 1.05, and only
a handful of the 20 pairs land within +/-5% of parity. "Median 1.05" invites the
reading that the two policies are nearly the same on every seed; the spread says
they are very different on most seeds and merely disagree about which way. Those
are opposite findings, and only one of them is true.
"""
function nw_paired(rows)
    out = Dict{String,Any}[]
    for t in nw_targets(rows), l in nw_leaf_policies(rows)
        u = Dict(r["seed"] => r for r in nw_select(rows, t, l, "unit"))
        s = Dict(r["seed"] => r for r in nw_select(rows, t, l, "stepsize"))
        seeds = sort(collect(intersect(keys(u), keys(s))))
        isempty(seeds) && continue
        rmin = [nw_per_kgrad(s[k], "ess_min") / nw_per_kgrad(u[k], "ess_min") for k in seeds]
        rmed = [nw_per_kgrad(s[k], "ess_median") / nw_per_kgrad(u[k], "ess_median") for k in seeds]
        push!(out, Dict{String,Any}(
            "target" => t, "leaf_policy" => l, "n_pairs" => length(seeds),
            "stepsize_wins_min" => count(>(1), rmin),
            "median_ratio_min" => median(rmin),
            "median_ratio_median" => median(rmed),
            "five_number_min" => nw_five_number(rmin),
            "five_number_median" => nw_five_number(rmed),
            # How many pairs are actually a tie, at a stated tolerance. A win
            # count alone cannot distinguish "12 clear wins" from "12 pairs that
            # differ in the sixth decimal".
            "within_5pct_min" => count(r -> abs(r - 1) <= 0.05, rmin),
            "within_5pct_median" => count(r -> abs(r - 1) <= 0.05, rmed)))
    end
    out
end

"""
    nw_five_number(v) -> Vector{Float64}

`[min, Q1, median, Q3, max]`. Quantiles are the default (linear interpolation,
Julia's `quantile`), so they match what a reader recomputes with `Statistics`.
"""
nw_five_number(v) = [minimum(v), quantile(v, 0.25), median(v),
                     quantile(v, 0.75), maximum(v)]

"""
    nw_discrimination(rows) -> Vector{Dict}

Whether a target can distinguish the policy arms at all.

This is the check that matters most and the one a summary table cannot carry.
If the four arms produce the SAME RUN on a target, that target reports a tie no
matter which policy is better -- and a tie read as evidence of equivalence is
exactly backwards. It is derivable only from the rows.
"""
function nw_discrimination(rows)
    out = Dict{String,Any}[]
    for t in nw_targets(rows)
        seeds = sort(unique(r["seed"] for r in rows if r["target"] == t))
        ident = 0
        cdiff = 0
        for k in seeds
            arms = [r for r in rows if r["target"] == t && r["seed"] == k]
            length(arms) < 2 && continue
            ref = first(arms)
            all(all(a[f] == ref[f] for f in NW_IDENTITY_FIELDS) for a in arms) && (ident += 1)
            length(unique(a["final_c"] for a in arms)) > 1 && (cdiff += 1)
        end
        push!(out, Dict{String,Any}(
            "target" => t, "n_seeds" => length(seeds),
            "arms_identical_seeds" => ident, "final_c_differs_seeds" => cdiff,
            "discriminates" => ident < length(seeds),
            "verdict" => ident == length(seeds) ?
                    "cannot discriminate — every arm identical" :
                cdiff == 0 ? "no centering ever differed" :
                    "discriminates on $(cdiff)/$(length(seeds)) seeds"))
    end
    out
end

"""
    nw_conclusion(rows) -> String

The study's conclusion, assembled from `nw_paired` and `nw_discrimination`
rather than asserted, so that a consumer loads this sentence instead of
re-typing it.

The rule applied is the one `nw_discrimination` argues for: a target that
cannot discriminate is not evidence, so it is **excluded** here rather than
counted as a tie. A policy is a universal winner only if it wins the paired
minimum-ESS comparison in EVERY discriminating (target, leaf-policy) cell --
one cell pointing the other way is enough to deny it.

Exact ties are counted and named rather than dropped, so the cells always add
up. They are not a rounding artifact: a target whose arms are bit-identical on
most seeds has a paired median pinned to exactly 1 **by construction**, since
more than half the ratios being summarised are exactly 1. Such a cell survives
the discrimination filter -- some seeds really did differ -- while its median
still carries no signal, and `nw_conclusion` says so instead of reading the tie
as evidence of equivalence.
"""
function nw_conclusion(rows)
    disc = nw_discrimination(rows)
    byname = Dict(d["target"] => d for d in disc)
    good = Set(d["target"] for d in disc if d["discriminates"])
    excluded = [d["target"] for d in disc if !d["discriminates"]]
    usable = [p for p in nw_paired(rows) if p["target"] in good]
    nstep = count(p -> p["median_ratio_min"] > 1, usable)
    nunit = count(p -> p["median_ratio_min"] < 1, usable)
    ties  = [p for p in usable if p["median_ratio_min"] == 1]

    s = if isempty(usable)
        "No target in this study discriminates between the policy arms, so it " *
        "supports no conclusion about which weighting is better."
    elseif nstep == length(usable)
        "`stepsize` won paired minimum-coordinate efficiency in all " *
        "$(length(usable)) discriminating (target, leaf-policy) cells."
    elseif nunit == length(usable)
        "`unit` won paired minimum-coordinate efficiency in all " *
        "$(length(usable)) discriminating (target, leaf-policy) cells."
    else
        "No universal winner: across the $(length(usable)) discriminating " *
        "(target, leaf-policy) cells, `stepsize` won paired minimum-coordinate " *
        "efficiency in $(nstep), `unit` in $(nunit), and $(length(ties)) were " *
        "exact ties. Keep `unit` as the conservative default and retain " *
        "`stepsize` as an experiment."
    end

    # "No universal winner" read off medians alone is the flattening this data
    # punishes: a cell can sit at parity in the median while almost every
    # individual pair is far from parity. Report the widest spread and how few
    # pairs are actually ties, so the sentence cannot be mistaken for "the two
    # policies behave alike".
    if !isempty(usable)
        widest = argmax(p -> p["five_number_min"][5] - p["five_number_min"][1], usable)
        fn = widest["five_number_min"]
        nties = sum(p["within_5pct_min"] for p in usable)
        npairs = sum(p["n_pairs"] for p in usable)
        s *= " That parity is an average, not a pattern: per-seed ratios on " *
             "`$(widest["target"])` / `$(widest["leaf_policy"])` span " *
             "$(round(fn[1]; digits = 2))–$(round(fn[5]; digits = 2)), and only " *
             "$(nties) of $(npairs) paired seeds land within ±5% of parity. The " *
             "policies differ substantially on most seeds and merely disagree " *
             "about which direction."
    end

    # A tie whose target is bit-identical on most seeds is pinned to 1 by those
    # seeds, not by the policies performing alike. Say which, and why.
    pinned = unique(p["target"] for p in ties
                    if 2 * byname[p["target"]]["arms_identical_seeds"] >=
                       byname[p["target"]]["n_seeds"])
    if !isempty(pinned)
        s *= " The ties come from " * join(("`" * t * "`" for t in pinned), ", ") *
             ", where the arms are bit-identical on at least half the seeds — " *
             "so the paired median is pinned to exactly 1 by those seeds and " *
             "measures nothing about the policies."
    end

    isempty(excluded) && return s
    s * " Excluded outright as non-discriminating: " *
        join(("`" * t * "`" for t in excluded), ", ") *
        " — every policy arm produced a bit-identical run there, so the ties " *
        "they report are an artifact of the target rather than evidence that " *
        "the policies are equivalent."
end

# --- tables -----------------------------------------------------------------
# Built here rather than with `docs/tables.jl`'s `md_table` so that the same
# functions render when this file is run outside the docs environment.

function _nw_table(headers, cells)
    io = IOBuffer()
    println(io, "| ", join(headers, " | "), " |")
    println(io, "|", join(fill("---", length(headers)), "|"), "|")
    for r in cells
        println(io, "| ", join(string.(r), " | "), " |")
    end
    Markdown.parse(String(take!(io)))
end

_nw_num(x; d = 2) = @sprintf("%.*f", d, x)

"Median ESS per 1000 gradients for every (target, leaf, metric) cell."
nw_aggregate_table(rows) = _nw_table(
    ["target", "leaves", "metric", "min ESS/kgrad", "median ESS/kgrad", "divergences"],
    [[ "`$(a["target"])`", "`$(a["leaf_policy"])`", "`$(a["metric_policy"])`",
       _nw_num(a["median_ess_min_per_kgrad"]),
       _nw_num(a["median_ess_median_per_kgrad"]), a["n_divergent"]]
     for a in nw_aggregate(rows)])

_nw_five(v) = "[" * join((_nw_num(x; d = 3) for x in v), ", ") * "]"

"""
Paired stepsize-vs-unit ratios. Above 1 favours `stepsize`.

Carries the five-number summary and the tie count, not just the median — see
`nw_paired` for why the median on its own reverses the finding.
"""
nw_paired_table(rows) = _nw_table(
    ["target", "leaves", "pairs", "stepsize wins (min ESS)",
     "min ESS ratio [min, Q1, med, Q3, max]", "within ±5%"],
    [[ "`$(p["target"])`", "`$(p["leaf_policy"])`", p["n_pairs"],
       "$(p["stepsize_wins_min"]) / $(p["n_pairs"])",
       _nw_five(p["five_number_min"]),
       "$(p["within_5pct_min"]) / $(p["n_pairs"])"]
     for p in nw_paired(rows)])

"Same, for median-coordinate rather than minimum-coordinate efficiency."
nw_paired_median_table(rows) = _nw_table(
    ["target", "leaves", "pairs",
     "median ESS ratio [min, Q1, med, Q3, max]", "within ±5%"],
    [[ "`$(p["target"])`", "`$(p["leaf_policy"])`", p["n_pairs"],
       _nw_five(p["five_number_median"]),
       "$(p["within_5pct_median"]) / $(p["n_pairs"])"]
     for p in nw_paired(rows)])

"Whether each target distinguishes the policy arms at all."
nw_discrimination_table(rows) = _nw_table(
    ["target", "seeds", "all arms identical", "centering differs", "verdict"],
    [[ "`$(d["target"])`", d["n_seeds"],
       "$(d["arms_identical_seeds"]) / $(d["n_seeds"])",
       "$(d["final_c_differs_seeds"]) / $(d["n_seeds"])", d["verdict"]]
     for d in nw_discrimination(rows)])

"""
    nw_summary(rows, config) -> Dict

Everything the report writes and the app renders, in one derived object. Held
in memory, never stored beside the rows.
"""
function nw_summary(rows, config)
    disc = nw_discrimination(rows)
    paired = nw_paired(rows)
    good = Set(d["target"] for d in disc if d["discriminates"])
    usable = [p for p in paired if p["target"] in good]
    Dict("note" => "Derived from results/nonlinear_weighting/rows.json at read " *
                   "time. Do not hand-edit: regenerate with " *
                   "nonlinear_weighting_report.jl.",
         "config" => config, "n_rows" => length(rows),
         "aggregate" => nw_aggregate(rows), "paired" => paired,
         "discrimination" => disc, "conclusion" => nw_conclusion(rows),
         "conclusion_facts" => Dict(
             "discriminating_targets" => sort(collect(good)),
             "excluded_targets" => [d["target"] for d in disc if !d["discriminates"]],
             "n_usable_cells" => length(usable),
             "stepsize_wins_cells" => count(p -> p["median_ratio_min"] > 1, usable),
             "unit_wins_cells" => count(p -> p["median_ratio_min"] < 1, usable)))
end
