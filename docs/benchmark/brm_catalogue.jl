# Definitions-only derivation layer for `run_brm_catalogue_benchmark.jl`.
# Loaded by Documenter through `docs/tables.jl`; do not add driver code here.
#
# Nothing here may import BayesianRegressionModels, StanBlocks or BridgeStan.
# This file runs inside the lightweight docs environment, which deliberately
# does not have them — the page renders from checked-in JSON, and the runner is
# the only thing that needs the heavy stack.

import Statistics

brmc_rows(d) = d["rows"]
brmc_models(d) = d["models"]

brmc_med(v) = isempty(v) ? NaN : Statistics.median(v)

brmc_spec_keys(d) = [m["spec"] for m in brmc_models(d)]

brmc_arm_order() = ["noncentered", "centered", "adaptive_centering"]

brmc_arm_label(a) = get(Dict(
    "noncentered" => "non-centered (BRM default)",
    "centered" => "static centered",
    "adaptive_centering" => "adaptive centering",
), a, a)

brmc_select(d, spec, arm, adapt) =
    [r for r in brmc_rows(d)
     if r["spec"] == spec && r["arm"] == arm && r["nonlinear_adapt"] == adapt && r["ok"]]

"""
One summary row per (spec, arm, nonlinear_adapt): medians over seeds.

`ess_min_per_grad` is the headline. It is already per-row in the artifact — this
takes the median of the per-seed ratios, not a ratio of medians, so a seed that
was slow for both reasons cannot cancel itself out.
"""
function brmc_summary(d)
    out = []
    for m in brmc_models(d), arm in brmc_arm_order(), adapt in (false, true)
        rs = brmc_select(d, m["spec"], arm, adapt)
        isempty(rs) && continue
        push!(out, (
            spec = m["spec"], arm = arm, adapt = adapt, n = length(rs),
            grad = brmc_med([r["grad_evals"] for r in rs]),
            ess = brmc_med([r["ess_min_shared_constrained"] for r in rs
                            if r["ess_min_shared_constrained"] !== nothing]),
            per_grad = brmc_med([r["ess_min_per_grad"] for r in rs
                                 if r["ess_min_per_grad"] !== nothing]),
            per_s = brmc_med([r["ess_min_shared_constrained"] / r["wall_s"] for r in rs
                              if r["ess_min_shared_constrained"] !== nothing &&
                                 r["wall_s"] !== nothing && r["wall_s"] > 0]),
            wall = brmc_med([r["wall_s"] for r in rs if r["wall_s"] !== nothing]),
            ndiv = sum(r["n_divergent"] for r in rs),
            nconst = maximum(r["n_constant"] for r in rs; init = 0),
        ))
    end
    out
end

"""Paired adaptive-on/off ratios for both gradient and wall-time efficiency."""
function brmc_paired_efficiency(d)
    out = []
    for m in brmc_models(d)
        off = Dict(r["seed"] => r for r in
                   brmc_select(d, m["spec"], "adaptive_centering", false))
        on = Dict(r["seed"] => r for r in
                  brmc_select(d, m["spec"], "adaptive_centering", true))
        seeds = sort(collect(intersect(keys(off), keys(on))))
        isempty(seeds) && continue
        ess_per_s(r) = r["ess_min_shared_constrained"] / r["wall_s"]
        push!(out, (
            spec = m["spec"], n = length(seeds),
            grad_ratio = brmc_med([on[s]["grad_evals"] / off[s]["grad_evals"]
                                   for s in seeds]),
            ess_per_grad_ratio = brmc_med([on[s]["ess_min_per_grad"] /
                                           off[s]["ess_min_per_grad"] for s in seeds]),
            wall_ratio = brmc_med([on[s]["wall_s"] / off[s]["wall_s"] for s in seeds]),
            ess_per_s_ratio = brmc_med([ess_per_s(on[s]) / ess_per_s(off[s])
                                        for s in seeds]),
            changed = count(s -> on[s]["grad_evals"] != off[s]["grad_evals"] ||
                                 on[s]["ess_min_shared_constrained"] !=
                                 off[s]["ess_min_shared_constrained"], seeds),
        ))
    end
    out
end

"""
The control: arms with no reparametrizer must be UNAFFECTED by `nonlinear_adapt`.

Returns one entry per (spec, arm) for the two control arms, with the number of
seeds whose flag-on and flag-off runs agree exactly on gradient count. Anything
below `n_seeds` means the flag reached something it should not have, and the
`adaptive_centering` result on that spec cannot be attributed to the flag.
"""
function brmc_controls(d)
    out = []
    for m in brmc_models(d), arm in ("noncentered", "centered")
        off = Dict(r["seed"] => r for r in brmc_select(d, m["spec"], arm, false))
        on = Dict(r["seed"] => r for r in brmc_select(d, m["spec"], arm, true))
        seeds = sort(collect(intersect(keys(off), keys(on))))
        same = count(s -> off[s]["grad_evals"] == on[s]["grad_evals"] &&
                          off[s]["ess_min_shared_constrained"] ==
                          on[s]["ess_min_shared_constrained"], seeds)
        push!(out, (spec = m["spec"], arm = arm, n = length(seeds), identical = same))
    end
    out
end

brmc_controls_clean(d) = all(c -> c.n > 0 && c.identical == c.n, brmc_controls(d))

"""
Paired flag-on / flag-off effect on the arm that actually carries a
reparametrizer. Paired by seed: same seed, same target, one knob.
"""
function brmc_paired(d, arm = "adaptive_centering")
    out = []
    for m in brmc_models(d)
        off = Dict(r["seed"] => r for r in brmc_select(d, m["spec"], arm, false))
        on = Dict(r["seed"] => r for r in brmc_select(d, m["spec"], arm, true))
        ratios = Float64[]
        for s in sort(collect(intersect(keys(off), keys(on))))
            a, b = off[s]["ess_min_per_grad"], on[s]["ess_min_per_grad"]
            (a === nothing || b === nothing || !(a > 0)) && continue
            push!(ratios, b / a)
        end
        isempty(ratios) && continue
        push!(out, (spec = m["spec"], n = length(ratios), med = Statistics.median(ratios),
                    lo = minimum(ratios), hi = maximum(ratios),
                    improved = count(>(1), ratios)))
    end
    out
end

"""
Where the adaptive arm lands between the two static parametrizations, per spec.

`recovered` is the fraction of the non-centered → centered gap that adaptive
centering with the flag on closes. It is only meaningful when centered actually
beats non-centered, so `nothing` is returned when it does not — a ratio computed
across a gap of the wrong sign is worse than no number.
"""
function brmc_gap(d)
    out = []
    for m in brmc_models(d)
        g(arm, adapt) = begin
            rs = brmc_select(d, m["spec"], arm, adapt)
            brmc_med([r["ess_min_per_grad"] for r in rs if r["ess_min_per_grad"] !== nothing])
        end
        nc, c, ad = g("noncentered", false), g("centered", false), g("adaptive_centering", true)
        any(isnan, (nc, c, ad)) && continue
        rec = c > nc ? (ad - nc) / (c - nc) : nothing
        push!(out, (spec = m["spec"], noncentered = nc, centered = c, adaptive = ad,
                    recovered = rec, centered_wins = c > nc))
    end
    out
end

"""
Verdict, counted from the rows rather than asserted.

Deliberately reports the count both ways. A benchmark over eight posteriors that
says only "it helps" is hiding the specs where it did not.
"""
function brmc_verdict(d)
    p = brmc_paired(d)
    helped = count(x -> x.med > 1, p)
    hurt = count(x -> x.med < 1, p)
    gaps = brmc_gap(d)
    beat_centered = count(x -> x.adaptive > x.centered, gaps)
    (n_specs = length(p), helped = helped, hurt = hurt,
     n_gaps = length(gaps), beat_centered = beat_centered,
     centered_wins = count(x -> x.centered_wins, gaps),
     controls_clean = brmc_controls_clean(d))
end

brmc_failures(d) = [r for r in brmc_rows(d) if !r["ok"]]

# DynamicHMC standard-warmup comparison artifact. Kept in this definitions-only
# harness because it shares the same generated model metadata and constrained-
# space ESS contract as the nonlinear-centering artifact above.

brmc_standard_arm_order() = [
    "warmuphmc_noncentered",
    "warmuphmc_centered",
    "warmuphmc_fixed_centering",
    "warmuphmc_adaptive_centering",
    "dynamichmc_noncentered",
    "dynamichmc_centered",
]

brmc_standard_arm_label(a) = get(Dict(
    "warmuphmc_noncentered" => "WarmupHMC — generated non-centered",
    "warmuphmc_centered" => "WarmupHMC — generated centered",
    "warmuphmc_fixed_centering" => "WarmupHMC — nonlinear wrapper fixed at c=0",
    "warmuphmc_adaptive_centering" => "WarmupHMC — adaptive centering",
    "dynamichmc_noncentered" => "DynamicHMC — generated non-centered",
    "dynamichmc_centered" => "DynamicHMC — generated centered",
), a, a)

brmc_standard_select(d, spec, arm) =
    [r for r in brmc_rows(d) if r["spec"] == spec && r["arm"] == arm && r["ok"]]

"""Median absolute diagnostics per generated model and standard-comparison arm."""
function brmc_standard_summary(d)
    out = []
    for m in brmc_models(d), arm in brmc_standard_arm_order()
        attempted = [r for r in brmc_rows(d)
                     if r["spec"] == m["spec"] && r["arm"] == arm]
        rs = brmc_standard_select(d, m["spec"], arm)
        isempty(rs) && continue
        push!(out, (
            spec = m["spec"], arm = arm, n = length(rs),
            n_total = length(attempted), n_failed = length(attempted) - length(rs),
            draws = brmc_med([r["n_draws_actual"] for r in rs]),
            grad = brmc_med([r["grad_evals"] for r in rs]),
            ess = brmc_med([r["ess_min_shared_constrained"] for r in rs]),
            per_grad = brmc_med([r["ess_min_per_grad"] for r in rs]),
            wall = brmc_med([r["wall_s"] for r in rs]),
            per_s = brmc_med([r["ess_min_shared_constrained"] / r["wall_s"]
                              for r in rs if r["wall_s"] > 0]),
            ndiv = sum(r["n_divergent"] for r in attempted),
        ))
    end
    out
end

# Historical-gallery coverage metadata. These read fields the runner records per
# model; nothing here re-derives structure from a formula string, so a coverage
# claim on the page can only come from something the runner actually measured on
# the real data it sampled.

"""
Grouping factors of one model as `name (levels)`, in the inventory's order.

`n_groups` is counted on the adapted data, so a level count here is the number of
distinct groups actually present in the rows that were sampled — not the number
the upstream dataset documents.
"""
function brmc_group_summary(m)
    n = get(m, "n_groups", Dict{String,Any}())
    cols = get(m, "grouping_factors", String[])
    isempty(cols) && return "—"
    join(["`$(g)` ($(get(n, g, "?")))" for g in cols], ", ")
end

"""
An inventory row key as a table cell — backticked, because a bare one corrupts.

`md_table` runs every cell through `md_cell`, which *parses* it as markdown so a
cell can legitimately carry code spans and links. Julia's markdown parser treats
INTRAWORD `_` as emphasis, so a key with two or more underscores is silently
rewritten: measured through `Markdown.parse`, `bambi:predict_new_groups` renders
as `bambi:predict*new*groups` and `mixed_models_jl:penicillin_crossed` as
`mixed*models*jl:penicillin_crossed`. The build stays green and the table renders;
the identifier a reader copies out simply does not exist.

Five of the sixteen published keys hit this, so it is the default cell for a spec
key anywhere on the page rather than a fix applied where someone noticed.
"""
brmc_spec_cell(spec::AbstractString) = "`" * spec * "`"

"""
Random-effect block widths of one model as `K=… on group`, in body order.

Empty string when the runner recorded no blocks, which would itself be a defect
worth seeing rather than hiding behind a dash.
"""
function brmc_block_summary(m)
    blocks = get(m, "random_effect_blocks", Any[])
    isempty(blocks) && return "—"
    join(["K=$(b["k"]) on `$(b["group"])`" *
          (b["correlated"] ? "" : " (uncorrelated)") for b in blocks], "; ")
end

brmc_max_k(m) = maximum((b["k"] for b in get(m, "random_effect_blocks", Any[]));
                        init = 0)

"""
Models whose generated block widths differ from the historical `lme4` reading.

Zipped positionally: the runner emits both lists from the same regex over the
same block order, so entry `i` of one describes the same block as entry `i` of the
other. A length mismatch means the generated body restructured the random-effect
part rather than merely reinterpreting a term, which is a different and larger
departure — it is reported as such instead of being compared element-wise.
"""
function brmc_block_width_departures(d)
    out = []
    for m in brmc_models(d)
        got = get(m, "random_effect_blocks", Any[])
        want = get(m, "historical_random_effect_blocks", Any[])
        if length(got) != length(want)
            push!(out, (spec = m["spec"], kind = "block count differs",
                        detail = "$(length(want)) historical block(s) vs " *
                                 "$(length(got)) generated"))
            continue
        end
        differing = [(w, g) for (w, g) in zip(want, got) if w["k"] != g["k"]]
        isempty(differing) && continue
        push!(out, (
            spec = m["spec"], kind = "block width differs",
            detail = join(["`($(g["terms"]) | $(g["group"]))` is width " *
                           "$(w["k"]) historically, $(g["k"]) as generated"
                           for (w, g) in differing], "; "),
        ))
    end
    out
end

"""
Per-model count of constrained coordinates dropped from the ESS minimum.

`n_constant` is the number of shared constrained coordinates that were constant
or non-finite across a trajectory's draws, and therefore excluded before taking
the minimum ESS. Excluding them is right — a coordinate that never moves has no
effective sample size to report, and including it would make every model's
headline ESS zero — but it does narrow what the published minimum is a minimum
*over*, and by a lot on some rows. That fraction belongs on the page rather than
in a sentence promising it was counted.

Returns one entry per model with a nonzero drop anywhere, largest share first;
a model that never dropped a coordinate is omitted so the table stays about the
rows where the caveat bites.
"""
function brmc_dropped_coordinates(d)
    out = []
    for m in brmc_models(d)
        rows = [r for r in brmc_rows(d) if r["spec"] == m["spec"]]
        isempty(rows) && continue
        worst = maximum(r["n_constant"] for r in rows)
        worst == 0 && continue
        push!(out, (spec = m["spec"], shared = m["n_names_shared"], max_dropped = worst,
                    cells = count(r -> r["n_constant"] > 0, rows), n_cells = length(rows),
                    share = worst / m["n_names_shared"]))
    end
    sort(out; by = x -> -x.share)
end

"""Models whose adapter departs from the historical categorical coding."""
brmc_categorical_departures(d) =
    [m for m in brmc_models(d) if !isempty(get(m, "categorical_departures", ""))]

# K>2 probe artifact. A separate runner and a separate results directory: it
# deliberately does NOT apply the publishable-row gate, because its whole purpose
# is to report what happens to a high-K card on real data rather than to exclude
# it beforehand.

brmc_high_k_candidates(d) = d["candidates"]

brmc_high_k_reached(c) = get(c, "reached", "")

"""Seed-paired numerator/denominator efficiency ratios for two sampler arms."""
function brmc_standard_ratios(d, numerator, denominator)
    out = []
    for m in brmc_models(d)
        num = Dict(r["seed"] => r for r in
                   brmc_standard_select(d, m["spec"], numerator))
        den = Dict(r["seed"] => r for r in
                   brmc_standard_select(d, m["spec"], denominator))
        seeds = sort(collect(intersect(keys(num), keys(den))))
        isempty(seeds) && continue
        ess_per_s(r) = r["ess_min_shared_constrained"] / r["wall_s"]
        push!(out, (
            spec = m["spec"], numerator = numerator, denominator = denominator,
            n = length(seeds),
            grad_ratio = brmc_med([num[s]["grad_evals"] / den[s]["grad_evals"]
                                   for s in seeds]),
            ess_per_grad_ratio = brmc_med([
                num[s]["ess_min_per_grad"] / den[s]["ess_min_per_grad"]
                for s in seeds]),
            wall_ratio = brmc_med([num[s]["wall_s"] / den[s]["wall_s"]
                                   for s in seeds]),
            ess_per_s_ratio = brmc_med([ess_per_s(num[s]) / ess_per_s(den[s])
                                        for s in seeds]),
        ))
    end
    out
end
