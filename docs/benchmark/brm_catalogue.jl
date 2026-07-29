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
    "warmuphmc_adaptive_centering",
    "dynamichmc_noncentered",
    "dynamichmc_centered",
]

brmc_standard_arm_label(a) = get(Dict(
    "warmuphmc_noncentered" => "WarmupHMC — generated non-centered",
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
        rs = brmc_standard_select(d, m["spec"], arm)
        isempty(rs) && continue
        push!(out, (
            spec = m["spec"], arm = arm, n = length(rs),
            draws = brmc_med([r["n_draws_actual"] for r in rs]),
            grad = brmc_med([r["grad_evals"] for r in rs]),
            ess = brmc_med([r["ess_min_shared_constrained"] for r in rs]),
            per_grad = brmc_med([r["ess_min_per_grad"] for r in rs]),
            wall = brmc_med([r["wall_s"] for r in rs]),
            per_s = brmc_med([r["ess_min_shared_constrained"] / r["wall_s"]
                              for r in rs if r["wall_s"] > 0]),
            ndiv = sum(r["n_divergent"] for r in rs),
        ))
    end
    out
end

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
