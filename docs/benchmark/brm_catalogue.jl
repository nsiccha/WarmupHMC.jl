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
            wall = brmc_med([r["wall_s"] for r in rs if r["wall_s"] !== nothing]),
            ndiv = sum(r["n_divergent"] for r in rs),
            nconst = maximum(r["n_constant"] for r in rs; init = 0),
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
