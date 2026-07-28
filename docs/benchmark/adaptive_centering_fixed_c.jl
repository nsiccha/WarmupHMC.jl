# Definitions-only derivation layer for `adaptive_centering_fixed_c_run.jl`.
# Loaded by Documenter through `docs/tables.jl`; do not add driver code here.

import Statistics

ace_chain_rows(d) = [r for r in d["rows"] if r["row_type"] == "chain"]
ace_pooled_rows(d) = [r for r in d["rows"] if r["row_type"] == "pooled"]

ace_arm_order() = [
    "exact_score_reference",
    "invariant_proxy",
    "whitened_noncentered",
    "fully_centered",
]

ace_arm_label(arm) = get(Dict(
    "exact_score_reference" => "exact-score / factorized reference",
    "invariant_proxy" => "strict-online invariant proxy",
    "whitened_noncentered" => "fully whitened noncentered",
    "fully_centered" => "fully centered",
), arm, arm)

ace_family_label(family) = family == "gaussian" ? "Gaussian" :
                           family == "student" ? "Student-t(5)" : family

ace_fmt_range(xs; digits = 2) = begin
    lo, hi = extrema(xs)
    string(round(lo; digits), "–", round(hi; digits))
end

function ace_summary_rows(d)
    chain_rows = ace_chain_rows(d)
    pooled_rows = ace_pooled_rows(d)
    out = Vector{Vector{Any}}()
    for family in d["config"]["families"], arm in ace_arm_order()
        rs = [r for r in chain_rows if r["family"] == family && r["arm"] == arm]
        isempty(rs) && continue
        pooled = only(r for r in pooled_rows if r["family"] == family && r["arm"] == arm)
        bulk_eff = [r["min_bulk_ess_per_1000_grad"] for r in rs]
        tail_eff = [r["min_tail_ess_per_1000_grad"] for r in rs]
        grad_draw = [r["gradients_per_draw"] for r in rs]
        push!(out, Any[
            ace_family_label(family),
            ace_arm_label(arm),
            Statistics.median(bulk_eff),
            ace_fmt_range(bulk_eff),
            Statistics.median(tail_eff),
            Statistics.median(grad_draw),
            ace_fmt_range(grad_draw),
            sum(r["n_divergent"] for r in rs),
            pooled["min_ess_bulk"],
            1000pooled["min_ess_bulk"] / pooled["total_gradient_evaluations"],
            pooled["min_ess_tail"],
            1000pooled["min_ess_tail"] / pooled["total_gradient_evaluations"],
            pooled["max_rhat"],
        ])
    end
    out
end

function ace_raw_pooled_rows(d)
    rows = sort(ace_pooled_rows(d); by = r -> (
        findfirst(==(r["family"]), d["config"]["families"]),
        findfirst(==(r["arm"]), ace_arm_order()),
    ))
    [Any[
        ace_family_label(r["family"]),
        ace_arm_label(r["arm"]),
        r["n_chains"],
        r["n_common_draws_per_chain"],
        r["total_divergent"],
        r["total_gradient_evaluations"],
        r["min_ess_bulk"],
        1000r["min_ess_bulk"] / r["total_gradient_evaluations"],
        r["min_ess_tail"],
        1000r["min_ess_tail"] / r["total_gradient_evaluations"],
        r["max_rhat"],
    ] for r in rows]
end

function ace_raw_chain_rows(d)
    rows = sort(ace_chain_rows(d); by = r -> (
        findfirst(==(r["family"]), d["config"]["families"]),
        findfirst(==(r["arm"]), ace_arm_order()),
        r["seed"],
    ))
    [Any[
        ace_family_label(r["family"]),
        ace_arm_label(r["arm"]),
        r["seed"],
        r["n_draws"],
        r["n_divergent"],
        r["gradient_evaluations"],
        r["total_transitions"],
        r["gradients_per_draw"],
        r["gradients_per_transition"],
        r["min_ess_bulk"],
        r["min_ess_tail"],
        r["min_bulk_ess_per_1000_grad"],
        r["min_tail_ess_per_1000_grad"],
        r["wall_seconds"],
    ] for r in rows]
end

function ace_validation_rows(d)
    [Any[
        ace_family_label(family),
        v["n_points"],
        v["max_common_frame_abs"],
        v["max_coordinate_roundtrip_abs"],
        v["max_transport_logdensity_abs"],
        v["max_gradient_fd_abs"],
    ] for (family, v) in sort!(collect(d["config"]["validation"]); by = first)]
end

function ace_proxy_comparison(d, family)
    rows = ace_chain_rows(d)
    optimum = Dict(r["seed"] => r for r in rows
                   if r["family"] == family && r["arm"] == "exact_score_reference")
    proxy = Dict(r["seed"] => r for r in rows
                 if r["family"] == family && r["arm"] == "invariant_proxy")
    seeds = sort!(collect(intersect(keys(optimum), keys(proxy))))
    ratios = [proxy[s]["min_bulk_ess_per_1000_grad"] /
              optimum[s]["min_bulk_ess_per_1000_grad"] for s in seeds]
    median_ratio = Statistics.median(ratios)
    n_low = count(<(0.8), ratios)
    n_high = count(>(1.25), ratios)
    required = ceil(Int, 0.75length(ratios))
    verdict = if median_ratio < 0.8 && n_low >= required
        "materially worse at the predeclared 20% threshold"
    elseif median_ratio > 1.25 && n_high >= required
        "materially better at the predeclared 25% threshold"
    else
        "not materially separated at the predeclared threshold"
    end
    Dict(
        "family" => family,
        "seeds" => seeds,
        "ratios" => ratios,
        "median_ratio" => median_ratio,
        "min_ratio" => minimum(ratios),
        "max_ratio" => maximum(ratios),
        "below_0_8" => n_low,
        "above_1_25" => n_high,
        "verdict" => verdict,
    )
end

function ace_comparison_rows(d)
    [begin
        c = ace_proxy_comparison(d, family)
        Any[
            ace_family_label(family),
            c["median_ratio"],
            ace_fmt_range(c["ratios"]; digits = 3),
            string(c["below_0_8"], "/", length(c["ratios"])),
            c["verdict"],
        ]
    end for family in d["config"]["families"]]
end

function ace_conclusion(d)
    comparisons = [ace_proxy_comparison(d, family) for family in d["config"]["families"]]
    if all(c -> occursin("not materially separated", c["verdict"]), comparisons)
        return "Across the synthetic controls, the strict-online invariant proxy was not " *
               "materially separated from the exact-score, factorized reference under the predeclared " *
               "20% paired efficiency criterion. This is a target-specific stress/control " *
               "result, not evidence that the proxy is generally optimal for BRM models."
    end
    join((string(ace_family_label(c["family"]), ": ", c["verdict"])
          for c in comparisons), " ")
end
