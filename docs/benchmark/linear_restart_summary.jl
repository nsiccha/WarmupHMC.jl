# Derived summaries for `linear_restart.json`.
#
# This file deliberately only defines functions and depends on Statistics. It
# is included both by the sampling driver and by Documenter's `@eval` blocks,
# so the checked result stays rows + configuration while every displayed
# figure is recomputed from those rows.

import Statistics

const LINEAR_SYNTHETIC_TARGETS = Set([
    "diag_gaussian", "diag_fallback_probe", "correlated_gaussian",
])

function _linear_values(rows, key)
    Float64[
        row[key] for row in rows
        if get(row, "ok", false) && get(row, key, nothing) isa Real &&
           isfinite(row[key])
    ]
end

function _linear_median(rows, key)
    values = _linear_values(rows, key)
    isempty(values) ? NaN : Statistics.median(values)
end

"Return one median row per target and policy from raw benchmark rows."
function linear_restart_summary(data::AbstractDict)
    rows = data["runs"]
    [begin
        selected = [row for row in rows
                    if row["target"] == target && row["arm"] == arm["name"]]
        successful = [row for row in selected if get(row, "ok", false)]
        active = sort!(unique!(String[row["active_transformation"] for row in successful]))
        base = Dict{String,Any}(
            "target" => target,
            "arm" => arm["name"],
            "source" => arm["source"],
            "weighting" => arm["weighting"],
            "n_runs" => length(successful),
            "grad_evals_median" => _linear_median(selected, "grad_evals"),
            "restarts_median" => _linear_median(selected, "n_restarts"),
            "min_ess_median" => _linear_median(selected, "ess_min"),
            "min_ess_per_kgrad_median" => _linear_median(selected, "ess_min_per_kgrad"),
            "wall_s_median" => _linear_median(selected, "wall_s"),
            "divergences_total" => sum(Int(row["divergences"]) for row in successful),
            "active_transformations" => join(active, ","),
            "adaptive_reflections_median" => _linear_median(selected, "adaptive_reflections"),
            "linear_metric_fallbacks_median" => _linear_median(selected, "linear_metric_fallbacks"),
        )
        if target in LINEAR_SYNTHETIC_TARGETS
            merge!(base, Dict(
                "covariance_relative_error_median" =>
                    _linear_median(selected, "covariance_relative_error"),
                "mean_scaled_error_median" =>
                    _linear_median(selected, "mean_scaled_error"),
            ))
        else
            merge!(base, Dict(
                "reference_mean_z_rmse_median" =>
                    _linear_median(selected, "reference_mean_z_rmse"),
                "reference_sd_log_rmse_median" =>
                    _linear_median(selected, "reference_sd_log_rmse"),
            ))
        end
        base
    end for target in String.(data["targets"]) for arm in data["arms"]]
end

function _linear_ratio_summary(pairs, key; higher_is_better)
    values = Float64[]
    for (test, baseline) in pairs
        numerator, denominator = get(test, key, nothing), get(baseline, key, nothing)
        numerator isa Real && denominator isa Real || continue
        isfinite(numerator) && isfinite(denominator) && numerator > 0 && denominator > 0 ||
            continue
        push!(values, numerator / denominator)
    end
    Dict{String,Any}(
        "median" => isempty(values) ? NaN : Statistics.median(values),
        "geomean" => isempty(values) ? NaN : exp(Statistics.mean(log, values)),
        "min" => isempty(values) ? NaN : minimum(values),
        "max" => isempty(values) ? NaN : maximum(values),
        "wins" => count(higher_is_better ? >(1) : <(1), values),
        "n" => length(values),
    )
end

function _linear_paired_comparison(rows, target, arm, baseline_arm, comparison)
    tests = Dict(Int(row["seed"]) => row for row in rows
                 if get(row, "ok", false) && row["target"] == target && row["arm"] == arm)
    baselines = Dict(Int(row["seed"]) => row for row in rows
                     if get(row, "ok", false) && row["target"] == target && row["arm"] == baseline_arm)
    seeds = sort!(collect(intersect(keys(tests), keys(baselines))))
    pairs = [(tests[seed], baselines[seed]) for seed in seeds]
    ess = _linear_ratio_summary(pairs, "ess_min_per_kgrad"; higher_is_better=true)
    gradients = _linear_ratio_summary(pairs, "grad_evals"; higher_is_better=false)
    wall = _linear_ratio_summary(pairs, "wall_s"; higher_is_better=false)
    accuracy_key = target in LINEAR_SYNTHETIC_TARGETS ?
        "covariance_relative_error" : "reference_mean_z_rmse"
    accuracy = _linear_ratio_summary(pairs, accuracy_key; higher_is_better=false)
    changed_pairs = [pair for pair in pairs
                     if pair[1]["n_restarts"] != pair[2]["n_restarts"]]
    changed_ess = _linear_ratio_summary(
        changed_pairs, "ess_min_per_kgrad"; higher_is_better=true,
    )
    Dict{String,Any}(
        "target" => target,
        "comparison" => comparison,
        "arm" => arm,
        "baseline_arm" => baseline_arm,
        "n_pairs" => length(pairs),
        "ess_efficiency_ratio_median" => ess["median"],
        "ess_efficiency_ratio_geomean" => ess["geomean"],
        "ess_efficiency_ratio_min" => ess["min"],
        "ess_efficiency_ratio_max" => ess["max"],
        "ess_efficiency_wins" => ess["wins"],
        "gradient_ratio_median" => gradients["median"],
        "wall_ratio_median" => wall["median"],
        "accuracy_metric" => accuracy_key,
        "accuracy_error_ratio_median" => accuracy["median"],
        "accuracy_error_ratio_geomean" => accuracy["geomean"],
        "accuracy_wins" => accuracy["wins"],
        "restart_count_differs" => length(changed_pairs),
        "changed_ess_efficiency_ratio_median" => changed_ess["median"],
        "changed_ess_efficiency_ratio_min" => changed_ess["min"],
        "changed_ess_efficiency_ratio_max" => changed_ess["max"],
    )
end

"Return seed-paired policy comparisons derived from raw benchmark rows."
function linear_restart_comparisons(data::AbstractDict)
    rows = data["runs"]
    targets = String.(data["targets"])
    arm_names = String[arm["name"] for arm in data["arms"]]
    vcat(
        [_linear_paired_comparison(rows, target, arm, "halo_unit", "arm_vs_halo")
         for target in targets for arm in arm_names[2:end]],
        [_linear_paired_comparison(rows, target, stepsize, unit, "stepsize_vs_unit")
         for target in targets
         for (stepsize, unit) in (("all_good_stepsize", "all_good_unit"),
                                  ("nuts_stepsize", "nuts_unit"))],
    )
end
