# AlgebraOfVega figures derived from the raw generated-BRM seed rows.
# Definitions only: Documenter includes this file at build time.

using AlgebraOfVega

brmc_plot_model_label(spec) = get(Dict(
    "lme4:dyestuff_re" => "dyestuff",
    "lme4:sleepstudy_slope" => "sleepstudy slope",
    "bambi:sleepstudy" => "sleepstudy (Bambi)",
    "mixed_models_jl:penicillin_crossed" => "penicillin crossed",
    "bambi:radon_partial" => "radon partial pooling",
    "bambi:radon_floor" => "radon floor",
    "bambi:radon_slopes" => "radon slopes",
    "bambi:dietox" => "dietox",
), spec, spec)

function brmc_standard_plot_rows(d, metric; multiplier=1.0)
    Row = NamedTuple{(:model, :arm, :seed, :value),
                     Tuple{String,String,Int,Float64}}
    values = Row[]
    for row in brmc_rows(d)
        row["ok"] || continue
        value = metric(row)
        value isa Real && isfinite(value) && value > 0 || error(
            "generated-BRM AoV figure requires a positive finite metric; " *
            "got $(repr(value)) for $(row["spec"]) / $(row["arm"]) / " *
            "seed $(row["seed"])",
        )
        push!(values, (
            model=brmc_plot_model_label(row["spec"]),
            arm=brmc_standard_arm_label(row["arm"]),
            seed=row["seed"],
            value=Float64(multiplier * value),
        ))
    end
    isempty(values) && error("generated-BRM AoV figure has zero rows")
    values
end

"""Seed-level constrained-space ESS per thousand generated-density gradients."""
function brmc_gradient_efficiency_plot(d)
    rows = brmc_standard_plot_rows(d, row -> row["ess_min_per_grad"];
                                   multiplier=1000.0)
    (
        data(rows) *
        mapping(
            :model => "generated BRM model",
            :value => "min shared ESS / 1,000 gradients";
            color=:arm => "default-warmup arm",
        ) *
        pointinterval(orientation=:vertical)
    ) * config(
        width=680,
        height=360,
        title="Gradient sampling efficiency across seeds",
        scales=scales(Y=(; scale=log10, nice=false, zero=false)),
    )
end

"""Seed-level constrained-space ESS per measured sampling second."""
function brmc_runtime_efficiency_plot(d)
    rows = brmc_standard_plot_rows(
        d,
        row -> row["ess_min_shared_constrained"] / row["wall_s"],
    )
    (
        data(rows) *
        mapping(
            :model => "generated BRM model",
            :value => "min shared ESS / second";
            color=:arm => "default-warmup arm",
        ) *
        pointinterval(orientation=:vertical)
    ) * config(
        width=680,
        height=360,
        title="Runtime sampling efficiency across seeds",
        scales=scales(Y=(; scale=log10, nice=false, zero=false)),
    )
end
