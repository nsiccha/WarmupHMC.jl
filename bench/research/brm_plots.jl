# AlgebraOfVega figures derived from the raw generated-BRM seed rows.
# Definitions only: Documenter includes this file at build time.

using AlgebraOfVega

"""
Short axis label per inventory spec key.

Falls through to the raw `source:key` so a newly benchmarked row always renders
— an unlabelled axis tick is a cosmetic defect, not a reason to fail the docs
build. `brmc_plot_unlabelled_specs` reports the fallthroughs so the page can
show them rather than let them pass unnoticed.
"""
brmc_plot_model_label(spec) = get(Dict(
    "lme4:dyestuff_re" => "dyestuff",
    "lme4:sleepstudy_slope" => "sleepstudy slope",
    "bambi:sleepstudy" => "sleepstudy (Bambi)",
    "mixed_models_jl:penicillin_crossed" => "penicillin crossed",
    "bambi:radon_partial" => "radon partial pooling",
    "bambi:radon_floor" => "radon floor",
    "bambi:radon_slopes" => "radon slopes",
    "bambi:dietox" => "dietox",
    # historical-gallery tranche
    "vasishth:meta_sbi" => "SBI meta-analysis",
    "kruschke:fruitfly_anhecova" => "fruitfly ANCOVA",
    "kruschke:therapeutic_touch" => "therapeutic touch",
    "bambi:hierarchical_binomial_partial" => "baseball binomial",
    "burkner_papers:epilepsy_simple" => "epilepsy counts",
    "mixed_models_jl:contraception_glmm" => "contraception",
    "bambi:predict_new_groups" => "pulmonary slopes",
    "vasishth:n400_crossed" => "N400 crossed",
), spec, spec)

brmc_plot_unlabelled_specs(d) =
    [s for s in brmc_spec_keys(d) if brmc_plot_model_label(s) == s]

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

# The model axis is HORIZONTAL (one band row per model), not vertical.
#
# With the historical-gallery tranche the matrix carries 16 models, and a
# vertical band axis gives each of them ~42 px of a 680 px plot — too narrow for
# "therapeutic touch" or "radon partial pooling", so Vega-Lite silently culls
# labels and the reader cannot tell which point-interval is which model. Widening
# the plot instead is not available: `.vega-figure` has no `overflow-x`, and the
# VitePress content column is ~688 px, so anything wider overflows the page.
# Putting the model on Y and the metric on a log X axis gives every label a full
# band row, and the figure then grows DOWNWARDS with the model count — vertical
# space on a docs page is free.
#
# The category must ride the `y=` KEYWORD channel, which is what
# `pointinterval`'s own docstring shows (`mapping(:value, y=:parameter)`). Giving
# it as a second positional instead — `mapping(:value, :model)` — is accepted,
# renders, and SILENTLY COLLAPSES the model dimension: measured on the published
# eight-model artifact, `data.values` fell from 48 groups (8 models × 6 arms,
# carrying a `model` field) to 6 arm-level groups with no `model` field at all.
# Reversing the positionals under `:horizontal` at least fails loudly
# ("value column \"model\" has non-numeric eltype String and cannot be reduced
# with `quantile`"). The keyword form is the only one of the three that is right.
#
# One band row must hold six `yOffset`-dodged arms, and the point mark is
# `size=80` — a ~10 px diameter — so a band needs ~64 px before the arms start
# overplotting each other.
brmc_plot_height(rows) = 80 + 64 * length(unique(r.model for r in rows))

"""
Fail the docs build if a figure's emitted spec has lost the per-model dimension.

The defect this guards is the one described above, and the reason it needs a
guard rather than a comment is that it has **no visible symptom in the build**:
the collapsed spec is valid Vega-Lite, `aov_figure` renders it, the page ships,
and the figure looks like a normal six-interval comparison — just of the wrong
thing. Nothing downstream can tell that the interval labelled
`warmuphmc_adaptive_centering` is now pooled over every model in the matrix.

So the check is on the *emitted spec*, not on the mapping call: it re-reads what
`aov_figure` will publish and compares `data.values` against the (model, arm)
pairs the rows actually contain. Pairs rather than a product, so a model that
lost an arm to a failure narrows the expectation instead of failing the build.
"""
function brmc_checked_figure(layer, rows)
    spec = JSON.parse(String(repr(MIME"application/vnd.vegalite.v5+json"(), layer)))
    values = spec["data"]["values"]
    expected = length(unique((r.model, r.arm) for r in rows))
    length(values) == expected || error("""
        generated-BRM AoV figure aggregated $(length(values)) group(s), expected \
        $(expected) — one per (model, arm) pair in the rows.

        A count this far off means a mapping channel stopped grouping. See the
        comment above `brmc_plot_height`: the model must ride `y=` as a keyword,
        never as a second positional under `orientation=:horizontal`.
        """)
    all(v -> haskey(v, "model"), values) || error("""
        generated-BRM AoV figure dropped the `model` field from its aggregated \
        data, so every interval is pooled across models.
        """)
    layer
end

"""Seed-level constrained-space ESS per thousand generated-density gradients."""
function brmc_gradient_efficiency_plot(d)
    rows = brmc_standard_plot_rows(d, row -> row["ess_min_per_grad"];
                                   multiplier=1000.0)
    brmc_checked_figure((
        data(rows) *
        mapping(
            :value => "min shared ESS / 1,000 gradients";
            y=:model => "generated BRM model",
            color=:arm => "default-warmup arm",
        ) *
        pointinterval(orientation=:horizontal)
    ) * config(
        width=560,
        height=brmc_plot_height(rows),
        title="Gradient sampling efficiency across seeds",
        scales=scales(X=(; scale=log10, nice=false, zero=false)),
    ), rows)
end

"""Seed-level constrained-space ESS per measured sampling second."""
function brmc_runtime_efficiency_plot(d)
    rows = brmc_standard_plot_rows(
        d,
        row -> row["ess_min_shared_constrained"] / row["wall_s"],
    )
    brmc_checked_figure((
        data(rows) *
        mapping(
            :value => "min shared ESS / second";
            y=:model => "generated BRM model",
            color=:arm => "default-warmup arm",
        ) *
        pointinterval(orientation=:horizontal)
    ) * config(
        width=560,
        height=brmc_plot_height(rows),
        title="Runtime sampling efficiency across seeds",
        scales=scales(X=(; scale=log10, nice=false, zero=false)),
    ), rows)
end
