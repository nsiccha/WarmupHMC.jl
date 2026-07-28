# Vega-Lite specs derived from the checked-in benchmark rows.
#
# Loaded by an `@eval` block through `load_harness("plots.jl")` and handed to
# `vega_figure` in docs/tables.jl, which emits the div the theme embedder
# renders. Definitions only — nothing here runs at include time, because
# include time IS docs-build time (see `load_harness`).
#
# WHY A PLAIN Dict AND NOT AlgebraOfVega
#
# A Vega-Lite spec is JSON, and JSON is what `docs/Project.toml` already has.
# Building the spec as a `Dict` keeps the docs environment at its current four
# dependencies; reaching for a plotting package to emit JSON that Julia can
# already write would put a rendering stack in the way of every docs build, for
# no expressiveness this needs. If a page ever wants a chart whose grammar is
# genuinely awkward here, that is the moment to reconsider — not before.
#
# WHY THE FIGURE IS DERIVED AND NOT CHECKED IN
#
# Same rule as every table on these pages: rows are the data, everything else is
# a function of the rows. A spec committed next to the JSON it plots is a second
# copy of those numbers that nothing forces to agree, and it would go stale in
# exactly the silent direction — a chart still renders when its numbers are old.

"""
    ratio_rows(rows) -> Vector{Dict}

Reduce `annotation_sweep.json` rows to the fields a chart needs, dropping any
row where either timing is absent.

A missing timing is a real state — a backend that could not run on that target
records `nothing` — and it must be dropped rather than plotted as zero, which
would read as "free" instead of "not measured".
"""
function ratio_rows(rows)
    out = Dict{String,Any}[]
    for r in rows
        (r["ns_const"] === nothing || r["ns_forwarddiff"] === nothing) && continue
        push!(out, Dict{String,Any}(
            "target" => r["target"],
            "d" => r["dim"],
            "c" => string(r["c_source"]),
            "ns_const" => Float64(r["ns_const"]),
            "ratio" => Float64(r["ns_const"]) / Float64(r["ns_forwarddiff"]),
        ))
    end
    out
end

"""
    backend_ratio_spec(rows; height=300) -> Dict

`Const ÷ ForwardDiff` against dimension, one point per sweep row.

This plots the column the page argues about in prose — that the ratio is below
parity everywhere, but that its variation does NOT track `d`. Both claims are
things a reader can check against the picture directly: the parity rule makes
the first a matter of which side of a line the points fall on, and the second
shows up as the vertical spread AT a single `d` being comparable to the spread
across all of them.

Encoding choices that are load-bearing rather than cosmetic:

  * **`x` is log-scaled.** The dimensions measured are 10, 26, 88, 89 — on a
    linear axis the last two land on top of each other and the first sits alone
    against the left edge, which reads as a gap in the data rather than as the
    sampling this sweep actually did.
  * **`y` includes zero.** A zoomed y-axis would exaggerate differences between
    ratios that are all within a factor of about two of each other, and the
    parity rule at `1×` is the reference the whole figure exists to be read
    against — it must be on the axis, not off the top of it.
  * **`c` is a SHAPE, not a second colour.** It is a nuisance parameter here;
    encoding it the same way as target identity would imply the two matter
    equally, which is the misreading this figure is meant to prevent.

`d` and the tooltip carry the raw `ns_const` too, so a reader who wants the
absolute cost behind a ratio does not have to go back to the table for it.
"""
function backend_ratio_spec(rows; height::Int = 300)
    values = ratio_rows(rows)
    Dict(
        "\$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
        "width" => "container",
        "height" => height,
        "data" => Dict("values" => values),
        "layer" => [
            Dict(
                "mark" => Dict("type" => "rule", "strokeDash" => [4, 4],
                               "opacity" => 0.6),
                "encoding" => Dict(
                    "y" => Dict("datum" => 1, "type" => "quantitative")),
            ),
            Dict(
                "mark" => Dict("type" => "point", "filled" => true,
                               "size" => 110, "opacity" => 0.85),
                "encoding" => Dict(
                    "x" => Dict(
                        "field" => "d", "type" => "quantitative",
                        "scale" => Dict("type" => "log", "nice" => false,
                                        "padding" => 24),
                        "axis" => Dict("values" => sort(unique(v["d"] for v in values)),
                                       "format" => "d", "grid" => false),
                        "title" => "dimension d"),
                    "y" => Dict(
                        "field" => "ratio", "type" => "quantitative",
                        "title" => "Const ÷ ForwardDiff (lower is faster)"),
                    "color" => Dict(
                        "field" => "target", "type" => "nominal",
                        # Target names are posteriordb identifiers, and Vega's
                        # default legend `labelLimit` of 160px ellipsises them
                        # in the middle — which for these names deletes exactly
                        # the part that distinguishes them (`…_partially_pooled`
                        # from `…_variable_intercept`). The legend is
                        # bottom-oriented, so the width available to a label is
                        # the content column, ~688px in this theme; 640 is that
                        # minus the symbol and padding. Chosen from the layout,
                        # not fitted to where truncation happened to stop.
                        "legend" => Dict("title" => "target",
                                         "orient" => "bottom",
                                         "columns" => 1,
                                         "labelLimit" => 640)),
                    "shape" => Dict(
                        "field" => "c", "type" => "nominal",
                        "legend" => Dict("title" => "c", "orient" => "bottom")),
                    "tooltip" => [
                        Dict("field" => "target", "type" => "nominal",
                             "title" => "target"),
                        Dict("field" => "d", "type" => "quantitative",
                             "title" => "d"),
                        Dict("field" => "c", "type" => "nominal", "title" => "c"),
                        Dict("field" => "ns_const", "type" => "quantitative",
                             "title" => "Const ns/grad", "format" => ".0f"),
                        Dict("field" => "ratio", "type" => "quantitative",
                             "title" => "Const ÷ ForwardDiff", "format" => ".3f"),
                    ],
                ),
            ),
        ],
    )
end

ace_plot_arm_label(arm) = get(Dict(
    "exact_score_reference" => "reference c*",
    "invariant_proxy" => "online proxy",
    "whitened_noncentered" => "whitened",
    "fully_centered" => "centered",
), arm, arm)

function ace_efficiency_plot_rows(d)
    rows = ace_chain_rows(d)
    isempty(rows) && error("adaptive-centering efficiency figure has zero chain rows")
    invalid = [r for r in rows if !(r["min_bulk_ess_per_1000_grad"] isa Real)]
    isempty(invalid) || error(
        "adaptive-centering efficiency figure has $(length(invalid)) " *
        "chain row(s) with missing or non-numeric bulk ESS/gradient",
    )
    values = [Dict{String,Any}(
        "family" => ace_family_label(r["family"]),
        "arm" => ace_plot_arm_label(r["arm"]),
        "seed" => r["seed"],
        "bulk_efficiency" => r["min_bulk_ess_per_1000_grad"],
    ) for r in rows]
    minimum(v["bulk_efficiency"] for v in values) > 0 || error(
        "adaptive-centering efficiency figure requires strictly positive " *
        "bulk ESS/gradient values for its log scale",
    )
    values
end

"""
    ace_bulk_ess_spec(d; height=280) -> Dict

Per-chain minimum-coordinate bulk ESS per thousand full-run gradient
evaluations, on a shared log scale. The boxes are computed by Vega-Lite from
the checked-in chain rows; no plotted summary is stored in the artifact.

Refuses empty, missing, zero and negative inputs so the log scale cannot
silently erase a failed chain.
"""
function ace_bulk_ess_spec(d; height::Int = 280)
    Dict(
        "\$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
        "data" => Dict("values" => ace_efficiency_plot_rows(d)),
        "facet" => Dict("column" => Dict(
            "field" => "family", "type" => "nominal",
            "sort" => ["Gaussian", "Student-t(5)"],
            "header" => Dict("title" => nothing),
        )),
        "spec" => Dict(
            "width" => 300,
            "height" => height,
            "mark" => Dict("type" => "boxplot", "extent" => "min-max",
                           "size" => 30),
            "encoding" => Dict(
                "x" => Dict(
                    "field" => "arm", "type" => "nominal",
                    "sort" => ["reference c*", "online proxy", "whitened", "centered"],
                    "title" => nothing,
                    "axis" => Dict("labelAngle" => -20, "labelLimit" => 110),
                ),
                "y" => Dict(
                    "field" => "bulk_efficiency", "type" => "quantitative",
                    "scale" => Dict("type" => "log"),
                    "title" => "min bulk ESS / 1,000 gradients",
                ),
                "color" => Dict(
                    "field" => "arm", "type" => "nominal",
                    "sort" => ["reference c*", "online proxy", "whitened", "centered"],
                    "legend" => nothing,
                ),
            ),
        ),
        "resolve" => Dict("scale" => Dict("y" => "shared")),
    )
end

function ace_ratio_plot_rows(d)
    isempty(ace_chain_rows(d)) &&
        error("adaptive-centering ratio figure has zero chain rows")
    values = Dict{String,Any}[]
    for family in d["config"]["families"]
        comparison = ace_proxy_comparison(d, family)
        for (i, (seed, ratio)) in enumerate(zip(
            comparison["seeds"], comparison["ratios"],
        ))
            push!(values, Dict{String,Any}(
                "family" => ace_family_label(family),
                "seed_index" => i,
                "seed" => seed,
                "ratio" => ratio,
            ))
        end
    end
    isempty(values) && error("adaptive-centering ratio figure has zero paired rows")
    values
end

"""
    ace_proxy_ratio_spec(d; height=260) -> Dict

Paired per-seed proxy/reference efficiency ratios. The dashed red rule is the
predeclared materially-worse threshold, the dotted grey rule is parity, and
the solid rule is the family median derived by Vega-Lite.
"""
function ace_proxy_ratio_spec(d; height::Int = 260)
    values = ace_ratio_plot_rows(d)
    ratios = [v["ratio"] for v in values]
    lo = min(0.8, minimum(ratios))
    hi = max(1.0, maximum(ratios))
    padding = max(0.02, 0.05(hi - lo))
    Dict(
        "\$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
        "data" => Dict("values" => values),
        "facet" => Dict("column" => Dict(
            "field" => "family", "type" => "nominal",
            "sort" => ["Gaussian", "Student-t(5)"],
            "header" => Dict("title" => nothing),
        )),
        "spec" => Dict(
            "width" => 300,
            "height" => height,
            "layer" => [
                Dict(
                    "mark" => Dict("type" => "rule", "color" => "#7f8c8d",
                                   "strokeDash" => [2, 3], "opacity" => 0.75),
                    "encoding" => Dict("y" => Dict(
                        "datum" => 1.0, "type" => "quantitative")),
                ),
                Dict(
                    "mark" => Dict("type" => "rule", "color" => "#c0392b",
                                   "strokeDash" => [6, 4], "opacity" => 0.85),
                    "encoding" => Dict("y" => Dict(
                        "datum" => 0.8, "type" => "quantitative")),
                ),
                Dict(
                    "transform" => [Dict(
                        "aggregate" => [Dict(
                            "op" => "median", "field" => "ratio", "as" => "median_ratio",
                        )],
                        "groupby" => ["family"],
                    )],
                    "mark" => Dict("type" => "rule", "color" => "#2c3e50",
                                   "strokeWidth" => 2.5),
                    "encoding" => Dict("y" => Dict(
                        "field" => "median_ratio", "type" => "quantitative")),
                ),
                Dict(
                    "mark" => Dict("type" => "point", "filled" => true,
                                   "size" => 80, "color" => "#2878b5", "opacity" => 0.85),
                    "encoding" => Dict(
                        "x" => Dict(
                            "field" => "seed_index", "type" => "quantitative",
                            "title" => "paired seed", "axis" => Dict(
                                "values" => collect(1:16), "format" => "d",
                                "labelOverlap" => true,
                            ),
                        ),
                        "y" => Dict(
                            "field" => "ratio", "type" => "quantitative",
                            "scale" => Dict("zero" => false,
                                            "domain" => [lo - padding, hi + padding]),
                            "title" => "proxy ÷ reference ESS/gradient",
                        ),
                        "tooltip" => [
                            Dict("field" => "family", "type" => "nominal",
                                 "title" => "family"),
                            Dict("field" => "seed", "type" => "quantitative",
                                 "title" => "seed", "format" => "d"),
                            Dict("field" => "ratio", "type" => "quantitative",
                                 "title" => "proxy ÷ reference", "format" => ".3f"),
                        ],
                    ),
                ),
            ],
        ),
        "resolve" => Dict("scale" => Dict("y" => "shared")),
    )
end
