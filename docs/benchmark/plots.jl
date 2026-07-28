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
