# Rendering checked-in benchmark JSON as documentation tables.
#
# Documenter never executes a plain ```julia fence, so any number typed into
# one is a hand-copy that nothing can check. Numbers that came out of a
# measurement therefore go through an `@eval` block, which Documenter *does*
# run, and which reads the same JSON the benchmark harness wrote and the web
# app consumes. There is one source of truth per number, and `make.jl` fails
# the build if it is missing or malformed.
#
# Use from a page:
#
#     ```@eval
#     Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
#     d = load_results("annotation_sweep.json")
#     md_table(["target", "d"], [[r["target"], r["dim"]] for r in d["rows"]])
#     ```
#
# `@__DIR__` inside an `@eval` block is `docs/build`, so `joinpath(@__DIR__,
# "..")` is `docs/` in both a local and a CI build. Do not use `pwd()`.

import JSON, Markdown, Printf

"""
Directory holding the checked-in benchmark result JSON.

`WHMC_BENCH_OUT` overrides it, which is what lets a CI job point the *consumer*
at results a generator has just written. The generators under `docs/benchmark/`
already honour that variable; without it on this side there is no job that can
run a probe and `makedocs` against the same files, and a generator is free to
change its output keys next to a stale checked-in JSON indefinitely. That is not
hypothetical — `655218e` changed `capture_boxing.jl`'s keys and nothing surfaced
it until `176a061` regenerated the results and took the docs build red.

Measured, that gap was **2 h 12 m on one day**, not the months an earlier draft
of this docstring claimed. The duration was never what made it dangerous. Two
other things did, and they are what this seam is actually for: the renaming
landed in one owner's file and the breakage surfaced in another's, and
`evidence.md` — which enumerates `results/` rather than naming keys — absorbed
the new shape in silence, so only the page that NAMED a removed key went red.
An enumerating consumer cannot fail; a naming one is the whole gate.

**A blank value is an ERROR, not a fallback.** `get(ENV, k, default)` returns
`""` for an exported-but-empty variable, and `joinpath("", "x")` is *relative*,
so the naive form silently resolves to the process's working directory. On the
generator side that scatters result files into the repo root. Here it is worse
in a subtler way: falling back to the checked-in results would let a CI guard
that regenerates into a temp directory build green against the very copies it
deliberately deleted — passing while checking nothing, which is the exact state
that guard exists to prevent. Refusing blank closes it independent of step
order.

Mirrors `env_dir` in `docs/benchmark/common.jl`. Duplicated rather than shared
on purpose: `common.jl` pulls PosteriorDB, BridgeStan and
StanLogDensityProblems, none of which `docs/Project.toml` has, so including it
would drag the whole measurement stack into `makedocs`.

This is not the only such copy. `env_dir` serves every generator that can
afford `common.jl`, and the scripts that cannot afford it inline their own
guard, each documenting its own reason. Change the rule and you must change all
of them, so find them with:

    grep -rl 'is set but blank' docs/

**Do not anchor that search on the variable name.** One rule guards several —
`WHMC_BENCH_OUT` here and in most generators, `WHMC_NW_OUT` in
`nonlinear_weighting_run.jl`, `WHMC_LINEAR_BENCH_OUT` in
`run_linear_restart_benchmark.jl` — so a variable grep silently returns a
*subset*, which is the failure direction that matters: a site you never see is
a site you never fix.

The error string cannot fail that way. It over-reports instead — a file that
merely *cites* this anchor matches it too — and that is the direction to
prefer, because an extra file to glance at costs seconds while a missed one
ships the bug. For the same reason this paragraph gives no count: a number in
prose has nothing to check it, and every count this note has carried was wrong
within the hour of being written.
"""
function results_dir()
    haskey(ENV, "WHMC_BENCH_OUT") ||
        return normpath(joinpath(@__DIR__, "benchmark", "results"))
    override = ENV["WHMC_BENCH_OUT"]
    isempty(strip(override)) && error("""
        WHMC_BENCH_OUT is set but blank.

        A blank value is almost always an unset shell variable that expanded to
        the empty string. It is refused rather than defaulted, because falling
        back to the checked-in results would let a guard that regenerates into a
        temporary directory pass while reading the copies it meant to replace.

        Unset WHMC_BENCH_OUT to read the checked-in results, or give it a real
        path.
        """)
    normpath(override)
end

"""
    load_results(relpath) -> Dict

Parse a checked-in result file under `docs/benchmark/results/`. Throws if the
file is absent — an `@eval` block that throws fails the docs build, which is
the point: a table whose data has gone missing must not render as an empty one.
"""
function load_results(relpath::AbstractString)
    path = joinpath(results_dir(), relpath)
    isfile(path) || error("""
        Benchmark results not found: $(path)

        Documentation tables are generated from the checked-in JSON under
        docs/benchmark/results/. Either the file was not committed, or a page
        names it wrongly. Do not replace this table with typed-in numbers.
        """)
    JSON.parse(read(path, String))
end

"""
    provenance(d; harness) -> String

One sentence naming what produced the numbers, built from the result file's own
metadata rather than from prose. A result file that records no `warmuphmc_sha`
says so instead of silently rendering an unattributed table.

Two artifact shapes are supported, because a harness should not have to repeat
itself to satisfy this function. If the file has a `config` object, provenance is
read from there — the preferred shape, since a single copy of `warmuphmc_sha`
cannot disagree with a duplicate of itself. Otherwise, or for any field `config`
omits, it falls back to the top level, which is the older flat shape.
"""
function provenance(d::AbstractDict; harness::AbstractString)
    cfg = get(d, "config", nothing)
    field(k) = cfg isa AbstractDict ? get(cfg, k, get(d, k, nothing)) : get(d, k, nothing)
    sha = field("warmuphmc_sha")
    where_ = sha === nothing ? "an **unrecorded** WarmupHMC revision" :
             "WarmupHMC `$(first(string(sha), 7))`"
    bits = ["Measured on $(where_)"]
    jl = field("julia")
    jl === nothing || push!(bits, "Julia $(jl)")
    bt = field("blas_threads")
    bt === nothing || push!(bits, "$(bt) BLAS thread$(bt == 1 ? "" : "s")")
    string(join(bits, ", "), ", by `docs/benchmark/", harness, "`.")
end

"""
    num(x; sig=4) -> String

Format a number for a table cell. Magnitudes at or above 1000 are rounded to a
whole number rather than shown in scientific notation — these are mostly
nanosecond counts, where `448636` is readable and `4.486e+05` is not.
"""
function num(x; sig::Int = 4)
    x === nothing && return "—"
    x isa Integer && return string(x)
    v = Float64(x)
    isfinite(v) || return "—"
    abs(v) >= 1000 && return string(Int(round(v)))
    v == round(v) && return string(Int(round(v)))
    Printf.@sprintf("%.*g", sig, v)
end

"""
    load_harness(relpath)

Make a benchmark script's derivation functions callable from an `@eval` block,
by including `docs/benchmark/<relpath>` into the calling module.

This exists so that a *derived* figure — a paired summary, a conclusion — is
computed from the rows at docs-build time by the same code the benchmark uses,
instead of being stored alongside them where it can drift. Rows are the data;
everything else is a function of the rows.

Two constraints on a file loaded this way, because the docs environment is not
the benchmark environment:

  * **It must only define things.** Anything that runs at include time — reading
    `ARGS`, writing output — runs during the docs build. Keep the driver and the
    derivation in separate files, as `summarize.jl` already does.
  * **It may only depend on what `docs/Project.toml` has**: `JSON`, `Markdown`,
    `Printf`, `Statistics`. Pulling in BridgeStan or PosteriorDB to render a
    table would make the docs build depend on the whole measurement stack.
"""
function load_harness(relpath::AbstractString)
    path = normpath(joinpath(@__DIR__, "benchmark", relpath))
    isfile(path) || error("""
        Benchmark harness not found: $(path)

        An `@eval` block asked for a derivation script under docs/benchmark/.
        Either it was not committed, or a page names it wrongly.
        """)
    Base.include(Base.@__MODULE__, path)
end

"""
    vega_figure(spec; caption="") -> Markdown.MD

Render a Vega-Lite `spec` (anything `JSON.json` accepts) as a figure, by
emitting it as a fenced ` ```vega-lite ` block. `setupVegaFigures` in
`docs/src/.vitepress/theme/vega-figure.ts` finds the rendered block and swaps a
chart in; the runtimes are CDN tags in `config.mts`.

**A code fence is not a stylistic choice — it is the only construct that
survives this pipeline.** The obvious shape, an `@eval` block returning a
`<div data-spec="…">`, cannot work, and both layers that break it break it
silently:

  * `Markdown.parse` destroys the JSON before anything sees it. `\$schema` is
    read as inline math and `ns_const` as emphasis, so the text that reaches
    the AST is already corrupt — measured, not feared.
  * DocumenterVitepress escapes `<` and `>` in *every* text node
    (`escape_markdown_text`, `writer.jl`), by design, because Vue would
    otherwise parse a bare `<` as a tag. A div therefore arrives at the browser
    as `&lt;div&gt;`.

Raw HTML reaches the page only through `@raw html`, which is a static fence and
cannot carry a computed value. Fenced code is the one path a generated string
crosses untouched: markdown does not interpret inside it, and VitePress marks
code blocks `v-pre`, so Vue does not either.

Figures are DERIVED, never checked in. Build the spec from the rows at
docs-build time — see [`load_harness`](@ref) — for the same reason a summary
table is derived: a spec stored beside the rows it plots is a second copy of
them that nothing forces to agree, and it goes stale in the silent direction,
because a chart still renders when its numbers are old.

If the runtime is unreachable the block stays a readable JSON dump rather than
becoming a blank gap, which is the right failure: the figure's data is still
on the page.
"""
function vega_figure(spec; caption::AbstractString = "")
    parts = Any[Markdown.Code("vega-lite", JSON.json(spec))]
    isempty(caption) || append!(parts, Markdown.parse("*" * caption * "*").content)
    Markdown.MD(parts)
end

"""
    md_table(headers, rows) -> Markdown.MD

Build a markdown table. Cells are passed through `string`, so pre-format
anything that needs it (see [`num`](@ref)).

**Zero rows is an error, not an empty table.** A headers-only table is valid
markdown, so it renders as a blank table and the build exits 0 — an artifact
that still parses but has lost its rows would ship as an empty table with
nothing anywhere saying so. [`load_results`](@ref) only guards the file being
*absent*; this guards it being present and empty.

Measured before this check existed: emptying `rows` in one artifact and
rebuilding took the build down only because a *derivation* called `median` on
an empty array two sections further down. The three tables above it rendered
blank and reported nothing. The catch was incidental to what the harness
happened to compute, which is not a guarantee.

If an empty result is meaningful somewhere, branch on `isempty` in the `@eval`
block and emit prose saying so — the derived blocks in `adaptive-centering.md`
do exactly that, and prose is the honest rendering of "there is nothing here".
"""
function md_table(headers::AbstractVector, rows::AbstractVector)
    isempty(rows) && error("""
        md_table was given zero rows (headers: $(join(string.(headers), ", "))).

        This is refused rather than rendered, because a headers-only table is
        valid markdown: the page would show a blank table and the build would
        succeed. If the underlying artifact can legitimately be empty, branch on
        `isempty` in the @eval block and emit prose instead of a table.
        """)
    io = IOBuffer()
    println(io, "| ", join(string.(headers), " | "), " |")
    println(io, "|", join(fill("---", length(headers)), "|"), "|")
    for r in rows
        length(r) == length(headers) ||
            error("row has $(length(r)) cells, expected $(length(headers)): $(r)")
        println(io, "| ", join(string.(r), " | "), " |")
    end
    Markdown.parse(String(take!(io)))
end
