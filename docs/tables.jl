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
it until the results were regenerated months later, which took the docs build
red.

**An empty value is treated as unset, deliberately.** `get(ENV, k, default)`
returns `""` for an exported-but-empty variable, and `joinpath("", "x")` is
*relative* — so the naive form silently resolves to the process's working
directory. On the generator side that scatters result files into the repo root;
here it would be worse, because the consumer would read whatever happened to be
there and the guard would pass while checking nothing.
"""
function results_dir()
    override = get(ENV, "WHMC_BENCH_OUT", "")
    isempty(override) || return normpath(override)
    normpath(joinpath(@__DIR__, "benchmark", "results"))
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
    md_table(headers, rows) -> Markdown.MD

Build a markdown table. Cells are passed through `string`, so pre-format
anything that needs it (see [`num`](@ref)).
"""
function md_table(headers::AbstractVector, rows::AbstractVector)
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
