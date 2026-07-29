module WarmupHMCBenchmarkEvidence

import JSON
using HTMXObjects: SemanticFields, SemanticGroup, SemanticLink, SemanticProse,
                   SemanticSection, SemanticTable, SemanticUnavailable

export benchmark_all_semantic, benchmark_index_semantic, benchmark_keys,
       benchmark_load, benchmark_path, benchmark_semantic, benchmark_table,
       results_dir

"""The checked-in benchmark-results directory, or the explicit CI override."""
function results_dir()
    haskey(ENV, "WHMC_BENCH_OUT") || return joinpath(@__DIR__, "results")
    override = ENV["WHMC_BENCH_OUT"]
    isempty(strip(override)) && error("WHMC_BENCH_OUT is set but blank")
    normpath(override)
end

benchmark_key(relpath) = replace(replace(relpath, ".json" => ""), '/' => '~')
benchmark_path(key) =
    joinpath(results_dir(), replace(String(key), '~' => '/') * ".json")

"""Every checked-in benchmark JSON key, sorted deterministically."""
function benchmark_keys()
    root = results_dir()
    isdir(root) || return String[]
    keys = String[]
    for (dir, _, files) in walkdir(root), file in files
        endswith(file, ".json") || continue
        push!(keys, benchmark_key(relpath(joinpath(dir, file), root)))
    end
    sort!(keys)
end

const PROVENANCE_ORDER =
    ["warmuphmc_sha", "julia", "host", "blas_threads", "n_seeds", "n_calls",
     "rounds", "n_draws_floor", "note"]
const COLUMN_ORDER = ["series", "target", "dim", "arm", "seed", "backend", "source"]

ordered(names, priority) = vcat(
    [name for name in priority if name in names],
    sort([name for name in names if !(name in priority)]),
)

cell(::Nothing) = ""
cell(x::Bool) = x ? "yes" : "no"
cell(x::Integer) = string(x)
cell(x::AbstractFloat) = begin
    isfinite(x) || return string(x)
    rounded = abs(x) >= 100 ? round(x; digits=1) : round(x; sigdigits=4)
    rounded == round(rounded) && abs(rounded) < 1e15 ?
        string(Int(round(rounded))) : string(rounded)
end
cell(x::AbstractVector) = join(cell.(x), ", ")
cell(x::AbstractDict) =
    join(["$key=$(cell(x[key]))" for key in sort(collect(keys(x)))], " ")
cell(x) = string(x)

"""Convert JSON row objects into a deterministic Tables.jl-compatible table."""
function benchmark_table(rows)
    names = String[]
    for row in rows, key in keys(row)
        key in names || push!(names, key)
    end
    columns = ordered(names, COLUMN_ORDER)
    NamedTuple(Symbol(column) => [cell(get(row, column, nothing)) for row in rows]
               for column in columns)
end

numeric_series(x) = x isa AbstractVector && !isempty(x) &&
    all(value -> value isa Real && !(value isa Bool), x)
is_series(x::AbstractDict) = !isempty(x) && all(numeric_series, values(x))
is_series(_) = false

function series_rows(series)
    rows = Dict{String,Any}[]
    for (name, values) in sort(collect(series); by=first)
        for (index, value) in enumerate(values)
            push!(rows, Dict("series" => string(name), "index" => index,
                             "value" => value))
        end
    end
    rows
end

"""
Load one result file into provenance plus named semantic-table sources.

The classification is structural: arrays of objects are row tables; numeric
arrays are collected into long `series/index/value` measurement tables;
objects whose values are all numeric arrays are measurement tables; other
objects are flattened provenance; everything else is scalar provenance.
"""
function benchmark_load(key)
    parsed = JSON.parsefile(benchmark_path(key))
    parsed isa AbstractVector && return (
        provenance = Pair{String,String}[],
        tables = ["rows" => benchmark_table(parsed)],
    )

    tables = Pair{String,Any}[]
    scalars = Pair{String,Any}[]
    nested = Pair{String,Any}[]
    top_level_series = Pair{String,Any}[]
    for key in sort(collect(keys(parsed)))
        value = parsed[key]
        if value isa AbstractVector && !isempty(value) &&
                all(row -> row isa AbstractDict, value)
            push!(tables, key => benchmark_table(value))
        elseif is_series(value)
            push!(tables, key => benchmark_table(series_rows(value)))
        elseif numeric_series(value)
            push!(top_level_series, key => value)
        elseif value isa AbstractDict
            append!(nested, [string(nested_key) => value[nested_key]
                             for nested_key in sort(collect(keys(value)))])
        else
            push!(scalars, key => value)
        end
    end
    isempty(top_level_series) ||
        push!(tables, "series" => benchmark_table(series_rows(top_level_series)))
    sort!(tables; by=first)
    nested_names = Set(first.(nested))
    flattened = vcat(nested, [pair for pair in scalars
                              if !(first(pair) in nested_names)])
    order = ordered(first.(flattened), PROVENANCE_ORDER)
    lookup = Dict(flattened)
    (
        provenance = [key => cell(lookup[key]) for key in order],
        tables,
    )
end

"""One result file represented once as HTMXObjects semantic data."""
function benchmark_semantic(key; max_rows=nothing)
    loaded = benchmark_load(key)
    body = Any[]
    for (name, table) in loaded.tables
        nrows = length(first(table))
        node = if !isnothing(max_rows) && nrows > max_rows
            SemanticUnavailable("$nrows rows × $(length(table)) columns — too " *
                "large to inline. The rows are in " *
                "`docs/benchmark/results/$(replace(String(key), '~' => '/')).json`, " *
                "and the app renders them at `/benchmark/$key`.")
        else
            SemanticTable(table)
        end
        push!(body, length(loaded.tables) == 1 ? node : SemanticSection(name, node))
    end
    SemanticSection(String(key),
        SemanticFields(; (Symbol(name) => value for (name, value) in loaded.provenance)...),
        body...,
    )
end

"""Index the checked-in evidence, optionally linking keys in an interactive app."""
function benchmark_index_semantic(link=nothing)
    result_keys = benchmark_keys()
    isempty(result_keys) && return SemanticSection("Benchmark evidence",
        SemanticUnavailable("No results checked in under docs/benchmark/results/."))
    rows = map(result_keys) do key
        loaded = benchmark_load(key)
        provenance = Dict(loaded.provenance)
        sha = string(get(provenance, "warmuphmc_sha", ""))
        (; file=key, sha=first(sha, min(7, length(sha))),
           tables=join([name for (name, _) in loaded.tables], ", "),
           rows=sum(table -> length(first(table)),
                    (table for (_, table) in loaded.tables); init=0))
    end
    SemanticSection("Benchmark evidence",
        SemanticProse("""
            Each row is one JSON written by a driver under `docs/benchmark/`. That
            file is the source: the app and the documentation project this same
            semantic value, so a published table cannot drift from the run that
            produced it.

            **Check the SHA before citing a row.** A results file is a snapshot of
            the tree it was measured on, and this directory keeps superseded runs
            beside current ones on purpose — the comparison is often the point."""),
        SemanticTable(rows),
        isnothing(link) ? SemanticGroup() :
            SemanticSection("Open a results file",
                SemanticGroup([SemanticLink(key, link(key)) for key in result_keys])),
    )
end

"""The index and every result file as one semantic tree."""
benchmark_all_semantic(; max_rows=40) = SemanticGroup(vcat(
    Any[benchmark_index_semantic()],
    Any[benchmark_semantic(key; max_rows) for key in benchmark_keys()],
))

end
